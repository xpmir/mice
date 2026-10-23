"""
ModernBERT MICE HuggingFace Implementation.
Custom PreTrainedModel and Modules supporting SDPA, Flash Attention 2, and ST export.
"""

import copy
from dataclasses import dataclass
import logging
import math
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from sentence_transformers.models import Dense, LayerNorm
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

from transformers.models.modernbert.modeling_modernbert import (
    ModernBertModel,
    ModernBertMLP,
    ModernBertAttention,
    ModernBertPreTrainedModel,
    ModernBertRotaryEmbedding,
    eager_attention_forward,
)
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import is_flash_attn_2_available

try:
    from transformers.modeling_flash_attention_utils import _unpad_input, _pad_input
except ImportError:
    _unpad_input, _pad_input = None, None

logger = logging.getLogger(__name__)


@dataclass
class ModernBertMiceOutput(SequenceClassifierOutput):
    """Output dataclass for ModernBertMiceForSequenceClassification.

    Attributes:
        loss (Optional[torch.FloatTensor]): Classification loss if labels provided.
        logits (torch.FloatTensor): Classification logits of shape (batch_size, num_labels).
        hidden_states (Optional[Tuple[torch.FloatTensor, ...]]): Hidden-states if requested.
        attentions (Optional[Tuple[torch.FloatTensor, ...]]): Attentions if requested.
        utilities (Optional[torch.FloatTensor]): Predicted token utility scores [B, L, 1] (or [total_tokens, 1]).
        p_active (Optional[torch.FloatTensor]): Exact active retention probabilities under Hard Concrete gate [B, L, 1].
        doc_hidden (Optional[torch.FloatTensor]): Pre-gating uncompressed document hidden states [B, L, D].
        pruned_utilities (Optional[torch.FloatTensor]): Post-pruned utility scores if hard pruning is active.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    utilities: Optional[torch.FloatTensor] = None
    p_active: Optional[torch.FloatTensor] = None
    doc_hidden: Optional[torch.FloatTensor] = None
    pruned_utilities: Optional[torch.FloatTensor] = None


def is_fa2_available() -> bool:
    """Checks if FlashAttention 2 is available and supported on the current CUDA GPU hardware (SM 8.0+)."""
    if torch.cuda.is_available():
        try:
            major_cc, _ = torch.cuda.get_device_capability(0)
            if major_cc < 8:
                return False
        except Exception:
            pass

    if is_flash_attn_2_available():
        return True
    try:
        import kernels  # noqa - HuggingFace kernels library providing kernels-community/flash-attn2

        return True
    except ImportError:
        pass
    try:
        import flash_attn  # noqa

        return True
    except ImportError:
        pass
    return False


def _unpad_input_with_posids(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Unpads 3D tensor [B, S, D] into 1D continuous token sequence [1, total_tokens, D].
    Calculates cu_seqlens [B+1] (torch.int32), max_seqlen (int), indices [total_tokens],
    and 1D position_ids [1, total_tokens] matching position indices within each sequence.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=hidden_states.device
        )
    else:
        attention_mask = attention_mask.bool()

    seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
            torch.cumsum(seqlens, dim=0, dtype=torch.int32),
        ]
    )
    max_seqlen = int(seqlens.max().item()) if len(seqlens) > 0 else 0

    mask_flat = attention_mask.view(-1)
    indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)

    unpadded_flat = hidden_states.view(-1, hidden_size)[indices]
    unpadded_hidden = unpadded_flat.unsqueeze(0)  # [1, total_tokens, hidden_size]

    row_pos_ids = (
        torch.arange(seq_len, device=hidden_states.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    position_ids = row_pos_ids.reshape(-1)[indices].unsqueeze(0)  # [1, total_tokens]

    return unpadded_hidden, cu_seqlens, max_seqlen, indices, position_ids


class UtilityPredictor(nn.Module):
    """Two-layer MLP Utility Predictor using ModernBERT's GeGLU activation
    and a Sigmoid or Hard Concrete (Louizos et al. 2018) output to produce per-token utility scores U in [0, 1].
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: Optional[int] = None,
        utility_temp: float = 1.0,
        utility_bias: float = 5.0,
        use_hard_concrete: bool = False,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
    ):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim * 2

        self.utility_bias = utility_bias
        self.use_hard_concrete = use_hard_concrete
        self.hc_gamma = hc_gamma
        self.hc_zeta = hc_zeta

        self.wi = nn.Linear(hidden_dim, 2 * bottleneck_dim, bias=False)
        self.wo = nn.Linear(bottleneck_dim, 1, bias=True)
        self.register_buffer(
            "temperature", torch.tensor(utility_temp, dtype=torch.float32)
        )
        self._init_weights()

    def _init_weights(self):
        # 1. Initialize wi with standard small weights
        nn.init.normal_(self.wi.weight, mean=0.0, std=0.02)

        # 2. Keep the output weights very small so input variance doesn't overpower the bias
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.01)

        # 3. Set the bias to the configured scalar.
        # logit = 5.0 -> sigmoid(5.0) = 0.9933 (keeps ~99% of utility)
        if self.wo.bias is not None:
            nn.init.constant_(self.wo.bias, self.utility_bias)

    # TODO MonkeyPatch for aval - Remove this in next release
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        bias_key = prefix + "wo.bias"
        if bias_key not in state_dict and self.wo.bias is not None:
            # Handle backward compatibility for legacy checkpoints saved with wo.bias=False:
            # Set bias to 0.0 to preserve the exact weights/logits of the trained legacy checkpoint
            state_dict[bias_key] = torch.tensor(
                [0.0], device=self.wo.weight.device, dtype=self.wo.weight.dtype
            )
        temp_key = prefix + "temperature"
        if temp_key not in state_dict:
            state_dict[temp_key] = torch.tensor(
                1.0, device=self.wo.weight.device, dtype=torch.float32
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self, h: torch.Tensor, return_p_active: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Args:
            h: Hidden state tensor of shape (batch_size, seq_len, hidden_dim)
               or (1, total_tokens, hidden_dim)
            return_p_active: If True, also returns the active probability tensor p_active.

        Returns:
            u: Utility scores of shape (batch_size, seq_len, 1) or (1, total_tokens, 1) in [0, 1]
               (or tuple of (u, p_active) if return_p_active=True).
        """
        x_val, x_gate = self.wi(h).chunk(2, dim=-1)
        geglu_out = x_val * torch.nn.functional.gelu(x_gate, approximate="none")
        logits = self.wo(geglu_out)
        beta = max(float(self.temperature.item()), 1e-4)

        if not self.use_hard_concrete:
            u = torch.sigmoid(logits / beta).to(h.dtype)
            if return_p_active:
                return u, None
            return u

        # Hard Concrete L0 formulation:
        # Closed-form retention probability: P(u > 0) = sigma(z - beta * log(-gamma / zeta))
        offset = beta * math.log(-self.hc_gamma / self.hc_zeta)
        p_active = torch.sigmoid(logits - offset).to(h.dtype)

        if self.training:
            # Clamped uniform noise to avoid log(0)
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((logits + torch.log(u) - torch.log(1.0 - u)) / beta)
        else:
            # Deterministic median evaluation (noise u = 0.5 -> log(u) - log(1-u) = 0)
            s = torch.sigmoid(logits / beta)

        s_bar = s * (self.hc_zeta - self.hc_gamma) + self.hc_gamma
        u_clamped = torch.clamp(s_bar, min=0.0, max=1.0).to(h.dtype)
        if return_p_active:
            return u_clamped, p_active
        return u_clamped


def _hard_prune_unpadded_documents(
    d_hidden: torch.Tensor,
    d_pos_ids: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    utilities: torch.Tensor,
    threshold: float = 0.5,
    return_pruned_utilities: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor],
]:
    """
    Hard-prunes document tokens whose utility score < threshold from 1D unpadded FA2 sequence.
    When threshold == 0.0 (Hard Concrete pruning), keeps tokens with utility > 0.0.
    Retains original positional indices (d_pos_ids) for RoPE embeddings in top layers.
    Rebuilds cu_seqlens_d and max_seqlen_d for remaining unpruned tokens.
    If return_pruned_utilities is True, also returns pruned_utilities.
    """
    batch_size = cu_seqlens_d.shape[0] - 1
    keep_mask = utilities.squeeze(-1).squeeze(0) > threshold  # [total_tokens]

    cu = cu_seqlens_d.cpu().tolist()
    kept_per_doc = []
    for i in range(batch_size):
        start, end = cu[i], cu[i + 1]
        if end > start:
            doc_keep = keep_mask[start:end]
            num_kept = doc_keep.sum().item()
            if num_kept == 0:
                # Ensure each doc keeps at least 1 token (token with highest utility)
                doc_utils = utilities[0, start:end, 0]
                best_idx = start + torch.argmax(doc_utils).item()
                keep_mask[best_idx] = True
                num_kept = 1
            kept_per_doc.append(num_kept)
        else:
            kept_per_doc.append(0)

    pruned_d_hidden = d_hidden[:, keep_mask, :]
    pruned_d_pos_ids = d_pos_ids[:, keep_mask] if d_pos_ids is not None else None

    new_cu_seqlens_d = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=d_hidden.device
    )
    new_cu_seqlens_d[1:] = torch.cumsum(
        torch.tensor(kept_per_doc, device=d_hidden.device, dtype=torch.int32), dim=0
    )
    new_max_seqlen_d = max(kept_per_doc) if len(kept_per_doc) > 0 else 0

    if return_pruned_utilities:
        pruned_utilities = utilities[:, keep_mask, :]
        return (
            pruned_d_hidden,
            pruned_d_pos_ids,
            new_cu_seqlens_d,
            new_max_seqlen_d,
            pruned_utilities,
        )

    return pruned_d_hidden, pruned_d_pos_ids, new_cu_seqlens_d, new_max_seqlen_d


def _hard_prune_padded_documents(
    d_hidden: torch.Tensor,
    doc_mask: torch.Tensor,
    utilities: torch.Tensor,
    threshold: float = 0.5,
    return_pruned_utilities: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Hard-prunes document tokens whose utility score < threshold from 3D padded sequence [B, L_d, D].
    When threshold == 0.0 (Hard Concrete pruning), keeps tokens with utility > 0.0.
    Constructs new 3D padded tensor [B, max_kept_len, D], new mask [B, max_kept_len],
    and retained position_ids [B, max_kept_len] preserving original token positions for RoPE.
    If return_pruned_utilities is True, also returns new_utilities [B, max_kept_len, 1].
    """
    batch_size, seq_len, hidden_size = d_hidden.shape
    valid_mask = (utilities.squeeze(-1) > threshold) & doc_mask.bool()  # [B, L_d]

    for i in range(batch_size):
        if valid_mask[i].sum().item() == 0:
            doc_valid = doc_mask[i].bool()
            if doc_valid.any():
                doc_utils = utilities[i, :, 0]
                doc_utils_masked = torch.where(
                    doc_valid, doc_utils, torch.tensor(-1.0, device=d_hidden.device)
                )
                best_idx = torch.argmax(doc_utils_masked).item()
                valid_mask[i, best_idx] = True

    num_kept = valid_mask.sum(dim=1)  # [B]
    max_kept_len = int(num_kept.max().item())

    new_d_hidden = torch.zeros(
        (batch_size, max_kept_len, hidden_size),
        dtype=d_hidden.dtype,
        device=d_hidden.device,
    )
    new_doc_mask = torch.zeros(
        (batch_size, max_kept_len), dtype=doc_mask.dtype, device=doc_mask.device
    )
    new_pos_ids = torch.zeros(
        (batch_size, max_kept_len), dtype=torch.long, device=d_hidden.device
    )
    if return_pruned_utilities:
        new_utilities = torch.zeros(
            (batch_size, max_kept_len, 1),
            dtype=utilities.dtype,
            device=utilities.device,
        )

    orig_pos = torch.arange(seq_len, device=d_hidden.device)

    for i in range(batch_size):
        kept_indices = torch.nonzero(valid_mask[i], as_tuple=False).squeeze(-1)
        k = len(kept_indices)
        if k > 0:
            new_d_hidden[i, :k] = d_hidden[i, kept_indices]
            new_doc_mask[i, :k] = 1
            new_pos_ids[i, :k] = orig_pos[kept_indices]
            if return_pruned_utilities:
                new_utilities[i, :k] = utilities[i, kept_indices]

    if return_pruned_utilities:
        return new_d_hidden, new_doc_mask, new_pos_ids, new_utilities

    return new_d_hidden, new_doc_mask, new_pos_ids


def _concat_query_doc_unpadded_old(
    q_hidden: torch.Tensor,
    d_hidden: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Reference loop-based implementation of unpadded query-doc concatenation (kept for output comparison)."""
    batch_size = cu_seqlens_q.shape[0] - 1
    device = q_hidden.device
    hidden_dim = q_hidden.shape[-1]

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    d_lens = cu_seqlens_d[1:] - cu_seqlens_d[:-1]
    d_kept_lens = torch.clamp(d_lens - 1, min=0)
    joint_lens = q_lens + d_kept_lens

    cu_seqlens_joint = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_joint[1:] = torch.cumsum(joint_lens, dim=0, dtype=torch.int32)
    max_seqlen_joint = int(joint_lens.max().item()) if len(joint_lens) > 0 else 0
    total_joint = int(cu_seqlens_joint[-1].item())

    joint_hidden = torch.empty(
        (1, total_joint, hidden_dim), dtype=q_hidden.dtype, device=device
    )
    pos_ids_joint = torch.empty((1, total_joint), dtype=torch.long, device=device)

    cu_q = cu_seqlens_q.cpu().tolist()
    cu_d = cu_seqlens_d.cpu().tolist()
    cu_j = cu_seqlens_joint.cpu().tolist()

    for i in range(batch_size):
        js, je = cu_j[i], cu_j[i + 1]
        qs, qe = cu_q[i], cu_q[i + 1]
        ds, de = cu_d[i], cu_d[i + 1]

        q_part = q_hidden[0, qs:qe]
        d_part = (
            d_hidden[0, ds + 1 : de]
            if de > ds + 1
            else q_hidden.new_empty((0, hidden_dim))
        )
        joint_hidden[0, js:je] = torch.cat([q_part, d_part], dim=0)
        pos_ids_joint[0, js:je] = torch.arange(je - js, device=device)

    return joint_hidden, cu_seqlens_joint, max_seqlen_joint, pos_ids_joint


def _concat_query_doc_unpadded(
    q_hidden: torch.Tensor,
    d_hidden: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Concatenates unpadded query and document representations along the sequence dimension
    for FlashAttention-2 execution using fully vectorized tensor operations (no CPU syncs or loops).

    Architecture context:
    - In MICE bottom layers, query and document are encoded independently:
        Query sequence: [CLS] q_1 ... q_m [SEP]
        Document sequence: [CLS] d_1 ... d_k [SEP]
    - To perform joint self-attention in top interaction layers without a duplicate [CLS],
      we drop the document's leading [CLS] (token index 0 of each doc sequence).
      The resulting concatenated sequence is exactly:
        [CLS] q_1 ... q_m [SEP] d_1 ... d_k [SEP]
      matching ModernBERT's native pair pretraining format.
    - FlashAttention-2 requires all tokens in sequence `i` to be contiguous in memory:
      [q_seq_0, d_seq_0, q_seq_1, d_seq_1, ...], with updated cumulative sequence lengths
      `cu_seqlens_joint` and 1D position IDs `0 .. len_i - 1` for RoPE embeddings.

    Returns:
        (joint_hidden, cu_seqlens_joint, max_seqlen_joint, pos_ids_joint)
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    device = q_hidden.device
    hidden_dim = q_hidden.shape[-1]

    # Calculate token counts per sequence; drop document CLS (-1)
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    d_lens = cu_seqlens_d[1:] - cu_seqlens_d[:-1]
    d_kept_lens = torch.clamp(d_lens - 1, min=0)
    joint_lens = q_lens + d_kept_lens

    # Build new cumulative sequence lengths for the combined sequence
    cu_seqlens_joint = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_joint[1:] = torch.cumsum(joint_lens, dim=0, dtype=torch.int32)
    max_seqlen_joint = int(joint_lens.max().item()) if len(joint_lens) > 0 else 0
    total_joint = int(cu_seqlens_joint[-1].item())

    joint_hidden = torch.empty(
        (1, total_joint, hidden_dim), dtype=q_hidden.dtype, device=device
    )

    # 1. Vectorized mapping for query tokens
    seq_ids_q = torch.repeat_interleave(torch.arange(batch_size, device=device), q_lens)
    offsets_q = torch.arange(q_hidden.shape[1], device=device) - cu_seqlens_q[seq_ids_q]
    dest_indices_q = cu_seqlens_joint[seq_ids_q] + offsets_q
    joint_hidden[0, dest_indices_q] = q_hidden[0]

    # 2. Vectorized mapping for document tokens (dropping leading [CLS] at offset 0)
    if d_hidden.shape[1] > 0:
        seq_ids_d_all = torch.repeat_interleave(
            torch.arange(batch_size, device=device), d_lens
        )
        offsets_d_all = (
            torch.arange(d_hidden.shape[1], device=device) - cu_seqlens_d[seq_ids_d_all]
        )
        valid_d_mask = offsets_d_all > 0
        if valid_d_mask.any():
            seq_ids_d = seq_ids_d_all[valid_d_mask]
            offsets_d = (
                offsets_d_all[valid_d_mask] - 1
            )  # 0-indexed within kept doc tokens
            dest_indices_d = cu_seqlens_joint[seq_ids_d] + q_lens[seq_ids_d] + offsets_d
            joint_hidden[0, dest_indices_d] = d_hidden[0, valid_d_mask]

    # 3. Vectorized position IDs: 0 .. joint_lens[i]-1 for each sequence (for RoPE)
    seq_ids_joint = torch.repeat_interleave(
        torch.arange(batch_size, device=device), joint_lens
    )
    pos_ids_joint = (
        torch.arange(total_joint, device=device) - cu_seqlens_joint[seq_ids_joint]
    ).unsqueeze(0)

    return joint_hidden, cu_seqlens_joint, max_seqlen_joint, pos_ids_joint


def _concat_query_doc_padded(
    q_hidden: torch.Tensor,
    d_hidden: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    doc_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenates padded query and document representations along sequence dimension for
    standard SDPA / eager attention execution using fully vectorized tensor operations.

    Why repacking is necessary:
    - Naively calling `torch.cat([q_hidden, d_hidden[:, 1:]], dim=1)` would leave query padding
      tokens in the *middle* of the sequence between query tokens and document tokens.
    - Padding tokens in the middle enlarge and distort relative positional distances in RoPE,
      violating the contiguous [CLS] q [SEP] d [SEP] pretraining layout.
    - This helper repacks each sequence so valid query tokens are immediately followed by valid
      document tokens (minus doc [CLS]), ensuring all padding is contiguous at the end of the row.

    Returns:
        (joint_hidden, joint_mask) where joint_hidden has shape [B, max_joint_len, D]
    """
    batch_size, _, hidden_dim = q_hidden.shape
    device = q_hidden.device

    if query_mask is None:
        query_mask = torch.ones(q_hidden.shape[:2], dtype=torch.long, device=device)
    if doc_mask is None:
        doc_mask = torch.ones(d_hidden.shape[:2], dtype=torch.long, device=device)

    # Compute valid lengths per sequence
    q_lens = query_mask.sum(dim=1)
    d_lens = doc_mask.sum(dim=1)
    # Drop doc [CLS] (token index 0)
    d_kept_lens = torch.clamp(d_lens - 1, min=0)
    joint_lens = q_lens + d_kept_lens
    max_joint_len = int(joint_lens.max().item()) if len(joint_lens) > 0 else 0

    joint_hidden = torch.zeros(
        (batch_size, max_joint_len, hidden_dim), dtype=q_hidden.dtype, device=device
    )
    joint_mask = (
        torch.arange(max_joint_len, device=device).unsqueeze(0)
        < joint_lens.unsqueeze(1)
    ).long()

    # Vectorized placement of valid query tokens
    q_valid = query_mask.bool()
    if q_valid.any():
        b_idx_q = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand_as(query_mask)[q_valid]
        )
        c_idx_q = (
            torch.arange(q_hidden.shape[1], device=device)
            .unsqueeze(0)
            .expand_as(query_mask)[q_valid]
        )
        joint_hidden[b_idx_q, c_idx_q] = q_hidden[q_valid]

    # Vectorized placement of valid document tokens (excluding doc CLS at col 0)
    if d_hidden.shape[1] > 1:
        d_valid = doc_mask[:, 1:].bool()
        if d_valid.any():
            b_idx_d = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand_as(doc_mask[:, 1:])[d_valid]
            )
            c_offset = (
                torch.arange(doc_mask.shape[1] - 1, device=device)
                .unsqueeze(0)
                .expand_as(doc_mask[:, 1:])[d_valid]
            )
            dest_col_d = q_lens[b_idx_d] + c_offset
            joint_hidden[b_idx_d, dest_col_d] = d_hidden[:, 1:][d_valid]

    return joint_hidden, joint_mask


def _pack_pair_inputs_padded(
    query_ids: torch.Tensor,
    query_mask: torch.Tensor,
    doc_ids: torch.Tensor,
    doc_mask: torch.Tensor,
    pad_token_id: int = 50283,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Packs valid query and doc tokens contiguously per batch row without middle padding:
    [CLS] q ... [SEP] [CLS] d ... [SEP] [PAD] ...
    Fully vectorized across batch dimensions.
    Returns:
        (unified_ids, q_lens, d_lens, max_joint_len)
    """
    B = query_ids.shape[0]
    device = query_ids.device
    q_lens = query_mask.sum(dim=1)
    d_lens = doc_mask.sum(dim=1)
    joint_lens = q_lens + d_lens
    max_joint_len = int(joint_lens.max().item()) if len(joint_lens) > 0 else 0

    unified_ids = torch.full(
        (B, max_joint_len), pad_token_id, dtype=query_ids.dtype, device=device
    )

    # Vectorized query token placement
    valid_q = query_mask.bool()
    if valid_q.any():
        batch_idx_q = torch.arange(B, device=device).unsqueeze(1).expand_as(query_ids)
        dest_cols_q = (
            torch.arange(query_ids.shape[1], device=device)
            .unsqueeze(0)
            .expand_as(query_ids)
        )
        unified_ids[batch_idx_q[valid_q], dest_cols_q[valid_q]] = query_ids[valid_q]

    # Vectorized doc token placement immediately following query tokens
    valid_d = doc_mask.bool()
    if valid_d.any():
        batch_idx_d = torch.arange(B, device=device).unsqueeze(1).expand_as(doc_ids)
        dest_cols_d = q_lens.unsqueeze(1) + torch.arange(
            doc_ids.shape[1], device=device
        ).unsqueeze(0)
        unified_ids[batch_idx_d[valid_d], dest_cols_d[valid_d]] = doc_ids[valid_d]

    return unified_ids, q_lens, d_lens, max_joint_len


def _create_mice_block_masks(
    q_lens: torch.Tensor,
    d_lens: torch.Tensor,
    seq_len: int,
    local_attention: Optional[int] = 128,
    device: Optional[torch.device] = None,
) -> dict:
    """Constructs 4D block-diagonal attention masks isolating Query and Document tokens:
    - Query tokens (col < q_len) attend ONLY to Query tokens.
    - Doc tokens (q_len <= col < q_len + d_len) attend ONLY to Doc tokens.
    - Padding rows (row >= q_len + d_len) attend to their own diagonal position to prevent NaN in softmax.
    Returns:
        {"full_attention": 4D boolean mask [B, 1, S, S], "sliding_attention": 4D boolean mask [B, 1, S, S]}
    """
    if device is None:
        device = q_lens.device
    B = q_lens.shape[0]
    row_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)  # [1, S, 1]
    col_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)  # [1, 1, S]
    q_l = q_lens.view(B, 1, 1)
    d_l = d_lens.view(B, 1, 1)

    is_row_q = row_idx < q_l
    is_col_q = col_idx < q_l
    is_row_d = (row_idx >= q_l) & (row_idx < (q_l + d_l))
    is_col_d = (col_idx >= q_l) & (col_idx < (q_l + d_l))

    block_mask = (is_row_q & is_col_q) | (is_row_d & is_col_d)
    is_row_pad = ~(is_row_q | is_row_d)
    diag = row_idx == col_idx
    full_mask = (block_mask | (is_row_pad & diag)).unsqueeze(1)

    if local_attention is not None and local_attention > 0:
        half_window = local_attention // 2
        band_mask = (row_idx - col_idx).abs() <= half_window
        sw_mask = ((block_mask & band_mask) | (is_row_pad & diag)).unsqueeze(1)
    else:
        sw_mask = full_mask

    return {"full_attention": full_mask, "sliding_attention": sw_mask}


def _create_block_position_ids(
    q_lens: torch.Tensor,
    d_lens: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Computes Rotary Position IDs resetting to 0 at the document boundary:
    - Query tokens: 0 .. q_lens[b] - 1
    - Doc tokens:   0 .. d_lens[b] - 1
    - Padding tokens: 0
    Returns: [B, seq_len] int64 tensor.
    """
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]
    q_l = q_lens.unsqueeze(1)  # [B, 1]
    d_l = d_lens.unsqueeze(1)  # [B, 1]

    is_q = col_idx < q_l
    is_d = (col_idx >= q_l) & (col_idx < (q_l + d_l))

    pos_q = col_idx
    pos_d = col_idx - q_l

    return torch.where(is_q, pos_q, torch.where(is_d, pos_d, torch.zeros_like(col_idx)))


class ModernBertMiceConfig(ModernBertConfig):
    model_type = "modernbert_mice"

    def __init__(
        self,
        n_contextualization_layers: int = 6,
        n_docs_ctx_layers: Optional[int] = None,
        n_interaction_layers: Optional[int] = None,
        bound_bottom_layers: bool = True,
        cross_attn_first: bool = True,
        utility_pred: bool = False,
        hard_pruning: Optional[bool] = None,
        hard_prune_threshold: float = 0.5,
        normalize_doc_embeds: bool = False,
        use_upper_rope: bool = True,
        top_layers_doc_ln: bool = False,
        bottom_final_norm: bool = True,
        top_final_norm: bool = False,
        unified_bottom_pass: bool = True,
        utility_temp: float = 1.0,
        utility_bias: float = 5.0,
        gate_target: str = "h",
        use_hard_concrete: bool = False,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
        compress_dim: float = 1.0,
        compress_init_option: str = "seed",
        doc_bottleneck_dim: Optional[int] = None,
        use_cross_attn: bool = True,
        head_type: str = "linear",
        num_labels: int = 1,
        **kwargs,
    ):
        if num_labels == 1:
            kwargs.pop("id2label", None)
            kwargs.pop("label2id", None)
            kwargs["id2label"] = {0: "LABEL_0"}
            kwargs["label2id"] = {"LABEL_0": 0}
        super().__init__(num_labels=num_labels, **kwargs)
        self.n_contextualization_layers = n_contextualization_layers
        self.n_docs_ctx_layers = n_docs_ctx_layers
        self.n_interaction_layers = n_interaction_layers
        self.bound_bottom_layers = bound_bottom_layers
        self.cross_attn_first = cross_attn_first
        self.utility_pred = utility_pred
        self.normalize_doc_embeds = normalize_doc_embeds
        self.use_upper_rope = use_upper_rope
        self.top_layers_doc_ln = top_layers_doc_ln
        self.bottom_final_norm = bottom_final_norm
        self.top_final_norm = top_final_norm
        self.unified_bottom_pass = unified_bottom_pass
        self.utility_temp = utility_temp
        self.utility_bias = utility_bias
        if gate_target not in ("h", "v"):
            raise ValueError(
                f"Unknown gate_target: '{gate_target}'. Expected 'h' or 'v'."
            )
        self.gate_target = gate_target
        if hc_zeta <= hc_gamma:
            raise ValueError(
                f"hc_zeta ({hc_zeta}) must be strictly greater than hc_gamma ({hc_gamma})"
            )
        self.use_hard_concrete = use_hard_concrete
        self.hc_gamma = hc_gamma
        self.hc_zeta = hc_zeta
        if hard_pruning is None:
            self.hard_pruning = True if use_hard_concrete else False
        else:
            self.hard_pruning = hard_pruning
        self.hard_prune_threshold = hard_prune_threshold
        assert compress_dim >= 1.0, f"compress_dim must be >= 1.0, got {compress_dim}"
        self.compress_dim = compress_dim
        self.compress_init_option = compress_init_option
        if doc_bottleneck_dim is not None:
            assert doc_bottleneck_dim > 0, (
                f"doc_bottleneck_dim must be > 0, got {doc_bottleneck_dim}"
            )
        self.doc_bottleneck_dim = doc_bottleneck_dim
        if self.doc_bottleneck_dim is not None and not self.normalize_doc_embeds:
            logger.warning(
                "⚠️ [MICE V2 CONFIG WARNING] `doc_bottleneck_dim` is enabled (%s) but "
                "`normalize_doc_embeds` is False! Document vectors stored in the index "
                "will not have unit L2 norm, which can severely degrade PLAID indexing/retrieval.",
                self.doc_bottleneck_dim,
            )
        self.use_cross_attn = use_cross_attn
        self.head_type = head_type


class ModernBertMiceCrossAttention(nn.Module):
    """
    Cross-attention module adapting ModernBERT's fused Wqkv linear projection
    to perform Cross-Attention (Query attending to Document).
    Supports SDPA and Flash Attention 2 backends via HF's ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self, config: ModernBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.deterministic_flash_attn = getattr(
            config, "deterministic_flash_attn", False
        )

        # Fused Wqkv projection matching ModernBERT
        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * self.head_dim * config.num_attention_heads,
            bias=config.attention_bias,
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        q_hidden: torch.Tensor,
        d_hidden: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        d_mask: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_d: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_d: Optional[int] = None,
        d_utilities: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        is_flash_impl = "flash" in str(attn_impl).lower()

        if (
            cu_seqlens_q is not None
            and cu_seqlens_d is not None
            and is_flash_impl
            and q_hidden.is_cuda
            and is_fa2_available()
        ):
            # Unpadded FlashAttention 2 path
            total_q = q_hidden.shape[1]
            total_d = d_hidden.shape[1]

            q_qkv = self.Wqkv(q_hidden).view(
                1, total_q, 3, self.num_heads, self.head_dim
            )
            query_states = q_qkv[:, :, 0].transpose(
                1, 2
            )  # [1, heads, total_q, head_dim]

            d_qkv = self.Wqkv(d_hidden).view(
                1, total_d, 3, self.num_heads, self.head_dim
            )
            key_states = d_qkv[:, :, 1].transpose(1, 2)  # [1, heads, total_d, head_dim]
            value_states = d_qkv[:, :, 2].transpose(
                1, 2
            )  # [1, heads, total_d, head_dim]

            if d_utilities is not None:
                # Value-only gating: broadcast [1, total_d, 1] -> [1, 1, total_d, 1] over heads & head_dim
                u_scale = d_utilities.squeeze(-1)[:, None, :, None].to(
                    value_states.dtype
                )
                value_states = value_states * u_scale

            if (
                query_states.dtype != value_states.dtype
                or query_states.dtype != key_states.dtype
            ):
                logger.warning(
                    f"[ModernBertMiceCrossAttention] Attention dtype mismatch in varlen path: "
                    f"query={query_states.dtype}, key={key_states.dtype}, value={value_states.dtype}. "
                    f"Casting key and value to {query_states.dtype} to prevent FlashAttention kernel failure."
                )
                key_states = key_states.to(query_states.dtype)
                value_states = value_states.to(query_states.dtype)

            attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_impl, eager_attention_forward
            )

            attn_output, _ = attn_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,  # Crucial: bypasses internal 2D _unpad_input
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.head_dim**-0.5,
                is_causal=False,
                cu_seq_lens_q=cu_seqlens_q.to(torch.int32),
                cu_seq_lens_k=cu_seqlens_d.to(torch.int32),
                max_length_q=int(max_seqlen_q) if max_seqlen_q is not None else total_q,
                max_length_k=int(max_seqlen_d) if max_seqlen_d is not None else total_d,
                **kwargs,
            )

            attn_output = attn_output.reshape(1, total_q, -1).contiguous()
            attn_output = self.out_drop(self.Wo(attn_output))
            return attn_output

        # Standard 3D Padded Path
        batch_size, q_len, _ = q_hidden.shape
        _, d_len, _ = d_hidden.shape

        # Extract Q from q_hidden
        q_qkv = self.Wqkv(q_hidden).view(
            batch_size, q_len, 3, self.num_heads, self.head_dim
        )
        query_states = q_qkv[:, :, 0].transpose(1, 2)  # [B, heads, q_len, head_dim]

        # Extract K, V from d_hidden
        d_qkv = self.Wqkv(d_hidden).view(
            batch_size, d_len, 3, self.num_heads, self.head_dim
        )
        key_states = d_qkv[:, :, 1].transpose(1, 2)  # [B, heads, d_len, head_dim]
        value_states = d_qkv[:, :, 2].transpose(1, 2)  # [B, heads, d_len, head_dim]

        if d_utilities is not None:
            # Value-only gating: broadcast [B, d_len, 1] -> [B, 1, d_len, 1] over heads & head_dim
            u_scale = d_utilities.squeeze(-1)[:, None, :, None].to(value_states.dtype)
            value_states = value_states * u_scale

        if q_mask is None:
            q_mask = torch.ones(
                (batch_size, q_len), dtype=torch.bool, device=q_hidden.device
            )
        if d_mask is None:
            d_mask = torch.ones(
                (batch_size, d_len), dtype=torch.bool, device=d_hidden.device
            )

        if (
            is_flash_impl
            and is_fa2_available()
            and q_hidden.is_cuda
            and _unpad_input is not None
            and _pad_input is not None
        ):
            # Per-layer unpad fallback if cu_seqlens is not provided directly
            q_unpadded, indices_q, cu_seqlens_q, max_seqlen_q, _ = _unpad_input(
                q_hidden, q_mask
            )
            d_unpadded, _, cu_seqlens_d, max_seqlen_d, _ = _unpad_input(
                d_hidden, d_mask
            )

            q_un_qkv = self.Wqkv(q_unpadded).view(-1, 3, self.num_heads, self.head_dim)
            Q_un = q_un_qkv[:, 0].unsqueeze(0)

            d_un_qkv = self.Wqkv(d_unpadded).view(-1, 3, self.num_heads, self.head_dim)
            K_un = d_un_qkv[:, 1].unsqueeze(0)
            V_un = d_un_qkv[:, 2].unsqueeze(0)

            if d_utilities is not None:
                u_unpadded, _, _, _, _ = _unpad_input(d_utilities, d_mask)
                V_un = V_un * u_unpadded.view(1, -1, 1, 1).to(V_un.dtype)

            if Q_un.dtype != V_un.dtype or Q_un.dtype != K_un.dtype:
                logger.warning(
                    f"[ModernBertMiceCrossAttention] Attention dtype mismatch in 3D unpad fallback path: "
                    f"query={Q_un.dtype}, key={K_un.dtype}, value={V_un.dtype}. "
                    f"Casting key and value to {Q_un.dtype} to prevent FlashAttention kernel failure."
                )
                K_un = K_un.to(Q_un.dtype)
                V_un = V_un.to(Q_un.dtype)

            attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_impl, eager_attention_forward
            )

            attn_output, _ = attn_interface(
                self,
                Q_un.transpose(1, 2),
                K_un.transpose(1, 2),
                V_un.transpose(1, 2),
                attention_mask=None,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.head_dim**-0.5,
                is_causal=False,
                cu_seq_lens_q=cu_seqlens_q.to(torch.int32),
                cu_seq_lens_k=cu_seqlens_d.to(torch.int32),
                max_length_q=int(max_seqlen_q),
                max_length_k=int(max_seqlen_d),
                **kwargs,
            )

            attn_output_flat = attn_output.reshape(-1, self.config.hidden_size)
            attn_output = _pad_input(attn_output_flat, indices_q, batch_size, q_len)
        else:
            if is_flash_impl:
                logger.warning(
                    f"[MICE CrossAttention] FlashAttention requested ('{attn_impl}'), but unpadding is unavailable "
                    f"(_unpad_input={_unpad_input is not None}). Falling back to SDPA with 4D cross-mask."
                )

            cross_mask = (
                q_mask.unsqueeze(-1).bool() & d_mask.unsqueeze(1).bool()
            ).unsqueeze(1)

            fallback_impl = "eager" if attn_impl == "eager" else "sdpa"
            attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                fallback_impl, eager_attention_forward
            )
            if (
                query_states.dtype != value_states.dtype
                or query_states.dtype != key_states.dtype
            ):
                logger.warning(
                    f"[ModernBertMiceCrossAttention] Attention dtype mismatch in SDPA fallback path: "
                    f"query={query_states.dtype}, key={key_states.dtype}, value={value_states.dtype}. "
                    f"Casting key and value to {query_states.dtype}."
                )
                key_states = key_states.to(query_states.dtype)
                value_states = value_states.to(query_states.dtype)

            attn_output, _ = attn_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=cross_mask,
                scaling=self.head_dim**-0.5,
                is_causal=False,
                **kwargs,
            )
            attn_output = attn_output.reshape(batch_size, q_len, -1).contiguous()

        attn_output = self.out_drop(self.Wo(attn_output))
        return attn_output


def create_sliding_window_mask(
    query_mask: torch.Tensor, local_attention: int = 128
) -> torch.Tensor:
    """Generates a 4D band-diagonal sliding window mask [B, 1, S, S] for SDPA."""
    seq_len = query_mask.shape[1]
    half_window = local_attention // 2
    row_idx = torch.arange(seq_len, device=query_mask.device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=query_mask.device).unsqueeze(0)
    band_mask = (row_idx - col_idx).abs() <= half_window
    key_mask = query_mask.unsqueeze(1).unsqueeze(2).bool()
    return key_mask & band_mask.unsqueeze(0).unsqueeze(0)


class ModernBertMiceTopLayer(nn.Module):
    """
    Top interaction layer for MICE: Cross-Attention + Self-Attention + MLP.
    """

    def __init__(self, config: ModernBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if getattr(config, "use_cross_attn", True):
            self.cross_attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
            if getattr(config, "top_layers_doc_ln", False):
                self.doc_cross_attn_norm = nn.LayerNorm(
                    config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
                )
            else:
                self.doc_cross_attn_norm = nn.Identity()

            self.cross_attn = ModernBertMiceCrossAttention(config, layer_idx=layer_idx)
        else:
            self.cross_attn_norm = None
            self.doc_cross_attn_norm = None
            self.cross_attn = None

        self.self_attn_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.self_attn = ModernBertAttention(config, layer_idx=layer_idx)

        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    def forward(
        self,
        q_hidden: torch.Tensor,
        d_hidden: Optional[torch.Tensor] = None,
        d_utilities: Optional[torch.Tensor] = None,
        q_mask: Optional[torch.Tensor] = None,
        d_mask: Optional[torch.Tensor] = None,
        q_attn_mask: Optional[torch.Tensor] = None,
        q_sw_attn_mask: Optional[torch.Tensor] = None,
        q_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_last_layer: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_d: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_d: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Cross Attention
        if self.cross_attn is not None and d_hidden is not None:
            norm_d_hidden = self.doc_cross_attn_norm(d_hidden)
            gate_target = getattr(self.config, "gate_target", "h")
            if gate_target == "h":
                if d_utilities is not None:
                    norm_d_hidden = norm_d_hidden * d_utilities.to(norm_d_hidden.dtype)
                cross_attn_utilities = None
            elif gate_target == "v":
                cross_attn_utilities = d_utilities
            else:
                raise ValueError(
                    f"Unknown gate_target: '{gate_target}'. Expected 'h' or 'v'."
                )

            cross_out = self.cross_attn(
                self.cross_attn_norm(q_hidden),
                norm_d_hidden,
                q_mask=q_mask,
                d_mask=d_mask,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_d=cu_seqlens_d,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_d=max_seqlen_d,
                d_utilities=cross_attn_utilities,
                **kwargs,
            )
            q_hidden = q_hidden + cross_out

        # 2. Self Attention
        if q_position_embeddings is None:
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            seq_len = q_hidden.shape[1]
            cos = torch.ones(
                (1, seq_len, head_dim), dtype=q_hidden.dtype, device=q_hidden.device
            )
            sin = torch.zeros(
                (1, seq_len, head_dim), dtype=q_hidden.dtype, device=q_hidden.device
            )
            q_position_embeddings = (cos, sin)

        if cu_seqlens_q is not None:
            # Unpadded Flash Attention 2 query self-attention
            self_out, _ = self.self_attn(
                self.self_attn_norm(q_hidden),
                position_embeddings=q_position_embeddings,
                attention_mask=None,
                cu_seq_lens_q=cu_seqlens_q.to(torch.int32),
                cu_seq_lens_k=cu_seqlens_q.to(torch.int32),
                max_length_q=(
                    int(max_seqlen_q) if max_seqlen_q is not None else q_hidden.shape[1]
                ),
                max_length_k=(
                    int(max_seqlen_q) if max_seqlen_q is not None else q_hidden.shape[1]
                ),
                **kwargs,
            )
        else:
            # Padded SDPA / Eager query self-attention
            target_mask = (
                q_sw_attn_mask
                if (
                    q_sw_attn_mask is not None
                    and getattr(self.self_attn, "sliding_window", None) is not None
                )
                else q_attn_mask
            )
            self_out, _ = self.self_attn(
                self.self_attn_norm(q_hidden),
                position_embeddings=q_position_embeddings,
                attention_mask=target_mask,
                **kwargs,
            )
        q_hidden = q_hidden + self_out

        # 3. MLP (Slice to CLS token on final top layer to optimize compute)
        if is_last_layer:
            if cu_seqlens_q is not None:
                # Unpadded 1D path: slice CLS tokens at cu_seqlens_q[:-1]
                cls_indices = cu_seqlens_q[:-1].long()
                q_hidden = q_hidden[:, cls_indices, :]  # Shape: [1, batch_size, D]
            else:
                # Padded 3D path: slice CLS token at index 0
                q_hidden = q_hidden[:, 0:1, :]  # Shape: [batch_size, 1, D]

        q_hidden = q_hidden + self.mlp(self.mlp_norm(q_hidden))

        return q_hidden


def _split_input_ids_vectorized(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sep_token_id: int = 50282,
    pad_token_id: int = 50283,
    cls_token_id: int = 50281,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically splits concatenated pair input_ids [B, S] into query and doc tensors.

    Respects exact tokenized sequence lengths without forcing arbitrary truncation/padding:
    - query_ids: [B, max_q_len_in_batch] starting with [CLS] ... [SEP]
    - doc_ids:   [B, max_d_len_in_batch] starting with [CLS] ... [SEP]
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    is_sep = input_ids == sep_token_id
    first_sep_idx = is_sep.long().argmax(dim=1, keepdim=True)

    seq_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    attn_bool = (
        attention_mask.bool()
        if attention_mask is not None
        else (input_ids != pad_token_id)
    )

    q_mask_raw = (seq_pos <= first_sep_idx) & attn_bool
    d_mask_raw = (seq_pos > first_sep_idx) & attn_bool

    # Compute dynamic max batch lengths
    q_lens = q_mask_raw.sum(dim=1)
    d_lens = d_mask_raw.sum(dim=1) + 1  # +1 for prepended CLS token

    max_q_len = int(q_lens.max().item())
    max_d_len = int(d_lens.max().item())

    # 1. Query extraction
    q_ids_raw = torch.where(q_mask_raw, input_ids, pad_token_id)
    q_ids = q_ids_raw[:, :max_q_len]
    q_mask = q_mask_raw[:, :max_q_len].long()

    # 2. Document extraction
    doc_offsets = seq_pos - (first_sep_idx + 1)

    d_ids = torch.full(
        (batch_size, max_d_len),
        pad_token_id,
        device=device,
        dtype=input_ids.dtype,
    )
    d_mask = torch.zeros((batch_size, max_d_len), device=device, dtype=torch.long)

    d_ids[:, 0] = cls_token_id
    d_mask[:, 0] = 1

    valid_doc_token_mask = (
        d_mask_raw & (doc_offsets >= 0) & (doc_offsets < (max_d_len - 1))
    )

    batch_indices, seq_indices = torch.where(valid_doc_token_mask)
    target_doc_indices = doc_offsets[batch_indices, seq_indices] + 1

    d_ids[batch_indices, target_doc_indices] = input_ids[batch_indices, seq_indices]
    d_mask[batch_indices, target_doc_indices] = 1

    return q_ids, q_mask, d_ids, d_mask


class ModernBertMiceForSequenceClassification(ModernBertPreTrainedModel):
    """
    HF-compatible Sequence Classification Model for MICE with ModernBERT backbone.
    Supports standard ST CrossEncoder forward pass as well as independent Q/D pass.
    Optimized for continuous 1D unpadded Flash Attention 2 execution when FA2 is available on CUDA.
    """

    config_class = ModernBertMiceConfig

    def __init__(self, config: ModernBertMiceConfig):
        attn_impl = getattr(config, "_attn_implementation", "sdpa")
        if "flash" in str(attn_impl).lower():
            fa2_avail = is_fa2_available()
            if not fa2_avail or not torch.cuda.is_available():
                logger.warning(
                    f"Flash Attention 2 requested ('{attn_impl}'), but is not supported in this environment "
                    f"(CUDA available: {torch.cuda.is_available()}, FA2 kernels available: {fa2_avail}). "
                    f"Falling back to `_attn_implementation='sdpa'`."
                )
                config._attn_implementation = "sdpa"

        # 1. Determine bottom and top layer counts
        n_context = config.n_contextualization_layers
        num_top = config.n_interaction_layers
        if num_top is None:
            num_top = config.num_hidden_layers - n_context

        # Update config total layers so num_hidden_layers represents total MICE layers
        config.num_hidden_layers = n_context + num_top
        if hasattr(config, "layer_types") and isinstance(
            config.layer_types, (list, tuple)
        ):
            if len(config.layer_types) != config.num_hidden_layers:
                multiplier = (
                    config.num_hidden_layers // max(1, len(config.layer_types))
                ) + 1
                config.layer_types = (list(config.layer_types) * multiplier)[
                    : config.num_hidden_layers
                ]

        super().__init__(config)
        self.config = config

        # 2. Build bottom ModernBertModel backbone with exactly n_contextualization_layers directly
        bottom_config = copy.deepcopy(config)
        bottom_config.num_hidden_layers = n_context
        if hasattr(bottom_config, "layer_types") and isinstance(
            bottom_config.layer_types, (list, tuple)
        ):
            bottom_config.layer_types = list(bottom_config.layer_types)[:n_context]

        self.model = ModernBertModel(bottom_config)

        # 3. Top interaction layers, adapter, and classification head
        self.build_top_layers()

        if getattr(config, "utility_pred", False):
            self.utility_predictor = UtilityPredictor(
                config.hidden_size,
                utility_temp=config.utility_temp,
                utility_bias=getattr(config, "utility_bias", 5.0),
                use_hard_concrete=getattr(config, "use_hard_concrete", False),
                hc_gamma=getattr(config, "hc_gamma", -0.1),
                hc_zeta=getattr(config, "hc_zeta", 1.1),
            )
        else:
            self.utility_predictor = None

        self.post_init()

    def build_top_layers(
        self,
        compress_dim: Optional[float] = None,
        compress_init_option: Optional[str] = None,
        n_interaction_layers: Optional[int] = None,
        head_type: Optional[str] = None,
        doc_bottleneck_dim: Optional[int] = None,
    ):
        """Builds (or rebuilds) the top interaction layers, adapter, top rotary embedding,
        document bottleneck, and classification head matching the specified or configured compression ratio."""
        if compress_dim is not None:
            self.config.compress_dim = compress_dim
        if compress_init_option is not None:
            self.config.compress_init_option = compress_init_option
        if n_interaction_layers is not None:
            self.config.n_interaction_layers = n_interaction_layers
        if head_type is not None:
            self.config.head_type = head_type
        if doc_bottleneck_dim is not None:
            self.config.doc_bottleneck_dim = doc_bottleneck_dim

        c_dim = getattr(self.config, "compress_dim", 1.0)
        if c_dim > 1.0:
            top_config = copy.deepcopy(self.config)
            top_config.hidden_size = int(self.config.hidden_size / c_dim)
            top_config.intermediate_size = int(self.config.intermediate_size / c_dim)
            top_config.num_attention_heads = int(
                self.config.num_attention_heads / c_dim
            )
            assert top_config.hidden_size % top_config.num_attention_heads == 0, (
                f"compress_dim={c_dim} produces top hidden_size="
                f"{top_config.hidden_size} not divisible by num_attention_heads="
                f"{top_config.num_attention_heads} (native hidden_size="
                f"{self.config.hidden_size}, native num_attention_heads="
                f"{self.config.num_attention_heads}). Choose a compress_dim that divides "
                f"evenly."
            )
        else:
            top_config = self.config
        self.top_config = top_config

        n_context = self.config.n_contextualization_layers
        num_top = self.config.n_interaction_layers
        if num_top is None:
            num_top = self.config.num_hidden_layers - n_context

        layer_offset = n_context
        self.top_layers = nn.ModuleList(
            [
                ModernBertMiceTopLayer(top_config, layer_idx=layer_offset + i)
                for i in range(num_top)
            ]
        )

        if top_config is not self.config:
            self.top_rotary_emb = ModernBertRotaryEmbedding(top_config)
            self.adapter = nn.Linear(self.config.hidden_size, top_config.hidden_size)
        else:
            self.top_rotary_emb = None
            self.adapter = None

        doc_bot_dim = getattr(self.config, "doc_bottleneck_dim", None)
        if doc_bot_dim is not None and doc_bot_dim > 0:
            self.doc_compress = nn.Linear(
                self.config.hidden_size, doc_bot_dim, bias=False
            )
            self.doc_decompress = nn.Linear(
                doc_bot_dim, top_config.hidden_size, bias=False
            )
        else:
            self.doc_compress = None
            self.doc_decompress = None

        head_dim = top_config.hidden_size
        num_labels = self.config.num_labels if hasattr(self.config, "num_labels") else 1
        head_type_val = getattr(self.config, "head_type", "linear")

        if head_type_val == "st_reranker":
            dense_inner = Dense(
                in_features=head_dim,
                out_features=head_dim,
                bias=False,
                activation_function=nn.GELU(),
                module_input_name="sentence_embedding",
                module_output_name="sentence_embedding",
            )
            norm = LayerNorm(dimension=head_dim)
            dense_score = Dense(
                in_features=head_dim,
                out_features=num_labels,
                bias=True,
                activation_function=nn.Identity(),
                module_input_name="sentence_embedding",
                module_output_name="scores",
            )
            self.output_modules = nn.Sequential(
                OrderedDict(
                    [
                        ("dense_inner", dense_inner),
                        ("norm", norm),
                        ("dense_score", dense_score),
                    ]
                )
            )
        else:
            modules = []
            if getattr(self.config, "top_final_norm", False):
                norm = LayerNorm(dimension=head_dim)
                norm.norm = nn.LayerNorm(
                    head_dim,
                    eps=getattr(self.config, "norm_eps", 1e-5),
                    bias=getattr(self.config, "norm_bias", False),
                )
                modules.append(("norm", norm))
            dense_score = Dense(
                in_features=head_dim,
                out_features=num_labels,
                bias=True,
                activation_function=nn.Identity(),
                module_input_name="sentence_embedding",
                module_output_name="scores",
            )
            modules.append(("dense_score", dense_score))
            self.output_modules = nn.Sequential(OrderedDict(modules))

        # Align device & dtype with bottom model if it exists
        if hasattr(self, "model") and hasattr(self.model, "parameters"):
            try:
                p = next(self.model.parameters())
                self.top_layers.to(device=p.device, dtype=p.dtype)
                if self.top_rotary_emb is not None:
                    self.top_rotary_emb.to(device=p.device, dtype=p.dtype)
                if self.adapter is not None:
                    self.adapter.to(device=p.device, dtype=p.dtype)
                if self.doc_compress is not None:
                    self.doc_compress.to(device=p.device, dtype=p.dtype)
                if self.doc_decompress is not None:
                    self.doc_decompress.to(device=p.device, dtype=p.dtype)
                self.output_modules.to(device=p.device, dtype=p.dtype)
            except StopIteration:
                pass

        # Initialize weights
        for layer in self.top_layers:
            self._init_weights(layer)
        if self.adapter is not None:
            self._init_weights(self.adapter)
        if self.doc_compress is not None:
            self._init_weights(self.doc_compress)
        if self.doc_decompress is not None:
            self._init_weights(self.doc_decompress)
        for m in self.output_modules.modules():
            if isinstance(m, nn.Linear):
                self._init_weights(m)

    def get_wv_matrices(self) -> List[torch.Tensor]:
        """Extract W_V weight matrices from top cross-attention (or self-attention) layers."""
        wv_list = []
        for top_layer in self.top_layers:
            attn_module = (
                top_layer.cross_attn
                if top_layer.cross_attn is not None
                else top_layer.self_attn
            )
            Wqkv = attn_module.Wqkv
            hidden_size = attn_module.config.hidden_size
            W_V = Wqkv.weight[2 * hidden_size : 3 * hidden_size, :]
            wv_list.append(W_V)
        return wv_list

    def get_wo_matrices(self) -> List[torch.Tensor]:
        """Extract W_O weight matrices from top cross-attention (or self-attention) layers."""
        return [
            (
                top_layer.cross_attn.Wo.weight
                if top_layer.cross_attn is not None
                else top_layer.self_attn.Wo.weight
            )
            for top_layer in self.top_layers
        ]

    @property
    def classifier(self):
        """Compatibility property returning the linear layer of dense_score."""
        if (
            hasattr(self, "output_modules")
            and "dense_score" in self.output_modules._modules
        ):
            return self.output_modules.dense_score.linear
        return None

    @property
    def top_final_norm(self):
        """Compatibility property returning the top final LayerNorm if top_final_norm is enabled."""
        if (
            getattr(self.config, "top_final_norm", False)
            and getattr(self.config, "head_type", "linear") == "linear"
        ):
            if (
                hasattr(self, "output_modules")
                and "norm" in self.output_modules._modules
            ):
                return self.output_modules.norm.norm
        return None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Remap legacy classifier.* and top_final_norm.* keys to output_modules for backward compatibility."""
        for old_suffix, new_suffix in [
            ("classifier.weight", "output_modules.dense_score.linear.weight"),
            ("classifier.bias", "output_modules.dense_score.linear.bias"),
            ("top_final_norm.weight", "output_modules.norm.norm.weight"),
            ("top_final_norm.bias", "output_modules.norm.norm.bias"),
        ]:
            old_key = prefix + old_suffix
            new_key = prefix + new_suffix
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, UtilityPredictor):
            module._init_weights()

    def _forward_bottom(
        self,
        sequence_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        pos_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """Process a single sequence (query or document) through bottom encoder layers.

        Args:
            sequence_input: 1D unpadded embeddings [Total_Tokens, Dim] if cu_seqlens is provided,
                           or 2D token IDs / 3D embeddings if cu_seqlens is None.
            mask: 2D attention mask [Batch, SeqLen] (used when cu_seqlens is None).
            cu_seqlens: Cumulative sequence lengths for 1D varlen FlashAttention.
            max_seqlen: Maximum sequence length in batch for 1D varlen FlashAttention.
            pos_ids: 1D unpadded position IDs for Rotary Position Embeddings.

        Returns:
            Tuple of (hidden_state, pos_emb_top)
        """
        # Top interaction layers may run at a compressed head_dim (compress_dim>1),
        # so pos_emb_top must come from a rotary module sized for top_config, not the
        # bottom backbone's own rotary_emb which stays at the native head_dim.
        rotary_for_top = (
            self.top_rotary_emb
            if self.top_rotary_emb is not None
            else self.model.rotary_emb
        )
        if cu_seqlens is not None:
            # Unpadded FlashAttention 2 Path
            position_embeddings = {
                l_type: self.model.rotary_emb(
                    sequence_input, pos_ids, layer_type=l_type
                )
                for l_type in set(self.config.layer_types)
            }
            hidden = sequence_input
            for encoder_layer in self.model.layers:
                layer_type = getattr(
                    encoder_layer,
                    "attention_type",
                    getattr(encoder_layer.attn, "layer_type", "full_attention"),
                )
                hidden = encoder_layer(
                    hidden,
                    position_embeddings=position_embeddings[layer_type],
                    attention_mask=None,
                    cu_seq_lens_q=cu_seqlens.to(torch.int32),
                    cu_seq_lens_k=cu_seqlens.to(torch.int32),
                    max_length_q=int(max_seqlen),
                    max_length_k=int(max_seqlen),
                    **kwargs,
                )
            if getattr(self.config, "bottom_final_norm", True):
                hidden = self.model.final_norm(hidden)

            pos_emb_top = rotary_for_top(
                sequence_input, pos_ids, layer_type="full_attention"
            )
            return hidden, pos_emb_top
        else:
            # Standard 3D Padded Path
            if getattr(self.config, "bottom_final_norm", True):
                if sequence_input.dtype in (torch.int32, torch.int64, torch.long):
                    outputs = self.model(
                        input_ids=sequence_input, attention_mask=mask, **kwargs
                    )
                else:
                    outputs = self.model(
                        inputs_embeds=sequence_input, attention_mask=mask, **kwargs
                    )

                last_hidden = outputs.last_hidden_state
            else:
                if sequence_input.dtype in (torch.int32, torch.int64, torch.long):
                    inputs_embeds = self.model.embeddings(input_ids=sequence_input)
                else:
                    inputs_embeds = sequence_input

                mask_kwargs = {
                    "config": self.config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": mask,
                }
                attention_mask_mapping = {
                    "full_attention": create_bidirectional_mask(**mask_kwargs),
                    "sliding_attention": create_bidirectional_sliding_window_mask(
                        **mask_kwargs
                    ),
                }

                position_ids = torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                ).unsqueeze(0)
                position_embeddings = {}
                for layer_type in set(self.config.layer_types):
                    position_embeddings[layer_type] = self.model.rotary_emb(
                        inputs_embeds, position_ids, layer_type
                    )

                hidden = inputs_embeds
                for encoder_layer in self.model.layers:
                    layer_type = getattr(
                        encoder_layer,
                        "attention_type",
                        getattr(encoder_layer.attn, "layer_type", "full_attention"),
                    )
                    hidden = encoder_layer(
                        hidden,
                        attention_mask=attention_mask_mapping[layer_type],
                        position_embeddings=position_embeddings[layer_type],
                        **kwargs,
                    )
                last_hidden = hidden

            position_ids = torch.arange(
                last_hidden.shape[1], device=last_hidden.device
            ).unsqueeze(0)
            pos_emb_top = rotary_for_top(
                last_hidden, position_ids, layer_type="full_attention"
            )
            return last_hidden, pos_emb_top

    def _forward_bottom_unified(
        self,
        query_ids: torch.LongTensor,
        query_mask: torch.Tensor,
        doc_ids: torch.LongTensor,
        doc_mask: torch.Tensor,
        use_unpadded: bool = False,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[int],
        Optional[int],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Any,
    ]:
        """Process both query and document tokens simultaneously through bottom encoder layers
        using a block-diagonal attention mask (for SDPA / Eager) or 2B-sequence varlen batch (for FA2).

        Returns:
            (q_hidden, d_hidden, cu_seqlens_q, cu_seqlens_d, max_seqlen_q, max_seqlen_d, q_mask, doc_mask, d_pos_ids, q_pos_emb_top)
        """
        rotary_for_top = (
            self.top_rotary_emb
            if self.top_rotary_emb is not None
            else self.model.rotary_emb
        )

        if use_unpadded:
            # 1. Unpadded FlashAttention-2 Path: concatenate 1D varlen sequences into 2B total sequences
            q_embeds = self.model.embeddings(input_ids=query_ids)
            (
                q_unpadded,
                cu_seqlens_q,
                max_seqlen_q,
                indices_q,
                q_pos_ids,
            ) = _unpad_input_with_posids(q_embeds, query_mask)

            d_embeds = self.model.embeddings(input_ids=doc_ids)
            (
                d_unpadded,
                cu_seqlens_d,
                max_seqlen_d,
                indices_d,
                d_pos_ids,
            ) = _unpad_input_with_posids(d_embeds, doc_mask)

            total_q = q_unpadded.shape[1]

            unified_tokens = torch.cat([q_unpadded, d_unpadded], dim=1)
            unified_pos_ids = torch.cat([q_pos_ids, d_pos_ids], dim=1)
            cu_seqlens_unified = torch.cat([cu_seqlens_q, cu_seqlens_d[1:] + total_q])
            max_seqlen = max(max_seqlen_q, max_seqlen_d)

            unified_hidden, _ = self._forward_bottom(
                unified_tokens,
                cu_seqlens=cu_seqlens_unified,
                max_seqlen=max_seqlen,
                pos_ids=unified_pos_ids,
                **kwargs,
            )

            q_hidden = unified_hidden[:, :total_q, :]
            d_hidden = unified_hidden[:, total_q:, :]

            if getattr(self.config, "use_upper_rope", True):
                q_pos_emb_top = rotary_for_top(
                    q_hidden, q_pos_ids, layer_type="full_attention"
                )
            else:
                q_pos_emb_top = None

            return (
                q_hidden,
                d_hidden,
                cu_seqlens_q,
                cu_seqlens_d,
                max_seqlen_q,
                max_seqlen_d,
                query_mask,
                doc_mask,
                d_pos_ids,
                q_pos_emb_top,
            )

        else:
            # 2. Padded SDPA / Eager Path: pack contiguously, compute block mask and block position IDs
            pad_id = getattr(self.config, "pad_token_id", 50283)
            unified_ids, q_lens, d_lens, max_joint_len = _pack_pair_inputs_padded(
                query_ids, query_mask, doc_ids, doc_mask, pad_token_id=pad_id
            )

            local_attn = getattr(self.config, "local_attention", 128)
            attn_mask_mapping = _create_mice_block_masks(
                q_lens=q_lens,
                d_lens=d_lens,
                seq_len=max_joint_len,
                local_attention=local_attn,
                device=unified_ids.device,
            )

            pos_ids = _create_block_position_ids(
                q_lens=q_lens,
                d_lens=d_lens,
                seq_len=max_joint_len,
                device=unified_ids.device,
            )

            if getattr(self.config, "bottom_final_norm", True):
                outputs = self.model(
                    input_ids=unified_ids,
                    attention_mask=attn_mask_mapping,
                    position_ids=pos_ids,
                    **kwargs,
                )
                unified_hidden = outputs.last_hidden_state
            else:
                inputs_embeds = self.model.embeddings(input_ids=unified_ids)
                position_embeddings = {}
                for layer_type in set(self.config.layer_types):
                    position_embeddings[layer_type] = self.model.rotary_emb(
                        inputs_embeds, pos_ids, layer_type
                    )
                hidden = inputs_embeds
                for encoder_layer in self.model.layers:
                    layer_type = getattr(
                        encoder_layer,
                        "attention_type",
                        getattr(encoder_layer.attn, "layer_type", "full_attention"),
                    )
                    hidden = encoder_layer(
                        hidden,
                        attention_mask=attn_mask_mapping[layer_type],
                        position_embeddings=position_embeddings[layer_type],
                        **kwargs,
                    )
                unified_hidden = hidden

            B = unified_ids.shape[0]
            max_q_len = int(q_lens.max().item())
            max_d_len = int(d_lens.max().item())

            col_idx_q = torch.arange(max_q_len, device=unified_ids.device).unsqueeze(0)
            new_q_mask = (col_idx_q < q_lens.unsqueeze(1)).long()
            q_hidden = unified_hidden[:, :max_q_len, :] * new_q_mask.unsqueeze(-1)

            col_idx_d = torch.arange(max_d_len, device=unified_ids.device).unsqueeze(0)
            new_d_mask = (col_idx_d < d_lens.unsqueeze(1)).long()
            source_cols = q_lens.unsqueeze(1) + col_idx_d
            valid_d = new_d_mask.bool()
            batch_idx = (
                torch.arange(B, device=unified_ids.device)
                .unsqueeze(1)
                .expand_as(source_cols)
            )

            d_hidden = torch.zeros(
                (B, max_d_len, unified_hidden.shape[-1]),
                dtype=unified_hidden.dtype,
                device=unified_hidden.device,
            )
            d_hidden[valid_d] = unified_hidden[batch_idx[valid_d], source_cols[valid_d]]

            if getattr(self.config, "use_upper_rope", True):
                q_position_ids = torch.arange(
                    q_hidden.shape[1], device=q_hidden.device
                ).unsqueeze(0)
                q_pos_emb_top = rotary_for_top(
                    q_hidden, q_position_ids, layer_type="full_attention"
                )
            else:
                q_pos_emb_top = None

            return (
                q_hidden,
                d_hidden,
                None,
                None,
                None,
                None,
                new_q_mask,
                new_d_mask,
                None,
                q_pos_emb_top,
            )

    def freeze_utility(self):
        """Freezes utility predictor parameters."""
        if self.utility_predictor is not None:
            for param in self.utility_predictor.parameters():
                param.requires_grad = False
            logger.info("[MICE v2] Frozen utility predictor.")
        else:
            logger.debug(
                "[MICE v2] freeze_utility called but utility_predictor is None."
            )

    def freeze_bottom_layers(self):
        """Freezes bottom contextualization layers, embeddings, and rotary embeddings."""
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        for param in self.model.layers.parameters():
            param.requires_grad = False

        if hasattr(self.model, "rotary_emb"):
            for param in self.model.rotary_emb.parameters():
                param.requires_grad = False

        if hasattr(self.model, "final_norm") and self.model.final_norm is not None:
            for param in self.model.final_norm.parameters():
                param.requires_grad = False

        logger.info("[MICE v2] Frozen embeddings and bottom contextualization layers.")

    def freeze_top_layers(self):
        """Freezes top interaction layers, rotary embeddings, adapter, and classification head,
        leaving document compression/decompression bottleneck parameters trainable."""
        for param in self.top_layers.parameters():
            param.requires_grad = False
        if self.top_rotary_emb is not None:
            for param in self.top_rotary_emb.parameters():
                param.requires_grad = False
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = False
        for param in self.output_modules.parameters():
            param.requires_grad = False
        logger.info("[MICE v2] Frozen top layers and classification head.")

    def _document_embed_post_process(
        self,
        d_hidden: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
        use_unpadded: bool = False,
        cu_seqlens_d: Optional[torch.Tensor] = None,
        max_seqlen_d: Optional[int] = None,
        d_pos_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[int],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Postprocessing of the contextualized token representations:
        First centering the hs and L2 norm for FastPLAID index and calculating
        token utilities and applies soft gating or hard token pruning.
        Returns updated (d_hidden, doc_mask, cu_seqlens_d, max_seqlen_d, d_pos_ids,
        utilities, p_active, d_hidden_pre_gate, pruned_utilities).
        """
        if getattr(self.config, "normalize_doc_embeds", False):
            d_hidden = d_hidden - torch.mean(d_hidden, dim=-1, keepdim=True)
            d_hidden = torch.nn.functional.normalize(d_hidden, p=2, dim=-1)

        d_hidden_pre_gate = d_hidden
        utilities = None
        p_active = None
        pruned_utilities = None

        if self.utility_predictor is not None:
            hard_prune = self.config.hard_pruning
            threshold = (
                0.0
                if getattr(self.config, "use_hard_concrete", False)
                else self.config.hard_prune_threshold
            )
            utilities, p_active = self.utility_predictor(d_hidden, return_p_active=True)

            if hard_prune:
                if use_unpadded and cu_seqlens_d is not None:
                    (
                        d_hidden,
                        d_pos_ids,
                        cu_seqlens_d,
                        max_seqlen_d,
                        pruned_utils,
                    ) = _hard_prune_unpadded_documents(
                        d_hidden,
                        d_pos_ids,
                        cu_seqlens_d,
                        utilities,
                        threshold=threshold,
                        return_pruned_utilities=True,
                    )
                else:
                    if doc_mask is None:
                        doc_mask = torch.ones(
                            (d_hidden.shape[0], d_hidden.shape[1]),
                            dtype=torch.bool,
                            device=d_hidden.device,
                        )
                    (
                        d_hidden,
                        doc_mask,
                        d_pos_ids,
                        pruned_utils,
                    ) = _hard_prune_padded_documents(
                        d_hidden,
                        doc_mask,
                        utilities,
                        threshold=threshold,
                        return_pruned_utilities=True,
                    )
                pruned_utilities = pruned_utils
                gate_target = getattr(self.config, "gate_target", "h")
                if gate_target == "h" and not getattr(
                    self.config, "top_layers_doc_ln", False
                ):
                    d_hidden = d_hidden * pruned_utilities.to(d_hidden.dtype)
            else:
                gate_target = getattr(self.config, "gate_target", "h")
                if gate_target == "h":
                    if not getattr(self.config, "top_layers_doc_ln", False):
                        # Legacy soft gating: multiply in post-processing
                        d_hidden = d_hidden * utilities.to(d_hidden.dtype)
                    else:
                        # Defer soft gating to top_layers (applied post-doc_cross_attn_norm)
                        pass
                elif gate_target == "v":
                    # Value-only gating: defer gating to cross-attention Value projections
                    pass
                else:
                    raise ValueError(
                        f"Unknown gate_target: '{gate_target}'. Expected 'h' or 'v'."
                    )

        if self.doc_compress is not None:
            d_hidden = self.doc_compress(d_hidden)
        elif self.adapter is not None:
            d_hidden = self.adapter(d_hidden)

        return (
            d_hidden,
            doc_mask,
            cu_seqlens_d,
            max_seqlen_d,
            d_pos_ids,
            utilities,
            p_active,
            d_hidden_pre_gate,
            pruned_utilities,
        )

    def encode_documents(
        self,
        doc_ids: torch.LongTensor,
        doc_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode documents through bottom contextualization layers.

        Returns (d_hidden, doc_mask): doc_mask is returned alongside d_hidden because
        hard pruning shortens d_hidden's sequence dimension, so callers need the
        post-pruning mask rather than the original tokenizer mask to know which
        positions are real tokens.
        """
        d_hidden, _ = self._forward_bottom(doc_ids, mask=doc_mask, **kwargs)
        d_hidden, doc_mask, _, _, _, _, _, _, _ = self._document_embed_post_process(
            d_hidden, doc_mask, use_unpadded=False
        )
        return d_hidden, doc_mask

    def _query_embed_post_process(self, q_hidden: torch.Tensor) -> torch.Tensor:
        """Mirrors `_document_embed_post_process`'s normalization + adapter
        projection, but never applies pruning/utility-gating -- those only
        ever make sense for documents, regardless of the pruning strategy."""
        if getattr(self.config, "normalize_doc_embeds", False):
            q_hidden = q_hidden - torch.mean(q_hidden, dim=-1, keepdim=True)
            q_hidden = torch.nn.functional.normalize(q_hidden, p=2, dim=-1)

        if self.adapter is not None:
            q_hidden = self.adapter(q_hidden)

        return q_hidden

    def encode_queries(
        self,
        query_ids: torch.LongTensor,
        query_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode queries through bottom contextualization layers.

        Returns (q_hidden, query_mask): query_mask is simply passed through
        unchanged (queries are never pruned, unlike documents) -- kept for
        interface symmetry with `encode_documents`.
        """
        q_hidden, _ = self._forward_bottom(query_ids, mask=query_mask, **kwargs)
        q_hidden = self._query_embed_post_process(q_hidden)
        return q_hidden, query_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_ids: Optional[torch.LongTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
        doc_hidden: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "return_dict", True)
        )

        # Handle ST concatenated inputs if query_ids is not passed
        if (
            query_ids is None or (doc_ids is None and doc_hidden is None)
        ) and input_ids is not None:
            query_ids, query_mask, doc_ids, doc_mask = _split_input_ids_vectorized(
                input_ids,
                attention_mask,
                sep_token_id=self.config.sep_token_id,
                pad_token_id=self.config.pad_token_id,
                cls_token_id=self.config.cls_token_id,
            )

        if query_mask is None and query_ids is not None:
            query_mask = torch.ones_like(query_ids, dtype=torch.long)
        if doc_mask is None and doc_ids is not None:
            doc_mask = torch.ones_like(doc_ids, dtype=torch.long)

        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        is_flash_impl = "flash" in str(attn_impl).lower()
        use_unpadded = (
            is_flash_impl
            and (query_ids.is_cuda if query_ids is not None else False)
            and is_fa2_available()
            and _unpad_input is not None
        )

        utilities = None
        p_active = None
        d_hidden_pre_gate = None
        pruned_utilities = None

        if doc_hidden is not None:
            # Path A: Pre-encoded document embeddings (from index / cache / encode_documents).
            # 1. Query Bottom Layers
            if use_unpadded:
                q_embeds = self.model.embeddings(input_ids=query_ids)
                (
                    q_unpadded,
                    cu_seqlens_q,
                    max_seqlen_q,
                    indices_q,
                    q_pos_ids,
                ) = _unpad_input_with_posids(q_embeds, query_mask)
                q_hidden, q_pos_emb_top = self._forward_bottom(
                    q_unpadded,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen_q,
                    pos_ids=q_pos_ids,
                    **kwargs,
                )
            else:
                if (
                    getattr(self.config, "_attn_implementation", "sdpa")
                    == "flash_attention_2"
                    and not query_ids.is_cuda
                ):
                    logger.warning(
                        "[MICE v2] Flash Attention 2 requested but inputs are not on CUDA. "
                        "Falling back to standard 3D padded pipeline."
                    )
                q_hidden, q_pos_emb_top = self._forward_bottom(
                    query_ids, mask=query_mask, **kwargs
                )
                cu_seqlens_q = max_seqlen_q = None

            if not getattr(self.config, "use_upper_rope", True):
                q_pos_emb_top = None

            # 2. Document Representation Pipeline
            if doc_mask is None:
                doc_mask = torch.ones(
                    (doc_hidden.shape[0], doc_hidden.shape[1]),
                    dtype=torch.bool,
                    device=doc_hidden.device,
                )

            if use_unpadded:
                (
                    d_hidden,
                    cu_seqlens_d,
                    max_seqlen_d,
                    _,
                    d_pos_ids,
                ) = _unpad_input_with_posids(doc_hidden, doc_mask)
            else:
                d_hidden = doc_hidden
                cu_seqlens_d = max_seqlen_d = None
                d_pos_ids = None

            if self.doc_decompress is not None:
                d_hidden = self.doc_decompress(d_hidden)
        else:
            # Path B: Raw text token IDs (end-to-end training or standard inference without cache).
            if getattr(self.config, "unified_bottom_pass", True):
                (
                    q_hidden,
                    d_hidden,
                    cu_seqlens_q,
                    cu_seqlens_d,
                    max_seqlen_q,
                    max_seqlen_d,
                    query_mask,
                    doc_mask,
                    d_pos_ids,
                    q_pos_emb_top,
                ) = self._forward_bottom_unified(
                    query_ids=query_ids,
                    query_mask=query_mask,
                    doc_ids=doc_ids,
                    doc_mask=doc_mask,
                    use_unpadded=use_unpadded,
                    **kwargs,
                )
            else:
                # Legacy separate bottom passes
                if use_unpadded:
                    q_embeds = self.model.embeddings(input_ids=query_ids)
                    (
                        q_unpadded,
                        cu_seqlens_q,
                        max_seqlen_q,
                        indices_q,
                        q_pos_ids,
                    ) = _unpad_input_with_posids(q_embeds, query_mask)
                    q_hidden, q_pos_emb_top = self._forward_bottom(
                        q_unpadded,
                        cu_seqlens=cu_seqlens_q,
                        max_seqlen=max_seqlen_q,
                        pos_ids=q_pos_ids,
                        **kwargs,
                    )

                    d_embeds = self.model.embeddings(input_ids=doc_ids)
                    (
                        d_unpadded,
                        cu_seqlens_d,
                        max_seqlen_d,
                        indices_d,
                        d_pos_ids,
                    ) = _unpad_input_with_posids(d_embeds, doc_mask)
                    d_hidden, _ = self._forward_bottom(
                        d_unpadded,
                        cu_seqlens=cu_seqlens_d,
                        max_seqlen=max_seqlen_d,
                        pos_ids=d_pos_ids,
                        **kwargs,
                    )
                else:
                    if (
                        getattr(self.config, "_attn_implementation", "sdpa")
                        == "flash_attention_2"
                        and not query_ids.is_cuda
                    ):
                        logger.warning(
                            "[MICE v2] Flash Attention 2 requested but inputs are not on CUDA. "
                            "Falling back to standard 3D padded pipeline."
                        )
                    q_hidden, q_pos_emb_top = self._forward_bottom(
                        query_ids, mask=query_mask, **kwargs
                    )
                    cu_seqlens_q = max_seqlen_q = None
                    d_hidden, _ = self._forward_bottom(doc_ids, mask=doc_mask, **kwargs)
                    cu_seqlens_d = max_seqlen_d = None
                    d_pos_ids = None

            if not getattr(self.config, "use_upper_rope", True):
                q_pos_emb_top = None

            # 2b. Full Document Post-Processing:
            (
                d_hidden,
                doc_mask,
                cu_seqlens_d,
                max_seqlen_d,
                d_pos_ids,
                utilities,
                p_active,
                d_hidden_pre_gate,
                pruned_utilities,
            ) = self._document_embed_post_process(
                d_hidden,
                doc_mask,
                use_unpadded=use_unpadded,
                cu_seqlens_d=cu_seqlens_d,
                max_seqlen_d=max_seqlen_d,
                d_pos_ids=d_pos_ids if use_unpadded else None,
            )

            # 2c. If bottleneck is active, decompress from storage dim d to interaction dim H_top.
            if self.doc_decompress is not None:
                d_hidden = self.doc_decompress(d_hidden)

        # 3. Top Cross-Attention Utilities (for Value-gating, if active)
        # Pre-encoded doc_hidden already baked in soft-gating or physical pruning during indexing.
        if doc_hidden is not None:
            d_utilities = None
        elif self.utility_predictor is not None:
            if getattr(self.config, "hard_pruning", False):
                if getattr(self.config, "use_hard_concrete", False):
                    # Hard Concrete Value-gated slicing: pass pruned utilities to top layers
                    d_utilities = (
                        pruned_utilities
                        if (
                            getattr(self.config, "top_layers_doc_ln", False)
                            or getattr(self.config, "gate_target", "h") == "v"
                        )
                        else None
                    )
                else:
                    # Legacy Phase 2 hard pruning: utilities discarded in top layers
                    d_utilities = None
            else:
                # Soft gating or unpruned ablation
                d_utilities = (
                    utilities
                    if (
                        getattr(self.config, "top_layers_doc_ln", False)
                        or getattr(self.config, "gate_target", "h") == "v"
                    )
                    else None
                )
        else:
            d_utilities = None

        # apply the projection for the query.
        # document are done in the post processing.
        if self.adapter is not None:
            q_hidden = self.adapter(q_hidden)

        rotary_for_top = (
            self.top_rotary_emb
            if self.top_rotary_emb is not None
            else self.model.rotary_emb
        )
        if not self.config.use_cross_attn:
            # When use_cross_attn is False (Concat-Self Attention mode):
            # 1. Apply any pending soft gating utilities to document representations
            if d_utilities is not None:
                if self.config.gate_target == "v":
                    logger.warning(
                        "[ModernBertMiceModel] gate_target='v' was configured but use_cross_attn=False (Concat-Self Attention mode). "
                        "Falling back to gating d_hidden directly."
                    )
                d_hidden = d_hidden * d_utilities.to(d_hidden.dtype)

            # 2. Concatenate [query, doc] into a single joint sequence [CLS] q [SEP] d [SEP]
            #    and recompute RoPE embeddings across the entire concatenated sequence.
            if use_unpadded:
                (
                    q_hidden,
                    cu_seqlens_q,
                    max_seqlen_q,
                    pos_ids_joint,
                ) = _concat_query_doc_unpadded(
                    q_hidden, d_hidden, cu_seqlens_q, cu_seqlens_d
                )
                if getattr(self.config, "use_upper_rope", True):
                    q_pos_emb_top = rotary_for_top(
                        q_hidden, pos_ids_joint, layer_type="full_attention"
                    )
                else:
                    q_pos_emb_top = None
                # Set d_hidden to None: signals top_layer to skip cross-attention and run
                # self-attention exclusively over the concatenated sequence in q_hidden.
                d_hidden = None
                cu_seqlens_d = None
                max_seqlen_d = None
            else:
                q_hidden, query_mask = _concat_query_doc_padded(
                    q_hidden, d_hidden, query_mask, doc_mask
                )
                joint_position_ids = torch.arange(
                    q_hidden.shape[1], device=q_hidden.device
                ).unsqueeze(0)
                if getattr(self.config, "use_upper_rope", True):
                    q_pos_emb_top = rotary_for_top(
                        q_hidden, joint_position_ids, layer_type="full_attention"
                    )
                else:
                    q_pos_emb_top = None
                # Set d_hidden to None to skip cross-attention in top_layer
                d_hidden = None
                doc_mask = None

        if use_unpadded:
            num_top = len(self.top_layers)
            for i, top_layer in enumerate(self.top_layers):
                q_hidden = top_layer(
                    q_hidden,
                    d_hidden,
                    d_utilities=d_utilities,
                    q_position_embeddings=q_pos_emb_top,
                    is_last_layer=(i == num_top - 1),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_d=cu_seqlens_d,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_d=max_seqlen_d,
                    **kwargs,
                )
        else:
            q_attn_mask = None
            q_sw_attn_mask = None
            if query_mask is not None:
                q_attn_mask = create_bidirectional_mask(
                    config=self.config,
                    inputs_embeds=q_hidden,
                    attention_mask=query_mask,
                )
                local_attn = getattr(self.config, "local_attention", 128)
                if local_attn is not None and local_attn > 0:
                    q_sw_attn_mask = create_sliding_window_mask(
                        query_mask=query_mask,
                        local_attention=local_attn,
                    )

            num_top = len(self.top_layers)
            for i, top_layer in enumerate(self.top_layers):
                q_hidden = top_layer(
                    q_hidden,
                    d_hidden,
                    d_utilities=d_utilities,
                    q_mask=query_mask,
                    d_mask=doc_mask,
                    q_attn_mask=q_attn_mask,
                    q_sw_attn_mask=q_sw_attn_mask,
                    q_position_embeddings=q_pos_emb_top,
                    is_last_layer=(i == num_top - 1),
                )

        if use_unpadded:
            cls_hidden = q_hidden.squeeze(0)  # Shape: [batch_size, D]
        else:
            cls_hidden = q_hidden[:, 0]  # Shape: [batch_size, D]

        features = {"sentence_embedding": cls_hidden}
        output_features = self.output_modules(features)
        logits = output_features["scores"]

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels.squeeze(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ModernBertMiceOutput(
            loss=loss,
            logits=logits,
            utilities=utilities,
            p_active=p_active,
            doc_hidden=d_hidden_pre_gate,
            pruned_utilities=pruned_utilities,
        )


AutoConfig.register("modernbert_mice", ModernBertMiceConfig)
AutoModel.register(ModernBertMiceConfig, ModernBertMiceForSequenceClassification)
AutoModelForSequenceClassification.register(
    ModernBertMiceConfig, ModernBertMiceForSequenceClassification
)

try:
    ModernBertMiceConfig.register_for_auto_class("AutoConfig")
    ModernBertMiceForSequenceClassification.register_for_auto_class("AutoModel")
    ModernBertMiceForSequenceClassification.register_for_auto_class(
        "AutoModelForSequenceClassification"
    )
except AttributeError:
    # When transformers is mocked in pre_experiment during experiment orchestration,
    # base classes are replaced with _FakeBaseClass which lacks register_for_auto_class
    pass
