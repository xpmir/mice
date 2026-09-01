import torch
from torch import nn
from typing import Optional
from transformers import AutoModelForCausalLM

from experimaestro import field, Param, LightweightTask
from xpm_torch.utils import to_device
from xpmir.letor.records import BaseItems

from .mice import MiceCrossEncoder, MICETokenizedTexts

import logging

logger = logging.getLogger(__name__)


class QwenCrossAttention(nn.Module):
    """
    Cross-attention layer for Qwen models, designed to attend to an external
    encoder's hidden states (e.g., a document representation in MICE).
    Supports Qwen3 features like q_norm, k_norm, and output gating.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        # Determine if bias is needed (Qwen2 uses bias, Qwen3 does not for Q/K/V)
        has_bias = getattr(config, "attention_bias", True)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=has_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=has_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=has_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Output gating (Qwen3)
        self.gate_proj = None
        if hasattr(config, "model_type") and config.model_type == "qwen3":
            self.gate_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )

        # Qwen3 specific: RMSNorm for Q and K
        # We use Qwen2RMSNorm if available, or try to import it
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as RMSNorm
        except ImportError:
            try:
                from transformers.models.qwen3.modeling_qwen3 import (
                    Qwen3RMSNorm as RMSNorm,
                )
            except ImportError:
                # Fallback implementation of RMSNorm if not found
                class RMSNorm(nn.Module):
                    def __init__(self, dim, eps=1e-6):
                        super().__init__()
                        self.eps = eps
                        self.weight = nn.Parameter(torch.ones(dim))

                    def forward(self, x):
                        return (
                            x
                            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                            * self.weight
                        )

        self.q_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if hasattr(config, "model_type") and config.model_type == "qwen3"
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if hasattr(config, "model_type") and config.model_type == "qwen3"
            else nn.Identity()
        )

    def forward(
        self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(encoder_hidden_states).view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(encoder_hidden_states).view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        )

        # Apply q_norm and k_norm if they are not Identity
        if not isinstance(self.q_norm, nn.Identity):
            query_states = self.q_norm(query_states)
        if not isinstance(self.k_norm, nn.Identity):
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Standard GQA: repeat key/value states to match query heads
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(
                self.num_key_value_groups, dim=1
            )

        # Using scaled_dot_product_attention (PyTorch 2.0+)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=encoder_attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        # Output gating (Qwen3)
        if self.gate_proj is not None:
            gate = self.gate_proj(hidden_states)
            attn_output = attn_output * torch.nn.functional.silu(gate)

        attn_output = self.o_proj(attn_output)

        return attn_output


class QwenCrossAttentionLayer(nn.Module):
    """
    A Qwen decoder layer augmented with cross-attention, following the MICE architecture.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine the attention class to use (Qwen2 or Qwen3)
        if hasattr(config, "model_type") and config.model_type == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import (
                Qwen3Attention,
                Qwen3RMSNorm,
                Qwen3MLP,
                Qwen3RotaryEmbedding,
            )

            self.input_layernorm = Qwen3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attn = Qwen3Attention(config, layer_idx)
            self.post_attention_layernorm = Qwen3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = Qwen3MLP(config)
            self.rotary_emb = Qwen3RotaryEmbedding(config)
        else:
            from transformers.models.qwen2.modeling_qwen2 import (
                Qwen2Attention,
                Qwen2RMSNorm,
                Qwen2MLP,
                Qwen2RotaryEmbedding,
            )

            self.input_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attn = Qwen2Attention(config, layer_idx)
            self.post_attention_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = Qwen2MLP(config)
            self.rotary_emb = Qwen2RotaryEmbedding(config)

        self.cross_attn = QwenCrossAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    ):
        # Prepare position embeddings for RoPE if not provided
        if position_embeddings is None and position_ids is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 1. Self-Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # 2. Cross-Attention Block (Mid-Fusion)
        if encoder_hidden_states is not None:
            residual = hidden_states
            # MICE: attend to document states using query states
            # We use the same pre-norm (input_layernorm) for query and doc
            hidden_states = self.cross_attn(
                self.input_layernorm(hidden_states),
                encoder_hidden_states=self.input_layernorm(encoder_hidden_states),
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states

        # 3. MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class QwenMiceCrossEncoder(MiceCrossEncoder):
    """Mid-Fusion Cross Encoder based on Qwen Architecture."""

    pooling_method: Param[Optional[str]] = field(default="cls", ignore_default=True)
    """Pooling method to use: cls or mean."""

    def __initialize__(self):
        super().__initialize__()

        # Structure setup
        # Weights copied later by InitTask
        temp_model = AutoModelForCausalLM.from_config(self.config)
        self.add_module("embeddings", temp_model.model.embed_tokens)

        # Build Bottom layers, either bound or distinct
        if self.bound_bottom_layers:
            if (
                self.n_docs_ctx_layers is not None
                and self.n_docs_ctx_layers != self.n_contextualization_layers
            ):
                raise ValueError(
                    "n_docs_ctx_layers must be equal to n_contextualization_layers if bound_bottom_layers is True"
                )
            self.add_module(
                "bottom_layers",
                nn.ModuleList(
                    [
                        type(temp_model.model.layers[0])(self.config, i)
                        for i in range(self.n_contextualization_layers)
                    ]
                ),
            )
        else:
            self.add_module(
                "query_bottom_layers",
                nn.ModuleList(
                    [
                        type(temp_model.model.layers[0])(self.config, i)
                        for i in range(self.n_contextualization_layers)
                    ]
                ),
            )
            n_doc_layers = (
                self.n_docs_ctx_layers
                if self.n_docs_ctx_layers is not None
                else self.n_contextualization_layers
            )
            self.add_module(
                "document_bottom_layers",
                nn.ModuleList(
                    [
                        type(temp_model.model.layers[0])(self.config, i)
                        for i in range(n_doc_layers)
                    ]
                ),
            )

        # Top layers: augmented with cross-attention
        if self.n_interaction_layers is not None:
            num_top = self.n_interaction_layers
        else:
            num_top = len(temp_model.model.layers) - self.n_contextualization_layers

        self.add_module(
            "top_layers",
            nn.ModuleList(
                [
                    QwenCrossAttentionLayer(
                        self.config, self.n_contextualization_layers + i
                    )
                    for i in range(num_top)
                ]
            ),
        )

        if self.extra_attn_bias:
            from .mice import ExactMatchAttentionHead

            self.add_module(
                "exact_match_heads",
                nn.ModuleList(
                    [
                        ExactMatchAttentionHead(self.config.hidden_size)
                        for _ in range(num_top)
                    ]
                ),
            )

        self.add_module("final_norm", temp_model.model.norm)

        if self.global_cls_token:
            self.global_cls = nn.Parameter(
                torch.randn(1, 1, self.config.hidden_size) * 0.02
            )

        # Rotary embeddings for the whole model
        if hasattr(self.config, "model_type") and self.config.model_type == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

            self.rotary_emb = Qwen3RotaryEmbedding(self.config)
        else:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

            self.rotary_emb = Qwen2RotaryEmbedding(self.config)

        self.dropout_layer = nn.Dropout(getattr(self.config, "classifier_dropout", 0.1))
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward_vanilla_transformer(
        self, x_q, x_d, cross_mask, q_pos, q_position_embeddings, exact_match_mask=None
    ):
        """Vanilla transformer architecture: Self-Attention followed by Cross-Attention."""
        num_top = len(self.top_layers)
        for i, layer in enumerate(self.top_layers):
            if self.extra_attn_bias and exact_match_mask is not None:
                exact_match_out = self.exact_match_heads[i](x_d, exact_match_mask)
                x_q = x_q + exact_match_out

            if i == num_top - 1 and self.pooling_method == "cls":
                # Optimized last layer: Self-Attention on full sequence,
                # then slice to CLS for Cross-Attention and MLP.

                # 1. Self-Attention Block (Full)
                residual = x_q
                normed_x_q = layer.input_layernorm(x_q)
                attn_out, _ = layer.self_attn(
                    normed_x_q,
                    attention_mask=None,
                    position_ids=q_pos,
                    position_embeddings=q_position_embeddings,
                )
                x_q = residual + attn_out

                # 2. Slice to CLS
                x_q = x_q[:, 0:1, :]

                # 3. Cross-Attention Block (CLS only)
                if x_d is not None and not self.mask_cls_to_doc:
                    residual = x_q
                    normed_x_q = layer.input_layernorm(x_q)
                    x_q = layer.cross_attn(
                        normed_x_q,
                        encoder_hidden_states=layer.input_layernorm(x_d),
                        encoder_attention_mask=cross_mask[:, :, 0:1, :],
                    )
                    x_q = residual + x_q

                # 4. MLP Block (CLS only)
                residual = x_q
                x_q = layer.mlp(layer.post_attention_layernorm(x_q))
                x_q = residual + x_q
            else:
                x_q = layer(
                    x_q,
                    attention_mask=None,
                    encoder_hidden_states=x_d,
                    encoder_attention_mask=cross_mask,
                    position_ids=q_pos,
                    position_embeddings=q_position_embeddings,
                )[0]
        return x_q

    def forward_inverted_transformer(
        self, x_q, x_d, cross_mask, q_pos, q_position_embeddings, exact_match_mask=None
    ):
        """Inverted transformer architecture: Cross-Attention followed by Self-Attention."""
        num_top = len(self.top_layers)
        for i, layer in enumerate(self.top_layers):
            if self.extra_attn_bias and exact_match_mask is not None:
                exact_match_out = self.exact_match_heads[i](x_d, exact_match_mask)
                x_q = x_q + exact_match_out

            if i == num_top - 1 and self.pooling_method == "cls":
                # Optimized last layer: Cross-Attention on full sequence,
                # then Self-Attention, then slice to CLS for MLP.

                # 1. Cross-Attention Block (Full)
                if x_d is not None:
                    residual = x_q
                    normed_x_q = layer.input_layernorm(x_q)
                    x_q = layer.cross_attn(
                        normed_x_q,
                        encoder_hidden_states=layer.input_layernorm(x_d),
                        encoder_attention_mask=cross_mask,
                    )
                    x_q = residual + x_q

                # 2. Self-Attention Block (Full)
                residual = x_q
                normed_x_q = layer.input_layernorm(x_q)
                attn_out, _ = layer.self_attn(
                    normed_x_q,
                    attention_mask=None,
                    position_ids=q_pos,
                    position_embeddings=q_position_embeddings,
                )
                x_q = residual + attn_out

                # 3. Slice to CLS
                x_q = x_q[:, 0:1, :]

                # 4. MLP Block (CLS only)
                residual = x_q
                x_q = layer.mlp(layer.post_attention_layernorm(x_q))
                x_q = residual + x_q
            else:
                # Inverted logic: Cross-attn followed by Self-attn
                # 1. Cross-Attention Block
                if x_d is not None:
                    residual = x_q
                    normed_x_q = layer.input_layernorm(x_q)
                    x_q = layer.cross_attn(
                        normed_x_q,
                        encoder_hidden_states=layer.input_layernorm(x_d),
                        encoder_attention_mask=cross_mask,
                    )
                    x_q = residual + x_q

                # 2. Self-Attention Block
                residual = x_q
                normed_x_q = layer.input_layernorm(x_q)
                attn_out, _ = layer.self_attn(
                    normed_x_q,
                    attention_mask=None,
                    position_ids=q_pos,
                    position_embeddings=q_position_embeddings,
                )
                x_q = residual + attn_out

                # 3. MLP Block
                residual = x_q
                hidden_states = layer.post_attention_layernorm(x_q)
                hidden_states = layer.mlp(hidden_states)
                x_q = residual + hidden_states
        return x_q

    def forward(
        self,
        inputs: BaseItems,
        tokenized: Optional[MICETokenizedTexts] = None,
    ):
        if tokenized is None:
            tokenized = self.batch_tokenize(inputs)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)
        tokenized_docs = to_device(tokenized.tokenized_docs, self.device)

        query_ids = tokenized_q.ids
        query_mask = tokenized_q.mask
        doc_ids = tokenized_docs.ids
        doc_mask = tokenized_docs.mask

        # Pos IDs for RoPE
        def _get_pos_ids(ids):
            b, s = ids.size()
            return torch.arange(s, device=ids.device).unsqueeze(0).expand(b, s)

        x_q = self.embeddings(query_ids)
        x_d = self.embeddings(doc_ids)

        q_pos = _get_pos_ids(query_ids)
        d_pos = _get_pos_ids(doc_ids)

        # Compute rotary embeddings once
        q_position_embeddings = self.rotary_emb(x_q, q_pos)
        d_position_embeddings = self.rotary_emb(x_d, d_pos)

        # Bottom
        if self.bound_bottom_layers:
            query_bottom_layers = self.bottom_layers
            document_bottom_layers = self.bottom_layers
        else:
            query_bottom_layers = self.query_bottom_layers
            document_bottom_layers = self.document_bottom_layers

        for layer in query_bottom_layers:
            x_q = layer(
                x_q,
                attention_mask=None,
                position_ids=q_pos,
                position_embeddings=q_position_embeddings,
            )
        for layer in document_bottom_layers:
            x_d = layer(
                x_d,
                attention_mask=None,
                position_ids=d_pos,
                position_embeddings=d_position_embeddings,
            )

        if self.global_cls_token:
            cls_token = self.global_cls.expand(x_q.shape[0], -1, -1)
            x_q = torch.cat([cls_token, x_q], dim=1)
            query_mask = torch.cat(
                [
                    torch.ones(
                        (query_mask.shape[0], 1),
                        dtype=query_mask.dtype,
                        device=query_mask.device,
                    ),
                    query_mask,
                ],
                dim=1,
            )
            # Recompute pos embeds for top layers
            q_pos = _get_pos_ids(query_mask)
            q_position_embeddings = self.rotary_emb(x_q, q_pos)

        # Top
        # Mask for Cross-Attention (Query attending to Doc) shape [batch, 1, seq_len_query, seq_len_doc]
        cross_mask = self.get_cross_attention_mask(query_mask, doc_mask, x_q.dtype)

        if self.extra_attn_bias:
            from .mice import compute_mask

            exact_match_mask = compute_mask(query_ids, doc_ids)
        else:
            exact_match_mask = None

        if self.cross_attn_first:
            x_q = self.forward_inverted_transformer(
                x_q, x_d, cross_mask, q_pos, q_position_embeddings, exact_match_mask
            )
        else:
            x_q = self.forward_vanilla_transformer(
                x_q, x_d, cross_mask, q_pos, q_position_embeddings, exact_match_mask
            )

        x_q = self.final_norm(x_q)

        # Pooling
        if self.pooling_method == "cls":
            pooled = x_q[:, 0]
        else:
            pooled = x_q.mean(dim=1)

        return self.classifier(self.dropout_layer(pooled)).squeeze(-1)


class InitMICEQwenFromHFID(LightweightTask):
    """Worker-node task to load weights into MICE Qwen model"""

    model: Param[QwenMiceCrossEncoder]

    def execute(self):
        model = self.model
        hf_id = model.hf_id

        # Build structure
        model.initialize()

        logger.info(f"Loading MICE Qwen weights from {hf_id}")

        full_backbone = AutoModelForCausalLM.from_pretrained(hf_id)

        # Embeddings
        if hasattr(full_backbone.model, "embed_tokens"):
            logger.info("Seeding embeddings from backbone")
            model.embeddings.load_state_dict(
                full_backbone.model.embed_tokens.state_dict()
            )
        else:
            logger.warning(
                f"Backbone {hf_id} has no 'embed_tokens' attribute; skipping seeding"
            )

        # Bottom layers
        if hasattr(full_backbone.model, "layers"):
            logger.info("Seeding bottom layers from backbone")
            if model.bound_bottom_layers:
                contextualization_module_names = ["bottom_layers"]
            else:
                contextualization_module_names = [
                    "document_bottom_layers",
                    "query_bottom_layers",
                ]

            for cntx_module_name in contextualization_module_names:
                contextualization_module = getattr(model, cntx_module_name)
                for i in range(len(contextualization_module)):
                    if i < len(full_backbone.model.layers):
                        contextualization_module[i].load_state_dict(
                            full_backbone.model.layers[i].state_dict()
                        )
                    else:
                        logger.warning(
                            f"Backbone has only {len(full_backbone.model.layers)} layers; cannot seed bottom layer {i}"
                        )
        else:
            logger.warning(
                f"Backbone {hf_id} has no layers; skipping bottom layer seeding"
            )

        # Top layers
        if hasattr(full_backbone.model, "layers"):
            start_idx = model.n_contextualization_layers
            if model.n_interaction_layers is not None:
                end_idx = start_idx + model.n_interaction_layers
                src_layers = full_backbone.model.layers[start_idx:end_idx]
            else:
                src_layers = full_backbone.model.layers[start_idx:]

            logger.info(f"Seeding {len(model.top_layers)} top layers from backbone")
            for i, target_layer in enumerate(model.top_layers):
                if i < len(src_layers):
                    if not model.random_top_layers:
                        self._copy_qwen_weights(src_layers[i], target_layer)
                else:
                    logger.warning(
                        f"Backbone has only {len(src_layers)} remaining layers; cannot seed top layer {i}"
                    )
        else:
            logger.warning(
                f"Backbone {hf_id} has no layers; skipping top layer seeding"
            )

        if hasattr(full_backbone.model, "norm"):
            logger.info("Seeding final_norm from backbone")
            model.final_norm.load_state_dict(full_backbone.model.norm.state_dict())
        else:
            logger.warning(
                f"Backbone {hf_id} has no 'norm' attribute; skipping final_norm seeding"
            )

        if model.global_cls_token:
            # Qwen might not have a CLS token; try BOS or EOS
            cls_token_id = (
                model.tokenizer.tokenizer.cls_token_id
                or model.tokenizer.tokenizer.bos_token_id
                or model.tokenizer.tokenizer.eos_token_id
            )
            if cls_token_id is not None:
                logger.info(
                    f"Seeding global_cls with token ID {cls_token_id} embedding"
                )
                with torch.no_grad():
                    cls_embedding = full_backbone.model.embed_tokens.weight[
                        cls_token_id
                    ]
                    model.global_cls.data.copy_(cls_embedding.view(1, 1, -1))
            else:
                logger.warning(
                    "global_cls is True but no CLS/BOS/EOS token found in tokenizer; skipping seeding"
                )

    def _copy_qwen_weights(self, src, target):
        """Copies weights and seeds cross-attention"""
        with torch.no_grad():
            # Copy standard components
            target.self_attn.load_state_dict(src.self_attn.state_dict())
            target.input_layernorm.load_state_dict(src.input_layernorm.state_dict())
            target.post_attention_layernorm.load_state_dict(
                src.post_attention_layernorm.state_dict()
            )
            target.mlp.load_state_dict(src.mlp.state_dict())

            # Seed cross-attention
            src_attn = src.self_attn
            target_cross = target.cross_attn

            # Qwen3: q_proj split
            if (
                hasattr(target_cross, "gate_proj")
                and target_cross.gate_proj is not None
            ):
                all_head = target_cross.num_heads * target_cross.head_dim
                if src_attn.q_proj.weight.shape[0] == 2 * all_head:
                    target_cross.q_proj.weight.copy_(src_attn.q_proj.weight[:all_head])
                    target_cross.gate_proj.weight.copy_(
                        src_attn.q_proj.weight[all_head:]
                    )
                else:
                    target_cross.q_proj.weight.copy_(src_attn.q_proj.weight)
            else:
                target_cross.q_proj.weight.copy_(src_attn.q_proj.weight)
                if (
                    hasattr(src_attn.q_proj, "bias")
                    and src_attn.q_proj.bias is not None
                ):
                    if target_cross.q_proj.bias is not None:
                        target_cross.q_proj.bias.copy_(src_attn.q_proj.bias)

            target_cross.k_proj.weight.copy_(src_attn.k_proj.weight)
            if hasattr(src_attn.k_proj, "bias") and src_attn.k_proj.bias is not None:
                if target_cross.k_proj.bias is not None:
                    target_cross.k_proj.bias.copy_(src_attn.k_proj.bias)

            target_cross.v_proj.weight.copy_(src_attn.v_proj.weight)
            if hasattr(src_attn.v_proj, "bias") and src_attn.v_proj.bias is not None:
                if target_cross.v_proj.bias is not None:
                    target_cross.v_proj.bias.copy_(src_attn.v_proj.bias)

            target_cross.o_proj.weight.copy_(src_attn.o_proj.weight)
            if hasattr(src_attn.o_proj, "bias") and src_attn.o_proj.bias is not None:
                if target_cross.o_proj.bias is not None:
                    target_cross.o_proj.bias.copy_(src_attn.o_proj.bias)

            # Qwen3 specific: copy q_norm and k_norm
            if hasattr(src_attn, "q_norm") and not isinstance(
                target_cross.q_norm, nn.Identity
            ):
                target_cross.q_norm.load_state_dict(src_attn.q_norm.state_dict())
            if hasattr(src_attn, "k_norm") and not isinstance(
                target_cross.k_norm, nn.Identity
            ):
                target_cross.k_norm.load_state_dict(src_attn.k_norm.state_dict())
