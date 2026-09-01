import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

# Configure logging
from logging_utils import setup_logging
setup_logging(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ExactMatchAttentionHead(nn.Module):
    def __init__(self, d_model, use_projections=True):
        super().__init__()
        self.use_projections = use_projections

        if self.use_projections:
            # Traditional heads transform the value and output space
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        """
        Args:
            x: Value states (usually from document) of shape (batch_size, seq_len_v, d_model)
            attn_mask: Pre-computed attention mask of shape (batch_size, 1, seq_len_q, seq_len_v)
        """
        B, Tv, C = x.shape
        _, _, Tq, _ = attn_mask.shape
        dtype = x.dtype
        device = x.device

        # 4. Compute Values
        V = self.v_proj(x) if self.use_projections else x
        # Reshape V for multi-head format expected by SDPA: (B, 1, Tv, C)
        V = V.unsqueeze(1)

        # 5. SDPA
        # Since we pre-computed the logit weights (0 and -inf), we bypass the implicit
        # QK^T scaling by passing a dummy Q and K of zeros, utilizing attn_mask.
        dummy_Q = torch.zeros(B, 1, Tq, C, dtype=dtype, device=device)
        dummy_K = torch.zeros(B, 1, Tv, C, dtype=dtype, device=device)

        # sdpa applies softmax over our custom logits (which are in attn_mask)
        out = F.scaled_dot_product_attention(
            dummy_Q, dummy_K, V,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # Remove head dimension -> (B, Tq, C)
        out = out.squeeze(1)

        if self.use_projections:
            out = self.out_proj(out)

        return out

@torch.compile
def compute_mask(q_ids, doc_ids, is_causal: bool = False):
    """
    Compute the exact match *cross-attention* mask.
    Returns a float mask (0.0 for match, -inf for no match).

    Args:
        q_ids: Query token IDs (B, Tq)
        doc_ids: Document token IDs (B, Td)
    """
    # 1. Compute the Exact Match Mask
    # Shape: (B, 1, Tq, Td) to match SDPA broadcast expectations
    tokens_q = q_ids.unsqueeze(2) # (B, Tq, 1)
    tokens_k = doc_ids.unsqueeze(1) # (B, 1, Td)
    match_mask = (tokens_q == tokens_k).unsqueeze(1) # (B, 1, Tq, Td)
    # 3. Create the Attention Logits
    # Initialize with -inf
    attn_mask = torch.full(match_mask.shape, float("-inf"), device=q_ids.device, dtype=torch.float32)
    attn_mask[match_mask] = 0.0

    # --- FIX FOR NO-MATCH EDGE CASE ---
    # If a token has no matches in the document, attn_logits will be all -inf, causing NaNs.
    # Fallback: attend to the first token (index 0) of the document.
    no_matches = ~match_mask.any(dim=-1) # (B, 1, Tq)

    if no_matches.any():
        # Force it to attend to index 0 for rows with no matches
        attn_mask[:, 0, :, 0] = torch.where(no_matches[:, 0, :], 0.0, attn_mask[:, 0, :, 0])
        # logger.info(f"Fallback applied to {no_matches.sum().item()} query tokens with no matches.")

    return attn_mask

def plot_debug_mask(mask, q_tokens, doc_tokens, path="attn_mask.png"):
    """Plot the mask for debugging"""
    m = mask[0, 0].cpu().numpy()

    # For visualization, replace -inf with a finite value
    m_vis = m.copy()
    m_vis[m_vis == float("-inf")] = -10

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(m_vis, cmap="viridis")

    # Labels
    ax.set_xticks(range(len(doc_tokens[0])))
    ax.set_yticks(range(len(q_tokens[0])))
    ax.set_xticklabels(doc_tokens[0].tolist())
    ax.set_yticklabels(q_tokens[0].tolist())

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Exact Match Mask Debug (0=Match, -10=Masked)")
    fig.tight_layout()
    plt.colorbar(im, label="Logits (0=Match, -10=Masked)")
    plt.savefig(path)
    logger.info(f"Debug plot saved to {path}")
    plt.close()

# --- Test Script ---
if __name__ == "__main__":
    torch.manual_seed(42)

    B, Tq, C = 1, 6, 8
    # Vocabulary tokens
    q_tokens = torch.tensor([[10, 25, 32, 45, 10, 25]])
    doc_tokens = torch.tensor([[10, 1, 4, 87, 212, 25, 56, 2]])

    Td = doc_tokens.shape[1]

    # Mock document features (Distinct vectors for each position)
    doc_features = torch.randn(B, Td, C)

    logger.info(f"Query tokens:    {q_tokens[0].tolist()}")
    logger.info(f"Document tokens: {doc_tokens[0].tolist()}")

    # 1. Compute mask
    attn_mask = compute_mask(q_tokens, doc_tokens)

    # 2. Plot mask
    plot_debug_mask(attn_mask, q_tokens, doc_tokens)

    # 3. Instantiate head without projections to easily see where data is copied from
    exact_match_head = ExactMatchAttentionHead(d_model=C, use_projections=False)

    with torch.no_grad():
        head_outputs = exact_match_head(doc_features, attn_mask)

    logger.info(f"Head Output Shape: {head_outputs.shape}")

    print("\n--- Verification ---")
    # q[0] = 10 matches doc[0] = 10
    match0 = torch.allclose(head_outputs[0,0], doc_features[0,0])
    print(f"q[0] matches doc[0]: {match0}")

    # q[1] = 25 matches doc[5] = 25
    match1 = torch.allclose(head_outputs[0,1], doc_features[0,5])
    print(f"q[1] matches doc[5]: {match1}")

    # q[2] = 32 has NO match. Fallback should be doc[0]
    match2 = torch.allclose(head_outputs[0,2], doc_features[0,0])
    print(f"q[2] (no match) fallback to doc[0]: {match2}")

    # q[5] = 25 matches doc[5] = 25
    match5 = torch.allclose(head_outputs[0,5], doc_features[0,5])
    print(f"q[5] matches doc[5]: {match5}")

    if all([match0, match1, match2, match5]):
        logger.info("All verifications passed!")
    else:
        logger.error("Some verifications FAILED!")
