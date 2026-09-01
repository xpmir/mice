"""
This script gathers utils for computing theoretical FLOPs for MICE and FrankenCrossScorer.
It extracts model parameters, attention mask patterns, and calculates the FLOPs for each layer considering the masking density (alpha) and interaction patterns.
The results are saved to a CSV file for analysis and comparison against baseline full cross-encoders.
"""

import torch
import yaml
import argparse
import sys
import pandas as pd
from pathlib import Path
from transformers import AutoConfig
from MICE.modeling.mice import MiceCrossEncoder
from models.franken import FrankenCrossScorer, AttentionPatch
from models.mask_scorer import HFMaskedMiniLMCrossScorer, HFMaskedEttinCrossScorer
import logging

# Setup logging
from logging_utils import setup_logging

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# SRC_DIR is the directory containing this script (src/)
SRC_DIR = Path(__file__).parent

# Add src to sys.path to import models
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def get_transformer_params(hf_id):
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    d = config.hidden_size
    d_ff = config.intermediate_size
    L = config.num_hidden_layers
    return d, d_ff, L


def get_alphas(model, query_len, doc_len):
    """Extracts the attention mask patterns from a FrankenCrossScorer to compute alpha values for each layer."""
    # Construct dummy input: [CLS] Q...Q [SEP] D...D [SEP]
    total_len = 1 + query_len + 1 + doc_len + 1
    input_ids = torch.zeros((1, total_len), dtype=torch.long)

    sep_id = model.tokenizer.sep_token_id
    cls_id = model.tokenizer.cls_token_id

    input_ids[0, 0] = cls_id
    input_ids[0, 1 : 1 + query_len] = 1  # Dummy Query tokens
    input_ids[0, 1 + query_len] = sep_id
    input_ids[0, 1 + query_len + 1 : 1 + query_len + 1 + doc_len] = (
        2  # Dummy Document tokens
    )
    input_ids[0, total_len - 1] = sep_id

    # Compute lookup matrix (codes for token type pairs)
    # We use CPU for these calculations
    lookup = model.build_lookup_matrix(input_ids, device="cpu")
    index_codes = lookup.view(-1)
    DEBUG = True
    DEBUG = False

    alphas = []
    for layer_idx in range(model.scorer.n_layers):
        mask_lookup = model.layer_mask_lookup[layer_idx]
        # Map codes to boolean allowed/blocked
        allowed = mask_lookup.gather(0, index_codes).view(lookup.shape[1:])
        if DEBUG:
            import matplotlib.pyplot as plt

            plt.imshow(allowed.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            plt.title(
                f"Layer {layer_idx} - Allowed Attention (white=allowed, black=blocked)"
            )
            plt.xlabel("Key Tokens")
            plt.ylabel("Query Tokens")
            plt.colorbar()
            plt.show()
        alpha = allowed.float().mean().item()
        alphas.append(alpha)
    return alphas


def attn_linear_flops(d, s):
    """Computes FLOPs for the linear layers in a standard transformer attention layer."""
    # Each token goes through 4 linear layers (Q, K, V, O) and the MLP (2 layers)
    return 2 * s * (4 * d**2)


def MLP_flops(d, d_ff, s):
    """Computes FLOPs for the linear layers in a standard transformer attention layer."""
    # Each token goes through 4 linear layers (Q, K, V, O) and the MLP (2 layers)
    return 2 * s * (2 * d * d_ff)


def compute_flops_self_attention(d, d_ff, s, alpha: float = 1.0):
    """Computes FLOPs for a single self-attention layer with masking."""
    linear_flops = attn_linear_flops(d, s)

    # Attention cost: 4 * alpha * S^2 * d for QK^T and AV operations
    attn_flops = 4 * alpha * (s**2) * d

    return linear_flops + attn_flops


def compute_flops_transformer_layer(d, d_ff, seq_len, alpha: float = 1.0):
    """
    Computes FLOPs for a single transformer layer.

    Args:
        d: Model dimension
        d_ff: MLP intermediate dimension
        s: Sequence length
        alpha: Masking density
        is_interaction: Whether this is a MICE interaction layer (frozen document)
    """
    linear_flops = compute_flops_self_attention(d, d_ff, seq_len, alpha)
    mlp_flops = MLP_flops(d, d_ff, seq_len)

    # print(
    #     f"Linear FLOPs: {linear_flops / 1e9:.2f} GFLOPs (alpha={alpha:.3f}, MLP FLOPs: {mlp_flops / 1e9:.2f} GFLOPs"
    # )
    return linear_flops + mlp_flops


def compute_flops_cross_attn_layer(d, query_len, doc_len):
    """
    Computes FLOPs for a single cross-attention layer.

    Args:
        d: Model dimension
        query_len: Query length
        doc_len: Document length
    """
    # linear only on document tokens: 2 * doc_len * (4 * d^2) for QKV and O projections
    linear_flops = attn_linear_flops(d, doc_len)

    # Attention cost: 4 * seq_len^2 * d for QK^T and AV operations
    self_attn_flops = 4 * (query_len**2) * d
    cross_attn_flops = 4 * query_len * doc_len * d

    attn_flops = self_attn_flops + cross_attn_flops

    logging.info(
        f"Cross-Attention Layer FLOPs: {linear_flops / 1e9:.2f} GFLOPs, Attention FLOPs: {attn_flops / 1e9:.2f} GFLOPs"
    )
    return linear_flops + attn_flops


# deprecated
def compute_franken_config_flops(config_path, query_len, max_seq_len):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    xp_id = cfg.get("id", config_path.stem)
    hf_id = cfg.get("base")
    if hf_id == "" or hf_id is None:
        raise ValueError(f"HF ID not found in config {config_path}")

    d, d_ff, L = get_transformer_params(hf_id)

    # Determine merge_layer and dropped layers
    merge_layer = cfg.get("merge_layer", cfg.get("split_layer", L))
    drop_layer_cfg = cfg.get("drop_layer", {})

    if isinstance(drop_layer_cfg, dict) and "values_range" in drop_layer_cfg:
        dropped_layers = list(
            range(
                drop_layer_cfg["values_range"][0], drop_layer_cfg["values_range"][1] + 1
            )
        )
    elif isinstance(drop_layer_cfg, int):
        dropped_layers = list(range(drop_layer_cfg + 1, L))
    else:
        dropped_layers = []

    # Build model to extract masks
    attn_patches_cfg = cfg.get("attn_patches", [])
    patches = []
    for p in attn_patches_cfg:
        start = p.get("start_layer", 0)
        if isinstance(start, dict):
            start = start.get("value", 0)
        end = p.get("end_layer", -1)
        if isinstance(end, dict):
            if "value" in end:
                end = end["value"]
            elif "values_range" in end:
                end = end["values_range"][0]

        patches.append(
            AttentionPatch.C(
                mask_attention_from=p.get("mask_attention_from"),
                mask_attention_to=p.get("mask_attention_to"),
                start_layer=start,
                end_layer=end,
            )
        )

    # Instantiate appropriate scorer class
    if "minilm" in hf_id.lower():
        scorer_cls = HFMaskedMiniLMCrossScorer
    elif "ettin" in hf_id.lower():
        scorer_cls = HFMaskedEttinCrossScorer
    else:
        scorer_cls = HFMaskedMiniLMCrossScorer

    for p in patches:
        logging.info(f"  - {p}")

    model_cfg = FrankenCrossScorer.C(
        scorer=scorer_cls.C(hf_id=hf_id), attention_patches=patches
    )
    model = model_cfg.instance()
    model.__initialize__()
    logging.info(
        f"Config: {config_path.name}, HF ID: {hf_id}, layers : {L}, Merge Layer: {merge_layer}, Dropped Layers: {drop_layer_cfg} ({dropped_layers}), Patches: {len(patches)}"
    )

    doc_len = max_seq_len - query_len - 3
    alphas = get_alphas(model, query_len, doc_len)

    total_flops = 0
    active_alphas = []
    s = max_seq_len
    n = query_len

    for layer in range(L):
        if layer in dropped_layers:
            continue

        alpha = alphas[layer]
        active_alphas.append(alpha)
        is_interaction = layer >= merge_layer
        if is_interaction:
            total_flops += compute_flops_cross_attn_layer(d, d_ff, s, n, alpha)
        else:
            total_flops += compute_flops_transformer_layer(d, d_ff, s, n, alpha)

    avg_alpha = sum(active_alphas) / len(active_alphas) if active_alphas else 0

    # Baseline: Full cross-encoder (L layers, alpha=1.0, full linear for all tokens)
    baseline_flops = L * compute_flops_transformer_layer(d, d_ff, s, n, 1.0)

    return total_flops, baseline_flops, hf_id, xp_id, avg_alpha


def compute_mice_config_flops(config_path, query_len, max_seq_len):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    grid_search = cfg.get("grid_search", {})
    if (
        "n_interaction_layers" in grid_search
        or "n_contextualization_layers" in grid_search
    ):
        logging.warning(
            f"Config {config_path} contains grid search over n_interaction_layers/n_contextualization_layers. Using default value for FLOPs calculation."
        )

    xp_id = cfg.get("id", config_path.stem)
    hf_id = cfg.get("base")
    if hf_id == "" or hf_id is None:
        raise ValueError(f"HF ID not found in config {config_path}")

    d, d_ff, nlayers = get_transformer_params(hf_id)

    # Determine n_contextualization_layers and n_interaction_layers
    n_contextualization_layers = cfg.get("n_contextualization_layers")
    n_interaction_layers = cfg.get("n_interaction_layers")

    if n_interaction_layers is None or n_contextualization_layers is None:
        raise ValueError(
            f"n_interaction_layers or n_contextualization_layers not found in config {config_path}"
        )

    n_docs_ctx_layers = cfg.get("n_docs_ctx_layers", n_contextualization_layers)
    extra_attn_bias = cfg.get("extra_attn_bias", False)

    mice_cfg = MiceCrossEncoder.C(
        n_contextualization_layers=n_contextualization_layers,
        n_interaction_layers=n_interaction_layers,
        n_docs_ctx_layers=n_docs_ctx_layers,
        extra_attn_bias=extra_attn_bias,
    )
    total_flops, baseline_flops = compute_flops_mice(
        mice_cfg,
        d,
        d_ff,
        nlayers,
        seq_len=max_seq_len,
        query_len=query_len,
    )

    return (
        total_flops,
        baseline_flops,
        hf_id,
        xp_id,
        1.0,
    )  # avg_alpha is 1.0 for MICE since we don't apply masking in the same way


def compute_prettr_config_flops(config_path, query_len, max_seq_len):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    xp_id = cfg.get("id", config_path.stem)
    hf_id = cfg.get("base")
    if hf_id == "" or hf_id is None:
        raise ValueError(f"HF ID not found in config {config_path}")

    d, d_ff, nlayers = get_transformer_params(hf_id)

    join_layer = cfg.get("join_layer")
    if join_layer is None:
        raise ValueError(f"join_layer not found in config {config_path}")

    total_flops, baseline_flops = compute_flops_prettr(
        join_layer,
        d,
        d_ff,
        nlayers,
        seq_len=max_seq_len,
        query_len=query_len,
    )

    return total_flops, baseline_flops, hf_id, xp_id, 1.0


def compute_flops_CrossEncoder(
    d,
    d_ff,
    nlayers,
    seq_len=512,
):
    baseline_flops = nlayers * compute_flops_transformer_layer(d, d_ff, seq_len, 1.0)

    return baseline_flops, baseline_flops


def compute_flops_prettr(
    join_layer,
    d,
    d_ff,
    nlayers,
    seq_len=512,
    query_len=32,
    precompute_docs=False,
):
    doc_len = seq_len - query_len - 3  # [CLS] q [SEP] doc [SEP]

    total_flops = 0
    # Layers < join_layer: independent encoding (masked joint)
    for _ in range(join_layer):
        # compute as if computations are done separately for query and document.
        total_flops += compute_flops_transformer_layer(d, d_ff, query_len, alpha=1.0)
        if not precompute_docs:
            total_flops += compute_flops_transformer_layer(d, d_ff, doc_len, alpha=1.0)

    # Layers >= join_layer: full joint self-attention
    for _ in range(join_layer, nlayers):
        total_flops += compute_flops_transformer_layer(d, d_ff, seq_len, alpha=1.0)

    # Baseline: Full cross-encoder
    baseline_flops = nlayers * compute_flops_transformer_layer(d, d_ff, seq_len, 1.0)

    return total_flops, baseline_flops


def compute_flops_colbert(
    d,
    d_ff,
    nlayers,
    seq_len=512,
    query_len=32,
    precompute_docs=False,
):
    doc_len = seq_len - query_len - 3  # [CLS] q [SEP] doc [SEP]

    total_flops = 0
    # 1. Query Encoding (independent)
    for _ in range(nlayers):
        total_flops += compute_flops_transformer_layer(d, d_ff, query_len)

    # 2. Document Encoding (independent)
    if not precompute_docs:
        for _ in range(nlayers):
            total_flops += compute_flops_transformer_layer(d, d_ff, doc_len)

    # 3. MaxSim operation
    # dot product: 2 * d * query_len * doc_len
    total_flops += 2 * d * query_len * doc_len

    # Baseline: Full cross-encoder
    baseline_flops = nlayers * compute_flops_transformer_layer(d, d_ff, seq_len, 1.0)

    return total_flops, baseline_flops


def compute_flops_mice(
    mice_cfg: MiceCrossEncoder,
    d,
    d_ff,
    nlayers,
    seq_len=512,
    query_len=32,
    precompute_docs=False,
):
    n_contextualization_layers = mice_cfg.n_contextualization_layers
    n_interaction_layers = mice_cfg.n_interaction_layers
    n_docs_ctx_layers = getattr(mice_cfg, "n_docs_ctx_layers", None)
    if n_docs_ctx_layers is None:
        n_docs_ctx_layers = n_contextualization_layers

    extra_attn_bias = getattr(mice_cfg, "extra_attn_bias", False)
    doc_len = seq_len - query_len - 3

    total_flops = 0

    # 1. Query Contextualization
    for _ in range(n_contextualization_layers):
        total_flops += compute_flops_transformer_layer(d, d_ff, query_len)

    # 2. Document Contextualization
    if not precompute_docs:
        for _ in range(n_docs_ctx_layers):
            total_flops += compute_flops_transformer_layer(d, d_ff, doc_len)

    # 3. Interaction Layers
    for _ in range(n_interaction_layers):
        # Self-Attention on Query tokens
        total_flops += compute_flops_self_attention(d, d_ff, query_len)
        total_flops += compute_flops_cross_attn_layer(d, query_len, doc_len)

        # MLP on Query tokens
        total_flops += MLP_flops(d, d_ff, query_len)

        # Extra Attention Bias (if enabled)
        if extra_attn_bias:
            # v_proj(m) + out_proj(n) + SDPA(n*m*d)
            bias_linear = 2 * (query_len + doc_len) * (d**2)
            bias_attn = 2 * query_len * doc_len * d
            total_flops += bias_linear + bias_attn

    # Baseline: Full cross-encoder (L layers, full linear for all tokens)
    # Using seq_len as the baseline sequence length
    baseline_flops = nlayers * compute_flops_transformer_layer(d, d_ff, seq_len, 1.0)

    return total_flops, baseline_flops


def calculate_config_flops(config_path, query_len, max_seq_len):
    """Determines the type of config (MICE or Franken) and calculates FLOPs accordingly."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    module = cfg.get("module", "")
    if "mice_training" in module:
        return compute_mice_config_flops(config_path, query_len, max_seq_len)
    elif "prettr_training" in module:
        return compute_prettr_config_flops(config_path, query_len, max_seq_len)
    elif "franken_cross_scorer" in module:
        return compute_franken_config_flops(config_path, query_len, max_seq_len)
    else:
        # Check for MICE specific parameters if module is ambiguous
        if "n_interaction_layers" in cfg or "n_contextualization_layers" in cfg:
            return compute_mice_config_flops(config_path, query_len, max_seq_len)
        raise ValueError(f"Unknown module type in config {config_path}: {module}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute theoretical FLOPs for MICE and masked models."
    )
    parser.add_argument(
        "--query_len", type=int, default=32, help="Typical query length"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=str(SRC_DIR / "MICE/experiments"),
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="*.yaml",
        help="Filter for YAML configs",
    )
    parser.add_argument(
        "--output", type=str, default="flops_results.csv", help="Output CSV file name"
    )
    args = parser.parse_args()
    config_paths = list(Path(args.config_dir).glob(args.filter))
    print(
        f"Found {len(config_paths)} config files in {args.config_dir} with filter {args.filter}."
    )
    print(f"\nComparing FLOPs for S={args.max_seq_len}, n={args.query_len}\n")

    results = []
    res = (
        ""
        f"{'XP ID':<35} | {'Base Model':<30} | {'Alpha':<8} | {'GFLOPs':<10} | {'% Base':<10} | {'Speedup':<8}"
    )
    res += "\n" + "-" * 125
    for cfg_path in sorted(config_paths):
        try:
            total, baseline, hf_id, xp_id, avg_alpha = calculate_config_flops(
                cfg_path, args.query_len, args.max_seq_len
            )
        except ValueError as e:
            logging.warning(f"Error processing {cfg_path}: {e}")
            continue

        speedup = baseline / total
        flops_pct = (total / baseline) * 100
        model_name = hf_id.split("/")[-1]
        res += f"\n{xp_id:<35} | {model_name:<30} | {avg_alpha:<8.1f} | {total / 1e9:<10.2f} | {flops_pct:<9.1f}% | {speedup:<8.2f}x"

        results.append(
            {
                "xp_id": xp_id,
                "config_file": cfg_path.name,
                "base_model": model_name,
                "avg_alpha": avg_alpha,
                "gflops": total / 1e9,
                "baseline_gflops": baseline / 1e9,
                "percentage_of_baseline": flops_pct,
                "speedup": speedup,
            }
        )
    print(res)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        # print(df[["xp_id", "base_model", "gflops", "percentage_of_baseline", "speedup"]])

    else:
        print("No configurations found. Please check the config directory path.")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
