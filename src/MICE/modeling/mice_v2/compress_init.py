"""Data-driven initialization of compressed MICE v2 top interaction layers.

When `compress_dim > 1.0`, the top interaction layers (and the adapter that
projects the bottom contextualization layers' output down to their size) have
a different shape than the pretrained ModernBERT backbone. This module
computes the initial values for those (already correctly-shaped) parameters
from the pretrained backbone weights, instead of leaving them at their
independent default init.

Two data-driven modes are supported:
- "random_projection": the projection matrices P (hidden) / P_int
  (intermediate) are themselves randomly initialized, but the same P/P_int
  are reused to derive every top-layer weight from the pretrained backbone
  weights (so the compressed weights start as a random low-rank projection of
  the pretrained ones, rather than being fully independent).
- "svd_projection": P/P_int are fit via SVD on the pretrained backbone
  weights to preserve as much variance as possible.

"random" mode is a no-op (handled by the caller) since every weight is just
left at its own independent default init.
"""

from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Projection matrix construction
# ---------------------------------------------------------------------------


def compute_random_projection(out_dim: int, in_dim: int) -> torch.Tensor:
    """A `(out_dim, in_dim)` projection matrix with PyTorch's ordinary default
    `nn.Linear` init (no special scheme) -- used for `P_int` in
    "random_projection" mode."""
    return nn.Linear(in_dim, out_dim).weight.detach().clone()


def compute_shared_projection_general(
    row_matrices: List[torch.Tensor], target_dim: int
) -> torch.Tensor:
    """Fits a semi-orthogonal projection `P` of shape `(target_dim, source_dim)`
    from a list of matrices that all share the same dim-0 size (`source_dim`),
    by SVD-ing their horizontal concatenation and keeping the top-`target_dim`
    left singular vectors."""
    stacked = torch.cat(row_matrices, dim=1)
    U, _S, _Vh = torch.linalg.svd(stacked.float(), full_matrices=False)
    P = U[:, :target_dim].T.contiguous()
    return P.to(row_matrices[0].dtype)


def compute_shared_projection(
    weight_matrices: List[torch.Tensor], target_dim: int
) -> torch.Tensor:
    """Same as `compute_shared_projection_general`, specialized for *square*
    matrices: also pools each matrix's transpose, so the fitted subspace
    captures variance from both the input and output axis (valid only when
    every matrix is square, i.e. shares one axis size on both sides)."""
    all_mats = []
    for w in weight_matrices:
        all_mats.append(w)
        all_mats.append(w.T)
    return compute_shared_projection_general(all_mats, target_dim)


# ---------------------------------------------------------------------------
# Per-matrix projection helpers
# ---------------------------------------------------------------------------


def project_square_both_sides(W: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Square `(d, d)` matrix, same `P` (target_dim, d) on both axes."""
    orig_dim, target_dim = W.shape[0], P.shape[0]
    scale = (orig_dim / target_dim) ** 0.5
    return scale * (P @ W @ P.T)


def project_two_sided(
    W: torch.Tensor, P_out: torch.Tensor, P_in: torch.Tensor
) -> torch.Tensor:
    """`(out, in)` matrix, two *different* projections, one per axis."""
    orig_out, orig_in = W.shape
    target_out, target_in = P_out.shape[0], P_in.shape[0]
    scale = (orig_out / target_out) ** 0.5 * (orig_in / target_in) ** 0.5
    return scale * (P_out @ W @ P_in.T)


def project_input_only(W: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """`(out, in)` matrix, only the input axis is compressed."""
    orig_dim, target_dim = W.shape[1], P.shape[0]
    scale = (orig_dim / target_dim) ** 0.5
    return scale * (W @ P.T)


def project_output_only(W: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """`(out, in)` matrix, only the output axis is compressed."""
    orig_dim, target_dim = W.shape[0], P.shape[0]
    scale = (orig_dim / target_dim) ** 0.5
    return scale * (P @ W)


def project_vector(v: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    orig_dim, target_dim = v.shape[0], P.shape[0]
    scale = (orig_dim / target_dim) ** 0.5
    return scale * (P @ v)


def project_fused_qkv_weight(
    Wqkv_weight: torch.Tensor, P: torch.Tensor
) -> torch.Tensor:
    """`(3d, d)` fused Q/K/V matrix -> chunk into 3 square `(d, d)` blocks,
    project each with the same shared `P` on both sides, re-concatenate."""
    chunks = Wqkv_weight.chunk(3, dim=0)
    return torch.cat([project_square_both_sides(c, P) for c in chunks], dim=0)


def project_fused_qkv_bias(
    bias: Optional[torch.Tensor], P: torch.Tensor
) -> Optional[torch.Tensor]:
    if bias is None:
        return None
    chunks = bias.chunk(3, dim=0)
    return torch.cat([project_vector(c, P) for c in chunks], dim=0)


def project_fused_mlp_wi(
    Wi_weight: torch.Tensor, P: torch.Tensor, P_int: torch.Tensor
) -> torch.Tensor:
    """`(2*intermediate, d)` fused GeGLU value/gate matrix -> chunk into 2
    `(intermediate, d)` blocks, project each with `P_int` on the intermediate
    (output) axis and `P` on the hidden (input) axis, re-concatenate."""
    chunks = Wi_weight.chunk(2, dim=0)
    return torch.cat([project_two_sided(c, P_int, P) for c in chunks], dim=0)


def project_mlp_wi_bias(
    bias: Optional[torch.Tensor], P_int: torch.Tensor
) -> Optional[torch.Tensor]:
    if bias is None:
        return None
    chunks = bias.chunk(2, dim=0)
    return torch.cat([project_vector(c, P_int) for c in chunks], dim=0)


def project_mlp_wo(
    Wo_weight: torch.Tensor, P: torch.Tensor, P_int: torch.Tensor
) -> torch.Tensor:
    """`(d, intermediate)` MLP down-projection: `P` on the hidden (output)
    axis, `P_int` on the intermediate (input) axis."""
    return project_two_sided(Wo_weight, P, P_int)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _collect_source_layers(full_backbone, n_contextualization_layers, n_top_layers):
    src_layers = []
    backbone_layers = full_backbone.model.layers
    for idx in range(n_top_layers):
        src_layer_idx = n_contextualization_layers + idx
        if src_layer_idx < len(backbone_layers):
            src_layers.append(backbone_layers[src_layer_idx])
    return src_layers


def _fit_projections(src_layers, hidden_target_dim, intermediate_target_dim):
    hidden_mats = []
    intermediate_mats = []
    for src_layer in src_layers:
        attn = src_layer.attn
        hidden_mats.extend(attn.Wqkv.weight.chunk(3, dim=0))
        hidden_mats.append(attn.Wo.weight)

        mlp = src_layer.mlp
        value_chunk, gate_chunk = mlp.Wi.weight.chunk(2, dim=0)
        intermediate_mats.append(value_chunk)
        intermediate_mats.append(gate_chunk)
        intermediate_mats.append(mlp.Wo.weight.T)

    P = compute_shared_projection(hidden_mats, hidden_target_dim)
    P_int = compute_shared_projection_general(
        intermediate_mats, intermediate_target_dim
    )
    return P, P_int


def _copy_norm(dst: nn.Module, src: nn.LayerNorm, P: torch.Tensor) -> None:
    with torch.no_grad():
        dst.weight.copy_(project_vector(src.weight, P))
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(project_vector(src.bias, P))


def _copy_attention(dst_attn, src_attn, P: torch.Tensor) -> None:
    with torch.no_grad():
        dst_attn.Wqkv.weight.copy_(project_fused_qkv_weight(src_attn.Wqkv.weight, P))
        if dst_attn.Wqkv.bias is not None:
            dst_attn.Wqkv.bias.copy_(project_fused_qkv_bias(src_attn.Wqkv.bias, P))
        dst_attn.Wo.weight.copy_(project_square_both_sides(src_attn.Wo.weight, P))
        if dst_attn.Wo.bias is not None:
            dst_attn.Wo.bias.copy_(project_vector(src_attn.Wo.bias, P))


def _copy_mlp(dst_mlp, src_mlp, P: torch.Tensor, P_int: torch.Tensor) -> None:
    with torch.no_grad():
        dst_mlp.Wi.weight.copy_(project_fused_mlp_wi(src_mlp.Wi.weight, P, P_int))
        if dst_mlp.Wi.bias is not None:
            dst_mlp.Wi.bias.copy_(project_mlp_wi_bias(src_mlp.Wi.bias, P_int))
        dst_mlp.Wo.weight.copy_(project_mlp_wo(src_mlp.Wo.weight, P, P_int))
        if dst_mlp.Wo.bias is not None:
            dst_mlp.Wo.bias.copy_(project_vector(src_mlp.Wo.bias, P))


def initialize_compressed_top_layers(
    hf_model,
    full_backbone,
    n_contextualization_layers: int,
    compress_init_option: str,
) -> None:
    """Initializes `hf_model.adapter` (if present), `hf_model.top_final_norm`
    (if present) and every `hf_model.top_layers[i]` from the pretrained
    `full_backbone`, by deriving each top-layer weight as a projection of the
    corresponding pretrained weight through a shared `P`/`P_int`.

    `hf_model.top_config.hidden_size`/`intermediate_size` may equal the native
    (uncompressed) size -- that happens when this is called at `compress_dim
    == 1.0` with an explicit `"random_projection"`/`"svd_projection"` request
    (rather than "seed", which just loads directly instead of calling this
    function at all). In that case `P`/`P_int` are full-size (semi-)orthogonal
    rotations rather than dimension-reducing projections, and since there is
    no `hf_model.adapter` to persist `P` into (it's only allocated once
    `compress_dim > 1.0`), `P` is computed standalone and simply not stored
    anywhere -- it only exists for the duration of this init pass, exactly
    like the source `full_backbone` weights it's derived from.

    Caller contract: `compress_init_option` must already be fully resolved by
    the caller (`mice_v2_scorer()` / `InitMICEV2ModernBERTFromHFID.execute()`)
    to one of "random", "random_projection", or "svd_projection" -- "seed" is a
    resolution state that only makes sense with knowledge of `compress_dim`,
    which this function does not have, so it never accepts that string.

    `compress_init_option`:
      - "random": no-op, every weight keeps its own independent default init.
      - "random_projection": `P`/`P_int` are randomly initialized, then reused
        to derive every top-layer weight from the pretrained backbone.
      - "svd_projection": `P`/`P_int` are fit via SVD on the pretrained
        backbone weights, then reused the same way.
    """
    if compress_init_option == "random":
        return

    if compress_init_option not in ("random_projection", "svd_projection"):
        raise ValueError(
            f"Unknown compress_init_option: {compress_init_option!r} (expected "
            "'random', 'random_projection', or 'svd_projection' -- 'seed' must "
            "already be resolved by the caller before calling this function)"
        )

    top_layers = hf_model.top_layers
    src_layers = _collect_source_layers(
        full_backbone, n_contextualization_layers, len(top_layers)
    )
    if not src_layers:
        return

    hidden_target_dim = hf_model.top_config.hidden_size
    intermediate_target_dim = hf_model.top_config.intermediate_size

    if compress_init_option == "random_projection":
        if hf_model.adapter is not None:
            P = hf_model.adapter.weight.detach().clone()
        else:
            P = compute_random_projection(
                hidden_target_dim, full_backbone.model.config.hidden_size
            )
        P_int = compute_random_projection(
            intermediate_target_dim, full_backbone.model.config.intermediate_size
        )
    else:
        P, P_int = _fit_projections(
            src_layers, hidden_target_dim, intermediate_target_dim
        )
        if hf_model.adapter is not None:
            with torch.no_grad():
                hf_model.adapter.weight.copy_(P)

    for top_layer, src_layer in zip(top_layers, src_layers):
        norm_to_copy = (
            src_layer.attn_norm
            if hasattr(src_layer, "attn_norm")
            and isinstance(src_layer.attn_norm, nn.LayerNorm)
            else (
                src_layer.mlp_norm
                if hasattr(src_layer, "mlp_norm")
                and isinstance(src_layer.mlp_norm, nn.LayerNorm)
                else None
            )
        )
        if norm_to_copy is not None:
            _copy_norm(top_layer.cross_attn_norm, norm_to_copy, P)
            _copy_norm(top_layer.self_attn_norm, norm_to_copy, P)
            if isinstance(top_layer.doc_cross_attn_norm, nn.LayerNorm):
                _copy_norm(top_layer.doc_cross_attn_norm, norm_to_copy, P)

        if hasattr(src_layer, "mlp_norm") and isinstance(
            src_layer.mlp_norm, nn.LayerNorm
        ):
            _copy_norm(top_layer.mlp_norm, src_layer.mlp_norm, P)

        _copy_attention(top_layer.self_attn, src_layer.attn, P)
        _copy_attention(top_layer.cross_attn, src_layer.attn, P)
        _copy_mlp(top_layer.mlp, src_layer.mlp, P, P_int)

    if getattr(hf_model, "top_final_norm", None) is not None and hasattr(
        full_backbone.model, "final_norm"
    ):
        _copy_norm(hf_model.top_final_norm, full_backbone.model.final_norm, P)
