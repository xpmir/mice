"""
Tests for MICE V2 compress_dim (compressed top interaction layers).
Run with: `uv run pytest tests/test_mice_v2_compress_dim.py -v`
"""

import pytest
import torch
import torch.nn as nn

from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from MICE.modeling.mice_v2.compress_init import (
    compute_random_projection,
    compute_shared_projection,
    compute_shared_projection_general,
    project_square_both_sides,
    project_two_sided,
    project_vector,
    project_fused_qkv_weight,
    project_fused_mlp_wi,
    project_mlp_wo,
    initialize_compressed_top_layers,
)


def _small_config(**kwargs):
    defaults = dict(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
    )
    defaults.update(kwargs)
    return ModernBertMiceConfig(**defaults)


def test_compress_dim_default_is_noop():
    config = _small_config()
    model = ModernBertMiceForSequenceClassification(config)
    assert model.adapter is None
    assert model.top_rotary_emb is None
    assert model.top_config is model.config


def test_compress_dim_shapes():
    config = _small_config(compress_dim=2.0, compress_init_option="random")
    model = ModernBertMiceForSequenceClassification(config)

    assert model.adapter is not None
    assert model.adapter.weight.shape == (16, 32)
    assert model.top_rotary_emb is not None

    top_layer = model.top_layers[0]
    assert top_layer.cross_attn.Wqkv.weight.shape == (3 * 16, 16)
    assert top_layer.cross_attn.Wo.weight.shape == (16, 16)
    assert top_layer.self_attn.Wqkv.weight.shape == (3 * 16, 16)
    assert top_layer.self_attn.Wo.weight.shape == (16, 16)
    assert top_layer.mlp.Wi.weight.shape == (2 * 32, 16)
    assert top_layer.mlp.Wo.weight.shape == (16, 32)
    assert model.classifier.weight.shape[1] == 16


def test_compress_dim_divisibility_assert():
    # hidden_size//compress_dim=21, num_attention_heads//compress_dim=2 -> 21 % 2 != 0
    config = _small_config(hidden_size=64, num_attention_heads=8, compress_dim=3.0)
    with pytest.raises(AssertionError):
        ModernBertMiceForSequenceClassification(config)


def test_compress_dim_negative_assert():
    with pytest.raises(AssertionError):
        _small_config(compress_dim=0.5)


@pytest.mark.parametrize("compress_dim", [1.0, 2.0])
@pytest.mark.parametrize("compress_init_option", ["random", "random_projection"])
def test_compress_dim_forward_finite(compress_dim, compress_init_option):
    config = _small_config(
        compress_dim=compress_dim, compress_init_option=compress_init_option
    )
    model = ModernBertMiceForSequenceClassification(config)
    model.eval()

    query_ids = torch.randint(0, 100, (2, 6))
    query_mask = torch.ones((2, 6), dtype=torch.long)
    doc_ids = torch.randint(0, 100, (2, 10))
    doc_mask = torch.ones((2, 10), dtype=torch.long)

    with torch.no_grad():
        out = model(
            query_ids=query_ids,
            query_mask=query_mask,
            doc_ids=doc_ids,
            doc_mask=doc_mask,
        )

    assert out.logits.shape == (2, 1)
    assert torch.isfinite(out.logits).all()


def test_scale_factor_and_projection_helpers():
    torch.manual_seed(0)

    # compute_random_projection: correct shape, no orthogonality requirement.
    P_rand = compute_random_projection(4, 8)
    assert P_rand.shape == (4, 8)

    # compute_shared_projection: SVD-derived, semi-orthogonal (P @ P.T ~= I).
    mats = [torch.randn(8, 8) for _ in range(3)]
    P = compute_shared_projection(mats, target_dim=4)
    assert P.shape == (4, 8)
    identity_approx = P @ P.T
    assert torch.allclose(identity_approx, torch.eye(4), atol=1e-4)

    # compute_shared_projection_general: rectangular matrices sharing dim0.
    rect_mats = [torch.randn(8, 5) for _ in range(3)]
    P_gen = compute_shared_projection_general(rect_mats, target_dim=4)
    assert P_gen.shape == (4, 8)

    # project_square_both_sides shape + scale sanity.
    W = torch.randn(8, 8)
    W_proj = project_square_both_sides(W, P)
    assert W_proj.shape == (4, 4)

    # project_two_sided with two different projections.
    P_out = compute_random_projection(3, 8)
    P_in = compute_random_projection(4, 8)
    W2 = torch.randn(8, 8)
    W2_proj = project_two_sided(W2, P_out, P_in)
    assert W2_proj.shape == (3, 4)

    # project_vector.
    v = torch.randn(8)
    v_proj = project_vector(v, P)
    assert v_proj.shape == (4,)

    # Fused QKV projection.
    Wqkv = torch.randn(24, 8)
    Wqkv_proj = project_fused_qkv_weight(Wqkv, P)
    assert Wqkv_proj.shape == (12, 4)

    # Fused MLP Wi projection (two different projections per axis).
    P_int = compute_random_projection(6, 16)
    Wi = torch.randn(32, 8)
    Wi_proj = project_fused_mlp_wi(Wi, P, P_int)
    assert Wi_proj.shape == (12, 4)

    # MLP Wo projection.
    Wo = torch.randn(8, 16)
    Wo_proj = project_mlp_wo(Wo, P, P_int)
    assert Wo_proj.shape == (4, 6)


def test_initialize_compressed_top_layers_shape_parity():
    config = _small_config(compress_dim=2.0, compress_init_option="svd_projection")
    model = ModernBertMiceForSequenceClassification(config)

    full_config = _small_config()
    full_backbone = ModernBertMiceForSequenceClassification(full_config)
    # Reuse the same bottom+top layer stack as a stand-in "pretrained backbone":
    # `full_backbone.model.layers` has num_hidden_layers=4 entries at native (32) width.
    full_backbone_model = full_backbone

    before_shapes = {
        name: p.shape for name, p in model.top_layers.named_parameters()
    }

    initialize_compressed_top_layers(
        model,
        full_backbone_model,
        n_contextualization_layers=2,
        compress_init_option="svd_projection",
    )

    after_shapes = {
        name: p.shape for name, p in model.top_layers.named_parameters()
    }
    assert before_shapes == after_shapes
    for p in model.top_layers.parameters():
        assert torch.isfinite(p).all()
