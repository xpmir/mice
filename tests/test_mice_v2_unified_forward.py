"""
Tests for Unified Bottom Contextualization Forward Pass using Block Attention Masks.
Run with: `uv run pytest tests/test_mice_v2_unified_forward.py -v`
"""

import pytest
import torch
from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
    _pack_pair_inputs_padded,
    _create_mice_block_masks,
    _create_block_position_ids,
)


def test_create_mice_block_masks():
    """Verify block mask structure, pad row isolation, and sliding window intersection."""
    B = 2
    seq_len = 8
    q_lens = torch.tensor([3, 2])
    d_lens = torch.tensor([4, 3])

    masks = _create_mice_block_masks(
        q_lens=q_lens,
        d_lens=d_lens,
        seq_len=seq_len,
        local_attention=4,
        device=torch.device("cpu"),
    )

    full_mask = masks["full_attention"]
    sw_mask = masks["sliding_attention"]

    assert full_mask.shape == (2, 1, 8, 8)
    assert sw_mask.shape == (2, 1, 8, 8)

    # Sample 0: Q is 0..2, D is 3..6, PAD is 7
    m0 = full_mask[0, 0]
    # Q cannot attend to D
    assert not m0[0:3, 3:7].any()
    # D cannot attend to Q
    assert not m0[3:7, 0:3].any()
    # Q attends to Q
    assert m0[0:3, 0:3].all()
    # D attends to D
    assert m0[3:7, 3:7].all()
    # Pad row 7 has diagonal True to prevent NaN in softmax
    assert m0[7, 7]
    assert not m0[7, :7].any()

    # Sliding window check: window is 4 -> half_window is 2
    sw0 = sw_mask[0, 0]
    # Token 0 can attend to 0, 1, 2 (|j - 0| <= 2)
    assert sw0[0, 0] and sw0[0, 1] and sw0[0, 2]
    # Token 3 (first doc token) can only attend to 3, 4, 5 (|j - 3| <= 2 within doc)
    assert sw0[3, 3] and sw0[3, 4] and sw0[3, 5]
    assert not sw0[3, 6]  # |6 - 3| = 3 > 2
    assert not sw0[3, 2]  # query token masked out


def test_create_block_position_ids():
    """Verify RoPE position IDs reset to 0 at the document segment boundary."""
    q_lens = torch.tensor([3, 2])
    d_lens = torch.tensor([4, 3])
    seq_len = 8

    pos_ids = _create_block_position_ids(
        q_lens=q_lens,
        d_lens=d_lens,
        seq_len=seq_len,
        device=torch.device("cpu"),
    )

    assert pos_ids.shape == (2, 8)
    # Sample 0: Q [0, 1, 2], D [0, 1, 2, 3], Pad [0]
    assert torch.equal(pos_ids[0], torch.tensor([0, 1, 2, 0, 1, 2, 3, 0]))
    # Sample 1: Q [0, 1], D [0, 1, 2], Pad [0, 0, 0]
    assert torch.equal(pos_ids[1], torch.tensor([0, 1, 0, 1, 2, 0, 0, 0]))


def test_pack_pair_inputs_padded():
    """Verify vectorized repacking of query and doc tokens without intermediate padding."""
    pad_id = 50283
    q_ids = torch.tensor([[10, 20, 30, pad_id], [40, 50, pad_id, pad_id]])
    q_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    d_ids = torch.tensor([[100, 200, pad_id], [300, 400, 500]])
    d_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    unified_ids, q_lens, d_lens, max_joint_len = _pack_pair_inputs_padded(
        q_ids, q_mask, d_ids, d_mask, pad_token_id=pad_id
    )

    assert max_joint_len == 5
    assert torch.equal(unified_ids[0], torch.tensor([10, 20, 30, 100, 200]))
    assert torch.equal(unified_ids[1], torch.tensor([40, 50, 300, 400, 500]))


@pytest.mark.parametrize("use_cross_attn", [True, False])
@pytest.mark.parametrize("gate_target", ["h", "v"])
def test_unified_vs_legacy_parity_query_doc_ids(use_cross_attn, gate_target):
    """Verify numerical parity to machine epsilon between unified and legacy bottom contextualization."""
    torch.manual_seed(42)
    base_kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        local_attention=4,
        _attn_implementation="sdpa",
        utility_pred=True,
        utility_bias=0.0,
        use_cross_attn=use_cross_attn,
        gate_target=gate_target,
    )

    cfg_unified = ModernBertMiceConfig(unified_bottom_pass=True, **base_kwargs)
    cfg_legacy = ModernBertMiceConfig(unified_bottom_pass=False, **base_kwargs)

    model_unified = ModernBertMiceForSequenceClassification(cfg_unified).eval()
    model_legacy = ModernBertMiceForSequenceClassification(cfg_legacy).eval()
    model_legacy.load_state_dict(model_unified.state_dict())

    q_ids = torch.tensor([[10, 20, 30], [40, 50, 0]])
    q_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    d_ids = torch.tensor([[100, 200, 300, 400], [500, 600, 700, 0]])
    d_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])

    with torch.no_grad():
        out_u = model_unified(
            query_ids=q_ids, query_mask=q_mask, doc_ids=d_ids, doc_mask=d_mask
        ).logits.squeeze(-1)
        out_l = model_legacy(
            query_ids=q_ids, query_mask=q_mask, doc_ids=d_ids, doc_mask=d_mask
        ).logits.squeeze(-1)

    diff = (out_u - out_l).abs().max().item()
    assert diff < 1e-5, f"Logits mismatch: diff={diff}"


def test_unified_vs_legacy_parity_input_ids():
    """Verify numerical parity when passing standard pair input_ids."""
    torch.manual_seed(42)
    base_kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        local_attention=4,
        _attn_implementation="sdpa",
        utility_pred=True,
        utility_bias=0.0,
    )

    cfg_unified = ModernBertMiceConfig(unified_bottom_pass=True, **base_kwargs)
    cfg_legacy = ModernBertMiceConfig(unified_bottom_pass=False, **base_kwargs)

    model_unified = ModernBertMiceForSequenceClassification(cfg_unified).eval()
    model_legacy = ModernBertMiceForSequenceClassification(cfg_legacy).eval()
    model_legacy.load_state_dict(model_unified.state_dict())

    input_ids = torch.tensor([
        [50281, 10, 20, 50282, 100, 200, 50282, 50283],
        [50281, 30, 50282, 400, 500, 600, 50282, 50283],
    ])
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    with torch.no_grad():
        out_u = model_unified(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        out_l = model_legacy(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

    diff = (out_u - out_l).abs().max().item()
    assert diff < 1e-5, f"Logits mismatch on input_ids: diff={diff}"


def test_unified_backward_gradient_flow():
    """Verify that gradients cleanly flow through the unified bottom contextualization layers."""
    torch.manual_seed(42)
    config = ModernBertMiceConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        local_attention=4,
        _attn_implementation="sdpa",
        utility_pred=True,
        unified_bottom_pass=True,
    )
    model = ModernBertMiceForSequenceClassification(config).train()

    q_ids = torch.tensor([[10, 20, 30], [40, 50, 0]])
    q_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    d_ids = torch.tensor([[100, 200, 300, 400], [500, 600, 700, 0]])
    d_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = torch.tensor([1.0, 0.0])

    out = model(
        query_ids=q_ids,
        query_mask=q_mask,
        doc_ids=d_ids,
        doc_mask=d_mask,
        labels=labels,
    )
    loss = out.loss
    loss.backward()

    # Assert gradients exist on embeddings, bottom layers, utility predictor, and top layers
    assert model.model.embeddings.tok_embeddings.weight.grad is not None
    assert model.model.layers[0].attn.Wqkv.weight.grad is not None
    assert model.utility_predictor.wi.weight.grad is not None
    assert model.top_layers[0].self_attn.Wqkv.weight.grad is not None
    assert model.output_modules.dense_score.linear.weight.grad is not None
