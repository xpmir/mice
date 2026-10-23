"""
Tests for MICE V2 Hard Concrete L0 Regularization and Temperature Annealing (Step 2).
Run with: `uv run pytest tests/test_mice_v2_hard_concrete.py -v`
"""

import math
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from MICE.modeling.mice_v2.modernbert_hf import (
    UtilityPredictor,
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
    _hard_prune_unpadded_documents,
    _hard_prune_padded_documents,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    mice_v2_scorer,
    ModernBertMiceScorer,
    UtilityRegularizer,
    compute_cosine_schedule,
    update_utility_temperature,
)
from MICE.experiments.micev2_training import MiceV2_FineTuning, get_name_from_tags
from xpm_torch.trainers import TrainerContext
from xpm_torch.metrics import Metrics


def test_hard_concrete_config_and_scorer():
    """Test config validation, parameter forwarding, and Experimaestro hash distinction."""
    # 1. Config validation
    cfg_default = ModernBertMiceConfig()
    assert cfg_default.use_hard_concrete is False
    assert cfg_default.hc_gamma == pytest.approx(-0.1)
    assert cfg_default.hc_zeta == pytest.approx(1.1)

    cfg_custom = ModernBertMiceConfig(use_hard_concrete=True, hc_gamma=-0.2, hc_zeta=1.2)
    assert cfg_custom.use_hard_concrete is True
    assert cfg_custom.hc_gamma == pytest.approx(-0.2)
    assert cfg_custom.hc_zeta == pytest.approx(1.2)

    # Invalid hc_zeta <= hc_gamma should raise ValueError
    with pytest.raises(ValueError, match="hc_zeta.*must be strictly greater than hc_gamma"):
        ModernBertMiceConfig(hc_gamma=1.1, hc_zeta=-0.1)

    # 2. Experimaestro Scorer and Hash Invariance
    from experimaestro.core.identifier import IdentifierComputer

    scorer_default, tasks_default = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
    )
    scorer_explicit_false, tasks_explicit_false = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
        use_hard_concrete=False,
    )
    scorer_hc, tasks_hc = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
        use_hard_concrete=True,
    )

    id_default = str(IdentifierComputer.compute(scorer_default))
    id_false = str(IdentifierComputer.compute(scorer_explicit_false))
    id_hc = str(IdentifierComputer.compute(scorer_hc))

    assert id_default == id_false, "Default and explicit use_hard_concrete=False must match hash"
    assert id_hc != id_default, "use_hard_concrete=True must produce a distinct hash"

    # 3. Model initialization forwards parameters
    inst = tasks_hc[0].instance()
    inst.execute()
    assert inst.scorer.use_hard_concrete is True
    assert inst.scorer.hf_model.config.use_hard_concrete is True
    assert inst.scorer.hf_model.utility_predictor.use_hard_concrete is True


def test_hard_concrete_sampling_bounds_and_exact_mass():
    """Verify utilities are strictly in [0.0, 1.0], exact 0.0 and 1.0 exist, and no NaNs occur."""
    torch.manual_seed(42)
    pred = UtilityPredictor(
        hidden_dim=32,
        utility_temp=0.67,
        utility_bias=0.0,
        use_hard_concrete=True,
        hc_gamma=-0.1,
        hc_zeta=1.1,
    )
    pred.train()

    # Broad variety of inputs to cover negative, zero, and positive logits
    h = torch.randn(100, 50, 32)
    u = pred(h)

    # 1. Hard bounds
    assert (u >= 0.0).all() and (u <= 1.0).all(), "Hard concrete samples must be in [0.0, 1.0]"
    assert not torch.isnan(u).any(), "Samples must not contain NaNs"

    # 2. Exact 0.0 and 1.0 mass (rectification)
    assert (u == 0.0).any(), "Stretching and rectifying must produce exact 0.0 values"
    assert (u == 1.0).any(), "Stretching and rectifying must produce exact 1.0 values"

    # 3. Stability under extreme logits
    with torch.no_grad():
        pred.wo.weight.fill_(0.0)
        # Extreme positive bias: logits = +100
        pred.wo.bias.fill_(100.0)
        u_high = pred(h)
        assert (u_high == 1.0).all()
        assert not torch.isnan(u_high).any()

        # Extreme negative bias: logits = -100
        pred.wo.bias.fill_(-100.0)
        u_low = pred(h)
        assert (u_low == 0.0).all()
        assert not torch.isnan(u_low).any()


def test_hard_concrete_deterministic_eval_mode():
    """Verify that model.eval() produces deterministic median outputs without sampling noise."""
    pred = UtilityPredictor(
        hidden_dim=32,
        utility_temp=0.5,
        utility_bias=1.0,
        use_hard_concrete=True,
        hc_gamma=-0.1,
        hc_zeta=1.1,
    )
    pred.eval()

    h = torch.randn(10, 20, 32)
    out1 = pred(h)
    out2 = pred(h)

    # In eval mode, outputs must be 100% identical (zero variance)
    assert torch.equal(out1, out2), "Evaluation mode must be completely deterministic (zero sampling noise)"

    # Check that output matches deterministic median formula: clamp(1.2 * sigmoid(logits / beta) - 0.1, 0, 1)
    with torch.no_grad():
        x_val, x_gate = pred.wi(h).chunk(2, dim=-1)
        geglu = x_val * F.gelu(x_gate, approximate="none")
        logits = pred.wo(geglu)
        beta = float(pred.temperature.item())
        expected = torch.clamp(
            torch.sigmoid(logits / beta) * (1.1 - (-0.1)) + (-0.1),
            min=0.0,
            max=1.0,
        )
    assert torch.allclose(out1, expected, atol=1e-6)


def test_p_active_closed_form_formula():
    """Verify last_p_active strictly follows the theoretical CDF sigma(z - beta * log(-gamma / zeta))."""
    gamma, zeta = -0.1, 1.1
    beta = 0.67
    pred = UtilityPredictor(
        hidden_dim=16,
        utility_temp=beta,
        use_hard_concrete=True,
        hc_gamma=gamma,
        hc_zeta=zeta,
    )

    h = torch.randn(8, 12, 16)
    _, p_active = pred(h, return_p_active=True)

    # Theoretical formula
    with torch.no_grad():
        x_val, x_gate = pred.wi(h).chunk(2, dim=-1)
        geglu = x_val * F.gelu(x_gate, approximate="none")
        logits = pred.wo(geglu)
        expected_p_active = torch.sigmoid(logits - beta * math.log(-gamma / zeta))

    assert p_active is not None
    assert torch.allclose(p_active, expected_p_active, atol=1e-6)
    assert (p_active > 0.0).all() and (p_active < 1.0).all()


def test_utility_regularizer_l0_loss_and_metrics():
    """Verify UtilityRegularizer computes L0 presence loss from last_p_active and logs distribution metrics."""
    config = ModernBertMiceConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        n_contextualization_layers=1,
        utility_pred=True,
        use_hard_concrete=True,
        _attn_implementation="sdpa",
    )
    model = ModernBertMiceForSequenceClassification(config)
    model.train()

    # Forward pass to populate last_utilities and last_p_active
    query_ids = torch.randint(0, 50, (2, 4))
    query_mask = torch.ones((2, 4), dtype=torch.long)
    doc_ids = torch.randint(0, 50, (2, 8))
    doc_mask = torch.ones((2, 8), dtype=torch.long)

    out = model(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

    assert out.utilities is not None
    assert out.p_active is not None

    # Hook test
    budget = 0.3
    reg_hook = UtilityRegularizer.C(
        lambda_reg=0.5,
        budget=budget,
        l2_reg=True,
    )

    ctx = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=model,
        optimizer=None,
        fabric=None,
    )
    ctx.metrics = Metrics()

    with ctx.losses() as losses:
        reg_hook(ctx, out.utilities, doc_mask=doc_mask, p_active=out.p_active, hf_model=model)

    # 1. Loss was added
    assert len(losses) == 1
    loss_item = losses[0]
    assert loss_item.name == "utility_reg"
    assert loss_item.weight == 0.5

    # Verify loss calculation uses p_active rather than utilities
    expected_mean_p = out.p_active.mean()
    expected_diff = F.relu(expected_mean_p - budget)
    expected_loss = expected_diff**2
    assert loss_item.value.item() == pytest.approx(expected_loss.item(), rel=1e-5)

    # 2. Verify L0 metrics logged
    m = {name: metric.compute() for name, metric in ctx.metrics.metrics.items()}
    assert "l0/p_active_mean" in m
    assert m["l0/p_active_mean"] == pytest.approx(expected_mean_p.item(), rel=1e-5)
    assert "l0/exact_zeros" in m
    assert "l0/exact_ones" in m
    assert "l0/intermediate" in m
    # Exact zeros + exact ones + intermediate must sum to 1.0
    total_ratio = m["l0/exact_zeros"] + m["l0/exact_ones"] + m["l0/intermediate"]
    assert total_ratio == pytest.approx(1.0, rel=1e-5)


def test_temperature_annealing_steepening():
    """Verify that lowering temperature to 0.1 steepens the distribution into near-binary step values."""
    torch.manual_seed(42)
    pred = UtilityPredictor(
        hidden_dim=32,
        utility_temp=1.0,
        utility_bias=0.0,
        use_hard_concrete=True,
        hc_gamma=-0.1,
        hc_zeta=1.1,
    )
    pred.train()

    h = torch.randn(20, 50, 32)

    # High temperature beta = 1.0
    pred.temperature.fill_(1.0)
    u_warm = pred(h)
    intermediate_warm = ((u_warm > 0.0) & (u_warm < 1.0)).float().mean().item()

    # Low temperature beta = 0.1
    pred.temperature.fill_(0.1)
    u_cold = pred(h)
    intermediate_cold = ((u_cold > 0.0) & (u_cold < 1.0)).float().mean().item()

    # Lower temperature must sharply decrease the proportion of intermediate tokens (>70% drop)
    assert intermediate_cold < intermediate_warm * 0.3, f"Intermediate tokens should drop by >70%, from {intermediate_warm:.2f} to {intermediate_cold:.2f}"
    assert intermediate_cold < 0.20


def test_hard_concrete_gradient_flow():
    """Verify gradients flow simultaneously from both task loss (via u) and L0 regularizer (via last_p_active)."""
    config = ModernBertMiceConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        n_contextualization_layers=1,
        utility_pred=True,
        gate_target="v",
        use_hard_concrete=True,
        _attn_implementation="sdpa",
    )
    model = ModernBertMiceForSequenceClassification(config)
    model.train()

    query_ids = torch.randint(0, 50, (2, 4))
    query_mask = torch.ones((2, 4), dtype=torch.long)
    doc_ids = torch.randint(0, 50, (2, 8))
    doc_mask = torch.ones((2, 8), dtype=torch.long)

    out = model(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

    # Joint loss: task_loss (from logits) + l0_loss (from p_active)
    task_loss = out.logits.sum()
    l0_loss = (out.p_active.mean() - 0.3) ** 2
    total_loss = task_loss + 0.1 * l0_loss

    total_loss.backward()

    # UtilityPredictor weights must have valid, non-zero gradients
    assert model.utility_predictor.wo.weight.grad is not None
    assert model.utility_predictor.wo.weight.grad.abs().sum() > 0.0
    assert model.utility_predictor.wi.weight.grad is not None
    assert model.utility_predictor.wi.weight.grad.abs().sum() > 0.0


def test_training_tag_suffix_hc():
    """Verify get_name_from_tags handles use_hard_concrete suffix '-hc'."""
    cfg_base = MiceV2_FineTuning(use_hard_concrete=False)
    cfg_hc = MiceV2_FineTuning(use_hard_concrete=True)

    tags_base = {"n_contextualization_layers": 2, "use_hard_concrete": False}
    tags_hc = {"n_contextualization_layers": 2, "use_hard_concrete": True}

    name_base = get_name_from_tags(tags_base, cfg_base)
    name_hc = get_name_from_tags(tags_hc, cfg_hc)

    assert "-hc" not in name_base, f"use_hard_concrete=False should not include '-hc', got {name_base}"
    assert "-hc" in name_hc, f"use_hard_concrete=True must include '-hc', got {name_hc}"


def test_hard_pruning_hc_padded_slicing():
    """Verify _hard_prune_padded_documents correctly prunes tokens with u=0 and returns pruned utilities."""
    d_hidden = torch.randn(2, 5, 8)
    doc_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    utilities = torch.tensor(
        [[[0.0], [0.8], [0.0], [0.3], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]]
    )

    new_d_hidden, new_doc_mask, new_pos_ids, new_utilities = (
        _hard_prune_padded_documents(
            d_hidden, doc_mask, utilities, threshold=0.0, return_pruned_utilities=True
        )
    )

    # Doc 0 keeps 2 tokens (indices 1 and 3). Doc 1 keeps 1 token (index 0 safeguard)
    assert new_d_hidden.shape == (2, 2, 8)
    assert new_doc_mask.shape == (2, 2)
    assert new_utilities.shape == (2, 2, 1)

    assert new_utilities[0, 0, 0] == pytest.approx(0.8)
    assert new_utilities[0, 1, 0] == pytest.approx(0.3)
    assert new_doc_mask[0].tolist() == [1, 1]
    assert new_doc_mask[1].tolist() == [1, 0]
    assert (new_d_hidden[0, 0] == d_hidden[0, 1]).all()
    assert (new_d_hidden[0, 1] == d_hidden[0, 3]).all()


def test_hard_pruning_hc_unpadded_slicing():
    """Verify _hard_prune_unpadded_documents correctly prunes 1D unpadded sequence and returns pruned utilities."""
    d_hidden = torch.randn(1, 6, 8)
    d_pos_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])
    cu_seqlens_d = torch.tensor([0, 3, 6], dtype=torch.int32)
    utilities = torch.tensor([[[0.0], [0.7], [0.0], [0.0], [0.0], [0.0]]])

    pruned_d_hidden, pruned_pos, new_cu, new_max, pruned_utils = (
        _hard_prune_unpadded_documents(
            d_hidden,
            d_pos_ids,
            cu_seqlens_d,
            utilities,
            threshold=0.0,
            return_pruned_utilities=True,
        )
    )

    # Doc 0 keeps index 1 (1 token). Doc 1 keeps index 3 (safeguard, 1 token)
    assert pruned_d_hidden.shape == (1, 2, 8)
    assert pruned_utils.shape == (1, 2, 1)
    assert new_cu.tolist() == [0, 1, 2]
    assert new_max == 1
    assert pruned_utils[0, 0, 0] == pytest.approx(0.7)


def test_hard_pruning_hc_model_forward_and_utilities_separation():
    """Verify full unpruned utilities remain on model while pruned utilities are passed to top layers."""
    torch.manual_seed(42)
    config = ModernBertMiceConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        n_contextualization_layers=1,
        utility_pred=True,
        gate_target="v",
        use_hard_concrete=True,
        # hard_pruning defaults to True when use_hard_concrete=True
        _attn_implementation="sdpa",
    )
    assert config.hard_pruning is True
    model = ModernBertMiceForSequenceClassification(config)
    model.eval()

    query_ids = torch.randint(0, 50, (2, 4))
    query_mask = torch.ones((2, 4), dtype=torch.long)
    doc_ids = torch.randint(0, 50, (2, 12))
    doc_mask = torch.ones((2, 12), dtype=torch.long)

    out = model(
        query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask
    )

    assert out.logits.shape == (2, 1)
    # Full document length must be preserved in utilities and p_active
    assert out.utilities.shape == (2, 12, 1)
    assert out.p_active.shape == (2, 12, 1)

    # Pruned utilities must exist and have sequence length <= 12
    assert out.pruned_utilities is not None
    assert out.pruned_utilities.shape[0] == 2
    assert out.pruned_utilities.shape[1] <= 12
    assert out.pruned_utilities.shape[2] == 1


def test_hard_pruning_hc_ablation_disabled():
    """Verify hard_pruning=False with use_hard_concrete=True retains full sequence without physical pruning."""
    config = ModernBertMiceConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        n_contextualization_layers=1,
        utility_pred=True,
        gate_target="v",
        use_hard_concrete=True,
        hard_pruning=False,
        _attn_implementation="sdpa",
    )
    assert config.hard_pruning is False
    model = ModernBertMiceForSequenceClassification(config)
    model.eval()

    query_ids = torch.randint(0, 50, (2, 4))
    query_mask = torch.ones((2, 4), dtype=torch.long)
    doc_ids = torch.randint(0, 50, (2, 10))
    doc_mask = torch.ones((2, 10), dtype=torch.long)

    out = model(
        query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask
    )

    assert out.logits.shape == (2, 1)
    assert out.utilities.shape == (2, 10, 1)
    # When hard_pruning is False, pruned_utilities must be None
    assert out.pruned_utilities is None


def test_hard_pruning_hc_experimaestro_hash_and_tags():
    """Verify hard_pruning tag generation and Experimaestro hash distinction."""
    from experimaestro.core.identifier import IdentifierComputer

    cfg_default = MiceV2_FineTuning(use_hard_concrete=True)
    cfg_no_prune = MiceV2_FineTuning(use_hard_concrete=True, hard_pruning=False)

    name_default = get_name_from_tags(
        {"use_hard_concrete": True, "hard_pruning": True}, cfg_default
    )
    name_no_prune = get_name_from_tags(
        {"use_hard_concrete": True, "hard_pruning": False}, cfg_no_prune
    )

    assert "-hc" in name_default
    assert "-nohardprune" not in name_default
    assert "-hc-nohardprune" in name_no_prune

    scorer_default, _ = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
        use_hard_concrete=True,
    )
    scorer_no_prune, _ = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
        use_hard_concrete=True,
        hard_pruning=False,
    )

    id_default = str(IdentifierComputer.compute(scorer_default))
    id_no_prune = str(IdentifierComputer.compute(scorer_no_prune))
    assert id_default != id_no_prune, "hard_pruning=False must alter Experimaestro identifier hash"


def test_hard_pruning_hc_end_to_end_gradient_flow():
    """Verify task loss and L0 regularizer gradients backpropagate with unified hard pruning active."""
    torch.manual_seed(42)
    config = ModernBertMiceConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        n_contextualization_layers=1,
        utility_pred=True,
        gate_target="v",
        use_hard_concrete=True,
        _attn_implementation="sdpa",
    )
    model = ModernBertMiceForSequenceClassification(config)
    model.train()

    query_ids = torch.randint(0, 50, (2, 4))
    query_mask = torch.ones((2, 4), dtype=torch.long)
    doc_ids = torch.randint(0, 50, (2, 8))
    doc_mask = torch.ones((2, 8), dtype=torch.long)

    out = model(
        query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask
    )

    task_loss = out.logits.sum()
    l0_loss = (out.p_active.mean() - 0.3) ** 2
    total_loss = task_loss + 0.1 * l0_loss
    total_loss.backward()

    assert model.utility_predictor.wo.weight.grad is not None
    assert model.utility_predictor.wo.weight.grad.abs().sum() > 0.0
    assert model.utility_predictor.wi.weight.grad is not None
    assert model.utility_predictor.wi.weight.grad.abs().sum() > 0.0
