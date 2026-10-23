"""
Tests for MICE V2 Soft Gate Strategy & Utility Predictor.
Run with: `uv run pytest tests/test_mice_v2_soft_gate.py -v`
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch

from MICE.modeling.mice_v2.modernbert_hf import (
    UtilityPredictor,
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    mice_v2_scorer,
    ModernBertMiceScorer,
    UtilityRegularizer,
    GatedValueRegularizer,
    compute_linear_schedule,
    compute_cosine_schedule,
    compute_temperature_schedule,
    update_utility_temperature,
)
from xpmir.letor.records import PointwiseItems
from xpm_torch.trainers import TrainerContext
from xpm_torch.metrics import Metrics, MetricsLogger
import logging

logging.basicConfig(level=logging.INFO)


def test_utility_predictor_direct():
    """Test UtilityPredictor module outputs, shapes, and value bounds."""
    batch_size, seq_len, d_model = 4, 32, 128
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    predictor = UtilityPredictor(hidden_dim=d_model)
    utilities = predictor(hidden_states)

    assert utilities.shape == (batch_size, seq_len, 1), f"Expected shape {(batch_size, seq_len, 1)}, got {utilities.shape}"
    assert (utilities >= 0.0).all() and (utilities <= 1.0).all(), "Utility scores must be strictly bounded in (0, 1)"
    assert utilities.mean().item() > 0.95, f"Expected initial mean utility > 0.95, got {utilities.mean().item()}"

    # Test unpadded 1D input tensor [1, total_tokens, d_model]
    unpadded_hidden = torch.randn(1, 100, d_model)
    unpadded_utilities = predictor(unpadded_hidden)
    assert unpadded_utilities.shape == (1, 100, 1)


def test_utility_regularizer_unpadded_fa2():
    """Test UtilityRegularizer handling 1D unpadded FA2 utilities with 2D doc_mask."""
    reg_hook = UtilityRegularizer.C(lambda_reg=0.1)

    # 1D unpadded FA2 utilities [1, total_unpadded_tokens, 1] vs 2D padded doc_mask [batch_size, max_seqlen]
    fa2_utilities = torch.rand((1, 37720, 1))
    doc_mask = torch.ones((256, 1024), dtype=torch.long)

    class MockPath:
        pass

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )

    with context.losses() as losses:
        reg_hook(context, fa2_utilities, doc_mask=doc_mask)

    assert len(losses) == 1
    assert losses[0].name == "utility_reg"
    assert losses[0].value.item() == pytest.approx(fa2_utilities.mean().item())


def test_utility_regularizer_histogram_logging():
    """Test UtilityRegularizer calling writer.add_histogram when step matches histogram_interval."""
    reg_hook = UtilityRegularizer.C(lambda_reg=0.1, histogram_interval=100)

    utilities = torch.rand((4, 16, 1))
    doc_mask = torch.ones((4, 16), dtype=torch.long)

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    context.metrics = Metrics()

    histogram_calls = []

    class MockLogger(MetricsLogger):
        def log_scalar(self, tag: str, value: float, step: int) -> None:
            pass

        def log_histogram(self, tag: str, values: object, step: int) -> None:
            histogram_calls.append((tag, getattr(values, "shape", None), step))

    context._logger = MockLogger()

    # Step 50: step % 100 != 0 -> should NOT log histogram
    context.state.steps = 50
    with context.losses():
        reg_hook(context, utilities, doc_mask=doc_mask)
    assert len(histogram_calls) == 0

    # Step 100: step % 100 == 0 -> SHOULD log histogram
    context.state.steps = 100
    with context.losses():
        reg_hook(context, utilities, doc_mask=doc_mask)
    assert len(histogram_calls) == 1
    assert histogram_calls[0][0] == "train/utility/distribution"
    assert histogram_calls[0][2] == 100


def test_utility_regularizer_bfloat16_histogram_logging():
    """Test UtilityRegularizer does not crash with TypeError: Got unsupported ScalarType BFloat16."""
    import numpy as np

    reg_hook = UtilityRegularizer.C(lambda_reg=0.1, histogram_interval=100)

    # utilities in bfloat16
    utilities = torch.rand((4, 16, 1), dtype=torch.bfloat16)
    doc_mask = torch.ones((4, 16), dtype=torch.long)

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    context.metrics = Metrics()

    histogram_calls = []

    class MockLogger(MetricsLogger):
        def log_scalar(self, tag: str, value: float, step: int) -> None:
            pass

        def log_histogram(self, tag: str, values: object, step: int) -> None:
            histogram_calls.append((tag, getattr(values, "dtype", None), step))

    context._logger = MockLogger()
    context.state.steps = 100
    with context.losses():
        reg_hook(context, utilities, doc_mask=doc_mask)

    assert len(histogram_calls) == 1
    assert histogram_calls[0][0] == "train/utility/distribution"
    assert histogram_calls[0][1] == np.float32


def test_modernbert_hf_soft_gate():
    """Test ModernBertMiceForSequenceClassification with utility_pred enabled."""
    config = ModernBertMiceConfig(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        utility_pred=True,
    )
    model = ModernBertMiceForSequenceClassification(config)
    model.eval()

    query_ids = torch.randint(0, 1000, (2, 8))
    query_mask = torch.ones((2, 8), dtype=torch.long)
    doc_ids = torch.randint(0, 1000, (2, 16))
    doc_mask = torch.ones((2, 16), dtype=torch.long)

    with torch.no_grad():
        out = model(
            query_ids=query_ids,
            query_mask=query_mask,
            doc_ids=doc_ids,
            doc_mask=doc_mask,
        )

    assert out.utilities is not None, "out.utilities should be populated when utility_pred=True"
    assert out.utilities.shape == (2, 16, 1), f"Expected utilities shape (2, 16, 1), got {out.utilities.shape}"
    assert (out.utilities >= 0.0).all() and (out.utilities <= 1.0).all()
    assert out.utilities.mean().item() > 0.95, f"Expected initial mean utility > 0.95, got {out.utilities.mean().item()}"


def test_modernbert_xpm_soft_gate_and_hook():
    """Test ModernBertMiceScorer with utility_pred=True and UtilityRegularizer hook."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    scorer.eval()

    queries = ["What is IR?", "How does soft gate work?"]
    documents = [
        "Information Retrieval is the activity of obtaining information resources relevant to an information need.",
        "Soft gating multiplies document token hidden states with predicted token utility scores.",
    ]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        context = TrainerContext(
            logpath=tmp_path / "logs",
            path=tmp_path / "ckpts",
            max_epoch=1,
            steps_per_epoch=10,
            trainer=None,
            model=scorer,
            optimizer=None,
            fabric=None,
        )
        context.metrics = Metrics()
        reg_hook = UtilityRegularizer.C(lambda_reg=0.1)
        context.add_hook(reg_hook)

        with context.losses() as losses:
            output = scorer(input_records)
            assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"

        # Verify UtilityRegularizer registered loss in context
        assert len(losses) == 1, f"Expected 1 loss registered in context, got {len(losses)}"
        reg_loss_item = losses[0]
        assert reg_loss_item.name == "utility_reg"
        assert reg_loss_item.weight == 0.1
        assert reg_loss_item.value > 0.0

        # Verify metrics logged to context
        assert "utility_reg" in context.metrics.metrics
        assert "utility/mean" in context.metrics.metrics
        assert "utility/sparsity" in context.metrics.metrics


def test_soft_gate_save_load():
    """Test saving and loading a soft-gated MICE v2 model persistence."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir)
        scorer.save_model(save_path)

        # Verify config.json contains utility_pred=True
        reloaded_config = ModernBertMiceConfig.from_pretrained(save_path)
        assert getattr(reloaded_config, "utility_pred", False) is True, "utility_pred flag must persist in config.json"

        # Reload model weights into new instance
        reloaded_model = ModernBertMiceForSequenceClassification.from_pretrained(save_path)
        assert reloaded_model.utility_predictor is not None, "Reloaded model must instantiate utility_predictor"


def test_legacy_checkpoint_loading_without_wo_bias():
    """Test loading a legacy checkpoint state_dict where utility_predictor.wo.bias is missing."""
    config = ModernBertMiceConfig(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        utility_pred=True,
    )
    model = ModernBertMiceForSequenceClassification(config)

    sd = model.state_dict()
    assert "utility_predictor.wo.bias" in sd
    del sd["utility_predictor.wo.bias"]

    new_model = ModernBertMiceForSequenceClassification(config)
    new_model.load_state_dict(sd, strict=True)
    assert new_model.utility_predictor.wo.bias is not None
    assert new_model.utility_predictor.wo.bias.item() == 0.0


def test_top_layers_doc_ln_toggle():
    """Test top_layers_doc_ln=True vs top_layers_doc_ln=False behavior and LayerNorm instantiation."""
    # 1. top_layers_doc_ln=False (Default legacy mode)
    config_false = ModernBertMiceConfig(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        utility_pred=True,
        top_layers_doc_ln=False,
    )
    model_false = ModernBertMiceForSequenceClassification(config_false)
    assert isinstance(model_false.top_layers[0].doc_cross_attn_norm, torch.nn.Identity)

    # 2. top_layers_doc_ln=True (Post-LN gating mode)
    config_true = ModernBertMiceConfig(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        utility_pred=True,
        top_layers_doc_ln=True,
    )
    model_true = ModernBertMiceForSequenceClassification(config_true)
    assert isinstance(model_true.top_layers[0].doc_cross_attn_norm, torch.nn.LayerNorm)

    # Verify forward pass for both
    query_ids = torch.randint(0, 1000, (2, 8))
    query_mask = torch.ones((2, 8), dtype=torch.long)
    doc_ids = torch.randint(0, 1000, (2, 16))
    doc_mask = torch.ones((2, 16), dtype=torch.long)

    with torch.no_grad():
        out_false = model_false(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)
        out_true = model_true(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

    assert out_false.logits.shape == (2, 1)
    assert out_true.logits.shape == (2, 1)
    assert out_false.utilities.shape == (2, 16, 1)
    assert out_true.utilities.shape == (2, 16, 1)


def test_top_layers_doc_ln_backbone_seeding():
    """Test InitMICEV2ModernBERTFromHFID copying LayerNorm weights into top_layer.doc_cross_attn_norm."""
    from transformers import AutoModelForSequenceClassification

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
        top_layers_doc_ln=True,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    full_backbone = AutoModelForSequenceClassification.from_pretrained("jhu-clsp/ettin-encoder-17m")

    assert isinstance(scorer.hf_model.top_layers[0].doc_cross_attn_norm, torch.nn.LayerNorm)
    # Check that weight was populated from backbone layer 2 attn_norm
    doc_ln = scorer.hf_model.top_layers[0].doc_cross_attn_norm
    src_attn_norm = full_backbone.model.layers[2].attn_norm
    assert doc_ln.weight is not None
    assert torch.equal(doc_ln.weight, src_attn_norm.weight)


def test_bottom_top_final_norm_toggles():
    """Test bottom_final_norm and top_final_norm toggles and backbone seeding."""
    from transformers import AutoModelForSequenceClassification

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
        bottom_final_norm=False,
        top_final_norm=True,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    assert scorer.hf_model.config.bottom_final_norm is False
    assert scorer.hf_model.config.top_final_norm is True
    assert isinstance(scorer.hf_model.top_final_norm, torch.nn.LayerNorm)

    # Test forward pass in unpadded and padded modes
    query_ids = torch.randint(0, 1000, (2, 8))
    query_mask = torch.ones((2, 8), dtype=torch.bool)
    doc_ids = torch.randint(0, 1000, (2, 16))
    doc_mask = torch.ones((2, 16), dtype=torch.bool)

    with torch.no_grad():
        out = scorer.hf_model(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)
    assert out.logits.shape == (2, 1)


def test_utility_regularizer_wv_monitoring():
    """Test that UtilityRegularizer logs W_V matrix Frobenius norms to metrics."""
    config = ModernBertMiceConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        utility_pred=True,
    )
    model = ModernBertMiceForSequenceClassification(config)

    reg_hook = UtilityRegularizer.C(lambda_reg=0.1)

    utilities = torch.rand((2, 8, 1))
    doc_mask = torch.ones((2, 8), dtype=torch.long)

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    context.metrics = Metrics()

    with context.losses():
        reg_hook(context, utilities, doc_mask=doc_mask, hf_model=model)

    metrics_dict = {name: metric.compute() for name, metric in context.metrics.metrics.items()}
    assert "wv_norm/mean" in metrics_dict, "wv_norm/mean should be recorded in metrics"
    assert "wv_norm/layer_0" in metrics_dict, "wv_norm/layer_0 should be recorded in metrics"

    expected_wv_list = model.get_wv_matrices()
    expected_layer0 = float(torch.norm(expected_wv_list[0], p="fro").item())
    assert metrics_dict["wv_norm/layer_0"] == pytest.approx(expected_layer0, rel=1e-4)


def test_gated_value_regularizer():
    """Test GatedValueRegularizer computing u * W_V * h norm loss and logging metrics."""
    config = ModernBertMiceConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        utility_pred=True,
    )
    model = ModernBertMiceForSequenceClassification(config)

    query_ids = torch.randint(0, 100, (2, 4))
    query_mask = torch.ones((2, 4), dtype=torch.bool)
    doc_ids = torch.randint(0, 100, (2, 8))
    doc_mask = torch.ones((2, 8), dtype=torch.bool)

    out = model(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)
    assert out.utilities is not None
    assert out.doc_hidden is not None

    reg_hook = GatedValueRegularizer.C(lambda_reg=0.1)

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=model,
        optimizer=None,
        fabric=None,
    )
    context.metrics = Metrics()

    with context.losses() as losses:
        reg_hook(
            context,
            model,
            out.utilities,
            out.doc_hidden,
            doc_mask=doc_mask,
        )

    assert len(losses) == 1
    assert losses[0].name == "gated_value_reg"
    assert losses[0].value.item() > 0.0

    metrics_dict = {name: metric.compute() for name, metric in context.metrics.metrics.items()}
    assert "value_norm/mean" in metrics_dict
    assert "utility/mean" in metrics_dict
    assert "utility/sparsity" in metrics_dict
    assert "wv_norm/mean" in metrics_dict
    assert "wv_norm/layer_0" in metrics_dict


def test_gated_value_scale_invariance():
    """Test that scaling down u while scaling up W_V leaves GatedValueRegularizer loss invariant."""
    config = ModernBertMiceConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        utility_pred=True,
    )
    model = ModernBertMiceForSequenceClassification(config)

    utilities = torch.full((2, 8, 1), 0.8)
    doc_hidden = torch.randn(2, 8, 64)
    doc_mask = torch.ones((2, 8), dtype=torch.long)

    val_reg = GatedValueRegularizer.C(lambda_reg=0.1)
    u_reg = UtilityRegularizer.C(lambda_reg=0.1)

    # 1. Baseline context
    ctx1 = TrainerContext(logpath=Path("/tmp/logs"), path=Path("/tmp/ckpts"), max_epoch=1, steps_per_epoch=10, trainer=None, model=model, optimizer=None, fabric=None)
    with ctx1.losses() as losses1:
        val_reg(ctx1, model, utilities, doc_hidden, doc_mask)
        u_reg(ctx1, utilities, doc_mask, hf_model=model)

    val_loss1 = losses1[0].value.item()
    u_loss1 = losses1[1].value.item()

    # 2. Scale u down by 10x and scale W_V up by 10x
    scaled_utilities = utilities * 0.1
    top_layer0_Wqkv = model.top_layers[0].cross_attn.Wqkv
    with torch.no_grad():
        top_layer0_Wqkv.weight[128:192, :] *= 10.0  # Scale W_V up by 10x

    ctx2 = TrainerContext(logpath=Path("/tmp/logs"), path=Path("/tmp/ckpts"), max_epoch=1, steps_per_epoch=10, trainer=None, model=model, optimizer=None, fabric=None)
    with ctx2.losses() as losses2:
        val_reg(ctx2, model, scaled_utilities, doc_hidden, doc_mask)
        u_reg(ctx2, scaled_utilities, doc_mask, hf_model=model)

    val_loss2 = losses2[0].value.item()
    u_loss2 = losses2[1].value.item()

    # UtilityRegularizer loss should drop by ~10x
    assert u_loss2 == pytest.approx(u_loss1 * 0.1, rel=1e-3)
    # GatedValueRegularizer loss should remain invariant
    assert val_loss2 == pytest.approx(val_loss1, rel=1e-3)


def test_micev2_fine_tuning_reg_target_config():
    """Test MiceV2_FineTuning reg_target default and options."""
    from MICE.experiments.micev2_training import MiceV2_FineTuning
    cfg = MiceV2_FineTuning()
    assert cfg.reg_target == "u"

    cfg_val = MiceV2_FineTuning(reg_target="u_wv_h")
    assert cfg_val.reg_target == "u_wv_h"


def test_soft_gate_eval_mode_shape_matching():
    """Test that model.eval() executes soft gating without shape mismatch across varying batch lengths."""
    config = ModernBertMiceConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        utility_pred=True,
        top_layers_doc_ln=True,
    )
    model = ModernBertMiceForSequenceClassification(config)
    model.eval()

    # Batch 1: doc length 16
    q1 = torch.randint(0, 100, (2, 4))
    qm1 = torch.ones((2, 4), dtype=torch.bool)
    d1 = torch.randint(0, 100, (2, 16))
    dm1 = torch.ones((2, 16), dtype=torch.bool)

    out1 = model(query_ids=q1, query_mask=qm1, doc_ids=d1, doc_mask=dm1)
    assert out1.logits.shape == (2, 1)

    # Batch 2: doc length 24 (different length) - must not crash with shape mismatch
    q2 = torch.randint(0, 100, (2, 4))
    qm2 = torch.ones((2, 4), dtype=torch.bool)
    d2 = torch.randint(0, 100, (2, 24))
    dm2 = torch.ones((2, 24), dtype=torch.bool)

    out2 = model(query_ids=q2, query_mask=qm2, doc_ids=d2, doc_mask=dm2)
    assert out2.logits.shape == (2, 1)


def test_compensation_diagnostics_metrics():
    """Test that wo_norm, doc_hidden_norm, gated_doc_norm, and gated_contrast_ratio are logged."""
    config = ModernBertMiceConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=4,
        n_contextualization_layers=2,
        utility_pred=True,
    )
    model = ModernBertMiceForSequenceClassification(config)

    q = torch.randint(0, 100, (2, 4))
    qm = torch.ones((2, 4), dtype=torch.bool)
    d = torch.randint(0, 100, (2, 8))
    dm = torch.ones((2, 8), dtype=torch.bool)

    out = model(query_ids=q, query_mask=qm, doc_ids=d, doc_mask=dm)

    reg_hook = UtilityRegularizer.C(lambda_reg=0.1)

    context = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=model,
        optimizer=None,
        fabric=None,
    )
    context.metrics = Metrics()

    with context.losses():
        reg_hook(context, out.utilities, doc_mask=dm, doc_hidden=out.doc_hidden, hf_model=model)

    metrics_dict = {name: metric.compute() for name, metric in context.metrics.metrics.items()}

    assert "wo_norm/mean" in metrics_dict, "wo_norm/mean should be logged"
    assert "wo_norm/layer_0" in metrics_dict, "wo_norm/layer_0 should be logged"
    assert "doc_hidden_norm/mean" in metrics_dict, "doc_hidden_norm/mean should be logged"
    assert "gated_doc_norm/mean" in metrics_dict, "gated_doc_norm/mean should be logged"


def test_utility_regularizer_budget_above_and_below():
    """Test UtilityRegularizer with target retention budget (over budget vs under budget)."""
    # 1. Over budget case: mean_utility = 0.5 > budget = 0.3
    reg_hook_budget = UtilityRegularizer.C(lambda_reg=0.1, budget=0.3)
    ctx_over = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    ctx_over.metrics = Metrics()
    u_over = torch.full((2, 10, 1), 0.5, requires_grad=True)

    with ctx_over.losses() as losses:
        reg_hook_budget(ctx_over, u_over)

    # Loss = lambda_reg * relu(0.5 - 0.3) = 0.1 * 0.2 = 0.02
    assert len(losses) == 1
    assert losses[0].value.item() == pytest.approx(0.2, rel=1e-5)
    loss_over = losses[0].value * 0.1
    loss_over.backward()
    assert u_over.grad is not None and u_over.grad.abs().sum() > 0

    metrics_over = {name: metric.compute() for name, metric in ctx_over.metrics.metrics.items()}
    assert metrics_over["utility/budget"] == pytest.approx(0.3)
    assert metrics_over["utility/over_budget"] == pytest.approx(0.2)
    assert metrics_over["utility_reg"] == pytest.approx(0.02)

    # 2. Under budget case: mean_utility = 0.2 <= budget = 0.3
    ctx_under = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    ctx_under.metrics = Metrics()
    u_under = torch.full((2, 10, 1), 0.2, requires_grad=True)

    with ctx_under.losses() as losses_under:
        reg_hook_budget(ctx_under, u_under)

    assert len(losses_under) == 1
    assert losses_under[0].value.item() == pytest.approx(0.0)
    loss_under = losses_under[0].value * 0.1
    loss_under.backward()
    # Gradient must be strictly zero when under budget
    assert u_under.grad is None or u_under.grad.abs().sum() == 0

    metrics_under = {name: metric.compute() for name, metric in ctx_under.metrics.metrics.items()}
    assert metrics_under["utility/budget"] == pytest.approx(0.3)
    assert metrics_under["utility/over_budget"] == pytest.approx(0.0)
    assert metrics_under["utility_reg"] == pytest.approx(0.0)

    # 3. Default budget=None case: legacy L1 loss
    reg_hook_default = UtilityRegularizer.C(lambda_reg=0.1)
    ctx_default = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    ctx_default.metrics = Metrics()
    with ctx_default.losses() as losses_default:
        reg_hook_default(ctx_default, u_over)

    assert len(losses_default) == 1
    assert losses_default[0].value.item() == pytest.approx(0.5)
    metrics_default = {name: metric.compute() for name, metric in ctx_default.metrics.metrics.items()}
    assert "utility/budget" not in metrics_default


def test_micev2_fine_tuning_reg_budget_config():
    """Test MiceV2_FineTuning reg_budget configuration."""
    from MICE.experiments.micev2_training import MiceV2_FineTuning
    cfg = MiceV2_FineTuning()
    assert cfg.reg_budget is None

    cfg_budget = MiceV2_FineTuning(reg_budget=0.3)
    assert cfg_budget.reg_budget == 0.3

    assert cfg.l2_reg is False
    cfg_l2 = MiceV2_FineTuning(l2_reg=True)
    assert cfg_l2.l2_reg is True


def test_utility_regularizer_l2_reg():
    """Test UtilityRegularizer with l2_reg=True (squared penalty, with and without budget)."""
    # 1. Without budget: mean_utility = 0.5 -> loss = 0.5**2 = 0.25
    reg_hook_l2 = UtilityRegularizer.C(lambda_reg=0.1, l2_reg=True)
    ctx1 = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    ctx1.metrics = Metrics()
    u = torch.full((2, 10, 1), 0.5, requires_grad=True)

    with ctx1.losses() as losses1:
        reg_hook_l2(ctx1, u)

    assert len(losses1) == 1
    # reg_loss = 0.5**2 = 0.25
    assert losses1[0].value.item() == pytest.approx(0.25, rel=1e-5)
    metrics1 = {name: metric.compute() for name, metric in ctx1.metrics.metrics.items()}
    assert metrics1["utility_reg"] == pytest.approx(0.1 * 0.25)

    # 2. With budget: mean_utility = 0.5, budget = 0.3 -> diff = 0.2 -> loss = 0.2**2 = 0.04
    reg_hook_l2_budget = UtilityRegularizer.C(lambda_reg=0.1, budget=0.3, l2_reg=True)
    ctx2 = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    ctx2.metrics = Metrics()
    u2 = torch.full((2, 10, 1), 0.5, requires_grad=True)

    with ctx2.losses() as losses2:
        reg_hook_l2_budget(ctx2, u2)

    assert len(losses2) == 1
    # reg_loss = (0.5 - 0.3)**2 = 0.04
    assert losses2[0].value.item() == pytest.approx(0.04, rel=1e-5)
    metrics2 = {name: metric.compute() for name, metric in ctx2.metrics.metrics.items()}
    assert metrics2["utility/budget"] == pytest.approx(0.3)
    assert metrics2["utility/over_budget"] == pytest.approx(0.2)
    assert metrics2["utility_reg"] == pytest.approx(0.1 * 0.04)

    # 3. Under budget: mean_utility = 0.2 <= budget = 0.3 -> loss = 0.0
    ctx3 = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    ctx3.metrics = Metrics()
    u3 = torch.full((2, 10, 1), 0.2, requires_grad=True)

    with ctx3.losses() as losses3:
        reg_hook_l2_budget(ctx3, u3)

    assert len(losses3) == 1
    assert losses3[0].value.item() == pytest.approx(0.0)
    metrics3 = {name: metric.compute() for name, metric in ctx3.metrics.metrics.items()}
    assert metrics3["utility/over_budget"] == pytest.approx(0.0)
    assert metrics3["utility_reg"] == pytest.approx(0.0)


def test_compute_linear_schedule():
    """Test compute_linear_schedule interpolation, clamping, and edge cases."""
    # Linear ramp from 1.0 to 0.1 over 100 steps
    assert compute_linear_schedule(1.0, 0.1, step=0, total_steps=100) == pytest.approx(1.0)
    assert compute_linear_schedule(1.0, 0.1, step=50, total_steps=100) == pytest.approx(0.55)
    assert compute_linear_schedule(1.0, 0.1, step=100, total_steps=100) == pytest.approx(0.1)

    # Beyond total steps: clamped to end_val
    assert compute_linear_schedule(1.0, 0.1, step=150, total_steps=100) == pytest.approx(0.1)

    # Negative step: clamped to start_val
    assert compute_linear_schedule(1.0, 0.1, step=-10, total_steps=100) == pytest.approx(1.0)

    # When end_val is None: remains fixed at start_val
    assert compute_linear_schedule(1.0, None, step=50, total_steps=100) == pytest.approx(1.0)

    # Edge case: total_steps <= 0 returns end_val
    assert compute_linear_schedule(1.0, 0.1, step=0, total_steps=0) == pytest.approx(0.1)


def test_compute_cosine_and_temperature_schedule():
    """Test compute_cosine_schedule, compute_temperature_schedule, and schedule dispatch."""
    import math

    # 1. Cosine boundaries and midpoint
    assert compute_cosine_schedule(1.0, 0.1, step=0, total_steps=100) == pytest.approx(1.0)
    assert compute_cosine_schedule(1.0, 0.1, step=50, total_steps=100) == pytest.approx(0.55)
    assert compute_cosine_schedule(1.0, 0.1, step=100, total_steps=100) == pytest.approx(0.1)

    # 2. S-curve property: slower initial decay than linear
    # At step 25 (25% progress):
    # linear is 1.0 - 0.25 * 0.9 = 0.775
    # cosine is 0.1 + 0.45 * (1 + cos(pi/4)) = 0.1 + 0.45 * (1 + sqrt(2)/2) ~= 0.8682
    cos_step25 = compute_cosine_schedule(1.0, 0.1, step=25, total_steps=100)
    lin_step25 = compute_linear_schedule(1.0, 0.1, step=25, total_steps=100)
    assert cos_step25 > lin_step25
    expected_cos_step25 = 0.1 + 0.5 * 0.9 * (1.0 + math.cos(math.pi * 0.25))
    assert cos_step25 == pytest.approx(expected_cos_step25)

    # 3. Beyond total steps and negative steps
    assert compute_cosine_schedule(1.0, 0.1, step=150, total_steps=100) == pytest.approx(0.1)
    assert compute_cosine_schedule(1.0, 0.1, step=-10, total_steps=100) == pytest.approx(1.0)
    assert compute_cosine_schedule(1.0, None, step=50, total_steps=100) == pytest.approx(1.0)
    assert compute_cosine_schedule(1.0, 0.1, step=0, total_steps=0) == pytest.approx(0.1)

    # 4. compute_temperature_schedule dispatch
    assert compute_temperature_schedule(1.0, 0.1, step=25, total_steps=100, schedule="linear") == pytest.approx(lin_step25)
    assert compute_temperature_schedule(1.0, 0.1, step=25, total_steps=100, schedule="cosine") == pytest.approx(cos_step25)

    # 5. Unknown schedule raises ValueError (no silent errors)
    with pytest.raises(ValueError, match="Unknown temperature schedule 'unknown'"):
        compute_temperature_schedule(1.0, 0.1, step=25, total_steps=100, schedule="unknown")


def test_utility_predictor_temperature_always_divides():
    """Test that UtilityPredictor registers temperature buffer, always divides by temperature, and classification model has no proxy helpers."""
    # 1. Default temperature is 1.0
    pred = UtilityPredictor(hidden_dim=64)
    assert hasattr(pred, "temperature")
    assert pred.temperature.item() == pytest.approx(1.0)

    # 2. Custom utility_temp
    pred_custom = UtilityPredictor(hidden_dim=64, utility_temp=0.5)
    assert pred_custom.temperature.item() == pytest.approx(0.5)

    # 3. Always divides by temperature:
    # With a fixed input, lowering temperature from 1.0 to 0.1 scales logits 10x,
    # pushing positive utilities even closer to 1.0.
    h = torch.randn(2, 5, 64)
    u_base = pred(h)
    pred.temperature.fill_(0.1)
    u_steep = pred(h)

    # For positive logits (where bias=5.0 dominates), u_steep > u_base
    assert (u_steep >= u_base).all()
    # At temp=0.1, sigmoid(50.0) is virtually 1.0
    assert u_steep.mean().item() == pytest.approx(1.0, rel=1e-3)

    # 4. Assert ModernBertMiceForSequenceClassification has NO set_temperature or get_temperature
    config = ModernBertMiceConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=3,
        n_contextualization_layers=2,
        utility_pred=True,
        utility_temp=0.7,
    )
    model = ModernBertMiceForSequenceClassification(config)
    assert not hasattr(model, "set_temperature")
    assert not hasattr(model, "get_temperature")
    assert model.utility_predictor.temperature.item() == pytest.approx(0.7)


def test_utility_regularizer_temperature_annealing():
    """Test that UtilityRegularizer anneals temperature across training steps and logs utility/temperature."""
    config = ModernBertMiceConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=3,
        n_contextualization_layers=2,
        utility_pred=True,
        utility_temp=1.0,
    )
    model = ModernBertMiceForSequenceClassification(config)

    reg_hook = UtilityRegularizer.C(
        lambda_reg=0.1,
        temp_start=1.0,
        temp_end=0.1,
    )

    ctx = TrainerContext(
        logpath=Path("/tmp/logs"),
        path=Path("/tmp/ckpts"),
        max_epoch=2,
        steps_per_epoch=50,  # 100 total steps
        trainer=None,
        model=model,
        optimizer=None,
        fabric=None,
    )
    u = torch.full((2, 5, 1), 0.8, requires_grad=True)

    # Step 0: temperature should be 1.0
    ctx.state.steps = 0
    ctx.metrics = Metrics()
    with ctx.losses():
        reg_hook(ctx, u, hf_model=model)
    assert model.utility_predictor.temperature.item() == pytest.approx(1.0)
    m0 = {name: metric.compute() for name, metric in ctx.metrics.metrics.items()}
    assert m0["utility/temperature"] == pytest.approx(1.0)

    # Step 50 (midpoint): temperature should be 0.55
    ctx.state.steps = 50
    ctx.metrics = Metrics()
    with ctx.losses():
        reg_hook(ctx, u, hf_model=model)
    assert model.utility_predictor.temperature.item() == pytest.approx(0.55)
    m50 = {name: metric.compute() for name, metric in ctx.metrics.metrics.items()}
    assert m50["utility/temperature"] == pytest.approx(0.55)

    # Step 100 (final step): temperature should be 0.1
    ctx.state.steps = 100
    ctx.metrics = Metrics()
    with ctx.losses():
        reg_hook(ctx, u, hf_model=model)
    assert model.utility_predictor.temperature.item() == pytest.approx(0.1)
    m100 = {name: metric.compute() for name, metric in ctx.metrics.metrics.items()}
    assert m100["utility/temperature"] == pytest.approx(0.1)

    # Test temp_schedule="cosine"
    reg_hook_cos = UtilityRegularizer.C(
        lambda_reg=0.1,
        temp_start=1.0,
        temp_end=0.1,
        temp_schedule="cosine",
    )
    ctx.state.steps = 25
    ctx.metrics = Metrics()
    with ctx.losses():
        reg_hook_cos(ctx, u, hf_model=model)
    expected_cos_25 = compute_cosine_schedule(1.0, 0.1, step=25, total_steps=100)
    assert model.utility_predictor.temperature.item() == pytest.approx(expected_cos_25)
    m_cos25 = {name: metric.compute() for name, metric in ctx.metrics.metrics.items()}
    assert m_cos25["utility/temperature"] == pytest.approx(expected_cos_25)


def test_modernbert_mice_config_owns_only_utility_temp():
    """Test that ModernBertMiceConfig owns utility_temp, and does not own schedule parameters."""
    cfg = ModernBertMiceConfig(utility_temp=0.8)
    assert cfg.utility_temp == 0.8
    assert not hasattr(cfg, "temp_start")
    assert not hasattr(cfg, "temp_end")
    assert not hasattr(cfg, "temp_anneal_steps")


def test_utility_bias_init():
    """Test that utility_bias initializes wo.bias in UtilityPredictor."""
    # Direct UtilityPredictor check
    p_default = UtilityPredictor(hidden_dim=64)
    assert p_default.utility_bias == 5.0
    assert p_default.wo.bias.item() == pytest.approx(5.0)

    p_zero = UtilityPredictor(hidden_dim=64, utility_bias=0.0)
    assert p_zero.utility_bias == 0.0
    assert p_zero.wo.bias.item() == pytest.approx(0.0)

    # Full model check with default 5.0
    cfg_default = ModernBertMiceConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, utility_pred=True)
    model_default = ModernBertMiceForSequenceClassification(cfg_default)
    assert model_default.config.utility_bias == 5.0
    assert model_default.utility_predictor.utility_bias == 5.0
    assert model_default.utility_predictor.wo.bias.item() == pytest.approx(5.0)

    # Full model check with custom bias 0.0
    cfg_zero = ModernBertMiceConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, utility_pred=True, utility_bias=0.0)
    model_zero = ModernBertMiceForSequenceClassification(cfg_zero)
    assert model_zero.config.utility_bias == 0.0
    assert model_zero.utility_predictor.utility_bias == 0.0
    assert model_zero.utility_predictor.wo.bias.item() == pytest.approx(0.0)

    # Full model check with negative bias -2.5
    cfg_neg = ModernBertMiceConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, utility_pred=True, utility_bias=-2.5)
    model_neg = ModernBertMiceForSequenceClassification(cfg_neg)
    assert model_neg.utility_predictor.wo.bias.item() == pytest.approx(-2.5)


def test_utility_bias_scorer_and_hash_invariance():
    """Test Experimaestro hash invariance and scorer execution with utility_bias."""
    from experimaestro.core.identifier import IdentifierComputer

    scorer_default, tasks_default = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
    )
    scorer_explicit_5, tasks_explicit_5 = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
        utility_bias=5.0,
    )
    scorer_zero, tasks_zero = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        utility_pred=True,
        utility_bias=0.0,
    )

    # Hashes: default and 5.0 must be identical because ignore_default=True
    id_default = str(IdentifierComputer.compute(scorer_default))
    id_explicit = str(IdentifierComputer.compute(scorer_explicit_5))
    id_zero = str(IdentifierComputer.compute(scorer_zero))

    assert id_default == id_explicit, "Default and explicit utility_bias=5.0 must produce identical scorer hash"
    assert id_default != id_zero, "utility_bias=0.0 must produce a distinct scorer hash"

    # Verify execution sets wo.bias to 0.0
    inst = tasks_zero[0].instance()
    inst.execute()
    assert inst.scorer.hf_model.utility_predictor.wo.bias.item() == pytest.approx(0.0)


def test_modernbert_statelessness_no_side_channels():
    """Verify that ModernBertMiceForSequenceClassification does not store side-channel states on self."""
    config = ModernBertMiceConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        n_contextualization_layers=1,
        utility_pred=True,
        gate_target="h",
        use_hard_concrete=True,
        hard_pruning=True,
        _attn_implementation="sdpa",
    )
    model = ModernBertMiceForSequenceClassification(config)
    q = torch.randint(0, 50, (2, 4))
    qm = torch.ones((2, 4), dtype=torch.long)
    d = torch.randint(0, 50, (2, 8))
    dm = torch.ones((2, 8), dtype=torch.long)

    out = model(query_ids=q, query_mask=qm, doc_ids=d, doc_mask=dm)

    # All outputs must be cleanly returned in out
    assert out.logits is not None
    assert out.utilities is not None
    assert out.p_active is not None
    assert out.doc_hidden is not None
    assert out.pruned_utilities is not None

    # Model itself must NOT have side-channel attributes
    assert not hasattr(model, "last_utilities")
    assert not hasattr(model, "last_p_active")
    assert not hasattr(model, "last_doc_hidden")
    assert not hasattr(model, "pruned_utilities")
    assert not hasattr(model.utility_predictor, "last_p_active")


def test_ambient_trainer_context_lifecycle_and_warning(tmp_path, caplog):
    """Test TrainerContext.current() ambient context lifecycle and loud warning when missing in train mode."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    # 1. Outside context: TrainerContext.current() must be None
    assert TrainerContext.current() is None

    context = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=scorer,
        optimizer=None,
        fabric=None,
    )
    context.metrics = Metrics()
    reg_hook = UtilityRegularizer.C(lambda_reg=0.1)
    context.add_hook(reg_hook)

    # 2. Inside with context: TrainerContext.current() is active
    with context:
        assert TrainerContext.current() is context
    assert TrainerContext.current() is None

    # 3. Inside with context.losses(): TrainerContext.current() is active and hook executes
    input_records = PointwiseItems.from_texts(
        topics=["What is MICE?"],
        documents=["MICE is a contextual compression cross-encoder."],
    )

    with context.losses() as losses:
        assert TrainerContext.current() is context
        out = scorer(input_records)
        assert out.shape == (1,)
    assert len(losses) == 1
    assert losses[0].name == "utility_reg"
    assert TrainerContext.current() is None

    # 4. Loud warning when model is in train mode (scorer.train()) and called outside context
    scorer.train()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        out_train = scorer(input_records)
        assert out_train.shape == (1,)
    assert any(
        "Regularization hooks (UtilityRegularizer, GatedValueRegularizer) will NOT execute"
        in record.message
        for record in caplog.records
    )


def test_reentrant_ambient_context_and_learner_simulation(tmp_path):
    """Test that Context is fully re-entrant across epoch, step, and losses scopes without token reuse errors."""
    mock_optimizer = MagicMock()
    context = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=2,
        steps_per_epoch=3,
        trainer=None,
        model=torch.nn.Linear(10, 1),
        optimizer=mock_optimizer,
        fabric=None,
    )

    # 1. Multi-level nested re-entrancy
    assert TrainerContext.current() is None
    with context:
        assert TrainerContext.current() is context
        with context:
            assert TrainerContext.current() is context
            with context.losses():
                assert TrainerContext.current() is context
            assert TrainerContext.current() is context
        assert TrainerContext.current() is context
    assert TrainerContext.current() is None

    # 2. Learner training loop simulation: epoch scope -> step -> losses across multiple epochs/batches
    for epoch in range(2):
        with context:
            assert TrainerContext.current() is context
            for b in range(3):
                metrics = Metrics()
                with context.step(metrics):
                    assert TrainerContext.current() is context
                    for microbatch in range(2):
                        with context.losses() as losses:
                            assert TrainerContext.current() is context
                            assert isinstance(losses, list)
                    assert TrainerContext.current() is context
            assert TrainerContext.current() is context
        # Outside epoch (e.g. during validation / listeners):
        assert TrainerContext.current() is None


def test_activation_checkpointing_backward_guard(tmp_path):
    """Test that activation checkpointing recompute during backward does not duplicate losses/metrics or fail assertions."""
    from torch.utils.checkpoint import checkpoint
    from xpm_torch.losses import Loss

    context = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=1,
        steps_per_epoch=1,
        trainer=None,
        model=None,
        optimizer=MagicMock(),
        fabric=None,
    )
    context.metrics = Metrics()

    loss_call_count = 0

    class DummyCheckpointModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            nonlocal loss_call_count
            ctx = TrainerContext.current()
            if ctx is not None:
                loss_call_count += 1
                dummy_loss = (self.linear(x) * 0.0).sum()
                ctx.add_loss(Loss("dummy_reg", 0.1, dummy_loss))
            return self.linear(x)

    module = DummyCheckpointModule()
    x = torch.randn(2, 8, requires_grad=True)

    with context:  # epoch level
        with context.losses() as losses:
            out = checkpoint(module, x, use_reentrant=True)
            main_loss = out.sum()
            total_loss = main_loss + sum(l.weight * l.value for l in losses)

        assert loss_call_count == 1
        assert len(losses) == 1
        # Backward triggers checkpoint recompute of forward; guard must silently bypass add_loss:
        context.backward(total_loss)
        assert len(losses) == 1


def test_metrics_logger_histogram_and_null_logger(tmp_path):
    """Test MetricsLogger.log_histogram on TensorboardLogger and NullMetricsLogger, plus writer property."""
    from xpm_torch.loggers import TensorboardLogger, NullMetricsLogger
    import numpy as np

    null_logger = NullMetricsLogger()
    null_logger.log_histogram("test/tag", np.zeros(10), step=1)
    null_logger.log_scalar("test/scalar", 1.0, step=1)

    tb_logger = TensorboardLogger(tmp_path / "tb")
    tb_logger.log_histogram("test/hist", np.random.randn(20), step=1)
    tb_logger.log_scalar("test/val", 0.42, step=1)
    assert tb_logger.writer is not None
    tb_logger.close()

    context = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=1,
        steps_per_epoch=1,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    assert context.writer is not None
    assert context.metrics_logger is not None


def test_cross_attention_dtype_preservation():
    """Test that value-only gating does not upcast bfloat16 attention states to float32."""
    from MICE.modeling.mice_v2.modernbert_hf import ModernBertMiceCrossAttention

    config = ModernBertMiceConfig(
        hidden_size=128,
        num_attention_heads=4,
        gate_target="v",
        _attn_implementation="sdpa",
    )
    cross_attn = ModernBertMiceCrossAttention(config, layer_idx=0).to(torch.bfloat16)

    # Simulate mixed precision: representations in bfloat16, utilities in float32
    q_hidden = torch.randn(2, 16, 128, dtype=torch.bfloat16)
    d_hidden = torch.randn(2, 32, 128, dtype=torch.bfloat16)
    d_utilities = torch.rand(2, 32, 1, dtype=torch.float32)

    # Test 3D padded path
    out = cross_attn(q_hidden, d_hidden, d_utilities=d_utilities)
    assert out.dtype == torch.bfloat16, f"Expected output dtype bfloat16, got {out.dtype}"

    # Test varlen unpadded path simulation
    q_unpad = torch.randn(1, 16, 128, dtype=torch.bfloat16)
    d_unpad = torch.randn(1, 32, 128, dtype=torch.bfloat16)
    d_utils_unpad = torch.rand(1, 32, 1, dtype=torch.float32)
    cu_q = torch.tensor([0, 16], dtype=torch.int32)
    cu_d = torch.tensor([0, 32], dtype=torch.int32)

    out_varlen = cross_attn(
        q_unpad,
        d_unpad,
        d_utilities=d_utils_unpad,
        cu_seqlens_q=cu_q,
        cu_seqlens_d=cu_d,
        max_seqlen_q=16,
        max_seqlen_d=32,
    )
    assert out_varlen.dtype == torch.bfloat16, f"Expected varlen output dtype bfloat16, got {out_varlen.dtype}"
