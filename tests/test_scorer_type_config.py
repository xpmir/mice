"""Unit tests for scorer_type configuration and factory dispatch."""

import pytest
from configuration import CE_FineTuning, ScorerType
from cross_encoders_training.experiment import (
    build_STCrossEncoder,
    build_STMultiVectorEncoder,
    get_ce_name,
)
from experimaestro.experiments.grid import generate_grid
from xpmir.neural.sentence_transformers import STCrossEncoder, STMultiVectorEncoder


def test_scorer_type_enum():
    assert ScorerType.stCrossEncoder.value == "stCrossEncoder"
    assert ScorerType.stMultiVectorEncoder.value == "stMultiVectorEncoder"
    assert ScorerType.hfCrossScorer.value == "hfCrossScorer"


def test_ce_finetuning_default_scorer_type():
    cfg = CE_FineTuning()
    assert cfg.scorer_type == ScorerType.stCrossEncoder.value


def test_build_st_cross_encoder():
    cfg = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L6-v2", max_length=128)
    scorer, init_tasks = build_STCrossEncoder(cfg)
    assert isinstance(scorer, STCrossEncoder.XPMConfig)
    assert len(init_tasks) == 1


def test_build_st_multivector_encoder():
    cfg = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=128)
    scorer, init_tasks = build_STMultiVectorEncoder(cfg)
    assert isinstance(scorer, STMultiVectorEncoder.XPMConfig)
    assert len(init_tasks) == 0
    assert scorer.query_maxlen == 32
    assert scorer.doc_maxlen == 128


def test_get_ce_name_prefix():
    tags_ce = {"base": "bert-base", "loss": "bce", "scorer_type": "stCrossEncoder"}
    tags_mv = {"base": "bert-base", "loss": "bce", "scorer_type": "stMultiVectorEncoder"}
    tags_default = {"base": "bert-base", "loss": "bce"}

    assert get_ce_name(tags_ce).startswith("cross-encoder-")
    assert get_ce_name(tags_mv).startswith("colbert-")
    assert get_ce_name(tags_default).startswith("cross-encoder-")


def test_invalid_scorer_type_raises():
    with pytest.raises(ValueError):
        ScorerType("invalid_scorer")


def test_grid_search_scorer_type():
    from experimaestro.experiments.grid import GenericParams

    # Inline GenericParams
    cfg = CE_FineTuning(
        base="cross-encoder/ms-marco-MiniLM-L6-v2",
        scorer_type=GenericParams(values_list=["stCrossEncoder", "stMultiVectorEncoder"]),
    )
    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    types_in_configs = {c.scorer_type for c in configs}
    assert types_in_configs == {"stCrossEncoder", "stMultiVectorEncoder"}

    # Via grid_search dict
    cfg_dict = CE_FineTuning(
        base="cross-encoder/ms-marco-MiniLM-L6-v2",
        grid_search={"scorer_type": ["stCrossEncoder", "stMultiVectorEncoder"]},
    )
    configs_dict, tags_dict = generate_grid(cfg_dict)
    assert len(configs_dict) == 2
    types_in_configs_dict = {c.scorer_type for c in configs_dict}
    assert types_in_configs_dict == {"stCrossEncoder", "stMultiVectorEncoder"}


def test_ettin_colbert_optimizer_regex():
    import torch.nn as nn
    from xpm_torch.experiments.configuration import TransformerOptimization
    from xpm_torch.optim import ScheduledOptimizer

    class MockEttinMultiVector(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleDict({
                "tok_embeddings": nn.Embedding(10, 16),
                "norm": nn.LayerNorm(16, elementwise_affine=True, bias=False),
            })
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "attn_norm": nn.LayerNorm(16, elementwise_affine=True, bias=False),
                    "Wqkv": nn.Linear(16, 48, bias=False),
                    "mlp_norm": nn.LayerNorm(16, elementwise_affine=True, bias=False),
                    "Wi": nn.Linear(16, 64, bias=False),
                })
            ])
            self.final_norm = nn.LayerNorm(16, elementwise_affine=True, bias=False)
            self.linear = nn.Linear(16, 16, bias=False)

    model = MockEttinMultiVector()
    opt_config = TransformerOptimization(re_no_l2_regularization=[r"\.bias$", r"(?i)norm"])
    optimizers = opt_config.optimizer

    sched_opt = ScheduledOptimizer()
    sched_opt.initialize(
        param_optimizers=optimizers,
        num_training_steps=100,
        module=model,
        use_scaler=False,
    )
    assert len(sched_opt.optimizers) == 2


def test_scheduled_optimizer_ignored_params_logging(caplog):
    import logging
    import torch.nn as nn
    from xpm_torch.experiments.configuration import TransformerOptimization
    from xpm_torch.optim import ScheduledOptimizer, RegexParameterFilter, ParameterOptimizer

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.trained = nn.Linear(4, 4)
            self.ignored = nn.Linear(4, 4)

    model = MockModel()
    param_opts = [
        ParameterOptimizer.C(
            optimizer=TransformerOptimization().get_optimizer(True),
            filter=RegexParameterFilter.C(excludes=[r"ignored"]),
        )
    ]
    sched = ScheduledOptimizer()
    with caplog.at_level(logging.INFO):
        sched.initialize(param_opts, 10, model, False)

    # 1. Ignored parameters must be logged as WARNING
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("Ignored parameters (2) NOT being optimized" in r.message for r in warning_records)
    assert any("ignored.weight" in r.message for r in warning_records)

    # 2. Optimized group info log must NOT dump individual parameter names
    info_records = [r for r in caplog.records if r.levelname == "INFO"]
    opt_info = [r for r in info_records if "Optimizing with" in r.message]
    assert len(opt_info) > 0
    for r in opt_info:
        assert "parameters)" in r.message
        assert "trained.weight" not in r.message  # No parameter names dumped at INFO

    # 3. When no parameters are ignored, INFO logs 'Ignored parameters: none'
    caplog.clear()
    all_param_opts = [
        ParameterOptimizer.C(
            optimizer=TransformerOptimization().get_optimizer(True),
        )
    ]
    with caplog.at_level(logging.INFO):
        sched.initialize(all_param_opts, 10, model, False)

    assert any("Ignored parameters: none" in r.message for r in caplog.records)


def test_build_st_multivector_encoder_meanmaxsim_for_mse_loss():
    from configuration import Losses

    # Default loss -> maxsim
    cfg_default = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L-6-v2")
    scorer_default, _ = build_STMultiVectorEncoder(cfg_default)
    assert scorer_default.similarity_fn_name == "maxsim"

    # MSE_mixedbread_large -> meanmaxsim
    cfg_mse = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L-6-v2")
    cfg_mse.learner.loss = Losses.MSE_mixedbread_large.value
    scorer_mse, _ = build_STMultiVectorEncoder(cfg_mse)
    assert scorer_mse.similarity_fn_name == "meanmaxsim"

    # Explicit override takes precedence
    cfg_override = CE_FineTuning(
        base="cross-encoder/ms-marco-MiniLM-L-6-v2", similarity_fn_name="maxsim"
    )
    cfg_override.learner.loss = Losses.MSE_mixedbread_large.value
    scorer_override, _ = build_STMultiVectorEncoder(cfg_override)
    assert scorer_override.similarity_fn_name == "maxsim"


def test_build_st_multivector_encoder_meanmaxsim_for_lighton_loss():
    from configuration import Losses

    # MSE_lighton_multilingual uses the same teacher calibration as
    # MSE_mixedbread_large, so it defaults to meanmaxsim too
    cfg = CE_FineTuning(base="cross-encoder/ms-marco-MiniLM-L-6-v2")
    cfg.learner.loss = Losses.MSE_lighton_multilingual.value
    scorer, _ = build_STMultiVectorEncoder(cfg)
    assert scorer.similarity_fn_name == "meanmaxsim"

    # Explicit meanmaxsim/maxsim override works with the lighton loss
    for name in ("meanmaxsim", "maxsim"):
        cfg_override = CE_FineTuning(
            base="cross-encoder/ms-marco-MiniLM-L-6-v2", similarity_fn_name=name
        )
        cfg_override.learner.loss = Losses.MSE_lighton_multilingual.value
        scorer_override, _ = build_STMultiVectorEncoder(cfg_override)
        assert scorer_override.similarity_fn_name == name


def test_build_trainer_pointwise_mse_scaling(monkeypatch):
    from configuration import Losses
    import training_utils
    from training_utils import build_trainer
    from xpmir.letor.distillation.pointwise import PointwiseDistillationTrainer
    from datamaestro_ir.data.huggingface import HuggingFacePointwiseDistillationSamples

    dummy_ds = HuggingFacePointwiseDistillationSamples.C(
        repo_id="dummy", split="train", score_field="label"
    )
    monkeypatch.setattr(training_utils, "build_ettin_reranker_v1_data", lambda *args, **kwargs: dummy_ds)

    # For stMultiVectorEncoder, teacher_scale and teacher_bias should be set
    cfg_mv = CE_FineTuning(scorer_type=ScorerType.stMultiVectorEncoder.value)
    cfg_mv.learner.loss = Losses.MSE_mixedbread_large.value
    trainer_mv = build_trainer(cfg_mv)
    assert isinstance(trainer_mv, PointwiseDistillationTrainer.XPMConfig)
    assert trainer_mv.lossfn.teacher_scale == 0.02
    assert trainer_mv.lossfn.teacher_bias == 0.70

    # For stCrossEncoder, teacher_scale and teacher_bias should be None
    cfg_ce = CE_FineTuning(scorer_type=ScorerType.stCrossEncoder.value)
    cfg_ce.learner.loss = Losses.MSE_mixedbread_large.value
    trainer_ce = build_trainer(cfg_ce)
    assert isinstance(trainer_ce, PointwiseDistillationTrainer.XPMConfig)
    assert trainer_ce.lossfn.teacher_scale is None
    assert trainer_ce.lossfn.teacher_bias is None


def test_pointwise_mse_loss_teacher_scaling():
    import torch
    from xpmir.letor.distillation.pointwise import PointwiseMSELoss

    # Without scaling
    loss_unscaled = PointwiseMSELoss.C().instance()
    loss_unscaled.initialize(None)
    s = torch.tensor([0.7, 0.9])
    y = torch.tensor([0.0, 10.0])
    # plain MSE between [0.7, 0.9] and [0.0, 10.0]
    expected_unscaled = ((0.7 - 0.0) ** 2 + (0.9 - 10.0) ** 2) / 2
    assert torch.isclose(loss_unscaled.compute(s, y, None), torch.tensor(expected_unscaled))

    # With scaling: y_scaled = 0.02 * y + 0.70
    loss_scaled = PointwiseMSELoss.C(teacher_scale=0.02, teacher_bias=0.70).instance()
    loss_scaled.initialize(None)
    # y=0.0 -> 0.70; y=10.0 -> 0.90
    # student: [0.7, 0.9] -> diffs: [0.0, 0.0] -> loss: 0.0
    assert torch.isclose(loss_scaled.compute(s, y, None), torch.tensor(0.0), atol=1e-5)
