import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import pytest

from experimaestro.core.identifier import IdentifierComputer
from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    ModernBertMiceScorer,
    compute_model_norm_metrics,
    mice_v2_scorer,
)
from sentence_transformers.models import Dense, LayerNorm


def _small_config(**kwargs):
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        n_contextualization_layers=1,
        n_interaction_layers=1,
    )
    defaults.update(kwargs)
    return ModernBertMiceConfig(**defaults)


def test_linear_head_output_modules():
    """Test default linear head structure and forward pass."""
    config = _small_config(head_type="linear", top_final_norm=False)
    model = ModernBertMiceForSequenceClassification(config)

    assert hasattr(model, "output_modules")
    assert isinstance(model.output_modules, nn.Sequential)
    assert len(model.output_modules) == 1
    assert "dense_score" in model.output_modules._modules
    assert "pooling" not in model.output_modules._modules
    assert isinstance(model.output_modules.dense_score, Dense)
    assert model.classifier is model.output_modules.dense_score.linear
    assert model.top_final_norm is None

    # Forward padded
    q_ids = torch.randint(0, 100, (2, 8))
    d_ids = torch.randint(0, 100, (2, 16))
    out = model(query_ids=q_ids, doc_ids=d_ids)
    assert out.logits.shape == (2, 1)


def test_linear_head_with_top_final_norm():
    """Test linear head with top_final_norm enabled."""
    config = _small_config(head_type="linear", top_final_norm=True)
    model = ModernBertMiceForSequenceClassification(config)

    assert len(model.output_modules) == 2
    assert "norm" in model.output_modules._modules
    assert "dense_score" in model.output_modules._modules
    assert isinstance(model.output_modules.norm, LayerNorm)
    assert model.top_final_norm is model.output_modules.norm.norm


def test_st_reranker_head_output_modules():
    """Test st_reranker head structure, module specs, and forward pass."""
    config = _small_config(head_type="st_reranker")
    model = ModernBertMiceForSequenceClassification(config)

    assert hasattr(model, "output_modules")
    assert isinstance(model.output_modules, nn.Sequential)
    assert len(model.output_modules) == 3

    names = list(model.output_modules._modules.keys())
    assert names == ["dense_inner", "norm", "dense_score"]

    dense_inner = model.output_modules.dense_inner
    norm = model.output_modules.norm
    dense_score = model.output_modules.dense_score

    assert isinstance(dense_inner, Dense)
    assert dense_inner.linear.in_features == 64
    assert dense_inner.linear.out_features == 64
    assert dense_inner.linear.bias is None
    assert isinstance(dense_inner.activation_function, nn.GELU)

    assert isinstance(norm, LayerNorm)
    assert norm.dimension == 64

    assert isinstance(dense_score, Dense)
    assert dense_score.linear.in_features == 64
    assert dense_score.linear.out_features == 1
    assert dense_score.linear.bias is not None
    assert isinstance(dense_score.activation_function, nn.Identity)

    # Forward padded
    q_ids = torch.randint(0, 100, (3, 10))
    d_ids = torch.randint(0, 100, (3, 20))
    out = model(query_ids=q_ids, doc_ids=d_ids)
    assert out.logits.shape == (3, 1)


def test_output_modules_direct_forward():
    """Test output_modules directly with sentence_embedding input."""
    config = _small_config(head_type="st_reranker")
    model = ModernBertMiceForSequenceClassification(config)

    cls_hidden = torch.randn(4, 64)
    features = {"sentence_embedding": cls_hidden}
    out = model.output_modules(features)
    assert "scores" in out
    assert out["scores"].shape == (4, 1)


def test_state_dict_backward_compatibility():
    """Test loading legacy state_dict containing classifier.* and top_final_norm.*."""
    config = _small_config(head_type="linear", top_final_norm=True)
    model = ModernBertMiceForSequenceClassification(config)

    # Construct synthetic legacy state dict
    legacy_sd = model.state_dict()
    legacy_sd["classifier.weight"] = legacy_sd.pop("output_modules.dense_score.linear.weight")
    legacy_sd["classifier.bias"] = legacy_sd.pop("output_modules.dense_score.linear.bias")
    legacy_sd["top_final_norm.weight"] = legacy_sd.pop("output_modules.norm.norm.weight")
    if "output_modules.norm.norm.bias" in legacy_sd:
        legacy_sd["top_final_norm.bias"] = legacy_sd.pop("output_modules.norm.norm.bias")

    # Load into fresh model
    new_model = ModernBertMiceForSequenceClassification(config)
    missing, unexpected = new_model.load_state_dict(legacy_sd, strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0


def test_experimaestro_hash_invariance():
    """Verify experimaestro task ID is identical when head_type is omitted vs explicitly 'linear'."""
    scorer_default, tasks_default = mice_v2_scorer(hf_id="jhu-clsp/ettin-encoder-17m")
    scorer_linear, tasks_linear = mice_v2_scorer(hf_id="jhu-clsp/ettin-encoder-17m", head_type="linear")
    scorer_st, tasks_st = mice_v2_scorer(hf_id="jhu-clsp/ettin-encoder-17m", head_type="st_reranker")

    # Check scorer identifier
    id_scorer_default = str(IdentifierComputer.compute(scorer_default))
    id_scorer_linear = str(IdentifierComputer.compute(scorer_linear))
    id_scorer_st = str(IdentifierComputer.compute(scorer_st))

    assert id_scorer_default == id_scorer_linear, "Default and linear head must produce identical scorer hash"
    assert id_scorer_default != id_scorer_st, "st_reranker head must produce a distinct scorer hash"

    # Check init task identifier
    id_task_default = str(IdentifierComputer.compute(tasks_default[0]))
    id_task_linear = str(IdentifierComputer.compute(tasks_linear[0]))
    id_task_st = str(IdentifierComputer.compute(tasks_st[0]))

    assert id_task_default == id_task_linear, "Default and linear head must produce identical task hash"
    assert id_task_default != id_task_st, "st_reranker head must produce a distinct task hash"


def test_classifier_norm_removed_from_metrics():
    """Verify classifier_norm is completely absent from compute_model_norm_metrics."""
    config = _small_config(head_type="linear")
    model = ModernBertMiceForSequenceClassification(config)
    assert not hasattr(model, "get_classifier_norm")

    metrics = compute_model_norm_metrics(model)
    assert "classifier_norm" not in metrics


def test_init_scorer_execution_both_head_types():
    """Verify InitModernBertMiceScorer executes cleanly for both linear and st_reranker heads."""
    for head in ["linear", "st_reranker"]:
        scorer_cfg, init_tasks = mice_v2_scorer(
            hf_id="jhu-clsp/ettin-encoder-17m",
            n_contextualization_layers=2,
            n_interaction_layers=2,
            head_type=head,
        )
        for task in init_tasks:
            inst_task = task.instance()
            inst_task.execute()
            scorer = inst_task.scorer

            assert scorer.hf_model.config.head_type == head
            # Verify forward pass works
            query_ids = torch.randint(0, 1000, (2, 8))
            query_mask = torch.ones((2, 8), dtype=torch.bool)
            doc_ids = torch.randint(0, 1000, (2, 16))
            doc_mask = torch.ones((2, 16), dtype=torch.bool)

            with torch.no_grad():
                out = scorer.hf_model(
                    query_ids=query_ids,
                    query_mask=query_mask,
                    doc_ids=doc_ids,
                    doc_mask=doc_mask,
                )
            assert out.logits.shape == (2, 1)


def test_unpadded_forward_simulation():
    """Verify that unpadded top-layer output with token count >> batch size passes cleanly through output_modules."""
    config = _small_config(head_type="st_reranker")
    model = ModernBertMiceForSequenceClassification(config)

    # 4 sequences, total 150 tokens. Sliced CLS output has shape [1, 4, 64]
    batch_size = 4
    dim = config.hidden_size
    mock_unpadded_cls = torch.randn(1, batch_size, dim)

    # In use_unpadded mode, cls_hidden = mock_unpadded_cls.squeeze(0) -> [4, 64]
    cls_hidden = mock_unpadded_cls.squeeze(0)
    features = {"sentence_embedding": cls_hidden}
    out = model.output_modules(features)
    assert out["scores"].shape == (batch_size, 1)
