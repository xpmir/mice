"""
Tests for ModernBERT MICE V2 document compress-decompress bottleneck.
Run with: `uv run pytest tests/test_mice_v2_doc_bottleneck.py -v`
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from datamaestro_ir.data.base import IDTextRecord
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder

from MICE.modeling.mice import MICETokenizedTexts, QueryDocInput
from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    mice_v2_scorer,
    ModernBertMiceScorer,
    InitMICEV2FromCheckpoint,
)
from xpmir.letor.records import PointwiseItems
from xpm_torch.module import SimpleModuleLoader


def _small_config(**kwargs):
    defaults = dict(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        normalize_doc_embeds=True,
    )
    defaults.update(kwargs)
    return ModernBertMiceConfig(**defaults)


def test_doc_bottleneck_default_is_noop():
    """When doc_bottleneck_dim is None, doc_compress and doc_decompress are None."""
    config = _small_config(doc_bottleneck_dim=None)
    model = ModernBertMiceForSequenceClassification(config)
    assert model.doc_compress is None
    assert model.doc_decompress is None


def test_doc_bottleneck_shapes():
    """Verify shapes and bias=False for doc_compress and doc_decompress."""
    config = _small_config(hidden_size=32, doc_bottleneck_dim=16)
    model = ModernBertMiceForSequenceClassification(config)

    assert model.doc_compress is not None
    assert model.doc_decompress is not None
    assert model.doc_compress.weight.shape == (16, 32)
    assert model.doc_decompress.weight.shape == (32, 16)
    assert model.doc_compress.bias is None
    assert model.doc_decompress.bias is None


def test_doc_bottleneck_warning_when_not_normalized(caplog):
    """Loud warning when doc_bottleneck_dim is set but normalize_doc_embeds is False."""
    import logging

    with caplog.at_level(logging.WARNING):
        _small_config(doc_bottleneck_dim=16, normalize_doc_embeds=False)

    assert any("⚠️ [MICE V2 CONFIG WARNING]" in record.message for record in caplog.records)


def test_doc_bottleneck_document_encoder():
    """Test MiceV2DocumentEncoder dimension property and embedding shape with bottleneck."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        doc_bottleneck_dim=64,
        normalize_doc_embeds=True,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    assert doc_encoder.dimension == 64, f"Expected dimension 64, got {doc_encoder.dimension}"

    docs = [
        IDTextRecord(id="doc1", text="Paris is the capital of France."),
        IDTextRecord(id="doc2", text="Berlin is the capital of Germany."),
    ]

    with torch.no_grad():
        token_embs = doc_encoder.document_token_embeddings(docs)

    assert len(token_embs) == 2
    assert token_embs[0].ndim == 2
    assert token_embs[0].shape[-1] == 64, f"Expected hidden dim 64, got {token_embs[0].shape[-1]}"
    assert token_embs[1].shape[-1] == 64


def test_doc_bottleneck_forward_parity():
    """Verify bit-exact / close score parity between end-to-end forward and pre-encoded doc_hidden_states forward."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        doc_bottleneck_dim=64,
        normalize_doc_embeds=True,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    queries = ["What is the capital of France?", "Where is Berlin?"]
    documents = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
    ]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    # 1. Standard forward pass (computes bottom layers -> compress -> decompress -> top layers)
    with torch.no_grad():
        standard_scores = scorer(input_records)

    # 2. Forward pass with pre-encoded doc_hidden_states (cached 64-dim vectors -> decompress -> top layers)
    doc_records = [IDTextRecord(id=f"doc_{i}", text=text) for i, text in enumerate(documents)]
    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    with torch.no_grad():
        doc_hidden_states = doc_encoder.document_token_embeddings(doc_records)
        tokenized_q = scorer.batch_tokenize(input_records, input_nature=QueryDocInput.QUERY)
        plaid_scores = scorer(
            tokenized=MICETokenizedTexts(tokenized_q=tokenized_q, tokenized_docs=None),
            doc_hidden_states=doc_hidden_states,
        )

    assert torch.allclose(standard_scores, plaid_scores, atol=1e-4), (
        f"Score mismatch between standard forward ({standard_scores}) and PLAID bottleneck forward ({plaid_scores})"
    )


def test_doc_bottleneck_combined_with_compress_dim():
    """Test compatibility when both compress_dim=2.0 and doc_bottleneck_dim=16 are active."""
    config = _small_config(hidden_size=32, compress_dim=2.0, doc_bottleneck_dim=16)
    model = ModernBertMiceForSequenceClassification(config)

    assert model.adapter is not None
    assert model.adapter.weight.shape == (16, 32)
    assert model.doc_compress.weight.shape == (16, 32)
    assert model.doc_decompress.weight.shape == (16, 16)  # top hidden size is 32/2 = 16

    query_ids = torch.randint(0, 100, (2, 6))
    query_mask = torch.ones((2, 6), dtype=torch.long)
    doc_ids = torch.randint(0, 100, (2, 10))
    doc_mask = torch.ones((2, 10), dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(query_ids=query_ids, query_mask=query_mask, doc_ids=doc_ids, doc_mask=doc_mask)

    assert out.logits.shape == (2, 1)
    assert not torch.isnan(out.logits).any()


def test_doc_bottleneck_backward_gradients():
    """Verify non-zero gradients flow through doc_compress and doc_decompress during training."""
    config = _small_config(hidden_size=32, doc_bottleneck_dim=16)
    model = ModernBertMiceForSequenceClassification(config)
    model.train()

    query_ids = torch.randint(0, 100, (2, 6))
    query_mask = torch.ones((2, 6), dtype=torch.long)
    doc_ids = torch.randint(0, 100, (2, 10))
    doc_mask = torch.ones((2, 10), dtype=torch.long)
    labels = torch.tensor([[1.0], [0.0]])

    out = model(
        query_ids=query_ids,
        query_mask=query_mask,
        doc_ids=doc_ids,
        doc_mask=doc_mask,
        labels=labels,
    )
    loss = out.loss
    loss.backward()

    assert model.doc_compress.weight.grad is not None
    assert model.doc_compress.weight.grad.abs().sum() > 0, "doc_compress must receive gradients"
    assert model.doc_decompress.weight.grad is not None
    assert model.doc_decompress.weight.grad.abs().sum() > 0, "doc_decompress must receive gradients"


def test_doc_bottleneck_persistence(tmp_path):
    """Verify saving to disk and reloading with AutoModel and CrossEncoder preserves bottleneck."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        doc_bottleneck_dim=32,
        normalize_doc_embeds=True,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    save_dir = tmp_path / "saved_mice_bot"
    scorer.save_model(save_dir)

    # 1. Reload via AutoModelForSequenceClassification
    reloaded_hf = AutoModelForSequenceClassification.from_pretrained(save_dir, trust_remote_code=True)
    assert reloaded_hf.config.doc_bottleneck_dim == 32
    assert reloaded_hf.doc_compress is not None
    assert reloaded_hf.doc_compress.weight.shape[0] == 32
    assert reloaded_hf.doc_decompress.weight.shape[1] == 32

    # 2. Reload via CrossEncoder
    reloaded_ce = CrossEncoder(str(save_dir), trust_remote_code=True)
    scores = reloaded_ce.predict([("What is Paris?", "Paris is the capital of France.")])
    assert len(scores) == 1
    assert not torch.isnan(torch.tensor(scores)).any()


def test_init_from_checkpoint_retrain_bottleneck(tmp_path):
    """Test loading a pretrained MICE v2 model without bottleneck and training the compression bottleneck on top."""
    # 1. Train/save a base MICE model without bottleneck
    base_scorer_cfg, base_init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        doc_bottleneck_dim=None,  # Base model has no bottleneck
        normalize_doc_embeds=True,
    )
    base_scorer = base_scorer_cfg.instance()
    for task in base_init_tasks:
        task.instance().execute()
    base_scorer.initialize()

    base_save_dir = tmp_path / "base_mice_no_bot"
    base_scorer.save_model(base_save_dir)
    assert (base_save_dir / "model.safetensors").exists()

    # 2. Instantiate a target model with doc_bottleneck_dim=32 and load base weights via InitMICEV2FromCheckpoint
    target_scorer_cfg, target_init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        doc_bottleneck_dim=32,  # Introduce 32-dim bottleneck
        normalize_doc_embeds=True,
    )
    for task in target_init_tasks:
        task.instance().execute()

    # Configure InitMICEV2FromCheckpoint with freeze_bottom_layers=True and freeze_top_layers=True
    init_task = InitMICEV2FromCheckpoint.C(
        scorer=target_scorer_cfg,
        checkpoint_loader=SimpleModuleLoader.C(value=target_scorer_cfg, path=base_save_dir),
        freeze_bottom_layers=True,
        freeze_top_layers=True,
    ).instance()
    init_task.execute()

    target_scorer = init_task.scorer

    hf_model = target_scorer.hf_model
    assert hf_model.doc_compress is not None
    assert hf_model.doc_decompress is not None
    assert hf_model.doc_compress.weight.shape[0] == 32
    assert hf_model.doc_decompress.weight.shape[1] == 32

    # Verify freeze states: bottom and top layers frozen, bottleneck trainable
    for p in hf_model.model.parameters():
        assert not p.requires_grad, "Bottom layers must be frozen"
    for p in hf_model.top_layers.parameters():
        assert not p.requires_grad, "Top layers must be frozen"
    for p in hf_model.output_modules.parameters():
        assert not p.requires_grad, "Head must be frozen"

    assert hf_model.doc_compress.weight.requires_grad, "doc_compress must be trainable"
    assert hf_model.doc_decompress.weight.requires_grad, "doc_decompress must be trainable"

    # 3. Verify forward and backward pass works and updates ONLY bottleneck
    target_scorer.train()
    queries = ["What is the capital of France?"]
    documents = ["Paris is the capital of France."]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    scores = target_scorer(input_records)
    loss = scores.sum()
    loss.backward()

    assert hf_model.doc_compress.weight.grad is not None
    assert hf_model.doc_compress.weight.grad.abs().sum() > 0
    assert hf_model.doc_decompress.weight.grad is not None
    assert hf_model.doc_decompress.weight.grad.abs().sum() > 0
