"""
Tests for MICE V2 Phase 2 Hard Token Pruning & Layer Freezing.
Run with: `uv run pytest tests/test_mice_v2_hard_pruning.py -v`
"""

import os
import tempfile
from pathlib import Path
import pytest
import torch


from MICE.modeling.mice_v2.modernbert_hf import (
    UtilityPredictor,
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
    _hard_prune_padded_documents,
    _hard_prune_unpadded_documents,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    mice_v2_scorer,
    ModernBertMiceScorer,
)
from xpmir.letor.records import PointwiseItems
from xpm_torch.trainers import TrainerContext


def test_hard_prune_padded_documents():
    """Test 3D padded document token dropping logic and RoPE position ID retention."""
    batch_size, seq_len, d_model = 2, 6, 16
    d_hidden = torch.randn(batch_size, seq_len, d_model)
    doc_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Utilities: Doc 0 has tokens [0.9, 0.2, 0.8, 0.1, 0.7, 0.4] -> keeps indices [0, 2, 4]
    # Doc 1 has tokens [0.1, 0.2, 0.1, 0.3, 0.1, 0.2] -> all below 0.5, keeps max utility index 3 (value 0.3)
    utilities = torch.tensor(
        [
            [[0.9], [0.2], [0.8], [0.1], [0.7], [0.4]],
            [[0.1], [0.2], [0.1], [0.3], [0.1], [0.2]],
        ]
    )

    new_d_hidden, new_doc_mask, new_pos_ids = _hard_prune_padded_documents(
        d_hidden, doc_mask, utilities, threshold=0.5
    )

    # Max kept tokens is 3 (doc 0 keeps 3 tokens)
    assert new_d_hidden.shape == (2, 3, d_model)
    assert new_doc_mask.shape == (2, 3)
    assert new_pos_ids.shape == (2, 3)

    # Doc 0 position IDs should match original kept positions [0, 2, 4]
    assert torch.equal(new_pos_ids[0], torch.tensor([0, 2, 4]))
    # Doc 1 position IDs should match best token index [3, 0, 0] with mask [1, 0, 0]
    assert new_pos_ids[1, 0].item() == 3
    assert torch.equal(new_doc_mask[1], torch.tensor([1, 0, 0]))


def test_hard_prune_unpadded_documents():
    """Test 1D unpadded FA2 document token dropping logic and RoPE position ID retention."""
    d_model = 16
    # Batch size 2: Doc 0 has 4 tokens, Doc 1 has 4 tokens -> total 8 tokens
    d_hidden = torch.randn(1, 8, d_model)
    d_pos_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    cu_seqlens_d = torch.tensor([0, 4, 8], dtype=torch.int32)

    # Doc 0: [0.8, 0.2, 0.9, 0.1] -> keeps [0, 2] (pos_ids: [0, 2])
    # Doc 1: [0.1, 0.7, 0.3, 0.6] -> keeps [1, 3] (pos_ids: [1, 3])
    utilities = torch.tensor([[[0.8], [0.2], [0.9], [0.1], [0.1], [0.7], [0.3], [0.6]]])

    pruned_d_hidden, pruned_d_pos_ids, new_cu_seqlens, new_max_len = (
        _hard_prune_unpadded_documents(
            d_hidden, d_pos_ids, cu_seqlens_d, utilities, threshold=0.5
        )
    )

    # Total kept tokens = 2 (doc 0) + 2 (doc 1) = 4
    assert pruned_d_hidden.shape == (1, 4, d_model)
    assert torch.equal(pruned_d_pos_ids, torch.tensor([[0, 2, 1, 3]]))
    assert torch.equal(new_cu_seqlens, torch.tensor([0, 2, 4], dtype=torch.int32))
    assert new_max_len == 2


def test_modernbert_hf_hard_pruning_and_freezing():
    """Test ModernBertMiceForSequenceClassification hard pruning and layer freezing."""
    config = ModernBertMiceConfig(
        n_contextualization_layers=2,
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        utility_pred=True,
        hard_pruning=True,
        hard_prune_threshold=0.5,
    )
    model = ModernBertMiceForSequenceClassification(config)

    # 1. Verify no automatic freezing upon initialization
    for param in model.parameters():
        assert param.requires_grad, "Model parameters must remain trainable upon initialization"

    # 2. Verify freeze_bottom_layers()
    model.freeze_bottom_layers()
    for param in model.model.embeddings.parameters():
        assert not param.requires_grad
    for param in model.model.layers.parameters():
        assert not param.requires_grad
    for param in model.top_layers.parameters():
        assert param.requires_grad

    # 3. Verify freeze_utility
    model.freeze_utility()
    for param in model.utility_predictor.parameters():
        assert not param.requires_grad

    # Verify top layers remain trainable
    for param in model.top_layers.parameters():
        assert param.requires_grad
    for param in model.classifier.parameters():
        assert param.requires_grad

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
        doc_encoded, doc_encoded_mask = model.encode_documents(doc_ids, doc_mask)

    assert out.logits.shape == (2, 1)
    assert doc_encoded.ndim == 3
    assert doc_encoded_mask.shape[:2] == doc_encoded.shape[:2]


def test_modernbert_xpm_hard_pruning():
    """Test ModernBertMiceScorer with hard_pruning=True and freezing."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
        hard_pruning=True,
        hard_prune_threshold=0.5,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        scorer = inst_task.scorer

    # Verify no automatic freezing upon initialization in XPM scorer
    for param in scorer.hf_model.parameters():
        assert param.requires_grad, "Scorer parameters must remain trainable upon initialization"

    # Test freeze utilitu
    scorer.freeze_utility()
    for param in scorer.hf_model.utility_predictor.parameters():
        assert not param.requires_grad
    for param in scorer.hf_model.model.embeddings.parameters():
        assert param.requires_grad
    for param in scorer.hf_model.model.layers.parameters():
        assert param.requires_grad

    # Test freeze with freeze_bottom_layers=True
    scorer.freeze_bottom_layers()

    # Verify frozen layers
    for param in scorer.hf_model.model.embeddings.parameters():
        assert not param.requires_grad
    for param in scorer.hf_model.model.layers.parameters():
        assert not param.requires_grad
    for param in scorer.hf_model.utility_predictor.parameters():
        assert not param.requires_grad

    # Verify trainable top layers
    for param in scorer.hf_model.top_layers.parameters():
        assert param.requires_grad

    scorer.eval()
    queries = ["What is IR?", "How does hard pruning work?"]
    documents = [
        "Information Retrieval obtains relevant document tokens.",
        "Hard token pruning drops tokens with predicted utility below half.",
    ]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    with torch.no_grad():
        output = scorer(input_records)
        assert output.shape == (2,)


def test_hard_pruning_save_load():
    """Test saving and reloading a hard-pruned MICE v2 model."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        utility_pred=True,
        hard_pruning=True,
        hard_prune_threshold=0.5,
    )
    for task in init_tasks:
        inst_task = task.instance()
        inst_task.execute()
        model = inst_task.scorer

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "hard_pruned_model"
        model.save_model(save_path)

        loader_config = scorer_cfg.loader_config(save_path)
        loader = loader_config.instance()
        loader.execute()
        reloaded_model = loader.model

        assert reloaded_model.hf_model.config.hard_pruning is True
        assert reloaded_model.hf_model.config.hard_prune_threshold == 0.5


def test_init_micev2_from_checkpoint_freezing(monkeypatch):
    """Test InitMICEV2FromCheckpoint parameter freezing with freeze_bottom_layers=True and False."""
    import json
    from MICE.modeling.mice_v2.modernbert_xpm import InitMICEV2FromCheckpoint, mice_v2_scorer
    from xpm_torch.module import SimpleModuleLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        ckpt_dir = Path(tmpdir) / "checkpoints" / "epoch-00000001"
        ckpt_dir.mkdir(parents=True)
        with (ckpt_dir / "info.json").open("wt") as fp:
            json.dump({"epoch": 1, "steps": 100}, fp)
        (ckpt_dir / "trainer.pth").touch()
        (ckpt_dir / "optimizer.pth").touch()

        # 1. Test freeze_bottom_layers=True (default)
        p2_scorer_cfg1, p2_init_tasks1 = mice_v2_scorer(
            hf_id="jhu-clsp/ettin-encoder-17m",
            n_contextualization_layers=2,
            utility_pred=True,
            hard_pruning=True,
        )
        for task in p2_init_tasks1:
            task.instance().execute()

        init_task_true = InitMICEV2FromCheckpoint.C(
            scorer=p2_scorer_cfg1,
            freeze_bottom_layers=True,
            checkpoint_loader=SimpleModuleLoader.C(value=p2_scorer_cfg1, path=ckpt_dir.parent),
        ).instance()
        init_task_true.execute()

        scorer1 = init_task_true.scorer
        for param in scorer1.hf_model.utility_predictor.parameters():
            assert not param.requires_grad
        for param in scorer1.hf_model.model.embeddings.parameters():
            assert not param.requires_grad
        for param in scorer1.hf_model.model.layers.parameters():
            assert not param.requires_grad
        for param in scorer1.hf_model.top_layers.parameters():
            assert param.requires_grad

        # 2. Test freeze_bottom_layers=False (utility frozen, bottom trainable)
        p2_scorer_cfg2, p2_init_tasks2 = mice_v2_scorer(
            hf_id="jhu-clsp/ettin-encoder-17m",
            n_contextualization_layers=2,
            utility_pred=True,
            hard_pruning=True,
        )
        for task in p2_init_tasks2:
            task.instance().execute()

        init_task_false = InitMICEV2FromCheckpoint.C(
            scorer=p2_scorer_cfg2,
            freeze_bottom_layers=False,
            checkpoint_loader=SimpleModuleLoader.C(value=p2_scorer_cfg2, path=ckpt_dir.parent),
        ).instance()
        init_task_false.execute()

        scorer2 = init_task_false.scorer
        for param in scorer2.hf_model.utility_predictor.parameters():
            assert not param.requires_grad
        for param in scorer2.hf_model.model.embeddings.parameters():
            assert param.requires_grad
        for param in scorer2.hf_model.model.layers.parameters():
            assert param.requires_grad
        for param in scorer2.hf_model.top_layers.parameters():
            assert param.requires_grad


def test_init_micev2_rank_zero_guard():
    """Test that is_rank_zero correctly freezes parameters across ranks while skipping file copies on non-zero ranks."""
    from MICE.modeling.mice_v2.modernbert_xpm import is_rank_zero, InitMICEV2FromCheckpoint, mice_v2_scorer
    from xpm_torch.module import SimpleModuleLoader

    # Standard environment rank 0
    assert is_rank_zero() is True

    # Simulate non-zero rank
    os.environ["RANK"] = "1"
    try:
        assert is_rank_zero() is False
        p2_scorer_cfg, init_tasks = mice_v2_scorer(
            hf_id="jhu-clsp/ettin-encoder-17m",
            n_contextualization_layers=2,
            utility_pred=True,
        )
        for task in init_tasks:
            task.instance().execute()

        init_task = InitMICEV2FromCheckpoint.C(
            scorer=p2_scorer_cfg,
            freeze_bottom_layers=True,
            checkpoint_loader=SimpleModuleLoader.C(value=p2_scorer_cfg, path=Path(".")),
        ).instance()
        # Should execute freezing on rank 1 and return early without attempting checkpoint copy
        init_task.execute()

        # Parameter freezing must be applied even on non-zero ranks
        for param in init_task.scorer.hf_model.utility_predictor.parameters():
            assert not param.requires_grad
        for param in init_task.scorer.hf_model.model.embeddings.parameters():
            assert not param.requires_grad
        for param in init_task.scorer.hf_model.top_layers.parameters():
            assert param.requires_grad
    finally:
        os.environ.pop("RANK", None)
