"""
Unit and integration tests for Evaluation Sparsity Hook & Results Reporting.
Run with: `uv run pytest tests/test_mice_v2_eval_sparsity.py -v`
"""

import json
import logging
from pathlib import Path
import pytest
import torch

from MICE.modeling.mice_v2.modernbert_xpm import (
    EvalSparsityAccumulator,
    InitEvalSparsityHook,
    ModernBertMiceScorer,
)
from training_utils import sync_eval_sparsity_metrics

logging.basicConfig(level=logging.INFO)


def test_eval_sparsity_accumulator_basic(tmp_path: Path):
    """Test EvalSparsityAccumulator token tracking, metric computation, and file flushing."""
    accumulator = EvalSparsityAccumulator(threshold=0.5)

    # 1. First batch: 4 tokens [0.8, 0.2, 0.6, 0.4] -> 2 pruned (<0.5), sum = 2.0
    utilities_1 = torch.tensor([[[0.8], [0.2], [0.6], [0.4]]])  # shape [1, 4, 1]
    doc_mask_1 = torch.tensor([[1, 1, 1, 1]])
    accumulator.update(utilities_1, doc_mask=doc_mask_1)

    assert accumulator.total_tokens == 4
    assert accumulator.pruned_tokens == 2
    assert pytest.approx(accumulator.sum_utilities, 1e-4) == 2.0

    # 2. Second batch with padding: 3 valid tokens [0.9, 0.1, 0.7] -> 1 pruned (<0.5), sum = 1.7
    utilities_2 = torch.tensor([[[0.9], [0.1], [0.7], [0.0]]])  # shape [1, 4, 1]
    doc_mask_2 = torch.tensor([[1, 1, 1, 0]])
    accumulator.update(utilities_2, doc_mask=doc_mask_2)

    assert accumulator.total_tokens == 7
    assert accumulator.pruned_tokens == 3
    assert pytest.approx(accumulator.sum_utilities, 1e-4) == 3.7

    # 3. Compute metrics
    stats, gather_succeeded = accumulator.compute()
    assert pytest.approx(stats["sparsity"], 1e-4) == 3 / 7
    assert pytest.approx(stats["kept_ratio"], 1e-4) == 4 / 7
    assert pytest.approx(stats["mean_utility"], 1e-4) == 3.7 / 7
    assert stats["total_tokens"] == 7
    assert gather_succeeded is True

    # 4. Flush to file (single-rank / rank 0 writes eval_sparsity.json)
    out_file = tmp_path / "eval_sparsity.json"
    accumulator.flush_to_file(output_filename=str(out_file.name))

    stats_computed, _ = accumulator.compute()
    with out_file.open("w") as fp:
        json.dump(stats_computed, fp)

    assert out_file.exists()
    loaded_stats = json.loads(out_file.read_text())
    assert loaded_stats["total_tokens"] == 7


def test_eval_sparsity_accumulator_multi_rank_fallback(tmp_path: Path, monkeypatch):
    """Test that rank 0 always writes eval_sparsity.json and rank!=0 writes eval_sparsity_{rank}.json when gather fails."""
    monkeypatch.chdir(tmp_path)

    # Mock Fabric with world_size=2
    class MockFabric:
        def __init__(self, rank: int, world_size: int = 2):
            self.global_rank = rank
            self.world_size = world_size
            self.is_global_zero = (rank == 0)

    # 1. Rank 1 runs flush_to_file with world_size=2 and uninitialized dist (gather fails)
    acc_rank1 = EvalSparsityAccumulator(threshold=0.5)
    acc_rank1.total_tokens = 5322242
    acc_rank1.pruned_tokens = 4502596
    acc_rank1.sum_utilities = 666595.4872

    fabric_rank1 = MockFabric(rank=1, world_size=2)
    acc_rank1.flush_to_file(fabric=fabric_rank1)

    # Rank 1 must write eval_sparsity_1.json, but NOT eval_sparsity.json
    assert (tmp_path / "eval_sparsity_1.json").exists()
    assert not (tmp_path / "eval_sparsity.json").exists()

    # 2. Rank 0 runs flush_to_file with world_size=2 and uninitialized dist (gather fails)
    acc_rank0 = EvalSparsityAccumulator(threshold=0.5)
    acc_rank0.total_tokens = 5425578
    acc_rank0.pruned_tokens = 4593339
    acc_rank0.sum_utilities = 675977.7672

    fabric_rank0 = MockFabric(rank=0, world_size=2)
    acc_rank0.flush_to_file(fabric=fabric_rank0)

    # Rank 0 must write eval_sparsity.json
    assert (tmp_path / "eval_sparsity.json").exists()


def test_sync_eval_sparsity_metrics(tmp_path: Path):
    """Test sync_eval_sparsity_metrics appends metrics to aggregated.txt cleanly."""
    task_dir = tmp_path / "eval_task"
    task_dir.mkdir(parents=True)

    results_file = task_dir / "aggregated.txt"
    results_file.write_text("nDCG@10                   all        0.4500\nRR@10                     all        0.3500\n")

    sparsity_file = task_dir / "eval_sparsity.json"
    sparsity_stats = {
        "sparsity": 0.6000,
        "mean_utility": 0.4200,
        "kept_ratio": 0.4000,
        "total_tokens": 1000,
    }
    sparsity_file.write_text(json.dumps(sparsity_stats))

    class MockEvaluation:
        def __init__(self, results_path):
            self.results = results_path

    class MockEvaluations:
        def __init__(self, eval_obj):
            self.results = [eval_obj]

    class MockCollection:
        def __init__(self, evaluations_dict):
            self.collection = evaluations_dict

    mock_tests = MockCollection({"msmarco": MockEvaluations(MockEvaluation(results_file))})

    # Run sync
    sync_eval_sparsity_metrics(mock_tests)

    content = results_file.read_text()
    assert "sparsity                  all        0.6000" in content
    assert "mean_utility" not in content
    assert "kept_ratio" not in content

    # Re-run sync to ensure no duplicate lines appended
    sync_eval_sparsity_metrics(mock_tests)
    lines = results_file.read_text().strip().split("\n")
    assert len(lines) == 3  # nDCG@10, RR@10, sparsity


def test_sync_eval_sparsity_metrics_multi_rank_merge(tmp_path: Path):
    """Verify that sync_eval_sparsity_metrics merges eval_sparsity.json and eval_sparsity_{rank}.json accurately."""
    task_dir = tmp_path / "eval_task"
    task_dir.mkdir(parents=True)

    results_file = task_dir / "aggregated.txt"
    results_file.write_text(
        "nDCG@10                   all        0.4500\nRR@10                     all        0.3500\n"
    )

    # Rank 0 file (written by rank 0 on failed gather)
    rank0_stats = {
        "sparsity": 0.8466082323394853,
        "mean_utility": 0.1245909223390451,
        "kept_ratio": 0.15339176766051468,
        "total_tokens": 5425578,
        "pruned_tokens": 4593339,
        "sum_utilities": 675977.7672,
    }
    (task_dir / "eval_sparsity.json").write_text(json.dumps(rank0_stats))

    # Rank 1 file (written by rank 1 on failed gather)
    rank1_stats = {
        "sparsity": 0.845996104649131,
        "mean_utility": 0.12524712089052267,
        "kept_ratio": 0.15400389535086906,
        "total_tokens": 5322242,
        "pruned_tokens": 4502596,
        "sum_utilities": 666595.4872,
    }
    (task_dir / "eval_sparsity_1.json").write_text(json.dumps(rank1_stats))

    class MockEvaluation:
        def __init__(self, results_path):
            self.results = results_path

    class MockEvaluations:
        def __init__(self, eval_obj):
            self.results = [eval_obj]

    class MockCollection:
        def __init__(self, evaluations_dict):
            self.collection = evaluations_dict

    mock_tests = MockCollection({"msmarco": MockEvaluations(MockEvaluation(results_file))})

    # Run sync
    sync_eval_sparsity_metrics(mock_tests)

    # 1. Verify merged eval_sparsity.json
    merged_data = json.loads((task_dir / "eval_sparsity.json").read_text())
    assert merged_data["total_tokens"] == 10747820
    assert merged_data["pruned_tokens"] == 9095935
    assert pytest.approx(merged_data["sparsity"], 1e-6) == 0.8463051111760338
    assert pytest.approx(merged_data["mean_utility"], 1e-6) == 0.12491586657996764
    assert pytest.approx(merged_data["kept_ratio"], 1e-6) == 0.15369488882396617

    # 2. Verify aggregated.txt
    content = results_file.read_text()
    assert "sparsity                  all        0.8463" in content


def test_init_eval_sparsity_hook_dynamic_threshold():
    """Verify InitEvalSparsityHook dynamically inherits threshold from ModernBertMiceScorer if threshold is None."""
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer, _ = mice_v2_scorer(hard_prune_threshold=0.3)
    hook = InitEvalSparsityHook.C(scorer=scorer)

    hook.execute()
    assert scorer.eval_sparsity_tracker is not None
    assert pytest.approx(scorer.eval_sparsity_tracker.threshold, 1e-4) == 0.3
