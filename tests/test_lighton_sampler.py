"""LightOn-style single-split batches (src/lighton_sampler.py, mmicep_brief.md step 3)."""

from functools import partial

import numpy as np
import pytest
from datamaestro_ir.data.huggingface import HuggingFacePointwiseDistillationSamples
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from lighton_sampler import (
    SingleSplitBatches,
    language_share_metrics,
    single_split_collate,
    split_probabilities,
    split_schedule,
)

class _Builder:
    """Same fields as EmbeddingsFineTuningPointwiseSamples, picklable."""

    query_field, document_field, score_field = "query", "document", "score"

    def _build_sample(self, row):
        return HuggingFacePointwiseDistillationSamples._build_sample(self, row)


def _split(name, n):
    return Dataset.from_dict(
        {
            "query": [f"q-{name}-{i}" for i in range(n)],
            "document": [f"{name}:{i}" for i in range(n)],
            "score": [float(i) for i in range(n)],
        }
    )


def _batches(names_sizes, batch_size=4, alpha=1.0, exhausted="replay", seed=0):
    names = [n for n, _ in names_sizes]
    return SingleSplitBatches(
        names=names,
        datasets=[_split(n, s) for n, s in names_sizes],
        builders=[_Builder() for _ in names],
        batch_size=batch_size,
        alpha=alpha,
        exhausted_split=exhausted,
        seed=seed,
    )


SPLITS = [("fiqa_fr", 40), ("nq_de", 120), ("msmarco_en", 400)]


def _rows(batch):
    """[(split, row)] of a list of (split index, sample)."""
    out = []
    for _, sample in batch:
        name, row = sample.document.document["text_item"].text.split(":")
        out.append((name, int(row)))
    return out


def _simulate(dataset, steps, world=1, workers=1):
    """Global batches in DataLoader order: step t comes from worker t % workers;
    the global batch concatenates the ranks' local batches."""
    B = dataset.batch_size
    iters = {
        (r, w): dataset.iterate(r, world, w, workers)
        for r in range(world)
        for w in range(workers)
    }
    batches = []
    for t in range(steps):
        batch = []
        for r in range(world):
            batch += [next(iters[r, t % workers]) for _ in range(B)]
        batches.append(batch)
    return batches


@pytest.mark.parametrize("world,workers", [(1, 1), (1, 2), (2, 2), (3, 2)])
def test_every_batch_comes_from_one_split(world, workers):
    dataset = _batches(SPLITS)
    for batch in _simulate(dataset, 200, world, workers):
        assert len({split for split, _ in batch}) == 1
        assert len({name for name, _ in _rows(batch)}) == 1


@pytest.mark.parametrize("world,workers", [(1, 2), (2, 2), (3, 2)])
def test_order_does_not_depend_on_workers_and_ranks_are_disjoint(world, workers):
    reference = [_rows(b) for b in _simulate(_batches(SPLITS, 4 * world), 60)]
    batches = [_rows(b) for b in _simulate(_batches(SPLITS, 4), 60, world, workers)]
    assert batches == reference
    # within the first pass of each split, no row is read twice
    seen = set()
    sizes = dict(SPLITS)
    for batch in batches:
        for name, row in batch:
            if len([1 for n, _ in seen if n == name]) < sizes[name]:
                assert (name, row) not in seen
                seen.add((name, row))


def test_first_pass_reads_the_on_disk_order():
    by_split = {}
    for batch in _simulate(_batches(SPLITS), 60):
        for name, row in _rows(batch):
            by_split.setdefault(name, []).append(row)
    for name, size in SPLITS:
        first_pass = by_split[name][:size]
        assert first_pass == list(range(len(first_pass)))


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
def test_split_frequencies_follow_size_power_alpha(alpha):
    sizes = [13008, 114030, 518851]
    p = split_probabilities(sizes, alpha)
    expected = np.array(sizes, dtype=float) ** alpha
    np.testing.assert_allclose(p, expected / expected.sum())

    steps = 200_000
    schedule = split_schedule(7, p)
    counts = np.bincount([next(schedule) for _ in range(steps)], minlength=len(p))
    chi2 = (((counts - steps * p) ** 2) / (steps * p)).sum()
    assert chi2 < 13.8  # chi-square, 2 degrees of freedom, p = 0.001


def test_schedule_is_seeded():
    p = split_probabilities([1, 2, 3], 1.0)
    first = [next(s) for s in [split_schedule(1, p)] for _ in range(1000)]
    again = [next(s) for s in [split_schedule(1, p)] for _ in range(1000)]
    other = [next(s) for s in [split_schedule(2, p)] for _ in range(1000)]
    assert first == again != other


def test_invalid_inputs():
    with pytest.raises(ValueError):
        split_probabilities([10, 0], 1.0)
    with pytest.raises(ValueError):
        _batches(SPLITS, exhausted="restart")


def _passes(exhausted, n=10, batch_size=4, steps=10):
    dataset = _batches([("fiqa_fr", n)], batch_size=batch_size, exhausted=exhausted)
    rows = [row for b in _simulate(dataset, steps) for _, row in _rows(b)]
    return [rows[i : i + n] for i in range(0, len(rows) - n + 1, n)]


def test_replay_reads_the_same_order_again():
    passes = _passes("replay")
    assert len(passes) >= 3
    assert all(p == list(range(10)) for p in passes)


def test_reshuffle_reads_a_new_order_each_pass():
    passes = _passes("reshuffle")
    assert passes[0] == list(range(10))
    for p in passes[1:]:
        assert sorted(p) == list(range(10))  # every row once per pass
    assert passes[1] != passes[0] and passes[1] != passes[2]
    assert _passes("reshuffle") == passes  # seeded


def _loader(dataset, workers=2):
    return StatefulDataLoader(
        dataset,
        batch_size=dataset.batch_size,
        num_workers=workers,
        collate_fn=partial(single_split_collate, names=dataset.names),
    )


def _documents(inputs):
    return [r["text_item"].text for r in inputs["records"].documents]


def test_dataloader_batches_match_the_schedule():
    dataset = _batches(SPLITS)
    expected = _simulate(_batches(SPLITS), 12, 1, 2)
    it = iter(_loader(dataset))
    for batch in expected:
        inputs = next(it)
        assert inputs["split"] == SPLITS[batch[0][0]][0]
        assert _documents(inputs) == [f"{n}:{r}" for n, r in _rows(batch)]
        assert inputs["teacher_scores"].tolist() == [float(r) for _, r in _rows(batch)]


@pytest.mark.parametrize("exhausted", ["replay", "reshuffle"])
def test_resume_from_dataloader_state(exhausted):
    splits = [("fiqa_fr", 10), ("nq_de", 30)]
    loader = _loader(_batches(splits, exhausted=exhausted))
    it = iter(loader)
    uninterrupted = [_documents(next(it)) for _ in range(25)]

    loader = _loader(_batches(splits, exhausted=exhausted))
    it = iter(loader)
    first = [_documents(next(it)) for _ in range(11)]
    state = loader.state_dict()
    del it, loader

    resumed = _loader(_batches(splits, exhausted=exhausted))
    resumed.load_state_dict(state)
    it = iter(resumed)
    rest = [_documents(next(it)) for _ in range(14)]
    assert first + rest == uninterrupted


def test_mixed_batch_is_refused():
    dataset = _batches(SPLITS)
    a = next(dataset.iterate(0, 1, 0, 1))
    b = ((a[0] + 1) % len(SPLITS), a[1])
    with pytest.raises(RuntimeError, match="mixes splits"):
        single_split_collate([a, b], dataset.names)


def test_language_share_metrics():
    class _Context:
        def __init__(self):
            self.metrics = {}

        def add_metric(self, metric):
            self.metrics[metric.key] = metric.compute()

    context = _Context()
    language_share_metrics(context, ["de", "en", "fr"], "nq_de")
    assert context.metrics == {
        "lighton/lang/de": 1.0,
        "lighton/lang/en": 0.0,
        "lighton/lang/fr": 0.0,
    }
