"""LightOn-style batches for mMICE-P (mmicep_brief.md, step 3).

Each batch is drawn entirely from one training split, the split of global step
``t`` being chosen with probability proportional to ``len(split) ** alpha``
(LightOn's ``MultinomialBatchSampler``). The schedule is computed from a seed, so
that every rank and every DataLoader worker agrees on it:

- worker ``w`` (out of ``W``) produces the steps ``t`` with ``t % W == w``, which
  is the order in which the DataLoader reads the workers' batches;
- if step ``t`` is the ``j``-th time split ``s`` is drawn, rank ``r`` reads the rows
  ``[(j * world + r) * B, +B)`` of ``s`` (``B`` = per-rank batch size): the ranks
  read disjoint rows, the global batch comes from a single split, and the order in
  which a split is read does not depend on the number of workers.

A split read to the end starts again (position ``g`` -> pass ``g // n``), either in
the same order (``replay``, MICE-P) or in a new order seeded by (seed, split, pass)
(``reshuffle``, LightOn). The split rows are shuffled once on disk
(:class:`multilingual_data.ShufflePointwiseSplits`).
"""

import logging
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from datamaestro_ir.data.lighton import EmbeddingsFineTuningPointwiseSamples
from experimaestro import Param
from torch.utils.data import get_worker_info
from xpm_torch.base import Sampler
from xpm_torch.datasets import ShardedIterableDataset
from xpm_torch.metrics import ScalarMetric
from xpm_torch.trainers import LossTrainer, TrainerContext
from xpmir.letor.distillation.pointwise import (
    DistillationPointwiseLoss,
    PointwiseDistillationTrainer,
    pointwise_distillation_collate,
)
from xpmir.letor.trainers.utils import wrap_collate_with_tokenizer

from configuration import ExhaustedSplit
from multilingual_data import split_identity, split_seed

logger = logging.getLogger(__name__)

_SCHEDULE_CHUNK = 1 << 16


def split_probabilities(sizes: Sequence[int], alpha: float) -> np.ndarray:
    """Probability of each split: ``size ** alpha``, normalised."""
    sizes = np.asarray(sizes, dtype=np.float64)
    if (sizes <= 0).any():
        raise ValueError(f"Every split must be non-empty, got sizes {sizes.tolist()}")
    weights = sizes**alpha
    return weights / weights.sum()


def split_schedule(seed: int, probabilities: np.ndarray) -> Iterator[int]:
    """The split of each global step ``t = 0, 1, ...``; the same for any reader."""
    rng = np.random.default_rng([seed, 0])
    cdf = np.cumsum(probabilities)
    last = len(probabilities) - 1
    while True:
        u = rng.random(_SCHEDULE_CHUNK) * cdf[-1]
        yield from np.minimum(np.searchsorted(cdf, u, side="right"), last).tolist()


class SingleSplitBatches(ShardedIterableDataset):
    """Yields ``(split index, sample)``, ``batch_size`` consecutive items per step,
    all from the split chosen for that step (see the module docstring). Infinite.

    The position (next step and item) is part of :meth:`state_dict`, so that a
    ``StatefulDataLoader`` resumes exactly where it stopped.
    """

    def __init__(
        self,
        names: List[str],
        datasets: List[Any],
        builders: List[Any],
        batch_size: int,
        alpha: float,
        exhausted_split: str,
        seed: int,
    ):
        super().__init__()
        if not (len(names) == len(datasets) == len(builders)):
            raise ValueError("names, datasets and builders must have the same length")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.names = names
        self.datasets = datasets
        self.builders = builders
        self.sizes = [len(d) for d in datasets]
        self.batch_size = batch_size
        self.exhausted_split = ExhaustedSplit(exhausted_split)
        self.seed = seed
        self.probabilities = split_probabilities(self.sizes, alpha)
        self._position = (0, 0)  # (next step, next item within that step)
        self._permutations: Dict[int, Tuple[int, np.ndarray]] = {}

    def state_dict(self) -> Dict[str, Any]:
        step, item = self._position
        return {"step": step, "item": item}

    def load_state_dict(self, state: Dict[str, Any]):
        self._position = (int(state["step"]), int(state["item"]))

    def _row_indices(self, split: int, start: int) -> np.ndarray:
        """Row indices for positions ``[start, start + batch_size)`` of ``split``."""
        n = self.sizes[split]
        positions = np.arange(start, start + self.batch_size, dtype=np.int64)
        passes, rows = np.divmod(positions, n)
        if self.exhausted_split is ExhaustedSplit.RESHUFFLE:
            for p in np.unique(passes):
                if p == 0:
                    continue  # the first pass reads the on-disk (shuffled) order
                mask = passes == p
                rows[mask] = self._permutation(split, int(p))[rows[mask]]
        return rows

    def _permutation(self, split: int, p: int) -> np.ndarray:
        """Order of pass ``p`` >= 1 of ``split``, seeded by (seed, split, pass)."""
        cached = self._permutations.get(split)
        if cached is None or cached[0] != p:
            name = self.names[split]
            rng = np.random.default_rng(split_seed(self.seed, f"{name}/pass{p}"))
            order = rng.permutation(self.sizes[split]).astype(np.int32)
            self._permutations[split] = cached = (p, order)
        return cached[1]

    def _read(self, split: int, rows: np.ndarray) -> Dict[str, list]:
        data = self.datasets[split]
        if rows[-1] - rows[0] == len(rows) - 1 and (np.diff(rows) == 1).all():
            return data[int(rows[0]) : int(rows[-1]) + 1]
        return data[rows.tolist()]

    def _resolve_layout(self) -> Tuple[int, int, int, int]:
        """``(rank, world_size, worker, num_workers)``."""
        shard_id, num_shards = self._resolve_sharding()
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        return (
            shard_id // num_workers,
            num_shards // num_workers,
            shard_id % num_workers,
            num_workers,
        )

    def iter_shard(self, shard_id: int, num_shards: int):  # pragma: no cover
        raise NotImplementedError("SingleSplitBatches is iterated with __iter__")

    def __iter__(self) -> Iterator[Tuple[int, Any]]:
        rank, world, worker, num_workers = self._resolve_layout()
        return self.iterate(rank, world, worker, num_workers)

    def iterate(
        self, rank: int, world: int, worker: int, num_workers: int
    ) -> Iterator[Tuple[int, Any]]:
        """The items of one (rank, worker), resuming from :attr:`_position` (read
        now, not when the iteration starts)."""
        return self._iterate(rank, world, worker, num_workers, *self._position)

    def _iterate(self, rank, world, worker, num_workers, start_step, start_item):
        B = self.batch_size
        counts = np.zeros(len(self.names), dtype=np.int64)
        for t, split in enumerate(split_schedule(self.seed, self.probabilities)):
            j = counts[split]
            counts[split] += 1
            if t % num_workers != worker or t < start_step:
                continue
            first = start_item if t == start_step else 0
            if first >= B:
                continue
            start = (int(j) * world + rank) * B
            if start // self.sizes[split] != (start + B - 1) // self.sizes[split]:
                logger.info(
                    "Split %s exhausted at step %d: starting pass %d (%s)",
                    self.names[split],
                    t,
                    (start + B - 1) // self.sizes[split],
                    self.exhausted_split.value,
                )
            columns = self._read(split, self._row_indices(split, start))
            builder = self.builders[split]
            for i in range(first, B):
                row = {key: values[i] for key, values in columns.items()}
                self._position = (t, i + 1)
                yield split, builder._build_sample(row)


class LightOnSplitSampler(Sampler):
    """One LightOn split per batch, drawn with probability ``len ** alpha``."""

    splits: Param[List[EmbeddingsFineTuningPointwiseSamples]]
    """The shuffled, on-disk pointwise splits (distinct names)."""

    alpha: Param[float] = 1.0
    """Split weights are ``len(split) ** alpha``."""

    exhausted_split: Param[str] = ExhaustedSplit.REPLAY.value
    """``replay`` (same order, MICE-P) or ``reshuffle`` (new order, LightOn)."""

    seed: Param[int]
    """Seed of the batch schedule and of the reshuffled passes."""

    def __validate__(self):
        ExhaustedSplit(self.exhausted_split)
        names = [s.name for s in self.splits]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"Splits must be non-empty with distinct names: {names}")

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.splits]

    def as_dataset(self, batch_size: int) -> SingleSplitBatches:
        """``batch_size`` is the per-rank batch size of the DataLoader."""
        dataset = SingleSplitBatches(
            names=self.names,
            datasets=[s.data for s in self.splits],
            builders=list(self.splits),
            batch_size=batch_size,
            alpha=self.alpha,
            exhausted_split=self.exhausted_split,
            seed=self.seed,
        )
        for name, size, p in zip(dataset.names, dataset.sizes, dataset.probabilities):
            logger.info("Split %s: %d rows, p=%.4f", name, size, p)
        return dataset


def single_split_collate(items: List[Tuple[int, Any]], names: List[str]):
    """:func:`pointwise_distillation_collate`, plus the ``split`` name of the batch."""
    splits = {split for split, _ in items}
    if len(splits) != 1:
        raise RuntimeError(
            f"A batch mixes splits {sorted(names[s] for s in splits)}: "
            "the batches are out of step with the sampler"
        )
    inputs = pointwise_distillation_collate([sample for _, sample in items])
    inputs["split"] = names[splits.pop()]
    return inputs


class LightOnPointwiseDistillationTrainer(PointwiseDistillationTrainer):
    """Pointwise distillation with one LightOn split per batch; logs the share of
    batches per language (``lighton/lang/<lang>``)."""

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        if not isinstance(self.sampler, LightOnSplitSampler):
            raise TypeError(
                f"{self.__class__.__name__} needs a LightOnSplitSampler, "
                f"got {type(self.sampler).__name__}"
            )
        # PointwiseDistillationTrainer.initialize, with the single-split dataset
        LossTrainer.initialize(self, random, context)
        self.lossfn.initialize(self.model)
        for loss in context.hooks(DistillationPointwiseLoss):
            loss.initialize(self.model)

        names = self.sampler.names
        self._languages = sorted({split_identity(n)[1] for n in names})
        local_batch_size = (
            context.get_local_batch_size(self.batch_size)
            if context is not None
            else self.batch_size
        )
        collate_fn = wrap_collate_with_tokenizer(
            self.model, partial(single_split_collate, names=names), log=logger
        )
        self._create_dataloader(
            self.sampler.as_dataset(local_batch_size), collate_fn=collate_fn
        )

    def process_batch(self, batch):
        language_share_metrics(self.context, self._languages, batch["split"])
        super().process_batch(batch)


def language_share_metrics(context, languages: List[str], split: str):
    """One 0/1 metric per language: averaged over a logging window, the share of
    batches drawn from that language."""
    lang = split_identity(split)[1]
    for language in languages:
        context.add_metric(
            ScalarMetric(f"lighton/lang/{language}", float(language == lang), 1)
        )


def build_lighton_trainer(
    cfg, lossfn, splits: Optional[Dict[str, EmbeddingsFineTuningPointwiseSamples]]
):
    """The trainer for ``Losses.MSE_lighton_multilingual``."""
    return LightOnPointwiseDistillationTrainer.C(
        sampler=LightOnSplitSampler.C(
            splits=list(splits.values()),
            alpha=float(cfg.lighton_alpha),
            exhausted_split=ExhaustedSplit(cfg.lighton_exhausted_split).value,
            seed=getattr(cfg, "seed", 42),
        ),
        lossfn=lossfn,
        batch_size=cfg.learner.optimization.batch_size,
    )
