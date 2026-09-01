"""Legacy pointwise distillation sampler and trainer adapters.
Kept only in dev branch for backward hash compatibility
"""

from datamaestro.data.huggingface import (
    FlattenAndShuffleDataset as DatamaestroFlattenAndShuffleDataset,
)
from xpmir.letor.distillation.pointwise import (
    PointwiseDistillationInputs,
    pointwise_distillation_collate,
    PreShuffledPointwiseDataset as XPMPreShuffledPointwiseDataset,
    PointwiseDistillationSampler as XPMPointwiseDistillationSampler,
    DistillationPointwiseLoss as XPMDistillationPointwiseLoss,
    PointwiseMSELoss as XPMPointwiseMSELoss,
    PointwiseDistillationTrainer as XPMPointwiseDistillationTrainer,
)

__all__ = [
    "PointwiseDistillationInputs",
    "pointwise_distillation_collate",
    "FlattenAndShuffleDataset",
    "PreShuffledPointwiseDataset",
    "PointwiseDistillationSampler",
    "DistillationPointwiseLoss",
    "PointwiseMSELoss",
    "PointwiseDistillationTrainer",
]


class FlattenAndShuffleDataset(DatamaestroFlattenAndShuffleDataset):
    """Subclass of FlattenAndShuffleDataset to preserve pointwise_distillation identifier hash."""

    pass


class PreShuffledPointwiseDataset(XPMPreShuffledPointwiseDataset):
    """Legacy adapter for PreShuffledPointwiseDataset."""

    pass


class PointwiseDistillationSampler(XPMPointwiseDistillationSampler):
    """Legacy adapter for PointwiseDistillationSampler to preserve task hash."""

    pass


class DistillationPointwiseLoss(XPMDistillationPointwiseLoss):
    """Legacy adapter for DistillationPointwiseLoss to preserve task hash."""

    pass


class PointwiseMSELoss(XPMPointwiseMSELoss):
    """Legacy adapter for PointwiseMSELoss to preserve task hash."""

    pass


class PointwiseDistillationTrainer(XPMPointwiseDistillationTrainer):
    """Legacy adapter for PointwiseDistillationTrainer to preserve task hash."""

    pass
