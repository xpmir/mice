"""
PreTTR (Pre-calculated Token-level Task-specific Representations) Training Experiment.

This module implements the training pipeline for PreTTR Cross-Encoders.
It uses joint tokenization but prevents cross-attention in early layers
to allow for offline document representation precomputation.
"""

import logging
from typing import List, Tuple

from experimaestro import LightweightTask
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment

from xpmir.papers.results import PaperResults
from xpmir.rankers import AbstractModuleScorer
from xpmir.text.huggingface.tokenizers import get_default_max_len
from xpmir.papers import configuration

from MICE.experiments.mice_training import TrainEvaluateLateInteractionScorer
from MICE.modeling.prettr_mice import prettr_scorer
from configuration import CE_FineTuning
from format import (
    loss_names,
    backbone_names_lower,
)

from logging_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class PreTTR_FineTuning(CE_FineTuning):
    ## PreTTR specific configuration
    join_layer: int = 6
    """The layer index at which full self-attention begins."""

    prettr_max_query_length: int = 32
    """The fixed offset used for offline document precomputation."""


def build_prettr(
    cfg: PreTTR_FineTuning,
) -> Tuple[AbstractModuleScorer, List[LightweightTask]]:
    """Build the PreTTR model."""
    default_max_len = get_default_max_len(cfg.base)
    max_len = (
        cfg.max_length
        if cfg.max_length and default_max_len > cfg.max_length
        else default_max_len
    )

    return prettr_scorer(
        hf_id=cfg.base,
        join_layer=cfg.join_layer,
        max_length=max_len,
        prettr_max_query_length=cfg.prettr_max_query_length,
    )


def get_name_from_tags(model_tags: dict, cfg: PreTTR_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    join_layer = model_tags.get("join_layer", cfg.join_layer)

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"

    return f"PreTTR-j{join_layer}-{base}{loss}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: PreTTR_FineTuning) -> PaperResults:
    """Main entry point for PreTTR training experiment."""
    return TrainEvaluateLateInteractionScorer(
        helper=helper,
        cfg=cfg,
        build_model_fn=build_prettr,
        get_model_name_fn=get_name_from_tags,
    )
