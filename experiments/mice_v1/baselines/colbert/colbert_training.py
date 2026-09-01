"""
ColBERT (Contextualized Late Interaction over BERT) Training Experiment.
"""

import logging
from typing import List, Tuple

from experimaestro import LightweightTask
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment

from xpmir.papers.results import PaperResults
from xpmir.rankers import AbstractModuleScorer
from xpmir.papers import configuration
from xpmir.neural.colbert import ColBERTEncoder

from MICE.experiments.mice_training import TrainEvaluateLateInteractionScorer
from configuration import CE_FineTuning
from format import (
    loss_names,
    backbone_names_lower,
)

from logging_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class Colbert_FineTuning(CE_FineTuning):
    dim: int = 128
    """Output dimension of the per-token projection."""

    query_maxlen: int = 32
    """Maximum number of tokens kept for a query."""

    doc_maxlen: int = 180
    """Maximum number of tokens kept for a document."""


def Colbert_scorer(
    hf_id: str,
    dim: int = 128,
    query_maxlen: int = 32,
    doc_maxlen: int = 180,
) -> Tuple[ColBERTEncoder, List[LightweightTask]]:
    from xpmir.text.huggingface.base import HFConfigID, HFModel, HFModelInitFromID
    from xpmir.text.huggingface.encoders import HFTokensEncoder
    from xpmir.text.huggingface.tokenizers import HFTokenizer, HFTokenizerAdapter
    from xpmir.text.adapters import TopicTextConverter
    from xpmir.text.encoders import TokenizedTextEncoder
    from xpmir.neural.colbert import ColBERTEncoder

    # Model and initialization
    config = HFConfigID.C(hf_id=hf_id)
    model = HFModel.C(config=config)
    init_tasks = [HFModelInitFromID.C(model=model)]

    # Tokenizer adapter for IDTextRecord
    tokenizer = HFTokenizerAdapter.C(
        tokenizer=HFTokenizer.C(model_id=hf_id), converter=TopicTextConverter.C()
    )

    # Encoder part
    hftokens_encoder = HFTokensEncoder.C(model=model)

    # Full tokenized text encoder
    tokenized_text_encoder = TokenizedTextEncoder.C(
        tokenizer=tokenizer, encoder=hftokens_encoder
    )

    colbert = ColBERTEncoder.C(
        encoder=tokenized_text_encoder,
        dim=dim,
        query_maxlen=query_maxlen,
        doc_maxlen=doc_maxlen,
    )

    return colbert, init_tasks


def build_colbert(
    cfg: Colbert_FineTuning,
) -> Tuple[AbstractModuleScorer, List[LightweightTask]]:
    """Build the Colbert model."""
    return Colbert_scorer(
        hf_id=cfg.base,
        dim=cfg.dim,
        query_maxlen=cfg.query_maxlen,
        doc_maxlen=cfg.doc_maxlen,
    )


def get_name_from_tags(model_tags: dict, cfg: Colbert_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    dim = model_tags.get("dim", cfg.dim)
    q_len = model_tags.get("query_maxlen", cfg.query_maxlen)
    d_len = model_tags.get("doc_maxlen", cfg.doc_maxlen)

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"

    return f"ColBERT-d{dim}-q{q_len}-d{d_len}-{base}{loss}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: Colbert_FineTuning) -> PaperResults:
    """Main entry point for ColBERT training experiment."""
    return TrainEvaluateLateInteractionScorer(
        helper=helper,
        cfg=cfg,
        build_model_fn=build_colbert,
        get_model_name_fn=get_name_from_tags,
    )
