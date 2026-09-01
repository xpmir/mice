from functools import lru_cache
from typing import Any, Optional

from experimaestro import stop_tags
from experimaestro.experiments import configuration

from xpmir.datasets.adapters import RandomFold
from xpmir.datasets.samplers import ValidationSample
from samplers import prepare_collection
from xpmir.letor.validation import AggregatorValidationListener, ValidationListener
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.rankers import scorer_retriever

from configuration import CE_FineTuning, Validation
from tests import NANO_BEIR_KEYS, CE_MEASURES

import logging

logger = logging.getLogger(__name__)


@lru_cache
def nano_msmarco_validation_datasets(
    cfg: ValidationSample, launcher=None, blocking: Optional[bool] = None
):
    """Return validation over msmarco."""

    dataset = prepare_collection("co.huggingface.nano-beir.msmarco", download=blocking)
    logger.info("Loaded: msmarco")
    _ = next(dataset.documents.iter_documents())  # Force load documents
    _ = next(dataset.topics.iter())  # Force load queries

    random_folds = RandomFold.C(
        dataset=dataset,
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
    ).submit(launcher=launcher)

    return random_folds, dataset.documents


@lru_cache
def nanobeir_validation_datasets(
    cfg: ValidationSample, launcher=None, blocking: Optional[bool] = None
):
    """Return validations over all the NANO_BEIR datasets."""

    random_folds = {}
    documents = {}

    for dataset_name, dataset_id in NANO_BEIR_KEYS.items():
        # Prepare dataset components
        dataset = prepare_collection(dataset_id, download=blocking)
        logger.info(f"Loaded: {dataset_name}")

        random_folds[dataset_name] = RandomFold.C(
            dataset=dataset,
            seed=cfg.seed,
            fold=0,
            sizes=[cfg.size],
        ).submit(launcher=launcher)

        documents[dataset_name] = dataset.documents

        # Force load documents, queries, and qrels (if available)
        _ = next(dataset.documents.iter_documents())
        _ = next(dataset.topics.iter())
        if hasattr(dataset, "assessments"):
            _ = dataset.assessments.iter()
    return random_folds, documents


@lru_cache
def nanobeir11_validation_datasets(
    cfg: ValidationSample, launcher=None, blocking: Optional[bool] = None
):
    """Return validations over all the NANO_BEIR datasets except arguana and touche."""
    random_folds, documents = nanobeir_validation_datasets(
        cfg, launcher=launcher, blocking=blocking
    )
    # Filter out arguana and touche
    to_exclude = ["nano_arguana", "nano_webis-touche2020"]
    random_folds = {k: v for k, v in random_folds.items() if k not in to_exclude}
    documents = {k: v for k, v in documents.items() if k not in to_exclude}
    return random_folds, documents


@configuration()
class ValidationSet:
    cfg: CE_FineTuning
    items: list[tuple[str, Any, Any]]  # (name, dataset, documents)

    @classmethod
    def load(cls, cfg: CE_FineTuning, launcher):
        items = []
        blocking = cfg.preprocessing.blocking_download
        if cfg.learner.validation == Validation.MSMARCO.value:
            ds_val, docs = nano_msmarco_validation_datasets(
                cfg.validation, launcher=launcher, blocking=blocking
            )
            items.append(("msmarco", ds_val, docs))
        elif cfg.learner.validation in [
            Validation.NanoBEIR.value,
            Validation.ALL.value,
        ]:
            validations, documents = nanobeir_validation_datasets(
                cfg.validation, launcher=launcher, blocking=blocking
            )
            for name in validations:
                items.append((name, validations[name], documents[name]))
        elif cfg.learner.validation == Validation.NanoBEIR11.value:
            validations, documents = nanobeir11_validation_datasets(
                cfg.validation, launcher=launcher, blocking=blocking
            )
            for name in validations:
                items.append((name, validations[name], documents[name]))
        logger.info(f"Loaded validation datasets: {[name for name, _, _ in items]}")
        return cls(cfg=cfg, items=items)

    def to_evaluations(self) -> EvaluationsCollection:
        """Returns an EvaluationsCollection for the validation datasets"""
        evals = {}
        for name, ds, _ in self.items:
            # Use standard measures for validation evaluations
            evals[name] = Evaluations(ds, measures=CE_MEASURES)
        return EvaluationsCollection(**evals)

    def build_listeners(
        self,
        scorer_model,
        val_retrievers_factory,
        retriever_tag,
    ) -> tuple[list[ValidationListener], dict[str, ValidationListener]]:
        """Build the validation Listeners based of the logic from Validation enum"""
        listeners = []
        msmarco_validation = None

        for name, ds, docs in self.items:
            if (
                self.cfg.learner.validation == Validation.MSMARCO.value
                and name == "msmarco"
            ):
                track = True
            else:
                track = False

            # build the listener
            retriever = scorer_retriever(
                documents=docs,
                retrievers=val_retrievers_factory,
                scorer=scorer_model,
                batch_size=self.cfg.retrieval.batch_size,
            ).tag("first_stage", retriever_tag)

            listener = ValidationListener.C(
                id=(
                    f"bestval_zs_{name}" if len(self.items) > 1 else "bestval"
                ),  # Maintain ID compatibility
                dataset=ds,
                retriever=stop_tags(retriever),  # remove dependency
                validation_interval=self.cfg.learner.validation_interval,
                metrics={"nDCG": track, "RR@10": False},
            )
            listeners.append(listener)

            if name == "msmarco":
                msmarco_validation = listener

        tracked_validations = {}

        if self.cfg.learner.validation in [
            Validation.NanoBEIR.value,
            Validation.NanoBEIR11.value,
            Validation.ALL.value,
        ]:
            aggregator = AggregatorValidationListener.C(
                listeners=listeners,
                id="aggregated_validation",
                validation_interval=self.cfg.learner.validation_interval,
                metrics={"nDCG": True, "RR@10": False},
            )
            listeners.append(aggregator)
            tracked_validations["nano-beir"] = aggregator

        if self.cfg.learner.validation in [
            Validation.MSMARCO.value,
            Validation.ALL.value,
        ]:
            tracked_validations["msmarco"] = msmarco_validation

        return listeners, tracked_validations
