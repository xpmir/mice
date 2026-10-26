from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from functools import lru_cache
import logging
from pathlib import Path
from typing import Annotated, Callable, List, Optional, Union

from configuration import Evaluation
from datamaestro import prepare_dataset
from datamaestro_ir.data import Adhoc
from datamaestro_ir.data.csv import Topics as CSVTopics
from datamaestro_ir.data.trec import TrecAdhocAssessments
from experimaestro import Launcher, Param, Task, pathgenerator
from samplers import prepare_collection
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import (
    Evaluations,
    EvaluationsCollection as XpmirEvaluationsCollection,
    Retriever,
    RetrieverFactory,
)
from xpmir.measures import RR, R, Success, nDCG

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# EvaluationsCollection with Launcher Routing
# -----------------------------------------------------------------------------


class EvaluationsCollection(XpmirEvaluationsCollection):
    """Custom EvaluationsCollection that routes long/big datasets to a specialized launcher"""

    long_evals: list[str]
    default_launcher: Launcher
    long_launcher: Launcher

    def __init__(
        self,
        *args,
        long_evals: Optional[list[str]] = None,
        long_launcher: Optional[Launcher] = None,
        default_launcher: Optional[Launcher] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.long_evals = (
            long_evals if long_evals is not None else ["fever", "ms_marco", "hotpotqa"]
        )
        self.long_launcher = long_launcher
        self.default_launcher = default_launcher

    def choose_launcher(self, dataset_name: str, launcher: Launcher = None) -> Launcher:
        """Choose the appropriate launcher for a given dataset name by checking trigger substrings."""
        if (
            any(trigger in dataset_name for trigger in self.long_evals)
            and self.long_launcher is not None
        ):
            logger.info(f"Using long launcher for dataset {dataset_name}")
            return self.long_launcher
        else:
            return (
                self.default_launcher if self.default_launcher is not None else launcher
            )

    def evaluate_retriever(
        self,
        retriever: Union[Retriever, RetrieverFactory],
        launcher: Launcher = None,
        model_id: Optional[str] = None,
        overwrite: bool = False,
        with_run: bool = False,
        init_tasks=[],
        fabric_config=None,
    ) -> list:
        if model_id is not None and not overwrite:
            assert model_id not in self.per_model, (
                f"Model with ID `{model_id}` was already evaluated"
            )

        results = []
        for key, evaluations in self.collection.items():
            launcher_to_use = self.choose_launcher(key, launcher)
            result = evaluations.evaluate_retriever(
                key,
                retriever,
                launcher_to_use,
                init_tasks=init_tasks,
                with_run=with_run,
                fabric_config=fabric_config,
            )
            results.append(result)

        if model_id is not None:
            self.per_model[model_id] = results

        return results


def check_datasets_docs(evaluations_collection: EvaluationsCollection):
    """Ensure that documents exist -> triggers any lazy loading issues now.
    ir_datasets should download them with prepare_dataset ... but not load them until accessed.
    """
    for evals in evaluations_collection.collection.values():
        try:
            _ = next(evals.dataset.documents.iter_documents())
        except FileNotFoundError as e:
            logger.error(
                f"{e}- cannot download {evals.dataset.documents.id}, please consider "
                f"adding it manually (may happen for proprietary datasets such as Robust04)"
            )


CE_MEASURES = [Success @ 5, nDCG @ 10, nDCG @ 20, RR @ 10]
RETRIEVERS_MEASURES = [R @ 100, R @ 1000]


def get_ce_measures(include_recall_100: bool = False) -> list:
    """CE_MEASURES, optionally with R@100 appended (see Evaluation.include_recall_100)."""
    return CE_MEASURES + [R @ 100] if include_recall_100 else CE_MEASURES


def get_fold(dataset, size, seed=0, launcher=None):
    if size > 0:
        (fold_config,) = RandomFold.folds(
            seed=seed, sizes=[size], dataset=dataset, submit=False
        )
        return fold_config.submit(launcher=launcher)
    return dataset


# -----------------------------------------------------------------------------
# Pre-filtering Tasks
# -----------------------------------------------------------------------------


class FilterUnassessedTopics(Task):
    """Filters a dataset to only keep topics that have at least one assessment/qrel"""

    dataset: Param[Adhoc]
    """The Adhoc dataset to filter"""

    topics: Annotated[Path, pathgenerator("topics.tsv")]
    """Filtered topics file"""

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    """Filtered assessments file"""

    def task_outputs(self, dep) -> Adhoc:
        return dep(
            Adhoc.C(
                id=self.dataset.id + ".filtered" if self.dataset.id else "filtered",
                topics=dep(CSVTopics.C(id="", path=self.topics)),
                assessments=dep(TrecAdhocAssessments.C(id="", path=self.assessments)),
                documents=self.dataset.documents,
            )
        )

    def execute(self):
        # 1. Identify all topic IDs with assessments
        assessed_topic_ids = set()
        for qrels in self.dataset.assessments.iter():
            if len(qrels.assessments) > 0:
                assessed_topic_ids.add(qrels.topic_id)

        logger.info(
            f"Filtering dataset {self.dataset.id or ''}: keeping {len(assessed_topic_ids)} topics with assessments"
        )

        # 2. Write filtered topics
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        written_topic_ids = set()
        with self.topics.open("wt") as fp:
            for topic in self.dataset.topics.iter():
                if topic["id"] in assessed_topic_ids:
                    fp.write(f"{topic['id']}\t{topic['text_item'].text}\n")
                    written_topic_ids.add(topic["id"])

        # 3. Write assessments matching the written topics
        with self.assessments.open("wt") as fp:
            for qrels in self.dataset.assessments.iter():
                if qrels.topic_id in written_topic_ids:
                    for qrel in qrels.assessments:
                        fp.write(f"{qrels.topic_id} 0 {qrel.doc_id} {qrel.rel}\n")


# -----------------------------------------------------------------------------
# Public Dataset Key Dictionaries (Backwards Compatibility)
# -----------------------------------------------------------------------------

INDOMAIN_KEYS = {
    "msmarco_dev": "com.microsoft.msmarco.passage.dev.small",
    "trec2019": "com.microsoft.msmarco.passage.trec2019.judged",
    "trec2020": "com.microsoft.msmarco.passage.trec2020.judged",
}

BEIR_KEYS = {
    "scifact": "org.beir.scifact.test",
    "touche": "org.beir.webis.touche2020.v2",
    "fiqa": "org.beir.fiqa.test",
    "nfcorpus": "org.beir.nfcorpus.test",
    "arguana": "org.beir.arguana",
    "climate_fever": "org.beir.climate.fever",
    "dbpedia": "org.beir.dbpedia.entity.test",
    "fever": "org.beir.fever.test",
    "hotpotqa": "org.beir.hotpotqa.test",
    "nq": "org.beir.nq",
    "quora": "org.beir.quora.test",
    "scidocs": "org.beir.scidocs",
    "trec_covid": "org.beir.trec.covid",
}

BEIR13_DECONTAMINATED_KEYS = {
    "arguana_decontaminated": "ai.lighton.beir_decontaminated.arguana.test",
    "climate_fever_decontaminated": "ai.lighton.beir_decontaminated.climate_fever.test",
    "dbpedia_decontaminated": "ai.lighton.beir_decontaminated.dbpedia_entity.test",
    "fever_decontaminated": "ai.lighton.beir_decontaminated.fever.test",
    "fiqa_decontaminated": "ai.lighton.beir_decontaminated.fiqa.test",
    "hotpotqa_decontaminated": "ai.lighton.beir_decontaminated.hotpotqa.test",
    "nfcorpus_decontaminated": "ai.lighton.beir_decontaminated.nfcorpus.test",
    "nq_decontaminated": "ai.lighton.beir_decontaminated.nq.test",
    "quora_decontaminated": "ai.lighton.beir_decontaminated.quora.test",
    "scidocs_decontaminated": "ai.lighton.beir_decontaminated.scidocs.test",
    "scifact_decontaminated": "ai.lighton.beir_decontaminated.scifact.test",
    "touche_decontaminated": "ai.lighton.beir_decontaminated.webis_touche2020.test",
    "trec_covid_decontaminated": "ai.lighton.beir_decontaminated.trec_covid.test",
}

LOTTE_KEYS = {
    "lotte_writing": "edu.stanford.lotte.writing.test.search",
    "lotte_recreation": "edu.stanford.lotte.recreation.test.search",
    "lotte_science": "edu.stanford.lotte.science.test.search",
    "lotte_technology": "edu.stanford.lotte.technology.test.search",
    "lotte_lifestyle": "edu.stanford.lotte.lifestyle.test.search",
}

NANO_BEIR_IDS = [
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "webis-touche2020",
]

NANO_BEIR_KEYS = {
    f"nano_{name}": f"co.huggingface.nano-beir.{name}" for name in NANO_BEIR_IDS
}

LIMIT_KEYS = {
    "limit": "co.huggingface.limit",
    "limit_small": "co.huggingface.limit_small",
}

ROBUST04_KEYS = {
    "robust04": "gov.nist.trec.adhoc.robust.2004.withstore",
}


# -----------------------------------------------------------------------------
# Lazy Dataset Specification & Registry
# -----------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Declarative specification for an evaluation dataset.

    Execution (loading, filtering, folding) is deferred until `build()` is called.
    """

    name: str
    datamaestro_id: str
    is_collection: bool = False
    filter_unassessed: bool = False
    aliases: list[str] = field(default_factory=list)
    loader: Optional[Callable] = None

    def build(
        self,
        test_topic_nb: int = 0,
        launcher: Optional[Launcher] = None,
        blocking: Optional[bool] = None,
        measures: Optional[list] = None,
    ) -> Evaluations:
        """Lazily instantiate the dataset and return an Evaluations object."""
        if self.loader is not None:
            ds = self.loader(
                test_topic_nb=test_topic_nb,
                launcher=launcher,
                blocking=blocking,
            )
        else:
            if self.is_collection:
                ds = prepare_collection(self.datamaestro_id, download=blocking)
            else:
                ds = prepare_dataset(self.datamaestro_id, download=blocking)

            if self.filter_unassessed:
                ds = FilterUnassessedTopics.C(dataset=ds).submit(launcher=launcher)

            ds = get_fold(ds, test_topic_nb, launcher=launcher)

        if measures is None:
            measures = CE_MEASURES

        return Evaluations(ds, measures=measures)


class TestRegistry:
    """Centralized registry for evaluation datasets and aggregations."""

    __test__ = False

    def __init__(self):
        self._datasets: dict[str, DatasetSpec] = {}
        self._dataset_aliases: dict[str, str] = {}
        self._aggregations: dict[str, list[str]] = {}
        self._agg_aliases: dict[str, str] = {}
        self._display_names: dict[str, str] = {}

    def register_dataset(
        self,
        name: str,
        datamaestro_id: str,
        *,
        is_collection: bool = False,
        filter_unassessed: bool = False,
        aliases: Optional[list[str]] = None,
        loader: Optional[Callable] = None,
    ) -> DatasetSpec:
        """Register a single dataset specification."""
        spec = DatasetSpec(
            name=name,
            datamaestro_id=datamaestro_id,
            is_collection=is_collection,
            filter_unassessed=filter_unassessed,
            aliases=aliases or [],
            loader=loader,
        )
        self._datasets[name] = spec
        self._dataset_aliases[name.lower()] = name
        for alias in spec.aliases:
            self._dataset_aliases[alias.lower()] = name
        return spec

    def register_dict(
        self,
        dataset_dict: dict[str, str],
        *,
        is_collection_fn: Optional[Callable[[str, str], bool]] = None,
        filter_unassessed: bool = False,
        aliases_map: Optional[dict[str, list[str]]] = None,
    ):
        """Batch-register dataset specifications from an existing {name: datamaestro_id} dictionary."""
        for name, dmid in dataset_dict.items():
            is_coll = is_collection_fn(name, dmid) if is_collection_fn else False
            aliases = aliases_map.get(name, []) if aliases_map else []
            self.register_dataset(
                name=name,
                datamaestro_id=dmid,
                is_collection=is_coll,
                filter_unassessed=filter_unassessed,
                aliases=aliases,
            )

    def register_aggregation(
        self,
        name: str,
        dataset_or_agg_names: list[str],
        display_name: Optional[str] = None,
        aliases: Optional[list[str]] = None,
    ):
        """Register a named aggregation of datasets (or sub-aggregations)."""
        canonical = name.lower().replace(" ", "_").replace("-", "_")
        self._aggregations[canonical] = list(dataset_or_agg_names)
        self._agg_aliases[name.lower()] = canonical
        self._agg_aliases[canonical] = canonical
        if display_name:
            self._display_names[canonical] = display_name
            self._agg_aliases[display_name.lower()] = canonical
        if aliases:
            for alias in aliases:
                self._agg_aliases[alias.lower()] = canonical
                self._agg_aliases[alias.lower().replace(" ", "_").replace("-", "_")] = (
                    canonical
                )

    def resolve_dataset_name(self, name: str) -> Optional[str]:
        """Look up canonical dataset name from alias or exact match."""
        if name in self._datasets:
            return name
        return self._dataset_aliases.get(name.lower(), None)

    def resolve_aggregation_name(self, name: str) -> Optional[str]:
        """Look up canonical aggregation name from alias or exact match."""
        if name in self._aggregations:
            return name
        canonical = name.lower().replace(" ", "_").replace("-", "_")
        if canonical in self._aggregations:
            return canonical
        return self._agg_aliases.get(name.lower(), None)

    def resolve_names(self, targets: Union[str, list[str]]) -> list[str]:
        """Recursively expand aggregations and resolve aliases into unique dataset names."""
        if isinstance(targets, str):
            targets = [targets]

        resolved = []
        seen = set()

        def _resolve_item(item: str, visited_aggs: set[str]):
            # 1. Check if item matches an aggregation not yet visited in this branch
            agg_key = self.resolve_aggregation_name(item)
            if agg_key and agg_key not in visited_aggs:
                new_visited = visited_aggs | {agg_key}
                for child in self._aggregations[agg_key]:
                    _resolve_item(child, new_visited)
                return

            # 2. Check if item is a dataset
            ds_name = self.resolve_dataset_name(item)
            if ds_name and ds_name in self._datasets:
                if ds_name not in seen:
                    seen.add(ds_name)
                    resolved.append(ds_name)
                return

            # 3. Support dynamic ad-hoc Datamaestro IDs (e.g. 'org.beir.scifact.test')
            if "." in item:
                dyn_name = item.split(".")[-1].replace("-", "_")
                if dyn_name not in self._datasets:
                    self.register_dataset(dyn_name, item)
                if dyn_name not in seen:
                    seen.add(dyn_name)
                    resolved.append(dyn_name)
                return

            all_known = (
                list(self._datasets.keys())
                + list(self._aggregations.keys())
                + list(self._agg_aliases.keys())
            )
            suggestions = difflib.get_close_matches(item, all_known, n=3, cutoff=0.5)
            sugg_str = (
                f"\n   Did you mean: {', '.join(repr(s) for s in suggestions)}?"
                if suggestions
                else ""
            )

            logger.warning(
                "\n" + "!" * 80 + "\n"
                f"⚠️  TEST REGISTRY WARNING: Target {item!r} not found in test registry!{sugg_str}\n"
                f"   Available aggregations: {sorted(list(self._aggregations.keys()))}\n"
                + "!" * 80
                + "\n"
            )

        for target in targets:
            _resolve_item(target, set())

        return resolved

    def build_collection(
        self,
        targets: Union[str, list[str]],
        test_topic_nb: int = 0,
        include_recall_100: bool = False,
        launcher: Optional[Launcher] = None,
        blocking: Optional[bool] = None,
        default_launcher: Optional[Launcher] = None,
        long_launcher: Optional[Launcher] = None,
        long_evals: Optional[list[str]] = None,
        check_docs: bool = False,
        measures: Optional[list] = None,
    ) -> EvaluationsCollection:
        """Lazily build and return an EvaluationsCollection for ONLY the requested targets."""
        dataset_names = self.resolve_names(targets)
        evals_dict = {}

        for name in dataset_names:
            spec = self._datasets[name]
            ds_measures = (
                measures
                if measures is not None
                else get_ce_measures(include_recall_100)
            )
            evals_dict[name] = spec.build(
                test_topic_nb=test_topic_nb,
                launcher=launcher,
                blocking=blocking,
                measures=ds_measures,
            )

        if long_evals is None:
            long_evals = ["fever", "msmarco", "hotpotqa"]

        collection = EvaluationsCollection(
            long_evals=long_evals,
            long_launcher=long_launcher,
            default_launcher=default_launcher or launcher,
            **evals_dict,
        )

        if check_docs:
            logger.info("Checking docs in datasets...")
            check_datasets_docs(collection)

        return collection

    def get_display_aggregations(self) -> dict[str, list[str]]:
        """Generate aggregations dictionary with display names for format.py and training_utils.py."""
        res = {}
        for agg_name, members in self._aggregations.items():
            display = self._display_names.get(agg_name, agg_name)
            res[display] = self.resolve_names(members)
        return res


# -----------------------------------------------------------------------------
# Global Test Registry Initialization
# -----------------------------------------------------------------------------

test_registry = TestRegistry()

# 1. Register all datasets from existing dictionaries
test_registry.register_dict(
    INDOMAIN_KEYS,
    is_collection_fn=lambda name, dmid: "dev.small" in dmid,
)
test_registry.register_dict(BEIR_KEYS)
test_registry.register_dict(BEIR13_DECONTAMINATED_KEYS, filter_unassessed=True)
test_registry.register_dict(LOTTE_KEYS)
test_registry.register_dict(NANO_BEIR_KEYS)
test_registry.register_dict(
    LIMIT_KEYS,
    aliases_map={"limit_small": ["limit-small"]},
)
test_registry.register_dict(ROBUST04_KEYS)

# 2. Register aggregations directly from the dictionary keys
test_registry.register_aggregation(
    "in_domain",
    list(INDOMAIN_KEYS.keys()),
    display_name="In Domain",
)
test_registry.register_aggregation(
    "beir13",
    list(BEIR_KEYS.keys()),
    display_name="BEIR13",
)
test_registry.register_aggregation(
    "beir13_decontaminated",
    list(BEIR13_DECONTAMINATED_KEYS.keys()),
    display_name="BEIR13 Decontaminated",
)
test_registry.register_aggregation(
    "lotte",
    list(LOTTE_KEYS.keys()),
    display_name="Lotte-S",
    aliases=["lotte_search", "lotte-search"],
)
test_registry.register_aggregation(
    "nano_beir",
    list(NANO_BEIR_KEYS.keys()),
    display_name="Nano BEIR",
)
test_registry.register_aggregation(
    "limit",
    list(LIMIT_KEYS.keys()),
    display_name="LIMIT",
)
test_registry.register_aggregation(
    "robust04",
    ["robust04"],
    display_name="Robust04",
)
test_registry.register_aggregation(
    "minified",
    [
        "msmarco_dev",
        "trec2019",
        "trec2020",
        "scifact",
        "touche",
        "fiqa",
        "nfcorpus",
    ],
    display_name="minified",
)
test_registry.register_aggregation(
    "paper",
    ["in_domain", "beir13", "robust04", "lotte"],
    display_name="Paper Tests",
)
test_registry.register_aggregation(
    "all_datasets",
    ["beir13", "in_domain", "robust04", "lotte"],
    display_name="All Datasets",
)

# Export aggregations dict for format.py / training_utils.py
aggregations = test_registry.get_display_aggregations()
aggregation_hf = {
    "Mean In Domain": aggregations["In Domain"],
    "BEIR 13": aggregations["BEIR13"],
    "BEIR 13 (Decontaminated)": aggregations["BEIR13 Decontaminated"],
    "LoTTE (OOD)": aggregations["Lotte-S"],
    "Nano BEIR": aggregations["Nano BEIR"],
    "LIMIT": aggregations["LIMIT"],
}


# -----------------------------------------------------------------------------
# Backwards-Compatible Evaluation Helpers
# -----------------------------------------------------------------------------


def get_decontaminated_test(
    key: str,
    test_topic_nb: int,
    launcher=None,
    blocking: Optional[bool] = None,
) -> Evaluations:
    """Build a single LightOn decontaminated BEIR dataset evaluation."""
    spec = test_registry._datasets[key]
    return spec.build(
        test_topic_nb=test_topic_nb,
        launcher=launcher,
        blocking=blocking,
    )


@lru_cache
def minified_tests(
    test_topic_nb: int,
    check_docs: bool = True,
    include_recall_100: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """Returns the pool of queries for the evaluations to use for testing."""
    return test_registry.build_collection(
        "minified",
        test_topic_nb=test_topic_nb,
        include_recall_100=include_recall_100,
        launcher=launcher,
        blocking=blocking,
        check_docs=check_docs,
    )


@lru_cache
def BEIR_13_tests(
    test_topic_nb: int,
    include_recall_100: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """All of BEIR (minus the 5 datasets not publicly available)"""
    return test_registry.build_collection(
        "beir13",
        test_topic_nb=test_topic_nb,
        include_recall_100=include_recall_100,
        launcher=launcher,
        blocking=blocking,
    )


@lru_cache
def BEIR13_decontaminated_tests(
    test_topic_nb: int,
    include_recall_100: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """All 13 LightOn decontaminated BEIR datasets."""
    return test_registry.build_collection(
        "beir13_decontaminated",
        test_topic_nb=test_topic_nb,
        include_recall_100=include_recall_100,
        launcher=launcher,
        blocking=blocking,
    )


@lru_cache
def Robust04_test(
    test_topic_nb: int,
    include_recall_100: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """Robust04 dataset"""
    return test_registry.build_collection(
        "robust04",
        test_topic_nb=test_topic_nb,
        include_recall_100=include_recall_100,
        launcher=launcher,
        blocking=blocking,
    )


@lru_cache
def LoTTE_tests(
    test_topic_nb: int,
    include_recall_100: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """LoTTE Search dataset"""
    return test_registry.build_collection(
        "lotte",
        test_topic_nb=test_topic_nb,
        include_recall_100=include_recall_100,
        launcher=launcher,
        blocking=blocking,
    )


@lru_cache
def nano_beir_tests(
    test_topic_nb: int,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """NanoBEIR datasets"""
    return test_registry.build_collection(
        "nano_beir",
        test_topic_nb=test_topic_nb,
        launcher=launcher,
        blocking=blocking,
    )


@lru_cache
def limit_tests(
    test_topic_nb: int,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """LIMIT theoretical stress-test benchmarks (full 50k and small 46 docs)."""
    return test_registry.build_collection(
        "limit",
        test_topic_nb=test_topic_nb,
        launcher=launcher,
        blocking=blocking,
    )


@lru_cache
def paper_tests(
    test_topic_nb: int,
    check_docs: bool = True,
    include_recall_100: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """Returns the pool of queries for the evaluations to include in the paper."""
    return test_registry.build_collection(
        "paper",
        test_topic_nb=test_topic_nb,
        include_recall_100=include_recall_100,
        launcher=launcher,
        blocking=blocking,
        check_docs=check_docs,
    )


# -----------------------------------------------------------------------------
# Main Test Builder Function
# -----------------------------------------------------------------------------


def build_tests(
    cfg: Evaluation,
    check_docs: bool = True,
    launcher=None,
    blocking: Optional[bool] = None,
    default_launcher=None,
    long_launcher=None,
    long_evals: Optional[List[str]] = None,
    measures: Optional[list] = None,
) -> EvaluationsCollection:
    """Build the tests to use for evaluation during training or at the end of it.

    Supports:
    - Suite selection via flags (`in_domain`, `beir13`, `nanobeir`, `lotte_search`, `limit`, etc.)
    - Cherry-picking specific datasets or aggregations via `cfg.datasets`
    - Lazy loading: only the resolved datasets are instantiated
    """
    targets = []

    # 1. Collect from boolean flags
    if cfg.in_domain:
        targets.append("in_domain")
    if cfg.beir13:
        targets.append("beir13")
    if cfg.beir13_decontaminated:
        targets.append("beir13_decontaminated")
    if cfg.robust04:
        targets.append("robust04")
    if cfg.lotte_search:
        targets.append("lotte")
    if cfg.nanobeir:
        targets.append("nano_beir")
    if cfg.limit:
        targets.append("limit")
    if cfg.all_datasets:
        targets.append("all_datasets")

    # 2. Collect cherry-picked datasets or aggregation names
    if cfg.datasets:
        targets.extend(cfg.datasets)

    # 3. Default fallback if nothing was selected
    if not targets:
        targets = ["minified"]

    include_recall_100 = getattr(cfg, "include_recall_100", False)

    return test_registry.build_collection(
        targets=targets,
        test_topic_nb=cfg.test_max_topics,
        include_recall_100=include_recall_100,
        launcher=launcher,
        blocking=blocking,
        default_launcher=default_launcher,
        long_launcher=long_launcher,
        long_evals=long_evals,
        check_docs=check_docs,
        measures=measures,
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def get_max_query_length(
    tokenizer, datasets: Union[EvaluationsCollection, List[str], str]
) -> dict[str, dict]:
    """Helper to find the maximum query length for each dataset in the collection.

    Arguments:
        tokenizer: HFTokenizer or raw transformers tokenizer
        datasets: EvaluationsCollection, list of dataset IDs, or a single dataset ID

    Returns:
        A dict mapping dataset names to a dict with:
        - 'max_len': the maximum token length
        - 'sample': a random query from the dataset
        - 'sample_len': the token length of the sample
    """
    import random
    from datamaestro import prepare_dataset
    from xpmir.text.huggingface.tokenizers import HFTokenizer

    res = {}

    if isinstance(tokenizer, HFTokenizer):
        if not hasattr(tokenizer, "tokenizer"):
            tokenizer.__initialize__()
        hf_tokenizer = tokenizer.tokenizer
    else:
        hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    dataset_dict = {}
    if isinstance(datasets, EvaluationsCollection):
        for name, evals in datasets.collection.items():
            dataset_dict[name] = evals.dataset
    elif isinstance(datasets, str):
        dataset_dict[datasets] = prepare_dataset(datasets)
    elif isinstance(datasets, list):
        for ds_id in datasets:
            dataset_dict[ds_id] = prepare_dataset(ds_id)

    for name, dataset in dataset_dict.items():
        try:
            if hasattr(dataset, "topics") and hasattr(dataset.topics, "iter_topics"):
                topics = dataset.topics.iter_topics()
            elif hasattr(dataset, "topics") and hasattr(dataset.topics, "instance"):
                topics = dataset.topics.instance().iter()
            else:
                topics = dataset.topics

            unique_queries = list({topic["text_item"].text for topic in topics})
        except Exception as e:
            logger.warning(f"Could not iterate topics for dataset {name}: {e}")
            res[name] = {"max_len": 0, "sample": "", "sample_len": 0}
            continue

        if not unique_queries:
            res[name] = {"max_len": 0, "sample": "", "sample_len": 0}
            continue

        lengths = [
            len(hf_tokenizer.encode(q, add_special_tokens=False))
            for q in unique_queries
        ]

        sample_query = random.choice(unique_queries)
        sample_len = len(hf_tokenizer.encode(sample_query, add_special_tokens=False))

        res[name] = {
            "max_len": max(lengths),
            "sample": sample_query,
            "sample_len": sample_len,
        }

    return res
