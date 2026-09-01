from functools import lru_cache
from datamaestro import prepare_dataset
from typing import Union, List, Optional
from xpmir.datasets.adapters import RandomFold
from samplers import prepare_collection
from xpmir.evaluation import (
    Evaluations,
    EvaluationsCollection as XpmirEvaluationsCollection,
    Retriever,
    RetrieverFactory,
)
from xpmir.measures import RR, nDCG, R, Success
from configuration import Evaluation
from experimaestro import Task, Param, pathgenerator, Launcher
from datamaestro_ir.data.csv import Topics as CSVTopics
from datamaestro_ir.data.trec import TrecAdhocAssessments
from datamaestro_ir.data import Adhoc
from pathlib import Path
from typing import Annotated

import logging

logger = logging.getLogger(__name__)


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

    # override the evaluate_retriever method with smart routing to the appropriate launcher based on dataset name
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
            # Determine which launcher to use
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

        # Adds to per model results
        if model_id is not None:
            self.per_model[model_id] = results

        return results


def check_datasets_docs(evaluations_collection: EvaluationsCollection):
    """Ensure that documents exists -> triggers any lazy loading issues now
    ir_datasets show download them with prepare_dataset ... but not load them until accessed
    """
    for evals in evaluations_collection.collection.values():
        try:
            _ = next(evals.dataset.documents.iter_documents())
        except FileNotFoundError as e:
            logger.error(
                f"{e}- cannot dowload {evals.dataset.documents.id}, please consider adding it manually (may happen for proprietary datasets such as Robust04)"
            )


CE_MEASURES = [Success @ 5, nDCG @ 10, nDCG @ 20, RR @ 10]
RETRIEVERS_MEASURES = [R @ 100, R @ 1000]


def get_fold(dataset, size, seed=0, launcher=None):
    if size > 0:
        (fold_config,) = RandomFold.folds(
            seed=seed, sizes=[size], dataset=dataset, submit=False
        )
        return fold_config.submit(launcher=launcher)
    return dataset


@lru_cache
def minified_tests(
    test_topic_nb: int,
    check_docs: bool = True,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """Returns the pool of queries for the evaluations to use for testing."""

    v1_devsmall_ds = prepare_collection(
        "com.microsoft.msmarco.passage.dev.small", download=blocking
    )
    dl19 = prepare_dataset(
        "com.microsoft.msmarco.passage.trec2019.judged", download=blocking
    )
    dl20 = prepare_dataset(
        "com.microsoft.msmarco.passage.trec2020.judged", download=blocking
    )

    v1_devsmall_ds = get_fold(v1_devsmall_ds, test_topic_nb, launcher=launcher)

    scifact = prepare_dataset("org.beir.scifact.test", download=blocking)  # 300 queries
    scifact = get_fold(scifact, test_topic_nb, launcher=launcher)

    touche = prepare_dataset("org.beir.webis.touche2020.v2", download=blocking)

    fiqa = prepare_dataset("org.beir.fiqa.test", download=blocking)  # 648 queries
    fiqa = get_fold(fiqa, test_topic_nb, launcher=launcher)

    nfcorpus = prepare_dataset(
        "org.beir.nfcorpus.test", download=blocking
    )  # 323 queries
    nfcorpus = get_fold(nfcorpus, test_topic_nb, launcher=launcher)

    measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
    tests = EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall_ds, measures=measures),
        trec2019=Evaluations(dl19, measures=measures),
        trec2020=Evaluations(dl20, measures=measures),
        scifact=Evaluations(scifact, measures=measures),
        touche=Evaluations(touche, measures=measures),
        fiqa=Evaluations(fiqa, measures=measures),
        nfcorpus=Evaluations(nfcorpus, measures=measures),
    )

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(tests)

    return tests


@lru_cache
def BEIR_13_tests(
    test_topic_nb: int,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """All of BEIR (minus the 5 datasets not publicly available)"""

    ## BEIR datasets
    scifact = prepare_dataset("org.beir.scifact.test", download=blocking)
    scifact = get_fold(scifact, test_topic_nb, launcher=launcher)

    touche = prepare_dataset("org.beir.webis.touche2020.v2", download=blocking)

    fiqa = prepare_dataset("org.beir.fiqa.test", download=blocking)
    fiqa = get_fold(fiqa, test_topic_nb, launcher=launcher)

    nfcorpus = prepare_dataset("org.beir.nfcorpus.test", download=blocking)
    nfcorpus = get_fold(nfcorpus, test_topic_nb, launcher=launcher)

    arguana = prepare_dataset("org.beir.arguana", download=blocking)
    arguana = get_fold(arguana, test_topic_nb, launcher=launcher)

    climate_fever = prepare_dataset("org.beir.climate.fever", download=blocking)
    climate_fever = get_fold(climate_fever, test_topic_nb, launcher=launcher)

    dbpedia = prepare_dataset("org.beir.dbpedia.entity.test", download=blocking)
    dbpedia = get_fold(dbpedia, test_topic_nb, launcher=launcher)

    fever = prepare_dataset("org.beir.fever.test", download=blocking)
    fever = get_fold(fever, test_topic_nb, launcher=launcher)

    hotpotqa = prepare_dataset("org.beir.hotpotqa.test", download=blocking)
    hotpotqa = get_fold(hotpotqa, test_topic_nb, launcher=launcher)

    nq = prepare_dataset("org.beir.nq", download=blocking)
    nq = get_fold(nq, test_topic_nb, launcher=launcher)

    quora = prepare_dataset("org.beir.quora.test", download=blocking)
    quora = get_fold(quora, test_topic_nb, launcher=launcher)

    scidocs = prepare_dataset("org.beir.scidocs", download=blocking)
    scidocs = get_fold(scidocs, test_topic_nb, launcher=launcher)

    trec_covid = prepare_dataset("org.beir.trec.covid", download=blocking)
    trec_covid = get_fold(trec_covid, test_topic_nb, launcher=launcher)

    measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
    return EvaluationsCollection(
        fever=Evaluations(fever, measures=measures),
        arguana=Evaluations(arguana, measures=measures),
        climate_fever=Evaluations(climate_fever, measures=measures),
        dbpedia=Evaluations(dbpedia, measures=measures),
        fiqa=Evaluations(fiqa, measures=measures),
        hotpotqa=Evaluations(hotpotqa, measures=measures),
        nfcorpus=Evaluations(nfcorpus, measures=measures),
        nq=Evaluations(nq, measures=measures),
        quora=Evaluations(quora, measures=measures),
        scidocs=Evaluations(scidocs, measures=measures),
        scifact=Evaluations(scifact, measures=measures),
        touche=Evaluations(touche, measures=measures),
        trec_covid=Evaluations(trec_covid, measures=measures),
    )


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

INDOMAIN_KEYS = {
    "msmarco_dev": "com.microsoft.msmarco.passage.dev.small",
    "trec2019": "com.microsoft.msmarco.passage.trec2019.judged",
    "trec2020": "com.microsoft.msmarco.passage.trec2020.judged",
}

LOTTE_KEYS = {
    "lotte_writing": "edu.stanford.lotte.writing.test.search",
    "lotte_recreation": "edu.stanford.lotte.recreation.test.search",
    "lotte_science": "edu.stanford.lotte.science.test.search",
    "lotte_technology": "edu.stanford.lotte.technology.test.search",
    "lotte_lifestyle": "edu.stanford.lotte.lifestyle.test.search",
}


def get_decontaminated_test(
    key: str,
    test_topic_nb: int,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> Evaluations:
    """Build a single LightOn decontaminated BEIR dataset evaluation."""
    measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
    dataset_id = BEIR13_DECONTAMINATED_KEYS[key]
    ds = prepare_dataset(dataset_id, download=blocking)
    # Apply the filter before taking the fold or evaluating
    ds = FilterUnassessedTopics.C(dataset=ds).submit(launcher=launcher)
    ds = get_fold(ds, test_topic_nb, launcher=launcher)
    return Evaluations(ds, measures=measures)


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


@lru_cache
def BEIR13_decontaminated_tests(
    test_topic_nb: int,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """All 13 LightOn decontaminated BEIR datasets."""
    evals = {}
    for key in BEIR13_DECONTAMINATED_KEYS:
        evals[key] = get_decontaminated_test(
            key,
            test_topic_nb,
            retrievers_only=retrievers_only,
            launcher=launcher,
            blocking=blocking,
        )

    return EvaluationsCollection(**evals)


@lru_cache
def Robust04_test(
    test_topic_nb: int,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """Robust04 dataset"""
    logger.info("Preparing Robust04 dataset...")
    robust04 = prepare_dataset(
        "gov.nist.trec.adhoc.robust.2004.withstore", download=blocking
    )
    robust04 = get_fold(robust04, test_topic_nb, launcher=launcher)

    return EvaluationsCollection(
        robust04=Evaluations(
            robust04, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
    )


@lru_cache
def LoTTE_tests(
    test_topic_nb: int,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """LoTTE Search dataset"""
    logger.info("Preparing LoTTE Search datasets...")
    lotte_writing = prepare_dataset(
        "edu.stanford.lotte.writing.test.search", download=blocking
    )
    lotte_writing = get_fold(lotte_writing, test_topic_nb, launcher=launcher)

    lotte_recreation = prepare_dataset(
        "edu.stanford.lotte.recreation.test.search", download=blocking
    )
    lotte_recreation = get_fold(lotte_recreation, test_topic_nb, launcher=launcher)

    lotte_science = prepare_dataset(
        "edu.stanford.lotte.science.test.search", download=blocking
    )
    lotte_science = get_fold(lotte_science, test_topic_nb, launcher=launcher)

    lotte_technology = prepare_dataset(
        "edu.stanford.lotte.technology.test.search", download=blocking
    )
    lotte_technology = get_fold(lotte_technology, test_topic_nb, launcher=launcher)

    lotte_lifestyle = prepare_dataset(
        "edu.stanford.lotte.lifestyle.test.search", download=blocking
    )
    lotte_lifestyle = get_fold(lotte_lifestyle, test_topic_nb, launcher=launcher)

    return EvaluationsCollection(
        lotte_writing=Evaluations(
            lotte_writing, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        lotte_recreation=Evaluations(
            lotte_recreation,
            CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES,
        ),
        lotte_science=Evaluations(
            lotte_science, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        lotte_technology=Evaluations(
            lotte_technology,
            CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES,
        ),
        lotte_lifestyle=Evaluations(
            lotte_lifestyle, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
    )


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

# Nano Beir Datamaestro Ids from id
NANO_BEIR_KEYS = {
    f"nano_{name}": f"co.huggingface.nano-beir.{name}" for name in NANO_BEIR_IDS
}


@lru_cache
def nano_beir_tests(
    test_topic_nb: int,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """NanoBEIR datasets"""

    measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
    evals = {}
    for key, dataset_id in NANO_BEIR_KEYS.items():
        # Map back to the dataset name (e.g., nano_arguana -> arguana)
        ds = prepare_dataset(dataset_id, download=blocking)
        ds = get_fold(ds, test_topic_nb, launcher=launcher)
        evals[key] = Evaluations(ds, measures=measures)

    return EvaluationsCollection(**evals)


@lru_cache
def paper_tests(
    test_topic_nb: int,
    check_docs: bool = True,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
) -> EvaluationsCollection:
    """Returns the pool of queries for the evaluations to include in the paper."""

    # In domain - MS Marco + TREC DL
    v1_dev = prepare_collection(
        "com.microsoft.msmarco.passage.dev.small", download=blocking
    )
    dl19 = prepare_dataset(
        "com.microsoft.msmarco.passage.trec2019.judged", download=blocking
    )
    dl20 = prepare_dataset(
        "com.microsoft.msmarco.passage.trec2020.judged", download=blocking
    )

    v1_dev = get_fold(v1_dev, test_topic_nb, launcher=launcher)
    dl19 = get_fold(dl19, test_topic_nb, launcher=launcher)
    dl20 = get_fold(dl20, test_topic_nb, launcher=launcher)

    # Out of domain - BEIR
    beir = BEIR_13_tests(
        test_topic_nb,
        retrievers_only=retrievers_only,
        launcher=launcher,
        blocking=blocking,
    )
    robust04 = Robust04_test(
        test_topic_nb,
        retrievers_only=retrievers_only,
        launcher=launcher,
        blocking=blocking,
    )
    lotte = LoTTE_tests(
        test_topic_nb,
        retrievers_only=retrievers_only,
        launcher=launcher,
        blocking=blocking,
    )

    paper_tests_res = EvaluationsCollection(
        msmarco_dev=Evaluations(
            v1_dev, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        trec2019=Evaluations(
            dl19, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        trec2020=Evaluations(
            dl20, CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        ),
        **beir.collection,
        **robust04.collection,
        **lotte.collection,
    )

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(paper_tests_res)
    return paper_tests_res


def build_tests(
    cfg: Evaluation,
    check_docs: bool = True,
    retrievers_only: bool = False,
    launcher=None,
    blocking: Optional[bool] = None,
    default_launcher=None,
    long_launcher=None,
    long_evals: Optional[List[str]] = None,
) -> EvaluationsCollection:
    """Build the tests to use for evaluation during training or at the end of it.
    :param cfg: Configuration for the evaluation
    :param check_docs: Whether to check that documents are accessible (triggers downloads if needed)
    :returns: The evaluations collection to use
    """
    all_evals = {}

    # Helper to add evals to the dictionary
    def add_evals(evals_collection: EvaluationsCollection):
        for name, evals in evals_collection.collection.items():
            if name not in all_evals:
                all_evals[name] = evals

    # 2. In-domain (MSMarco + TREC DL)
    if cfg.in_domain or cfg.all_datasets:
        v1_dev = prepare_collection(
            "com.microsoft.msmarco.passage.dev.small", download=blocking
        )
        dl19 = prepare_dataset(
            "com.microsoft.msmarco.passage.trec2019.judged", download=blocking
        )
        dl20 = prepare_dataset(
            "com.microsoft.msmarco.passage.trec2020.judged", download=blocking
        )

        v1_dev = get_fold(v1_dev, cfg.test_max_topics, launcher=launcher)
        dl19 = get_fold(dl19, cfg.test_max_topics, launcher=launcher)
        dl20 = get_fold(dl20, cfg.test_max_topics, launcher=launcher)

        measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
        add_evals(
            EvaluationsCollection(
                msmarco_dev=Evaluations(v1_dev, measures),
                trec2019=Evaluations(dl19, measures),
                trec2020=Evaluations(dl20, measures),
            )
        )

    # 3. BEIR13 (All of it)
    if cfg.beir13 or cfg.all_datasets:
        add_evals(
            BEIR_13_tests(
                cfg.test_max_topics,
                retrievers_only=retrievers_only,
                launcher=launcher,
                blocking=blocking,
            )
        )

    # 3b. BEIR13 Decontaminated
    if cfg.beir13_decontaminated:
        add_evals(
            BEIR13_decontaminated_tests(
                cfg.test_max_topics,
                retrievers_only=retrievers_only,
                launcher=launcher,
                blocking=blocking,
            )
        )

    if cfg.robust04 or cfg.all_datasets:
        add_evals(
            Robust04_test(
                cfg.test_max_topics,
                retrievers_only=retrievers_only,
                launcher=launcher,
                blocking=blocking,
            )
        )

    if cfg.lotte_search or cfg.all_datasets:
        add_evals(
            LoTTE_tests(
                cfg.test_max_topics,
                retrievers_only=retrievers_only,
                launcher=launcher,
                blocking=blocking,
            )
        )

    # 1. NanoBEIR
    if cfg.nanobeir:
        add_evals(
            nano_beir_tests(
                cfg.test_max_topics,
                retrievers_only=retrievers_only,
                launcher=launcher,
                blocking=blocking,
            )
        )
    # 4. Filter or add specific datasets if provided
    if cfg.datasets:
        beir13_all = None
        for ds_name in cfg.datasets:
            if ds_name in all_evals:
                continue

            if ds_name in BEIR13_DECONTAMINATED_KEYS:
                all_evals[ds_name] = get_decontaminated_test(
                    ds_name,
                    cfg.test_max_topics,
                    retrievers_only=retrievers_only,
                    launcher=launcher,
                    blocking=blocking,
                )
            elif ds_name in BEIR_KEYS:
                if beir13_all is None:
                    beir13_all = BEIR_13_tests(
                        cfg.test_max_topics,
                        retrievers_only=retrievers_only,
                        launcher=launcher,
                        blocking=blocking,
                    )
                all_evals[ds_name] = beir13_all.collection[ds_name]
            elif ds_name in INDOMAIN_KEYS:
                ds_id = INDOMAIN_KEYS[ds_name]
                if "dev.small" in ds_id:
                    ds = prepare_collection(ds_id, download=blocking)
                else:
                    ds = prepare_dataset(ds_id, download=blocking)
                ds = get_fold(ds, cfg.test_max_topics, launcher=launcher)
                measures = CE_MEASURES if not retrievers_only else RETRIEVERS_MEASURES
                all_evals[ds_name] = Evaluations(ds, measures=measures)
            elif ds_name in NANO_BEIR_KEYS:
                nano_all = nano_beir_tests(
                    cfg.test_max_topics,
                    retrievers_only=retrievers_only,
                    launcher=launcher,
                    blocking=blocking,
                )
                all_evals[ds_name] = nano_all.collection[ds_name]
            elif ds_name in LOTTE_KEYS:
                lotte_all = LoTTE_tests(
                    cfg.test_max_topics,
                    retrievers_only=retrievers_only,
                    launcher=launcher,
                    blocking=blocking,
                )
                all_evals[ds_name] = lotte_all.collection[ds_name]
            elif ds_name == "robust04":
                rob_all = Robust04_test(
                    cfg.test_max_topics,
                    retrievers_only=retrievers_only,
                    launcher=launcher,
                    blocking=blocking,
                )
                all_evals[ds_name] = rob_all.collection["robust04"]
            else:
                logger.warning(f"Dataset {ds_name} not found in supported dataset list")

    # If nothing was selected, use minified_tests as default (original behavior)
    if not all_evals and not cfg.all_datasets:
        return minified_tests(
            cfg.test_max_topics,
            check_docs=check_docs,
            retrievers_only=retrievers_only,
            launcher=launcher,
            blocking=blocking,
        )

    if long_evals is None:
        long_evals = ["fever", "ms_marco", "hotpotqa"]

    tests = EvaluationsCollection(
        long_evals=long_evals,
        long_launcher=long_launcher,
        default_launcher=default_launcher or launcher,
        **all_evals,
    )

    if check_docs:
        logger.info("Checking docs in datasets...")
        check_datasets_docs(tests)

    return tests


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
    from xpmir.text.huggingface.tokenizers import HFTokenizer
    from datamaestro import prepare_dataset
    import random

    res = {}

    # Get the underlying transformers tokenizer
    if isinstance(tokenizer, HFTokenizer):
        if not hasattr(tokenizer, "tokenizer"):
            tokenizer.__initialize__()
        hf_tokenizer = tokenizer.tokenizer
    else:
        hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    # Normalize datasets to a dictionary of dataset objects
    dataset_dict = {}
    if isinstance(datasets, EvaluationsCollection):
        for name, evals in datasets.collection.items():
            dataset_dict[name] = evals.dataset
    elif isinstance(datasets, str):
        dataset_dict[datasets] = prepare_dataset(datasets)
    elif isinstance(datasets, list):
        for ds_id in datasets:
            # For list of strings, use the ID as name
            dataset_dict[ds_id] = prepare_dataset(ds_id)

    for name, dataset in dataset_dict.items():
        try:
            # Try different ways to iterate topics
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

        # Tokenize all queries and find the maximum length
        lengths = [
            len(hf_tokenizer.encode(q, add_special_tokens=False))
            for q in unique_queries
        ]

        # Pick a random sample
        sample_query = random.choice(unique_queries)
        sample_len = len(hf_tokenizer.encode(sample_query, add_special_tokens=False))

        res[name] = {
            "max_len": max(lengths),
            "sample": sample_query,
            "sample_len": sample_len,
        }

    return res
