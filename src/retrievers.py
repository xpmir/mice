"""This file contains utilities for building Retrievers that can be used in various experiments"""

from typing import Any, Dict, List, Optional
from experimaestro import Launcher, stop_tags
from datamaestro_ir.data import Documents
from xpmir.rankers import Retriever
import xpmir.interfaces.anserini as anserini
from xpmir.index.sparse import SparseRetriever, SeismicSparseRetriever
from xpmir.rankers.standard import BM25, Model
from xpmir.index.bow import BOWRetriever, BOWSparseRetrieverIndexBuilder
from xpmir.evaluation import EvaluationsCollection, RetrieverFactory
from xpmir.index.plaid import PlaidIndexBuilder, PlaidRetriever
from xpm_torch.configuration import FabricConfiguration

from MICE.plaid_mice import MicePlaidRetrieverv2
from configuration import CE_FineTuning, PlaidConfiguration, ScorerType
from index_utils import get_splade_index, get_plaid_index
import logging

logger = logging.getLogger(__name__)

### Factories for bm25, Splade, and PLAID retrievers


def plaid_retriever(
    cfg,
    name: str,
    encoder,
    documents: Documents,
    launcher_index,
    init_tasks: list = None,
    topk: int = None,
    seed: int = 42,
    tags: Optional[Dict[str, Any]] = None,
    long_launcher_index: Optional[Launcher] = None,
    long_evals: Optional[List[str]] = None,
    key: Optional[str] = None,
    **kwargs,
) -> Retriever.C:
    """Factory for PLAID Retriever, given the current configuration"""

    retriever = (
        PlaidRetriever.C(
            encoder=encoder,
            index=stop_tags(
                get_plaid_index(
                    documents=documents,
                    encoder=encoder,
                    plaid_cfg=cfg.plaid,
                    launcher_index=launcher_index,
                    indexation_cfg=getattr(cfg, "indexation", None),
                    init_tasks=init_tasks,
                    seed=seed,
                    long_launcher_index=long_launcher_index,
                    long_evals=long_evals,
                    key=key,
                )
            ),
            topk=topk or cfg.retrieval.k,
            n_ivf_probe=cfg.plaid.n_ivf_probe,
            n_full_scores=cfg.plaid.n_full_scores,
            fabric_config=FabricConfiguration.C(precision="bf16-true"),
        )
        .tag("first_stage", "plaid")
        .tag("scorer", name)
        .tag("scorer_type", ScorerType.stMultiVectorEncoder.value)
        .tag("plaid_retriever", True)
    )
    if tags:
        for k, v in tags.items():
            if k not in ("first_stage", "scorer", "scorer_type", "plaid_retriever"):
                retriever.tag(k, v)
    return retriever


def splade_retriever(
    cfg: CE_FineTuning,
    name: str,
    encoder: Model,
    documents: Documents,
    launcher_index,
    init_tasks: list = None,
    topk: int = None,
    in_memory: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    long_launcher_index: Optional[Launcher] = None,
    long_evals: Optional[List[str]] = None,
    key: Optional[str] = None,
    **kwargs,
) -> Retriever.C:
    seismic_cfg = getattr(cfg, "seismic", None)
    is_seismic = seismic_cfg is not None and getattr(seismic_cfg, "enabled", False)

    index_obj = stop_tags(  # drop 'index_documents' tag that clutters eval.
        get_splade_index(
            documents,
            splade_encoder=encoder,
            indexation_cfg=cfg.indexation,
            launcher_index=launcher_index,
            init_tasks=init_tasks,
            long_launcher_index=long_launcher_index,
            long_evals=long_evals,
            key=key,
            seismic_cfg=seismic_cfg,
        )
    )

    batch_size = cfg.retrieval.batch_size
    in_memory = getattr(cfg.retrieval, "in_memory", False) or in_memory

    if is_seismic:
        retriever = (
            SeismicSparseRetriever.C(
                index=index_obj,
                topk=topk or cfg.retrieval.k,
                batchsize=batch_size,
                encoder=encoder,
                in_memory=in_memory,
                query_cut=seismic_cfg.query_cut,
                heap_factor=seismic_cfg.heap_factor,
                n_knn=seismic_cfg.n_knn,
            )
            .tag("first_stage", name)
            .tag("approximate", "seismic")
        )
    else:
        retriever = SparseRetriever.C(
            index=index_obj,
            topk=topk or cfg.retrieval.k,
            batchsize=batch_size,
            encoder=encoder,
            in_memory=in_memory,
        ).tag("first_stage", name)
    if tags:
        for k, v in tags.items():
            if k != "first_stage":
                retriever.tag(k, v)
    return retriever


def bm25_retriever(
    cfg: CE_FineTuning,
    name: str,
    documents: Documents,
    launcher_index=None,
    topk: int = None,
    tags: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Retriever.C:
    """Factory for BM25 Retriever, given the current configuration"""

    # -----The baseline------

    bow_index = BOWSparseRetrieverIndexBuilder.C(
        documents=documents,
        max_docs=cfg.indexation.max_indexed,
    ).submit(launcher=launcher_index)

    retriever = BOWRetriever.C(
        index=stop_tags(bow_index),
        model=BM25.C(),
        topk=topk or cfg.retrieval.k,
    ).tag("first_stage", name)
    if tags:
        for k, v in tags.items():
            if k != "first_stage":
                retriever.tag(k, v)
    return retriever


def anserini_bm25_retriever(
    cfg: CE_FineTuning,
    name: str,
    documents: Documents,
    launcher_index,
    topk: int = None,
    **kwargs,
) -> Retriever.C:
    """Factory for BM25 Retriever, given the current configuration"""
    index = anserini.index_builder(launcher=launcher_index)(documents)
    return anserini.AnseriniRetriever.C(
        k=topk or cfg.retrieval.k,
        model=BM25.C(),
        index=stop_tags(index),
        store=documents,
    ).tag("first_stage", name)


class MicePlaidRetrieverFactory(RetrieverFactory):
    """A factory that Manages the Plaid Indexes for Mice models"""

    def __init__(
        self,
        mice_model,
        first_stage_factory: RetrieverFactory,
        topk: int,
        batchsize: int,
        n_ivf_probe: int = 8,
        n_full_scores: int = 0,
        max_query_length: Optional[int] = None,
        max_doc_length: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ):
        self.mice_model = mice_model
        self.topk = topk
        self.batchsize = batchsize
        self.first_stage_factory = first_stage_factory
        self.n_ivf_probe = n_ivf_probe
        self.n_full_scores = n_full_scores
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.chunk_size = chunk_size
        self.doc_encoder = mice_model.get_document_encoder(
            max_doc_length=max_doc_length
        )
        self.indexes: Dict[str] = {}
        self.documents: Dict[str, Documents] = {}

    # def build_index(self, key: str, documents: Documents, run: AdhocRun):
    #     """Register a run for a given document collection"""
    #     if key in self.indexes.keys():
    #         logger.warning(
    #             f"Plaid index for dataset '{key}' already stored for {self.mice_model}"
    #         )
    #     self.runs[key] = run
    #     self.documents[key] = documents

    def build_indexes(
        self,
        evaluations: EvaluationsCollection,
        cfg: PlaidConfiguration,
        batch_size: int,
        seed,
        launcher_index,
        init_tasks,
        long_launcher_index: Optional[Launcher] = None,
        long_evals: Optional[List[str]] = None,
    ):
        effective_max_doc = cfg.max_doc_length or self.max_doc_length
        if effective_max_doc is not None:
            self.doc_encoder = self.mice_model.get_document_encoder(
                max_doc_length=effective_max_doc
            )

        # Triggers default to evaluation's long_evals if not explicitly overridden for indexation
        effective_long_evals = (
            long_evals
            if long_evals is not None
            else getattr(evaluations, "long_evals", ["fever", "msmarco", "hotpotqa"])
        )

        use_colbert = cfg.use_colbert_first_stage
        if use_colbert:
            if cfg.compress_only:
                logger.warning(
                    "use_colbert_first_stage is enabled but compress_only is True; "
                    "forcing compress_only=False to allow ColBERT first-stage search."
                )
            effective_compress_only = False
        else:
            effective_compress_only = True
            logger.info("Enforcing compress_only=True for MICE PLAID indexation.")

        # 1) Build the index and retriever for this model
        for (
            dataset_key,
            collection,
        ) in evaluations.collection.items():
            documents = collection.dataset.documents

            if documents.id not in self.indexes.keys():
                is_long = bool(
                    long_launcher_index is not None
                    and any(trigger in dataset_key for trigger in effective_long_evals)
                )
                launcher = long_launcher_index if is_long else launcher_index
                logging.info(
                    f"Indexing dataset {documents.id} (key='{dataset_key}') using "
                    f"{'long_launcher_index' if is_long else 'launcher_index'}: {launcher}"
                )
                self.indexes[dataset_key] = stop_tags(
                    PlaidIndexBuilder.C(
                        documents=documents,
                        encoder=self.doc_encoder,
                        buffer_size=cfg.buffer_size,
                        batch_size=batch_size,
                        fast_plaid_batch_size=cfg.fast_plaid_batch_size,
                        n_bits=cfg.n_bits,
                        kmeans_niters=cfg.kmeans_niters,
                        n_samples_kmeans=cfg.n_samples_kmeans,
                        max_points_per_centroid=cfg.max_points_per_centroid,
                        seed=seed,
                        compress_only=effective_compress_only,
                        force_cpu_indexing=cfg.force_cpu_indexing,
                        fabric_config=FabricConfiguration.C(precision="bf16-true"),
                    )
                    .tag("index_documents", documents.id)
                    .submit(
                        launcher=launcher,
                        init_tasks=init_tasks,
                        # transient=TransientMode.REMOVE,
                    )
                )
            else:
                logging.warning(f"Dataset {documents.id} already indexed with Plaid")

    def _resolve_index(self, dataset: Documents, key: str = None):
        # Try to find the run by key first, then by dataset ID
        index = self.indexes.get(key) if key else None
        if index is None:
            # Fallback to dataset ID if key not provided or not found
            for k, docs in self.documents.items():
                if docs.id == dataset.id:
                    index = self.indexes[k]
                    break

        if index is None:
            raise KeyError(
                f"No run found for dataset key='{key}' or id='{dataset.id}'"
                f"Available: {','.join(self.runs.keys())}"
            )

        return index

    def __call__(self, dataset: Documents, key: str = None) -> MicePlaidRetrieverv2:
        index = self._resolve_index(dataset, key)
        return MicePlaidRetrieverv2.C(
            store=dataset,
            index=index,
            scorer=self.mice_model,
            retriever=self.first_stage_factory(dataset, key=key),
            top_k=self.topk,
            batchsize=self.batchsize,
            max_query_length=self.max_query_length,
            chunk_size=self.chunk_size,
        ).tag("plaid_retriever", True)

    def colbert_retriever(self, dataset: Documents, key: str = None) -> PlaidRetriever:
        """Genuine ColBERT-style (MaxSim/IVF) first-stage retriever, searching
        the same PLAID index built by `build_indexes`, using MICE's own
        bottom-layer contextualized embeddings for both queries and documents."""
        query_encoder = (
            self.mice_model.get_query_encoder(max_query_length=self.max_query_length)
            if hasattr(self.mice_model, "get_query_encoder")
            else self.mice_model
        )
        return PlaidRetriever.C(
            encoder=query_encoder,
            index=self._resolve_index(dataset, key),
            topk=self.topk,
            n_ivf_probe=self.n_ivf_probe,
            n_full_scores=self.n_full_scores,
            fabric_config=FabricConfiguration.C(precision="16-true"),
        ).tag("first_stage", "mice-colbert")

    # @classmethod
    # def from_results(cls, name: str, results: List) -> "MicePlaidRetrieverFactory":
    #     factory = cls(name)
    #     for res in results:
    #         factory.add_run(res.key, res.task.dataset.documents, res.run)
    #     return factory
