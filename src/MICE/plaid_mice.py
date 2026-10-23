import contextlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from MICE.modeling.mice import MICETokenizedTexts, QueryDocInput
from datamaestro_ir.data.base import IDTextRecord, ScoredDocument
from experimaestro import Param, field
from experimaestro.core.arguments import Meta
from lightning_fabric import Fabric
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from xpm_torch.configuration import FabricConfiguration
from xpmir.index.plaid import PlaidIndex
from xpmir.letor.records import PointwiseItem
from xpmir.rankers import AbstractModuleScorer
from xpmir.rankers.retriever import Retriever
from xpmir.rankers.scorer import (
    ReRankingDataset,
    RerankingInputs,
    TwoStageRetriever,
    reranking_collate,
)
from xpmir.text.encoders import TokenizerOptions

logger = logging.getLogger(__name__)


class MicePlaidRetriever(Retriever):
    """Implements Plaid retrieval using a MICE BERT model.
    At the moment, the PLAID index is only used for compressing documents,
    retrieval is run with a sparse retriever and documents are gathered with the
    `get_document_tokens()` method of the index.
    """

    scorer: Param[AbstractModuleScorer]
    """The appropriate Mice scorer."""

    first_stage_retriever: Param[Retriever]
    """A first stage retriever to fetch candidate documents for the PLAID index."""

    index: Param[PlaidIndex]
    """The fast-plaid index to search."""

    topk: Param[int]
    """Number of documents to return per query."""

    fabric_config: Meta[FabricConfiguration] = field(
        default_factory=FabricConfiguration.C
    )
    """Control the device for the model encoding and fast-plaid index."""

    def initialize(self):
        super().initialize()
        if not self.index.compress_only:
            raise RuntimeError(
                "Search with PLAID isn't supported at this stage.\
                The index must be built with compress_only=True and retrieval is done with a sparse retriever."
            )

        logger.info("PLAID retriever: initializing the encoder")
        # 1. Initialize Fabric first
        fabric = self.fabric_config.get_fabric()
        fabric.launch()

        with fabric.init_module():
            self.scorer.initialize()
            self.scorer = fabric.setup(self.scorer)
            self.scorer.eval()

    def _store(self):
        return self.index.documents

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        # Retrieves the documents
        scoredDocuments = self.first_stage_retriever.retrieve(record)

        # Decompressing the document tokens embedding from the index by their doc_id
        doc_list: list = []
        for doc_id in scoredDocuments:
            token_embeddings = self.index.get_document_tokens(doc_id)
            doc_list.append(
                token_embeddings.to(self.scorer.device)
            )  # Move token embeddings to scorer device

        # Tokenize the queries
        tokenized_queries = self.scorer.batch_tokenize(
            [record], input_nature=QueryDocInput.QUERY
        )

        with torch.no_grad():
            # Pass the tokenized queries and the doc embeddings through the MICE encoder to get the scores
            results = self.scorer(
                tokenized=MICETokenizedTexts(
                    tokenized_q=tokenized_queries, tokenized_docs=None
                ),
                doc_hidden_states=doc_list,
            )

        single = results[0] if results else []
        out: List[ScoredDocument] = []
        for doc_index, score in single:
            out.append(
                ScoredDocument(self._store().document_int(int(doc_index)), float(score))
            )

        out.sort(reverse=True)
        return out[: (self.top_k or len(out))]


class PlaidReRankingDataset(ReRankingDataset):
    """A dataset that yields PointwiseItem records for re-ranking.
    Inherits from ReRankingDataset for backwards compatibility.
    """

    def __init__(
        self,
        candidates: Union[
            List[Tuple[IDTextRecord, List[ScoredDocument]]],
            Dict[str, IDTextRecord],
        ],
        retriever: Optional[Retriever] = None,
    ):
        if isinstance(candidates, dict):
            # Backwards compatibility: if queries dict was passed, retrieve sequentially
            assert retriever is not None, (
                "retriever must be provided if queries dict is passed"
            )
            retrieved_candidates = []
            for _, query in candidates.items():
                retrieved_candidates.append((query, retriever.retrieve(query)))
            candidates = retrieved_candidates
        super().__init__(candidates)


class MiceRerankingCollate:
    """Picklable collate function for MICE-PLAID reranking DataLoader."""

    def __init__(
        self,
        scorer,
        index: Optional[PlaidIndex] = None,
        max_query_length: Optional[int] = None,
    ):
        self.scorer = scorer.module if hasattr(scorer, "module") else scorer
        self.index = index
        self.max_query_length = max_query_length

    def __call__(self, batch: List[PointwiseItem]) -> RerankingInputs:
        inputs = reranking_collate(batch)
        doc_list: list = []
        queries_list: list = []
        for record in inputs["records"]:
            doc_list.append(record.document["id"])
            queries_list.append(record.topic)

        if issubclass(type(self.scorer), AbstractModuleScorer):
            options = (
                TokenizerOptions(max_length=self.max_query_length)
                if self.max_query_length is not None
                else None
            )
            tokenized_queries = self.scorer.batch_tokenize(
                queries_list, options=options, input_nature=QueryDocInput.QUERY
            )
        else:
            tokenized_queries = None

        inputs["tokenized_queries"] = tokenized_queries
        inputs["doc_ids"] = doc_list

        if self.index is not None and doc_list:
            inputs["doc_token_embeddings"] = self.index.get_document_tokens(
                doc_list, device="cpu"
            )

        return inputs


class MicePlaidRetrieverv2(TwoStageRetriever):
    index: Param[PlaidIndex]
    """The fast-plaid index to search."""

    max_query_length: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Maximum query length for reranking. If None, uses scorer's default."""

    def build_reranking_dataloader(
        self, candidates: List[Tuple[IDTextRecord, List[ScoredDocument]]]
    ):
        """Builds a dataloader for re-ranking all documents for pre-retrieved candidates.
        Allows for efficient two-stage retrieval with cross-query batching on GPU when the scorer
        supports it (i.e. is an AbstractModuleScorer and batchsize > 0).
        """
        dataset = ReRankingDataset(candidates)
        collate_fn = MiceRerankingCollate(
            self.scorer, self.index, self.max_query_length
        )

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=self.batchsize,
            num_workers=min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 4),
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn,
        )

        return dataloader

    def score_batch(
        self, inputs: Dict[str, Any], scorer_type: type
    ) -> Optional[np.ndarray]:
        fabric: Optional[Fabric] = getattr(self, "fabric", None) or getattr(
            self.scorer, "fabric", None
        )
        with fabric.autocast() if fabric else contextlib.nullcontext():
            if issubclass(scorer_type, AbstractModuleScorer):
                batch_items = inputs["records"]
                tokenized_queries = inputs.get("tokenized_queries")
                doc_ids = inputs.get("doc_ids")
                doc_token_embeddings = inputs.get("doc_token_embeddings")
                if doc_token_embeddings is None and doc_ids:
                    doc_token_embeddings = self.index.get_document_tokens(doc_ids)

                if doc_token_embeddings is None:
                    raise RuntimeError(
                        f"MicePlaidRetrieverv2 could not obtain document token embeddings "
                        f"for batch of size {len(batch_items)}"
                    )

                doc_token_embeddings = [
                    t.to(self.scorer.device, non_blocking=True)
                    for t in doc_token_embeddings
                ]
                return (
                    self.scorer(
                        batch_items,
                        tokenized=MICETokenizedTexts(
                            tokenized_q=tokenized_queries, tokenized_docs=None
                        ),
                        doc_hidden_states=doc_token_embeddings,
                    )
                    .cpu()
                    .float()
                    .numpy()
                )
            return None
