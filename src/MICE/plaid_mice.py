import os
from typing import Dict, List, Optional
from MICE.modeling.mice import MICETokenizedTexts, QueryDocInput
from datamaestro_ir.data.base import IDTextRecord, ScoredDocument
from experimaestro.core.arguments import Meta
from lightning_fabric import Fabric
import torch
import logging

from experimaestro import Param, field
from tqdm import tqdm
from xpm_torch.configuration import FabricConfiguration
from xpm_torch.datasets import ShardedIterableDataset
from xpmir.index.plaid import PlaidIndex
from xpmir.rankers.retriever import Retriever
from xpmir.rankers.scorer import RerankingInputs, TwoStageRetriever, reranking_collate
from xpmir.letor.records import PointwiseItem
from xpmir.rankers import AbstractModuleScorer
from xpm_torch.utils import to_device
from torchdata.stateful_dataloader import StatefulDataLoader

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


class PlaidReRankingDataset(ShardedIterableDataset):
    """A dataset that yields PointwiseItem records for re-ranking"""

    def __init__(self, queries: Dict[str, IDTextRecord], retriever: Retriever):
        super().__init__()
        self.queries = list(queries.items())
        self.retriever = retriever

    def iter_shard(self, shard_id: int, num_shards: int):
        import os

        rank = os.environ.get(
            "SLURM_PROCID", os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        )
        logger.debug(
            "ReRankingDataset (rank %s): shard %d/%d starting (total queries: %d)",
            rank,
            shard_id,
            num_shards,
            len(self.queries),
        )
        for i in range(shard_id, len(self.queries), num_shards):
            qid, query = self.queries[i]
            # Pull first-stage results on-the-fly to avoid materialising everything
            scored_docs = self.retriever.retrieve(query)
            for sd in scored_docs:
                yield PointwiseItem(query, sd.document, sd.score)


class MicePlaidRetrieverv2(TwoStageRetriever):
    index: Param[PlaidIndex]
    """The fast-plaid index to search."""

    def build_reranking_dataloader(self, queries: Dict[str, IDTextRecord]):
        """Builds a dataloader for re-ranking all documents for a set of queries
        Allows for efficient two-stage retrieval with cross-query batching on GPU when the scorer supports it (i.e. is an AbstractModuleScorer and batchsize > 0).
        """
        # We don't materialise everything, but iterate on the fly
        dataset = PlaidReRankingDataset(queries, self.retriever)

        # get underlying module if wrapped (e.g. Fabric)
        scorer = self.scorer.module if hasattr(self.scorer, "module") else self.scorer

        def tokenization_fn(
            records: List[IDTextRecord],
        ) -> Optional[MICETokenizedTexts]:
            doc_list: list = []
            queries_list: list = []
            for record in records:
                doc_list.append(record.document["id"])
                queries_list.append(record.topic)

            if issubclass(type(scorer), AbstractModuleScorer):
                return scorer.batch_tokenize(
                    queries_list, input_nature=QueryDocInput.QUERY
                ), doc_list
            return None

        def collate_fn(batch: List[PointwiseItem]) -> RerankingInputs:
            inputs = reranking_collate(batch)
            inputs["tokenized_queries"], inputs["doc_ids"] = tokenization_fn(
                inputs["records"]
            )
            return inputs

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=self.batchsize,
            num_workers=min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 4),
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn,
        )

        fabric: Fabric = getattr(self, "fabric", None)
        if fabric:
            dataloader = fabric.setup_dataloaders(dataloader)

        return dataloader

    @torch.no_grad()
    def retrieve_all(
        self, queries: Dict[str, IDTextRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        """Retrieves documents for all queries in an efficient two - stage fashion:
        - populate a `PointWiseItem` dataset with the documents from first stage
        - reranks them on the fly with the scorer with given batch size
        - if self.batchsize is 0, scores all documents from the same query at once (will cause OOM large top_k first stages)
        """
        # Scorer in evaluation mode
        self.scorer.eval()

        if self.batchsize == 0:
            # Fallback to per-query retrieval if no batchsize
            return super().retrieve_all(queries)

        # get underlying module if wrapped (e.g. Fabric)
        scorer = self.scorer.module if hasattr(self.scorer, "module") else self.scorer
        scorer_type = type(scorer)

        fabric: Fabric = getattr(self, "fabric", None)
        dataloader = self.build_reranking_dataloader(queries)

        # Process in batches
        scored_results = {qid: [] for qid in queries}
        seen_qids = set()

        disable_tqdm = fabric is not None and not fabric.is_global_zero

        # Calculate local total for the progress bar to reach 100% on rank 0
        total_to_process = len(queries)
        if fabric and fabric.world_size > 1:
            # Number of queries this specific rank (0) will handle
            total_to_process = len(
                range(fabric.global_rank, len(queries), fabric.world_size)
            )

        batch_size_info = (
            f"batch size {self.batchsize}"
            if issubclass(scorer_type, AbstractModuleScorer)
            else "rsv (one-by-one)"
        )
        logger.info(
            f"Rank {fabric.global_rank if fabric else 0}: Re-Ranking with '{scorer_type.__name__}' using {batch_size_info}..."
        )

        pbar = (
            tqdm(
                total=total_to_process,
                desc="Re-ranking",
                unit="query",
            )
            if not disable_tqdm
            else None
        )

        for inputs in dataloader:
            batch_items = inputs["records"]
            batch = inputs["batch"]
            tokenized_queries = inputs.get("tokenized_queries")
            doc_ids = inputs.get("doc_ids")
            doc_token_embeddings = [
                t.to(self.scorer.device)
                for t in self.index.get_document_tokens(doc_ids)
            ]

            if issubclass(scorer_type, AbstractModuleScorer):
                # Use scorer.forward if it's an AbstractModuleScorer to batch across queries

                scores = (
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

                for score, item in zip(scores, batch):
                    qid = item.topic["id"]
                    if qid not in seen_qids:
                        seen_qids.add(qid)
                        if pbar is not None:
                            pbar.update(1)
                    scored_results[qid].append(
                        to_device(
                            ScoredDocument(item.document, float(score.item())),
                            "cpu",
                        )
                    )
            else:
                # Fallback: group by query and use rsv (score one by one)
                by_query = {}
                for item in batch:
                    qid = item.topic["id"]
                    if qid not in seen_qids:
                        seen_qids.add(qid)
                        if pbar is not None:
                            pbar.update(1)
                    by_query.setdefault(qid, []).append(
                        to_device(
                            ScoredDocument(item.document, item.relevance),
                            "cpu",
                        )
                    )

                for qid, docs in by_query.items():
                    scored_results[qid].extend(self.scorer.rsv(queries[qid], docs))

        if pbar is not None:
            pbar.close()

        if fabric and fabric.world_size > 1:
            # Gather results from all GPUs
            # We use a list of tuples and filter empty results to reduce
            # communication overhead. Using torch.distributed directly for
            # objects is more robust than fabric.all_gather which is tensor-oriented.
            local_results = [
                (qid, docs) for qid, docs in scored_results.items() if docs
            ]
            gathered_data = [None] * fabric.world_size
            torch.dist.all_gather_object(gathered_data, local_results)

            if fabric.is_global_zero:
                # Merge them back into one master dictionary
                final_results = {qid: [] for qid in queries}
                for rank_results in gathered_data:
                    for qid, docs in rank_results:
                        final_results[qid].extend(docs)
                scored_results = final_results
            else:
                return {}

        # Sort and truncate
        for qid in scored_results:
            scored_results[qid].sort(reverse=True)
            if self.top_k:
                scored_results[qid] = scored_results[qid][: self.top_k]

        return scored_results
