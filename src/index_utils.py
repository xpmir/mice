"""Utilities for index management."""

import logging
from xpmir.index.sparse import SparseRetrieverIndexBuilder
from xpmir.rankers import Documents

logger = logging.getLogger(__name__)

# cache the indexes
_indexes = {}


def get_splade_index(
    documents: Documents,
    splade_encoder,
    indexation_cfg,
    launcher_index,
    init_tasks: list = [],
):
    """Build an index for given documents, using a given Sparse retriever model
    Caches it to avoid submitting job twice.
    """

    index_cfg = SparseRetrieverIndexBuilder.C(
        batch_size=indexation_cfg.batch_size,
        # batcher=PowerAdaptativeBatcher.C(),
        encoder=splade_encoder,
        documents=documents,
        ordered_index=False,
        max_docs=indexation_cfg.max_indexed,
    )

    indexer_id = index_cfg.__identifier__()

    if indexer_id not in _indexes:
        logger.info(
            "Indexing %s (%s documents) with %s",
            documents.id,
            documents.count,
            launcher_index,
        )
        index = index_cfg.tag("index_documents", documents.id).submit(
            launcher=launcher_index, init_tasks=init_tasks
        )
        _indexes[indexer_id] = index
    else:
        index = _indexes[indexer_id]

    return index
