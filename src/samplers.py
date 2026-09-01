from functools import lru_cache
from typing import Union

from datamaestro import prepare_dataset
from datamaestro_ir.data import Documents, Adhoc

from xpmir.datasets.adapters import MemoryTopicStore
from xpmir.letor.distillation.samplers import (
    DistillationNegativesSampler,
    ListwiseDistillationSamplesTSVWithAnnotations,
)
from xpmir.letor.samplers.adapters import SamplerAdapter
from xpmir.letor.processors import StoreHydrator


from typing import Optional


@lru_cache
def prepare_collection(
    prepare_str: str, download: Optional[bool] = None
) -> Union[Documents, Adhoc]:
    """Prepare a dataset and caches the result"""
    return prepare_dataset(prepare_str, download=download)


@lru_cache
def msmarco_rankdistillm_sampled_colbert50(passages_per_query: int) -> SamplerAdapter:
    """Build a DistillationNegativesSampler
    - sampling negatives from the RankZephyr reranked ColBERTv2 top 50
    - using MS MARCO qrels."""
    unannotated = prepare_dataset(
        "com.github.webis-de.rank-distillm.rankzephyr.colbert10000.sampled50.annotated"
    )

    # Load qrels separately
    qrels = prepare_dataset("com.microsoft.msmarco.passage.train.qrels")

    # Create the annotated dataset on the fly
    train_ranks_distil = ListwiseDistillationSamplesTSVWithAnnotations.C(
        id="",
        qrels=qrels,
        top_k=unannotated.top_k,
        with_docid=unannotated.with_docid,
        with_queryid=unannotated.with_queryid,
        path=unannotated.path,
    )

    # Access to topic text
    train_topics = prepare_dataset("com.microsoft.msmarco.passage.train.queries")

    # Generate a sampler from the samples, hydrating with stores
    raw_sampler = DistillationNegativesSampler.C(
        samples=train_ranks_distil, passages_per_query=passages_per_query
    )
    hydrator = StoreHydrator.C(
        documentstore=prepare_collection("com.microsoft.msmarco.passage.documents"),
        querystore=MemoryTopicStore.C(topics=train_topics),
    )

    return SamplerAdapter.C(sampler=raw_sampler, processors=[hydrator])
