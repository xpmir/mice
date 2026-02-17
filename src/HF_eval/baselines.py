from typing import List
import logging
from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from attrs import Factory
from xpmir.papers import configuration

from tests import minified_tests, paper_tests
from configuration import *
from xpmir.papers.helpers import NeuralIRExperiment
from xpmir.neural.huggingface import HFCrossScorer

from transformers import AutoConfig

from functools import partial

from experimaestro.launcherfinder import find_launcher
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.rankers.standard import BM25, Model

import xpmir.interfaces.anserini as anserini
from datamaestro_text.data.ir import Documents
from xpmir.rankers import scorer_retriever, document_cache, Retriever
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.neural.splade import MaxAggregation, SpladeTextEncoderV2
from xpmir.text.huggingface.base import HFMaskedLanguageModel
from xpmir.text.adapters import TopicTextConverter
from xpmir.learning.devices import BestDevice
from xpmir.text.huggingface import (
    HFTokenizerAdapter,
    HFTokenizer
)
logging.basicConfig(level=logging.INFO)


@configuration()
class BaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)

    evaluation: Evaluation = Factory(Evaluation)

    scorers_hf_id: List[str] = ["cross-encoder/ms-marco-MiniLM-L12-v2"]
    
    retrievers_hf_id: List[str] = [""]

    test_max_topics: int = 0
    """Development test size (0 to leave it like this)"""


def build_splade_retriever_cfg(
        cfg: BaselinesConfig,
        retriever_hf_id: str,
        ):
    
    launcher_index = find_launcher(cfg.indexation.requirements)

    tokenizer = HFTokenizer.C(model_id=retriever_hf_id)
    splade_encoder = SpladeTextEncoderV2.C(
        tokenizer=HFTokenizerAdapter.C(
            tokenizer=tokenizer, converter=TopicTextConverter.C()
        ),
        encoder=HFMaskedLanguageModel.from_pretrained_id(retriever_hf_id),
        aggregation=MaxAggregation.C(),
        maxlen=256,
    )

    @document_cache
    def splade_index(documents: Documents):
        device = BestDevice.C()

        logging.info(
            "Indexing %s (%s documents) with %s",
            documents.id,
            documents.count,
            launcher_index,
        )

        index = SparseRetrieverIndexBuilder.C(
            batch_size=cfg.indexation.batch_size,
            batcher=PowerAdaptativeBatcher.C(),
            encoder=splade_encoder,
            device=device,
            documents=documents,
            ordered_index=False,
            max_docs=cfg.indexation.max_indexed,
        ).submit(launcher=launcher_index)

        return index

    def splade_retriever(
        name,
        encoder,
        documents: Documents,
    ) -> Retriever.C:
        return (
            SparseRetriever.C(
                index=splade_index()(documents),
                topk=cfg.retrieval.k,
                batchsize=1,
                encoder=encoder,
                in_memory=False,
                device=cfg.device,
            )
            .tag("first-stage", name)
            .tag("data", documents.id)
        )
    
    def splade_val_retrievers(
        documents: Documents,
        *,
        model: Model = None,
    ) -> Retriever.C:
        return (
            SparseRetriever.C(
                index=splade_index()(documents),
                topk=cfg.learner.validation_top_k,
                batchsize=1,
                encoder=model,
                in_memory=True,
                device=cfg.device,
            )

        )


    # Caches the Splade index task for a document collection
    val_retrievers = partial(
        splade_val_retrievers, 
        model=splade_encoder,
    )

    return partial(splade_retriever, retriever_hf_id, splade_encoder)


@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: BaselinesConfig
) -> PaperResults:
    

    device = cfg.device

    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)
    if cfg.evaluation.all_datasets:
        tests = paper_tests(cfg.evaluation.test_max_topics)
    else:
        tests = minified_tests(cfg.evaluation.test_max_topics)

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher.C(),
        device=device,
    ) #: Model-based retrievers

    ### BM25 Retriever

    @document_cache
    def index_builder(documents: Documents):
        return anserini.IndexCollection.C(
            documents=documents,
        ).submit(launcher=launcher_index,)

    def bm25Retriever(
        name,
        model,
        documents: Documents,
    ) -> Retriever.C:
        return anserini.AnseriniRetriever.C(
            index=index_builder()(documents),
            model=model,
            k=cfg.retrieval.k,
            store=documents,
        ).tag("first-stage", name)


    ### Build the retrievers list 
    retrievers = [partial(bm25Retriever, "bm25", BM25.C())]
    
    #### SPLADE Retriever
    for retriever_hf_id in cfg.retrievers_hf_id:
        retrievers.append(
            build_splade_retriever_cfg(
                cfg,
                retriever_hf_id,
            )
        )
    
    for retriever in retrievers:

        # Eval First stage only
        tests.evaluate_retriever(
            retriever,
            launcher=launcher_evaluate,
            )
        logging.info(f"First stage only evaluation done for {retriever}")
        logging.info(f"Evaluating model-based retrievers {cfg.scorers_hf_id}")
        # Eval With cross-encoder
        for scorer_hf_id in cfg.scorers_hf_id:
            # evaluating the zero-shot ability

            config = AutoConfig.from_pretrained(scorer_hf_id)
            scorer = HFCrossScorer.C(
                hf_id=scorer_hf_id,
                max_length=config.max_position_embeddings,
            )

            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer.tag("model", scorer_hf_id),
                    retrievers=retriever,
                    device=device,
                ),
                launcher=launcher_evaluate,
                # model_id=f"{scorer_hf_id}", #need to be unique for each eval
                init_tasks=[],
            )



    return PaperResults(
        models={
            "minilm-zs-RR@10": scorer,
        }, 
        evaluations=tests,
        tb_logs=None, 
    )
