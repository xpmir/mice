
"""One experiment to rule them all, merge all possible configurations here"""

import logging
from functools import partial

from experimaestro.launcherfinder import find_launcher

import pandas as pd
from transformers import AutoConfig
import xpmir.interfaces.anserini as anserini
from xpmir.experiments.ir import IRExperimentHelper, ir_experiment
from xpmir.rankers import scorer_retriever
from xpmir.neural.huggingface import HFCrossScorer
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import (
    HFTokenizerAdapter,
    HFTokenizer,
)
from xpmir.text.huggingface.base import HFMaskedLanguageModel
from xpmir.text.adapters import TopicTextConverter
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndexBuilder,
    Sparse2BMPConverter,
)
from xpmir.rankers import Documents, Retriever, document_cache
from xpmir.learning.devices import BestDevice

from xpmir.neural.splade import MaxAggregation, SpladeTextEncoderV2

# PROBLEM - may need two different batcher class here..
from xpmir.learning.batchers import (
    PowerAdaptativeBatcher as xpmir_PowerAdaptativeBatcher,
)

from tests import minified_tests, paper_tests
from models.franken import AttentionPatch, FrankenCrossScorer
from models.mask_scorer import HFMaskedEttinCrossScorer, HFMaskedMiniLMCrossScorer, MAX_SEQ_LEN
from format import dataframe_to_latex
from configuration import *
from xpmir.papers.helpers import NeuralIRExperiment


logging.basicConfig(level=logging.INFO)

TEST_CORPUS = [
    "irds.msmarco-passage.documents",
    # "irds.msmarco-passage-v2.documents",
    "irds.beir.webis-touche2020.v2",
    "irds.beir.fiqa.test",
    "irds.beir.nfcorpus.test",
    "irds.beir.scifact.test",
]


@configuration()
class MaskedBaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)

    evaluation: Evaluation = Factory(Evaluation)

    scorers_hf_id: List[str] = ["cross-encoder/ms-marco-MiniLM-L12-v2"]

    attn_patches: List[Attn_patch] = []    

    retrievers_hf_id: List[str] = [""]

    test_max_topics: int = 0
    """Development test size (0 to leave it like this)"""


def get_model_based_retrievers(cfg: FrankenCE_Finetuning):
    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=xpmir_PowerAdaptativeBatcher.C(),
        device=cfg.device,
    )  #: Model-based retrievers

    return model_based_retrievers


def build_scorer_model(hf_id: str, cfg: FrankenCE_Finetuning, use_attn_patches: bool, tag:str = "scorer_model") -> FrankenCrossScorer:
    """Builds the FrankenCrossScorer model according to the given configuration"""

    logging.info(f"Building scorer model from base: {hf_id}..")
    
    attention_patches = []
    if use_attn_patches and len(cfg.attn_patches) > 0:
        for attn_patch_cfg in cfg.attn_patches:
            attention_patches.append(
                AttentionPatch.C(
                    mask_attention_from=attn_patch_cfg.mask_attention_from,
                    mask_attention_to=attn_patch_cfg.mask_attention_to,
                    start_layer=attn_patch_cfg.start_layer,
                    end_layer=attn_patch_cfg.end_layer,
                )
            )
        logging.info(f"Using attention patches:")
        for ap in attention_patches:
            logging.info(f" - from {ap.mask_attention_from} to {ap.mask_attention_to} in layers {ap.start_layer} to {ap.end_layer}")
    else:
        logging.info("No attention patches used.")

    try:
        if "ettin" in hf_id.lower():
            scorer_cls = HFMaskedEttinCrossScorer
        elif "minilm" in hf_id.lower() or "bert-base" in hf_id.lower():
            scorer_cls = HFMaskedMiniLMCrossScorer
        else:
            raise NotImplementedError(f"Unsupported HF model ID for scorer instantiation: {hf_id}")

        scorer_model = FrankenCrossScorer.C(
                    scorer=scorer_cls.C(hf_id=hf_id),
                    attention_patches=attention_patches,
                ).tag("scorer", tag) 
    except Exception as e:
        logging.error(f"Error building scorer model: {e}")
        raise e

    return scorer_model


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: MaskedBaselinesConfig):
    """MiniLM-v2 model training"""
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_bmp = find_launcher(cfg.indexation.sparse2bmp_requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)

    #: Model-based retrievers
    device = cfg.device
    if cfg.evaluation.all_datasets:
        tests = paper_tests(cfg.evaluation.test_max_topics)
    else:
        tests = minified_tests(cfg.evaluation.test_max_topics)

    # Setup indices and validation/test base retrievers
    model_based_retrievers = get_model_based_retrievers(cfg)

    ## FIRST STAGE RETRIEVERS
    if retriever_id := cfg.retrievers_hf_id[0] != "":
        # We don't use BM25, but a given sparse retriever
        tokenizer = HFTokenizer.C(model_id=retriever_id)
        splade_encoder = SpladeTextEncoderV2.C(
            tokenizer=HFTokenizerAdapter.C(
                tokenizer=tokenizer, converter=TopicTextConverter.C()
            ),
            encoder=HFMaskedLanguageModel.from_pretrained_id(retriever_id),
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
                batcher=xpmir_PowerAdaptativeBatcher.C(),
                encoder=splade_encoder,
                device=device,
                documents=documents,
                ordered_index=False,
                max_docs=cfg.indexation.max_indexed,
            ).submit(launcher=launcher_index)

            # Just submit the convertion for now
            Sparse2BMPConverter.C(
                index=index, block_size=32, compress_range=True
            ).submit(launcher=launcher_bmp)

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
                    device=device,
                )
                .tag("model", name)
                .tag("data", documents.id)
            )

        first_stage_retriever = partial(splade_retriever, retriever_tag, splade_encoder)
    else:
        base_model = BM25.C()

        def bm25_retriever(name, documents: Documents) -> Retriever.C:        
            return (
                anserini.AnseriniRetriever.C(
                    k=cfg.retrieval.k,
                    model=base_model,
                    index=anserini.index_builder(launcher=launcher_index)(documents),
                    store=documents,
                )
                .tag("model", name)
                .tag("data", documents.id)
            )

        retriever_tag = "bm25"
        first_stage_retriever = partial(bm25_retriever, retriever_tag)

    # evaluate base retrievers alone
    tests.evaluate_retriever(
        first_stage_retriever,
        launcher=launcher_evaluate,
    )
    
    check_HF_vs_vanilla = False
    scorer_models = []

    for hf_id in cfg.scorers_hf_id:
        hf_config = AutoConfig.from_pretrained(hf_id)

        # First eval with xpmir CrossScorer class 
        # to check that Franken model with no patches is equivalent to HF CrossScorer
        scorer_models.append(
            HFCrossScorer.C(
                hf_id=hf_id,
                max_length=min(MAX_SEQ_LEN, hf_config.max_position_embeddings),
            ).tag("scorer", "raw_" + hf_id)
        )

        # Then our Custom Backbone
        for use_attn_patches in ([True, False] if check_HF_vs_vanilla else [True]):
            model_tag = ("masked_" if use_attn_patches else "vanilla_") + hf_id
            # Build the model
            scorer_models.append(
                build_scorer_model(
                    hf_id,
                    cfg, 
                    use_attn_patches= use_attn_patches,
                    tag = model_tag
                )
            )

    for scorer_model in scorer_models:
        # Evaluate the neural model on test collections
        # load_model = outputs.checkpoints[200]
        print(scorer_model)
        tests.evaluate_retriever(
            partial(
                model_based_retrievers,
                scorer=scorer_model,
                retrievers=first_stage_retriever,
                device=device,
            ),
            launcher_evaluate,
        )

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    df = tests.to_dataframe()
    metric_cols = [("metric", "AP"), ("metric", "RR@10"), ("metric", "nDCG@10")]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")
    df_grouped = (
        df.groupby(
            ["dataset", ("tag", "model"), ("tag", "scorer")],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    logging.info(df_grouped)
    #save the results in a csv file
    if not helper.xp.resultspath.exists():
        helper.xp.resultspath.mkdir(parents=True, exist_ok=True)
    output_file = helper.xp.resultspath / "results.csv"
    df_grouped.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

    # # Generate and save LaTeX table
    latex_table = dataframe_to_latex(df_grouped, caption="Evaluation Results", label="tab:eval_results")
    latex_output_file = helper.xp.resultspath / "results.tex"
    with open(latex_output_file, "w") as f:
        f.write(latex_table)
    logging.info(f"LaTeX table saved to {latex_output_file}")
