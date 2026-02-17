"""One experiment to rule them all, merge all possible configurations here"""

import logging
from functools import partial
from typing import List, Tuple

from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher

import numpy as np
import pandas as pd
import xpmir.interfaces.anserini as anserini
from xpmir.experiments.ir import IRExperimentHelper, ir_experiment
from xpmir.learning.optim import GradientLogHook, GradientClippingHook
from xpmir.papers.helpers.samplers import (
    msmarco_v1_docpairs_efficient_sampler,
    msmarco_v1_validation_dataset,
    prepare_collection,
    msmarco_hofstaetter_ensemble_hard_negatives,
)
from xpmir.rankers import scorer_retriever
from xpmir.rankers.standard import BM25, Model
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
from xpmir.learning import Random

from xpmir.neural.splade import MaxAggregation, SpladeTextEncoderV2

# PROBLEM - may need two different batcher class here..
from xpmir.learning.batchers import (
    PowerAdaptativeBatcher as xpmir_PowerAdaptativeBatcher,
)
from format import dataframe_to_latex
from stats import run_statistical_tests
from xpm_torch.batchers import PowerAdaptativeBatcher

from xpm_torch.trainers.distillation import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpm_torch.trainers.pairwise import PairwiseTrainer, PointwiseCrossEntropyLoss
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.validation import AggregatorValidationListener, ValidationListener

from models.franken import FrankenCrossScorer, AttentionPatch
from models.mask_scorer import HFMaskedEttinCrossScorer, HFMaskedMiniLMCrossScorer

from configuration import (
    Attn_patch,
    Layer_params,
    Losses,
    FrankenCE_Finetuning,
    PoolingMethod,
)
from tests import build_tests, nfcorpus_validation_dataset


logging.basicConfig(level=logging.INFO)

TEST_CORPUS = [
    "irds.msmarco-passage.documents",
    # "irds.msmarco-passage-v2.documents",
    "irds.beir.webis-touche2020.v2",
    "irds.beir.fiqa.test",
    "irds.beir.nfcorpus.test",
    "irds.beir.scifact.test",
]
target_dir = "/home/vast/franken_minilm/results/"


def get_model_based_retrievers(cfg: FrankenCE_Finetuning):
    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=xpmir_PowerAdaptativeBatcher.C(),
        device=cfg.device,
    )  #: Model-based retrievers

    return model_based_retrievers


def build_grid_search_attention_patches(attn_patches_cfg: list[Attn_patch]):
    """Parse the attention patches from the configuration. Importantly, this method also parses the
    attention patches that will be applied in a grid search manner (i.e., those with range of layers).

    Important: At the moment, the code doesn't work for more than one grid search at the same time.
    """

    overall_attention_patches = (
        []
    )  # Store non-grid patches, under the form of list of AttentionPatch
    grid_search_attention_patches = (
        []
    )  # Stores grid search patches, under the form of a list (if multiple grid search are performed at the same time) of list of AttentionPatch
    for attn_patch_cfg in attn_patches_cfg:
        parsed_attention_patches = []
        slp = Layer_params.from_any(attn_patch_cfg.start_layer, default=0)
        slp._validate()
        elp = Layer_params.from_any(attn_patch_cfg.end_layer, default=1)
        elp._validate()

        start_layers = (
            range(slp.get_content()[0], slp.get_content()[1])
            if isinstance(slp.get_content(), tuple)
            else (slp.get_content(),)
        )

        end_layers = (
            range(elp.get_content()[0], elp.get_content()[1])
            if isinstance(elp.get_content(), tuple)
            else (elp.get_content(),)
        )
        for start_layer in start_layers:
            for end_layer in end_layers:
                parsed_attention_patches.append(
                    AttentionPatch.C(
                        mask_attention_from=attn_patch_cfg.mask_attention_from,
                        mask_attention_to=attn_patch_cfg.mask_attention_to,
                        start_layer=start_layer,
                        end_layer=end_layer,
                    )
                )

        if len(parsed_attention_patches) == 1:
            overall_attention_patches.append(parsed_attention_patches[0])
        else:
            grid_search_attention_patches.extend(parsed_attention_patches)

    return overall_attention_patches, grid_search_attention_patches


def build_trainer(cfg: FrankenCE_Finetuning) -> LossTrainer:
    try:
        loss_member = Losses(cfg.learner.loss)
    except ValueError:
        raise ValueError(
            f"Unknown loss function: {cfg.learner.loss}. Accepted values are: {[e.value for e in Losses]}"
        )

    if loss_member is Losses.marginMSE:
        # define the trainer for monomlm
        return DistillationPairwiseTrainer.C(
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
        )

    # TODO: name properly the BCE loss function in the configuration as well
    elif loss_member is Losses.PointWiseMSE:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        return PairwiseTrainer.C(
            lossfn=PointwiseCrossEntropyLoss.C(),
            sampler=msmarco_v1_docpairs_efficient_sampler(
                sample_rate=cfg.learner.sample_rate,
                sample_max=cfg.learner.sample_max,
                launcher=launcher_preprocessing,
            ),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learner.optimization.batch_size,
        )

    else:
        raise NotImplementedError(
            f"Loss function {cfg.learner.loss} is not implemented yet."
        )


def build_scorer_model(
    cfg: FrankenCE_Finetuning, attn_patches: List[AttentionPatch], suffix: str = ""
) -> FrankenCrossScorer:
    """Builds the FrankenCrossScorer model and apply to it a list of attention patches"""

    hf_id = cfg.base
    logging.info(f"Building scorer model from base: {hf_id}.")

    logging.info(f"Using attention patches:")
    for ap in attn_patches:
        logging.info(
            f" - from {ap.mask_attention_from} to {ap.mask_attention_to} in layers {ap.start_layer} to {ap.end_layer}"
        )
    try:
        if "ettin" in hf_id.lower():
            try:
                pooling_method = PoolingMethod(cfg.pooling_method)
            except ValueError:
                raise ValueError(
                    f"Unknown pooling method: {cfg.pooling_method}. Accepted values are: {[e.value for e in PoolingMethod]}"
                )
            scorer_cls = HFMaskedEttinCrossScorer
            kwargs = {"hf_id": hf_id, "pooling_method": pooling_method.value}
        elif "minilm" in hf_id.lower() or "bert-base" in hf_id.lower():
            scorer_cls = HFMaskedMiniLMCrossScorer
            kwargs = {
                "hf_id": hf_id,
            }
        else:
            raise NotImplementedError(
                f"Unsupported HF model ID for scorer instantiation: {hf_id}"
            )
        scorer_model = (
            FrankenCrossScorer.C(
                scorer=scorer_cls.C(**kwargs),
                attention_patches=attn_patches,
            )
            .tag("scorer", cfg.id)
            .tag("grid_search", suffix if suffix else "no_grid_search")
        )

    except Exception as e:
        logging.error(f"Error building scorer model: {e}")
        raise e

    return scorer_model


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: FrankenCE_Finetuning):
    """MiniLM-v2 model training"""
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_bmp = find_launcher(cfg.indexation.sparse2bmp_requirements)
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    #: Model-based retrievers
    device = cfg.device
    train_documents = prepare_collection("irds.msmarco-passage.documents")

    ds_val = msmarco_v1_validation_dataset(
        cfg.validation, launcher=launcher_preprocessing
    )

    val_documents = prepare_collection("irds.beir.nfcorpus.documents")
    ds_val_zs = nfcorpus_validation_dataset(
        cfg.validation, launcher=launcher_preprocessing
    )
    
    tests = build_tests(cfg.evaluation)
    # Setup indices and validation/test base retrievers
    model_based_retrievers = get_model_based_retrievers(cfg)

    ## FIRST STAGE RETRIEVERS
    if cfg.retriever:
        # We don't use BM25, but a given sparse retriever
        tokenizer = HFTokenizer.C(model_id=cfg.retriever)
        splade_encoder = SpladeTextEncoderV2.C(
            tokenizer=HFTokenizerAdapter.C(
                tokenizer=tokenizer, converter=TopicTextConverter.C()
            ),
            encoder=HFMaskedLanguageModel.from_pretrained_id(cfg.retriever),
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
                .tag("first_stage", name)
                .tag("data", documents.id)
            )

        def splade_val_retrievers(
            documents: Documents,
            *,
            model: Model = None,
        ) -> Retriever.C:
            return SparseRetriever.C(
                index=splade_index()(documents),
                topk=cfg.learner.validation_top_k,
                batchsize=1,
                encoder=model,
                in_memory=True,
                device=device,
            )

        retriever_tag = cfg.retriever
        # Caches the Splade index task for a document collection
        val_retrievers = partial(
            splade_val_retrievers,
            model=splade_encoder,
        )
        val_retrievers_zs = val_retrievers

        test_retrievers = partial(splade_retriever, retriever_tag, splade_encoder)
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
                .tag("first_stage", name)
                .tag("data", documents.id)
            )

        retrievers = partial(
            anserini.retriever,
            anserini.index_builder(launcher=launcher_index),
            model=base_model,
        )

        val_retrievers = partial(
            retrievers, store=train_documents, k=cfg.learner.validation_top_k
        )
        val_retrievers_zs = partial(
            retrievers, store=val_documents, k=cfg.learner.validation_top_k
        )

        retriever_tag = "bm25"
        test_retrievers = partial(bm25_retriever, retriever_tag)

    # evaluate base retrievers alone
    tests.evaluate_retriever(
        test_retrievers,
        launcher=launcher_evaluate,
    )

    attention_patches, grid_search_attention_patches = (
        build_grid_search_attention_patches(cfg.attn_patches)
    )

    ### Create the list of lists of attention patches to use in the xp based on the grid search attention patches ###
    list_of_attention_patches_dicts = []
    if len(grid_search_attention_patches) > 0:
        logging.info("Grid search over attention patches will be performed.")
        for grid_search_attention_patch in grid_search_attention_patches:
            if grid_search_attention_patch is None:
                continue
            logging.info(
                " - from %s to %s in layers %s to %s",
                grid_search_attention_patch.mask_attention_from,
                grid_search_attention_patch.mask_attention_to,
                grid_search_attention_patch.start_layer,
                grid_search_attention_patch.end_layer,
            )

            tmp = attention_patches
            suffix = f"attn_{grid_search_attention_patch.mask_attention_from}2{grid_search_attention_patch.mask_attention_to}_l{grid_search_attention_patch.start_layer}-{grid_search_attention_patch.end_layer}"
            if (
                grid_search_attention_patch.start_layer
                == grid_search_attention_patch.end_layer
            ):
                logging.warning(
                    "Skipping attention patch with identical start and end layers: %d",
                    grid_search_attention_patch.start_layer,
                )
                list_of_attention_patches_dicts.append(
                    {"attention_patches": tmp, "suffix": suffix}
                )
            else:
                list_of_attention_patches_dicts.append(
                    {
                        "attention_patches": tmp + [grid_search_attention_patch],
                        "suffix": suffix,
                    }
                )
    else:
        logging.info("No grid search over attention patches will be performed.")
        list_of_attention_patches_dicts.append(
            {"attention_patches": attention_patches, "suffix": ""}
        )

    ### TRAINING CROSS ENCODER
    for i in range(cfg.nb_repetitions):
        seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
        random = Random.C(seed=seed).tag("seed", seed)

        ce_trainer: LossTrainer = build_trainer(cfg)

        def train_and_evaluate(scorer_model: FrankenCrossScorer, suffix: str = ""):
            # validation listeners
            validation = ValidationListener.C(
                id="bestval",
                dataset=ds_val,
                retriever=model_based_retrievers(
                    documents=train_documents,
                    retrievers=val_retrievers,
                    scorer=scorer_model,
                    device=device,
                ).tag("retriever", retriever_tag),
                validation_interval=cfg.learner.validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG": False},
            )

            validation_zs = ValidationListener.C(
                id="bestval_zs",
                dataset=ds_val_zs,
                retriever=model_based_retrievers(
                    documents=val_documents,
                    retrievers=val_retrievers_zs,
                    scorer=scorer_model,
                    device=device,
                ).tag("retriever", retriever_tag),
                validation_interval=cfg.learner.validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG": False},
            )

            aggregator_validation = AggregatorValidationListener.C(
                listeners=[validation, validation_zs],
                id="aggregated_validation",
                validation_interval=cfg.learner.validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG": False},
            )

            hooks = [setmeta(GradientLogHook.C(), True)]
            if cfg.learner.max_grad_norm > 0:
                hooks.append(GradientClippingHook.C(max_norm=cfg.learner.max_grad_norm))

            learner = Learner.C(
                random=random,
                trainer=ce_trainer,
                model=scorer_model,
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=cfg.learner.optimization.optimizer,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                listeners=[validation, validation_zs, aggregator_validation],
                hooks=hooks,
                strategy=cfg.learner.strategy,
                precision=cfg.learner.precision,
                accelerator=cfg.learner.accelerator,
            ).tag("grid_search", suffix if suffix else "no_grid_search")

            outputs = learner.submit(launcher=launcher_learner)
            scorer_id = scorer_model.tags().get('scorer', 'no_id')
            # evaluate saved models produced by validation
            for metric_name in validation.monitored():
                load_model = outputs.listeners[validation.id][metric_name]
                tests.evaluate_retriever(
                    partial(
                        model_based_retrievers,
                        scorer=scorer_model,
                        retrievers=test_retrievers,
                        device=device,
                    ),
                    launcher_evaluate,
                    model_id=f"{scorer_id}-{metric_name}-{seed}",
                    init_tasks=[load_model],
                )

            helper.tensorboard_service.add(learner, learner.logpath)

        # Build the model
        for i, dict in enumerate(list_of_attention_patches_dicts):
            scorer_id = f"{cfg.id}-{i}"
            if i ==0: 
                logging.info(f"Baseline run (First patch): {dict}")
                scorer_id = f"baseline-{scorer_id}"

            scorer_model = build_scorer_model(
                cfg, attn_patches=dict["attention_patches"], suffix=dict["suffix"]
            ).tag("scorer", scorer_id)
            
            train_and_evaluate(scorer_model, suffix=dict["suffix"])

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    df = tests.to_dataframe()

    metric_cols = [("metric", "AP"), ("metric", "RR@10"), ("metric", "nDCG@10")]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")
    df_grouped = (
        df.groupby(
            [
                "dataset",
                ("tag", "first_stage"),
                ("tag", "scorer"),
                ("tag", "grid_search"),
            ],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    logging.info(df_grouped)

    # save results
    if not helper.xp.resultspath.exists():
        helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

    logging.info("per_model:")
    logging.info(tests.per_model.keys())

    if cfg.compare_with_baseline:
        logging.info("Loading detailed results for each runs, across each setups")
        detailed_df = run_statistical_tests(
            tests.per_model, nb_repetitions=cfg.nb_repetitions
        )

        logging.info("Detailed results across each setups:")
        logging.info(detailed_df)

        detailed_output_file = (
            helper.xp.resultspath / "statistical_significance_results.csv"
        )
        detailed_df.to_csv(detailed_output_file, index=True)
        logging.info(
            f"Statistical significance results saved to {detailed_output_file}"
        )

    output_file = helper.xp.resultspath / "results.csv"
    df_grouped.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

    # Generate and save LaTeX table
    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results",
        label="tab:eval_results",
        sig_df=detailed_df if cfg.compare_with_baseline else None,
    )
    latex_output_file = helper.xp.resultspath / "results.tex"
    with open(latex_output_file, "w") as f:
        f.write(latex_table)
    logging.info(f"LaTeX table saved to {latex_output_file}")
