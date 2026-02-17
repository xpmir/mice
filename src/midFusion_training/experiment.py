"""One experiment to rule them all, merge all possible configurations here"""

import logging
from functools import partial

from experimaestro import setmeta, tagspath
from experimaestro.launcherfinder import find_launcher

import numpy as np
import pandas as pd
import xpmir.interfaces.anserini as anserini
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
from xpm_torch.batchers import PowerAdaptativeBatcher

from xpm_torch.trainers.distillation import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpm_torch.trainers.pairwise import PairwiseTrainer, PointwiseCrossEntropyLoss
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.validation import AggregatorValidationListener, ValidationListener
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from models.midFusion_modeling import (
    EttinMidFusionCrossEncoder,
    MiniLMMidFusionCrossEncoder,
)
from ir_training.experiment import build_scorer_model as build_Franken_scorer_model
from configuration import Layer_params, Losses, MidFusionCE_Finetuning, PoolingMethod
from tests import build_tests, minified_tests, nfcorpus_validation_dataset, paper_tests
from stats import run_statistical_tests
logging.basicConfig(level=logging.INFO)

# TODO: Use xpm_torch Batcher
# TODO: Use xpm_torch Experiment handler


def get_model_based_retrievers(cfg: MidFusionCE_Finetuning):
    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=xpmir_PowerAdaptativeBatcher.C(),
        device=cfg.device,
    )  #: Model-based retrievers

    return model_based_retrievers


def build_trainer(cfg: MidFusionCE_Finetuning) -> LossTrainer:
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


def build_MICE_scorer(
    hf_id: str,
    drop_layer: int,
    merge_layer: int,
    cfg: MidFusionCE_Finetuning,
    **kwargs,
) -> MiniLMMidFusionCrossEncoder:
    """Builds the MidFusionCrossEncoder model according to the given configuration
    Args:
        hf_id (str): HuggingFace model identifier
        drop_layer (int): Layer at which to drop backbone layers
        merge_layer (int): Layer at which to merge the two branches
        cfg (MidFusionCE_Finetuning): Configuration object
        **kwargs: Additional keyword arguments for model instantiation
    Returns:
    """

    logging.info(f"Building Mid_Fusion scorer model from HF id {hf_id}..")
    kwargs |= {
        "hf_id": hf_id,
        "merge_layer": merge_layer,
        "drop_layer": drop_layer,
        "use_self_attention": cfg.use_self_attention,
        "random_top_layers": cfg.random_top_layers,
    }
    try:
        if "ettin" in hf_id.lower():
            try:
                pooling_method = PoolingMethod(cfg.pooling_method)
            except ValueError:
                raise ValueError(
                    f"Unknown pooling method: {cfg.pooling_method}. Accepted values are: {[e.value for e in PoolingMethod]}"
                )
            scorer_cls = EttinMidFusionCrossEncoder
            kwargs |= {
                "pooling_method": pooling_method.value,
            }
        elif "minilm" in hf_id.lower() or "bert-base" in hf_id.lower():
            scorer_cls = MiniLMMidFusionCrossEncoder
        else:
            raise NotImplementedError(
                f"Unsupported HF model ID for scorer instantiation: {hf_id}"
            )

        scorer_model = scorer_cls.C(**kwargs)

    except Exception as e:
        logging.error(f"Error building scorer model: {e}")
        raise e

    return scorer_model


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: MidFusionCE_Finetuning):
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

    ## grid search params + baseline (no drop = 0)
    drop_layer_params = Layer_params.from_any(cfg.drop_layer).get_content_as_list()
    # if cfg.compare_with_baseline:
    #     drop_layer_params = [0] + drop_layer_params
    # -> baseline is MiniLM vanialla now 

    compress_dim_params = Layer_params.from_any(cfg.compress_dim).get_content_as_list()
    logging.info(f"Compress dim param range: {compress_dim_params}")
    logging.info(f"Drop layer param range: {drop_layer_params}")

    if "minilm" in cfg.base.lower():
        backbone = "miniLM"
    elif "ettin" in cfg.base:
        backbone = "ettin"
    else:
        backbone = ""

    def train_and_evaluate(scorer_model) -> None:

        logging.info(f"Train and evaluate model: {scorer_model}: {tagspath(scorer_model)} ")
        # The validation listener evaluates the full retriever
        # (retriever + scorer) and keep the best performing model
        # on the validation set
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

        hooks = [
            # setmeta(DistributedHook.C(models=[scorer_model]), True),
            setmeta(GradientLogHook.C(), True),
        ]

        if cfg.learner.max_grad_norm > 0:
            gradient_clipping_hook = GradientClippingHook.C(
                max_norm=cfg.learner.max_grad_norm
            )
            hooks.append(gradient_clipping_hook)

        # The learner trains the model
        learner = Learner.C(
            # Misc settings
            random=random,
            # How to train the model
            trainer=ce_trainer,
            # The model to train
            model=scorer_model,
            # Optimization settings
            steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
            optimizers=cfg.learner.optimization.optimizer,
            max_epochs=cfg.learner.optimization.max_epochs,
            checkpoint_interval=cfg.learner.checkpoint_interval,
            # The listeners (here, for validation)
            listeners=[validation, validation_zs, aggregator_validation],
            # The hook used for evaluation
            hooks=hooks,
            # fabric settings
            strategy=cfg.learner.strategy,
            precision=cfg.learner.precision,
            accelerator=cfg.learner.accelerator,
        )

        # Submit job and link
        outputs = learner.submit(launcher=launcher_learner)

        # Evaluate the neural model on test collections
        for metric_name in validation.monitored():
            load_model = outputs.listeners[validation.id][metric_name]
            # load_model = outputs.checkpoints[200]
            tests.evaluate_retriever(
                partial(
                    model_based_retrievers,
                    scorer=scorer_model,
                    retrievers=test_retrievers,
                    device=device,
                ),
                launcher_evaluate,
                model_id=f"{tagspath(scorer_model)}-{metric_name}-seed{seed}",
                init_tasks=[load_model],
            )

        # this links the tensorboard run dir to in the xp/results/run folder, so that we can access it easily.
        # the linking works only if the task was generated and scheduled by experimaestro, so that the learner.logpath is set.
        helper.tensorboard_service.add(learner, learner.logpath)


    ce_trainer: LossTrainer = build_trainer(cfg)

    for i in range(cfg.nb_repetitions):
        seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
        random = Random.C(seed=seed).tag("seed", seed)
        
        if cfg.compare_with_baseline:
            logging.info("Comparing with baseline models enabled.")
            #Build Baseline model (no drop, no compression)
            scorer_model = build_Franken_scorer_model(
                cfg,
                attn_patches=[],
            )
            scorer_model.tag("scorer", f"baseline-vanilla-{backbone}")

            train_and_evaluate(scorer_model)
    
        ### TRAINING MICE MODELS WITH DIFFERENT PARAMS
        for drop_layer in drop_layer_params:
            for compress_dim in compress_dim_params:

                scorer_model = (
                    build_MICE_scorer(
                        hf_id=cfg.base,
                        drop_layer=drop_layer,
                        merge_layer=cfg.merge_layer,
                        cfg=cfg,
                        freeze_base=cfg.freeze_base,
                        compress_dim=compress_dim,
                    )
                    .tag("drop", drop_layer)
                    .tag("merge", cfg.merge_layer)
                )
                # Custom tagging for compression and self-attention
                if len(compress_dim_params) > 1 or compress_dim > 1:
                    scorer_model.tag("compress", compress_dim)
                if not cfg.use_self_attention:
                    scorer_model.tag("SelfAttn", cfg.use_self_attention)

                #finally tag the model with scorer name - unique per parameter set
                scorer_model.tag("scorer", f"{backbone}-{tagspath(scorer_model)}")
                
                train_and_evaluate(scorer_model)
                ###  add baseline depending on xp params
                ###  not considering these as baselines anymore
                # if (len(drop_layer_params) > 1 and drop_layer == 0) or (
                #     len(compress_dim_params) > 1 and int(compress_dim) == 1
                # ):
                #     model_id = "baseline-" + model_id

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    df = tests.to_dataframe()
    metric_cols = [("metric", "AP"), ("metric", "RR@10"), ("metric", "nDCG@10")]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")
    df_grouped = (
        df.groupby(
            ["dataset", ("tag", "first_stage"), ("tag", "scorer")],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "var"])
        .reset_index()
    )
    logging.info(df_grouped)

    # save results
    if not helper.xp.resultspath.exists():
        helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

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
