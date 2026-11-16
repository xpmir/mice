"""
Standard Cross-Encoder Training Experiment.

This module provides the primary training pipeline for Cross-Encoders, supporting
first-stage retrieval (BM25, SPLADE) followed by neural re-ranking. It includes
support for multiple loss functions, multi-dataset validation, and
hyperparameter tuning via grid search.

"""

import logging
from functools import partial
import numpy as np

from experimaestro import setmeta, stop_tags, RunMode
from experimaestro.launcherfinder import find_launcher
from experimaestro.experiments.grid import generate_grid

from xpm_torch import Random
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.optim import GradientLogHook, GradientClippingHook
from xpm_torch.huggingface import prepare_hf_model

from xpmir.papers.results import PaperResults
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.rankers import scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory
from xpmir.neural.splade import splade_encoder_from_pretrained_hf

from retrievers import splade_retriever, bm25_retriever, plaid_retriever
from validations import ValidationSet
from configuration import CE_FineTuning, CheckpointSelection, ScorerType, Losses
from tests import build_tests
from format import loss_names, backbone_names_lower

from training_utils import (
    build_trainer,
    process_experiment_results,
    find_checkpoint,
    FreezeParameterHook,
)

from typing import Optional
from experimaestro import LightweightTask, Param
from xpmir.neural.sentence_transformers import (
    STCrossEncoder,
    STMultiVectorEncoder,
    st_multivector_scorer,
)
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderModelCardData
from sentence_transformers.models import Transformer, Pooling, Dense, LayerNorm
from torch import nn

from xp_utils import setup_logging

setup_logging(level=logging.INFO)

logger = logging.getLogger(__name__)


class InitDistilledSTCrossEncoder(LightweightTask):
    """Initializes the STCrossEncoder with a custom distillation architecture."""

    model: Param[STCrossEncoder]
    pref_attn_implementation: Param[Optional[str]] = None

    def execute(self):
        model = self.model
        # Initialize basic component properties without triggering the default CrossEncoder load
        super(STCrossEncoder, model).__initialize__()
        if model.tokenizer is not None:
            model.tokenizer.initialize()

        encoder_size = model.model_id.split("-")[-1]

        transformer = Transformer(
            model.model_id,
            max_seq_length=model.max_length,
            model_kwargs={"attn_implementation": self.pref_attn_implementation},
        )
        transformer.model.config.num_labels = 1
        embedding_dimension = transformer.get_embedding_dimension()

        pooling = Pooling(embedding_dimension=embedding_dimension, pooling_mode="cls")
        dense_inner = Dense(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
            activation_function=nn.GELU(),
            module_input_name="sentence_embedding",
            module_output_name="sentence_embedding",
        )
        norm = LayerNorm(dimension=embedding_dimension)
        dense_score = Dense(
            in_features=embedding_dimension,
            out_features=1,
            bias=True,
            activation_function=nn.Identity(),
            module_input_name="sentence_embedding",
            module_output_name="scores",
        )

        # Override the native CrossEncoder with our custom pipeline
        model.st_model = CrossEncoder(
            modules=[transformer, pooling, dense_inner, norm, dense_score],
            num_labels=1,
            activation_fn=nn.Identity(),
            model_card_data=CrossEncoderModelCardData(
                model_name=f"Ettin Reranker {encoder_size} distilled from mxbai-rerank-large-v2",
                language="en",
                license="apache-2.0",
            ),
        )
        # Mark as initialized so the framework doesn't try to call __initialize__ again later
        model._initialized = True


def build_STCrossEncoder(cfg) -> tuple[STCrossEncoder, list]:
    """Helper factory to build the distilled STCrossEncoder."""
    from xpmir.neural.huggingface import HFQueryDocTokenizer

    attn_impl = getattr(cfg, "pref_attn_implementation", "sdpa")
    tokenizer = HFQueryDocTokenizer.C(
        model_id=cfg.base,
        max_length=cfg.max_length,
        max_query_length=getattr(cfg, "max_query_length", None),
        max_doc_length=getattr(cfg, "max_doc_length", None),
        pref_attn_implementation=attn_impl,
    )
    scorer = STCrossEncoder.C(
        model_id=cfg.base,
        max_length=cfg.max_length,
        tokenizer=tokenizer,
        pref_attn_implementation=attn_impl,
    )
    init_task = InitDistilledSTCrossEncoder.C(
        model=scorer, pref_attn_implementation=attn_impl
    )
    return scorer, [init_task]


def build_STMultiVectorEncoder(cfg) -> tuple[STMultiVectorEncoder, list]:
    """Helper factory to build the STMultiVectorEncoder."""
    attn_impl = getattr(cfg, "pref_attn_implementation", None)
    query_maxlen = getattr(cfg, "max_query_length", None) or 32
    if query_maxlen > 128:
        logger.warning(
            f"max_query_length {query_maxlen} is unusually large for multi-vector encoder (ColBERT queries are typically 32 tokens). Clamping to 32."
        )
        query_maxlen = 32

    doc_maxlen = (
        getattr(cfg, "max_doc_length", None) or getattr(cfg, "max_length", None) or 300
    )
    if doc_maxlen > 512:
        logger.warning(
            f"max_doc_length {doc_maxlen} is unusually large for multi-vector encoder passages. Clamping to 300."
        )
        doc_maxlen = 512

    loss_val = getattr(cfg.learner, "loss", None) if hasattr(cfg, "learner") else None
    if isinstance(loss_val, Losses):
        loss_val = loss_val.value

    similarity_fn = getattr(cfg, "similarity_fn_name", None) or (
        "meanmaxsim"
        if loss_val
        in (
            Losses.MSE_mixedbread_large.value,
            Losses.pearson_mixedbread_large.value,
            Losses.MSE_lighton_multilingual.value,
        )
        else "maxsim"
    )

    # TODO: Remove explicit `dim` override once legacy runs with dim=128 are deprecated
    dim = cfg.plaid.dim
    if dim is not None:
        logger.info(
            f"MultiVector scorer: using explicit projection dimension dim={dim}"
        )

    scorer = st_multivector_scorer(
        model_id=cfg.base,
        dim=dim,
        query_maxlen=query_maxlen,
        doc_maxlen=doc_maxlen,
        pref_attn_implementation=attn_impl,
        similarity_fn_name=similarity_fn,
        query_expansion=None,
    )
    return scorer, []


def get_ce_name(model_tags: dict) -> str:
    """Creates the HF id from tags using formatting conventions."""
    loss = model_tags.get("loss", "")
    base = model_tags.get("base", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"
    base = backbone_names_lower.get(base, base).replace("/", "-")

    extra_parts = []
    if model_tags.get("freeze_LN"):
        extra_parts.append("freezeLN")
    if "batch_size" in model_tags:
        extra_parts.append(f"bs{model_tags['batch_size']}")
    if "checkpoint" in model_tags and str(model_tags["checkpoint"]) != "last":
        extra_parts.append(str(model_tags["checkpoint"]))
    if str(model_tags.get("plaid_retriever")).lower() in ("true", "1"):
        extra_parts.append("plaid")

    extra = f"-{'-'.join(extra_parts)}" if extra_parts else ""
    scorer_type = model_tags.get("scorer_type", ScorerType.stCrossEncoder.value)
    prefix = (
        "colbert"
        if scorer_type == ScorerType.stMultiVectorEncoder.value
        else "cross-encoder"
    )
    return f"{prefix}-{base}{loss}{extra}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: CE_FineTuning) -> PaperResults:
    launcher_index = find_launcher(cfg.indexation.requirements)
    long_reqs_index = getattr(cfg.indexation, "long_requirements", None)
    long_launcher_index = find_launcher(long_reqs_index) if long_reqs_index else None
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)

    long_reqs = getattr(cfg.retrieval, "long_requirements", None)
    long_launcher = find_launcher(long_reqs) if long_reqs else None

    tests = build_tests(
        cfg.evaluation,
        launcher=launcher_preprocessing,
        default_launcher=launcher_evaluate,
        long_launcher=long_launcher,
        long_evals=getattr(cfg.retrieval, "long_evals", None),
        blocking=cfg.preprocessing.blocking_download,
    )

    # cache the indexes
    learners = []
    all_weights = []

    def run_one_config(
        helper: LearningExperimentHelper,
        cfg: CE_FineTuning,
        grid_search_id: str,
        cfg_tags: dict,
    ):
        """Main process for Cross-encoder training"""

        if cfg.retriever:
            # We don't use BM25, but a given sparse retriever
            retriever_tag = cfg.retriever
            splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(
                cfg.retriever
            )

            # Caches the Splade index task for a document collection
            val_retrievers_factory = partial(
                splade_retriever,
                cfg,
                retriever_tag,
                splade_encoder,
                launcher_index=launcher_index,
                init_tasks=retriever_init_tasks,
                topk=cfg.learner.validation_top_k,
                in_memory=True,
            )

            effective_long_evals = getattr(
                cfg.indexation, "long_evals", None
            ) or getattr(cfg.retrieval, "long_evals", None)

            test_retrievers_factory = partial(
                splade_retriever,
                cfg,
                retriever_tag,
                splade_encoder,
                launcher_index=launcher_index,
                long_launcher_index=long_launcher_index,
                long_evals=effective_long_evals,
                topk=cfg.retrieval.k,
                init_tasks=retriever_init_tasks,
            )
        else:
            retriever_tag = "bm25"
            retriever_init_tasks = []  # no init task for BM25

            val_retrievers_factory = partial(
                bm25_retriever,
                cfg,
                retriever_tag,
                launcher_index=launcher_index,
                topk=cfg.learner.validation_top_k,
            )

            test_retrievers_factory = partial(
                bm25_retriever,
                cfg,
                retriever_tag,
                launcher_index=launcher_index,
                topk=cfg.retrieval.k,
            )

        # evaluate base retrievers alone and precompute runs for faster evaluation
        logging.info(f"Precomputing first stage runs for {retriever_tag}")
        test_runs = tests.evaluate_retriever(
            test_retrievers_factory,
            launcher=launcher_evaluate,
            init_tasks=retriever_init_tasks,
            with_run=cfg.precompute_first_stage,
            fabric_config=cfg.evaluation.fabric.get_config(),
        )
        if cfg.precompute_first_stage:
            test_run_retriever_factory = MultiRunRetrieverFactory.from_results(
                retriever_tag, test_runs
            )

        ### Validation ###
        validation_set = ValidationSet.load(cfg, launcher_preprocessing)
        val_tests = validation_set.to_evaluations()

        # Evaluate First stage on validation and store the topk
        # this enables faster validations during training
        val_runs = val_tests.evaluate_retriever(
            val_retrievers_factory,
            launcher=launcher_evaluate,
            init_tasks=retriever_init_tasks,
            with_run=True,
            fabric_config=cfg.evaluation.fabric.get_config(),
        )

        if cfg.precompute_first_stage:
            val_run_retriever_factory = MultiRunRetrieverFactory.from_results(
                retriever_tag, val_runs
            )

        ### TRAINING CROSS ENCODER

        ce_trainer: LossTrainer = build_trainer(cfg)

        hooks = [setmeta(GradientLogHook.C(), True)]
        if cfg.learner.max_grad_norm > 0:
            gradient_clipping_hook = GradientClippingHook.C(
                max_norm=cfg.learner.max_grad_norm
            )
            hooks.append(gradient_clipping_hook)

        if cfg.freeze_LN:
            freeze_hook = FreezeParameterHook.C(pattern=cfg.freeze_LN_pattern)
            hooks.append(freeze_hook)
            if "freeze_LN" not in cfg_tags:
                cfg_tags["freeze_LN"] = True

        # Build the model
        scorer_type = ScorerType(cfg.scorer_type)
        if scorer_type == ScorerType.stCrossEncoder:
            scorer_model, scorer_hf_init_tasks = build_STCrossEncoder(cfg)
        elif scorer_type == ScorerType.stMultiVectorEncoder:
            scorer_model, scorer_hf_init_tasks = build_STMultiVectorEncoder(cfg)
        elif scorer_type == ScorerType.hfCrossScorer:
            scorer_model, scorer_hf_init_tasks = hf_cross_scorer(
                hf_id=cfg.base,
                max_length=cfg.max_length,
                max_query_length=cfg.max_query_length,
                max_doc_length=cfg.max_doc_length,
                pref_attn_implementation=getattr(cfg, "pref_attn_implementation", None),
            )
        else:
            raise ValueError(
                f"Unknown scorer_type: '{cfg.scorer_type}'. "
                f"Expected one of {[e.value for e in ScorerType]}"
            )

        for k, v in cfg_tags.items():
            scorer_model.tag(k, v)
        # Run one Training and eval per seed
        for i in range(cfg.nb_repetitions):
            seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
            random = Random.C(seed=seed).tag("seed", seed)

            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            validations, tracked_validations = validation_set.build_listeners(
                scorer_model,
                (
                    val_run_retriever_factory
                    if cfg.precompute_first_stage
                    else val_retrievers_factory
                ),
                retriever_tag,
            )

            # The learner trains the model
            learner = Learner.C(
                # Misc settings
                random=random,
                trainer=ce_trainer,  # How to train the model
                model=scorer_model,  # The model to train
                # Optimization settings
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=cfg.learner.optimization.optimizer,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                # The listeners (here, for validation)
                listeners=stop_tags(validations),  # don't grab tags for validation
                # The hook used for evaluation
                hooks=hooks,
                # fabric settings
                fabric_config=cfg.learner.fabric.get_config(),
            )
            learners.append(learner)

            # Submit job and link
            outputs = learner.submit(
                launcher=launcher_learner,
                init_tasks=scorer_hf_init_tasks,
            )
            # this links the tensorboard run dir to in the xp/results/run folder, so that we can access it easily.
            if helper.xp.run_mode == RunMode.NORMAL:
                helper.tensorboard_service.add(learner, learner.logpath)
            else:
                logging.debug(
                    "Skipping TensorBoard service registration in dry-run mode"
                )

            # Build list of models to evaluate: (model_id, load_model)
            models_to_evaluate = []
            checkpoint_modes = cfg.evaluation.get_checkpoint_modes()

            for mode in checkpoint_modes:
                # 1. Best validation checkpoints
                if mode == CheckpointSelection.VAL:
                    for name, tracked_validation in tracked_validations.items():
                        logging.info(f"Adding validation model for evaluation: {name}")
                        for metric_name in tracked_validation.monitored():
                            load_model = (
                                outputs.listeners[tracked_validation.id][metric_name]
                                .tag("checkpoint", name)
                                .tag("seed", seed)
                                .tag("first_stage", retriever_tag)
                            )
                            model_id = f"{grid_search_id}-{name}-{metric_name}-{seed}"
                            models_to_evaluate.append((model_id, load_model))

                # 2. Last checkpoint
                elif mode == CheckpointSelection.LAST:
                    logging.info("Adding last checkpoint for evaluation")
                    last_model = (
                        outputs.learned_model.tag("checkpoint", "last")
                        .tag("seed", seed)
                        .tag("first_stage", retriever_tag)
                    )
                    model_id = f"{grid_search_id}-last-{seed}"
                    models_to_evaluate.append((model_id, last_model))

                # 3. Specific epoch checkpoint
                elif isinstance(mode, int):
                    target_epoch = mode
                    logging.info(
                        f"Adding checkpoint for epoch {target_epoch} for evaluation"
                    )
                    load_model = find_checkpoint(
                        outputs=outputs,
                        target_epoch=target_epoch,
                        learner=learner,
                    )
                    load_model = (
                        load_model.tag("checkpoint", f"epoch-{target_epoch}")
                        .tag("seed", seed)
                        .tag("first_stage", retriever_tag)
                    )
                    model_id = f"{grid_search_id}-epoch{target_epoch}-{seed}"
                    models_to_evaluate.append((model_id, load_model))

            # Shared evaluation execution loop
            effective_long_evals = getattr(
                cfg.indexation, "long_evals", None
            ) or getattr(cfg.retrieval, "long_evals", None)

            for model_id, load_model in models_to_evaluate:
                load_model.tag("scorer", cfg.base)
                if cfg.use_plaid:
                    load_model.tag("plaid_retriever", False)
                all_weights.append(load_model)
                tests.evaluate_retriever(
                    partial(
                        scorer_retriever,
                        scorer=scorer_model,
                        retrievers=(
                            test_run_retriever_factory
                            if cfg.precompute_first_stage
                            else test_retrievers_factory
                        ),
                        batch_size=cfg.retrieval.batch_size,
                    ),
                    launcher_evaluate,
                    model_id=model_id,
                    init_tasks=[load_model],
                    with_run=cfg.save_runs,
                    fabric_config=cfg.evaluation.fabric.get_config(),
                )

                # Option A: Standalone PLAID first-stage evaluation
                if cfg.use_plaid and scorer_type == ScorerType.stMultiVectorEncoder:
                    logging.info(
                        f"Scheduling PLAID first-stage evaluation for model {model_id}"
                    )
                    # Create an independent loader config so tags don't mutate load_model in-place
                    plaid_load_model = scorer_model.loader_config(
                        load_model.path,
                        settings=getattr(load_model, "settings", None),
                    )
                    for k, v in load_model.tags().items():
                        if k not in ("first_stage", "scorer", "plaid_retriever"):
                            plaid_load_model.tag(k, v)
                    plaid_load_model.tag("first_stage", cfg.base).tag(
                        "scorer", cfg.base
                    ).tag("plaid_retriever", True)
                    all_weights.append(plaid_load_model)

                    tests.evaluate_retriever(
                        partial(
                            plaid_retriever,
                            cfg,
                            cfg.base,
                            scorer_model,
                            launcher_index=launcher_index,
                            long_launcher_index=long_launcher_index,
                            long_evals=effective_long_evals,
                            init_tasks=[plaid_load_model],
                            topk=cfg.retrieval.k,
                            seed=seed,
                        ),
                        launcher_evaluate,
                        model_id=f"{model_id}-plaid",
                        init_tasks=[plaid_load_model],
                        with_run=cfg.save_runs,
                        fabric_config=cfg.evaluation.fabric.get_config(),
                    )

    all_configs, all_tags = generate_grid(cfg)

    # Simplify tags keys
    new_all_tags = []
    for cfg_tags in all_tags:
        simple_tags = {}
        for k, v in cfg_tags.items():
            simple_key = k.split(".")[-1]
            if simple_key in simple_tags:
                simple_tags[k] = v
            else:
                simple_tags[simple_key] = v
        new_all_tags.append(simple_tags)
    all_tags = new_all_tags

    config_map = {}

    for config, cfg_tags in zip(all_configs, all_tags):
        # Update cfg_tags with base if not present
        if "base" not in cfg_tags:
            cfg_tags["base"] = config.base

        # just run the config
        tagspath = "_".join(f"{k}={v}" for k, v in cfg_tags.items())
        config_map[frozenset(cfg_tags.items())] = config
        logging.info(
            f"Running config with tags:\n- {'\n- '.join(f'{k}: {v}' for k, v in cfg_tags.items())}"
        )
        prepare_hf_model(config.base)
        run_one_config(
            helper=helper, cfg=config, grid_search_id=tagspath, cfg_tags=cfg_tags
        )

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()
    # Process Results shared
    process_experiment_results(
        tests, all_tags, config_map, cfg, helper, learners, all_weights, get_ce_name
    )
