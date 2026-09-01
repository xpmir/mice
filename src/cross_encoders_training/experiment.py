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

from xpmir.papers.results import PaperResults
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.rankers import scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory
from xpmir.neural.splade import splade_encoder_from_pretrained_hf

from retrievers import splade_retriever, bm25_retriever
from validations import ValidationSet
from configuration import CE_FineTuning, CheckpointSelection
from tests import build_tests
from format import loss_names, backbone_names_lower

from training_utils import build_trainer, process_experiment_results

from typing import Optional
from experimaestro import LightweightTask, Param
from xpmir.neural.sentence_transformers import STCrossEncoder
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderModelCardData
from sentence_transformers.models import Transformer, Pooling, Dense, LayerNorm
from torch import nn

from logging_utils import setup_logging

setup_logging(level=logging.INFO)


class InitDistilledSTCrossEncoder(LightweightTask):
    """Initializes the STCrossEncoder with a custom distillation architecture."""

    model: Param[STCrossEncoder]
    pref_attn_implementation: Param[Optional[str]] = None

    def execute(self):
        # Initialize basic component properties without triggering the default CrossEncoder load
        super(STCrossEncoder, self.model).__initialize__()
        # Mark as initialized so the framework doesn't try to call __initialize__ again later
        self.model._initialized = True

        encoder_size = self.model.model_id.split("-")[-1]

        transformer = Transformer(
            self.model.model_id,
            max_seq_length=self.model.max_length,
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
        self.model.st_model = CrossEncoder(
            modules=[transformer, pooling, dense_inner, norm, dense_score],
            num_labels=1,
            activation_fn=nn.Identity(),
            model_card_data=CrossEncoderModelCardData(
                model_name=f"Ettin Reranker {encoder_size} distilled from mxbai-rerank-large-v2",
                language="en",
                license="apache-2.0",
            ),
        )


def build_STCrossEncoder(cfg) -> tuple[STCrossEncoder, list]:
    """Helper factory to build the distilled STCrossEncoder."""
    attn_impl = getattr(cfg, "pref_attn_implementation", "sdpa")
    scorer = STCrossEncoder.C(
        model_id=cfg.base, max_length=cfg.max_length, pref_attn_implementation=attn_impl
    )
    init_task = InitDistilledSTCrossEncoder.C(
        model=scorer, pref_attn_implementation=attn_impl
    )
    return scorer, [init_task]


def get_ce_name(model_tags: dict) -> str:
    """Creates the HF id from tags using formatting conventions."""
    loss = model_tags.get("loss", "")
    base = model_tags.get("base", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"
    base = backbone_names_lower.get(base, base).replace("/", "-")

    extra_parts = []
    if "batch_size" in model_tags:
        extra_parts.append(f"bs{model_tags['batch_size']}")
    if "checkpoint" in model_tags and str(model_tags["checkpoint"]) != "last":
        extra_parts.append(str(model_tags["checkpoint"]))

    extra = f"-{'-'.join(extra_parts)}" if extra_parts else ""
    return f"cross-encoder-{base}{loss}{extra}"


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: CE_FineTuning) -> PaperResults:
    launcher_index = find_launcher(cfg.indexation.requirements)
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

            test_retrievers_factory = partial(
                splade_retriever,
                cfg,
                retriever_tag,
                splade_encoder,
                launcher_index=launcher_index,
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

        # Build the model
        if getattr(cfg, "use_st_scorer", False):
            scorer_model, scorer_hf_init_tasks = build_STCrossEncoder(cfg)
        else:
            scorer_model, scorer_hf_init_tasks = hf_cross_scorer(
                hf_id=cfg.base,
                max_length=cfg.max_length,
                max_query_length=cfg.max_query_length,
                max_doc_length=cfg.max_doc_length,
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
            checkpoint_mode = cfg.evaluation.get_checkpoint_mode()

            # 1. Best validation checkpoints
            if checkpoint_mode in (CheckpointSelection.VAL, CheckpointSelection.BOTH):
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
            if checkpoint_mode in (CheckpointSelection.LAST, CheckpointSelection.BOTH):
                logging.info("Adding last checkpoint for evaluation")
                last_model = (
                    outputs.learned_model.tag("checkpoint", "last")
                    .tag("seed", seed)
                    .tag("first_stage", retriever_tag)
                )
                model_id = f"{grid_search_id}-last-{seed}"
                models_to_evaluate.append((model_id, last_model))

            # Shared evaluation execution loop
            for model_id, load_model in models_to_evaluate:
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
        run_one_config(
            helper=helper, cfg=config, grid_search_id=tagspath, cfg_tags=cfg_tags
        )

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()
    # Process Results shared
    process_experiment_results(
        tests, all_tags, config_map, cfg, helper, learners, all_weights, get_ce_name
    )
