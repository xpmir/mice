"""
MICE V2 (ModernBERT-based) Training Experiment.
Implements training pipeline for MICE V2 models.
"""

import logging
from pathlib import Path
from functools import partial
from typing import Optional, Callable
import numpy as np

from experimaestro import setmeta, stop_tags, RunMode
from experimaestro.launcherfinder import find_launcher

from xpm_torch import Random
from xpm_torch.experiments.helpers import LearningExperimentHelper, learning_experiment
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import Learner
from xpm_torch.optim import GradientLogHook, GradientClippingHook
from xpm_torch.huggingface import prepare_hf_model

from xpmir.papers.results import PaperResults
from xpmir.rankers import scorer_retriever
from xpmir.evaluation import MultiRunRetrieverFactory
from xpmir.text.huggingface.tokenizers import get_default_max_len
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.papers import configuration

from retrievers import splade_retriever, bm25_retriever, MicePlaidRetrieverFactory
from validations import ValidationSet
from experimaestro.experiments.grid import generate_grid, GridSearch
from configuration import CE_FineTuning, CheckpointSelection
from tests import build_tests
from format import (
    loss_names,
    backbone_names_lower,
)

from training_utils import (
    build_trainer,
    process_experiment_results,
    find_checkpoint,
    resolve_trained_model_loader,
    validate_and_align_config,
    FreezeParameterHook,
)

from MICE.modeling.mice_v2.modernbert_xpm import (
    mice_v2_scorer,
    UtilityRegularizer,
    GatedValueRegularizer,
    InitMICEV2FromCheckpoint,
    InitEvalSparsityHook,
)
from xp_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class MiceV2_FineTuning(CE_FineTuning):
    ## MICE V2 specific configuration
    n_contextualization_layers: GridSearch[int] = 4
    """Number of bottom encoder layers that process query and document independently"""

    n_docs_ctx_layers: GridSearch[Optional[int]] = None
    """Number of bottom encoder layers for the document. If None, use n_contextualization_layers."""

    n_interaction_layers: GridSearch[Optional[int]] = None
    """Number of top encoder layers with cross-attention. If None, use all remaining layers from backbone."""

    bound_bottom_layers: GridSearch[bool] = True
    """Whether to bound bottom query and document encoding layers"""

    attn_implementation: GridSearch[Optional[str]] = None
    """Attention implementation (e.g. 'flash_attention_2', 'sdpa', 'eager')"""

    utility_pred: GridSearch[bool] = False
    """Whether to enable 2-layer MLP soft-gating utility predictor on document tokens"""

    lambda_reg: GridSearch[float] = 0.0
    """Regularization weight for the Utility predictor"""

    reg_target: GridSearch[str] = "u"
    """Target for regularization: 'u' for mean utility loss, 'u_wv_h' for gated value norm loss (u * W_V * h)"""

    gate_target: GridSearch[str] = "h"
    """Target representation for gating: 'h' for full document hidden states (scales K and V), 'v' for cross-attention Value vectors only"""

    use_hard_concrete: GridSearch[bool] = False
    """Whether to use Hard Concrete L0 distribution sampling and regularization"""

    hc_gamma: GridSearch[float] = -0.1
    """Hard concrete lower stretch parameter gamma (default -0.1)"""

    hc_zeta: GridSearch[float] = 1.1
    """Hard concrete upper stretch parameter zeta (default 1.1)"""

    hard_pruning: GridSearch[Optional[bool]] = None
    """Whether to physically drop tokens. Defaults to True if use_hard_concrete=True, else False."""

    reg_budget: GridSearch[Optional[float]] = None
    """Optional target retention budget for utility regularizer (e.g. 0.3 for 30% retention). If None, standard linear L1 regularization is used."""

    l2_reg: GridSearch[bool] = False
    """Whether to square the regularization penalty (L2 instead of L1) for smoother optimization"""

    phase2_training: GridSearch[bool] = False
    """Whether to run Phase 2 hard token pruning fine-tuning after Phase 1 soft gating"""

    hard_prune_threshold: GridSearch[float] = 0.5
    """Utility threshold below which document tokens are hard dropped in Phase 2"""

    normalize_doc_embeds: GridSearch[bool] = False
    """Whether to L2-normalize document hidden states before utility prediction and upper interaction layers"""

    phase2_max_epochs: GridSearch[int] = 3
    """Number of training epochs for Phase 2 top layer fine-tuning"""

    freeze_bottom_layers: GridSearch[bool] = True
    """Whether to freeze bottom contextualization layers during Phase 2 training (utility predictor is always frozen)"""

    freeze_bottom_layers_phase1: GridSearch[bool] = False
    """Whether to freeze bottom contextualization layers during Phase 1 training (embeddings and bottom layers frozen, leaving top layers, utility predictor, and head trainable)"""

    freeze_top_layers: GridSearch[bool] = False
    """Whether to freeze top interaction layers, adapter, rotary embeddings, and classification head (leaving bottleneck parameters trainable)"""

    eval_phase_1: GridSearch[bool] = False
    """Whether to also evaluate the Phase 1 model (soft gating without hard pruning) when phase2_training is enabled."""

    use_upper_rope: GridSearch[bool] = True
    """Whether to use Rotary Position Embedding (RoPE) in upper interaction layers"""

    top_layers_doc_ln: GridSearch[bool] = False
    """Whether to use a document LayerNorm in top interaction layers and multiply token utilities post-LayerNorm"""

    bottom_final_norm: GridSearch[bool] = True
    """Whether to apply LayerNorm at the output of bottom contextualization layers"""

    top_final_norm: GridSearch[bool] = False
    """Whether to apply LayerNorm at the output of top interaction layers before classifier"""

    utility_bias: GridSearch[float] = 5.0
    """Initial bias scalar for utility predictor output projection wo.bias"""

    temp_start: GridSearch[float] = 1.0
    """Initial temperature for utility predictor sigmoid (default 1.0)"""

    temp_end: GridSearch[float] = 0.1
    """Final temperature for utility predictor sigmoid after annealing (default 0.1)"""

    temp_schedule: GridSearch[str] = "cosine"
    """Schedule type for temperature annealing: 'linear', 'cosine', or 'exp'"""

    temp_anneal_steps: GridSearch[Optional[int]] = None
    """Number of training steps to anneal temperature over. If None, anneals over total training steps."""

    compress_dim: GridSearch[float] = 1.0
    """Compression ratio (>=1.0) applied to hidden_size/intermediate_size/num_attention_heads of the top interaction layers only"""

    compress_init_option: GridSearch[str] = "seed"
    """How top interaction layers are initialized: 'seed' (load from the pretrained
    checkpoint when compress_dim == 1.0, else falls back to 'random'), 'random' (always
    skip seeding, keep each layer's own default init), 'random_projection', or
    'svd_projection' (both only meaningful when compress_dim > 1.0)"""

    doc_bottleneck_dim: GridSearch[Optional[int]] = None
    """Optional bottleneck dimension for document token embeddings (d). If set, compresses document embeddings to d dimensions before storage/cross-attention."""

    use_cross_attn: GridSearch[bool] = True
    """Whether top layers use Cross+Self attention (True) or Concatenated Self-Attention only (False)"""

    head_type: GridSearch[str] = "linear"
    """Type of classification head: 'linear' (default) or 'st_reranker'"""

    eval_sparsity: bool = False
    """Whether to compute token sparsity, mean utility, and token retention ratio during evaluation benchmarks"""

    query_prompt: GridSearch[Optional[str]] = None
    """Optional query prompt template or prefix (e.g. '[Q] ' or 'query: ')"""

    doc_prompt: GridSearch[Optional[str]] = None
    """Optional document prompt template or prefix (e.g. '[D] ' or 'passage: ')"""


def build_mice_v2(
    cfg: MiceV2_FineTuning,
    phase_2: bool = False,
    cfg_tags: Optional[dict] = None,
):
    """Factory helper to build MICE V2 scorer with all architectural flags."""
    from_exp = getattr(cfg, "from_experiment", None)
    checkpoint_loader = None
    if from_exp and not phase_2:
        checkpoint_loader = resolve_trained_model_loader(from_exp)
        validate_and_align_config(
            cfg, checkpoint_loader.model, tags=cfg_tags, from_ref=from_exp
        )

    default_max_len = get_default_max_len(cfg.base)
    if cfg.max_length and default_max_len > cfg.max_length:
        max_len = cfg.max_length
    else:
        max_len = default_max_len

    attn_impl = cfg.attn_implementation or cfg.pref_attn_implementation

    freeze_bottom = (
        cfg.freeze_bottom_layers
        if (phase_2 or cfg.hard_pruning)
        else cfg.freeze_bottom_layers_phase1
    )

    kwargs = dict(
        hf_id=cfg.base,
        n_contextualization_layers=cfg.n_contextualization_layers,
        n_docs_ctx_layers=cfg.n_docs_ctx_layers,
        n_interaction_layers=cfg.n_interaction_layers,
        bound_bottom_layers=cfg.bound_bottom_layers,
        attn_implementation=attn_impl,
        utility_pred=cfg.utility_pred,
        normalize_doc_embeds=cfg.normalize_doc_embeds,
        use_upper_rope=cfg.use_upper_rope,
        top_layers_doc_ln=cfg.top_layers_doc_ln,
        bottom_final_norm=cfg.bottom_final_norm,
        top_final_norm=cfg.top_final_norm,
        utility_temp=cfg.temp_start,
        utility_bias=cfg.utility_bias,
        gate_target=cfg.gate_target,
        use_hard_concrete=cfg.use_hard_concrete,
        hc_gamma=cfg.hc_gamma,
        hc_zeta=cfg.hc_zeta,
        compress_dim=cfg.compress_dim,
        compress_init_option=cfg.compress_init_option,
        doc_bottleneck_dim=cfg.doc_bottleneck_dim,
        use_cross_attn=cfg.use_cross_attn,
        head_type=cfg.head_type,
        max_length=max_len,
        max_query_length=getattr(cfg, "max_query_length", None),
        max_doc_length=getattr(cfg, "max_doc_length", None),
        query_prompt=getattr(cfg, "query_prompt", None),
        doc_prompt=getattr(cfg, "doc_prompt", None),
        freeze_bottom_layers=freeze_bottom,
    )

    if phase_2:
        logging.info(
            f"[Phase 2] Building MICE V2 model for hard pruning fine-tuning (threshold={cfg.hard_prune_threshold})"
        )
        kwargs.update(
            utility_pred=True,
            hard_pruning=True,
            hard_prune_threshold=cfg.hard_prune_threshold,
        )
    elif cfg.use_hard_concrete:
        logging.info(
            "[Single-Phase HC] Building MICE V2 model with Hard Concrete pruning"
        )
        kwargs.update(
            hard_pruning=True if cfg.hard_pruning is None else cfg.hard_pruning,
            hard_prune_threshold=0.0,
        )
    elif cfg.hard_pruning:
        logging.info(
            f"[Hard Pruning] Building MICE V2 model with hard pruning (threshold={cfg.hard_prune_threshold})"
        )
        kwargs.update(
            utility_pred=True,
            hard_pruning=True,
            hard_prune_threshold=cfg.hard_prune_threshold,
        )
    else:
        logging.info(
            f"[Phase 1] Building MICE V2 model for soft gating training (utility_pred={cfg.utility_pred})"
        )

    scorer, init_tasks = mice_v2_scorer(**kwargs)

    if checkpoint_loader is not None:
        init_tasks = [
            InitMICEV2FromCheckpoint.C(
                scorer=scorer,
                checkpoint_loader=stop_tags(checkpoint_loader),
                freeze_bottom_layers=freeze_bottom,
                freeze_utility=phase_2,
                freeze_top_layers=cfg.freeze_top_layers,
            )
        ]

    return scorer, init_tasks


def get_name_from_tags(model_tags: dict, cfg: MiceV2_FineTuning) -> str:
    """Creates the HF id from tags using formatting conventions."""
    base = model_tags.get("base", "")
    base = backbone_names_lower.get(base, base).replace("/", "-")

    n_ctx_layers = model_tags.get(
        "n_contextualization_layers", cfg.n_contextualization_layers
    )
    n_docs_layers = model_tags.get("n_docs_ctx_layers", cfg.n_docs_ctx_layers)
    doc_suffix = f"-d{n_docs_layers}" if n_docs_layers is not None else ""

    use_upper_rope = model_tags.get("use_upper_rope", cfg.use_upper_rope)
    rope_suffix = "" if use_upper_rope else "-norope"

    top_layers_doc_ln = model_tags.get("top_layers_doc_ln", cfg.top_layers_doc_ln)
    doc_ln_suffix = "-docln" if top_layers_doc_ln else ""

    bottom_final_norm = model_tags.get("bottom_final_norm", cfg.bottom_final_norm)
    nobottomln_suffix = "" if bottom_final_norm else "-nobottomln"

    top_final_norm = model_tags.get("top_final_norm", cfg.top_final_norm)
    topln_suffix = "-topln" if top_final_norm else ""

    gate_target = model_tags.get("gate_target", getattr(cfg, "gate_target", "h"))
    gate_suffix = "-gatev" if gate_target == "v" else ""

    use_hard_concrete = model_tags.get(
        "use_hard_concrete", getattr(cfg, "use_hard_concrete", False)
    )
    hc_suffix = "-hc" if use_hard_concrete else ""

    hard_pruning = model_tags.get("hard_pruning", getattr(cfg, "hard_pruning", None))
    prune_suffix = (
        "-nohardprune" if (use_hard_concrete and hard_pruning is False) else ""
    )

    head_type = model_tags.get("head_type", getattr(cfg, "head_type", "linear"))
    head_suffix = f"-{head_type}" if head_type and head_type != "linear" else ""

    freeze_bottom_phase1 = model_tags.get(
        "freeze_bottom_layers_phase1",
        getattr(cfg, "freeze_bottom_layers_phase1", False),
    )
    freeze_bottom_suffix = "-freezebottom" if freeze_bottom_phase1 else ""

    freeze_top = model_tags.get(
        "freeze_top_layers",
        getattr(cfg, "freeze_top_layers", False),
    )
    freeze_top_suffix = "-freezetop" if freeze_top else ""

    compress_dim = model_tags.get("compress_dim", getattr(cfg, "compress_dim", 1.0))
    compress_suffix = (
        f"-cdim{round(float(compress_dim), 2)}" if float(compress_dim) > 1.0 else ""
    )

    doc_bot_dim = model_tags.get(
        "doc_bottleneck_dim", getattr(cfg, "doc_bottleneck_dim", None)
    )
    dbot_suffix = f"-dbot{doc_bot_dim}" if doc_bot_dim else ""

    utility_bias = model_tags.get("utility_bias", getattr(cfg, "utility_bias", 5.0))
    ubias_suffix = f"-ubias{utility_bias}" if utility_bias != 5.0 else ""

    attn_impl = model_tags.get("attn_implementation", cfg.attn_implementation)
    attn_suffix = f"-{attn_impl}" if attn_impl else ""

    loss = model_tags.get("loss", "")
    loss = loss_names.get(loss, loss).replace("/", "-")
    if len(loss):
        loss = f"-{loss}"

    return f"MiceV2-l{n_ctx_layers}{doc_suffix}{rope_suffix}{doc_ln_suffix}{nobottomln_suffix}{topln_suffix}{gate_suffix}{hc_suffix}{prune_suffix}{freeze_bottom_suffix}{freeze_top_suffix}{compress_suffix}{dbot_suffix}{head_suffix}{ubias_suffix}{attn_suffix}-{base}{loss}"


def TrainEvaluateMiceV2(
    helper: LearningExperimentHelper,
    cfg: CE_FineTuning,
    get_model_name_fn: Callable,
) -> PaperResults:
    """Generic training and evaluation experiment for late interaction models."""
    launcher_index = find_launcher(cfg.indexation.requirements)
    long_reqs_index = cfg.indexation.long_requirements
    long_launcher_index = find_launcher(long_reqs_index) if long_reqs_index else None
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    long_reqs = cfg.retrieval.long_requirements
    long_launcher = find_launcher(long_reqs) if long_reqs else None

    # I - Build the model and initialization tasks
    all_configs, all_tags = generate_grid(cfg)
    if not all_configs:
        raise ValueError(
            f"generate_grid returned 0 configurations for {cfg.id}. "
            "Please check your grid_search configuration or parameter definitions."
        )

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

    ## II - common evaluation for all configs (e.g., first stage retriever runs, validation runs)
    base_config, base_tags = all_configs[0], all_tags[0]
    logging.info(
        f"Running common evaluation for all configs (first stage retriever runs, validation runs) with base config: {base_config.id}, tags: {base_tags} (should not impact first stage retriever runs)"
    )

    tests = build_tests(
        base_config.evaluation,
        launcher=launcher_preprocessing,
        default_launcher=launcher_evaluate,
        long_launcher=long_launcher,
        blocking=base_config.preprocessing.blocking_download,
    )

    # cache the indexes
    learners = []
    all_weights = []

    if base_config.retriever:
        # We don't use BM25, but a given sparse retriever
        retriever_tag = base_config.retriever
        splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(
            base_config.retriever
        )

        # Caches the Splade index task for a document collection
        val_retrievers_factory = partial(
            splade_retriever,
            base_config,
            retriever_tag,
            splade_encoder,
            launcher_index=launcher_index,
            init_tasks=retriever_init_tasks,
            topk=base_config.learner.validation_top_k,
            in_memory=True,
        )

        test_retrievers_factory = partial(
            splade_retriever,
            base_config,
            retriever_tag,
            splade_encoder,
            launcher_index=launcher_index,
            topk=base_config.retrieval.k,
            init_tasks=retriever_init_tasks,
        )
    else:
        retriever_tag = "bm25"
        retriever_init_tasks = []  # no init task for BM25

        val_retrievers_factory = partial(
            bm25_retriever,
            base_config,
            retriever_tag,
            launcher_index=launcher_index,
            topk=base_config.learner.validation_top_k,
        )

        test_retrievers_factory = partial(
            bm25_retriever,
            base_config,
            retriever_tag,
            launcher_index=launcher_index,
            topk=base_config.retrieval.k,
        )

    # evaluate base retrievers alone and precompute runs for faster evaluation (once for the experiment)
    logging.info(f"Precomputing first stage test runs for {retriever_tag}")
    test_runs = tests.evaluate_retriever(
        test_retrievers_factory,
        launcher=launcher_evaluate,
        init_tasks=retriever_init_tasks,
        with_run=True,
        fabric_config=base_config.evaluation.fabric.get_config(),
    )
    test_run_retriever_factory = MultiRunRetrieverFactory.from_results(
        retriever_tag, test_runs
    )

    ### Validation (once for the experiment) ###
    logging.info(f"Precomputing first stage validation runs for {retriever_tag}")
    validation_set = ValidationSet.load(base_config, launcher_preprocessing)
    val_tests = validation_set.to_evaluations()
    val_runs = val_tests.evaluate_retriever(
        val_retrievers_factory,
        launcher=launcher_evaluate,
        init_tasks=retriever_init_tasks,
        with_run=True,
        fabric_config=base_config.evaluation.fabric.get_config(),
    )

    val_run_retriever_factory = MultiRunRetrieverFactory.from_results(
        retriever_tag, val_runs
    )

    def run_one_config(
        helper: LearningExperimentHelper,
        cfg: CE_FineTuning,
        grid_search_id: str,
        cfg_tags: dict,
    ):
        """Main process for one configuration of the late interaction model."""

        ### TRAINING CROSS ENCODER

        ce_trainer: LossTrainer = build_trainer(cfg)

        # Base hooks shared by all training phases across all seed repetitions
        learner_hooks = [setmeta(GradientLogHook.C(), True)]
        if cfg.learner.max_grad_norm > 0:
            gradient_clipping_hook = GradientClippingHook.C(
                max_norm=cfg.learner.max_grad_norm
            )
            learner_hooks.append(gradient_clipping_hook)

        if cfg.freeze_LN:
            freeze_hook = FreezeParameterHook.C(pattern=cfg.freeze_LN_pattern)
            learner_hooks.append(freeze_hook)

        # Phase 1 hooks: base hooks + UtilityRegularizer (if soft gating utility prediction is active)

        if cfg.utility_pred and cfg.lambda_reg > 0.0:
            histogram_interval = (
                cfg.learner.validation_interval
                * cfg.learner.optimization.steps_per_epoch
            )
            if cfg.reg_target == "u":
                ce_trainer.hooks.append(
                    UtilityRegularizer.C(
                        lambda_reg=cfg.lambda_reg,
                        budget=cfg.reg_budget,
                        l2_reg=cfg.l2_reg,
                        temp_start=cfg.temp_start,
                        temp_end=cfg.temp_end,
                        temp_anneal_steps=cfg.temp_anneal_steps,
                        temp_schedule=cfg.temp_schedule,
                        histogram_interval=histogram_interval,
                        sparsity_threshold=cfg.hard_prune_threshold,
                    )
                )
            elif cfg.reg_target == "u_wv_h":
                ce_trainer.hooks.append(
                    GatedValueRegularizer.C(
                        lambda_reg=cfg.lambda_reg,
                        budget=cfg.reg_budget,
                        l2_reg=cfg.l2_reg,
                        temp_start=cfg.temp_start,
                        temp_end=cfg.temp_end,
                        temp_anneal_steps=cfg.temp_anneal_steps,
                        temp_schedule=cfg.temp_schedule,
                        histogram_interval=histogram_interval,
                        sparsity_threshold=cfg.hard_prune_threshold,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown reg_target: '{cfg.reg_target}'. Expected 'u' or 'u_wv_h'."
                )

        # Build the model using the provided factory
        model, model_init_tasks = build_mice_v2(cfg, cfg_tags=cfg_tags)

        # tag the model with the configuration tags for monitoring and logging
        for k, v in cfg_tags.items():
            model.tag(k, v)

        # Run one Training and eval per seed
        for i in range(cfg.nb_repetitions):
            seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
            random = Random.C(seed=seed).tag("seed", seed)
            phases_to_evaluate = []

            # The validation listener evaluates the full retriever
            # (retriever + scorer) and keep the best performing model
            # on the validation set
            validations, tracked_validations = validation_set.build_listeners(
                model,
                val_run_retriever_factory,
                retriever_tag,
            )

            optimizers = cfg.learner.optimization.optimizer
            if cfg.phase2_training:
                total_epochs = (
                    cfg.learner.optimization.max_epochs + cfg.phase2_max_epochs
                )
                total_steps = total_epochs * cfg.learner.optimization.steps_per_epoch
                optimizers = cfg.learner.optimization.get_optimizers_with_total_steps(
                    total_steps
                )

            # The learner trains the model
            learner = Learner.C(
                # Misc settings
                random=random,
                trainer=ce_trainer,  # How to train the model
                model=model,  # The model to train
                # Optimization settings
                steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                optimizers=optimizers,
                max_epochs=cfg.learner.optimization.max_epochs,
                checkpoint_interval=cfg.learner.checkpoint_interval,
                # The listeners (here, for validation)
                listeners=stop_tags(validations),  # don't grab tags for validation
                # The hook used for evaluation
                hooks=learner_hooks,
                # fabric settings
                fabric_config=cfg.learner.fabric.get_config(),
            )
            if cfg.phase2_training:
                learner.tag("phase", 1)

            learners.append(learner)

            # Submit job and link
            outputs = learner.submit(
                launcher=launcher_learner,
                init_tasks=model_init_tasks,
            )
            # this links the tensorboard run dir to in the xp/results/run folder, so that we can access it easily.
            if helper.xp.run_mode == RunMode.NORMAL:
                helper.tensorboard_service.add(learner, learner.logpath)

            if cfg.phase2_training:
                total_epochs = (
                    cfg.learner.optimization.max_epochs + cfg.phase2_max_epochs
                )

                logging.info(
                    f"[Phase 2] Launching Hard Pruning Fine-Tuning (threshold={cfg.hard_prune_threshold}, max_epochs={cfg.phase2_max_epochs}, total_epochs={total_epochs})"
                )
                # Build the Phase 2 model for hard pruning fine-tuning
                model_phase2, _ = build_mice_v2(cfg, phase_2=True)

                for k, v in cfg_tags.items():
                    model_phase2.tag(k, v)

                validations_p2, tracked_validations_p2 = validation_set.build_listeners(
                    model_phase2,
                    val_run_retriever_factory,
                    retriever_tag,
                )

                learner_p2 = Learner.C(
                    random=random,
                    # Build a new trainer for phase 2, as it may have different hooks (e.g., no utility regularizer)
                    trainer=build_trainer(cfg),
                    model=model_phase2,
                    steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                    optimizers=optimizers,
                    max_epochs=cfg.phase2_max_epochs,
                    checkpoint_interval=cfg.learner.checkpoint_interval,
                    listeners=stop_tags(validations_p2),
                    hooks=learner_hooks,
                    fabric_config=cfg.learner.fabric.get_config(),
                ).tag("phase", 2)

                learners.append(learner_p2)

                p2_init_task = InitMICEV2FromCheckpoint.C(
                    scorer=model_phase2,
                    checkpoint_loader=stop_tags(
                        outputs.learned_model
                    ),  # use outputs from phase 1 as init for phase 2
                    freeze_bottom_layers=cfg.freeze_bottom_layers,
                    freeze_top_layers=cfg.freeze_top_layers,
                )

                outputs_p2 = learner_p2.submit(
                    launcher=launcher_learner,
                    init_tasks=[p2_init_task],
                )
                if helper.xp.run_mode == RunMode.NORMAL:
                    helper.tensorboard_service.add(learner_p2, learner_p2.logpath)

                phases_to_evaluate.append(
                    (2, model_phase2, outputs_p2, tracked_validations_p2)
                )

                if cfg.eval_phase_1:
                    phases_to_evaluate.append((1, model, outputs, tracked_validations))

            else:
                phases_to_evaluate.append((1, model, outputs, tracked_validations))

            # Build list of models to evaluate: (model_id, load_model, label, eval_model)

            models_to_evaluate = []
            checkpoint_modes = cfg.evaluation.get_checkpoint_modes()

            for (
                phase_num,
                current_model,
                current_outputs,
                current_tracked_validations,
            ) in phases_to_evaluate:
                phase_prefix = f"-p{phase_num}" if cfg.phase2_training else ""
                current_learner = learner_p2 if phase_num == 2 else learner

                for mode in checkpoint_modes:
                    # 1. Best validation checkpoints
                    if mode == CheckpointSelection.VAL:
                        for (
                            name,
                            tracked_validation,
                        ) in current_tracked_validations.items():
                            logging.info(
                                f"Adding validation model for evaluation (Phase {phase_num}): {name}"
                            )
                            for metric_name in tracked_validation.monitored():
                                load_model = (
                                    current_outputs.listeners[tracked_validation.id][
                                        metric_name
                                    ]
                                    .tag("checkpoint", name)
                                    .tag("seed", seed)
                                    .tag("first_stage", retriever_tag)
                                )
                                if cfg.phase2_training:
                                    load_model.tag("phase", phase_num)
                                for k, v in cfg_tags.items():
                                    load_model.tag(k, v)
                                model_id = f"{grid_search_id}{phase_prefix}-{name}-{metric_name}-{seed}"
                                models_to_evaluate.append(
                                    (model_id, load_model, name, current_model)
                                )

                    # 2. Last checkpoint
                    elif mode == CheckpointSelection.LAST:
                        logging.info(
                            f"Adding last checkpoint for evaluation (Phase {phase_num})"
                        )
                        last_model = (
                            current_outputs.learned_model.tag("checkpoint", "last")
                            .tag("seed", seed)
                            .tag("first_stage", retriever_tag)
                        )
                        if cfg.phase2_training:
                            last_model.tag("phase", phase_num)

                        for k, v in cfg_tags.items():
                            last_model.tag(k, v)
                        model_id = f"{grid_search_id}{phase_prefix}-last-{seed}"
                        models_to_evaluate.append(
                            (model_id, last_model, "last", current_model)
                        )

                    # 3. Specific epoch checkpoint
                    elif isinstance(mode, int):
                        target_epoch = mode
                        logging.info(
                            f"Adding checkpoint for epoch {target_epoch} for evaluation (Phase {phase_num})"
                        )
                        load_model = find_checkpoint(
                            outputs=current_outputs,
                            target_epoch=target_epoch,
                            learner=current_learner,
                        )
                        load_model = (
                            load_model.tag("checkpoint", f"epoch-{target_epoch}")
                            .tag("seed", seed)
                            .tag("first_stage", retriever_tag)
                        )
                        if cfg.phase2_training:
                            load_model.tag("phase", phase_num)
                        for k, v in cfg_tags.items():
                            load_model.tag(k, v)
                        model_id = (
                            f"{grid_search_id}{phase_prefix}-epoch{target_epoch}-{seed}"
                        )
                        models_to_evaluate.append(
                            (
                                model_id,
                                load_model,
                                f"epoch-{target_epoch}",
                                current_model,
                            )
                        )

            # Shared evaluation execution loop (Standard & PLAID)
            for model_id, load_model, label, eval_model in models_to_evaluate:
                if cfg.use_plaid:
                    load_model.tag("plaid_retriever", False)

                eval_init_tasks = [load_model]
                if cfg.eval_sparsity:
                    eval_init_tasks.append(
                        InitEvalSparsityHook.C(
                            scorer=eval_model,
                            threshold=cfg.hard_prune_threshold,
                        )
                    )

                all_weights.append(load_model)

                tests.evaluate_retriever(
                    partial(
                        scorer_retriever,
                        scorer=eval_model,
                        retrievers=test_run_retriever_factory,
                        batch_size=cfg.retrieval.batch_size,
                    ),
                    launcher_evaluate,
                    model_id=model_id,
                    init_tasks=eval_init_tasks,
                    with_run=cfg.save_runs,
                    fabric_config=cfg.evaluation.fabric.get_config(),
                )

                if cfg.use_plaid:
                    load_model.tag("plaid_retriever", True)
                    logging.info(f"Running PLAID-style evaluation for: {label}")
                    if cfg.plaid.max_doc_length is not None:
                        logging.info(
                            f"[PLAID] Enforcing max_doc_length={cfg.plaid.max_doc_length} for PLAID indexation "
                            f"(base model default: {eval_model.tokenizer.max_doc_length})"
                        )
                    if cfg.plaid.max_query_length is not None:
                        logging.info(
                            f"[PLAID] Enforcing max_query_length={cfg.plaid.max_query_length} for PLAID search/reranking "
                            f"(base model default: {eval_model.tokenizer.max_query_length})"
                        )

                    # 1) Build the Retriever factory for first stage + model + Plaid
                    mice_plaid_factory = MicePlaidRetrieverFactory(
                        mice_model=eval_model,
                        first_stage_factory=test_run_retriever_factory,
                        topk=cfg.retrieval.k,
                        batchsize=cfg.retrieval.batch_size,
                        n_ivf_probe=cfg.plaid.n_ivf_probe,
                        n_full_scores=cfg.plaid.n_full_scores,
                        max_query_length=cfg.plaid.max_query_length,
                        max_doc_length=cfg.plaid.max_doc_length,
                    )
                    # 2) Build the plaid indexes for all datasets in the collection
                    mice_plaid_factory.build_indexes(
                        evaluations=tests,
                        cfg=cfg.plaid,
                        batch_size=cfg.retrieval.batch_size,
                        seed=seed,
                        launcher_index=launcher_index,
                        init_tasks=[load_model],
                        long_launcher_index=long_launcher_index,
                        long_evals=getattr(cfg.indexation, "long_evals", None),
                    )

                    # 2) Run tests with the factory
                    all_weights.append(load_model)
                    tests.evaluate_retriever(
                        retriever=mice_plaid_factory,
                        launcher=launcher_evaluate,
                        model_id=f"{model_id}-Plaid",
                        init_tasks=eval_init_tasks,
                        with_run=cfg.save_runs,
                        fabric_config=cfg.evaluation.fabric.get_config(),
                    )

                    if cfg.plaid.use_colbert_first_stage:
                        logging.info(
                            f"Running genuine ColBERT-style first-stage evaluation for: {label}"
                        )
                        colbert_tag = "mice-colbert"

                        # the tags for the first stage plaid retriever
                        load_model.tag("first_stage", colbert_tag)
                        load_model.tag("plaid_retriever", "")

                        # Precompute the ColBERT-style first-stage run once (same
                        # pattern as the SPLADE precompute above) -- no model_id,
                        # so it isn't registered as its own report row, exactly
                        # like the SPLADE-alone run isn't either; the measures
                        # are still computed and stored.
                        colbert_runs = tests.evaluate_retriever(
                            mice_plaid_factory.colbert_retriever,
                            launcher=launcher_evaluate,
                            init_tasks=eval_init_tasks,
                            with_run=True,
                            fabric_config=cfg.evaluation.fabric.get_config(),
                        )
                        colbert_run_retriever_factory = (
                            MultiRunRetrieverFactory.from_results(
                                colbert_tag, colbert_runs
                            )
                        )

                        # for the second stage reranker
                        load_model.tag("plaid_retriever", True)

                        # Rerank of ColBERT-retrieved candidates via MICE's top
                        # interaction layers -- reuses the cached run above (no
                        # first-stage re-retrieval). build_indexes submits an
                        # identical PlaidIndexBuilder config to the one above, so
                        # it resolves to the same experimaestro task (no re-indexing).
                        # We have to create a new factory here as it use different
                        # first stage factory compare to previous one.
                        mice_colbert_rerank_factory = MicePlaidRetrieverFactory(
                            mice_model=eval_model,
                            first_stage_factory=colbert_run_retriever_factory,
                            topk=cfg.retrieval.k,
                            batchsize=cfg.retrieval.batch_size,
                            n_ivf_probe=cfg.plaid.n_ivf_probe,
                            n_full_scores=cfg.plaid.n_full_scores,
                            max_query_length=cfg.plaid.max_query_length,
                            max_doc_length=cfg.plaid.max_doc_length,
                        )
                        mice_colbert_rerank_factory.build_indexes(
                            evaluations=tests,
                            cfg=cfg.plaid,
                            batch_size=cfg.retrieval.batch_size,
                            seed=seed,
                            launcher_index=launcher_index,
                            init_tasks=[load_model],
                            long_launcher_index=long_launcher_index,
                            long_evals=getattr(cfg.indexation, "long_evals", None),
                        )
                        tests.evaluate_retriever(
                            retriever=mice_colbert_rerank_factory,
                            launcher=launcher_evaluate,
                            model_id=f"{model_id}-ColBERT-Plaid",
                            init_tasks=eval_init_tasks,
                            with_run=cfg.save_runs,
                            fabric_config=cfg.evaluation.fabric.get_config(),
                        )

    for config, cfg_tags in zip(all_configs, all_tags):
        # Update cfg_tags with base if not present
        if "base" not in cfg_tags:
            cfg_tags["base"] = config.base
        for k, v in cfg_tags.items():
            cfg_tags[k] = str(v)  # ensure cfg_tags has all keys from config
        # just run the config
        tagspath = "_".join(f"{k}={v}" for k, v in cfg_tags.items())
        logging.info(
            f"Running config with tags:\n- {'\n- '.join(f'{k}: {v}' for k, v in cfg_tags.items())}"
        )
        config_map[frozenset(cfg_tags.items())] = config

        if config.base and not Path(config.base).exists():
            prepare_hf_model(config.base)

        run_one_config(
            helper=helper, cfg=config, grid_search_id=tagspath, cfg_tags=cfg_tags
        )

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    process_experiment_results(
        tests=tests,
        all_tags=all_tags,
        config_map=config_map,
        cfg=cfg,
        helper=helper,
        learners=learners,
        all_weights=all_weights,
        get_name_fn=partial(get_model_name_fn, cfg=all_configs[0]),
    )


@learning_experiment()
def run(helper: LearningExperimentHelper, cfg: MiceV2_FineTuning) -> PaperResults:
    """Main entry point for MICE V2 training experiment."""
    return TrainEvaluateMiceV2(
        helper=helper,
        cfg=cfg,
        get_model_name_fn=get_name_from_tags,
    )
