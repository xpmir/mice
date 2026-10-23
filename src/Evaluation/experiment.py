"""
Two-Stage Information Retrieval Evaluation Experiment.

This module provides the infrastructure to evaluate Information Retrieval (IR) systems,
supporting both first-stage retrievers (BM25, SPLADE) and two-stage systems
integrating Cross-Encoders for re-ranking.

The experiment workflow includes:
1. Configuration of retrieval, indexation, and preprocessing parameters.
2. Building test collections across multiple datasets.
3. First-stage evaluation: Running and caching results for base retrievers.
4. Second-stage evaluation: Applying Cross-Encoders (scorers) to the first-stage runs.
5. Result processing: Aggregating metrics (RR@10, nDCG@10, R@1000) and
   generating standardized outputs (CSV, LaTeX tables).

Usage:
    This module is designed to be invoked via experimaestro:
    `uv run experimaestro run-experiment src/Evaluation/cross-encoders.yaml`
"""

from typing import List, Optional, Dict, Any, Tuple
from attrs import Factory, field
from functools import partial
import pandas as pd
from pathlib import Path

from experimaestro import RunMode, stop_tags
from experimaestro.launcherfinder import find_launcher
from experimaestro.experiments.grid import generate_grid, GridSearch
from experimaestro.core.objects.config import clone

from xpm_torch.huggingface import prepare_hf_model
from xpm_torch.configuration import FabricConfiguration

from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from xpmir.neural.splade import splade_encoder_from_pretrained_hf
from xpmir.papers import configuration
from xpmir.experiments.helpers import NeuralIRExperiment
from xpmir.neural.huggingface import hf_cross_scorer
from xpmir.neural.sentence_transformers import (
    st_cross_scorer,
    st_multivector_scorer,
    STCrossEncoder,
    STMultiVectorEncoder,
)
from xpmir.rankers import scorer_retriever
from xpmir.rankers.scorer import Scorer
from xpmir.evaluation import MultiRunRetrieverFactory

from format import dataframe_to_latex, aggregations
from tests import build_tests, EvaluationsCollection
from configuration import (
    Retrieval,
    Indexation,
    Preprocessing,
    Evaluation,
    ScorerType,
    PlaidConfiguration,
    SeismicConfiguration,
)
from retrievers import (
    splade_retriever,
    bm25_retriever,
    plaid_retriever,
    MicePlaidRetrieverFactory,
)
from index_utils import sync_index_size_metrics
from latency_utils import sync_latency_metrics
from training_utils import (
    add_dataset_aggregations,
    resolve_trained_model_loader,
    validate_and_align_config,
    parse_experiment_ref,
    sync_eval_sparsity_metrics,
    find_candidate_experiment_dirs,
)
from MICE.modeling.mice_v2.modernbert_xpm import (
    ModernBertMiceScorer,
    InitEvalSparsityHook,
)

import logging

from xp_utils import setup_logging

setup_logging(level=logging.INFO)


@configuration()
class BaselinesConfig(NeuralIRExperiment):
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)
    preprocessing: Preprocessing = Factory(Preprocessing)

    scorer_id: GridSearch[str] = ""
    """Identifier for the scorer model: Hugging Face model ID, ST model ID, local folder, or experiment ID."""

    scorer_type: GridSearch[str] = ScorerType.stCrossEncoder.value
    """Type of scorer model: stCrossEncoder, hfCrossScorer, stMultiVectorEncoder, miceV2, or from_experiment."""

    from_experiment: GridSearch[Optional[str]] = None
    """Optional experiment ID (e.g. 'exp_id' or 'exp_id/run_id' to pin a specific run), local model directory path, or HF Hub model ID to load a trained model."""

    from_run_id: Optional[str] = None
    """Optional run ID within from_experiment (e.g. '20260924_180227')."""

    pref_attn_implementation: GridSearch[Optional[str]] = None
    """Preferred attention implementation (e.g. 'flash_attention_2', 'sdpa', 'eager')."""

    precision: Optional[str] = None
    """Precision setting for evaluation (e.g. 'bf16-mixed')."""

    max_length: GridSearch[Optional[int]] = None
    """Maximum length for cross-encoders."""

    max_query_length: GridSearch[Optional[int]] = None
    """Maximum query length for cross-encoders."""

    max_doc_length: GridSearch[Optional[int]] = None
    """Maximum document length for cross-encoders."""

    query_prompt: GridSearch[Optional[str]] = None
    """Optional query prompt template or prefix (e.g. '[Q] ' or 'query: ')."""

    doc_prompt: GridSearch[Optional[str]] = None
    """Optional document prompt template or prefix (e.g. '[D] ' or 'passage: ')."""

    hard_pruning: GridSearch[Optional[bool]] = None
    """Whether to enforce hard token pruning for MICE models during evaluation."""

    hard_prune_threshold: GridSearch[Optional[float]] = None
    """Utility threshold below which tokens are hard-pruned for MICE models."""

    eval_sparsity: bool = False
    """Whether to compute token sparsity and retention ratio during evaluation benchmarks."""

    get_latency: GridSearch[bool] = False
    """Whether to extract evaluation query latency from task logs and sync into aggregated.txt"""

    save_runs: bool = False
    """Whether to save the evaluation runs in the resultspath."""

    retriever: str = ""
    """First-stage candidate retriever model (e.g. 'naver/splade-v3-distilbert' for SPLADE, or 'bm25'/empty for BM25)."""

    use_plaid: GridSearch[bool] = False
    """Whether to use PLAID multi-vector indexing and retrieval."""

    precompute_first_stage: bool = True
    """If True, precomputes and saves candidate runs for first-stage retrievers to disk and reloads them for reranking.
    If False, evaluates first-stage dynamically on-the-fly without saving intermediate candidate runs."""

    seismic: SeismicConfiguration = Factory(SeismicConfiguration)
    """Configuration for Seismic approximate search on SPLADE index."""

    compare_with_baseline: bool = False
    """If True and use_plaid=True with MICE, also evaluates standard reranking without PLAID."""

    plaid: PlaidConfiguration = Factory(PlaidConfiguration)
    """Configuration for PLAID indexing and retrieval."""

    include_recall_100: bool = False
    """Whether to additionally report R@100 alongside CE_MEASURES."""

    evaluation: Evaluation = Factory(Evaluation)

    grid_search: Dict[str, Any] = field(factory=dict)
    """Grid search parameters mapping parameter paths to values or config_dicts combinations."""


def build_first_stage_retrievers(
    cfg: BaselinesConfig,
    launcher_index,
    cfg_tags: Optional[Dict[str, Any]] = None,
    long_launcher_index=None,
) -> List[Tuple[str, Any, List[Any]]]:
    """Instantiate the first-stage candidate retriever (SPLADE or BM25).

    PLAID is never built here; PLAID is an indexing/retrieval engine for multi-vector
    scorers (ColBERT) or token storage for MICE rerankers.
    """
    retriever_name = (cfg.retriever or "").strip()
    if retriever_name and retriever_name.lower() != "bm25":
        logging.info(
            f"Instantiating SPLADE first-stage candidate retriever: '{retriever_name}'"
        )
        prepare_hf_model(retriever_name)
        splade_encoder, retriever_init_tasks = splade_encoder_from_pretrained_hf(
            retriever_name
        )
        return [
            (
                retriever_name,
                partial(
                    splade_retriever,
                    cfg,
                    retriever_name,
                    splade_encoder,
                    launcher_index=launcher_index,
                    long_launcher_index=long_launcher_index,
                    long_evals=getattr(cfg.indexation, "long_evals", None),
                    topk=cfg.retrieval.k,
                    init_tasks=retriever_init_tasks,
                    tags=cfg_tags,
                ),
                retriever_init_tasks,
            )
        ]

    logging.info("Instantiating BM25 first-stage candidate retriever")
    return [
        (
            "bm25",
            partial(
                bm25_retriever,
                cfg,
                "bm25",
                launcher_index=launcher_index,
                tags=cfg_tags,
            ),
            [],
        )
    ]


def resolve_evaluation_scorer(
    exp_ref: str,
    cfg: BaselinesConfig,
    cfg_tags: Optional[Dict[str, Any]] = None,
) -> Tuple[Scorer, List[Any], Any, str]:
    """Resolves an experiment ID, local path, or HF Hub model, aligns configuration,
    and returns the configured Scorer and its init tasks."""
    loader = resolve_trained_model_loader(exp_ref)
    source_model = loader.model

    # Shared configuration validation and alignment with training
    validate_and_align_config(cfg, source_model, tags=cfg_tags, from_ref=exp_ref)

    scorer = clone(source_model)

    is_mice_v2 = isinstance(scorer, ModernBertMiceScorer)
    is_st_mv = isinstance(scorer, STMultiVectorEncoder)
    is_st_ce = isinstance(scorer, STCrossEncoder)

    # Length overrides
    if cfg.max_length is not None:
        if hasattr(scorer, "max_length"):
            scorer.max_length = cfg.max_length
        if hasattr(scorer, "tokenizer") and hasattr(scorer.tokenizer, "max_length"):
            scorer.tokenizer.max_length = cfg.max_length
    if cfg.max_query_length is not None:
        if hasattr(scorer, "max_query_length"):
            scorer.max_query_length = cfg.max_query_length
        if hasattr(scorer, "tokenizer") and hasattr(
            scorer.tokenizer, "max_query_length"
        ):
            scorer.tokenizer.max_query_length = cfg.max_query_length
    if cfg.max_doc_length is not None:
        if hasattr(scorer, "max_doc_length"):
            scorer.max_doc_length = cfg.max_doc_length
        if hasattr(scorer, "tokenizer") and hasattr(scorer.tokenizer, "max_doc_length"):
            scorer.tokenizer.max_doc_length = cfg.max_doc_length

    # Prompt overrides
    if (
        cfg.query_prompt is not None
        and hasattr(scorer, "tokenizer")
        and hasattr(scorer.tokenizer, "query_prompt")
    ):
        scorer.tokenizer.query_prompt = cfg.query_prompt
    if (
        cfg.doc_prompt is not None
        and hasattr(scorer, "tokenizer")
        and hasattr(scorer.tokenizer, "doc_prompt")
    ):
        scorer.tokenizer.doc_prompt = cfg.doc_prompt

    # Attention implementation
    if cfg.pref_attn_implementation is not None:
        if hasattr(scorer, "pref_attn_implementation"):
            scorer.pref_attn_implementation = cfg.pref_attn_implementation
        elif hasattr(scorer, "attn_implementation"):
            scorer.attn_implementation = cfg.pref_attn_implementation

    # MICE v2 pruning overrides
    if is_mice_v2:
        if cfg.hard_pruning is not None:
            scorer.hard_pruning = cfg.hard_pruning
        elif cfg.hard_prune_threshold is not None:
            scorer.hard_pruning = True
        if cfg.hard_prune_threshold is not None:
            scorer.hard_prune_threshold = cfg.hard_prune_threshold

    # Build independent loader
    model_loader = scorer.loader_config(
        loader.path, settings=getattr(loader, "settings", None)
    )
    init_tasks = [stop_tags(model_loader)]

    if cfg.eval_sparsity and is_mice_v2:
        init_tasks.append(
            InitEvalSparsityHook.C(
                scorer=scorer,
                threshold=getattr(scorer, "hard_prune_threshold", 0.5),
            )
        )

    resolved_type_str = (
        "mice_v2"
        if is_mice_v2
        else (
            ScorerType.stMultiVectorEncoder.value
            if is_st_mv
            else (
                ScorerType.stCrossEncoder.value
                if is_st_ce
                else ScorerType.hfCrossScorer.value
            )
        )
    )
    return scorer, init_tasks, model_loader, resolved_type_str


def validate_scorer_init_tasks(scorer: Scorer, init_tasks: List[Any]) -> None:
    """Validates that any ModuleLoader or scorer-dependent hook in init_tasks points to the exact scorer instance being evaluated."""
    from experimaestro.core.objects.config_utils import ConfigWrapper

    for task in init_tasks:
        raw_task = (
            task.value
            if isinstance(task, ConfigWrapper)
            else getattr(task, "task", task)
        )
        if hasattr(raw_task, "value") and isinstance(raw_task.value, Scorer):
            if raw_task.value is not scorer:
                raise ValueError(
                    f"Init task {type(raw_task).__name__} is bound to a different scorer instance "
                    f"(id={id(raw_task.value)}) than the evaluation scorer (id={id(scorer)}). "
                    "Weights would not be loaded into the evaluated model!"
                )
        if hasattr(raw_task, "scorer") and isinstance(raw_task.scorer, Scorer):
            if raw_task.scorer is not scorer:
                raise ValueError(
                    f"Init task {type(raw_task).__name__} hook points to a different scorer instance "
                    f"(id={id(raw_task.scorer)}) than the evaluation scorer (id={id(scorer)})."
                )


def tag_scorer(
    scorer: Scorer,
    target_id: str,
    resolved_type_str: str,
    is_from_exp: bool,
    cfg_tags: Dict[str, Any],
):
    """Tags a scorer instance with model metadata and active grid search parameters."""
    scorer.tag("scorer", target_id)
    scorer.tag("scorer_type", resolved_type_str)
    if is_from_exp:
        exp_id, run_id = parse_experiment_ref(target_id)
        if run_id:
            scorer.tag("from_run_id", run_id)

    for k, v in cfg_tags.items():
        if k not in (
            "scorer",
            "scorer_type",
            "from_experiment",
            "from_run_id",
            "plaid_retriever",
        ):
            scorer.tag(k, v)


def two_stage_reranker(
    documents,
    batch_size: int,
    scorer: Scorer,
    retrievers: MultiRunRetrieverFactory,
    key: str = None,
    chunk_size: Optional[int] = None,
    **kwargs,
):
    """Wraps scorer_retriever to tag the resulting two-stage retriever with plaid_retriever=False."""
    retriever = scorer_retriever(
        documents=documents,
        batch_size=batch_size,
        scorer=scorer,
        retrievers=retrievers,
        key=key,
        chunk_size=chunk_size,
        **kwargs,
    )
    return retriever.tag("plaid_retriever", False)


def evaluate_scorer(
    cfg: BaselinesConfig,
    cfg_tags: Dict[str, Any],
    tests: EvaluationsCollection,
    first_stage_factories: Dict[str, Tuple[Any, List[Any]]],
    launcher_evaluate,
    launcher_index,
    fabric_config,
    long_launcher_index=None,
):
    """Evaluates a single scorer configuration.

    Supports four execution branches:
    1. ColBERT + PLAID (use_plaid=True):
       Evaluated directly as an end-to-end multi-vector retriever with PLAID indexing.
       No first-stage candidate retrieval needed.
    2. ColBERT without PLAID (use_plaid=False):
       Evaluated as a second-stage reranker over candidate runs from first_stage_factories.
       Logs a loud warning recommending PLAID multi-vector retrieval.
    3. MICE + PLAID (use_plaid=True):
       Uses first-stage candidates (SPLADE or BM25) and builds a PLAID index for token storage
       at the specified n_bits. If compare_with_baseline=True, also evaluates standard reranking.
    4. Standard Reranking (use_plaid=False, Cross-Encoders or MICE):
       Full-text cross-encoder reranking over first-stage candidate runs.
    """
    target_id = ""
    is_from_exp = False
    if cfg.from_experiment:
        target_id = cfg.from_experiment
        if cfg.from_run_id and "/" not in target_id:
            target_id = f"{target_id}/{cfg.from_run_id}"
        is_from_exp = True
    elif cfg.scorer_id:
        target_id = cfg.scorer_id
    else:
        raise ValueError(
            "Neither 'scorer_id' nor 'from_experiment' was specified in config."
        )

    scorer_type_enum = ScorerType.from_string(cfg.scorer_type)
    if scorer_type_enum in (ScorerType.from_experiment, ScorerType.miceV2):
        is_from_exp = True
    elif scorer_type_enum == ScorerType.auto:
        exp_id, _ = parse_experiment_ref(target_id)
        target_path = Path(target_id).expanduser()
        if (target_path / "experimaestro.json").exists() or bool(
            find_candidate_experiment_dirs(exp_id)
        ):
            is_from_exp = True

    model_loader = None
    if is_from_exp or scorer_type_enum in (
        ScorerType.miceV2,
        ScorerType.from_experiment,
    ):
        scorer, ce_init_tasks, model_loader, resolved_type_str = (
            resolve_evaluation_scorer(target_id, cfg, cfg_tags=cfg_tags)
        )
    elif scorer_type_enum == ScorerType.hfCrossScorer:
        if not Path(target_id).is_dir():
            prepare_hf_model(target_id)
        scorer, ce_init_tasks = hf_cross_scorer(
            hf_id=target_id,
            max_length=cfg.max_length,
            max_doc_length=cfg.max_doc_length,
            max_query_length=cfg.max_query_length,
            pref_attn_implementation=cfg.pref_attn_implementation,
        )
        resolved_type_str = ScorerType.hfCrossScorer.value
    elif scorer_type_enum == ScorerType.stCrossEncoder:
        if not Path(target_id).is_dir():
            prepare_hf_model(target_id)
        st_tok = None
        has_chat_template = False
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
            has_chat_template = bool(tok.chat_template)
        except Exception as e:
            logging.debug(f"Could not inspect chat_template for {target_id}: {e}")

        effective_max_length = cfg.max_length
        if cfg.max_query_length is not None and cfg.max_doc_length is not None:
            combined_bound = cfg.max_query_length + cfg.max_doc_length + 3
            if effective_max_length is None or effective_max_length > combined_bound:
                effective_max_length = combined_bound
        elif (cfg.max_query_length is not None) ^ (cfg.max_doc_length is not None):
            if effective_max_length is None:
                raise ValueError(
                    f"Incomplete sequence length configuration for '{target_id}': "
                    f"both 'max_query_length' ({cfg.max_query_length}) and 'max_doc_length' ({cfg.max_doc_length}) "
                    "must be specified when 'max_length' is not provided."
                )

        if cfg.max_query_length is not None or cfg.max_doc_length is not None:
            from xpmir.neural.huggingface import HFQueryDocTokenizer

            st_tok = HFQueryDocTokenizer.C(
                model_id=target_id,
                max_query_length=cfg.max_query_length,
                max_doc_length=cfg.max_doc_length,
                max_length=effective_max_length,
                query_prompt=cfg.query_prompt,
                doc_prompt=cfg.doc_prompt,
                pref_attn_implementation=cfg.pref_attn_implementation,
            )
            logging.info(
                f"Configured HFQueryDocTokenizer for '{target_id}' with max_query_length={cfg.max_query_length}, "
                f"max_doc_length={cfg.max_doc_length}, max_length={effective_max_length}"
            )
        elif has_chat_template:
            logging.info(
                f"Model '{target_id}' defines a chat template; using native SentenceTransformers "
                f"tokenization with max_length={effective_max_length}."
            )

        scorer = st_cross_scorer(
            model_id=target_id,
            max_length=effective_max_length,
            max_query_length=cfg.max_query_length,
            max_doc_length=cfg.max_doc_length,
            pref_attn_implementation=cfg.pref_attn_implementation,
            tokenizer=st_tok,
        )
        ce_init_tasks = []
        resolved_type_str = ScorerType.stCrossEncoder.value
    elif scorer_type_enum == ScorerType.stMultiVectorEncoder:
        if not Path(target_id).is_dir():
            prepare_hf_model(target_id)
        query_maxlen = cfg.max_query_length
        doc_maxlen = cfg.max_doc_length or cfg.max_length

        if query_maxlen is not None and query_maxlen > 512:
            logging.warning(
                f"MultiVector scorer '{target_id}': max_query_length ({query_maxlen}) exceeds 512 tokens. "
                "ColBERT queries are typically 32 tokens; consider checking your length config."
            )
        if doc_maxlen is not None and doc_maxlen > 512:
            logging.warning(
                f"MultiVector scorer '{target_id}': max_doc_length ({doc_maxlen}) exceeds 512 tokens. "
                "Consider checking your length config."
            )

        # TODO: Remove explicit `dim` override once legacy runs with dim=128 are deprecated
        dim = cfg.plaid.dim
        if dim is not None:
            logging.info(
                f"MultiVector scorer '{target_id}': using explicit projection dimension dim={dim}"
            )

        scorer = st_multivector_scorer(
            model_id=target_id,
            dim=dim,
            query_maxlen=query_maxlen,
            doc_maxlen=doc_maxlen,
            pref_attn_implementation=cfg.pref_attn_implementation,
        )
        ce_init_tasks = []
        resolved_type_str = ScorerType.stMultiVectorEncoder.value
    else:
        raise ValueError(f"Unsupported scorer_type: '{scorer_type_enum}'")

    is_plaid = bool(cfg.use_plaid)
    is_colbert = resolved_type_str == ScorerType.stMultiVectorEncoder.value
    is_mice = resolved_type_str == "mice_v2"

    tag_scorer(
        scorer,
        target_id=target_id,
        resolved_type_str=resolved_type_str,
        is_from_exp=is_from_exp,
        cfg_tags=cfg_tags,
    )
    validate_scorer_init_tasks(scorer, ce_init_tasks)

    # -------------------------------------------------------------------------
    # Branch 1: ColBERT + PLAID (End-to-End Multi-Vector Retrieval)
    # ColBERT model is evaluated directly with PLAID index and retrieval.
    # No candidate retriever (SPLADE or BM25) is needed or used.
    # -------------------------------------------------------------------------
    if is_plaid and is_colbert:
        logging.info(
            f"Scheduling PLAID multi-vector retrieval evaluation for ColBERT scorer: {target_id}"
        )
        plaid_retriever_factory = partial(
            plaid_retriever,
            cfg,
            target_id,
            scorer,
            launcher_index=launcher_index,
            long_launcher_index=long_launcher_index,
            long_evals=getattr(cfg.indexation, "long_evals", None),
            topk=cfg.retrieval.k,
            init_tasks=ce_init_tasks,
            tags=cfg_tags,
        )
        tests.evaluate_retriever(
            plaid_retriever_factory,
            launcher=launcher_evaluate,
            init_tasks=ce_init_tasks,
            fabric_config=fabric_config,
            with_run=cfg.save_runs,
        )
        return

    # All subsequent branches require a first-stage candidate retriever
    if not first_stage_factories:
        raise ValueError(
            f"Scorer '{target_id}' ({resolved_type_str}) requires a first-stage candidate "
            "retriever, but none was configured. Please specify 'retriever: ...' in your config."
        )

    for retriever_name, (
        run_retriever_factory,
        retriever_init_tasks,
    ) in first_stage_factories.items():
        # Build unified init_tasks for downstream evaluation
        eval_init_tasks = list(ce_init_tasks)
        if not cfg.precompute_first_stage and retriever_init_tasks:
            for task in retriever_init_tasks:
                if task not in eval_init_tasks:
                    eval_init_tasks.append(task)
            logging.info(
                f"Attached {len(retriever_init_tasks)} live first-stage retriever init tasks to evaluation: "
                f"{[type(t).__name__ for t in retriever_init_tasks]}"
            )

        # ---------------------------------------------------------------------
        # Branch 2: ColBERT without PLAID (Second-Stage Reranker over Candidates)
        # Evaluates ColBERT by re-ranking first-stage candidate runs.
        # ---------------------------------------------------------------------
        if is_colbert and not is_plaid:
            logging.warning(
                f"Evaluating ColBERT model '{target_id}' as a second-stage reranker over first-stage "
                f"candidates from '{retriever_name}' (use_plaid=False). "
                "If you intended end-to-end multi-vector retrieval, set 'use_plaid: true'."
            )
            two_stage_retriever_factory = partial(
                two_stage_reranker,
                batch_size=cfg.retrieval.batch_size,
                chunk_size=getattr(cfg.retrieval, "chunk_size", None),
                scorer=scorer,
                retrievers=run_retriever_factory,
            )
            tests.evaluate_retriever(
                two_stage_retriever_factory,
                launcher=launcher_evaluate,
                init_tasks=eval_init_tasks,
                fabric_config=fabric_config,
                with_run=cfg.save_runs,
            )
            continue

        # ---------------------------------------------------------------------
        # Branch 3: MICE + PLAID (Token Storage Indexing & Candidate Reranking)
        # First-stage retriever provides candidates (SPLADE or BM25).
        # PLAID is used as token storage for MICE embeddings (at n_bits).
        # ---------------------------------------------------------------------
        if is_plaid and is_mice:
            # Baseline evaluation without PLAID (if requested)
            if cfg.compare_with_baseline:
                logging.info(
                    f"Scheduling baseline 2-stage evaluation: first_stage={retriever_name}, "
                    f"scorer={target_id} ({resolved_type_str})"
                )
                two_stage_retriever_factory = partial(
                    two_stage_reranker,
                    batch_size=cfg.retrieval.batch_size,
                    chunk_size=getattr(cfg.retrieval, "chunk_size", None),
                    scorer=scorer,
                    retrievers=run_retriever_factory,
                )
                tests.evaluate_retriever(
                    two_stage_retriever_factory,
                    launcher=launcher_evaluate,
                    init_tasks=eval_init_tasks,
                    fabric_config=fabric_config,
                    with_run=cfg.save_runs,
                )

            logging.info(
                f"Scheduling MICE-PLAID evaluation for: {target_id} (n_bits={cfg.plaid.n_bits}) "
                f"over first_stage={retriever_name}"
            )
            mice_plaid_factory = MicePlaidRetrieverFactory(
                mice_model=scorer,
                first_stage_factory=run_retriever_factory,
                topk=cfg.retrieval.k,
                batchsize=cfg.retrieval.batch_size,
                n_ivf_probe=cfg.plaid.n_ivf_probe,
                n_full_scores=cfg.plaid.n_full_scores,
                max_query_length=cfg.plaid.max_query_length,
                max_doc_length=cfg.plaid.max_doc_length,
                chunk_size=getattr(cfg.retrieval, "chunk_size", None),
            )
            mice_plaid_factory.build_indexes(
                evaluations=tests,
                cfg=cfg.plaid,
                batch_size=cfg.indexation.batch_size or cfg.retrieval.batch_size,
                seed=0,
                launcher_index=launcher_index,
                init_tasks=ce_init_tasks,
                long_launcher_index=long_launcher_index,
                long_evals=getattr(cfg.indexation, "long_evals", None),
            )
            tests.evaluate_retriever(
                retriever=mice_plaid_factory,
                launcher=launcher_evaluate,
                init_tasks=eval_init_tasks,
                fabric_config=fabric_config,
                with_run=cfg.save_runs,
            )

            if cfg.plaid.use_colbert_first_stage:
                logging.info(
                    f"Scheduling MICE-ColBERT first-stage evaluation for: {target_id}"
                )
                colbert_runs = tests.evaluate_retriever(
                    mice_plaid_factory.colbert_retriever,
                    launcher=launcher_evaluate,
                    init_tasks=ce_init_tasks,
                    with_run=True,
                    fabric_config=fabric_config,
                )
                colbert_run_factory = MultiRunRetrieverFactory.from_results(
                    "mice-colbert", colbert_runs
                )
                mice_colbert_rerank_factory = MicePlaidRetrieverFactory(
                    mice_model=scorer,
                    first_stage_factory=colbert_run_factory,
                    topk=cfg.retrieval.k,
                    batchsize=cfg.retrieval.batch_size,
                    n_ivf_probe=cfg.plaid.n_ivf_probe,
                    n_full_scores=cfg.plaid.n_full_scores,
                    max_query_length=cfg.plaid.max_query_length,
                    max_doc_length=cfg.plaid.max_doc_length,
                    chunk_size=getattr(cfg.retrieval, "chunk_size", None),
                )
                mice_colbert_rerank_factory.build_indexes(
                    evaluations=tests,
                    cfg=cfg.plaid,
                    batch_size=cfg.retrieval.batch_size,
                    seed=0,
                    launcher_index=launcher_index,
                    init_tasks=ce_init_tasks,
                    long_launcher_index=long_launcher_index,
                    long_evals=cfg.indexation.long_evals,
                )
                tests.evaluate_retriever(
                    retriever=mice_colbert_rerank_factory,
                    launcher=launcher_evaluate,
                    init_tasks=ce_init_tasks,
                    fabric_config=fabric_config,
                    with_run=cfg.save_runs,
                )
            continue

        # ---------------------------------------------------------------------
        # Branch 4: Standard Two-Stage Reranking (Cross-Encoders or MICE without PLAID)
        # Evaluates full-text cross-encoder reranking over first-stage candidate runs.
        # ---------------------------------------------------------------------
        logging.info(
            f"Scheduling standard 2-stage evaluation: first_stage={retriever_name}, "
            f"scorer={target_id} ({resolved_type_str})"
        )
        two_stage_retriever_factory = partial(
            two_stage_reranker,
            batch_size=cfg.retrieval.batch_size,
            chunk_size=getattr(cfg.retrieval, "chunk_size", None),
            scorer=scorer,
            retrievers=run_retriever_factory,
        )
        tests.evaluate_retriever(
            two_stage_retriever_factory,
            launcher=launcher_evaluate,
            init_tasks=eval_init_tasks,
            fabric_config=fabric_config,
            with_run=cfg.save_runs,
        )


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: BaselinesConfig) -> PaperResults:
    all_configs, all_tags = generate_grid(cfg)
    if not all_configs:
        raise ValueError(
            f"generate_grid returned 0 configurations for {getattr(cfg, 'id', 'evaluation')}. "
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

    from experimaestro.launchers.direct import DirectLauncher
    from experimaestro.connectors.local import LocalConnector

    def safe_find_launcher(reqs):
        if not reqs or not reqs.strip():
            return DirectLauncher(connector=LocalConnector.instance())
        return find_launcher(reqs) or DirectLauncher(
            connector=LocalConnector.instance()
        )

    base_config = all_configs[0]
    launcher_evaluate = safe_find_launcher(base_config.retrieval.requirements)
    launcher_index = safe_find_launcher(base_config.indexation.requirements)
    long_reqs_index = getattr(base_config.indexation, "long_requirements", None)
    long_launcher_index = (
        safe_find_launcher(long_reqs_index) if long_reqs_index else None
    )
    launcher_preprocessing = safe_find_launcher(base_config.preprocessing.requirements)

    fabric_config = (
        base_config.evaluation.fabric.get_config()
        if hasattr(base_config.evaluation, "fabric")
        else (
            FabricConfiguration.C(precision=base_config.precision)
            if getattr(base_config, "precision", None)
            else None
        )
    )

    long_reqs = getattr(base_config.retrieval, "long_requirements", None)
    long_launcher = safe_find_launcher(long_reqs) if long_reqs else None

    if base_config.include_recall_100:
        base_config.evaluation.include_recall_100 = True

    # Build tests collections ONCE
    tests = build_tests(
        base_config.evaluation,
        launcher=launcher_preprocessing,
        default_launcher=launcher_evaluate,
        long_launcher=long_launcher,
        long_evals=getattr(base_config.retrieval, "long_evals", None),
        check_docs=False,
        blocking=base_config.preprocessing.blocking_download,
    )

    has_scorers = any(bool(c.scorer_id or c.from_experiment) for c in all_configs)

    if not has_scorers:
        logging.info(
            "No scorers configured; evaluating retrievers directly across grid search."
        )
        for cfg_item, cfg_tags in zip(all_configs, all_tags):
            retrievers = build_first_stage_retrievers(
                cfg_item, launcher_index, cfg_tags=cfg_tags
            )
            for retriever_name, retriever_factory, retriever_init_tasks in retrievers:
                tests.evaluate_retriever(
                    retriever_factory,
                    launcher=launcher_evaluate,
                    init_tasks=retriever_init_tasks,
                    with_run=cfg_item.save_runs,
                    fabric_config=fabric_config,
                )
    else:
        # Check if any scorer requires first-stage candidate retrieval
        # ColBERT with PLAID is an end-to-end retriever and does not require candidates.
        needs_first_stage = False
        for c in all_configs:
            st_type = ScorerType.from_string(c.scorer_type) if c.scorer_type else None
            is_colbert_plaid = (
                st_type == ScorerType.stMultiVectorEncoder
            ) and c.use_plaid
            if not is_colbert_plaid:
                needs_first_stage = True
                break

        first_stage_factories: Dict[str, Tuple[Any, List[Any]]] = {}

        if needs_first_stage:
            all_retrievers = build_first_stage_retrievers(
                base_config,
                launcher_index,
                long_launcher_index=long_launcher_index,
            )
            logging.info(
                f"Evaluating first-stage candidate retrievers: {[name for name, _, _ in all_retrievers]}"
            )

            # Evaluate first-stage retrievers and prepare factories for downstream reranking
            for (
                retriever_name,
                retriever_factory,
                retriever_init_tasks,
            ) in all_retrievers:
                if base_config.precompute_first_stage:
                    eval_results = tests.evaluate_retriever(
                        retriever_factory,
                        launcher=launcher_evaluate,
                        init_tasks=retriever_init_tasks,
                        with_run=True,
                        fabric_config=fabric_config,
                    )
                    run_retriever_factory = MultiRunRetrieverFactory.from_results(
                        retriever_name, eval_results
                    )
                else:
                    eval_results = tests.evaluate_retriever(
                        retriever_factory,
                        launcher=launcher_evaluate,
                        init_tasks=retriever_init_tasks,
                        with_run=base_config.save_runs,
                        fabric_config=fabric_config,
                    )
                    run_retriever_factory = retriever_factory

                first_stage_factories[retriever_name] = (
                    run_retriever_factory,
                    retriever_init_tasks,
                )
                logging.info(
                    f"First-stage candidate setup done for {retriever_name} (precompute={base_config.precompute_first_stage})"
                )
        else:
            logging.info(
                "All scorers are ColBERT + PLAID; skipping first-stage candidate retrieval."
            )

        # Evaluate scorers across all grid configurations
        for cfg_item, cfg_tags in zip(all_configs, all_tags):
            evaluate_scorer(
                cfg_item,
                cfg_tags=cfg_tags,
                tests=tests,
                first_stage_factories=first_stage_factories,
                launcher_evaluate=launcher_evaluate,
                launcher_index=launcher_index,
                fabric_config=fabric_config,
                long_launcher_index=long_launcher_index,
            )

    # Wait for all tasks to complete
    helper.xp.wait()

    if any(getattr(c, "eval_sparsity", False) for c in all_configs):
        sync_eval_sparsity_metrics(tests)

    # When use_plaid is True, sync on-disk index sizes from the dependency graph
    is_plaid = any(c.use_plaid for c in all_configs)
    if is_plaid:
        sync_index_size_metrics(tests)

    if any(c.get_latency for c in all_configs):
        sync_latency_metrics(tests)

    df = tests.to_dataframe()

    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    # Identify available metric columns and convert to numeric
    all_metric_cols = [col for col in df.columns if col[0] == "metric"]
    df[all_metric_cols] = df[all_metric_cols].apply(pd.to_numeric, downcast="float")

    # Flatten MultiIndex columns and remove duplicates (e.g., 'dataset' might be both a tag and a column)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if col[1] else col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    cols_to_drop = [col for col in df.columns if "index_doc" in str(col).lower()]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if not has_scorers:
        tag_names = ["first_stage"]
    else:
        tag_names = [
            "first_stage",
            "scorer",
            "scorer_type",
            "from_experiment",
            "from_run_id",
            "plaid_retriever",
        ]
    for t_dict in all_tags:
        for k in t_dict.keys():
            if k not in tag_names:
                tag_names.append(k)
    group_by_tags = [col for col in tag_names if col in df.columns]

    # Add aggregations
    df = add_dataset_aggregations(
        df,
        group_by_cols=group_by_tags,
        aggregations=aggregations,
        add_mean=True,
    )

    # Reorder columns: tags first, metrics after
    tag_cols = ["dataset"] + group_by_tags
    tag_cols = [c for c in tag_cols if c in df.columns]
    metric_cols = [c for c in df.columns if c not in tag_cols]
    df = df[tag_cols + metric_cols]

    logging.info(f"Final DataFrame:\n{df}")

    # save results
    if helper.xp.run_mode == RunMode.NORMAL:
        if not helper.xp.resultspath.exists():
            helper.xp.resultspath.mkdir(parents=True, exist_ok=True)

        output_file = helper.xp.resultspath / "results.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

        # Generate and save LaTeX table
        metric_col = "nDCG@10" if has_scorers else "R@100"
        if metric_col not in df.columns:
            for fallback in ["nDCG@10", "R@100", "R@1000", "RR@10", "Success@5"]:
                if fallback in df.columns:
                    metric_col = fallback
                    break

        latex_table = dataframe_to_latex(
            df,
            caption="Evaluation Results",
            label="tab:eval_results",
            sig_df=None,
            metric_col=metric_col,
        )
        latex_output_file = helper.xp.resultspath / "results.tex"
        with open(latex_output_file, "w") as f:
            f.write(latex_table)
        logging.info(f"LaTeX table saved to {latex_output_file}")
    else:
        logging.info("Dry-run mode: skipping saving results to disk")
