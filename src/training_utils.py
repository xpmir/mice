"""Utility functions for training and result processing."""

import logging
import shutil
import json
from typing import Callable, Optional, Any, Dict, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
import re
from functools import lru_cache
import torch
from torch import Tensor
from experimaestro import RunMode, Param, field
from experimaestro.experiments.mockmodule import FakeModule
from experimaestro.launcherfinder import find_launcher
from experimaestro.annotations import tags as get_tags
from xpm_torch.context import InitializationHook, Context
from xpm_torch.trainers import TrainerContext
from xpmir.rankers import AbstractModuleScorer

from experimaestro.settings import get_settings
from xpm_torch import ModuleLoader
from xpm_torch.huggingface import TorchHFHub
from xpm_torch.trainers import LossTrainer
from xpm_torch.learner import LearnerOutput

from xpmir.letor.trainers.batchwise import BatchwiseTrainer
from xpmir.letor.trainers.pairwise import PairwiseTrainer
from xpmir.evaluation import EvaluationsCollection
from xpm_torch.losses.batchwise import SoftmaxCrossEntropy
from xpm_torch.losses.pairwise import HingeLoss, PointwiseCrossEntropyLoss

from xpmir.datasets.samplers import (
    msmarco_colbertv2_annotated,
    msmarco_rankdistillm_colbert_top50,
    msmarco_hofstaetter_ensemble_hard_negatives,
    msmarco_v1_docpairs_efficient_sampler,
)
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler
from xpmir.letor.distillation.listwise import (
    ADR_MSE,
    DistillRankNetLoss,
    DistillationListwiseTrainer,
    ListwiseSoftmaxCrossEntropy,
    ListwiseHingeLoss,
    ListwiseBCE,
)
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.letor.distillation.pointwise import (
    DistillationPointwiseLoss,
    PointwiseDistillationSampler,
    PointwiseDistillationTrainer,
    PointwiseMSELoss,
)
from datamaestro.data.huggingface import FlattenAndShuffleDataset

from samplers import prepare_collection
from configuration import Losses, CE_FineTuning, ScorerType
from samplers import msmarco_rankdistillm_sampled_colbert50
from format import (
    aggregations,
    aggregation_hf,
    dataframe_to_latex,
    loss_names,
)

from xpm_torch.utils.utils import Initializable

logger = logging.getLogger(__name__)


class FreezeParameterHook(InitializationHook):
    """Freezes model parameters matching a regex pattern prior to optimizer initialization."""

    pattern: Param[str] = field(default=r"(?i)norm", ignore_default=True)
    """Regex pattern for parameters to freeze."""

    def before(self, context: Context):
        if hasattr(context, "model"):
            model = context.model
        elif hasattr(context, "state") and hasattr(context.state, "model"):
            model = context.state.model
        else:
            raise AttributeError(
                f"[FreezeParameterHook] Cannot resolve model from context of type {type(context).__name__}. "
                "Expected 'context.model' or 'context.state.model'."
            )

        if isinstance(model, Initializable):
            model.initialize()

        regex = re.compile(self.pattern)
        frozen_names = []
        total_elements = 0

        for name, param in model.named_parameters():
            if regex.search(name):
                param.requires_grad = False
                frozen_names.append(name)
                total_elements += param.numel()

        if frozen_names:
            logger.info(
                f"[FreezeParameterHook] Successfully frozen {len(frozen_names)} parameter tensors "
                f"({total_elements:,} elements) matching pattern '{self.pattern}' (e.g '{frozen_names[-1]}')"
            )
        else:
            logger.warning(
                f"[FreezeParameterHook] WARNING: Pattern '{self.pattern}' matched 0 parameters in model {type(model).__name__}!"
            )


def get_task_by_tags(tasks: list, tags: dict):
    """Return the first task in tasks that has all its tags matching the given tags."""

    for task in tasks:
        task_tags = get_tags(task)
        if not task_tags:
            continue
        # Check if all given tags are present and match in the task's tags
        if all(str(task_tags.get(tag)) == str(value) for tag, value in tags.items()):
            return task
    return None


class PearsonLoss(DistillationPointwiseLoss):
    """Pearson correlation loss: L = 1 - r(student, teacher).

    Computes the sample Pearson correlation coefficient across the batch and
    returns 1 - r as the loss. Scale and shift invariant.

    Attributes:
        eps: Numerical stability epsilon added to variance to prevent division by zero
             and gradient explosion when student variance collapses (default: 1e-6).
    """

    NAME = "Pearson"

    eps: Param[float] = field(default=1e-6, ignore_default=True)
    """Small epsilon added to variance to prevent division by zero or NaN gradients."""

    def initialize(self, ranker: AbstractModuleScorer):
        super().initialize(ranker)

    def compute(
        self,
        student_scores: Tensor,
        teacher_scores: Tensor,
        context: Optional[TrainerContext] = None,
    ) -> Tensor:
        """Compute the Pearson correlation loss (1 - r) between student and teacher scores.

        Arguments:
            student_scores: A batch tensor of student predictions (B,) or (B, 1)
            teacher_scores: A batch tensor of teacher scores (B,) or (B, 1)
        """
        s = student_scores.view(-1)
        y = teacher_scores.view(-1)

        # Center both scores
        s_centered = s - s.mean()
        y_centered = y - y.mean()

        # Compute sample variances with epsilon to prevent division by zero or NaN gradients
        s_var = torch.mean(s_centered**2)
        y_var = torch.mean(y_centered**2)

        s_std = torch.sqrt(s_var + self.eps)
        y_std = torch.sqrt(y_var + self.eps)

        cov = torch.mean(s_centered * y_centered)
        r = cov / (s_std * y_std)

        # Clamp r to [-1.0, 1.0] for numerical stability against floating point inaccuracies
        r = torch.clamp(r, -1.0, 1.0)

        return 1.0 - r


def build_trainer(cfg: CE_FineTuning) -> LossTrainer:
    """
    Builds a trainer based on the configuration's loss function.
    """
    try:
        loss_member = Losses(cfg.learner.loss)
    except ValueError:
        raise ValueError(
            f"Unknown loss function: {cfg.learner.loss}. Accepted values are: {[e.value for e in Losses]}"
        )
    ### Pointwise losses
    if loss_member is Losses.BCE:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        return PairwiseTrainer.C(
            lossfn=PointwiseCrossEntropyLoss.C(),
            sampler=msmarco_v1_docpairs_efficient_sampler(
                sample_rate=cfg.learner.sample_rate,
                sample_max=cfg.learner.sample_max,
                launcher=launcher_preprocessing,
            ),
            batch_size=cfg.learner.optimization.batch_size,
        )

    ### Pairwise losses ###
    elif loss_member is Losses.hingeLoss:
        launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
        return PairwiseTrainer.C(
            lossfn=HingeLoss.C(),
            sampler=msmarco_v1_docpairs_efficient_sampler(
                sample_rate=cfg.learner.sample_rate,
                sample_max=cfg.learner.sample_max,
                launcher=launcher_preprocessing,
            ),
            batch_size=cfg.learner.optimization.batch_size,
        )

    ### Pairwise distillation losses ###
    elif loss_member is Losses.marginMSE:
        return DistillationPairwiseTrainer.C(
            batch_size=cfg.learner.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
        )

    ### Listwise losses with ColBERT negatives ###
    elif loss_member in (
        Losses.BCE_Colbertv2Neg,
        Losses.hingeLoss_Colbertv2Neg,
        Losses.infoNCE_Colbertv2Neg,
        Losses.infoNCE_RankDistiLLM,
    ):
        passages_per_query = 8
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch:
            batch_size = batch_size // passages_per_query
            logger.warning(
                f"normalized batch size to {batch_size} to get {batch_size * passages_per_query} docs per batch"
            )
        else:
            logger.warning(
                f"Not normalizing docs per batch, {passages_per_query} docs x {batch_size} = {batch_size * passages_per_query} docs per batch"
            )

        if loss_member in (Losses.infoNCE_Colbertv2Neg, Losses.infoNCE_RankDistiLLM):
            loss_fn = ListwiseSoftmaxCrossEntropy.C()
        elif loss_member is Losses.hingeLoss_Colbertv2Neg:
            loss_fn = ListwiseHingeLoss.C()
        else:
            loss_fn = ListwiseBCE.C()

        if loss_member is Losses.infoNCE_RankDistiLLM:
            sampler = msmarco_rankdistillm_sampled_colbert50(
                passages_per_query=passages_per_query
            )
        else:
            sampler = msmarco_colbertv2_annotated(passages_per_query=passages_per_query)

        return DistillationListwiseTrainer.C(
            sampler=sampler,
            lossfn=loss_fn,
            batch_size=batch_size,
        )

    ### Listwise distillation losses ###
    elif loss_member is Losses.distillRankNET:
        logger.warning(
            "Using loss function DistillRankNET, switching to batch size = 1 (i.e. 50 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=DistillRankNetLoss.C(),
        )

    elif loss_member is Losses.ADR_MSE:
        logger.warning(
            "Using loss function ADR_MSE, switching to batch size = 1 (i.e. 50 passages per batch)."
        )
        return DistillationListwiseTrainer.C(
            batch_size=1,
            sampler=msmarco_rankdistillm_colbert_top50(),
            lossfn=ADR_MSE.C(),
        )
    ## Not using this one
    elif loss_member is Losses.infoNCE:
        batch_size = cfg.learner.optimization.batch_size

        if cfg.normalize_docs_per_batch:
            batch_size = int(np.sqrt(batch_size))
            passages_per_batch = batch_size * batch_size
            logger.warning(
                f"normalized batch size to {batch_size} to get {passages_per_batch} docs per batch"
            )
        else:
            passages_per_batch = batch_size * batch_size
            logger.warning(
                f"Not normalizing docs per batch for InfoNCE, {batch_size}**2 docs = {passages_per_batch} docs per batch"
            )

        return BatchwiseTrainer.C(
            sampler=PairwiseInBatchNegativesSampler.C(
                sampler=msmarco_v1_docpairs_efficient_sampler(),
            ),
            lossfn=SoftmaxCrossEntropy.C(),
            batch_size=batch_size,
            hooks=[],
        )

    elif loss_member in (Losses.MSE_mixedbread_large, Losses.pearson_mixedbread_large):
        # Load the Ettin v1 training configurations
        from datamaestro_ir.config.cross_encoder.ettin_reranker_v1_data import CONFIGS

        # Filter out excluded subsets
        excluded = set(cfg.ettin_subset_exclude)
        selected_configs = [name for name in CONFIGS if name not in excluded]

        if not selected_configs:
            raise ValueError("All Ettin subsets are excluded. Nothing to train on!")

        shuffled_dataset_config = build_ettin_reranker_v1_data(
            tuple(selected_configs),
            seed=getattr(cfg, "seed", 42),
            requirements=cfg.preprocessing.requirements,
            download=True,
        )

        scorer_type = getattr(cfg, "scorer_type", ScorerType.stCrossEncoder.value)
        if isinstance(scorer_type, ScorerType):
            scorer_type = scorer_type.value

        if loss_member is Losses.pearson_mixedbread_large:
            eps = getattr(cfg, "pearson_eps", 1e-6)
            logger.info(
                f"Using PearsonLoss (1 - r) with eps={eps} for {scorer_type} distillation on Ettin dataset."
            )
            lossfn = PearsonLoss.C(eps=eps)
        elif scorer_type == ScorerType.stMultiVectorEncoder.value:
            teacher_scale = 0.02
            teacher_bias = 0.70
            logger.info(
                f"STMultiVectorEncoder detected: scaling teacher scores with scale={teacher_scale}, bias={teacher_bias} to align with MeanMaxSim range."
            )
            lossfn = PointwiseMSELoss.C(
                teacher_scale=teacher_scale, teacher_bias=teacher_bias
            )
        else:
            lossfn = PointwiseMSELoss.C()

        return PointwiseDistillationTrainer.C(
            sampler=PointwiseDistillationSampler.C(dataset=shuffled_dataset_config),
            lossfn=lossfn,
            batch_size=cfg.learner.optimization.batch_size,
        )

    else:
        raise NotImplementedError(
            f"Loss function {cfg.learner.loss} is not implemented yet."
        )


@lru_cache(maxsize=None)
def build_ettin_reranker_v1_data(
    selected_configs: tuple[str],
    seed: int = 42,
    requirements: str = None,
    download: bool = True,
):
    """Builds the Ettin v1 reranker dataset based on selected configurations."""

    # Using streaming=false to download and cache the data locally as requested
    samples_list = []
    for name in selected_configs:
        dataset_id = (
            f"cross_encoder.ettin_reranker_v1_data[name={name},streaming=false]"
        )
        dataset_config = prepare_collection(dataset_id, download=download)
        samples_list.append(dataset_config)

    # Submit the preprocessing task to the indexing launcher
    launcher_indexing = find_launcher(requirements)

    return FlattenAndShuffleDataset.C(samples=samples_list, seed=seed).submit(
        launcher=launcher_indexing
    )


def save_raw_results(df: pd.DataFrame, resultspath: Path):
    """Formats and saves the raw experimental results to disk."""
    if not resultspath.exists():
        resultspath.mkdir(parents=True, exist_ok=True)

    output_file = resultspath / "raw_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Raw results saved to {output_file}")


def identify_best_models(
    df: pd.DataFrame, dataset: str, metric: str, group_by_tags: list
) -> pd.DataFrame:
    """Identifies the best model for each configuration based on a specific dataset and metric."""
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        logger.warning(f"Dataset {dataset} not found in results for model selection")
        return pd.DataFrame()

    if ("metric", metric) in df.columns:
        metric_col = [("metric", metric)]
    elif metric in df.columns:
        metric_col = [metric]
    else:
        logger.warning(f"Metric {metric} not found in columns")
        return pd.DataFrame()

    best_models_indices = subset.groupby(group_by_tags, dropna=False)[
        metric_col
    ].idxmax()

    if isinstance(best_models_indices, pd.DataFrame):
        best_models_indices = best_models_indices.iloc[:, 0]

    return subset.loc[best_models_indices]


def add_dataset_aggregations(
    df: pd.DataFrame,
    group_by_cols: list = None,
    aggregations: dict[str, list[str]] = None,
    add_mean: bool = True,
) -> pd.DataFrame:
    """Adds aggregate rows (e.g., mean across datasets) to the results dataframe."""

    # Handle the mean aggregation by recursion
    if add_mean:
        unique_datasets = sorted(df["dataset"].unique().tolist())
        if len(unique_datasets) > 1:
            aggregations = (aggregations or {}).copy()
            if "mean" not in aggregations:
                aggregations["mean"] = unique_datasets
        return add_dataset_aggregations(df, group_by_cols, aggregations, add_mean=False)

    if not aggregations:
        return df

    new_rows = []

    for agg_name, datasets in aggregations.items():
        present_datasets = df["dataset"].unique()
        missing = [ds for ds in datasets if ds not in present_datasets]
        if missing:
            logger.debug(
                f"Aggregation {agg_name} skipped because the following datasets are missing globally: {missing}"
            )
            continue

        mask = df["dataset"].isin(datasets)
        subset = df[mask]

        if group_by_cols:
            grouped = subset.groupby(group_by_cols, dropna=False)
            # Ensure to compute the means if and only if ALL datasets in the aggregation are present
            # for each specific group (model)
            counts = grouped["dataset"].nunique()
            agg = grouped.mean(numeric_only=True)
            agg = agg[counts == len(datasets)].reset_index()
        else:
            # Global mean
            if subset["dataset"].nunique() == len(datasets):
                agg = subset.mean(numeric_only=True).to_frame().T
            else:
                agg = pd.DataFrame()

        if not agg.empty:
            agg["dataset"] = agg_name
            new_rows.append(agg)

    if new_rows:
        new_rows = [row[row.columns.intersection(df.columns)] for row in new_rows]
        return pd.concat([df] + new_rows, ignore_index=True)

    return df


def format_model_results(
    model_df: pd.DataFrame, aggregations: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processes results for a single model, adding aggregations and formatting for MD."""
    if isinstance(model_df.columns, pd.MultiIndex):
        flat_cols = []
        for col in model_df.columns:
            if col[0] == "metric":
                flat_cols.append(col[1])
            elif col[0] == "dataset":
                flat_cols.append("dataset")
            else:
                flat_cols.append("_".join(str(x) for x in col if x))
        model_df.columns = flat_cols

    metrics_to_show = ["Success@5", "RR@10", "nDCG@10"]
    cols_to_keep = [c for c in ["dataset"] + metrics_to_show if c in model_df.columns]

    if not cols_to_keep:
        return pd.DataFrame(), pd.DataFrame()

    results = model_df[cols_to_keep].copy()
    agg_names = list(aggregations.keys())
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    results[numeric_cols] = results[numeric_cols] * 100

    csv_results = results.copy()
    csv_results[numeric_cols] = csv_results[numeric_cols].round(2)

    md_results = results.copy()
    for col in numeric_cols:
        md_results[col] = md_results[col].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else x
        )

    for agg_name in agg_names:
        idx = md_results[md_results["dataset"] == agg_name].index
        if not idx.empty:
            md_results.loc[idx, "dataset"] = f"**{agg_name}**"
            for col in numeric_cols:
                md_results.loc[idx, col] = md_results.loc[idx, col].apply(
                    lambda x: f"**{x}**" if pd.notna(x) else x
                )

    return csv_results, md_results


def export_model(
    best_tags: dict,
    model_name: str,
    csv_results: pd.DataFrame,
    md_results: pd.DataFrame,
    learners: list,
    all_weights: list,
    best_cfg: CE_FineTuning,
    resultspath: Path,
    save_runs: bool = False,
    tests: EvaluationsCollection = None,
):
    """Exports all artifacts (weights, logs, readme, config) for a best model."""
    models_path = resultspath / "models"
    best_model_path = models_path / model_name

    old_readme = best_model_path / "README.md"
    if old_readme.exists():
        logger.warning(
            f"Pre-existing README.md found in {best_model_path}; removing it to ensure fresh generation."
        )
        old_readme.unlink()

    best_model_path.mkdir(parents=True, exist_ok=True)

    csv_results.to_csv(best_model_path / "results.csv", index=False)
    logger.info(f"Model results saved to {best_model_path / 'results.csv'}")

    eval_only_tags = {
        "validation",
        "metric",
        "first_stage",
        "checkpoint",
        "plaid_retriever",
    }
    learner_tags = {k: v for k, v in best_tags.items() if k not in eval_only_tags}
    logger.info(f"Looking up Learner task with tags: {learner_tags}")
    best_model_learner = get_task_by_tags(learners, learner_tags)
    if best_model_learner:
        logger.info(f"Found matching Learner task: {best_model_learner.jobpath.name}")
        tb_path = best_model_learner.logpath
        if not tb_path.exists():
            logger.warning(f"Learner logpath does not exist: {tb_path}")
    else:
        logger.warning(
            f"Could not find Learner task matching tags {learner_tags} among {len(learners)} registered learners"
        )

    tb_log_path = (
        best_model_learner.logpath
        if (best_model_learner and best_model_learner.logpath.exists())
        else None
    )

    # Export to HF format using ExportAction
    best_model_loader = get_task_by_tags(all_weights, best_tags)
    if best_model_loader:
        if hasattr(best_model_loader, "model"):
            model_obj = best_model_loader.model
            model_obj._last_export_dir = best_model_path
            model_obj.model_name = model_name

            retriever_name = (
                best_tags.get("first_stage")
                or (best_cfg.retriever if best_cfg else None)
                or "bm25"
            )
            if not retriever_name or str(retriever_name).lower() == "nan":
                retriever_name = "bm25"

            top_k = (
                best_tags.get("k")
                or best_tags.get("retrieval.k")
                or (best_cfg.retrieval.k if best_cfg else "N/A")
            )

            raw_loss = (
                best_tags.get("loss")
                or best_tags.get("learner.loss")
                or (best_cfg.learner.loss if best_cfg else "N/A")
            )
            loss_name = loss_names.get(str(raw_loss), str(raw_loss))

            export_metadata = {
                "loss": loss_name,
                "training_strat": loss_name,
                "k": top_k,
                "retriever": retriever_name,
                "tb_path": str(tb_log_path) if tb_log_path else None,
                "results": md_results.to_markdown(index=False),
            }

            if tb_log_path:
                model_obj._tb_path = tb_log_path
            model_obj._export_metadata = export_metadata

        if isinstance(torch, FakeModule):
            logger.warning(
                "\n" + "=" * 80 + "\n"
                f"[MODEL EXPORT SKIPPED - MOCKS DETECTED]\n"
                f"Model '{model_name}' was NOT exported to {best_model_path} because\n"
                "modules (torch, transformers) are mocked in this experiment manager process (pre_experiment).\n"
                "Instantiating and exporting PyTorch weights to safetensors requires real PyTorch and Transformers.\n\n"
                "All metrics, raw_results.csv, and model results.csv have been saved successfully.\n"
                + "="
                * 80
            )
        else:
            logger.info(f"Exporting model using ExportAction to {best_model_path}")
            action_cfg = best_model_loader.export_action(default_name=model_name)
            action_inst = action_cfg.instance()

            # Attach metadata to action_inst loader & model so it is accessible during write_hub_extras
            if hasattr(action_inst, "loader") and action_inst.loader is not None:
                if (
                    hasattr(action_inst.loader, "model")
                    and action_inst.loader.model is not None
                ):
                    action_inst.loader.model._export_metadata = export_metadata
                    action_inst.loader.model.model_name = model_name
                    if tb_log_path:
                        action_inst.loader.model._tb_path = tb_log_path

            action_inst.export_to_folder(best_model_path)

            # Hydrate predecessor task stubs and anonymize cluster paths in experimaestro.json
            from xp_utils import sanitize_and_hydrate_model_export

            try:
                sanitize_and_hydrate_model_export(best_model_path)
            except Exception as e:
                logger.warning(
                    f"Failed to sanitize and hydrate experimaestro.json in {best_model_path}: {e}",
                    exc_info=True,
                )
    else:
        logger.warning(f"Could not find model task for tags {best_tags} in all_weights")

    def get_runs_per_tags(model_tags: dict):
        runs = {}
        detailed = {}
        for dataset, evals in tests.collection.items():
            for eval_tags, evaluate in evals.per_tags.items():
                if all(
                    str(eval_tags.get(tag)) == str(value)
                    for tag, value in model_tags.items()
                ):
                    job_path = Path(evaluate.results).parent
                    run_path = job_path / "run.txt"
                    if run_path.exists():
                        runs[dataset] = run_path
                    detailed_path = job_path / "detailed.dat"
                    if detailed_path.exists():
                        detailed[dataset] = detailed_path
        return runs, detailed

    # 5. Export runs
    if save_runs:
        if not tests:
            logger.error("save_runs is True but no tests provided, skipping...")
        runs, detailed = get_runs_per_tags(learner_tags)
        if not runs:
            logging.warning(f"not runs retrieved for model with tags {learner_tags}")
            return
        runs_dir = best_model_path / "evals"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for dataset, dpath in detailed.items():
            shutil.copy(dpath, runs_dir / f"detailed_{dataset}.dat")
        logger.info(f"Copied {len(list(runs.keys()))} runs to {runs_dir}")


def merge_eval_sparsity_files(task_dir: Path) -> Optional[Dict[str, float]]:
    """Loads eval_sparsity.json and merges any eval_sparsity_{rank}.json (rank > 0) files if present."""
    sparsity_file = task_dir / "eval_sparsity.json"
    rank_files = sorted(task_dir.glob("eval_sparsity_*.json"))

    if not sparsity_file.exists() and not rank_files:
        return None

    # If only eval_sparsity.json exists and no separate rank files exist, return directly
    if sparsity_file.exists() and not rank_files:
        try:
            with sparsity_file.open("r") as fp:
                return json.load(fp)
        except Exception as e:
            logger.warning(f"Failed to read {sparsity_file}: {e}")
            return None

    # Gather rank 0 (eval_sparsity.json) and other ranks
    all_files = []
    if sparsity_file.exists():
        all_files.append(sparsity_file)
    for rf in rank_files:
        if rf != sparsity_file:
            all_files.append(rf)

    total_tokens = 0
    pruned_tokens = 0
    sum_utilities = 0.0

    for f in all_files:
        try:
            with f.open("r") as fp:
                data = json.load(fp)
            t = data.get("total_tokens", 0)
            p = data.get("pruned_tokens", int(round(t * data.get("sparsity", 0.0))))
            s = data.get("sum_utilities", float(t * data.get("mean_utility", 0.0)))
            total_tokens += t
            pruned_tokens += p
            sum_utilities += s
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")

    if total_tokens == 0:
        return None

    denom = max(total_tokens, 1)
    merged_stats = {
        "sparsity": pruned_tokens / denom,
        "mean_utility": sum_utilities / denom,
        "kept_ratio": (total_tokens - pruned_tokens) / denom,
        "total_tokens": total_tokens,
        "pruned_tokens": pruned_tokens,
        "sum_utilities": sum_utilities,
    }

    try:
        with sparsity_file.open("w") as fp:
            json.dump(merged_stats, fp, indent=2)
        logger.info(
            f"[merge_eval_sparsity_files] Merged {len(all_files)} sparsity files into {sparsity_file}: {merged_stats}"
        )
    except Exception as e:
        logger.warning(f"Failed to write merged {sparsity_file}: {e}")

    return merged_stats


def sync_eval_sparsity_metrics(tests: EvaluationsCollection):
    """Synchronize eval_sparsity.json into aggregated.txt so to_dataframe() parses it natively."""
    if not hasattr(tests, "collection"):
        logger.warning(
            "[sync_eval_sparsity_metrics] tests object has no 'collection' attribute"
        )
        return

    for dataset_name, evaluations in tests.collection.items():
        eval_list = getattr(evaluations, "results", [])
        if not eval_list and hasattr(evaluations, "per_tags"):
            eval_list = list(evaluations.per_tags.values())

        for evaluation in eval_list:
            results_path = getattr(evaluation, "results", None)
            if results_path is not None:
                results_path = Path(results_path)
                if results_path.exists():
                    stats = merge_eval_sparsity_files(results_path.parent)
                    if stats is not None:
                        try:
                            existing_text = results_path.read_text()
                            lines_to_append = []
                            for metric_name in ["sparsity"]:
                                if (
                                    metric_name in stats
                                    and f"{metric_name:25s}" not in existing_text
                                ):
                                    lines_to_append.append(
                                        "{:25s} {:10s} {:.4f}\n".format(
                                            metric_name,
                                            "all",
                                            float(stats[metric_name]),
                                        )
                                    )
                            if lines_to_append:
                                with results_path.open("a") as fp:
                                    fp.writelines(lines_to_append)
                                logger.debug(
                                    f"[sync_eval_sparsity_metrics] Synced eval sparsity metrics ({list(stats.keys())}) into {results_path}"
                                )
                            else:
                                logger.debug(
                                    f"[sync_eval_sparsity_metrics] Metrics already present in {results_path}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"[sync_eval_sparsity_metrics] Could not sync eval sparsity for {dataset_name}: {e}"
                            )
                    else:
                        logger.debug(
                            f"[sync_eval_sparsity_metrics] No eval sparsity data found in {results_path.parent}"
                        )


def process_experiment_results(
    tests: EvaluationsCollection,
    all_tags: list,
    config_map: dict,
    cfg: CE_FineTuning,
    helper,
    learners: list,
    all_weights: list,
    get_name_fn: Callable,
):
    """Shared post-experiment results processing and reporting."""

    # 0. Sync eval sparsity metrics into aggregated results files before building DataFrame
    sync_eval_sparsity_metrics(tests)

    # When use_plaid is True, sync on-disk index sizes from the dependency graph
    is_plaid = bool(
        getattr(cfg, "use_plaid", False)
        or any(getattr(c, "use_plaid", False) for c in config_map.values())
    )
    if is_plaid:
        from index_utils import sync_index_size_metrics

        sync_index_size_metrics(tests)

    # When get_latency is True, sync query latency metrics from task logs
    is_latency = bool(
        getattr(cfg, "get_latency", False)
        or any(getattr(c, "get_latency", False) for c in config_map.values())
    )
    if is_latency:
        from latency_utils import sync_latency_metrics

        sync_latency_metrics(tests)

    # 1. Exctract data
    df = tests.to_dataframe()

    if df.empty:
        logging.info("No results found, Ending experiment")
        return

    logging.info(f"Evaluated models: \n- {'\n- '.join(tests.per_model.keys())}")

    # Identify available metric columns and convert to numeric
    metric_cols = [col for col in df.columns if col[0] == "metric"]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")

    # Flatten MultiIndex columns and remove duplicates
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if col[1] else col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop useless columns
    cols_to_drop = [col for col in df.columns if "index_doc" in str(col).lower()]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Identify grid search keys from all_tags
    grid_keys = set()
    for tags in all_tags:
        grid_keys.update(tags.keys())

    # Core model identity tags (grid keys + evaluation checkpoint tags if present)
    eval_identity_tags = {"first_stage", "checkpoint", "plaid_retriever"}
    model_id_tags = sorted(
        list((grid_keys | eval_identity_tags) & set(df.columns))
        + (["seed"] if "seed" in df.columns else [])
    )

    # Tags to group by when aggregating across seeds
    group_by_tags = [tag for tag in model_id_tags if tag != "seed"]

    # Add both specific aggregations (ID, BEIR, etc.) and the global mean
    df_with_aggs = add_dataset_aggregations(
        df,
        group_by_cols=model_id_tags,
        aggregations=aggregations,
        add_mean=True,  # will add 'mean' dataset at the end
    )

    # keep only results with a base tag (all our models have it)
    # We ensure it's not NaN and not an empty string to exclude first-stage only results
    scorer_only_df = df_with_aggs.copy()[
        df_with_aggs["base"].notna()
        & (df_with_aggs["base"].astype(str) != "")
        & (df_with_aggs["base"].astype(str) != "nan")
    ]

    # Determine target dataset for selecting best model
    val_dataset = getattr(cfg.learner, "validation", None)
    best_model_dataset = getattr(
        getattr(cfg, "evaluation", None), "best_model_dataset", None
    )

    present_datasets = [str(x) for x in scorer_only_df["dataset"].values]
    val_dataset_str = str(val_dataset) if val_dataset is not None else None
    best_model_dataset_str = (
        str(best_model_dataset) if best_model_dataset is not None else None
    )

    if best_model_dataset_str and best_model_dataset_str in present_datasets:
        target_dataset = best_model_dataset_str
    elif val_dataset_str and val_dataset_str in present_datasets:
        target_dataset = val_dataset_str
    elif "BEIR13" in present_datasets:
        target_dataset = "BEIR13"
    else:
        target_dataset = "mean"

    logging.info(f"Selecting best model based on dataset: {target_dataset}")

    # Group by hyperparameter tags excluding seed and checkpoint so that
    # a single best (seed, checkpoint) combination is selected per model configuration
    best_model_group_tags = [tag for tag in group_by_tags if tag != "checkpoint"]
    best_models_df = identify_best_models(
        scorer_only_df,
        dataset=target_dataset,
        metric="nDCG@10",
        group_by_tags=best_model_group_tags,
    )
    logging.info(f"df with only best models:\n{best_models_df}")

    if helper.xp.run_mode != RunMode.NORMAL:
        logger.info("Dry-run mode: skipping disk writes for best models and results")
        return
    # Save raw results to disk
    save_raw_results(df_with_aggs, helper.xp.resultspath)

    best_models_list = []
    if not best_models_df.empty:
        # 1. Collect all best candidates per configuration for the raw results summary
        for _, best_row in best_models_df.iterrows():
            mask = pd.Series(True, index=df_with_aggs.index)
            for tag in model_id_tags:
                mask &= df_with_aggs[tag].astype(str) == str(best_row[tag])
            best_model_df = df_with_aggs[mask].copy()
            best_models_list.append(best_model_df)

        if best_models_list:
            all_best_models_df = pd.concat(best_models_list, ignore_index=True)
            all_best_models_df.to_csv(
                helper.xp.resultspath / "best_models_per_scorer_raw_results.csv",
                index=False,
            )

        # 2. Export model artifacts if enabled
        if cfg.export_trained_models:
            models_path = helper.xp.resultspath / "models"
            if models_path.exists():
                shutil.rmtree(models_path)
                logging.info(f"Deleted existing models directory: {models_path}")

            export_only_best = getattr(cfg, "export_only_best", True)
            if export_only_best and len(best_models_df) > 1:
                metric_col = "nDCG@10"
                if metric_col in best_models_df.columns:
                    best_idx = best_models_df[metric_col].astype(float).idxmax()
                    export_models_df = best_models_df.loc[[best_idx]]
                    winner_score = export_models_df[metric_col].values[0]
                    logging.info(
                        f"export_only_best=True: selected single best model among {len(best_models_df)} candidates "
                        f"({metric_col}={winner_score:.4f} on {target_dataset})"
                    )
                else:
                    logging.warning(
                        f"Metric {metric_col} not found in best_models_df; exporting all {len(best_models_df)} candidate models"
                    )
                    export_models_df = best_models_df
            else:
                export_models_df = best_models_df

            # Check for potential naming collisions among models to be exported
            candidate_names = [
                get_name_fn({tag: r[tag] for tag in model_id_tags})
                for _, r in export_models_df.iterrows()
            ]
            has_name_collision = len(candidate_names) != len(set(candidate_names))
            if has_name_collision:
                logging.warning(
                    "Naming collision detected among exported models! Disambiguating names using grid search tags."
                )

            grid_keys = set()
            for tags in all_tags:
                grid_keys.update(tags.keys())

            for _, best_row in export_models_df.iterrows():
                best_tags = {tag: best_row[tag] for tag in model_id_tags}
                logging.info(f"Exporting model with tags: {best_tags}")

                best_grid_tags = {k: best_tags[k] for k in grid_keys if k in best_tags}
                best_cfg = config_map.get(frozenset(best_grid_tags.items()))

                mask = pd.Series(True, index=df_with_aggs.index)
                for tag in model_id_tags:
                    mask &= df_with_aggs[tag].astype(str) == str(best_row[tag])
                best_model_df = df_with_aggs[mask].copy()

                csv_results, md_results = format_model_results(
                    best_model_df, aggregations=aggregation_hf
                )

                model_name = get_name_fn(best_tags)
                if has_name_collision and best_grid_tags:
                    tag_suffix = "_".join(
                        f"{k}-{v}" for k, v in sorted(best_grid_tags.items())
                    )
                    model_name = f"{model_name}_{tag_suffix}"

                export_model(
                    best_tags=best_tags,
                    model_name=model_name,
                    csv_results=csv_results,
                    md_results=md_results,
                    learners=learners,
                    all_weights=all_weights,
                    best_cfg=best_cfg,
                    resultspath=helper.xp.resultspath,
                    save_runs=cfg.save_runs,
                    tests=tests,
                )

    # Final aggregation and LaTeX table generation
    metric_names = [col[1] for col in metric_cols]
    group_by_tags = ["dataset"] + group_by_tags
    counts = df_with_aggs.groupby(group_by_tags, dropna=False).size().to_frame("n_runs")
    df_grouped_raw = df_with_aggs.groupby(group_by_tags, dropna=False)[
        metric_names
    ].agg(["mean", "var"])
    df_grouped_raw[("n_runs", "")] = counts["n_runs"]
    df_grouped = df_grouped_raw.reset_index()

    # Reorder columns: tags first, metrics after
    tag_cols_expected = group_by_tags + ["n_runs"]
    tag_cols = []
    for col in tag_cols_expected:
        if (col, "") in df_grouped.columns:
            tag_cols.append((col, ""))
        elif col in df_grouped.columns:
            tag_cols.append(col)

    metric_cols_grouped = [c for c in df_grouped.columns if c not in tag_cols]
    df_grouped = df_grouped[tag_cols + metric_cols_grouped]

    logging.info(f"Aggregated Results:\n{df_grouped}")
    df_grouped.to_csv(helper.xp.resultspath / "results.csv", index=False)

    latex_table = dataframe_to_latex(
        df_grouped,
        caption="Evaluation Results",
        label="tab:eval_results",
        sig_df=None,
    )
    with open(helper.xp.resultspath / "results.tex", "w") as f:
        f.write(latex_table)

    logging.info(f"Saved aggregated results to {helper.xp.resultspath / 'results.csv'}")
    logging.info("Experiment completed successfully.")


def find_checkpoint(
    outputs: LearnerOutput,
    target_epoch: int,
    learner: Optional[Any] = None,
) -> Any:
    """Finds a checkpoint task for a specific epoch.

    Search order:
    1. Direct match in outputs.checkpoints (by int or str key).
    2. Check if target_epoch equals learner.max_epochs (using outputs.learned_model).
    3. Check if the checkpoint exists on disk in learner.checkpointspath.
    4. If not found, raises a descriptive ValueError listing available scheduled checkpoints and interval info.

    Args:
        outputs: The LearnerOutput from learner.submit().
        target_epoch: Target epoch number to evaluate.
        learner: Optional Learner instance for disk and max_epochs fallback.

    Returns:
        The ModuleLoader task for the requested epoch.
    """
    # 1. Check in outputs.checkpoints
    if outputs.checkpoints:
        if target_epoch in outputs.checkpoints:
            logger.info(
                f"Found scheduled checkpoint for epoch {target_epoch} in learner outputs."
            )
            return outputs.checkpoints[target_epoch]
        for k, v in outputs.checkpoints.items():
            if str(k) == str(target_epoch):
                logger.info(
                    f"Found scheduled checkpoint for epoch {target_epoch} in learner outputs."
                )
                return v

    # 2. Check if target_epoch corresponds to max_epochs (last)
    if learner is not None and getattr(learner, "max_epochs", None) == target_epoch:
        logger.info(
            f"Requested epoch {target_epoch} matches learner.max_epochs. Using last checkpoint (learned_model)."
        )
        return outputs.learned_model

    # 3. Disk fallback
    if learner is not None and hasattr(learner, "checkpointspath"):
        from xpm_torch.trainers.context import TrainerContext, TrainState
        from xpm_torch.learner import CheckpointSettings

        chk_path = (
            TrainerContext.get_checkpoint_path(
                Path(learner.checkpointspath), target_epoch
            )
            / TrainState.MODEL_DIR
        )
        if chk_path.exists():
            logger.info(
                f"Found checkpoint for epoch {target_epoch} on disk at {chk_path}"
            )
            return learner.model.loader_config(
                chk_path, settings=CheckpointSettings.C(epoch=target_epoch)
            )

    # 4. Not found -> raise informative error
    available = sorted(outputs.checkpoints.keys()) if outputs.checkpoints else []
    interval_info = (
        f" Learner checkpoint_interval is {learner.checkpoint_interval}, max_epochs is {learner.max_epochs}."
        if learner is not None and hasattr(learner, "checkpoint_interval")
        else ""
    )
    raise ValueError(
        f"Checkpoint for epoch {target_epoch} could not be found! "
        f"Available scheduled checkpoints in learner outputs: {available}.{interval_info} "
        f"Please verify that target_epoch was scheduled by checkpoint_interval or exists on disk."
    )


def find_candidate_experiment_dirs(exp_id: str) -> List[Path]:
    """Find candidate directories for an experiment ID across Experimaestro workspaces and local roots."""
    candidate_exp_dirs = []
    try:
        settings = get_settings()
        # Check workspaces in settings
        for ws in settings.workspaces:
            if ws.path:
                p_exp = Path(ws.path) / "experiments" / exp_id
                if p_exp.exists():
                    candidate_exp_dirs.append(p_exp)
                p_direct = Path(ws.path) / exp_id
                if p_direct.exists():
                    candidate_exp_dirs.append(p_direct)
    except Exception as e:
        logger.debug(f"Could not inspect settings.workspaces: {e}")

    # Check default experiment roots (~/code/experiments/*)
    default_experiments_root = Path.home() / "code" / "experiments"
    if default_experiments_root.exists():
        for sub in default_experiments_root.iterdir():
            if sub.is_dir():
                sub_exp = sub / exp_id
                if sub_exp.exists() and sub_exp not in candidate_exp_dirs:
                    candidate_exp_dirs.append(sub_exp)
                sub_experiments_exp = sub / "experiments" / exp_id
                if (
                    sub_experiments_exp.exists()
                    and sub_experiments_exp not in candidate_exp_dirs
                ):
                    candidate_exp_dirs.append(sub_experiments_exp)

    return candidate_exp_dirs


def parse_experiment_ref(ref: str) -> Tuple[str, Optional[str]]:
    """Parse an experiment reference into (exp_id, run_id).

    If ref contains a slash:
    - Check for direct absolute path like /path/to/exp_dir/run_id where parent directory exists.
    - Check if it matches 'exp_id/run_id' where exp_id is a known local experiment in workspaces.
    Otherwise returns (ref, None).
    """
    if "/" not in ref:
        return ref, None

    # Check for direct absolute path like /path/to/exp_dir/run_id
    p = Path(ref).expanduser()
    if p.is_absolute() and p.parent.is_dir():
        return str(p.parent), p.name

    # Check for exp_id/run_id where exp_id is a known local experiment
    parts = ref.strip("/").split("/")
    if len(parts) == 2:
        candidate_exp, candidate_run = parts[0], parts[1]
        if find_candidate_experiment_dirs(candidate_exp):
            return candidate_exp, candidate_run

    return ref, None


def _find_exported_models_in_run(run_dir: Path) -> List[Path]:
    """Find directories containing experimaestro.json inside run_dir/results/models."""
    models_dir = (
        run_dir / "results" / "models"
        if (run_dir / "results").exists()
        else (run_dir / "models" if run_dir.name == "results" else run_dir)
    )
    if not models_dir.exists() or not models_dir.is_dir():
        return []
    return [
        d
        for d in models_dir.iterdir()
        if d.is_dir() and (d / "experimaestro.json").exists()
    ]


def find_experiment_results_model_dir(experiment_ref: str) -> Path:
    """Finds the exported model directory inside an experiment's results/models path.

    Args:
        experiment_ref: Experiment ID (e.g. 'train_mice_ettin32m_l4+4'),
                       optional run-specific ID ('exp_id/run_id'),
                       or direct directory path.

    Returns:
        Path to the exported model directory containing experimaestro.json.
    """
    ref_path = Path(experiment_ref).expanduser()

    # 1. Direct path to a model folder (contains experimaestro.json)
    if ref_path.is_dir() and (ref_path / "experimaestro.json").exists():
        logger.info(f"Using direct model path: {ref_path}")
        return ref_path

    # 2. Direct path to a run directory containing results/models
    if ref_path.is_dir():
        models = _find_exported_models_in_run(ref_path)
        if models:
            logger.info(f"Found exported model in {models[0]}")
            return models[0]

    # 3. Experiment ID / run lookup across workspaces and local experiment directories
    exp_id, run_id = parse_experiment_ref(experiment_ref)
    candidate_exp_dirs = find_candidate_experiment_dirs(exp_id)

    if not candidate_exp_dirs:
        if Path(exp_id).is_dir():
            exp_base = Path(exp_id)
        elif ref_path.is_dir():
            exp_base = ref_path
        else:
            settings = get_settings()
            searched_ws = [str(ws.path) for ws in settings.workspaces if ws.path]
            raise FileNotFoundError(
                f"Experiment '{exp_id}' not found in any workspace or local experiment paths. "
                f"Searched workspaces: {searched_ws}"
            )
    else:
        exp_base = candidate_exp_dirs[0]

    # If run_id is explicitly specified, load directly from it
    if run_id:
        run_dir = exp_base / run_id
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Run ID '{run_id}' not found under experiment base directory {exp_base}"
            )
        models = _find_exported_models_in_run(run_dir)
        if not models:
            raise FileNotFoundError(
                f"No valid exported models with experimaestro.json found in run '{run_id}' at {run_dir}"
            )
        selected_model_dir = models[0]
        logger.info(
            f"Resolved experiment '{experiment_ref}' (run '{run_id}') to model at {selected_model_dir}"
        )
        return selected_model_dir

    # When run_id is omitted:
    # First, check 'current'
    current_path = exp_base / "current"
    if current_path.exists():
        current_run = current_path.resolve()
        models = _find_exported_models_in_run(current_run)
        if models:
            selected_model_dir = models[0]
            logger.info(
                f"Resolved experiment '{experiment_ref}' (current run '{current_run.name}') to model at {selected_model_dir}"
            )
            return selected_model_dir
        else:
            logger.warning(
                f"'current' run in {exp_base} ('{current_run.name}') has no exported models yet "
                "(run may still be in progress). Searching for previous completed runs..."
            )

    # Search completed runs among siblings, sorted by mtime descending
    run_dirs = sorted(
        [
            d
            for d in exp_base.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name not in ("current", "dry-run")
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for candidate_run in run_dirs:
        models = _find_exported_models_in_run(candidate_run)
        if models:
            selected_model_dir = models[0]
            logger.info(
                f"Resolved experiment '{experiment_ref}' to latest completed run '{candidate_run.name}' at {selected_model_dir}"
            )
            return selected_model_dir

    raise FileNotFoundError(
        f"Experiment '{exp_id}' found at {exp_base}, but no completed runs with exported models "
        "(containing experimaestro.json) were found."
    )


def resolve_trained_model_loader(source: str) -> ModuleLoader:
    """Unified resolution of an experiment ID, local path, or Hugging Face Hub repo into a ModuleLoader config.

    Args:
        source: Experiment ID ('train_mice_ettin32m_l4+4'),
                pinned run ID ('train_micev2_ettin68m_l10+4_v3/20260924_180227'),
                local folder path, or HF Hub repo ('VictorMorand/MiceV2-l10p4-ettin-68m').

    Returns:
        ModuleLoader Config whose `.model` attribute is the Model Config.
    """
    path_candidate = Path(source).expanduser()

    # If it's a direct path to an existing directory
    if path_candidate.is_dir():
        target = find_experiment_results_model_dir(source)
    else:
        exp_id, run_id = parse_experiment_ref(source)
        candidates = find_candidate_experiment_dirs(exp_id)
        if candidates or Path(exp_id).is_dir():
            # Experiment is local. Do not swallow FileNotFoundError to avoid false HF Hub fallback.
            target = find_experiment_results_model_dir(source)
        else:
            # Fall back to Hugging Face Hub repo ID
            logger.info(
                f"'{source}' not found locally as experiment; treating as Hugging Face Hub repo ID."
            )
            target = source

    logger.info(f"Loading pretrained ModuleLoader from {target}")
    return TorchHFHub.pretrained_loader(target, as_instance=False)


def validate_and_align_config(
    xp_config: Any,
    xpm_model_config: Any,
    tags: Optional[dict] = None,
    from_ref: Optional[str] = None,
) -> None:
    """Validate user overrides and align default parameters between an experiment config (xp config)
    and a loaded model Experimaestro Config (xpm Config).

    Args:
        xp_config: Dataclass experiment config (xp config, e.g. MiceV2_FineTuning).
        xpm_model_config: Experimaestro Config of the loaded model (xpm Config).
        tags: Optional dict of tracking tags to synchronize with the resolved model.
        from_ref: Optional string identifier of the checkpoint source for logging.
    """
    source_label = from_ref or getattr(xp_config, "from_experiment", "loaded model")
    default_cfg = type(xp_config)()

    # 1. Base model alignment / override detection
    src_base = getattr(xpm_model_config, "hf_id", None) or getattr(
        xpm_model_config, "base", None
    )
    if src_base and not isinstance(src_base, field):
        cfg_base = getattr(xp_config, "base", None)
        default_base = getattr(default_cfg, "base", None)
        if not cfg_base or cfg_base == default_base or cfg_base == src_base:
            if cfg_base != src_base:
                logger.info(
                    "[Model Alignment] Inherited base '%s' from loaded model '%s' (was default %r)",
                    src_base,
                    source_label,
                    cfg_base,
                )
            xp_config.base = src_base
        else:
            logger.warning(
                "[PARAM OVERRIDE] Config explicitly overrides base model! "
                "Loaded model (%s) has %r, but config specifies %r.",
                source_label,
                src_base,
                cfg_base,
            )

    # 2. Architectural & operational attributes
    aligned_attrs = [
        "n_contextualization_layers",
        "n_docs_ctx_layers",
        "n_interaction_layers",
        "bound_bottom_layers",
        "use_upper_rope",
        "top_layers_doc_ln",
        "bottom_final_norm",
        "top_final_norm",
        "compress_dim",
        "compress_init_option",
        "doc_bottleneck_dim",
        "use_cross_attn",
        "utility_pred",
        "gate_target",
        "head_type",
        "hard_pruning",
        "hard_prune_threshold",
        "attn_implementation",
    ]

    for attr in aligned_attrs:
        if not hasattr(xp_config, attr):
            continue

        src_val = getattr(xpm_model_config, attr, None)
        if src_val is None or isinstance(src_val, field):
            continue

        cfg_val = getattr(xp_config, attr)
        default_val = getattr(default_cfg, attr)

        # Special case: phase2_training explicitly configures hard_prune_threshold
        if attr == "hard_prune_threshold" and getattr(
            xp_config, "phase2_training", False
        ):
            if cfg_val != src_val:
                logger.warning(
                    "[PARAM OVERRIDE] Phase 2 hard_prune_threshold (%r) overrides loaded model value (%r) from '%s'",
                    cfg_val,
                    src_val,
                    source_label,
                )
            continue

        if cfg_val == default_val:
            # Parameter was left at default -> inherit from loaded model
            if cfg_val != src_val:
                logger.info(
                    "[Model Alignment] Inherited '%s' from loaded model '%s': %r (was default %r)",
                    attr,
                    source_label,
                    src_val,
                    default_val,
                )
                setattr(xp_config, attr, src_val)
        else:
            # Parameter was explicitly set by user in config
            if cfg_val != src_val:
                logger.warning(
                    "[PARAM OVERRIDE] Config explicitly overrides loaded model parameter '%s'! "
                    "Loaded model (%s) value: %r -> Config override value: %r.",
                    attr,
                    source_label,
                    src_val,
                    cfg_val,
                )

    # 3. Context/tokenization lengths and prompt attributes
    tok = getattr(xpm_model_config, "tokenizer", None)
    length_and_prompt_attrs = [
        "max_query_length",
        "max_doc_length",
        "max_length",
        "query_prompt",
        "doc_prompt",
    ]
    for attr in length_and_prompt_attrs:
        if not hasattr(xp_config, attr):
            continue

        cfg_val = getattr(xp_config, attr)
        default_val = getattr(default_cfg, attr)

        src_val = getattr(xpm_model_config, attr, None)
        if (src_val is None or isinstance(src_val, field)) and tok is not None:
            src_val = getattr(tok, attr, None)

        if src_val is None or isinstance(src_val, field):
            continue

        if cfg_val == default_val:
            if cfg_val != src_val:
                logger.info(
                    "[Model Alignment] Inherited '%s' from loaded model '%s': %r (was default %r)",
                    attr,
                    source_label,
                    src_val,
                    default_val,
                )
                setattr(xp_config, attr, src_val)
        else:
            if cfg_val != src_val:
                logger.warning(
                    "[PARAM OVERRIDE] Config explicitly overrides loaded model parameter '%s'! "
                    "Loaded model (%s) value: %r -> Config override value: %r.",
                    attr,
                    source_label,
                    src_val,
                    cfg_val,
                )

    # 4. Synchronize tags
    if tags is not None:
        if "base" not in tags or tags["base"] != getattr(xp_config, "base", None):
            tags["base"] = getattr(xp_config, "base", None)
        ref_to_parse = tags.get("from_experiment", source_label)
        if ref_to_parse:
            parsed_exp_id, run_id = parse_experiment_ref(str(ref_to_parse))
            tags["from_experiment"] = parsed_exp_id
            if run_id:
                tags["from_run_id"] = run_id
            if hasattr(xp_config, "from_run_id") and run_id:
                xp_config.from_run_id = run_id
