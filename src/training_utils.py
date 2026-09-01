"""Utility functions for training and result processing."""

import logging
import shutil
from typing import Callable
from pathlib import Path
import numpy as np
import pandas as pd
from functools import lru_cache
from experimaestro import RunMode
from experimaestro.launcherfinder import find_launcher
from experimaestro.annotations import tags as get_tags

from xpm_torch.trainers import LossTrainer

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
from pointwise_distillation import (
    PointwiseDistillationSampler,
    PointwiseDistillationTrainer,
    PointwiseMSELoss,
    FlattenAndShuffleDataset,
)

from samplers import prepare_collection
from configuration import Losses, CE_FineTuning
from samplers import msmarco_rankdistillm_sampled_colbert50
from format import (
    aggregations,
    aggregation_hf,
    dataframe_to_latex,
    loss_names,
)

logger = logging.getLogger(__name__)


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

    elif loss_member is Losses.MSE_mixedbread_large:
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

        return PointwiseDistillationTrainer.C(
            sampler=PointwiseDistillationSampler.C(dataset=shuffled_dataset_config),
            lossfn=PointwiseMSELoss.C(),
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
    if not best_models_df.empty and cfg.export_trained_models:
        # Check if model folder exists before saving models, delete if so
        models_path = helper.xp.resultspath / "models"
        if models_path.exists():
            shutil.rmtree(models_path)
            logging.info(f"Deleted existing models directory: {models_path}")

        for _, best_row in best_models_df.iterrows():
            # Extract tags for this best model
            best_tags = {tag: best_row[tag] for tag in model_id_tags}

            logging.info(f"Best evaluated model is {best_tags}")

            # Reconstruct the grid tags to find the original config
            grid_keys = set()
            for tags in all_tags:
                grid_keys.update(tags.keys())
            best_grid_tags = {k: best_tags[k] for k in grid_keys if k in best_tags}
            best_cfg = config_map.get(frozenset(best_grid_tags.items()))

            # Filter the original dataframe for this specific best model (all datasets)
            mask = pd.Series(True, index=df_with_aggs.index)
            for tag in model_id_tags:
                mask &= df_with_aggs[tag].astype(str) == str(best_row[tag])

            best_model_df = df_with_aggs[mask].copy()
            best_models_list.append(best_model_df)

            # Format and Export artifacts
            csv_results, md_results = format_model_results(
                best_model_df, aggregations=aggregation_hf
            )

            export_model(
                best_tags=best_tags,
                model_name=get_name_fn(best_tags),
                csv_results=csv_results,
                md_results=md_results,
                learners=learners,
                all_weights=all_weights,
                best_cfg=best_cfg,
                resultspath=helper.xp.resultspath,
                save_runs=cfg.save_runs,
                tests=tests,
            )

        if best_models_list:
            best_model_df = pd.concat(best_models_list, ignore_index=True)
            best_model_df.to_csv(
                helper.xp.resultspath / "best_models_per_scorer_raw_results.csv",
                index=False,
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
