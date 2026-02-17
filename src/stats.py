from pathlib import Path
import pandas as pd
import logging
from xpm_torch.utils.logging import easylog
logger = easylog()

def compute_t_test_threshold(mean, var, n, alpha=0.05):
    """Compute the t-test threshold for a one-sided t-test given mean, variance, sample size, and significance level alpha."""
    from scipy import stats

    if n <= 1:
        raise ValueError("Sample size n must be greater than 1 to compute t-test threshold.")

    df = n - 1  # degrees of freedom
    t_critical = stats.t.ppf(1 - alpha, df)
    standard_error = (var / n) ** 0.5
    threshold = mean + t_critical * standard_error
    return threshold

def load_measures_from_file(path: Path) -> pd.Series:
    """Load all measures from a detailed file as a Series indexed by (measure, query)."""
    df = pd.read_csv(path, sep='\s+', header=None, names=["measure", "query", "value"])
    df["measure"] = df["measure"].astype(str).str.strip()
    df["query"] = df["query"].astype(str).str.strip()
    df["value"] = df["value"].astype(float)
    idx = pd.MultiIndex.from_arrays([df["measure"], df["query"]], names=["measure", "query"])
    return pd.Series(df["value"].values, index=idx)

def combine_measures(tests_per_model: dict, nb_repetitions: int = 1) -> pd.DataFrame:
    """
    Combine all measures across models and datasets.

    Columns are a MultiIndex (dataset_key, group_key).
    Grouping: if 'baseline' appears anywhere in model_id -> group_key "baseline",
              otherwise remove last "-<suffix>" from model_id.
    Aggregates (sums) per-group then divides by nb_repetitions to get averages.

    Now handles that each list item in tests_per_model[model_id] may refer to a
    different dataset (entry.key).
    """
    cols = {}  # mapping (dataset_key, group_key) -> pd.Series

    for model_id, lst in tests_per_model.items():
        if not lst:
            continue

        # determine group_key once per model_id
        if "baseline" in model_id:
            group_key = "baseline"
        else:
            group_key = model_id.rsplit("-", 1)[0] if "-" in model_id else model_id

        # iterate all entries for this model_id (may be different datasets)
        for entry in lst:
            res = getattr(entry, "result", entry)
            detailed = getattr(res, "detailed", None)
            dataset_key = getattr(entry, "key", "unknown")
            if detailed is None:
                continue
            path = Path(detailed)
            if not path.exists():
                continue

            s = load_measures_from_file(path)

            # set column name as tuple (dataset_key, group_key)
            s.name = (dataset_key, group_key)
            key = (dataset_key, group_key)
            if key in cols:
                cols[key] = cols[key].add(s, fill_value=0.0)
            else:
                cols[key] = s

    if not cols:
        return pd.DataFrame()

    # Build DataFrame: index = (measure, query) MultiIndex; columns = MultiIndex (dataset, group)
    df = pd.DataFrame(cols)

    # average across repetitions
    if nb_repetitions and nb_repetitions > 0:
        df = df.astype(float) / float(nb_repetitions)

    # add per-dataset summary columns (baselines_mean, others_mean)
    datasets = sorted({col[0] for col in df.columns})
    for ds in datasets:
        ds_cols = [col for col in df.columns if col[0] == ds]
        baseline_cols = [col for col in ds_cols if col[1] == "baseline"]
        other_cols = [col for col in ds_cols if col[1] not in ("baseline", "baselines_mean", "others_mean")]
        if baseline_cols:
            df[(ds, "baselines_mean")] = df.loc[:, baseline_cols].mean(axis=1)
        if other_cols:
            df[(ds, "others_mean")] = df.loc[:, other_cols].mean(axis=1)

    # Ensure columns are a proper MultiIndex and ordered
    df.columns = pd.MultiIndex.from_tuples([tuple(c) for c in df.columns])
    return df

def run_statistical_tests(tests_per_model: dict, nb_repetitions: int = 1) -> pd.DataFrame:
    """
    Run t-tests between baseline and each other group, per dataset and per measure (across queries).
    Returns a DataFrame indexed by (dataset, measure, model) with columns [t_stat, p_value, n_baseline, n_other].
    """
    from scipy import stats

    df = combine_measures(tests_per_model, nb_repetitions=nb_repetitions)
    if df.empty:
        logger.warning("No data available to run statistical tests.")
        return pd.DataFrame()

    results = []
    datasets = sorted({col[0] for col in df.columns})
    measures = df.index.get_level_values("measure").unique()

    logger.info(f"Running statistical tests on datasets: {datasets} and measures: {measures.tolist()}")

    for dataset_key in datasets:
        # subset columns for this dataset (second level are group keys)
        if dataset_key not in df.columns.get_level_values(0):
            continue
        subset = df.xs(dataset_key, axis=1, level=0)
        if subset.empty:
            continue

        # identify baseline and other group columns (exclude summary *_mean)
        baseline_cols = [c for c in subset.columns if str(c) == "baseline"]
        other_cols = [c for c in subset.columns if c not in baseline_cols and not str(c).endswith("_mean")]
        if not baseline_cols:
            # no baseline for this dataset, skip
            logger.warning(f"No baseline group found for dataset {dataset_key}, skipping statistical tests for this dataset.")
            continue
        baseline_col = baseline_cols[0]

        for measure in measures:
            try:
                measure_subset = subset.xs(measure, level="measure")
            except KeyError:
                # measure not present for this dataset
                logger.warning(f"Measure {measure} not found for dataset {dataset_key}, skipping.")
                continue

            if measure_subset.empty:
                continue

            if baseline_col not in measure_subset.columns:
                continue
            baseline_series = measure_subset[baseline_col]

            for col in other_cols:
                if col not in measure_subset.columns:
                    continue
                other_series = measure_subset[col]

                # align queries and drop NaNs
                common_idx = baseline_series.dropna().index.intersection(other_series.dropna().index)
                if len(common_idx) < 2:
                    results.append(
                        {
                            "dataset": dataset_key,
                            "measure": measure,
                            "model": col,
                            "t_stat": float("nan"),
                            "p_value": float("nan"),
                            "n_baseline": int(len(baseline_series.dropna())),
                            "n_other": int(len(other_series.dropna())),
                        }
                    )
                    continue

                b = baseline_series.loc[common_idx].astype(float)
                o = other_series.loc[common_idx].astype(float)

                t_stat, p_value = stats.ttest_ind(b, o, equal_var=False, nan_policy="omit")
                results.append(
                    {
                        "dataset": dataset_key,
                        "measure": measure,
                        "model": col,
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                        "n_baseline": int(len(b)),
                        "n_other": int(len(o)),
                    }
                )

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return pd.DataFrame()
    res_df = res_df.set_index(["dataset", "measure", "model"]).sort_index()

    return res_df