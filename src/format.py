"""Formatting utilities for experiment results"""

import pandas as pd

from tests import NANO_BEIR_KEYS

### Some dicts for formatting

loss_names = {
    "bce": "BCE",
    "hingeLoss": "Hinge",
    "infoNCE_Colbertv2Neg": "InfoNCE",
    "marginMSE": "MarginMSE",
    "distillRankNET": "DistillRankNET",
    "ADR_MSE": "ADR-MSE",
    "MSE_mixedbread_large": "MSE (mixedbread-large)",
}

backbone_names = {
    "jhu-clsp/ettin-encoder-17m": "Ettin-17M",
    "jhu-clsp/ettin-encoder-32m": "Ettin-32M",
    "jhu-clsp/ettin-encoder-68m": "Ettin-68M",
    "jhu-clsp/ettin-encoder-150m": "Ettin-150M",
    "microsoft/MiniLM-L12-H384-uncased": "MiniLM-L12 (33M)",
    "google/electra-base-discriminator": "ELECTRA (110M)",
    "bert-base-uncased": "BERT-Base (110M)",
    "FacebookAI/roberta-base": "RoBERTa (125M)",
    "microsoft/deberta-v3-base": "DeBERTav3 (184M)",
}

# condensed version for mice figure
backbone_names_mice = {
    "microsoft/MiniLM-L12-H384-uncased": "MiniLM",
    "bert-base-uncased": "BERT",
    "google/electra-base-discriminator": "ELECTRA",
    "microsoft/deberta-v3-base": "DeBERTav3",
    "FacebookAI/roberta-base": "RoBERTa (125M)",
    # Ettin
    "jhu-clsp/ettin-encoder-17m": "Ettin17",
    "jhu-clsp/ettin-encoder-32m": "Ettin32",
    "jhu-clsp/ettin-encoder-68m": "Ettin68",
    "jhu-clsp/ettin-encoder-150m": "Ettin150",
}

backbone_names_lower = {
    "jhu-clsp/ettin-encoder-17m": "ettin-17m",
    "jhu-clsp/ettin-encoder-32m": "ettin-32m",
    "microsoft/MiniLM-L12-H384-uncased": "MiniLM-L12",
    "jhu-clsp/ettin-encoder-68m": "ettin-68m",
    "bert-base-uncased": "bert-base",
    "google/electra-base-discriminator": "ELECTRA",
    "FacebookAI/roberta-base": "RoBERTa",
    "jhu-clsp/ettin-encoder-150m": "ettin-150m",
    "microsoft/deberta-v3-base": "DeBERTav3",
}

backbone_nlayers = {
    "jhu-clsp/ettin-encoder-17m": 7,
    "jhu-clsp/ettin-encoder-32m": 10,
    "microsoft/MiniLM-L12-H384-uncased": 12,
    "jhu-clsp/ettin-encoder-68m": 19,
    "bert-base-uncased": 12,
    "google/electra-base-discriminator": 12,
    "FacebookAI/roberta-base": 12,
    "jhu-clsp/ettin-encoder-150m": 22,
    "microsoft/deberta-v3-base": 12,
}

aggregations = {
    "In Domain": ["msmarco_dev", "trec2019", "trec2020"],
    "minified": [
        "msmarco_dev",
        "trec2019",
        "trec2020",
        "scifact",
        "touche",
        "fiqa",
        "nfcorpus",
    ],
    "BEIR13": [
        "arguana",
        "climate_fever",
        "dbpedia",
        "fever",
        "fiqa",
        "hotpotqa",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "touche",
        "trec_covid",
    ],
    "BEIR13 Decontaminated": [
        "arguana_decontaminated",
        "climate_fever_decontaminated",
        "dbpedia_decontaminated",
        "fever_decontaminated",
        "fiqa_decontaminated",
        "hotpotqa_decontaminated",
        "nfcorpus_decontaminated",
        "nq_decontaminated",
        "quora_decontaminated",
        "scidocs_decontaminated",
        "scifact_decontaminated",
        "touche_decontaminated",
        "trec_covid_decontaminated",
    ],
    "Lotte-S": [
        "lotte_lifestyle",
        "lotte_recreation",
        "lotte_science",
        "lotte_technology",
        "lotte_writing",
    ],
    "Nano BEIR": list(NANO_BEIR_KEYS.keys()),
}

## Aggregations used in the HF Card (keep short)
aggregation_hf = {
    "Mean In Domain": aggregations["In Domain"],
    "BEIR 13": aggregations["BEIR13"],
    "BEIR 13 (Decontaminated)": aggregations["BEIR13 Decontaminated"],
    "LoTTE (OOD)": aggregations["Lotte-S"],
    "Nano BEIR": aggregations["Nano BEIR"],
}

DATASET_TO_ABB = {
    "fiqa": "Fi",
    "msmarco_dev": "MSM",
    "trec2019": "DL19",
    "trec2020": "DL20",
    "nfcorpus": "NFC",
    "scifact": "SF",
    "arguana": "Ar",
    "touche": "T-v2",
    "climate_fever": "CF",
    "dbpedia": "DB",
    "fever": "FE",
    "hotpotqa": "HPQ",
    "nq": "NQ",
    "quora": "Q",
    "scidocs": "SD",
    "trec_covid": "T-C",
    "robust04": "R04",
    "lotte_lifestyle": "Life.",
    "lotte_recreation": "Rec.",
    "lotte_science": "Sci.",
    "lotte_technology": "Tech.",
    "lotte_writing": "Writ.",
    "fiqa_decontaminated": "Fi*",
    "nfcorpus_decontaminated": "NFC*",
    "scifact_decontaminated": "SF*",
    "arguana_decontaminated": "Ar*",
    "touche_decontaminated": "T-v2*",
    "climate_fever_decontaminated": "CF*",
    "dbpedia_decontaminated": "DB*",
    "fever_decontaminated": "FE*",
    "hotpotqa_decontaminated": "HPQ*",
    "nq_decontaminated": "NQ*",
    "quora_decontaminated": "Q*",
    "scidocs_decontaminated": "SD*",
    "trec_covid_decontaminated": "T-C*",
}


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if pd.isna(text):
        return "-"
    text = str(text)
    # Escape underscore and other special characters
    replacements = {
        "_": "\\_",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str = "Results",
    label: str = "tab:results",
    sig_df: pd.DataFrame = None,
    metric_col: str = "nDCG@10",
) -> str:
    """
    Convert a grouped dataframe with multi-level columns to a LaTeX table.

    Args:
        df: DataFrame with hierarchical columns (metric, AP/RR@10/nDCG@10, mean/var)
        caption: Table caption
        label: Table label for referencing
        sig_df: Optional DataFrame with statistical significance results
        include_pm: Whether to include +/- variance values in cells

    Returns:
        LaTeX table string
    """
    # Sort the MultiIndex columns to avoid performance warnings
    df = df.sort_index(axis=1)

    # Normalize significance DataFrame if provided: reset index when it's a
    # MultiIndex (the output of `run_statistical_tests` uses a MultiIndex),
    # and ensure a `p_value` column exists for downstream checks.
    if sig_df is not None:
        try:
            if isinstance(sig_df.index, pd.MultiIndex):
                sig_df = sig_df.reset_index()
        except Exception:
            sig_df = sig_df.copy()

        # Ensure there's a `p_value` column (tolerant renaming if needed)
        if "p_value" not in sig_df.columns:
            for col in sig_df.columns:
                if "p" in str(col).lower() and (
                    "value" in str(col).lower() or "val" in str(col).lower()
                ):
                    sig_df = sig_df.rename(columns={col: "p_value"})
                    break

    # Helper to find a column by level value (works with tuples and single-level names)
    def find_col_by_value(value: str):
        for col in df.columns:
            try:
                if isinstance(col, tuple):
                    if any(str(x) == value for x in col):
                        return col
                else:
                    if str(col) == value:
                        return col
            except Exception:
                continue
        return None

    dataset_col = find_col_by_value("dataset") or df.columns[0]
    first_stage_col = find_col_by_value("first_stage")
    scorer_col = find_col_by_value("scorer")
    base_col = find_col_by_value("base")

    # Find nDCG@10 mean/var columns
    ndcg_mean_col = None
    ndcg_var_col = None
    for col in df.columns:
        if isinstance(col, tuple):
            if (
                len(col) >= 3
                and col[0] == "metric"
                and col[1] == metric_col
                and col[2] == "mean"
            ):
                ndcg_mean_col = col
            if (
                len(col) >= 3
                and col[0] == "metric"
                and col[1] == metric_col
                and col[2] == "var"
            ):
                ndcg_var_col = col

    # Fallback: search for strings containing both tokens
    if ndcg_mean_col is None or ndcg_var_col is None:
        for col in df.columns:
            s = " ".join(map(str, col)) if isinstance(col, tuple) else str(col)
            if metric_col in s and "mean" in s and ndcg_mean_col is None:
                ndcg_mean_col = col
            if metric_col in s and "var" in s and ndcg_var_col is None:
                ndcg_var_col = col

    # Final fallback: just look for the metric name itself if no mean/var columns found
    if ndcg_mean_col is None:
        ndcg_mean_col = find_col_by_value(metric_col)

    # Identify tag columns (non-metric, non-dataset, non-n_runs)
    excluded_col_names = {"dataset", "n_runs", "seed"}
    tag_cols = []
    for col in df.columns:
        col_name = col[0] if isinstance(col, tuple) else str(col)
        col_stat = col[1] if isinstance(col, tuple) and len(col) > 1 else ""
        if (
            col_name not in excluded_col_names
            and col_name != "metric"
            and col_stat not in ("mean", "var")
        ):
            if (
                col not in (dataset_col, ndcg_mean_col, ndcg_var_col)
                and col not in tag_cols
            ):
                tag_cols.append(col)
    table: dict = {}
    table_values: dict = {}  # numeric mean values for averaging
    table_meta: dict = {}  # store first_stage/scorer for each model label
    sig_pvals: dict = {}
    datasets = set()

    for _, row in df.iterrows():
        # dataset value
        try:
            dataset_val = row[dataset_col]
        except Exception:
            # fallback: try locating by name in index
            dataset_val = row.get("dataset", None)

        if pd.isna(dataset_val):
            continue
        dataset = str(dataset_val)
        datasets.add(dataset)

        # model label construction
        parts = []
        if first_stage_col is not None:
            try:
                fs = row[first_stage_col]
                if pd.notna(fs) and str(fs) != "nan":
                    parts.append(str(fs))
            except Exception:
                pass

        scorer_val = None
        if scorer_col is not None:
            try:
                sc = row[scorer_col]
                if pd.notna(sc) and str(sc) != "nan":
                    scorer_val = str(sc)
                    parts.append(scorer_val)
            except Exception:
                pass

        # If scorer is absent or NaN (fine-tuning mode), use base model
        model_parts = []
        if scorer_val is None and base_col is not None:
            try:
                b = row[base_col]
                if pd.notna(b) and str(b) != "nan":
                    model_parts.append(str(b))
            except Exception:
                pass

        # Add any grid search / tag columns that vary across rows in df
        extra_tags = []
        for tc in tag_cols:
            if tc in (first_stage_col, scorer_col, base_col):
                continue
            tc_name = tc[0] if isinstance(tc, tuple) else str(tc)
            try:
                val = row[tc]
                if pd.notna(val) and str(val) != "nan":
                    if df[tc].nunique(dropna=True) > 1:
                        extra_tags.append(f"{tc_name}={val}")
            except Exception:
                pass

        if model_parts:
            if extra_tags:
                model_parts.append(f"({', '.join(extra_tags)})")
            parts.append(" ".join(model_parts))
        elif extra_tags:
            parts.append(", ".join(extra_tags))

        model_label = " / ".join(parts) if parts else "model"

        # record meta for significance lookup
        table_meta.setdefault(model_label, {})
        try:
            if first_stage_col is not None:
                fs_val = row[first_stage_col]
                table_meta[model_label]["first_stage"] = (
                    None if pd.isna(fs_val) else str(fs_val)
                )
            else:
                table_meta[model_label]["first_stage"] = None
        except Exception:
            table_meta[model_label]["first_stage"] = None
        try:
            if scorer_col is not None:
                sc_val = row[scorer_col]
                table_meta[model_label]["scorer"] = (
                    None if pd.isna(sc_val) else str(sc_val)
                )
            else:
                table_meta[model_label]["scorer"] = None
        except Exception:
            table_meta[model_label]["scorer"] = None

        # get mean and var
        mean_val = None
        try:
            if ndcg_mean_col is not None:
                mean_val = row[ndcg_mean_col]
        except Exception:
            mean_val = None
        # var_val = None
        # try:
        #     if ndcg_var_col is not None:
        #         var_val = row[ndcg_var_col]
        # except Exception:
        #     var_val = None

        # Format cell: display as percentage (multiply by 100) with one decimal, no variance
        try:
            if pd.isna(mean_val):
                cell = "-"
            else:
                m = float(mean_val)
                cell = f"{100 * m:.1f}"
        except Exception:
            cell = "-"

        # If significance DataFrame provided, check p-value for this dataset/metric and model
        # We'll record p-values for post-processing (arrows) rather than modify cell here
        if sig_df is not None and cell != "-":
            try:
                metric_name = metric_col
                sd = sig_df[sig_df["dataset"].astype(str) == dataset]
                sd = sd[sd["measure"].astype(str) == metric_name]
                scorer_name = table_meta.get(model_label, {}).get("scorer")
                first_stage_name = table_meta.get(model_label, {}).get("first_stage")
                sig_row = None
                if scorer_name:
                    mask = sd["model"].astype(str).str.contains(scorer_name, na=False)
                    if mask.any():
                        sig_row = sd[mask]
                # fallback: try matching first_stage
                if sig_row is None or sig_row.empty:
                    if first_stage_name:
                        mask2 = (
                            sd["model"]
                            .astype(str)
                            .str.contains(first_stage_name, na=False)
                        )
                        if mask2.any():
                            sig_row = sd[mask2]
                if sig_row is not None and not sig_row.empty:
                    # take smallest p-value among matches
                    pval = float(sig_row["p_value"].astype(float).min())
                    # store p-value for later arrow assignment
                    try:
                        sig_pvals.setdefault(model_label, {})[dataset] = pval
                    except Exception:
                        pass
            except Exception:
                pass

        table.setdefault(model_label, {})[dataset] = cell
        # store numeric mean for averages
        try:
            numeric_mean = None if pd.isna(mean_val) else float(mean_val)
        except Exception:
            numeric_mean = None
        table_values.setdefault(model_label, {})[dataset] = numeric_mean

    # Build LaTeX table
    # Order datasets: MSMARCO first, then BEIR and finally LoTTE.
    preferred = ["msmarco_dev", "trec2019", "trec2020"]
    ordered = [d for d in preferred if d in datasets]
    # BEIR-like datasets: all datasets except preferred and those containing 'lotte'
    beir = sorted([d for d in datasets if d not in preferred and "lotte" not in d])
    # Lotte datasets go at the end (just before the Avg columns)
    lotte = sorted([d for d in datasets if "lotte" in d])
    datasets = ordered + beir + lotte
    model_labels = sorted(table.keys())

    # Post-process significance markers: replace bolding with directional arrows
    if sig_df is not None:
        for model in model_labels:
            for d in datasets:
                try:
                    pval = sig_pvals.get(model, {}).get(d, None)
                except Exception:
                    pval = None
                if pval is None or pval >= 0.05:
                    continue
                # numeric mean for this cell
                mean_val = table_values.get(model, {}).get(d)
                if mean_val is None:
                    continue
                # find baseline mean for this dataset
                baseline_mean = None
                for m_label, meta in table_meta.items():
                    sc_name = meta.get("scorer") or ""
                    if "baseline" in sc_name.lower():
                        b = table_values.get(m_label, {}).get(d)
                        if b is not None:
                            baseline_mean = b
                            break
                if baseline_mean is None:
                    continue
                if mean_val > baseline_mean:
                    arrow = " $\\uparrow$"
                elif mean_val < baseline_mean:
                    arrow = " $\\downarrow$"
                else:
                    arrow = ""
                if arrow:
                    cur = table.get(model, {}).get(d, "-")
                    if cur != "-":
                        table[model][d] = f"{cur}{arrow}"

    latex_lines = []
    latex_lines.append("\\begin{table*}[h]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")

    # Add two extra columns for Avg. and Avg. (OOD)
    col_spec = "r" + "c" * len(datasets) + "cccc"
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    # Use abbreviations for dataset display names when available
    header = ["Model"] + [escape_latex(DATASET_TO_ABB.get(d, d)) for d in datasets]
    header.extend(
        [
            "Avg. (ID)",
            "Avg. (BEIR OOD)",
            "Avg. (LoTTE OOD)",
            "Avg. (OOD)",
        ]
    )
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")

    for model in model_labels:
        row_parts = [escape_latex(model)]
        for d in datasets:
            cell = table.get(model, {}).get(d, "-")
            row_parts.append(cell)

        # In-domain average (preferred only)
        id_ds = [d for d in datasets if d in preferred]
        id_vals = [
            table_values.get(model, {}).get(d)
            for d in id_ds
            if table_values.get(model, {}).get(d) is not None
        ]
        if id_vals:
            avg = sum(id_vals) / len(id_vals)
            row_parts.append(f"{100 * avg:.1f}")
        else:
            row_parts.append("-")

        # BEIR-only OOD (already computed in `beir` list above)
        beir_vals = [
            table_values.get(model, {}).get(d)
            for d in beir
            if table_values.get(model, {}).get(d) is not None
        ]
        if beir_vals:
            beir_avg = sum(beir_vals) / len(beir_vals)
            row_parts.append(f"{100 * beir_avg:.1f}")
        else:
            row_parts.append("-")

        # LoTTE-only OOD
        lotte_vals = [
            table_values.get(model, {}).get(d)
            for d in lotte
            if table_values.get(model, {}).get(d) is not None
        ]
        if lotte_vals:
            lotte_avg = sum(lotte_vals) / len(lotte_vals)
            row_parts.append(f"{100 * lotte_avg:.1f}")
        else:
            row_parts.append("-")

        # All OOD average: BEIR + LoTTE (exclude preferred)
        all_ood_ds = [d for d in datasets if d not in preferred]
        all_ood_vals = [
            table_values.get(model, {}).get(d)
            for d in all_ood_ds
            if table_values.get(model, {}).get(d) is not None
        ]
        if all_ood_vals:
            all_ood_avg = sum(all_ood_vals) / len(all_ood_vals)
            row_parts.append(f"{100 * all_ood_avg:.1f}")
        else:
            row_parts.append("-")

        latex_lines.append(" & ".join(row_parts) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)
