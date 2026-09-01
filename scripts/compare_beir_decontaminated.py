#!/usr/bin/env python3
"""
Script to compare topic count, qrel count, and total judgments for all 14 BEIR
datasets between `ai.lighton.beir_decontaminated.<name>` and `org.beir.<name>`.
"""

import argparse
import logging
import sys
from pathlib import Path

from datamaestro import prepare_dataset

# Add src to python path for logging_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from logging_utils import setup_logging

    setup_logging(level=logging.INFO)
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("compare_beir_decontaminated")

# Mapping of dataset name to (LightOn dataset base ID, BEIR dataset ID)
BEIR_DATASET_MAPPING = {
    "arguana": ("ai.lighton.beir_decontaminated.arguana", "org.beir.arguana"),
    "climate_fever": (
        "ai.lighton.beir_decontaminated.climate_fever",
        "org.beir.climate.fever",
    ),
    "dbpedia_entity": (
        "ai.lighton.beir_decontaminated.dbpedia_entity",
        "org.beir.dbpedia.entity.test",
    ),
    "fever": ("ai.lighton.beir_decontaminated.fever", "org.beir.fever.test"),
    "fiqa": ("ai.lighton.beir_decontaminated.fiqa", "org.beir.fiqa.test"),
    "hotpotqa": (
        "ai.lighton.beir_decontaminated.hotpotqa",
        "org.beir.hotpotqa.test",
    ),
    "msmarco": (
        "ai.lighton.beir_decontaminated.msmarco",
        "org.beir.nano-beir.msmarco",
    ),
    "nfcorpus": (
        "ai.lighton.beir_decontaminated.nfcorpus",
        "org.beir.nfcorpus.test",
    ),
    "nq": ("ai.lighton.beir_decontaminated.nq", "org.beir.nq"),
    "quora": ("ai.lighton.beir_decontaminated.quora", "org.beir.quora.test"),
    "scidocs": ("ai.lighton.beir_decontaminated.scidocs", "org.beir.scidocs"),
    "scifact": ("ai.lighton.beir_decontaminated.scifact", "org.beir.scifact.test"),
    "trec_covid": (
        "ai.lighton.beir_decontaminated.trec_covid",
        "org.beir.trec.covid",
    ),
    "webis_touche2020": (
        "ai.lighton.beir_decontaminated.webis_touche2020",
        "org.beir.webis.touche2020.v2",
    ),
}


def prepare_raw_resources(name: str):
    """Pre-download raw parquet files for a dataset without building the docstore."""
    logger.info("[%s] Pre-downloading raw files (prepare-only)...", name)

    # Prepare queries and qrels
    queries_id = f"ai.lighton.beir_decontaminated.{name}.queries"
    qrels_id = f"ai.lighton.beir_decontaminated.{name}.qrels.test"

    logger.info("  Downloading queries: %s", queries_id)
    queries_obj = prepare_dataset(queries_id)

    logger.info("  Downloading qrels: %s", qrels_id)
    qrels_obj = prepare_dataset(qrels_id)

    logger.info("  Downloading corpus parquet (without building docstore)...")
    try:
        import datamaestro_ir.config.ai.lighton.beir_decontaminated as mod

        cls_name = f"LightOnDecontaminated_{name}_Docs"
        if hasattr(mod, cls_name):
            cls = getattr(mod, cls_name)
            cls.CORPUS.download()
            logger.info("  Successfully downloaded corpus parquet!")
        else:
            logger.warning("  Could not find class %s in module", cls_name)
    except Exception as e:
        logger.warning("  Could not trigger raw CORPUS download: %s", e)

    return queries_obj, qrels_obj


def get_dataset_stats(name: str, dataset_id: str, is_lighton: bool = True):
    """Load topics, qrels, and document counts for dataset.

    Avoids building heavy compressed document stores for large corpora (e.g. Wikipedia)
    on HPC login nodes by reading Parquet metadata directly for LightOn datasets.
    """
    logger.info("Preparing dataset stats for: %s", dataset_id)
    doc_count = 0
    try:
        if is_lighton:
            queries_id = f"ai.lighton.beir_decontaminated.{name}.queries"
            qrels_id = f"ai.lighton.beir_decontaminated.{name}.qrels.test"

            logger.info("  [%s] Reading topics from %s...", name, queries_id)
            topics_obj = prepare_dataset(queries_id)
            topics = list(topics_obj.iter())

            logger.info("  [%s] Reading qrels from %s...", name, qrels_id)
            qrels_obj = prepare_dataset(qrels_id)
            qrels = list(qrels_obj.iter())

            # Read document count from corpus.parquet metadata
            try:
                import pyarrow.parquet as pq
                import datamaestro_ir.config.ai.lighton.beir_decontaminated as mod

                cls_name = f"LightOnDecontaminated_{name}_Docs"
                if hasattr(mod, cls_name):
                    cls = getattr(mod, cls_name)
                    if cls.CORPUS.path.exists():
                        doc_count = pq.read_metadata(cls.CORPUS.path).num_rows
                        logger.info("  [%s] Found %d documents (from Parquet metadata)", name, doc_count)
            except Exception as e:
                logger.warning("  [%s] Could not read document count: %s", name, e)
        else:
            ds = prepare_dataset(dataset_id)
            logger.info("  [%s] Reading topics from %s...", name, dataset_id)
            topics = list(ds.topics.iter())
            logger.info("  [%s] Reading assessments/qrels...", name)
            qrels = list(ds.assessments.iter())
            try:
                if ds.documents.path.exists():
                    doc_count = ds.documents.documentcount
            except Exception:
                doc_count = 0

        total_judgments = sum(len(q.assessments) for q in qrels)
        logger.info(
            "  [%s] Found %d docs, %d topics, %d qrel topics (%d judgments)",
            name,
            doc_count,
            len(topics),
            len(qrels),
            total_judgments,
        )

        return {
            "status": "OK",
            "docs": doc_count,
            "topics": len(topics),
            "qrels_topics": len(qrels),
            "judgments": total_judgments,
        }
    except Exception as e:
        logger.exception("Failed to load dataset '%s'", dataset_id)
        return {
            "status": "ERROR",
            "docs": 0,
            "topics": 0,
            "qrels_topics": 0,
            "judgments": 0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compare all 14 LightOn decontaminated BEIR datasets against BEIR originals."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset name to compare (default: compare all 14 datasets).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download raw dataset files only without building heavy document stores.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    try:
        from logging_utils import setup_logging

        setup_logging(level=log_level)
    except ImportError:
        logging.basicConfig(level=log_level)

    logger.setLevel(log_level)

    targets = (
        {args.dataset: BEIR_DATASET_MAPPING[args.dataset]}
        if args.dataset and args.dataset in BEIR_DATASET_MAPPING
        else BEIR_DATASET_MAPPING
    )

    logger.info(
        "Comparing %d dataset(s) (prepare_only=%s, debug=%s)...",
        len(targets),
        args.prepare_only,
        args.debug,
    )

    if args.prepare_only:
        logger.info("Executing --prepare-only: downloading raw files...")
        for name in targets:
            prepare_raw_resources(name)
        logger.info("All raw resources downloaded successfully!")
        return

    summary_rows = []

    for name, (lighton_id, beir_id) in targets.items():
        logger.info("==================================================")
        logger.info(
            "Processing '%s': LightOn (%s) vs BEIR (%s)", name, lighton_id, beir_id
        )

        lighton_stats = get_dataset_stats(name, lighton_id, is_lighton=True)
        beir_stats = get_dataset_stats(name, beir_id, is_lighton=False)

        summary_rows.append(
            {
                "name": name,
                "lighton_docs": (
                    f"{lighton_stats['docs']:,}"
                    if lighton_stats["status"] == "OK" and lighton_stats["docs"] > 0
                    else ("N/A" if lighton_stats["status"] == "OK" else "ERROR")
                ),
                "beir_docs": (
                    f"{beir_stats['docs']:,}"
                    if beir_stats["status"] == "OK" and beir_stats["docs"] > 0
                    else ("N/A" if beir_stats["status"] == "OK" else "ERROR")
                ),
                "lighton_topics": (
                    lighton_stats["topics"]
                    if lighton_stats["status"] == "OK"
                    else "ERROR"
                ),
                "beir_topics": (
                    beir_stats["topics"] if beir_stats["status"] == "OK" else "ERROR"
                ),
                "lighton_qrels": (
                    lighton_stats["qrels_topics"]
                    if lighton_stats["status"] == "OK"
                    else "ERROR"
                ),
                "beir_qrels": (
                    beir_stats["qrels_topics"]
                    if beir_stats["status"] == "OK"
                    else "ERROR"
                ),
                "lighton_judgments": (
                    lighton_stats["judgments"]
                    if lighton_stats["status"] == "OK"
                    else "ERROR"
                ),
                "beir_judgments": (
                    beir_stats["judgments"] if beir_stats["status"] == "OK" else "ERROR"
                ),
            }
        )

    # Print markdown table
    print("\n" + "=" * 135)
    print(
        "                      FULL BEIR DECONTAMINATED COMPARISON TABLE                      "
    )
    print("=" * 135)
    header = f"| {'Dataset':<18} | {'LightOn Docs':<14} | {'BEIR Docs':<14} | {'LightOn Topics':<14} | {'BEIR Topics':<11} | {'LightOn Qrels':<13} | {'BEIR Qrels':<10} | {'LightOn Judg.':<13} | {'BEIR Judg.':<10} |"
    divider = (
        "|:"
        + "-" * 18
        + ":|:"
        + "-" * 14
        + ":|:"
        + "-" * 14
        + ":|:"
        + "-" * 14
        + ":|:"
        + "-" * 11
        + ":|:"
        + "-" * 13
        + ":|:"
        + "-" * 10
        + ":|:"
        + "-" * 13
        + ":|:"
        + "-" * 10
        + ":|"
    )
    print(header)
    print(divider)

    for row in summary_rows:
        line = f"| {row['name']:<18} | {row['lighton_docs']:<14} | {row['beir_docs']:<14} | {str(row['lighton_topics']):<14} | {str(row['beir_topics']):<11} | {str(row['lighton_qrels']):<13} | {str(row['beir_qrels']):<10} | {str(row['lighton_judgments']):<13} | {str(row['beir_judgments']):<10} |"
        print(line)

    print("=" * 135 + "\n")


if __name__ == "__main__":
    main()
