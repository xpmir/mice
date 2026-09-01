#!/usr/bin/env python3
"""
Script to inspect the first samples of the "cross_encoder.ettin_reranker_v1_data" dataset
and pre-download/cache it locally using datamaestro.
"""

import argparse
import logging
import sys
from datamaestro import prepare_dataset
from datamaestro_ir.config.cross_encoder.ettin_reranker_v1_data import CONFIGS

# Set up logging
from logging_utils import setup_logging
setup_logging(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inspect_ettin")

def inspect_config(name: str, num_samples: int, no_streaming: bool):
    streaming_val = "false" if no_streaming else "true"
    dataset_id = f"cross_encoder.ettin_reranker_v1_data[name={name},streaming={streaming_val}]"
    logger.info("Preparing dataset with ID: %s", dataset_id)

    try:
        data = prepare_dataset(dataset_id).instance()
    except Exception as e:
        logger.error("Failed to load dataset '%s'.", dataset_id)
        raise e

    logger.info("Dataset '%s' loaded successfully. Iterating first %d samples...", name, num_samples)

    count = 0
    try:
        for i, sample in enumerate(data):
            if i >= num_samples:
                break

            query_text = sample.query["text_item"].text
            doc_text = sample.document.document["text_item"].text
            score = sample.document.score

            if num_samples > 0:
                print("-" * 80)
                print(f"[{name}] Sample #{i + 1}")
                print(f"Query:    {query_text}")
                print(f"Document: {doc_text}")
                print(f"Score:    {score:.4f}")
            count += 1

        logger.info("Successfully loaded/inspected subset '%s'.", name)
    except Exception as e:
        logger.exception("An error occurred while iterating through '%s'.", name)
        raise e

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and pre-download the 'cross_encoder.ettin_reranker_v1_data' dataset configurations."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rerank_msmarco",
        help="The config name of the dataset to load (default: 'rerank_msmarco')."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pre-download and inspect ALL 39 configs of the Ettin reranker dataset."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to print per loaded config (set to 0 for downloading only)."
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (downloads/caches the dataset locally instead)."
    )
    args = parser.parse_args()

    if args.all:
        logger.info("Pre-downloading/inspecting ALL 39 Ettin configurations...")
        for name in CONFIGS:
            try:
                inspect_config(name, args.num_samples, args.no_streaming)
            except Exception as e:
                logger.error("Error processing config '%s': %s", name, str(e))
                logger.info("Continuing with remaining configurations...")
    else:
        inspect_config(args.name, args.num_samples, args.no_streaming)

if __name__ == "__main__":
    main()
