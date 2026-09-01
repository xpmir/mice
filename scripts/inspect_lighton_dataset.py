#!/usr/bin/env python3
"""
Script to inspect the first samples of the "ai.lighton.embeddings_pre_training" dataset
using datamaestro.
"""

import argparse
import logging
import sys
from datamaestro import prepare_dataset

# Set up logging to print to stderr/stdout nicely
from logging_utils import setup_logging
setup_logging(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inspect_lighton")

def main():
    parser = argparse.ArgumentParser(
        description="Inspect the first samples of the 'ai.lighton.embeddings_pre_training' dataset using datamaestro."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="agnews",
        help="The name variant of the dataset to load (one of 73 values, e.g., 'agnews', 'amazon_qa', 'wiki_qa', etc.)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to inspect and print."
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (downloads/caches the dataset locally instead)."
    )
    args = parser.parse_args()

    # Build the dataset ID string with parameters
    streaming_val = "false" if args.no_streaming else "true"
    dataset_id = f"ai.lighton.embeddings_pre_training[name={args.name},streaming={streaming_val}]"
    logger.info("Preparing dataset with ID: %s", dataset_id)

    try:
        # Load the dataset instance using datamaestro
        data = prepare_dataset(dataset_id).instance()
    except Exception as e:
        logger.error("Failed to load dataset '%s'. Make sure datamaestro and the datasets library are installed properly.", dataset_id)
        raise e

    logger.info("Dataset loaded successfully! Iterating first %d samples...", args.num_samples)

    count = 0
    try:
        for i, sample in enumerate(data):
            if i >= args.num_samples:
                break

            # Accessing fields from PointwiseDistillationSample
            query_text = sample.query["text_item"].text
            doc_text = sample.document.document["text_item"].text
            score = sample.document.score

            print("-" * 80)
            print(f"Sample #{i + 1}")
            print(f"Query:    {query_text}")
            print(f"Document: {doc_text}")
            print(f"Score:    {score:.4f}")
            count += 1

        logger.info("Successfully inspected %d samples.", count)
    except Exception as e:
        logger.exception("An error occurred while iterating through the dataset.")
        raise e

if __name__ == "__main__":
    main()
