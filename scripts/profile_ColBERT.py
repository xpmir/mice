#!/usr/bin/env python
"""Profiling helpers for PyLate ColBERT ranking models."""

import logging
import torch
import sys
import os

# Add the script directory to path to allow importing profiling_utils if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ColBERT import PyLateColBERT
from profiling_utils import benchmark_model, DummyBatch

try:
    from datamaestro_text.data.ir import TextItem
except ImportError:
    class TextItem:
        pass


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    query_text = "what is artificial intelligence?"
    doc_text = (
        "Artificial intelligence studies intelligent agents blah -, lorem ipsum dolor sit amet consectetur adipiscing elit "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua . Ut enim ad minim veniam quis nostrud exercitation "
        "ullamco laboris nisi ut aliquip ex ea commodo consequat . Duis aute irure dolor in reprehenderit in voluptate velit esse "
        "cillum dolore eu fugiat nulla pariatur . Excepteur sint occaecat cupidatat non proident , sunt in culpa qui officia deserunt "
        "mollit anim id est laborum ."
    )

    batch_size = 128
    logger.info(f"Using batch size: {batch_size}")

    batch = DummyBatch.build(batch_size, query_text, doc_text)

    model_name = "microsoft/MiniLM-L12-H384-uncased"
    model_name = "lightonai/colbertv2.0"
    model_name = "colbert-ir/colbertv2.0"
    model_name = "answerdotai/answerai-colbert-small-v1"

    warmup_steps = 2
    num_runs = 10

    benchmark_model(
        PyLateColBERT,
        "PyLateColBERT",
        batch,
        device,
        model_name,
        warmup_steps,
        num_runs,
        print_model_summary=True,
        # max_length=512,
    )


if __name__ == "__main__":
    main()
