"""
Verification script for hf_cross_scorer and HFCrossScorer pref_attn_implementation and Fabric FA2 fallback.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
import pytest
import logging
from transformers import AutoConfig
from lightning.fabric import Fabric

from xpmir.neural.huggingface import hf_cross_scorer, HFCrossScorer
from xpmir.neural.sentence_transformers import st_cross_scorer, STCrossEncoder
from xpmir.letor.records import PointwiseItems
from xpm_torch.module import fallback_fa2_if_incompatible_precision


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_hf_cross_scorer_fa2")

MODEL_ID = "jhu-clsp/ettin-encoder-17m"


def test_hf_cross_scorer_pref_attn_implementation():
    """Verify that hf_cross_scorer properly sets pref_attn_implementation on HFCrossScorer."""
    logger.info("--- Test 1: hf_cross_scorer with pref_attn_implementation='sdpa' ---")
    scorer, init_tasks = hf_cross_scorer(
        hf_id=MODEL_ID,
        pref_attn_implementation="sdpa",
    )
    assert scorer.pref_attn_implementation == "sdpa"

    for t in init_tasks:
        t.instance().execute()

    scorer = scorer.instance()
    scorer.initialize()

    # Verify that the underlying HuggingFace model config recorded 'sdpa'
    actual_attn = getattr(scorer.encoder.model.config, "_attn_implementation", None)
    logger.info(f"Actual _attn_implementation in HF config: {actual_attn}")
    assert actual_attn == "sdpa"

    # Forward pass sanity check
    records = PointwiseItems.from_texts(
        topics=["Which planet is the Red Planet?"],
        documents=["Mars is known as the Red Planet."],
    )
    with torch.no_grad():
        out = scorer(records)
    assert isinstance(out, torch.Tensor)
    logger.info(f"Forward pass output: {out}")


def test_fabric_fa2_fallback_fp32():
    """Verify that setup_with_fabric correctly downgrades FA2 to SDPA under FP32 Fabric precision."""
    logger.info("--- Test 2: Fabric setup_with_fabric FP32 fallback ---")
    scorer, init_tasks = hf_cross_scorer(
        hf_id=MODEL_ID,
        pref_attn_implementation="flash_attention_2",
    )
    for t in init_tasks:
        t.instance().execute()

    scorer = scorer.instance()
    scorer.initialize()

    # Manually simulate flash_attention_2 in config to test fallback logic
    scorer.encoder.model.config._attn_implementation = "flash_attention_2"
    assert scorer.encoder.model.config._attn_implementation == "flash_attention_2"

    # Simulate Fabric under FP32 precision ('32-true')
    fabric = Fabric(accelerator="cpu", precision="32-true")

    # Run setup_with_fabric
    wrapped_scorer = scorer.setup_with_fabric(fabric)

    # Verify fallback to 'sdpa'
    actual_attn = getattr(scorer.encoder.model.config, "_attn_implementation", None)
    logger.info(f"Attention implementation after setup_with_fabric under FP32: {actual_attn}")
    assert actual_attn == "sdpa"


if __name__ == "__main__":
    logger.info("Running HFCrossScorer FA2 tests...")
    test_hf_cross_scorer_pref_attn_implementation()
    test_fabric_fa2_fallback_fp32()
    logger.info("✅ All HFCrossScorer FA2 tests passed successfully!")
