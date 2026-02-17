#!/usr/bin/env python
"""Profiling helpers for mid-fusion ranking models."""

import gc
import logging
import statistics
import time
import torch
import sys
import os

from xpm_torch import parameters

# Add the script directory to path to allow importing profiling_utils if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModel
from typing import Optional, List, Tuple, Union, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

from models.midFusion_modeling import MiniLMMidFusionCrossEncoder, EttinMidFusionCrossEncoder
from xpmir.learning import ModuleInitMode, ModuleInitOptions
from profiling_utils import benchmark_model, DummyBatch

try:
    from datamaestro_text.data.ir import TextItem
except ImportError:
    class TextItem:
        pass


_reference_state_cache: dict[str, dict[str, torch.Tensor]] = {}

def _load_reference_weights(hf_id: str) -> dict[str, torch.Tensor] | None:
    try:
        print(f"[info] loading reference model weights for verification from {hf_id}...")
        reference = AutoModel.from_pretrained(hf_id)
    except Exception as exc:
        print(f"[warn] failed to load reference model for verification ({exc}).")
        return None
    
    state = reference.state_dict()
    _reference_state_cache[hf_id] = {k: v.detach().cpu() for k,v in state.items()}
    return _reference_state_cache[hf_id]

def verify_loaded_weights(model, hf_id, name):
    ref_state = _reference_state_cache.get(hf_id)
    if not ref_state:
        ref_state = _load_reference_weights(hf_id)
        if not ref_state:
            return

    print(f"[info] Verifying weights for {name}...")
    
    # Identify split layer from model if possible, assuming 6 as per default in script
    split_layer = getattr(model, "split_layer", 6)

    # Helper to check equality
    def check_param(param_name, ref_key, target_tensor):
        if ref_key not in ref_state:
            # print(f"[warn] {name}: Reference key {ref_key} not found for {param_name}")
            return
        
        ref_tensor = ref_state[ref_key]
        if ref_tensor.shape != target_tensor.shape:
             print(f"[warn] {name}: Shape mismatch for {param_name}: {target_tensor.shape} vs ref {ref_tensor.shape}")
             return

        diff = (ref_tensor - target_tensor.cpu()).abs().max().item()
        if diff > 1e-5:
             print(f"[warn] {name}: Parameter {param_name} differs from reference (max diff {diff:.3e})")
    
    # Check embeddings
    for k, v in model.embeddings.state_dict().items():
        ref_key = f"embeddings.{k}"
        check_param(f"embeddings.{k}", ref_key, v)

    # Check bottom layers
    for i, layer in enumerate(model.bottom_layers):
        for k, v in layer.state_dict().items():
            ref_key = f"encoder.layer.{i}.{k}"
            check_param(f"bottom_layers.{i}.{k}", ref_key, v)
            
    is_asymmetric = "Asymmetric" in name
    
    for j, layer in enumerate(model.top_layers):
        original_idx = split_layer + j
        for k, v in layer.state_dict().items():
            ref_key = f"encoder.layer.{original_idx}.{k}"
            
            if is_asymmetric:
                # In MiniLMMidFusionCrossEncoder, we only copy self attention and FFN.
                if "crossattention" in k:
                    continue 
                check_param(f"top_layers.{j}.{k}", ref_key, v)
            else:
                 check_param(f"top_layers.{j}.{k}", ref_key, v)

    print(f"[info] Weight verification for {name} complete.")


def test_compression_dimensions():
    logger = logging.getLogger(__name__)
    logger.info("--- Testing Compression Dimensions ---")
    model_name = "microsoft/MiniLM-L12-H384-uncased"
    compress_dim = 2
    
    # Expected original dimensions for MiniLM-L12-H384-uncased
    # hidden_size: 384
    # intermediate_size: 1536
    # num_attention_heads: 12
    
    model = MiniLMMidFusionCrossEncoder.from_kwargs(
        hf_id=model_name,
        compress_dim=compress_dim,
        random_top_layers=True, # Avoid weight copy mismatch issues for this test
        merge_layer=6
    )
    model.initialize(ModuleInitOptions(mode=ModuleInitMode.DEFAULT))

    head_config = model.head_config
    
    expected_hidden = 384 // compress_dim
    expected_intermediate = 1536 // compress_dim
    expected_heads = 12 // compress_dim

    logger.info(f"Checking hidden_size: {head_config.hidden_size} (expected {expected_hidden})")
    assert head_config.hidden_size == expected_hidden, f"hidden_size mismatch: {head_config.hidden_size} != {expected_hidden}"
    
    logger.info(f"Checking intermediate_size: {head_config.intermediate_size} (expected {expected_intermediate})")
    assert head_config.intermediate_size == expected_intermediate, f"intermediate_size mismatch: {head_config.intermediate_size} != {expected_intermediate}"
    
    logger.info(f"Checking num_attention_heads: {head_config.num_attention_heads} (expected {expected_heads})")
    assert head_config.num_attention_heads == expected_heads, f"num_attention_heads mismatch: {head_config.num_attention_heads} != {expected_heads}"
    
    logger.info("Compression Dimensions Test Passed!")


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
    model_name = "bert-base-uncased"
    model_name = "nreimers/MiniLM-L6-H384-uncased"
    model_name = "jhu-clsp/ettin-encoder-150m"

    # tokenize documents to see the shapes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(
        [doc_text for _ in range(batch_size)],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    logger.info(f"Tokenized document shape: {encoded['input_ids'].shape}")
    logger.info(f"Tokenized attention_mask shape: {encoded['attention_mask'].shape}")

    # Tokenization is now handled inside the model's forward pass
    logger.info("Tokenization will be handled by the model.")

    warmup_steps = 5
    num_runs = 100


    kwargs = dict(
        # print_model_summary=True,
        random_top_layers=False,
        # compress_dim = 2.0,
        drop_layer=9,
        merge_layer=6,
    )
    
    if kwargs.get("compress_dim", 1) > 1: test_compression_dimensions()

    if "minilm" in model_name.lower():
        cls = MiniLMMidFusionCrossEncoder
        name = "MiniLMMidFusionCrossEncoder"
    elif "ettin" in model_name.lower():
        cls = EttinMidFusionCrossEncoder
        name = "EttinMidFusionCrossEncoder"
    else:
        raise ValueError(f"Model {model_name} not supported in this profiling script.")
    
    benchmark_model(
        cls,
        name,
        batch,
        device,
        model_name,
        warmup_steps,
        num_runs,
        verify_weights_fn=verify_loaded_weights,
        print_model_summary=True,
        **kwargs,
    )


if __name__ == "__main__":
    main()
