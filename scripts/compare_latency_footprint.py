#!/usr/bin/env python
"""
Comparison script for MidFusion and ColBERT latency and memory footprint.
"""

import logging
import torch
import sys
import os
from tabulate import tabulate
from transformers import AutoTokenizer

# Add the script directory to path to allow importing profiling_utils if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from profiling_utils import benchmark_model, DummyBatch
from models.ColBERT import PyLateColBERT
from models.midFusion_modeling import MiniLMMidFusionCrossEncoder, EttinMidFusionCrossEncoder
from xpmir.neural.huggingface import HFCrossScorer

def main():
    # Query and Document similar to previous profiling scripts
    query_text = "what is artificial intelligence?"
    doc_text = (
        10 * "Artificial intelligence studies intelligent agents blah -, lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua . Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat . Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur . Excepteur sint occaecat cupidatat non proident , sunt in culpa qui officia deserunt mollit anim id est laborum ."
    )

    batch = DummyBatch.build(batch_size, query_text, doc_text)
    


    # tokenize documents to see the shapes
    tokenizer = AutoTokenizer.from_pretrained(midfusion_model_name)
    encoded = tokenizer(
        [doc_text for _ in range(batch_size)],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )

    print(f"Batch Size: {batch_size}")
    print(f"Query Length (chars): {len(query_text)}")
    print(f"Doc Length (chars): {len(doc_text)}")
    print(f"Tokenized document shape: {encoded['input_ids'].shape}")
    print(f"Tokenized attention_mask shape: {encoded['attention_mask'].shape}")
    doc_length = encoded['input_ids'].shape[1]

    
    results = []

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)

    if use_precomputed_docs:
        doc_hidden_states = torch.randn(
            batch_size,
            doc_length,
            model_dim,
            device=device
        )
        colbert_doc_hidden_states = torch.randn(
            batch_size,
            doc_length,
            colbert_dim,
            device=device
        )
        print(f"[info] Using precomputed document hidden states with shape {doc_hidden_states.shape} for MidFusion and ColBERT.")

    else:
        print("[info] computing document hidden states along the way.")
        doc_hidden_states = None
        colbert_doc_hidden_states = None

    print(f"\nStarting benchmarks on device: {device} ({device_name})\n")

    # --- Benchmark MidFusion ---
    # Using MiniLM L6 as per profile_midFusion.py example
    
    
    print(f"Benchmarking MidFusion ({midfusion_model_name}, {midfusion_kwargs})...")
    n_upper = midfusion_kwargs['drop_layer'] - midfusion_kwargs['merge_layer']
    
    if 'ettin' in midfusion_model_name:
        cls = EttinMidFusionCrossEncoder
        label = "MidFusion (Ettin)"
    else:
        cls = MiniLMMidFusionCrossEncoder
        label = "MidFusion (MiniLM-L12)"

    midfusion_stats = benchmark_model(
        cls,
        label,
        batch,
        device,
        midfusion_model_name,
        warmup_steps,
        num_runs,
        print_model_summary=False,
        doc_hidden_states=doc_hidden_states,
        **midfusion_kwargs
    )
    midfusion_stats["Model"] = f"MICE-l{midfusion_kwargs['merge_layer']}+{n_upper} {midfusion_model_name}"
    results.append(midfusion_stats)
    

    # --- Benchmark ColBERT ---
    print(f"Benchmarking ColBERT ({colbert_model_name})...")
    
    colbert_stats = benchmark_model(
        PyLateColBERT,
        "ColBERT (Small)",
        batch,
        device,
        colbert_model_name,
        warmup_steps,
        num_runs,
        print_model_summary=False,
        doc_hidden_states=colbert_doc_hidden_states,
        # max_length=512 # ColBERT handles lengths internally/via defaults usually
        document_length=max_length,
    )
    colbert_stats["Model"] = f"ColBERT {colbert_model_name}"
    results.append(colbert_stats)
    
    # --- Benchmark Vanilla Cross-Encoder ---
    print(f"Benchmarking Vanilla Cross-Encoder ({vanilla_model_name})...")
    
    if doc_hidden_states is not None:
        print("[warn] Vanilla Cross-Encoder does not use precomputed document states; ignoring them.")

    vanilla_stats = benchmark_model(
        HFCrossScorer,
        "Vanilla Cross-Encoder",
        batch,
        device,
        vanilla_model_name,
        warmup_steps,
        num_runs,
        print_model_summary=True,
        max_length=512,
    )
    vanilla_stats["Model"] = f"Vanilla {vanilla_model_name}"
    results.append(vanilla_stats)
    

    # --- Display Results ---
    if results:
        print("\n\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        
        table_data = []
        for res in results:
            table_data.append([
                res["Model"],
                f"{res['num_params']/1e6:.1f}M",
                f"{res['mean_time']*1000:.2f} ms",
                f"{res['std_time']*1000:.2f} ms",
                f"{res['max_memory_mb']:.2f} MB",
                f"{res['memory_increase_mb']:.2f} MB",
                f"{res['theoretical_docs_per_second']:.2f} docs/sec"
            ])
        
        headers = ["Model", "Params", "Mean Latency", "Std Dev", "Max Memory", "Memory Increase", "Docs/sec"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nPrecomputed Docs: {use_precomputed_docs}, Runs: {num_runs}, Batch Size: {batch_size}, Doc Length: {doc_length}, Device: {device_name}")
        print("="*50 + "\n")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Configuration ---
    batch_size = 64
    warmup_steps = 5
    num_runs = 10
    max_length = 512
    model_dim = 384  # MiniLM-L12-H384-uncased dimension
    colbert_dim = 128  # ColBERT input dimension
    
    base_name = "microsoft/MiniLM-L12-H384-uncased"
    base_name = "jhu-clsp/ettin-encoder-32m"
    base_name = "jhu-clsp/ettin-encoder-17m"
    base_name = "jhu-clsp/ettin-encoder-68m"
    base_name = "jhu-clsp/ettin-encoder-150m"

    colbert_model_name = base_name
    midfusion_model_name = base_name
    vanilla_model_name = base_name
    
    use_precomputed_docs = True
    use_precomputed_docs = False

    # override 
    # colbert_model_name = "answerdotai/answerai-colbert-small-v1"
    # colbert_model_name = "colbert-ir/colbertv2.0"
    # midfusion_model_name = "nreimers/MiniLM-L6-H384-uncased"
    
    midfusion_kwargs = dict(
        merge_layer=11,
        drop_layer=14,
        random_top_layers=False 
    )
    
    main()
