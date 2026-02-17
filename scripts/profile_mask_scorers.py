#!/usr/bin/env python
"""Profiling helpers for Franken MiniLM scorers.

This script benchmarks the forward pass cost of the baseline MiniLM scorer
versus the FrankenMiniLM scorer that applies attention masking.  It focuses on
highlighting the overhead introduced by the custom masking pipeline
(`torch.isin`, per-layer mask construction, host<->device copies).
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)
from types import SimpleNamespace

from datamaestro_text.data.ir import TextItem

from models.franken import AttentionPatch, FrankenCrossScorer
from models.mask_scorer import HFMaskedScorer, HFMaskedMiniLMCrossScorer
from xpmir.text.huggingface import HFStringTokenizer
from xpmir.neural.huggingface import HFCrossScorer
from xpmir.learning import ModuleInitMode, ModuleInitOptions
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def build_attention_patches(num_layers: int) -> List[AttentionPatch]:
    """Return a representative set of patches matching current experiments."""
    return [
        AttentionPatch.C(
            mask_attention_from=["query", "sep_1", "document", "sep_2"],
            mask_attention_to=["cls"],
            start_layer=0,
            end_layer=num_layers,
        ),
        AttentionPatch.C(
            mask_attention_from=["document"],
            mask_attention_to=["sep_1"],
            start_layer=0,
            end_layer=num_layers,
        ),
        AttentionPatch.C(
            mask_attention_from=["query"],
            mask_attention_to=["sep_2"],
            start_layer=0,
            end_layer=num_layers,
        ),
        AttentionPatch.C(
            mask_attention_from=["sep_1"],
            mask_attention_to=["cls", "query", "document", "sep_2"],
            start_layer=0,
            end_layer=num_layers,
        ),
        AttentionPatch.C(
            mask_attention_from=["sep_2"],
            mask_attention_to=["cls", "query", "sep_1", "document"],
            start_layer=0,
            end_layer=num_layers,
        ),
    ]


@dataclass
class DummyBatch:
    topics: Sequence[dict]
    documents: Sequence[dict]


    @property
    def queries(self):
        """Deprecated: use topics"""
        return self.topics
    
    @classmethod
    def build(cls, batch_size: int, query: str, document: str) -> "DummyBatch":
        def make_item(text: str) -> dict:
            return {TextItem: SimpleNamespace(text=text)}

        topics = [make_item(query) for _ in range(batch_size)]
        documents = [make_item(document) for _ in range(batch_size-1)]
        #first elem is twice bigger document
        documents.insert(0, make_item(document * 2))

        return cls(topics=topics, documents=documents)


_reference_state_cache: dict[str, dict[str, dict[str, torch.Tensor]]] = {}


def _load_reference_weights(hf_id: str, label: str) -> dict[str, dict[str, torch.Tensor]] | None:
    try:
        print(f"[info] {label}: loading reference model weights for verification from {hf_id}...")
        reference = AutoModelForSequenceClassification.from_pretrained(
            hf_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        print(f"[warn] {label}: failed to load reference model for verification ({exc}).")
        return None

    state = reference.state_dict()
    base_prefix = getattr(reference, "base_model_prefix", "")
    prefix = f"{base_prefix}." if base_prefix else ""

    backbone_state: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        tensor_cpu = tensor.detach().cpu()
        if prefix and key.startswith(prefix):
            backbone_state[key[len(prefix):]] = tensor_cpu
        elif not prefix and not key.startswith("classifier"):
            backbone_state[key] = tensor_cpu

    classifier_state: dict[str, torch.Tensor] = {}
    classifier_module = getattr(reference, "classifier", None)
    if classifier_module is not None:
        classifier_state = {
            k: v.detach().cpu()
            for k, v in classifier_module.state_dict().items()
        }

    _reference_state_cache[hf_id] = {
        "backbone": backbone_state,
        "classifier": classifier_state,
    }
    print(
        f"[info] {label}: reference model loaded successfully ({reference.__class__.__name__})."
    )
    del reference
    return _reference_state_cache[hf_id]


def verify_loaded_weights(module: torch.nn.Module, hf_id: str, label: str):
    scorer = getattr(module, "scorer", None)
    if scorer is None or not hasattr(scorer, "model"):
        print(f"[warn] {label}: scorer model not available for weight verification.")
        return

    target_model = scorer.model
    target_state = {k: v.detach().cpu() for k, v in target_model.state_dict().items()}

    cache_entry = _reference_state_cache.get(hf_id)
    if cache_entry is None:
        cache_entry = _load_reference_weights(hf_id, label)
        if cache_entry is None:
            return

    reference_backbone = cache_entry["backbone"]
    reference_keys = set(reference_backbone.keys())
    target_keys = set(target_state.keys())

    missing = sorted(reference_keys - target_keys)
    unexpected = sorted(target_keys - reference_keys)
    if missing:
        examples = ", ".join(missing[:3])
        print(
            f"[warn] {label}: missing {len(missing)} backbone keys (e.g. {examples})."
        )
    if unexpected:
        examples = ", ".join(unexpected[:3])
        print(
            f"[warn] {label}: unexpected {len(unexpected)} backbone keys (e.g. {examples})."
        )

    shared_keys = reference_keys & target_keys
    if not shared_keys:
        print(f"[warn] {label}: no shared parameters found during weight verification.")
        return

    tolerance = 1e-5
    max_diff = 0.0
    worst_key = None
    mismatched_samples: List[tuple[str, float]] = []

    for key in sorted(shared_keys):
        ref_tensor = reference_backbone[key].to(dtype=torch.float32)
        tgt_tensor = target_state[key].to(dtype=torch.float32)
        if ref_tensor.shape != tgt_tensor.shape:
            print(
                f"[warn] {label}: shape mismatch for parameter '{key}' ({ref_tensor.shape} vs {tgt_tensor.shape})."
            )
            return
        diff = (ref_tensor - tgt_tensor).abs().max().item()
        if diff > tolerance and len(mismatched_samples) < 5:
            mismatched_samples.append((key, diff))
        if diff > max_diff:
            max_diff = diff
            worst_key = key

    for key, diff in mismatched_samples:
        print(f"[warn] {label}: backbone parameter '{key}' |Δ|={diff:.3e}")

    if missing or unexpected:
        print(
            f"[warn] {label}: parameter mismatch (missing={len(missing)}, unexpected={len(unexpected)})."
        )
    elif max_diff > tolerance and worst_key is not None:
        print(
            f"[warn] {label}: backbone parameters differ from reference (max |Δ|={max_diff:.3e} at '{worst_key}')."
        )
    else:
        print(
            f"[info] {label}: backbone parameters match reference checkpoint (max |Δ|={max_diff:.3e})."
        )

    classifier_module = None
    if hasattr(module, "classifier"):
        classifier_module = module.classifier
    elif hasattr(scorer, "classifier"):
        classifier_module = scorer.classifier

    reference_classifier = cache_entry.get("classifier", {})
    if classifier_module is None or not reference_classifier:
        return

    target_classifier_state = {
        k: v.detach().cpu() for k, v in classifier_module.state_dict().items()
    }
    ref_classifier_keys = set(reference_classifier.keys())
    tgt_classifier_keys = set(target_classifier_state.keys())

    missing_cls = sorted(ref_classifier_keys - tgt_classifier_keys)
    unexpected_cls = sorted(tgt_classifier_keys - ref_classifier_keys)
    if missing_cls:
        examples = ", ".join(missing_cls[:3])
        print(
            f"[warn] {label}: missing {len(missing_cls)} classifier keys (e.g. {examples})."
        )
    if unexpected_cls:
        examples = ", ".join(unexpected_cls[:3])
        print(
            f"[warn] {label}: unexpected {len(unexpected_cls)} classifier keys (e.g. {examples})."
        )

    shared_cls = ref_classifier_keys & tgt_classifier_keys
    if not shared_cls:
        print(f"[warn] {label}: no shared classifier parameters found during verification.")
        return

    cls_max_diff = 0.0
    cls_worst_key = None
    for key in sorted(shared_cls):
        ref_tensor = reference_classifier[key].to(dtype=torch.float32)
        tgt_tensor = target_classifier_state[key].to(dtype=torch.float32)
        if ref_tensor.shape != tgt_tensor.shape:
            print(
                f"[warn] {label}: classifier shape mismatch for '{key}' ({ref_tensor.shape} vs {tgt_tensor.shape})."
            )
            return
        diff = (ref_tensor - tgt_tensor).abs().max().item()
        if diff > cls_max_diff:
            cls_max_diff = diff
            cls_worst_key = key

    if missing_cls or unexpected_cls:
        print(
            f"[warn] {label}: classifier parameter mismatch (missing={len(missing_cls)}, unexpected={len(unexpected_cls)})."
        )
    elif cls_max_diff > tolerance and cls_worst_key is not None:
        print(
            f"[warn] {label}: classifier parameters differ from reference (max |Δ|={cls_max_diff:.3e} at '{cls_worst_key}')."
        )
    else:
        print(
            f"[info] {label}: classifier parameters match reference checkpoint (max |Δ|={cls_max_diff:.3e})."
        )


# ---------------------------------------------------------------------------
# Profiling utilities
# ---------------------------------------------------------------------------


def instantiate_scorer(hf_id: str, attention_patches=None):
    """Instantiate a FrankenCrossScorer model from the given class and HF model ID."""
    if "minilm" in hf_id.lower():
        scorer_cls = HFMaskedMiniLMCrossScorer
    else:
        raise NotImplementedError(f"Unsupported HF model ID for scorer instantiation: {hf_id}")

    scorer_cfg = FrankenCrossScorer.C(
                scorer=scorer_cls.C(hf_id=hf_id),
                attention_patches=attention_patches,
            )
    scorer = scorer_cfg.instance()
    scorer.__initialize__(ModuleInitOptions(mode=ModuleInitMode.DEFAULT))
    return scorer


def time_forward(model, batch, repeats: int, device: torch.device) -> List[float]:
    timings: List[float] = []
    with torch.inference_mode():
        # Warm-up once to trigger lazy initialization / compilation
        _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
    return timings


def profile_forward(model, batch, steps: int, device: torch.device, trace_dir: str | None, label: str):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    profile_kwargs = {"activities": activities, "record_shapes": True}
    if trace_dir:
        trace_path = Path(trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        profile_kwargs["on_trace_ready"] = tensorboard_trace_handler(str(trace_path))
        profile_kwargs["with_stack"] = True
        profile_kwargs["with_flops"] = True

    profile_kwargs.setdefault("with_stack", False)
    profile_kwargs.setdefault("with_flops", False)

    with profile(**profile_kwargs) as prof:
        with torch.inference_mode():
            for _ in range(steps):
                with record_function("forward"):
                    _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    return prof


def collect_event_stats(prof) -> dict:
    stats = {}
    for event in prof.key_averages():
        stats[event.key] = {
            "cpu": event.self_cpu_time_total,
            "cuda": getattr(event, "self_cuda_time_total", 0.0),
            "count": event.count,
        }
    return stats


def maybe_compile_model(model: torch.nn.Module, name: str, enabled: bool):
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print(f"[warn] torch.compile not available; skipping compilation for {name}.")
        return model
    try:
        compiled = torch.compile(model)
    except Exception as exc:
        print(f"[warn] torch.compile failed for {name}: {exc}")
        return model
    print(f"[info] torch.compile succeeded for {name}.")
    return compiled


def print_comparison_table(baseline_stats: dict, franken_stats: dict, limit: int, device_type: str):
    rows = []
    keys = set(baseline_stats).union(franken_stats)
    for key in keys:
        base_cpu = baseline_stats.get(key, {}).get("cpu", 0.0)
        base_cuda = baseline_stats.get(key, {}).get("cuda", 0.0)
        base_count = baseline_stats.get(key, {}).get("count", 0)
        fr_cpu = franken_stats.get(key, {}).get("cpu", 0.0)
        fr_cuda = franken_stats.get(key, {}).get("cuda", 0.0)
        fr_count = franken_stats.get(key, {}).get("count", 0)

        total_base = base_cpu + base_cuda
        total_fr = fr_cpu + fr_cuda
        ratio = (total_fr / total_base) if total_base > 0 else (float("inf") if total_fr > 0 else 1.0)

        rows.append(
            (
                key,
                base_cpu / 1000.0,
                fr_cpu / 1000.0,
                base_cuda / 1000.0,
                fr_cuda / 1000.0,
                base_count,
                fr_count,
                ratio,
            )
        )

    rows.sort(key=lambda r: r[2] + r[4], reverse=True)

    header = "op".ljust(45)
    header += "baseline_cpu(ms)".rjust(16)
    header += "franken_cpu(ms)".rjust(16)
    if device_type == "cuda":
        header += "baseline_cuda(ms)".rjust(18)
        header += "franken_cuda(ms)".rjust(18)
    header += "count(b/fr)".rjust(16)
    header += "ratio".rjust(10)
    print(header)
    print("-" * len(header))

    for row in rows[:limit]:
        key, base_cpu, fr_cpu, base_cuda, fr_cuda, base_count, fr_count, ratio = row
        line = key.ljust(45)
        line += f"{base_cpu:15.3f}{fr_cpu:16.3f}"
        if device_type == "cuda":
            line += f"{base_cuda:18.3f}{fr_cuda:18.3f}"
        line += f"{base_count:7d}/{fr_count:<7d}"
        if ratio == float("inf"):
            line += "    inf"
        else:
            line += f"{ratio:10.2f}"
        print(line)


def profile_memory(model: torch.nn.Module, batch: DummyBatch, device: torch.device, label: str, train: bool = False):
    if device.type != "cuda":
        print(f"[{label}] Memory profiling skipped (requires CUDA).")
        return

    # Reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Function to run one step
    def run_step():
        if train:
            model.train()
            # Enable gradients
            with torch.set_grad_enabled(True):
                output = model(batch)
                # Assume output is scores or has a loss. If just scores, sum them to get a scalar for backward
                if isinstance(output, torch.Tensor):
                    loss = output.sum()
                elif hasattr(output, "loss"):
                    loss = output.loss
                else:
                    raise ValueError("Unknown output format for backward pass")
                
                loss.backward()
                model.zero_grad() # Clear gradients
        else:
            model.eval()
            with torch.inference_mode():
                _ = model(batch)

    # Warmup / Allocation
    try:
        run_step()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[{label}] OOM during memory profiling warmup!")
            return
        raise e

    # Measurement
    torch.cuda.reset_peak_memory_stats(device)
    run_step()
    
    max_mem = torch.cuda.max_memory_allocated(device)
    print(f"[{label}] Peak GPU Memory ({'Train' if train else 'Inference'}): {max_mem / 1024**3:.3f} GB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    config = AutoConfig.from_pretrained(args.hf_id)
    baseline =  HFCrossScorer.C(
                hf_id=args.hf_id,
                max_length=config.max_position_embeddings,
            ).instance()
    baseline.initialize(ModuleInitOptions(mode=ModuleInitMode.DEFAULT))
    baseline.to(device)
    # baseline = maybe_compile_model(baseline, "MiniLMCrossScorer", args.compile)

    print(f"loaded baseline model from {args.hf_id}:\n\n")
    franken = instantiate_scorer(
        args.hf_id,
        attention_patches=build_attention_patches(args.num_layers),
    )
    franken.to(device)

    print(f"loaded model from {args.hf_id}: {franken.scorer}")
    verify_loaded_weights(franken, args.hf_id, "franken")

    # franken = maybe_compile_model(franken, "FrankenMiniLMCrossScorer", args.compile)

    # Ensure document is long enough to hit max_length (usually 512 for BERT)
    # 512 tokens ~ 300-400 words. Repeating the default text 5 times should be safe.
    long_document = args.document * 5
    batch = DummyBatch.build(args.batch_size, args.query, long_document)
    
    print(f"\n=== Profiling Configuration ===")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mode: {'Training' if args.train else 'Inference'}")
    print(f"Device: {device}")
    print("===============================\n")

    # Memory Profiling
    print("=== Memory Usage ===")
    profile_memory(baseline, batch, device, "Baseline", train=args.train)
    profile_memory(franken, batch, device, "Franken", train=args.train)
    print("")

    # Timing benchmark (Inference only for now, unless we want to time training steps too)
    # If training mode is requested, we skip pure inference timing or adapt it?
    # For now, let's keep the original timing as inference latency check, 
    # but strictly speaking, if user wants training, we should probably time training steps.
    # Let's add a condition.
    
    if args.train:
        print("Skipping inference timing benchmark because --train mode is active.")
    else:
        baseline_times = time_forward(baseline, batch, args.repeats, device)
        franken_times = time_forward(franken, batch, args.repeats, device)

        baseline_ms = statistics.mean(baseline_times) * 1e3
        franken_ms = statistics.mean(franken_times) * 1e3

        print("=== Inference Timing (ms per forward) ===")
        print(f"Baseline MiniLMCrossScorer : {baseline_ms:.2f} ms")
        print(f"FrankenMiniLMCrossScorer  : {franken_ms:.2f} ms")
        print(f"Slowdown factor            : {franken_ms / baseline_ms:.2f}x")

    if args.profile and not args.train:
        print("\n=== torch.profiler summary (MiniLMCrossScorer) ===")
        baseline_prof = profile_forward(baseline, batch, args.profile_steps, device, args.trace_dir if args.trace_baseline else None, label="baseline")
        print(baseline_prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.summary_limit))

        print("\n=== torch.profiler summary (FrankenMiniLMCrossScorer) ===")
        franken_prof = profile_forward(franken, batch, args.profile_steps, device, args.trace_dir or None, label="franken")
        print(franken_prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.summary_limit))

        print("\n=== Self-time comparison (ms) ===")
        baseline_stats = collect_event_stats(baseline_prof)
        franken_stats = collect_event_stats(franken_prof)
        print_comparison_table(baseline_stats, franken_stats, args.summary_limit, device.type)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-id", default="microsoft/MiniLM-L12-H384-uncased", # cross-encoder/ms-marco-MiniLM-L12-v2  microsoft/MiniLM-L12-H384-uncased
                        help="Hugging Face model identifier to load")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=12,
                        help="Number of transformer layers to patch")
    parser.add_argument("--repeats", type=int, default=50,
                        help="Number of timed forward iterations")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler on both scorers and report comparison")
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace-dir", default="",
                        help="If set, export torch.profiler traces for TensorBoard at this path")
    parser.add_argument("--trace-baseline", action="store_true",
                        help="Also export baseline traces alongside Franken traces")
    parser.add_argument("--summary-limit", type=int, default=15,
                        help="Number of profiler ops to display in summaries")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution even if CUDA is available")
    parser.add_argument("--compile", action="store_true",
                        help="Attempt torch.compile on both scorers before timing/profiling")
    parser.add_argument("--query", default="what is artificial intelligence?",
                        help="Dummy query text for synthetic batch")
    parser.add_argument("--document", default="Artificial intelligence studies intelligent agents blah -, lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua . Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat . Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur . Excepteur sint occaecat cupidatat non proident , sunt in culpa qui officia deserunt mollit anim id est laborum .",
                        help="Dummy document text for synthetic batch")
    parser.add_argument("--train", action="store_true", help="Run in training mode (forward + backward) to measure training memory")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
