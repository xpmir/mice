import gc
import logging
import statistics
import time
import torch
from typing import Optional, List, Tuple, Union, Sequence, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from xpmir.learning import ModuleInitMode, ModuleInitOptions

try:
    from datamaestro_text.data.ir import TextItem
except ImportError:
    class TextItem:
        pass

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


def benchmark_model(
    model_cls,
    name,
    batch,
    device,
    model_name,
    warmup_steps,
    num_runs,
    print_model_summary=False,
    doc_hidden_states: Optional[torch.Tensor] = None,
    verify_weights_fn: Optional[Callable] = None,
    **model_kwargs,
):
    logger = logging.getLogger(__name__)
    logger.info(f"--- Benchmarking {name} ---")
    try:
        cuda_available = torch.cuda.is_available()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        #clear CUDA memory before loading model
        gc.collect()
        torch.cuda.empty_cache()

        if hasattr(model_cls, "from_kwargs"):
            model = model_cls.from_kwargs(
                hf_id=model_name, 
                **model_kwargs,
            )
            if hasattr(model, "initialize"):
                model.initialize(ModuleInitOptions(mode=ModuleInitMode.DEFAULT))
        else:
            # Fallback for classes using experimaestro .C() pattern (like PyLateColBERT)
            model = model_cls.C(
                hf_id=model_name,
                **model_kwargs,
            ).instance()

        
        # Verify weights before moving to device (or after, just need to be careful with cpu/cuda)
        if verify_weights_fn:
            verify_weights_fn(model, model_name, name)
        

        if not cuda_available:
            print("[warn] CUDA not available, running on CPU may be slow.")
            # We continue even if CPU, but warn.
            
        model.to(device)
        model.eval()

        if print_model_summary:
            logger.info(f"Model Summary for {name}:")
            print(model.__repr__())

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"{name} Parameters: {num_params:,}")
        
        # Try to detect attention implementation
        attn_impl = "N/A"
        try:
            if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
                attn_impl = model.config._attn_implementation
            elif hasattr(model, "bottom_layers") and hasattr(model.bottom_layers, "config") and hasattr(model.bottom_layers.config, "_attn_implementation"):
                 attn_impl = model.bottom_layers.config._attn_implementation
            # Special case for some HF models that store it in a different place or wrappers
        except Exception:
            pass
        logger.info(f"{name} Attention Implementation: {attn_impl}")

        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Record memory after loading but before inference
            mem_after_loading = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()


        kwargs = {"doc_hidden_states": doc_hidden_states} if doc_hidden_states is not None else {}
        with torch.no_grad():
            if warmup_steps > 0:
                logger.info(f"{name} warmup iterations: {warmup_steps}")
                for _ in range(warmup_steps):
                    _ = model(batch, **kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            # Reset again after warmup to measure pure inference peak if desired, 
            # or keep it to include warmup's peak. 
            # Usually, peak is stable after warmup.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            timings = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(batch, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start_time)

        if timings:
            mean = statistics.mean(timings)
            stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
            logger.info(
                f"{name} Timings (s): mean={mean:.4f}, std={stdev:.4f}, min={min(timings):.4f}, max={max(timings):.4f}"
            )
        else:
            mean, stdev = 0.0, 0.0

        max_mem_mb = 0.0
        mem_increase_mb = 0.0
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated()
            max_mem_mb = max_mem / 1024 / 1024
            batch_size = len(batch.topics)
            theoretical_docs_per_second = batch_size / mean if mean > 0 else 0.0
            # Increase relative to memory after loading
            mem_increase_mb = (max_mem - mem_after_loading) / 1024 / 1024
            logger.info(f"{name} Max Memory: {max_mem_mb:.2f} MB")
            logger.info(f"{name} Memory Increase (Inference): {mem_increase_mb:.2f} MB")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "mean_time": mean,
            "std_time": stdev,
            "min_time": min(timings) if timings else 0.0,
            "max_time": max(timings) if timings else 0.0,
            "max_memory_mb": max_mem_mb,
            "memory_increase_mb": mem_increase_mb,
            "theoretical_docs_per_second": theoretical_docs_per_second,
            "num_params": num_params,
            "attn_impl": attn_impl,
        }

    except Exception as exc:
        logger.error(f"Failed to benchmark {name}: {exc}")
        raise
