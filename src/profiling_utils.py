import gc
import logging
import statistics
import time
import torch
from pathlib import Path
from typing import Optional, Sequence, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from xpmir.letor.records import PointwiseItems
from xpmir.rankers.scorer import AbstractModuleScorer
from xpm_torch.huggingface import TorchHFHub

logger = logging.getLogger(__name__)
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
        documents = [make_item(document) for _ in range(batch_size - 1)]
        # first elem is twice bigger document
        documents.insert(0, make_item(document * 2))

        return cls(topics=topics, documents=documents)


def get_attn_implementation(model):
    attn_impl = "N/A"
    try:
        if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
            attn_impl = model.config._attn_implementation
        elif (
            hasattr(model, "bottom_layers")
            and hasattr(model.bottom_layers, "config")
            and hasattr(model.bottom_layers.config, "_attn_implementation")
        ):
            attn_impl = model.bottom_layers.config._attn_implementation
        # Special case for some HF models that store it in a different place or wrappers
    except Exception:
        pass
    return attn_impl


def benchmark_model(
    batch,
    tokenized=None,
    warmup_steps: int = 5,
    num_runs: int = 10,
    name: str = None,
    model_name_or_path: str = None,
    model: AbstractModuleScorer = None,
    model_cls=None,
    print_model_summary=False,
    doc_hidden_states: Optional[torch.Tensor] = None,
    verify_weights_fn: Optional[Callable] = None,
    model_kwargs=None,
):
    logger.info(f"--- Benchmarking {name} ---")
    try:
        cuda_available = torch.cuda.is_available()
        # start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        # clear CUDA memory before loading model
        gc.collect()
        torch.cuda.empty_cache()

        # 1. Model initialization and loading
        if model is not None:
            logger.info(f"Using provided model instance for {name}")
            assert isinstance(model, AbstractModuleScorer), (
                "Provided model must be an instance of AbstractModuleScorer"
            )
        elif model_name_or_path is not None:
            if not name:
                if Path(model_name_or_path).is_dir():
                    name = Path(model_name_or_path).name
                else:
                    name = model_name_or_path

            loader_cfg = TorchHFHub.pretrained_loader(
                model_name_or_path, as_instance=False
            )
            loader = loader_cfg.instance()
            loader.execute()
            model = loader.model

        elif model_cls is not None:
            logger.info(
                f"Instantiating model from class {model_cls.__name__} for {name}"
            )
            # deprecated - use TorchHFHub.from_pretrained
            if hasattr(model_cls, "from_kwargs"):
                model = model_cls.from_kwargs(
                    hf_id=model_name_or_path,
                    **model_kwargs,
                )
                if hasattr(model, "initialize"):
                    model.initialize()
            else:
                # Fallback for classes using experimaestro .C() pattern (like PyLateColBERT)
                model = model_cls.C(
                    hf_id=model_name_or_path,
                    **model_kwargs,
                ).instance()
        else:
            raise ValueError(
                "Must provide either model instance, model_name_or_path, or model_cls"
            )

        # Verify weights before moving to device (or after, just need to be careful with cpu/cuda)
        if verify_weights_fn:
            verify_weights_fn(model, model_name_or_path, name)

        if cuda_available:
            model.cuda()
        else:
            logger.warning("[warn] CUDA not available, running on CPU may be slow.")
            # We continue even if CPU, but warn.

        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        attn_impl = get_attn_implementation(model)

        if print_model_summary:
            logger.info(
                f"Model Summary for {name}:\n"
                f" - Parameters: {num_params:,}\n"
                f" - {name} Attention Implementation: {attn_impl}"
            )

            # Try to detect attention implementation
            logger.info(model.__repr__())

        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Record memory after loading but before inference
            mem_after_loading = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        kwargs = {}
        if doc_hidden_states is not None:
            kwargs["doc_hidden_states"] = doc_hidden_states
        if tokenized is not None:
            kwargs["tokenized"] = tokenized

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
            logger.debug(
                f"{name} Timings (s): mean={mean:.4f}, std={stdev:.4f}, min={min(timings):.4f}, max={max(timings):.4f}"
            )
        else:
            mean, stdev = 0.0, 0.0

        batch_size = len(batch.topics)
        theoretical_docs_per_second = batch_size / mean if mean > 0 else 0.0

        max_mem_mb = 0.0
        mem_increase_mb = 0.0
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated()
            max_mem_mb = max_mem / 1024 / 1024
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


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch = DummyBatch.build(batch_size=4, query="What is AI?", document="AI stands for Artificial Intelligence.")
    queries = ["What is the capital of France?"]
    documents = ["Paris is the capital and most populous city of France."]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    results = benchmark_model(
        model_name_or_path="/Users/victor/code/experiments/JZ/baseline_small_CE/20260428_185727/results/models/cross-encoder-MiniLM-L12",
        batch=input_records,
        device=device,
        num_runs=20,
        print_model_summary=True,
    )
    print(results)
