"""
Tests for the MICE model.
run manually with `pytest tests/test_mice.py` or `uv run pytest tests/test_mice.py -v`
"""

import torch
import logging
import tempfile
import pytest
from pathlib import Path
from MICE.modeling.mice import mice_scorer, MiceCrossEncoder
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param
from xpm_torch.huggingface import TorchHFHub


class TestMiceForwardTask(LightweightTask):
    """Tests Mice Loading, scoring a PointWiseItem, and compare outputs after reloading for checkpointing"""

    scorer: Param[MiceCrossEncoder]

    def execute(self):
        """
        Task execution for instantiating a dataset and running a forward pass for a MICE model.
        """
        print(f"Step 1: Instantiating the MICE model (ID: {id(self.scorer)})...")

        # Initialize the model components (experimaestro way)
        self.scorer.initialize()

        print("Step 2: Preparing the dataset...")
        queries = ["What is the capital of France?"]
        documents = ["Paris is the capital and most populous city of France."]

        # Create the input records
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        print("Step 3: Running a forward pass...")
        self.scorer.eval()

        # Run forward pass without gradient computation
        with torch.no_grad():
            output = self.scorer(input_records)

        print(f"Model output (score): {output}")

        # Validation
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"

        print("\nForward pass successful!")


@pytest.mark.parametrize(
    "model_id",
    [
        "jhu-clsp/ettin-encoder-17m",  # ModernBERT
        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # BERT
        "google/electra-small-discriminator",  # Electra
        # "Qwen/Qwen2.5-0.5B-Instruct",           # Qwen
    ],
)
@pytest.mark.parametrize("cross_attn_first", [True])
@pytest.mark.parametrize("mask_cls_to_doc", [False])
@pytest.mark.parametrize("extra_attn_bias", [True])
@pytest.mark.parametrize(
    "n_contextualization_layers, n_docs_ctx_layers, bound_bottom_layers",
    [
        (2, None, True),
        (2, None, False),
        (2, 3, False),
    ],
)
def test_mice(
    model_id,
    n_contextualization_layers,
    n_docs_ctx_layers,
    cross_attn_first: bool,
    mask_cls_to_doc: bool,
    extra_attn_bias: bool,
    bound_bottom_layers,
):
    """Tests MICE loading, forward pass, and weight persistence across a grid of parameters."""

    # Initialize scorer configuration
    scorer_cfg, init_tasks = mice_scorer(
        hf_id=model_id,
        # n_interaction_layers=3, #automatic
        n_contextualization_layers=n_contextualization_layers,
        n_docs_ctx_layers=n_docs_ctx_layers,
        cross_attn_first=cross_attn_first,
        bound_bottom_layers=bound_bottom_layers,
        mask_cls_to_doc=mask_cls_to_doc,
        extra_attn_bias=extra_attn_bias,
    )

    # Create and run the forward pass task
    test_forward_task = TestMiceForwardTask.C(scorer=scorer_cfg).instance()

    # Run initialization tasks (weight loading)
    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()

    # Verify layer counts
    model = test_forward_task.scorer
    if bound_bottom_layers:
        assert len(model.bottom_layers) == n_contextualization_layers
    else:
        assert len(model.query_bottom_layers) == n_contextualization_layers
        assert len(model.document_bottom_layers) == (
            n_docs_ctx_layers or n_contextualization_layers
        )

    # --- Persistence Verification ---

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_path = tmp_path / "model"
        hf_save_path = tmp_path / "hf_model"

        # 1. Save model weights
        model.save_model(save_path)

        # 2. Reload model using loader_config
        loader_config = scorer_cfg.loader_config(save_path)
        loader = loader_config.instance()
        loader.execute()
        reloaded_model = loader.model

        # 3. Verify weights are identical
        assert torch.allclose(
            model.classifier.weight, reloaded_model.classifier.weight
        ), "Classifier weights mismatch after standard reload!"

        # 4. HF Export/Import Roundtrip
        hub = TorchHFHub(loader)
        hub.save_pretrained(hf_save_path)

        hf_loader = TorchHFHub.pretrained_loader(hf_save_path, as_instance=True)
        hf_loader.execute()
        hf_model = hf_loader.model

        assert torch.allclose(
            model.classifier.weight, hf_model.classifier.weight
        ), "Classifier weights mismatch after HF roundtrip!"


def test_lexical_bias():
    from MICE.modeling.mice import mice_scorer
    from xpmir.letor.records import PointwiseItems
    import tempfile
    from pathlib import Path

    # 1. Instantiate the MICE model with lexical_bias=True
    scorer_cfg, init_tasks = mice_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        lexical_bias=True
    )

    for task in init_tasks:
        task.instance().execute()

    model = scorer_cfg.instance()
    model.initialize()

    # Verify model has the parameter and is initialized to zero
    assert hasattr(model.top_layers[0].crossattention, "lexical_bias_weight"), "lexical_bias_weight not found on top layer cross-attention"
    bias_weight = model.top_layers[0].crossattention.lexical_bias_weight
    assert torch.allclose(bias_weight, torch.zeros_like(bias_weight)), "lexical_bias_weight not initialized to zero"

    # 2. Verify backward pass and optimizer update
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    query = "What is the capital of France?"
    document = "Paris is the capital of France."
    input_records = PointwiseItems.from_texts(topics=[query], documents=[document])

    # Forward pass
    scores = model(input_records)

    # Dummy loss to compute gradients
    loss = scores.sum()
    loss.backward()

    # Verify gradients are computed for the bias weight
    assert bias_weight.grad is not None, "Gradients not computed for lexical_bias_weight"
    assert not torch.allclose(bias_weight.grad, torch.zeros_like(bias_weight.grad)), "Gradients are all zero for lexical_bias_weight"

    # Step the optimizer
    prev_weight = bias_weight.clone()
    optimizer.step()

    # Verify parameter updated
    assert not torch.allclose(bias_weight, prev_weight), "lexical_bias_weight did not update"

    # 3. Persistence roundtrip check
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_path = tmp_path / "model"

        # Save model
        model.save_model(save_path)

        # Reload model
        loader_config = scorer_cfg.loader_config(save_path)
        loader = loader_config.instance()
        loader.execute()
        reloaded_model = loader.model

        # Verify reloaded weights are identical
        reloaded_weight = reloaded_model.top_layers[0].crossattention.lexical_bias_weight
        assert torch.allclose(bias_weight, reloaded_weight), "lexical_bias_weight mismatch after save/load"
