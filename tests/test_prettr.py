"""
Tests for the PreTTR model using the MICE test framework.
"""

import torch
import pytest
from MICE.modeling.prettr_mice import prettr_scorer, PreTTRCrossEncoder
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param

class TestPreTTRForwardTask(LightweightTask):
    """Tests PreTTR Loading and scoring a PointWiseItem"""

    scorer: Param[PreTTRCrossEncoder]

    def execute(self):
        # Initialize the model components
        self.scorer.initialize()

        queries = ["What is the capital of France?"]
        documents = ["Paris is the capital and most populous city of France."]

        # Create the input records
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        self.scorer.eval()

        # Run forward pass
        with torch.no_grad():
            output = self.scorer(input_records)

        print(f"Model output (score): {output}")

        # Validation
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"

@pytest.mark.parametrize(
    "model_id",
    [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "google/electra-small-discriminator",
    ],
)
@pytest.mark.parametrize("join_layer", [2, 6])
def test_prettr(model_id, join_layer):
    """Tests PreTTR loading and forward pass."""

    # Initialize scorer configuration
    scorer_cfg, init_tasks = prettr_scorer(
        hf_id=model_id,
        join_layer=join_layer,
        max_length=64
    )

    # Create and run the forward pass task
    test_forward_task = TestPreTTRForwardTask.C(scorer=scorer_cfg).instance()

    # Run initialization tasks (weight loading)
    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()
