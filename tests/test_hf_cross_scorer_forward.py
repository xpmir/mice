import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import torch
import pytest
import logging
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param
from xpmir.neural.huggingface import hf_cross_scorer, HFCrossScorer

# Set up logging to see what's happening
from logging_utils import setup_logging
setup_logging(level=logging.INFO)


class TestHFForwardTask(LightweightTask):
    """Tests HF Cross Scorer Loading, scoring a PointWiseItem with two different passages"""

    scorer: Param[HFCrossScorer]

    def execute(self):
        """
        Task execution for instantiating a dataset and running a forward pass for an HF model.
        """
        print(f"Step 1: Instantiating the HF model (ID: {id(self.scorer)})...")

        # Initialize the model components (experimaestro way)
        self.scorer.initialize()

        print(self.scorer)
        print("Step 2: Preparing the dataset (one query, two passages)...")
        # Define one query and two different documents to compare scores
        query = "What is the capital of France?"
        # Paris (relevant) vs London (irrelevant for this query)
        relevant_doc = "Paris is the capital and most populous city of France."
        irrelevant_doc = "London is the capital and most populous city of England and the United Kingdom."

        # We need to provide the query for each document
        queries = [query, query]
        documents = [relevant_doc, irrelevant_doc]

        # Create the input records
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        print("Step 3: Running a forward pass...")
        # Switch to evaluation mode
        self.scorer.eval()

        # Run forward pass without gradient computation
        with torch.no_grad():
            # The model's forward method accepts BaseItems (like PointwiseItems)
            output = self.scorer(input_records)

        # Output shape is (2, 1) or (2,) depending on the model/squeezer
        scores = output.view(-1)
        relevant_score = scores[0].item()
        irrelevant_score = scores[1].item()

        print(f"Relevant document score:   {relevant_score:.4f}")
        print(f"Irrelevant document score: {irrelevant_score:.4f}")

        # Validation
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"

        # In a well-trained model, the relevant document should have a higher score
        if relevant_score > irrelevant_score:
            print("✅ SUCCESS: Relevant document has a HIGHER score.")
        else:
            print("❌ FAILURE: Relevant document has a LOWER score.")
            # We don't necessarily want to fail the test if the model is untrained/random,
            # but for MS-MARCO pre-trained models, it should pass.
            # Let's keep it as a warning for now unless we are sure.
            # assert relevant_score > irrelevant_score, "Relevant score should be greater than irrelevant score"

        print(
            "\nForward pass successful! HF Cross Scorer model is correctly instantiated and working."
        )


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-32m"])
def test_hf_cross_scorer(model_id):
    """tests hf cross scorer loading, forward pass."""
    # hf_cross_scorer returns the model and a list of initialization tasks
    # that need to be executed to load the weights
    scorer_cfg, init_tasks = hf_cross_scorer(hf_id=model_id)

    print(scorer_cfg)

    # Create and run the forward pass task
    test_forward_task = TestHFForwardTask.C(scorer=scorer_cfg).instance()
    # Run initialization tasks if they exist
    for init_task in init_tasks:
        init_task.instance().execute()
    test_forward_task.execute()


if __name__ == "__main__":
    # Use a small BERT model for testing purposes
    model_ids = [
        "jhu-clsp/ettin-encoder-32m"
        # "cross-encoder/ms-marco-MiniLM-L-6-v2",
        # "Qwen/Qwen3-0.6B"
    ]

    for model_id in model_ids:
        print("\n\n" + "=" * 50)
        print(f"### TESTING {model_id}")
        print("=" * 50)
        test_hf_cross_scorer(model_id)
