import torch
import logging
from experimaestro import LightweightTask, Param
from MICE.modeling.mice import mice_scorer, MiceCrossEncoder
from xpmir.letor.records import PointwiseItems


class TestMiceForwardTask(LightweightTask):
    scorer: Param[MiceCrossEncoder]

    def execute(self):
        self.scorer.initialize()

        # Verify global_cls exists
        assert hasattr(self.scorer, "global_cls"), (
            "global_cls should be present in the model"
        )
        assert isinstance(self.scorer.global_cls, torch.nn.Parameter), (
            "global_cls should be a torch.nn.Parameter"
        )
        print("✓ global_cls exists and is a Parameter")

        # Prepare input
        queries = ["What is the capital of France?"]
        documents = ["Paris is the capital of France."]
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        # Forward pass
        self.scorer.eval()
        with torch.no_grad():
            output = self.scorer(input_records)

        print(f"Model output: {output}")
        assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"
        print("✓ Forward pass successful with global_cls_token=True")


def test_global_cls_token_all():
    from logging_utils import setup_logging
setup_logging(level=logging.INFO)

    model_ids = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "jhu-clsp/ettin-encoder-32m",
    ]
    merge_layer = 3

    for model_id in model_ids:
        print(f"\n\n### TESTING {model_id} ###")
        # Instantiate with global_cls_token=True
        scorer_cfg, init_tasks = mice_scorer(
            hf_id=model_id, merge_layer=merge_layer, global_cls_token=True
        )

        # Create and run the forward pass task
        test_forward_task = TestMiceForwardTask.C(scorer=scorer_cfg).instance()

        # Run initialization tasks if they exist
        for init_task in init_tasks:
            init_task.instance().execute()

        test_forward_task.execute()


if __name__ == "__main__":
    test_global_cls_token_all()
