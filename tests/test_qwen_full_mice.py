import torch
from MICE.modeling.mice import mice_scorer
import logging

from logging_utils import setup_logging
setup_logging(level=logging.INFO)


def test_qwen_full_mice():
    # Use a small Qwen model for testing
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"Testing full MICE model with {model_id}...")

    # 1. Create the scorer and its initialization task
    scorer, init_tasks = mice_scorer(
        hf_id=model_id, n_contextualization_layers=4, max_length=128
    )

    # 2. Run the initialization task to load weights
    print("Running initialization task...")
    for task in init_tasks:
        task.instance().execute()

    # 3. Prepare dummy inputs
    # xpmir BaseItems requires some structure, let's mock it or use dummy tensors
    # Actually, let's test the forward directly with tensors first to ensure the plumbing works

    batch_size = 2
    q_len = 16
    d_len = 32

    tokenized_q = type(
        "Tokenized",
        (),
        {
            "ids": torch.randint(0, 1000, (batch_size, q_len)),
            "mask": torch.ones(batch_size, q_len),
        },
    )
    tokenized_docs = type(
        "Tokenized",
        (),
        {
            "ids": torch.randint(0, 1000, (batch_size, d_len)),
            "mask": torch.ones(batch_size, d_len),
        },
    )

    tokenized = type(
        "MICETokenized",
        (),
        {"tokenized_q": tokenized_q, "tokenized_docs": tokenized_docs},
    )

    print("Performing forward pass...")
    model = scorer.instance()
    model.initialize()
    model.to("cpu")  # Ensure it's on CPU for the test
    output = model(inputs=None, tokenized=tokenized)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size,)
    print("Full MICE Qwen model forward pass successful!")


if __name__ == "__main__":
    test_qwen_full_mice()
