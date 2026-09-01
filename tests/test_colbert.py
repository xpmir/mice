"""
Tests for the ColBERT model using the MICE test framework.
"""

import torch
import pytest
from typing import List, Tuple
from experimaestro import LightweightTask, Param
from datamaestro_ir.data.base import IDTextRecord
from xpmir.letor.records import PointwiseItems
from xpmir.neural.colbert import ColBERTEncoder

def Colbert_scorer(
    hf_id: str,
    dim: int = 128,
    query_maxlen: int = 32,
    doc_maxlen: int = 180,
) -> Tuple[ColBERTEncoder, List[LightweightTask]]:
    from xpmir.text.huggingface.base import HFConfigID, HFModel, HFModelInitFromID
    from xpmir.text.huggingface.encoders import HFTokensEncoder
    from xpmir.text.huggingface.tokenizers import HFTokenizer, HFTokenizerAdapter
    from xpmir.text.adapters import TopicTextConverter
    from xpmir.text.encoders import TokenizedTextEncoder
    from xpmir.neural.colbert import ColBERTEncoder

    # Model and initialization
    config = HFConfigID.C(hf_id=hf_id)
    model = HFModel.C(config=config)
    init_tasks = [HFModelInitFromID.C(model=model)]

    # Tokenizer adapter for IDTextRecord
    tokenizer = HFTokenizerAdapter.C(
        tokenizer=HFTokenizer.C(model_id=hf_id),
        converter=TopicTextConverter.C()
    )

    # Encoder part
    hftokens_encoder = HFTokensEncoder.C(model=model)

    colbert = ColBERTEncoder.C(
        encoder= TokenizedTextEncoder.C(
            tokenizer=tokenizer,
            encoder=hftokens_encoder
        ),
        dim=dim,
        query_maxlen=query_maxlen,
        doc_maxlen=doc_maxlen,
    )

    return colbert, init_tasks

class TestColBERTForwardTask(LightweightTask):
    """Tests ColBERT Loading and scoring a PointWiseItem"""

    scorer: Param[ColBERTEncoder]

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
    ],
)
def test_colbert(model_id):
    """Tests ColBERT loading and forward pass."""

    # Initialize scorer configuration
    scorer_cfg, init_tasks = Colbert_scorer(
        hf_id=model_id,
        dim=128,
        query_maxlen=32,
        doc_maxlen=180
    )

    # Create and run the forward pass task
    test_forward_task = TestColBERTForwardTask.C(scorer=scorer_cfg).instance()

    # Run initialization tasks (weight loading)
    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()
