"""
Tests for ModernBERT MICE V2 Concat-Self-Attention Architecture (use_cross_attn=False).
Run with: `uv run pytest tests/test_mice_v2_concat.py -v`
"""

import tempfile
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer, ModernBertMiceScorer
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param

import logging

logging.basicConfig(level=logging.INFO)


class TestMiceV2ConcatForwardTask(LightweightTask):
    """Tests Mice V2 Concat Loading, scoring a PointWiseItem, and reload persistence"""

    scorer: Param[ModernBertMiceScorer]

    def execute(self):
        self.scorer.initialize()
        logging.info(f"Initialized MICE V2 Concat scorer: {self.scorer}")
        assert not self.scorer.hf_model.config.use_cross_attn, (
            "use_cross_attn should be False"
        )
        assert self.scorer.hf_model.top_layers[0].cross_attn is None, (
            "cross_attn should be None when use_cross_attn=False"
        )

        queries = ["What is the capital of France?", "Where is Paris located?"]
        documents = [
            "Paris is the capital and most populous city of France.",
            "Paris is located in northern central France on the Seine river.",
        ]
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        self.scorer.eval()
        with torch.no_grad():
            output = self.scorer(input_records)

        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
@pytest.mark.parametrize(
    "n_contextualization_layers",
    [2],
)
def test_mice_v2_concat(
    model_id,
    n_contextualization_layers,
):
    """Tests MICE V2 concat-self-attention forward pass, weight persistence, and standalone ST/HF rebuild."""
    logging.info(
        f"Testing MICE V2 Concat with model_id={model_id}, n_contextualization_layers={n_contextualization_layers}"
    )

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=n_contextualization_layers,
        use_cross_attn=False,
    )

    test_forward_task = TestMiceV2ConcatForwardTask.C(scorer=scorer_cfg).instance()

    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()

    scorer = test_forward_task.scorer

    # Test doc_hidden pass (offline doc caching)
    tokenized = scorer.batch_tokenize(
        PointwiseItems.from_texts(
            topics=["What is France?"],
            documents=["France is a country in Europe."],
        )
    )
    with torch.no_grad():
        doc_hidden, _ = scorer.encode_documents(
            tokenized.tokenized_docs.ids, tokenized.tokenized_docs.mask
        )
        # Pass doc_hidden as list of 2D tensors matching fast-plaid decompression
        doc_list = [doc_hidden[0]]
        out_cached = scorer(
            tokenized=tokenized,
            doc_hidden_states=doc_list,
        )
        assert out_cached.shape == (1,)

    # --- Persistence Verification ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_path = tmp_path / "model"
        st_save_path = tmp_path / "st_model"

        # 1. Save model and tokenizer
        scorer.save_model(save_path)

        assert (save_path / "config.json").exists(), (
            "config.json missing from save_path!"
        )
        assert (save_path / "model.safetensors").exists() or (
            save_path / "pytorch_model.bin"
        ).exists(), "Weights missing from save_path!"
        assert (save_path / "tokenizer_config.json").exists(), (
            "tokenizer_config.json missing from save_path!"
        )

        # 2. Verify standalone rebuild without base model using HF Auto Classes
        standalone_tok = AutoTokenizer.from_pretrained(save_path)
        assert standalone_tok is not None, "Failed to load tokenizer from save_path!"

        standalone_model = AutoModelForSequenceClassification.from_pretrained(save_path)
        assert standalone_model is not None, (
            "Failed to rebuild model from save_path without base model!"
        )
        assert not standalone_model.config.use_cross_attn, (
            "use_cross_attn should be False in rebuilt model config"
        )
        assert standalone_model.top_layers[0].cross_attn is None, (
            "cross_attn should be None in rebuilt model"
        )

        # 3. Reload ModernBertMiceScorer
        reloaded_scorer = ModernBertMiceScorer.C(
            hf_id=model_id,
            tokenizer=scorer_cfg.tokenizer,
            n_contextualization_layers=n_contextualization_layers,
            use_cross_attn=False,
        ).instance()
        reloaded_scorer.load_model(save_path)

        assert torch.allclose(
            scorer.hf_model.classifier.weight,
            reloaded_scorer.hf_model.classifier.weight,
        ), "Classifier weights mismatch after standard reload!"

        # 4. SentenceTransformers Export & Import Roundtrip
        scorer.export_to_st(str(st_save_path))

        st_cross_encoder = CrossEncoder(str(st_save_path))
        scores = st_cross_encoder.predict(
            [["What is the capital of France?", "Paris is the capital of France."]]
        )
        assert scores is not None
        assert len(scores) == 1


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_dim", [64, 768])
def test_concat_unpadded_equivalence_with_old(batch_size: int, hidden_dim: int):
    """Compares the new vectorized _concat_query_doc_unpadded against the reference
    _concat_query_doc_unpadded_old to guarantee exact numerical and dimensional equivalence."""
    from MICE.modeling.mice_v2.modernbert_hf import (
        _concat_query_doc_unpadded,
        _concat_query_doc_unpadded_old,
    )

    torch.manual_seed(42)
    q_lens = torch.randint(5, 30, (batch_size,))
    # Include potential edge case where doc has only 1 token (only CLS) or short docs
    d_lens = torch.randint(1, 100, (batch_size,))

    cu_q = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_q[1:] = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    cu_d = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_d[1:] = torch.cumsum(d_lens, dim=0, dtype=torch.int32)

    total_q = int(cu_q[-1].item())
    total_d = int(cu_d[-1].item())

    q_hidden = torch.randn(1, total_q, hidden_dim)
    d_hidden = torch.randn(1, total_d, hidden_dim)

    # Run old reference
    old_hidden, old_cu, old_max_len, old_pos = _concat_query_doc_unpadded_old(
        q_hidden, d_hidden, cu_q, cu_d
    )

    # Run new vectorized
    new_hidden, new_cu, new_max_len, new_pos = _concat_query_doc_unpadded(
        q_hidden, d_hidden, cu_q, cu_d
    )

    assert torch.allclose(new_hidden, old_hidden, atol=1e-6), (
        "Vectorized joint_hidden does not match old loop implementation!"
    )
    assert torch.equal(new_cu, old_cu), (
        f"cu_seqlens_joint mismatch: new={new_cu}, old={old_cu}"
    )
    assert new_max_len == old_max_len, (
        f"max_seqlen_joint mismatch: new={new_max_len}, old={old_max_len}"
    )
    assert torch.equal(new_pos, old_pos), (
        "pos_ids_joint mismatch between new and old implementation!"
    )
