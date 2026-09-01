"""
Tests for MICEQueryDocTokenizer: functionality with custom query/doc max lengths,
input modes, and comparative analysis against default STCrossEncoder pair tokenization.
"""

import pytest
import torch
from transformers import AutoTokenizer

from MICE.modeling.mice import MICEQueryDocTokenizer, QueryDocInput, MICETokenizedTexts
from xpmir.letor.records import PointwiseItems
from xpmir.text.encoders import TokenizedTexts


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_mice_tokenizer_default_length_fallback():
    """Test that unspecified max_query_length / max_doc_length fall back to max_length (or HF default 512)."""
    tokenizer = MICEQueryDocTokenizer.C(model_id=MODEL_ID).instance()
    tokenizer.initialize()

    assert tokenizer.max_length == 512
    assert tokenizer.max_query_length == 512
    assert tokenizer.max_doc_length == 512


def test_mice_tokenizer_custom_lengths():
    """Test MICEQueryDocTokenizer with explicit max_query_length and max_doc_length."""
    max_q = 12
    max_d = 24
    tokenizer = MICEQueryDocTokenizer.C(
        model_id=MODEL_ID,
        max_query_length=max_q,
        max_doc_length=max_d,
    ).instance()
    tokenizer.initialize()

    assert tokenizer.max_query_length == max_q
    assert tokenizer.max_doc_length == max_d

    # Create long inputs that exceed both max_q and max_d
    query = "what is the capital city of France and why is it famous worldwide?"
    doc = (
        "Paris is the capital and most populous city of France, with an estimated population "
        "of 2,165,423 residents in an area of 105 square kilometers."
    )

    records = PointwiseItems.from_texts(topics=[query], documents=[doc])
    res = tokenizer.tokenize(records)

    assert isinstance(res, MICETokenizedTexts)
    assert res.tokenized_q.ids.shape == (1, max_q)
    assert res.tokenized_docs.ids.shape == (1, max_d)
    assert res.tokenized_q.lens == [max_q]
    assert res.tokenized_docs.lens == [max_d]


def test_mice_tokenizer_input_natures():
    """Test QueryDocInput.PAIRS, QUERY, and DOCUMENT modes."""
    tokenizer = MICEQueryDocTokenizer.C(
        model_id=MODEL_ID,
        max_query_length=10,
        max_doc_length=20,
    ).instance()
    tokenizer.initialize()

    query = "Search query string"
    doc = "Document text content for retrieval"

    # PAIRS mode
    records = PointwiseItems.from_texts(topics=[query], documents=[doc])
    pairs_res = tokenizer.tokenize(records, input_nature=QueryDocInput.PAIRS)
    assert isinstance(pairs_res, MICETokenizedTexts)
    assert pairs_res.tokenized_q is not None
    assert pairs_res.tokenized_docs is not None

    # QUERY mode
    q_records = [{"text_item": type("T", (), {"text": query})()}]
    q_res = tokenizer.tokenize(q_records, input_nature=QueryDocInput.QUERY)
    assert isinstance(q_res, TokenizedTexts)
    assert torch.equal(q_res.ids, pairs_res.tokenized_q.ids)

    # DOCUMENT mode
    doc_records = [{"text_item": type("T", (), {"text": doc})()}]
    d_res = tokenizer.tokenize(doc_records, input_nature=QueryDocInput.DOCUMENT)
    assert isinstance(d_res, TokenizedTexts)
    assert torch.equal(d_res.ids, pairs_res.tokenized_docs.ids)


def test_mice_vs_st_cross_encoder_tokenization():
    """Compare MICEQueryDocTokenizer separate tokenization against standard CrossEncoder joint tokenization."""
    max_q = 10
    max_d = 20
    joint_max = 30

    mice_tok = MICEQueryDocTokenizer.C(
        model_id=MODEL_ID,
        max_query_length=max_q,
        max_doc_length=max_d,
    ).instance()
    mice_tok.initialize()

    hf_tok = AutoTokenizer.from_pretrained(MODEL_ID)

    query = "Which planet is known as the Red Planet?"
    doc = "Mars, known for its reddish appearance, is often referred to as the Red Planet in our solar system."

    records = PointwiseItems.from_texts(topics=[query], documents=[doc])
    mice_res = mice_tok.tokenize(records)

    # Standard ST / CrossEncoder joint tokenization
    st_enc = hf_tok(
        query,
        doc,
        max_length=joint_max,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # 1. Check structural differences
    # ST CrossEncoder produces 1 joint tensor: [CLS] Query [SEP] Doc [SEP]
    st_ids = st_enc["input_ids"][0]
    st_decoded = hf_tok.decode(st_ids)

    # MICE produces 2 independent tensors: [CLS] Query [SEP] and [CLS] Doc [SEP]
    q_ids = mice_res.tokenized_q.ids[0]
    d_ids = mice_res.tokenized_docs.ids[0]
    q_decoded = hf_tok.decode(q_ids)
    d_decoded = hf_tok.decode(d_ids)

    # Assertions
    assert len(q_ids) == max_q
    assert len(d_ids) == max_d
    assert len(st_ids) == joint_max

    # MICE Query has [CLS] at 0 and [SEP] at the end of query sequence
    assert q_ids[0] == hf_tok.cls_token_id
    assert q_ids[-1] == hf_tok.sep_token_id

    # MICE Doc also has its own [CLS] at 0 and [SEP] at the end
    assert d_ids[0] == hf_tok.cls_token_id
    assert d_ids[-1] == hf_tok.sep_token_id

    # ST Joint sequence has 2 [SEP] tokens (one after query, one after doc)
    sep_indices = (st_ids == hf_tok.sep_token_id).nonzero(as_tuple=True)[0]
    assert len(sep_indices) == 2

    print("\n--- Tokenization Comparison Summary ---")
    print(f"ST Joint Encoded ({joint_max} tokens):\n  {st_decoded}")
    print(f"MICE Query Encoded ({max_q} tokens):\n  {q_decoded}")
    print(f"MICE Doc Encoded ({max_d} tokens):\n  {d_decoded}")
