"""
Tests for ModernBERT MICE V2 PLAID Indexation & Retrieval.
Run with: `uv run pytest tests/test_mice_v2_plaid.py -v`
"""

import tempfile
import torch
import pytest
from datamaestro_ir.data.base import IDTextRecord
from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer, MiceV2DocumentEncoder
from MICE.modeling.mice import MICETokenizedTexts, QueryDocInput
from xpmir.letor.records import PointwiseItems
from xpmir.text.encoders import TokenizerOptions

import logging
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_document_encoder(model_id):
    """Test get_document_encoder and document token embedding generation."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    docs = [
        IDTextRecord(id="doc1", text="Paris is the capital of France."),
        IDTextRecord(id="doc2", text="Berlin is the capital of Germany."),
    ]

    with torch.no_grad():
        token_embs = doc_encoder.document_token_embeddings(docs)

    assert len(token_embs) == 2, f"Expected 2 document embeddings, got {len(token_embs)}"
    assert isinstance(token_embs[0], torch.Tensor), "Embedding should be a Tensor"
    assert token_embs[0].ndim == 2, f"Expected shape [seq_len, dim], got {token_embs[0].shape}"
    logging.info(f"Document 1 token embeddings shape: {token_embs[0].shape}")


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_doc_hidden_states_forward(model_id):
    """Test forward pass with decompressed document hidden states (PLAID pipeline)."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    queries = ["What is the capital of France?", "Where is Berlin?"]
    documents = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
    ]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    # 1. Standard forward pass
    with torch.no_grad():
        standard_scores = scorer(input_records)

    # 2. Forward pass with pre-encoded doc_hidden_states using encode_documents
    doc_records = [IDTextRecord(id=f"doc_{i}", text=text) for i, text in enumerate(documents)]
    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    with torch.no_grad():
        doc_hidden_states = doc_encoder.document_token_embeddings(doc_records)
        tokenized_q = scorer.batch_tokenize(input_records, input_nature=QueryDocInput.QUERY)
        plaid_scores = scorer(
            tokenized=MICETokenizedTexts(tokenized_q=tokenized_q, tokenized_docs=None),
            doc_hidden_states=doc_hidden_states,
        )

    logging.info(f"Standard scores: {standard_scores}")
    logging.info(f"PLAID doc_hidden scores: {plaid_scores}")

    assert torch.allclose(standard_scores, plaid_scores, atol=1e-4), \
        f"Mismatch between standard forward ({standard_scores}) and PLAID doc_hidden forward ({plaid_scores})"


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_normalize_doc_embeds(model_id):
    """Test document state L2 normalization when normalize_doc_embeds=True."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
        normalize_doc_embeds=True,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    docs = [
        IDTextRecord(id="doc1", text="Paris is the capital of France."),
        IDTextRecord(id="doc2", text="Berlin is the capital of Germany."),
    ]
    doc_encoder = scorer_cfg.get_document_encoder().instance()
    doc_encoder.model = scorer
    doc_encoder.eval()

    with torch.no_grad():
        token_embs = doc_encoder.document_token_embeddings(docs)

    for i, emb in enumerate(token_embs):
        norms = torch.norm(emb, p=2, dim=-1)
        logging.info(f"Doc {i} token norms (normalize_doc_embeds=True): {norms}")
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Expected unit L2 norm (1.0), got {norms}"

    # Verify parity between standard forward pass and pre-encoded doc_hidden_states
    queries = ["What is the capital of France?", "Where is Berlin?"]
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    with torch.no_grad():
        standard_scores = scorer(input_records)
        tokenized_q = scorer.batch_tokenize(input_records, input_nature=QueryDocInput.QUERY)
        plaid_scores = scorer(
            tokenized=MICETokenizedTexts(tokenized_q=tokenized_q, tokenized_docs=None),
            doc_hidden_states=token_embs,
        )

    assert torch.allclose(standard_scores, plaid_scores, atol=1e-4), \
        f"Score mismatch with normalize_doc_embeds=True: {standard_scores} vs {plaid_scores}"


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_plaid_doc_length_override(model_id):
    """Test that max_doc_length on get_document_encoder forces truncating stored vectors."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
        max_doc_length=512,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    long_doc = [
        IDTextRecord(
            id="long_doc",
            text=" ".join(["word"] * 100),  # 100 words -> ~100 tokens
        )
    ]

    # 1. Default encoder: should produce > 50 tokens
    default_encoder = scorer_cfg.get_document_encoder().instance()
    default_encoder.model = scorer
    default_encoder.eval()
    with torch.no_grad():
        default_embs = default_encoder.document_token_embeddings(long_doc)
    assert default_embs[0].shape[0] > 50, f"Expected > 50 tokens, got {default_embs[0].shape[0]}"

    # 2. PLAID encoder with max_doc_length=15: should force at most 15 tokens
    plaid_doc_encoder = scorer_cfg.get_document_encoder(max_doc_length=15).instance()
    plaid_doc_encoder.model = scorer
    plaid_doc_encoder.eval()
    with torch.no_grad():
        truncated_embs = plaid_doc_encoder.document_token_embeddings(long_doc)

    assert truncated_embs[0].shape[0] <= 15, (
        f"Expected <= 15 vectors stored for PLAID, got {truncated_embs[0].shape[0]}"
    )
    # 3. Verify base model tokenizer was not mutated in-place
    assert scorer.tokenizer.max_doc_length == 512, (
        f"Base model tokenizer was mutated! Expected 512, got {scorer.tokenizer.max_doc_length}"
    )


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_v2_plaid_query_length_override(model_id):
    """Test that max_query_length on get_query_encoder truncates query vectors for PLAID search."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
        max_query_length=64,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    scorer.eval()

    long_query = [
        IDTextRecord(
            id="long_q",
            text=" ".join(["query_token"] * 30),
        )
    ]

    # 1. Default query token embeddings
    with torch.no_grad():
        default_q_embs = scorer.query_token_embeddings(long_query)
    assert default_q_embs.shape[1] > 10, f"Expected > 10 query tokens, got {default_q_embs.shape[1]}"

    # 2. PLAID query encoder with max_query_length=8
    plaid_q_encoder = scorer_cfg.get_query_encoder(max_query_length=8).instance()
    plaid_q_encoder.model = scorer
    plaid_q_encoder.initialize()
    plaid_q_encoder.eval()

    with torch.no_grad():
        truncated_q_embs = plaid_q_encoder.query_token_embeddings(long_query)

    assert truncated_q_embs.shape[1] <= 8, (
        f"Expected <= 8 query vectors, got {truncated_q_embs.shape[1]}"
    )
    # 3. Base model tokenizer should be intact
    assert scorer.tokenizer.max_query_length == 64, (
        f"Base model tokenizer was mutated! Expected 64, got {scorer.tokenizer.max_query_length}"
    )


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
def test_mice_query_doc_tokenizer_options(model_id):
    """Test that MICEQueryDocTokenizer respects options.max_length for queries and docs."""
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
        max_doc_length=128,
        max_query_length=64,
    )
    scorer = scorer_cfg.instance()
    for task in init_tasks:
        task.instance().execute()
    scorer.initialize()
    tokenizer = scorer.tokenizer

    records = [IDTextRecord(id="r1", text=" ".join(["test"] * 50))]

    # Default doc tokenization
    t_doc_default = tokenizer.tokenize(records, input_nature=QueryDocInput.DOCUMENT)
    assert t_doc_default.ids.shape[1] > 20

    # Overridden doc tokenization with options
    t_doc_truncated = tokenizer.tokenize(
        records, options=TokenizerOptions(max_length=10), input_nature=QueryDocInput.DOCUMENT
    )
    assert t_doc_truncated.ids.shape[1] <= 10

    # Overridden query tokenization with options
    t_q_truncated = tokenizer.tokenize(
        records, options=TokenizerOptions(max_length=8), input_nature=QueryDocInput.QUERY
    )
    assert t_q_truncated.ids.shape[1] <= 8
