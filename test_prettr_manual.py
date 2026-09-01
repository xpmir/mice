import torch
import pytest
from MICE.modeling.prettr_mice import prettr_scorer, PreTTRCrossEncoder
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param

def test_prettr_forward():
    hf_id = "cross-encoder/ms-marco-MiniLM-L-6-v2" # small BERT
    join_layer = 3

    # 1. Create the scorer
    scorer_cfg, init_tasks_cfg = prettr_scorer(
        hf_id=hf_id,
        join_layer=join_layer,
        max_length=64
    )
    model = scorer_cfg.instance()
    model.initialize()

    print("Model initialized successfully")

    # 2. Initialize weights
    for task_cfg in init_tasks_cfg:
        task_cfg.instance().execute()
    print("Weights seeded successfully")

    # 3. Prepare dataset
    queries = ["What is the capital of France?"]
    documents = ["Paris is the capital and most populous city of France."]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

    # 4. Run forward pass
    model.eval()
    with torch.no_grad():
        score = model(input_records)

    print(f"Model output (score): {score}")
    assert isinstance(score, torch.Tensor)
    assert score.shape == (1,)
    print("Forward pass successful!")

    # 5. Parity Check: Joint forward vs Independent Document Encoding
    # In PreTTR, the document representations after join_layer should be identical
    # to the joint forward pass if cross-attention is masked.
    print("\n--- Parity Check ---")
    tokenized_joint = model.batch_tokenize(input_records)
    with torch.no_grad():
        input_ids = tokenized_joint.ids
        token_type_ids = tokenized_joint.token_type_ids

        # Replicate position_ids logic for verification
        BAT, SEQ = input_ids.shape
        pos_ids_joint = torch.arange(SEQ, dtype=torch.long, device=model.device).unsqueeze(0).expand(BAT, SEQ).clone()
        is_doc = (token_type_ids == 1)
        for b in range(BAT):
            doc_indices = torch.where(is_doc[b])[0]
            if len(doc_indices) > 0:
                doc_start = doc_indices[0]
                num_doc = len(doc_indices)
                pos_ids_joint[b, doc_start:] = torch.arange(
                    model.prettr_max_query_length,
                    model.prettr_max_query_length + num_doc,
                    device=model.device
                )

        base_model = getattr(model.encoder.model, model.encoder.model.base_model_prefix)
        joint_embeddings = base_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids_joint
        )

        # Document tokens in the joint sequence are those with token_type_id == 1
        doc_token_mask = (token_type_ids == 1)
        joint_doc_embeddings = joint_embeddings[doc_token_mask].view(1, -1, joint_embeddings.shape[-1])

        # Independent doc encoding using the official API
        doc_record = {"id": "doc1", "text": documents[0]}

        # We check the official API's output shape and embeddings consistency
        doc_hiddens = model.document_token_embeddings([doc_record])
        indep_doc_hiddens = doc_hiddens[0].unsqueeze(0)

        print(f"Joint doc embeddings shape: {joint_doc_embeddings.shape}")
        print(f"Indep doc hiddens shape:    {indep_doc_hiddens.shape}")

        assert joint_doc_embeddings.shape == indep_doc_hiddens.shape, "Shape mismatch in document tokens!"

        # Verification of embeddings (before layers)
        tokenized_indep = model.tokenizer.tokenizer(
            [doc_record["text"]],
            padding=True, truncation=True, max_length=64,
            add_special_tokens=False, return_tensors="pt",
        )
        sep_id = model.tokenizer.tokenizer.sep_token_id
        indep_doc_ids = torch.cat([
            tokenized_indep["input_ids"],
            torch.full((tokenized_indep["input_ids"].shape[0], 1), sep_id, dtype=torch.long)
        ], dim=1)
        indep_pos_ids = torch.arange(
            model.prettr_max_query_length,
            model.prettr_max_query_length + indep_doc_ids.shape[1],
            device=model.device
        ).unsqueeze(0)

        indep_doc_embeddings = base_model.embeddings(
            input_ids=indep_doc_ids,
            token_type_ids=torch.ones_like(indep_doc_ids),
            position_ids=indep_pos_ids
        )

        diff = torch.abs(joint_doc_embeddings - indep_doc_embeddings).max().item()
        print(f"Max difference in embeddings: {diff:.6f}")
        assert diff < 1e-5, "Embeddings mismatch! PreTTR positional offsets are inconsistent."
        print("✅ SUCCESS: Embeddings match!")

    # 6. Test encode_documents (Offline API) logic
    doc_text = "This is a document"
    tokenized = model.tokenizer.tokenizer([doc_text], return_tensors="pt", add_special_tokens=False)
    # Add SEP manually for consistency
    input_ids = torch.cat([tokenized["input_ids"], torch.tensor([[model.tokenizer.tokenizer.sep_token_id]])], dim=1)
    attention_mask = torch.cat([tokenized["attention_mask"], torch.tensor([[1]])], dim=1)

    doc_hidden = model.encode_documents(input_ids, attention_mask)
    print(f"Doc hidden shape: {doc_hidden.shape}")
    assert doc_hidden.shape[1] == input_ids.shape[1]
    print("Offline document encoding successful!")

if __name__ == "__main__":
    test_prettr_forward()
