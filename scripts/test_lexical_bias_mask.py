"""
Diagnostic script to verify the lexical bias mask and attention mask generation.
Usage:
    uv run python scripts/test_lexical_bias_mask.py
"""

import torch
import logging
from pathlib import Path
from MICE.modeling.mice import mice_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_lexical_bias_mask")

def main():
    logger.info("Initializing ModernBERT MICE model with lexical_bias=True...")
    scorer_cfg, init_tasks = mice_scorer(
        hf_id="jhu-clsp/ettin-encoder-17m",
        n_contextualization_layers=2,
        lexical_bias=True
    )

    # Initialize weights
    for task in init_tasks:
        task.instance().execute()

    model = scorer_cfg.instance()
    model.initialize()
    model.eval()

    # Enable a distinct non-zero lexical bias value for testing
    with torch.no_grad():
        model.top_layers[0].crossattention.lexical_bias_weight.fill_(2.5)

    # Prepare sample inputs
    query = "What is the capital of France?"
    document = "Paris is the capital and most populous city of France."
    logger.info(f"Query: '{query}'")
    logger.info(f"Doc:   '{document}'")

    from xpmir.letor.records import PointwiseItems
    input_records = PointwiseItems.from_texts(topics=[query], documents=[document])

    # Tokenize manually to inspect tokens
    tokenized = model.batch_tokenize(input_records)
    query_ids = tokenized.tokenized_q.ids
    doc_ids = tokenized.tokenized_docs.ids

    # Decode tokens
    tokenizer = model.tokenizer.tokenizer
    q_tokens = [tokenizer.decode([tid]) for tid in query_ids[0]]
    d_tokens = [tokenizer.decode([tid]) for tid in doc_ids[0]]

    # 1. Print Tokens
    logger.info("\n--- Query Tokens ---")
    logger.info(", ".join([f"{i}:'{t}'" for i, t in enumerate(q_tokens)]))
    logger.info("\n--- Doc Tokens ---")
    logger.info(", ".join([f"{i}:'{t}'" for i, t in enumerate(d_tokens)]))

    # 2. Compute All Masks
    # Raw token match (unfiltered)
    lexical_match_mask_raw = (query_ids.unsqueeze(2) == doc_ids.unsqueeze(1)).unsqueeze(1) # [B, 1, Tq, Td]
    raw_bool_mask = lexical_match_mask_raw[0, 0] # [Tq, Td]

    # Filtered match (ignoring special tokens)
    lexical_match_mask = model.get_lexical_match_mask(query_ids, doc_ids)
    filtered_bool_mask = lexical_match_mask[0, 0] # [Tq, Td]

    # compute_mask (for extra_attn_bias with fallback)
    from MICE.modeling.mice import compute_mask
    exact_match_mask = compute_mask(query_ids, doc_ids)
    compute_mask_val = exact_match_mask[0, 0] # [Tq, Td]
    compute_mask_bool = (exact_match_mask == 0.0)[0, 0] # [Tq, Td]

    # Print raw token match grid
    logger.info("\n--- RAW TOKEN EQUALITY MASK GRID ---")
    header_raw = f"{'Query Token':<15} | " + " ".join([f"{t[:4]:<4}" for t in d_tokens])
    logger.info(header_raw)
    logger.info("-" * len(header_raw))
    for q_idx, q_tok in enumerate(q_tokens):
        row_vals = []
        for d_idx in range(len(d_tokens)):
            is_match = raw_bool_mask[q_idx, d_idx].item()
            row_vals.append("[X] " if is_match else "[ ] ")
        logger.info(f"{q_tok[:15]:<15} | " + "".join(row_vals))

    # Print filtered lexical_match_mask grid
    logger.info("\n--- FILTERED LEXICAL_MATCH_MASK GRID (ignoring special tokens) ---")
    header_filt = f"{'Query Token':<15} | " + " ".join([f"{t[:4]:<4}" for t in d_tokens])
    logger.info(header_filt)
    logger.info("-" * len(header_filt))
    for q_idx, q_tok in enumerate(q_tokens):
        row_vals = []
        for d_idx in range(len(d_tokens)):
            is_match = filtered_bool_mask[q_idx, d_idx].item()
            row_vals.append("[X] " if is_match else "[ ] ")
        logger.info(f"{q_tok[:15]:<15} | " + "".join(row_vals))

    # Print compute_mask grid (where it is 0.0)
    logger.info("\n--- COMPUTE_MASK GRID (with fallback) ---")
    header_comp = f"{'Query Token':<15} | " + " ".join([f"{t[:4]:<4}" for t in d_tokens])
    logger.info(header_comp)
    logger.info("-" * len(header_comp))
    for q_idx, q_tok in enumerate(q_tokens):
        row_vals = []
        for d_idx in range(len(d_tokens)):
            is_match = compute_mask_bool[q_idx, d_idx].item()
            raw_val = compute_mask_val[q_idx, d_idx].item()
            row_vals.append(f"[X]({raw_val: .0f}) " if is_match else f"[ ]({raw_val: .0f}) ")
        logger.info(f"{q_tok[:15]:<15} | " + "".join(row_vals))

    # 3. Verify resulting attention mask and bias application
    cross_mask = model.get_cross_attention_mask(tokenized.tokenized_q.mask, tokenized.tokenized_docs.mask, torch.float32)
    bias = model.top_layers[0].crossattention.lexical_bias_weight.view(1, -1, 1, 1) * lexical_match_mask.to(torch.float32)
    biased_mask = cross_mask + bias

    # Print the values in the biased mask for all matched positions in filtered mask
    logger.info("\n--- Biased Mask Values at Filtered Match Positions ---")
    matches = torch.nonzero(filtered_bool_mask)
    for match in matches:
        q_idx, d_idx = match[0].item(), match[1].item()
        base_val = cross_mask[0, 0, q_idx, d_idx].item()
        biased_val = biased_mask[0, 0, q_idx, d_idx].item()
        logger.info(
            f"Match: '{q_tokens[q_idx]}' == '{d_tokens[d_idx]}' "
            f"(q={q_idx}, d={d_idx}) | Base Mask: {base_val:.1f} -> Biased Mask: {biased_val:.1f}"
        )

    # Let's also print the biased mask values for the first layer's CLS token to ensure no bias was added
    logger.info("\n--- Biased Mask Values for Special Tokens (e.g. CLS at q=0, d=0) ---")
    base_cls_val = cross_mask[0, 0, 0, 0].item()
    biased_cls_val = biased_mask[0, 0, 0, 0].item()
    logger.info(f"CLS at (0, 0) | Base Mask: {base_cls_val:.1f} -> Biased Mask: {biased_cls_val:.1f}")

    logger.info("\nDiagnostic script completed successfully!")

if __name__ == "__main__":
    main()
