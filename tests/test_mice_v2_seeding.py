"""
Seeding tests for ModernBERT MICE V2 architecture.
Run with: `uv run pytest tests/test_mice_v2_seeding.py -v`
"""

import torch
import logging
from transformers import AutoModel
from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

logger = logging.getLogger(__name__)


def test_mice_v2_seeding():
    logger.info("### Testing ModernBERT MICE V2 Seeding ###")
    hf_id = "jhu-clsp/ettin-encoder-17m"
    n_contextualization_layers = 2

    # 1. Initialize MICE V2 model configuration
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=hf_id, n_contextualization_layers=n_contextualization_layers
    )

    init_task_instances = [t.instance() for t in init_tasks]
    for it in init_task_instances:
        it.execute()

    scorer = init_task_instances[0].scorer
    hf_model = scorer.hf_model

    # 2. Load backbone for comparison
    backbone = AutoModel.from_pretrained(hf_id)

    # 3. Compare Embeddings
    assert torch.allclose(
        hf_model.model.embeddings.tok_embeddings.weight,
        backbone.embeddings.tok_embeddings.weight,
    ), "Embeddings weights mismatch!"
    logger.info("✅ Embeddings match")

    # 4. Compare Bottom Layers (Layer 0)
    assert torch.allclose(
        hf_model.model.layers[0].attn.Wqkv.weight,
        backbone.layers[0].attn.Wqkv.weight,
    ), "Bottom layer 0 Wqkv mismatch!"
    logger.info("✅ Bottom layers match")

    # 5. Compare Top Layers and Cross-Attention Seeding (Layer n_contextualization_layers)
    top_0 = hf_model.top_layers[0]
    backbone_top = backbone.layers[n_contextualization_layers]

    # Self-attention part
    assert torch.allclose(
        top_0.self_attn.Wqkv.weight, backbone_top.attn.Wqkv.weight
    ), "Top layer self-attention Wqkv mismatch!"

    # Cross-attention part (Wqkv)
    assert torch.allclose(
        top_0.cross_attn.Wqkv.weight, backbone_top.attn.Wqkv.weight
    ), "Top layer cross-attention Wqkv mismatch!"

    logger.info("✅ Top layers and Cross-Attention seeding match successfully!")


if __name__ == "__main__":
    test_mice_v2_seeding()
