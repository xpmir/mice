import torch
from typing import List, Optional
from transformers import AutoConfig
from functools import lru_cache

from experimaestro import Config, Constant, Param
from datamaestro_text.data.ir import TextItem
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer

from xpmir.text.encoders import (
    TokenizedTexts,
)

from models.mask_scorer import HFMaskedScorer
from xpm_torch.xpmModel import xpmTorchHubModule
import logging
from xpm_torch.utils.logging import easylog

logger = easylog()
logger.setLevel(logging.INFO)

INPUT_PART_TO_POSITION = {"cls": 0, "query": 1, "sep_1": 2, "document": 3, "sep_2": 4}
N = len(INPUT_PART_TO_POSITION.keys())

@torch.compile
def build_lookup_matrix_core(
        input_ids: torch.Tensor,
        sep_token_id: int,
        cls_token_id: int,
        pad_token_id: int,
        ) -> torch.Tensor:
    """Builds a N*N lookup matrix for segment token types based on CLS, SEP, and PAD tokens.
    Arguments:
        input_ids: A (batch x seq_len) tensor of input token IDs
        sep_token_id: The token ID for the SEP token
        cls_token_id: The token ID for the CLS token
        pad_token_id: The token ID for the PAD token
    Returns: 
        A (batch x seq_len x seq_len) lookup matrix tensor of type long
    """
    # 1. Add 1.0 for SEP or CLS tokens
    A = ((input_ids == sep_token_id) | (input_ids == cls_token_id)).float()
    
    # 2. Add -inf for PAD tokens
    A = torch.where((input_ids == pad_token_id), -torch.inf, A)

    # Use A.new_zeros() for better device/dtype handling instead of explicit device transfer
    A = torch.hstack([A.new_zeros(A.shape[0], 1), A])
    A = A[:, 1:] + A[:, :-1]
    A = A.cumsum(dim=1) - 1
    A = A.unsqueeze(dim=1) + (A * N).unsqueeze(dim=2) 
    
    A = torch.where(A == -torch.inf, float(N**2), A)
    return A.long()


@lru_cache(maxsize=32)
def n_layers_from_hf_id(hf_id: str) -> int:
    """
    Retrieves the number of transformer layers from a Hugging Face model ID's configuration.

    Args:
        hf_id: The Hugging Face model ID (e.g., 'bert-base-uncased', 'gpt2').

    Returns:
        The number of layers (int), or -1 if the attribute is not found.
    """
    try:
        config = AutoConfig.from_pretrained(hf_id)
    except Exception as e:
        logger.debug(f"Error loading configuration for '{hf_id}': {e}")
        raise e

    # Define a list of common configuration attributes for the number of layers
    layer_keys = [
        "num_hidden_layers", # Common for BERT, RoBERTa, etc.
        "n_layer",           # Common for GPT-2, LLaMA, T5, etc.
        "num_layers",        # Less common, but good to include
    ]

    # Check the configuration for the layer attributes
    for key in layer_keys:
        if hasattr(config, key):
            num_layers = getattr(config, key)
            logger.debug(f"Found layer attribute '{key}' in config.")
            return num_layers

    logger.debug(f"Warning: Could not find a common layer attribute in the configuration for '{hf_id}'.")
    logger.debug(f"The full config attributes are: {list(config.__dict__.keys())}")

    raise ValueError(f"Could not determine number of layers for model ID '{hf_id}'.")


class AttentionPatch(Config):
    """One Attention patch for FrankenMiniLM models.
    
    Will mask attention from tokens in `mask_attention_from` to tokens in `mask_attention_to`
    in layers from `start_layer` to `end_layer`
    """
    mask_attention_from: Param[Optional[List[str]]] = None

    mask_attention_to: Param[Optional[List[str]]] = None

    start_layer: Param[int] = 0

    end_layer: Param[int] = -1


class FrankenCrossScorer(xpmTorchHubModule, LearnableScorer):
    """Base class where we will control everything related to the masking of 
    attention parts within MiniLM."""
    
    scorer: Param[HFMaskedScorer]
    """an encoder for encoding the concatenated query-document tokens which
    doesn't contains the final linear layer"""

    attention_patches: Param[Optional[List[AttentionPatch]]] = None
    """List of attention patches to apply during the forward pass, can be None if no patching is needed (baseline standard behavior)"""

    version: Constant[int] = 2

    @property
    def tokenizer(self):
        return self.scorer.tokenizer
    
    def __initialize__(self, options):
        super().__initialize__(options)
        self.scorer.initialize(options)

        logging.info(f"Initialized FrankenCrossScorer with {len(self.attention_patches) if self.attention_patches else 0} attention patches.")
      

        #getting number of layers for model, works for modernBert and bert, 
        # condider using n_layers_from_hf_id if needed
        self.masks_per_layer = {layer_idx: [] for layer_idx in range(self.scorer.n_layers)}

        if self.attention_patches:
            for attention_patch in self.attention_patches:
                mask_from = attention_patch.mask_attention_from
                mask_to = attention_patch.mask_attention_to
                if attention_patch.end_layer == -1:
                    layer_range = range(attention_patch.start_layer, self.scorer.n_layers)
                else:
                    layer_range = range(attention_patch.start_layer, attention_patch.end_layer)
                for input_part_from in mask_from:
                    for input_part_to in mask_to:
                        code = INPUT_PART_TO_POSITION[input_part_from] * N + INPUT_PART_TO_POSITION[input_part_to]
                        for layer_idx in layer_range:
                            self.masks_per_layer[layer_idx].append(code)

        mask_lookup = torch.ones((self.scorer.n_layers, N * N + 1), dtype=torch.bool)
        #Always mask PAD code (N*N) to False
        mask_lookup[:, N * N] = False

        for layer_idx, codes in self.masks_per_layer.items():
            if codes:
                indices = torch.as_tensor(codes, dtype=torch.long)
                mask_lookup[layer_idx, indices] = False

        self.register_buffer("layer_mask_lookup", mask_lookup, persistent=False)


    def build_lookup_matrix(self, input_ids, device) -> torch.Tensor:
        """Builds a N*N lookup matrix for segment token types based on CLS, SEP, and PAD tokens.
        Arguments:
            input_ids: A (batch x seq_len) tensor of input token IDs
        Returns: 
            A (batch x seq_len x seq_len) lookup matrix tensor
        """
        return build_lookup_matrix_core(
            input_ids,
            self.tokenizer.sep_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.pad_token_id,
        ).to(device)

    def forward(
        self, 
        inputs: BaseRecords, 
        info: TrainerContext = None,
    ):  
        
        r = self.tokenizer(
            [
                (tr[TextItem].text, dr[TextItem].text)
                for tr, dr in zip(inputs.topics, inputs.documents)
            ], 
            max_length=self.scorer.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=True,
        )

        tokenized_inputs = TokenizedTexts(
            None,
            r["input_ids"],
            r["length"],
            r.get("attention_mask", None),
            r.get("token_type_ids", None),
        ).to(self.scorer.device)
        
        parsed_inputs = self.build_lookup_matrix(tokenized_inputs.ids, device=self.scorer.device)
        index_codes = parsed_inputs.view(-1)
        mask_shape = parsed_inputs.shape

        #Cache attention masks to avoid redundant computations and allocations on GPU
        # ex with seq_len=2K, batch size 64:  each mask is 1.9Gb ...
        # we thus keep only one copy per unique lookup row
        attention_masks = {}
        def get_layer_mask_lookup(lookup_row):
            key = lookup_row.cpu().numpy().tobytes()
            if key not in attention_masks:
                allowed_flat = lookup_row.gather(0, index_codes)
                attention_masks[key] = allowed_flat.view(mask_shape).to(parsed_inputs.dtype)
            return attention_masks[key]

        layer_attention_masks = []
        for layer_idx in range(self.scorer.n_layers):
            lookup_row = self.layer_mask_lookup[layer_idx]
            layer_attention_masks.append(
                get_layer_mask_lookup(lookup_row)
                )

        #delete to free memory
        del parsed_inputs
        del index_codes

        #A Scorer outputs tensor of shape (batch,) of scores
        return self.scorer(
            tokenized_inputs,
            layer_attention_masks=layer_attention_masks,
        )