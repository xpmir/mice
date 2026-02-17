import torch
import torch.nn as nn
from typing import Optional, List

from experimaestro import Param
from transformers import AutoModelForSequenceClassification
from xpmir.learning import Module
from xpmir.text.encoders import (
    TextsRepresentationOutput,
    TokenizedTexts,
)
from dataclasses import InitVar
from xpmir.learning.optim import ModuleInitOptions

from configuration import PoolingMethod
from models.mask_modeling import CustomMaskModel, CustomMaskBertModel, CustomMaskModernBertModel

try:
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

import logging
from xpm_torch.utils.logging import easylog

logger = easylog()
logger.setLevel(logging.INFO)

MAX_SEQ_LEN = 1024

class HFMaskedScorer(Module):
    """Custom class for masked scorers
    Based on Huggingface `AutoModelForSequenceClassification` architecture with a masked model backbone
    """

    hf_id: Param[str]
    """Model ID from huggingface"""

    model: InitVar[CustomMaskModel]
    """The HF model"""

    tokenizer: AutoTokenizer
    """the tokenizer attached to the model"""

    hf_config: AutoConfig
    """the HuggingFave model configuration"""

    @property
    def dimension(self):
        return self.hf_config.hidden_size

    @property
    def n_layers(self):
        return self.hf_config.num_hidden_layers
    
    @property
    def max_length(self):
        """Returns the maximum length that the model can process"""
        return min(self.hf_config.max_position_embeddings, MAX_SEQ_LEN)

    @property
    def device(self):
        return self.model.device


    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer

        Args:
            options: loader options
        """
        super().__initialize__(options)

        self.hf_config = AutoConfig.from_pretrained(self.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)      


    def forward(self, tokenized: TokenizedTexts, layer_attention_masks: Optional[List[torch.FloatTensor]]) -> TextsRepresentationOutput:
        raise NotImplementedError("HFMaskedScorer is an abstract base class and does not implement forward()")


class HFMaskedMiniLMCrossScorer(HFMaskedScorer):
    """Custom class for masked MiniLM Cross Scorer
    Based on Huggingface `AutoModelForSequenceClassification` architecture with a masked MiniLM backbone
    """

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer

        Args:
            options: loader options
        """
        super().__initialize__(options)
        
        self.model = CustomMaskBertModel(self.hf_config)
        self.classifier = nn.Linear(self.hf_config.hidden_size, 1)
        self.dropout_layer = nn.Dropout(self.hf_config.hidden_dropout_prob)

        # Load pretrained weights in a SequenceClassification Model, separating backbone and classifier head
        reference_model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            self.hf_id,
            dtype=getattr(self.model, "dtype", None),
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        base_prefix = getattr(reference_model, "base_model_prefix", "")
        prefix = f"{base_prefix}." if base_prefix else ""
        backbone_state = {}
        for key, value in reference_model.state_dict().items():
            if prefix and key.startswith(prefix):
                backbone_state[key[len(prefix):]] = value
            elif not prefix and not key.startswith("classifier"):
                backbone_state[key] = value
        missing_backbone, unexpected_backbone = self.model.load_state_dict(backbone_state, strict=False)

        if missing_backbone:
            logging.warning("Missing backbone keys after HF load: %s", missing_backbone)
        if unexpected_backbone:
            logging.warning("Unexpected backbone keys after HF load: %s", unexpected_backbone)
        if not unexpected_backbone and not missing_backbone:
            logging.info("Successfully loaded HF model backbone weights into HFMaskedScorer backbone.")

        classifier_module = getattr(reference_model, "classifier", None)
        loading_dict = loading_info if isinstance(loading_info, dict) else {}
        missing_keys = loading_dict.get("missing_keys", [])
        mismatched_keys = loading_dict.get("mismatched_keys", [])
        classifier_missing = any(key.startswith("classifier.") for key in missing_keys)
        classifier_mismatched = any(
            (key if isinstance(key, str) else key[0]).startswith("classifier.")
            for key in mismatched_keys
        )

        if classifier_module is not None and not classifier_missing and not classifier_mismatched:
            classifier_state = classifier_module.state_dict()
            missing_cls, unexpected_cls = self.classifier.load_state_dict(classifier_state, strict=False)
            if missing_cls:
                logging.warning("Missing classifier keys after HF load: %s", missing_cls)
            if unexpected_cls:
                logging.warning("Unexpected classifier keys after HF load: %s", unexpected_cls)
            if not unexpected_cls and not missing_cls:
                logging.info("Successfully loaded HF model classifier weights into HFMaskedScorer classifier.")
        else:
            logging.info("Skipping HF classifier head weights (missing or mismatched in checkpoint).")

        del reference_model
        logging.info(f"Initialized HFMaskedScorer with model ID: {self.hf_id}, hidden size: {self.hf_config.hidden_size}, number of layers: {self.hf_config.num_hidden_layers}, dropout: {self.hf_config.hidden_dropout_prob}")
        

    def forward(self, tokenized: TokenizedTexts, layer_attention_masks: Optional[List[torch.FloatTensor]]) -> TextsRepresentationOutput:
        tokenized = tokenized.to(self.device)
        y = self.model(
            tokenized.ids,
            token_type_ids=tokenized.token_type_ids,
            layer_attention_masks=layer_attention_masks
        )

        pooled = y.pooler_output if hasattr(y, "pooler_output") and y.pooler_output is not None else y.last_hidden_state[:, 0]
        pooled = self.dropout_layer(pooled)
        logits = self.classifier(pooled).squeeze(1)

        return logits

class HFMaskedEttinCrossScorer(HFMaskedScorer):
    """Custom class for masked Ettin Cross Scorer
    Based on Huggingface `AutoModelForSequenceClassification` architecture with a masked Ettin backbone
    """

    pooling_method: Param[Optional[str]] = None
    """Pooling method to use for the Ettin based scorer: cls or mean.
    Leave it to None for models coming from the Hub, as it will be inferred from the model config."""

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer

        Args:
            options: loader options
        """
        super().__initialize__(options)
        self.model = CustomMaskModernBertModel(self.hf_config)

        if self.pooling_method is None:
            if self.hf_config.classifier_pooling == PoolingMethod.CLS.value:
                self.pooling_function = lambda x, masks: x[:, 0]
            elif self.hf_config.classifier_pooling == PoolingMethod.MEAN.value:
                # Rationale here, is that layer_attention_masks[-1][:,0,:] corresponds to the attention mask (dim=[BS,SEQ_LEN,SEQ_LEN]) of the last layer (-1), 
                # in particular over the information sent to the CLS token (0), from all tokens (:). 
                self.pooling_function = lambda x, masks: (x * masks[-1][:,0,:].unsqueeze(-1)).sum(dim=1) / masks[-1][:,0,:].sum(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported pooling method in model config: {self.hf_config.classifier_pooling}")
        else:
            if self.pooling_method == PoolingMethod.CLS.value:
                self.pooling_function = lambda x, masks: x[:, 0]
            elif self.pooling_method == PoolingMethod.MEAN.value:
                self.pooling_function = lambda x, masks: (x * masks[-1][:,0,:].unsqueeze(-1)).sum(dim=1) / masks[-1][:,0,:].sum(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported pooling method provided: {self.pooling_method}")
            
        # ModernBertPredictionHead
        self.head = ModernBertPredictionHead(
            self.hf_config,
        )

        self.classifier = nn.Linear(self.hf_config.hidden_size, 1)
        self.dropout_layer = nn.Dropout(self.hf_config.classifier_dropout)

        # Load pretrained weights, excluding classifier head
        reference_model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            self.hf_id,
            dtype=getattr(self.model, "dtype", None),
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        base_prefix = getattr(reference_model, "base_model_prefix", "")
        prefix = f"{base_prefix}." if base_prefix else ""
        backbone_state = {}
        for key, value in reference_model.state_dict().items():
            if prefix and key.startswith(prefix):
                backbone_state[key[len(prefix):]] = value
            elif prefix and not key.startswith("classifier"):
                backbone_state[key] = value
        missing_backbone_model, unexpected_backbone = self.model.load_state_dict(backbone_state, strict=False)

        unexpected_backbone_state = {k: v for k, v in backbone_state.items() if k in unexpected_backbone}
        
        head_prefix = "head."
        remaining_backbone_state = {}
        for key, value in unexpected_backbone_state.items():
            if key.startswith(head_prefix):
                remaining_backbone_state[key[len(head_prefix):]] = value
            else:
                remaining_backbone_state[key] = value
        missing_backbone_head, unexpected_backbone = self.head.load_state_dict(remaining_backbone_state, strict=False)

        if missing_backbone_model:
            logging.warning("Missing backbone keys after HF load (model): %s", missing_backbone_model)
        if missing_backbone_head:
            logging.warning("Missing backbone keys after HF load (head): %s", missing_backbone_head)
        if unexpected_backbone:
            logging.warning("Unexpected backbone keys after HF load (model & head): %s", unexpected_backbone)
        if not unexpected_backbone and not missing_backbone_model and not missing_backbone_head:
            logging.info("Successfully loaded HF model backbone weights into HFMaskedScorer backbone.")

        classifier_module = getattr(reference_model, "classifier", None)
        loading_dict = loading_info if isinstance(loading_info, dict) else {}
        missing_keys = loading_dict.get("missing_keys", [])
        mismatched_keys = loading_dict.get("mismatched_keys", [])
        classifier_missing = any(key.startswith("classifier.") for key in missing_keys)
        classifier_mismatched = any(
            (key if isinstance(key, str) else key[0]).startswith("classifier.")
            for key in mismatched_keys
        )

        if classifier_module is not None and not classifier_missing and not classifier_mismatched:
            classifier_state = classifier_module.state_dict()
            missing_cls, unexpected_cls = self.classifier.load_state_dict(classifier_state, strict=False)
            if missing_cls:
                logging.warning("Missing classifier keys after HF load: %s", missing_cls)
            if unexpected_cls:
                logging.warning("Unexpected classifier keys after HF load: %s", unexpected_cls)
            if not unexpected_cls and not missing_cls:
                logging.info("Successfully loaded HF model classifier weights into HFMaskedScorer classifier.")
        else:
            logging.info("Skipping HF classifier head weights (missing or mismatched in checkpoint).")

        del reference_model
        logging.info(f"Initialized HFMaskedScorer with model ID: {self.hf_id}, hidden size: {self.hf_config.hidden_size}, number of layers: {self.hf_config.num_hidden_layers}, dropout: {self.hf_config.classifier_dropout}")

    def forward(self, tokenized: TokenizedTexts, layer_attention_masks: Optional[List[torch.FloatTensor]]) -> TextsRepresentationOutput:
        tokenized = tokenized.to(self.device)
        outputs = self.model(
            tokenized.ids,
            token_type_ids=tokenized.token_type_ids,
            layer_attention_masks=layer_attention_masks
        )

        last_hidden_state = outputs[0]

        pooled_last_hidden_state = self.pooling_function(last_hidden_state, layer_attention_masks)
            
        head_output = self.head(pooled_last_hidden_state)
        head_output = self.dropout_layer(head_output)
        logits = self.classifier(head_output).squeeze(1)

        return logits
