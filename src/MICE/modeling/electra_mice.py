import logging
import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from transformers.models.electra.modeling_electra import ElectraLayer

from experimaestro import Param, field
from .mice import (
    BertMiceCrossEncoder,
    InitMICEBERTFromHFID,
)

logger = logging.getLogger(__name__)


class ElectraMiceCrossEncoder(BertMiceCrossEncoder):
    """Mice Cross Encoder based on Electra Architecture."""

    def __initialize__(self):
        # We don't call super().__initialize__() because it would use BertLayer
        # instead we call the grandparent's initialize
        super(BertMiceCrossEncoder, self).__initialize__()

        self.config = AutoConfig.from_pretrained(self.hf_id)
        self.head_config = AutoConfig.from_pretrained(self.hf_id)
        self.head_config.is_decoder = True
        self.head_config.add_cross_attention = True

        if self.compress_dim > 1:
            self.head_config.hidden_size = int(
                self.head_config.hidden_size / self.compress_dim
            )
            self.head_config.intermediate_size = int(
                self.head_config.intermediate_size / self.compress_dim
            )
            self.head_config.num_attention_heads = int(
                self.head_config.num_attention_heads / self.compress_dim
            )
            self.adapter = nn.Linear(
                self.config.hidden_size, self.head_config.hidden_size
            )
        else:
            self.adapter = None

        # Build skeleton (initially random)
        temp_model = AutoModel.from_config(self.config)
        self.add_module("embeddings", temp_model.embeddings)
        if hasattr(temp_model, "embeddings_project"):
            self.embeddings_project = temp_model.embeddings_project
        else:
            self.embeddings_project = None

        # Build Bottom layers
        if self.bound_bottom_layers:
            self.add_module(
                "bottom_layers",
                nn.ModuleList(
                    [
                        ElectraLayer(self.config)
                        for _ in range(self.n_contextualization_layers)
                    ]
                ),
            )
        else:
            self.add_module(
                "query_bottom_layers",
                nn.ModuleList(
                    [
                        ElectraLayer(self.config)
                        for _ in range(self.n_contextualization_layers)
                    ]
                ),
            )
            n_doc_layers = (
                self.n_docs_ctx_layers
                if self.n_docs_ctx_layers is not None
                else self.n_contextualization_layers
            )
            self.add_module(
                "document_bottom_layers",
                nn.ModuleList(
                    [ElectraLayer(self.config) for _ in range(n_doc_layers)],
                ),
            )

        if self.n_interaction_layers is not None:
            num_top = self.n_interaction_layers
        else:
            num_top = len(temp_model.encoder.layer) - self.n_contextualization_layers

        self.add_module(
            "top_layers",
            nn.ModuleList([ElectraLayer(self.head_config) for _ in range(num_top)]),
        )

        if self.extra_attn_bias:
            from .mice import ExactMatchAttentionHead

            self.add_module(
                "exact_match_heads",
                nn.ModuleList(
                    [
                        ExactMatchAttentionHead(self.head_config.hidden_size)
                        for _ in range(num_top)
                    ]
                ),
            )

        # Standard BertMice components (pooler, classifier, dropout)
        self.pooler = getattr(temp_model, "pooler", None)
        if self.pooler:
            if self.compress_dim > 1:
                self.pooler.dense = nn.Linear(
                    self.head_config.hidden_size,
                    self.head_config.hidden_size,
                    bias=True,
                )
            self.add_module("pooler", self.pooler)

        self.dropout_layer = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.head_config.hidden_size, 1)

        if self.global_cls_token:
            self.global_cls = nn.Parameter(
                torch.randn(1, 1, self.head_config.hidden_size) * 0.02
            )

    def encode_queries(self, input_ids, attention_mask):
        """Compute bottom layers (independent encoding)"""
        x = self.embeddings(input_ids)
        if self.embeddings_project is not None:
            x = self.embeddings_project(x)

        # Standard BERT extended mask logic
        ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)
        if self.bound_bottom_layers:
            query_bottom_layers = self.bottom_layers
        else:
            query_bottom_layers = self.query_bottom_layers

        for layer in query_bottom_layers:
            x = layer(x, ext_mask)
        return x

    def encode_documents(self, input_ids, attention_mask):
        """Compute bottom layers (independent encoding)"""
        x = self.embeddings(input_ids)
        if self.embeddings_project is not None:
            x = self.embeddings_project(x)

        # Standard BERT extended mask logic
        ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)

        if self.bound_bottom_layers:
            document_bottom_layers = self.bottom_layers
        else:
            document_bottom_layers = self.document_bottom_layers

        for layer in document_bottom_layers:
            x = layer(x, ext_mask)
        return x


class InitMICEElectraFromHFID(InitMICEBERTFromHFID):
    """Worker-node task to load weights into MICE Electra model"""

    model: Param[ElectraMiceCrossEncoder] = field(overrides=True)

    def execute(self):
        # We need to mostly re-implement execute because it's hard to call super()
        # and handle the embeddings_project correctly without duplicating model load

        model = self.model
        hf_id = model.hf_id

        # Ensure configs are available
        if not hasattr(model, "config") or model.config is None:
            model.config = AutoConfig.from_pretrained(hf_id)

        if not hasattr(model, "head_config") or model.head_config is None:
            model.head_config = AutoConfig.from_pretrained(hf_id)
            model.head_config.is_decoder = True
            model.head_config.add_cross_attention = True

        # Build the model structure first
        model.initialize()

        logger.info(f"Building MICE from pretrained Electra model: {hf_id}")

        full_backbone = AutoModel.from_pretrained(hf_id)

        if hasattr(full_backbone, "embeddings"):
            logger.info("Seeding embeddings from backbone")
            model.embeddings.load_state_dict(full_backbone.embeddings.state_dict())

        if (
            hasattr(full_backbone, "embeddings_project")
            and full_backbone.embeddings_project is not None
        ):
            logger.info("Seeding embeddings_project from backbone")
            model.embeddings_project.load_state_dict(
                full_backbone.embeddings_project.state_dict()
            )

        # Copy bottom layers
        if hasattr(full_backbone, "encoder") and hasattr(
            full_backbone.encoder, "layer"
        ):
            logger.info(
                f"Seeding {model.n_contextualization_layers} bottom layers from backbone"
            )
            if model.bound_bottom_layers:
                contextualization_module_names = ["bottom_layers"]
            else:
                contextualization_module_names = [
                    "document_bottom_layers",
                    "query_bottom_layers",
                ]

            for cntx_module_name in contextualization_module_names:
                contextualization_module = getattr(model, cntx_module_name)
                for i in range(len(contextualization_module)):
                    if i < len(full_backbone.encoder.layer):
                        contextualization_module[i].load_state_dict(
                            full_backbone.encoder.layer[i].state_dict()
                        )
                    else:
                        logger.warning(
                            f"Backbone has only {len(full_backbone.encoder.layer)} layers; cannot seed bottom layer {i}"
                        )

        # Load original top layers to copy weights from
        start_idx = model.n_contextualization_layers
        if model.n_interaction_layers is not None:
            end_idx = start_idx + model.n_interaction_layers
            original_top_layers = full_backbone.encoder.layer[start_idx:end_idx]
        else:
            original_top_layers = full_backbone.encoder.layer[start_idx:]

        for i in range(len(original_top_layers)):
            target_layer = model.top_layers[i]
            if not model.random_top_layers:
                logger.info(
                    f"Copying weights from original Electra to Interaction top layer {i}"
                )
                self._copy_bert_weights(original_top_layers[i], target_layer)

        # pooler
        model.pooler = getattr(full_backbone, "pooler", None)

        if model.global_cls_token:
            cls_token_id = model.tokenizer.tokenizer.cls_token_id
            if cls_token_id is not None:
                logger.info(
                    f"Seeding global_cls with [CLS] embedding (ID {cls_token_id})"
                )
                with torch.no_grad():
                    cls_embedding = full_backbone.embeddings.word_embeddings.weight[
                        cls_token_id
                    ]
                    model.global_cls.data.copy_(cls_embedding.view(1, 1, -1))
