from collections.abc import Callable
from typing import Optional
import torch
import torch.nn as nn

from experimaestro import Constant, Param
from datamaestro_text.data.ir import TextItem
from transformers import GradientCheckpointingLayer
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer
from configuration import PoolingMethod
from xpm_torch.xpmModel import xpmTorchHubModule
from xpm_torch.utils.logging import easylog
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.pytorch_utils import apply_chunking_to_forward
import logging

logger = easylog()
logger.setLevel(logging.INFO)

# check if transformers is installed
try:
    from transformers import (
        ModernBertConfig,
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        AutoModelForSequenceClassification,
    )
    from transformers.models.bert.modeling_bert import BertLayer
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertPredictionHead,
        ModernBertMLP,
        ModernBertAttention,
    )
    from transformers.models.modernbert_decoder.modeling_modernbert_decoder import (
        eager_attention_forward,
        ModernBertDecoderRotaryEmbedding,
        rotate_half,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


class MidFusionCrossEncoder(xpmTorchHubModule, LearnableScorer):
    """
    Mid-Fusion Cross Encoder base Architecture, with Cross-Attention in the top layers.
    The bottom layers encode query and document independently, while the top layers
    allow the query to attend to the document via cross-attention mechanisms.
    """

    hf_id: Param[str] = "bert-base-uncased"
    """Hugging Face checkpoint identifier that provides weights and config. Must be a BERT-like model."""

    merge_layer: Param[int] = 6
    """Mid-fusion index: encoder layers are split into bottom (independent) and top (cross-attention)"""

    drop_layer: Param[int] = 0
    """Layer at which to drop backbone layers"""

    mask_cls_to_doc: Param[bool] = True
    """Whether to mask the [CLS] token from attending to document tokens."""

    mask_query_to_cls: Param[bool] = True
    """Whether to mask query tokens from attending to the [CLS] token (using it as a sink)"""

    use_self_attention: Param[bool] = True
    """Whether to use self-attention in the top layers along with cross-attention."""

    freeze_base: Param[bool] = False
    """Whether to freeze the bottom layers during finetuning"""

    random_top_layers: Param[bool] = False
    """Whether to initialize top layers randomly instead of copying from backbone"""

    compress_dim: Param[float] = 1.0
    """Factor by which to divide the hidden dimensions of the top layers"""

    _version: Constant[int] = 2
    """Model version"""

    embeddings: nn.Module
    """Shared embeddings for query and document"""

    bottom_layers: nn.ModuleList
    """Bottom layers: independent encoding"""

    top_layers: nn.ModuleList
    """Top layers: cross-attention encoding"""

    def __initialize__(self, options):
        super().__initialize__(options)

        ### Gather all the components shared across implementations:
        assert (
            self.merge_layer > 0
        ), "merge_layer must be > 0, otherwise no bottom layers exist"
        assert not (
            self.drop_layer and self.drop_layer < self.merge_layer
        ), "drop_layer must be > merge_layer"

        # 1. Load Base Configuration
        self.config = AutoConfig.from_pretrained(self.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)

        # 2. The Top "Head" (Query attends to Document)
        # We need to create NEW layers that support Cross-Attention

        # Enable decoder mode for the top layers configuration
        self.head_config = AutoConfig.from_pretrained(self.hf_id)
     
        # Ensure attention implementation matches base config if not present
        if (
            not hasattr(self.head_config, "_attn_implementation")
            or self.head_config._attn_implementation is None
        ):
            if (
                hasattr(self.config, "_attn_implementation")
                and self.config._attn_implementation is not None
            ):
                self.head_config._attn_implementation = self.config._attn_implementation
            else:
                self.head_config._attn_implementation = "eager"

    @property
    def max_length(self) -> int:
        return getattr(
            self.config, "max_position_embeddings", self.tokenizer.model_max_length
        )

    def get_extended_attention_mask(self, mask, dtype):
        """Helper to create the -inf mask for transformers"""
        inverted_mask = 1.0 - mask[:, None, None, :]
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )

    def get_self_attention_mask(self, mask, dtype):
        """Helper to create the -inf mask for transformers"""
        mask_bool = mask.to(torch.bool)
        # Build a square attention map where both query and key positions must be valid tokens
        valid_pairs = mask_bool[:, None, :, None] & mask_bool[:, None, None, :]
        if self.mask_query_to_cls:
            # Additionally mask out all query tokens from attending to the [CLS] token
            valid_pairs[:, :, 1:, 0] = False
        attn_mask = torch.zeros(valid_pairs.shape, dtype=dtype, device=mask.device)
        attn_mask.masked_fill_(~valid_pairs, torch.finfo(dtype).min)
        return attn_mask

    def get_cross_attention_mask(self, query_mask, doc_mask, dtype):
        """Helper to create the -inf mask for cross-attention with rectangular support"""
        q_valid = query_mask.to(torch.bool)[:, :, None]
        d_valid = doc_mask.to(torch.bool)[:, None, :]
        valid_pairs = q_valid & d_valid
        if self.mask_cls_to_doc:
            # Additionally mask out the [CLS] token from attending to document tokens
            valid_pairs[:, 0, :] = False
        attn_mask = torch.zeros(
            (valid_pairs.size(0), 1, valid_pairs.size(1), valid_pairs.size(2)),
            dtype=dtype,
            device=query_mask.device,
        )
        attn_mask.masked_fill_(~valid_pairs[:, None, :, :], torch.finfo(dtype).min)
        return attn_mask


class MiniLMMidFusionCrossEncoder(MidFusionCrossEncoder):
    """
    Mid-Fusion Cross Encoder based on BERT Architecture.
    """

    def __initialize__(self, options):
        super().__initialize__(options)

        # for BertLayer, we can just activate is_decoder and add_cross_attention in the config, oof !
        self.head_config.is_decoder = True
        self.head_config.add_cross_attention = True


        if self.compress_dim > 1:
            self.head_config.hidden_size = int(self.head_config.hidden_size / self.compress_dim)
            self.head_config.intermediate_size = int(self.head_config.intermediate_size / self.compress_dim)
            self.head_config.num_attention_heads = int(self.head_config.num_attention_heads / self.compress_dim)
            logging.info(
                f"Compressing top layer dimensions by factor {self.compress_dim}: hidden_size={self.head_config.hidden_size}, intermediate_size={self.head_config.intermediate_size}, num_attention_heads={self.head_config.num_attention_heads}"
            )
            # self.adapter = nn.Sequential(
            #     nn.Linear(
            #         self.config.hidden_size, self.head_config.hidden_size, bias=True
            #     ),
            #     nn.GELU(),
            # )
            self.adapter = nn.Linear(
                self.config.hidden_size, self.head_config.hidden_size, bias=True
            )
        else:
            self.adapter = None
            

        # 2. The Bottom Encoders (Shared weights for Query and Doc)
        # We load the pre-trained BERT model for the bottom half
        full_bert = AutoModel.from_pretrained(self.hf_id)
        self.embeddings = full_bert.embeddings

        # Take layers 0 to merge_layer-1
        self.bottom_layers = nn.ModuleList(full_bert.encoder.layer[: self.merge_layer])

        self.top_layers = nn.ModuleList()

        # Load original top layers to copy weights from
        if self.drop_layer > 0:
            assert (
                self.drop_layer >= self.merge_layer
            ), "drop_layer must be >= merge_layer"
            assert self.drop_layer < len(
                full_bert.encoder.layer
            ), f"drop_layer {self.drop_layer} exceeds number of layers in the backbone: {len(full_bert.encoder.layer)}"
            original_top_layers = full_bert.encoder.layer[
                self.merge_layer : self.drop_layer
            ]
            logging.info(
                f"Dropping backbone layers {self.drop_layer}-{len(full_bert.encoder.layer)-1}"
            )
        else:
            original_top_layers = full_bert.encoder.layer[self.merge_layer :]

        for i in range(len(original_top_layers)):
            # Instantiate a fresh layer with Cross-Attention enabled
            new_layer = BertLayer(self.head_config)

            # COPY trained weights (Self-Attention + FFN) from original BERT to new layer
            # Note: The Cross-Attention block (new_layer.crossattention) will remain random!
            if not self.random_top_layers:
                logger.info(f"Copying weights from original BERT to Mid-Fusion top layer {i}")
                self._copy_weights(original_top_layers[i], new_layer)
            else:
                logger.info(f"Initializing Mid-Fusion top layer {i} randomly")

            self.top_layers.append(new_layer)

        # pooler 
        # Preserve the pretrained pooler to keep the original [CLS] projection
        self.pooler = getattr(full_bert, "pooler", None)

        if self.pooler is None:
            logger.warning(
                "No pooler found in the base model; using [CLS] token directly."
            )
        elif self.compress_dim > 1:
            #we must adjust the pooler to the new hidden size
            self.pooler.dense = nn.Linear(
                self.head_config.hidden_size, self.head_config.hidden_size, bias=True   
            )

        # Classifier
        self.dropout_layer = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.head_config.hidden_size, 1)

        if self.freeze_base:
            logger.info("Freezing base model (bottom layers) parameters.")
            self.bottom_layers.requires_grad_(False)
            self.embeddings.requires_grad_(False)

    def _copy_weights(self, src_layer, target_layer):
        """
        Copies Self-Attention and FFN weights from src to target.
        Leaves Cross-Attention weights (only in target) initialized randomly.
        """
        # Copy Self-Attention
        target_layer.attention.self.load_state_dict(
            src_layer.attention.self.state_dict()
        )
        target_layer.attention.output.load_state_dict(
            src_layer.attention.output.state_dict()
        )

        # Copy Intermediate (FFN part 1)
        target_layer.intermediate.load_state_dict(src_layer.intermediate.state_dict())

        # Copy Output (FFN part 2)
        target_layer.output.load_state_dict(src_layer.output.state_dict())

        # Seed cross-attention with self-attention weights when available
        if (
            hasattr(target_layer, "crossattention")
            and target_layer.crossattention is not None
        ):
            try:
                target_layer.crossattention.self.load_state_dict(
                    src_layer.attention.self.state_dict()
                )
                target_layer.crossattention.output.load_state_dict(
                    src_layer.attention.output.state_dict()
                )
            except Exception as err:
                logger.exception("Cross-attention weight copy failed: %s", err)
            else:
                logger.info("Cross-attention weights seeded from self-attention")

    def forward_bottom(self, input_ids, attention_mask):
        """Compute bottom layers (independent encoding)"""
        x = self.embeddings(input_ids)
        # Standard BERT extended mask logic
        ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)
        # Compute position ids and pass them to ModernBertEncoderLayer which expects them
        batch, seq_len = input_ids.size()

        for layer in self.bottom_layers:
            x = layer(x, ext_mask)[0]
        return x

    def forward_bertLayer_wo_selfAttention(
        self, 
        bertLayer: BertLayer, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        """ Same forward as modeling_bert.BertLayer but without self-attention"""
        # self_attention_outputs = bertLayer.attention(
        #     hidden_states,
        #     attention_mask=attention_mask,
        #     head_mask=head_mask,
        #     output_attentions=output_attentions,
        #     past_key_values=past_key_values,
        #     cache_position=cache_position,
        # )
        # attention_output = self_attention_outputs[0]
        # outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        assert bertLayer.is_decoder, "bertLayer must be instantiated as a decoder (is_decoder=True)"
        assert encoder_hidden_states is not None, "encoder_hidden_states must be provided for cross-attention"

        if not hasattr(bertLayer, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {bertLayer} has to be instantiated with cross-attention layers"
                " by setting `config.add_cross_attention=True`"
            )

        cross_attention_outputs = bertLayer.crossattention(
            hidden_states,
            attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            bertLayer.feed_forward_chunk, bertLayer.chunk_size_feed_forward, bertLayer.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def tokenze_texts(self, texts):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
    
    def forward(self, inputs: BaseRecords, 
                info: TrainerContext = None,
                tokenized_queries: Optional[dict] = None,
                tokenized_docs: Optional[dict] = None,
                doc_hidden_states: Optional[torch.Tensor] = None,
                ):
        """
        Forward pass of the Mid-Fusion Cross Encoder.
        inputs: BaseRecords containing 'topics' and 'documents' with TextItems.
        tokenized_queries: Optional pre-tokenized queries to skip tokenization step.
        tokenized_docs: Optional pre-tokenized documents to skip tokenization step.
        doc_hidden_states: Optional pre-computed document hidden states from bottom layers.
        info: TrainerContext for additional context (not used here).
        """

        # Prepare inputs

        # Tokenize
        if tokenized_docs is None:
            docs = [d[TextItem].text for d in inputs.documents]
            tokenized_docs = self.tokenze_texts(docs)
        
        if tokenized_queries is None:
            queries = [t[TextItem].text for t in inputs.topics]
            tokenized_queries = self.tokenze_texts(queries)

        query_ids = tokenized_queries.input_ids.to(self.device)
        query_mask = tokenized_queries.attention_mask.to(self.device)
        doc_ids = tokenized_docs.input_ids.to(self.device)
        doc_mask = tokenized_docs.attention_mask.to(self.device)

        # 1. Process Query through Bottom Layers
        q_hidden = self.forward_bottom(query_ids, query_mask)
        # Mask for Self-Attention (Query) shape [batch, 1, seq_len_query, seq_len_query]
        q_ext_mask = self.get_self_attention_mask(query_mask, q_hidden.dtype)

        if doc_hidden_states is None:
            # Process Doc through Bottom Layers
            doc_hidden_states = self.forward_bottom(doc_ids, doc_mask) # shape [batch, seq_len_doc, dim]
            # 2. Prepare Masks for Top Layers

        # Mask for Cross-Attention (Query attending to Doc) shape [batch, 1, seq_len_query, seq_len_doc]
        d_ext_mask = self.get_cross_attention_mask(query_mask, doc_mask, q_hidden.dtype)

        # 3. Process Query through Top Layers (with Cross-Attention to Doc)
        if self.adapter is not None:
            q_hidden = self.adapter(q_hidden)
            doc_hidden_states = self.adapter(doc_hidden_states)


        for layer in self.top_layers:
            # BertLayer with is_decoder=True accepts:
            # (hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
            if self.use_self_attention:
                layer_out = layer(
                    hidden_states=q_hidden,  # Query (Self-Attn)
                    attention_mask=q_ext_mask,
                    encoder_hidden_states=doc_hidden_states,  # Document (Cross-Attn Key/Value)
                    encoder_attention_mask=d_ext_mask,
                )
            else:
                layer_out = self.forward_bertLayer_wo_selfAttention(
                    bertLayer=layer,
                    hidden_states=q_hidden,  # Query (Cross-Attn)
                    encoder_hidden_states=doc_hidden_states,  # Document (Cross-Attn Key/Value)
                    encoder_attention_mask=d_ext_mask,
                )

            q_hidden = layer_out[0]

        # 4. Score (Use [CLS] of the Query)
        if self.pooler is not None:
            pooled = self.pooler(q_hidden)
        else:
            pooled = q_hidden[:, 0, :]

        pooled = self.dropout_layer(pooled)
        score = self.classifier(pooled)
        return score.squeeze(-1)


class ModernBertCrossAttentionLayer(GradientCheckpointingLayer):
    """
    A ModernBERT encoder layer with added cross-attention for mid-fusion ranking.

    This layer combines:
    1. Bidirectional self-attention on the query (with optional local/global patterns from ModernBERT)
    2. Cross-attention from query to document (always global)
    3. Feed-forward network (MLP)
    """

    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config

        # Layer normalization (first layer uses identity for pre-norm architecture)
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )

        # Self-attention on query: inherits local/global behavior from ModernBERT encoder
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)

        # Cross-attention from query to document: always uses global (full) attention
        self.crossattention = ModernBertCrossAttention(config=config, layer_idx=layer_id)
        
        # Feed-forward network
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))
    
    def forward_modernbertLayer_wo_selfAttention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        """ Same forward as modeling_bert.BertLayer but without self-attention"""
        # Self-attention on query (bidirectional)
        # attn_outputs = self.attn(
        #     self.attn_norm(hidden_states),
        #     attention_mask=attention_mask,
        #     sliding_window_mask=sliding_window_mask,
        #     position_ids=position_ids,
        #     cu_seqlens=cu_seqlens,
        #     max_seqlen=max_seqlen,
        #     output_attentions=output_attentions,
        # )

        # attention_output = attn_outputs[0]
        # outputs = attn_outputs[1:]

        # Apply self-attention residual immediately
        # hidden_states = hidden_states + attention_output

        # Prepare position embeddings for query and encoder (rotary)
        if position_ids is None:
            # create position ids for query if missing
            b_q, s_q, _ = hidden_states.shape
            position_ids = (
                torch.arange(s_q, device=hidden_states.device)
                .unsqueeze(0)
                .expand(b_q, s_q)
            )

        # Cross-attention from query to document (always global/full attention)
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.attn_norm(encoder_hidden_states)

            cross_attention_outputs = self.crossattention(
                query=self.attn_norm(hidden_states),
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            cross_attention_output = cross_attention_outputs[0]
            # normalize outputs tuples to concatenation
            outputs = cross_attention_outputs[1:]
            hidden_states = hidden_states + cross_attention_output

        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + outputs  # add attentions if outputted

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        
        # Self-attention on query (bidirectional)
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # Apply self-attention residual immediately
        hidden_states = hidden_states + attention_output

        # Prepare position embeddings for query and encoder (rotary)
        if position_ids is None:
            # create position ids for query if missing
            b_q, s_q, _ = hidden_states.shape
            position_ids = (
                torch.arange(s_q, device=hidden_states.device)
                .unsqueeze(0)
                .expand(b_q, s_q)
            )

        # Cross-attention from query to document (always global/full attention)
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.attn_norm(encoder_hidden_states)

            cross_attention_outputs = self.crossattention(
                query=self.attn_norm(hidden_states),
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            cross_attention_output = cross_attention_outputs[0]
            # normalize outputs tuples to concatenation
            outputs = outputs + cross_attention_outputs[1:]
            hidden_states = hidden_states + cross_attention_output

        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + outputs  # add attentions if outputted


class ModernBertCrossAttention(nn.Module):
    """Cross-attention wrapper for ModernBERT that accepts separate query/key/value tensors.

    Mirrors the projection and attention interface of ModernBertDecoderAttention but
    allows providing distinct key/value tensors coming from an encoder.
    """

    # TODO: Consider changing the config type, as we add new attributes to it and only use a subset of ModernBertConfig
    def __init__(self, config: ModernBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.scaling = (
            self.head_dim**-0.5
        )  # See https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py#L311
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )

        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )

        # Note: Cross-attention is always global (full attention from query to all document tokens)
        # No sliding window restriction is applied here

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Handle different sequence lengths for query / key / value
        q_input_shape = query.shape[:-1]
        q_hidden_shape = (*q_input_shape, -1, self.head_dim)
        k_input_shape = key.shape[:-1]
        k_hidden_shape = (*k_input_shape, -1, self.head_dim)
        v_input_shape = value.shape[:-1]
        v_hidden_shape = (*v_input_shape, -1, self.head_dim)

        query_states = self.q_proj(query).view(q_hidden_shape).transpose(1, 2)
        key_states = self.k_proj(key).view(k_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(value).view(v_hidden_shape).transpose(1, 2)

        # Fallback to a direct (explicit) cross-attention implementation
        # This handles different query/key/value sequence lengths and
        # doesn't rely on attention helpers that expect packed qkv tensors.
        output_attentions = kwargs.get("output_attentions", False)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            # sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*q_input_shape, -1).contiguous()
        attn_output = self.out_drop(self.Wo(attn_output))

        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class EttinMidFusionCrossEncoder(MidFusionCrossEncoder):
    """
    Mid-Fusion Cross Encoder based on ModernBERT Architecture.
    """

    pooling_method: Param[Optional[str]] = None
    """Pooling method to use for the Ettin based scorer: cls or mean.
    Leave it to None for models coming from the Hub, as it will be inferred from the model config."""

    _version: Constant[int] = 10
    """Model version"""

    def __initialize__(self, options):
        super().__initialize__(options)
        if self.pooling_method is None:
            if self.config.classifier_pooling == PoolingMethod.CLS.value:
                self.pooling_function = lambda x: x[:, 0]
            elif self.config.classifier_pooling == PoolingMethod.MEAN.value:
                self.pooling_function = lambda x: (x).mean(dim=1)
            else:
                raise ValueError(
                    f"Unsupported pooling method in model config: {self.config.classifier_pooling}"
                )
        else:
            if self.pooling_method == PoolingMethod.CLS.value:
                self.pooling_function = lambda x: x[:, 0]
            elif self.pooling_method == PoolingMethod.MEAN.value:
                self.pooling_function = lambda x: (x).mean(dim=1)
            else:
                raise ValueError(
                    f"Unsupported pooling method provided: {self.pooling_method}"
                )

        # The Bottom Encoders (Shared weights for Query and Doc)
        # We load the pre-trained BERT model for the bottom half
        full_modernbert = AutoModelForSequenceClassification.from_pretrained(self.hf_id)
        self.embeddings = full_modernbert.model.embeddings

        # Take layers 0 to merge_layer-1
        self.bottom_layers = nn.ModuleList(
            full_modernbert.model.layers[: self.merge_layer]
        )

        self.top_layers = nn.ModuleList()
        # Load original top layers to copy weights from
        if self.drop_layer > 0:
            assert (
                self.drop_layer >= self.merge_layer
            ), "drop_layer must be >= merge_layer"
            assert self.drop_layer < len(
                full_modernbert.model.layers
            ), f"drop_layer {self.drop_layer} exceeds number of layers in the backbone: {len(full_modernbert.model.layers)}"
            original_top_layers = full_modernbert.model.layers[
                self.merge_layer : self.drop_layer
            ]
            logging.info(
                f"Dropping backbone layers {self.drop_layer}-{len(full_modernbert.model.layers)-1}"
            )
        else:
            original_top_layers = full_modernbert.model.layers[self.merge_layer :]

        for i in range(len(original_top_layers)):
            # Instantiate a fresh layer with Cross-Attention enabled
            # Re-uses the same config from the original layer
            new_layer = ModernBertCrossAttentionLayer(
                full_modernbert.model.layers[self.merge_layer + i].config,
                layer_id=self.merge_layer + i,
            )

            # COPY trained weights (Self-Attention + FFN) from original BERT to new layer
            # Note: The Cross-Attention block (new_layer.crossattention) will remain random!
            if not self.random_top_layers:
                logger.info(f"Copying weights from original ModernBERT to Mid-Fusion top layer {i}")
                self._copy_weights(original_top_layers[i], new_layer)
            else:
                logger.info(f"Initializing Mid-Fusion top layer {i} randomly")

            self.top_layers.append(new_layer)

        self.final_norm = full_modernbert.model.final_norm

        # ModernBertPredictionHead
        self.head: ModernBertPredictionHead = full_modernbert.head

        # Classifier
        self.dropout_layer = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        if self.freeze_base:
            logger.info("Freezing base model (bottom layers) parameters.")
            self.bottom_layers.requires_grad_(False)
            self.embeddings.requires_grad_(False)

        del full_modernbert  # free memory
        torch.cuda.empty_cache()

    def _init_decoder_qkv_from_encoder(self, decoder_attention, encoder_attention):
        """Initialize decoder's separate Q, K, V projections from encoder's combined Wqkv."""
        all_head_size = encoder_attention.all_head_size

        with torch.no_grad():
            # Split and copy weights
            decoder_attention.q_proj.weight.copy_(
                encoder_attention.Wqkv.weight[0:all_head_size, :]
            )
            decoder_attention.k_proj.weight.copy_(
                encoder_attention.Wqkv.weight[all_head_size : 2 * all_head_size, :]
            )
            decoder_attention.v_proj.weight.copy_(
                encoder_attention.Wqkv.weight[2 * all_head_size : 3 * all_head_size, :]
            )

            # Split and copy biases if they exist
            if encoder_attention.Wqkv.bias is not None:
                decoder_attention.q_proj.bias.copy_(
                    encoder_attention.Wqkv.bias[0:all_head_size]
                )
                decoder_attention.k_proj.bias.copy_(
                    encoder_attention.Wqkv.bias[all_head_size : 2 * all_head_size]
                )
                decoder_attention.v_proj.bias.copy_(
                    encoder_attention.Wqkv.bias[2 * all_head_size : 3 * all_head_size]
                )

    def _copy_weights(self, src_layer, target_layer):
        """
        Copies Attention and MLP weights from src to target.
        Leaves Cross-Attention weights (only in target) initialized randomly.
        """
        # Copy Self-Attention
        target_layer.attn_norm.load_state_dict(src_layer.attn_norm.state_dict())
        target_layer.attn.load_state_dict(src_layer.attn.state_dict())

        target_layer.mlp_norm.load_state_dict(src_layer.mlp_norm.state_dict())
        # Copy MLP
        target_layer.mlp.load_state_dict(src_layer.mlp.state_dict())

        # Seed cross-attention with self-attention weights when available
        if (
            hasattr(target_layer, "crossattention")
            and target_layer.crossattention is not None
        ):
            try:
                self._init_decoder_qkv_from_encoder(target_layer.crossattention, src_layer.attn)
                # Copy output projection and dropout
                target_layer.crossattention.Wo.load_state_dict(
                    src_layer.attn.Wo.state_dict()
                )
                target_layer.crossattention.out_drop.load_state_dict(
                    src_layer.attn.out_drop.state_dict()
                )
                logger.info("Cross-attention weights initialized from self-attention")
            except Exception:
                logger.exception("Cross-attention weight copy failed")

    def forward_bottom(self, input_ids, attention_mask):
        """Compute bottom layers (independent encoding)"""
        x = self.embeddings(input_ids)
        # Standard BERT extended mask logic
        ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)
        # Compute position ids and pass them to ModernBertEncoderLayer which expects them
        batch, seq_len = input_ids.size()
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch, seq_len)
        )

        for layer_idx, layer in enumerate(self.bottom_layers):
            x = layer(x, ext_mask, position_ids=position_ids)[0]

        return x

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        # Prepare inputs
        queries = [t[TextItem].text for t in inputs.topics]
        docs = [d[TextItem].text for d in inputs.documents]

        # Tokenize
        q_out = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        d_out = self.tokenizer(
            docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        query_ids = q_out.input_ids.to(self.device)
        query_mask = q_out.attention_mask.to(self.device)
        doc_ids = d_out.input_ids.to(self.device)
        doc_mask = d_out.attention_mask.to(self.device)

        # 1. Process Query through Bottom Layers
        q_hidden = self.forward_bottom(query_ids, query_mask)

        # Process Doc through Bottom Layers
        doc_hidden_states = self.forward_bottom(doc_ids, doc_mask)
        # 2. Prepare Masks for Top Layers

        # Mask for Self-Attention (Query) shape [batch, 1, 1, seq_len_query]
        q_ext_mask = self.get_self_attention_mask(query_mask, q_hidden.dtype)

        # Mask for Cross-Attention (Query attending to Doc) shape [batch, 1, 1, seq_len_doc]
        d_ext_mask = self.get_cross_attention_mask(query_mask, doc_mask, q_hidden.dtype)

        # 3. Process Query through Top Layers (with Cross-Attention to Doc)
        x = q_hidden
        # Prepare position ids for the query (used by rotary / RoPE implementations)
        q_batch, q_seq = query_ids.size()
        q_position_ids = (
            torch.arange(q_seq, device=query_ids.device)
            .unsqueeze(0)
            .expand(q_batch, q_seq)
        )

        for layer_idx, layer in enumerate(self.top_layers):
            layer: ModernBertCrossAttentionLayer 
            # ModernBertCrossAttentionLayer: self-attention + cross-attention + MLP
            if self.use_self_attention:
                layer_out = layer(
                    hidden_states=x,  # Query (Self-Attn)
                    attention_mask=q_ext_mask,
                    encoder_hidden_states=doc_hidden_states,  # Document (Cross-Attn Key/Value)
                    encoder_attention_mask=d_ext_mask,
                    position_ids=q_position_ids,
                )
            else:
                layer_out = layer.forward_modernbertLayer_wo_selfAttention(
                    hidden_states=x,  # Query (Self-Attn)
                    attention_mask=q_ext_mask,
                    encoder_hidden_states=doc_hidden_states,  # Document (Cross-Attn Key/Value)
                    encoder_attention_mask=d_ext_mask,
                    position_ids=q_position_ids,
                )
            
            x = layer_out[0]

        x = self.final_norm(x)
        pooled_last_hidden_state = self.pooling_function(x)

        head_output = self.head(pooled_last_hidden_state)
        head_output = self.dropout_layer(head_output)
        logits = self.classifier(head_output).squeeze(1)

        return logits
