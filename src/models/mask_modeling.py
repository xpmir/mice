from abc import abstractmethod
import torch
from typing import Optional, List, Tuple, Union

import logging
import torch.nn as nn

try:
    from transformers import BertModel, ModernBertConfig, ModernBertModel
    from transformers.models.bert.modeling_bert import BertEncoder
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertEmbeddings, 
        ModernBertEncoderLayer, 
        ModernBertRotaryEmbedding, 
    )
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
    from transformers import PreTrainedModel

except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


class CustomMaskModel(PreTrainedModel):
    """
    Base class for models that support layer-specific attention masks.
    """

    @abstractmethod
    def forward(self, *args, layer_attention_masks: Optional[List[torch.FloatTensor]] = None, **kwargs):
        """
        Forward method that accepts layer-specific attention masks.
        
        Args:
            layer_attention_masks: Optional list of attention masks, one per encoder layer.
                                   Each mask should have shape [batch_size, seq_len, seq_len].
        """
        raise NotImplementedError("CustomMaskModel is an abstract base class and does not implement forward()")

class CustomMaskBertEncoder(BertEncoder):
    """
    Modified BertEncoder that accepts layer-specific attention masks.
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_attention_masks: Optional[List[torch.FloatTensor]],
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        next_decoder_cache = () if use_cache else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_attention_masks[i],
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=layer_attention_masks[i],
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_value,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class CustomMaskBertModel(BertModel, CustomMaskModel):
    """
    Modified BertModel that supports layer-specific attention masks.
    
    Args:
        config: BertConfig instance
        
    Forward Args:
        layer_attention_masks: Optional list of attention masks, one per encoder layer.
                              Each mask should have shape [batch_size, num_heads, seq_len, seq_len]
                              or [batch_size, 1, seq_len, seq_len].
                              If None, uses the standard attention_mask for all layers.
    """
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # Replace the encoder with our custom encoder
        self.encoder = CustomMaskBertEncoder(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        layer_attention_masks: Optional[List[torch.Tensor]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Taken from transformers BertModel.forward (https://github.com/huggingface/transformers/blob/40dc11cd3eb4126652aa41ef8272525affd4a636/src/transformers/models/bert/modeling_bert.py#L639)
        with modifications to accept layer-specific attention masks.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, layer_attention_masks[0])
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Prepare layer-specific attention masks if provided
        if layer_attention_masks is not None:
            # Get the extended 4D ? mask for this layer
            extended_layer_masks = [
                self.get_extended_attention_mask(layer_mask, input_shape)
                for layer_mask in layer_attention_masks]
        else:
            # could implement inputting states without layer masks
            raise ValueError("layer_attention_masks must be provided for CustomMaskBertModel")

        # Prepare encoder attention mask if this model is being used as a decoder
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            layer_attention_masks=extended_layer_masks,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

### SAME FOR ETTIN-BASED MODELS ###
class CustomMaskModernBertModel(ModernBertModel, CustomMaskModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.rotary_emb = ModernBertRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        layer_attention_masks: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor, ...], BaseModelOutput]:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """
        if self.config._attn_implementation == "flash_attention_2":
            raise ValueError("Flash Attention 2 is not supported with layer-specific attention masks.")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self._maybe_set_compile()

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        layer_attention_masks, sliding_window_masks = self._update_layer_attention_masks(
            layer_attention_masks, output_attentions=output_attentions
        )

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=layer_attention_masks[idx],
                sliding_window_mask=sliding_window_masks[idx],
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
    def _expand_masks(self, attention_masks: List[torch.Tensor], dtype: torch.dtype) -> List[torch.Tensor]:
        expanded_masks = []
        for attention_mask in attention_masks:
            # Expand to 4D tensor for multi-head attention
            expanded_mask = attention_mask[:, None, :, :]
            inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask

            expanded_masks.append(inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min))
        return expanded_masks

    def _update_layer_attention_masks(self, attention_masks: List[torch.Tensor], output_attentions: bool) -> torch.Tensor:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logging.warning_once(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logging.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )

        global_attention_masks = self._expand_masks(attention_masks, dtype=self.dtype)

        # Create position indices
        sliding_window_masks = []
        for global_attention_mask in global_attention_masks:
            rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
            # Calculate distance between positions
            distance = torch.abs(rows - rows.T)

            # Create sliding window mask (1 for positions within window, 0 outside)
            window_mask = (
                (distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_masks[0].device)
            )
            # Combine with existing mask
            sliding_window_masks.append(global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min))

        return global_attention_masks, sliding_window_masks
