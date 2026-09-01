import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class QwenCrossAttention(nn.Module):
    """
    Cross-attention layer for Qwen models, designed to attend to an external
    encoder's hidden states (e.g., a document representation in MICE).
    Supports Qwen3 features like q_norm, k_norm, and output gating.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        # Determine if bias is needed (Qwen2 uses bias, Qwen3 does not for Q/K/V)
        has_bias = getattr(config, "attention_bias", True)

        # In Qwen3-0.6B: num_heads=16, kv_heads=8, head_dim=128
        # But weights are q=[2048, 1024], k=[1024, 1024], v=[1024, 1024]
        # This means q_proj (before gate split) is 2048 (16 * 128), and k/v are 1024 (8 * 128).
        # WAIT: 8 * 128 = 1024. 16 * 128 = 2048.
        # My previous code used self.num_heads * self.head_dim = 16 * 128 = 2048.
        # And self.num_key_value_heads * self.head_dim = 8 * 128 = 1024.
        # So why did k_proj.weight.copy_ fail with (512) vs (1024)?
        # Ah! 1024 / 16 = 64? No, 1024 / 8 = 128.
        # Let's check config.hidden_size // config.num_attention_heads
        # 1024 // 16 = 64.
        # So head_dim is 64? But the norm says 128.
        # Let's check the config again.

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=has_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=has_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=has_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Output gating (Qwen3)
        self.gate_proj = None
        if hasattr(config, "model_type") and config.model_type == "qwen3":
            # In Qwen3, q_proj in backbone is [2048, 1024], so gate is another 1024.
            self.gate_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )

        # Qwen3 specific: RMSNorm for Q and K
        self.q_norm = (
            Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if hasattr(config, "model_type") and config.model_type == "qwen3"
            else nn.Identity()
        )
        self.k_norm = (
            Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if hasattr(config, "model_type") and config.model_type == "qwen3"
            else nn.Identity()
        )

    def forward(
        self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(encoder_hidden_states).view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(encoder_hidden_states).view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        )

        # Apply q_norm and k_norm if they are not Identity
        if not isinstance(self.q_norm, nn.Identity):
            query_states = self.q_norm(query_states)
        if not isinstance(self.k_norm, nn.Identity):
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Standard GQA: repeat key/value states to match query heads
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(
                self.num_key_value_groups, dim=1
            )

        # Using scaled_dot_product_attention (PyTorch 2.0+)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=encoder_attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        # Output gating (Qwen3)
        if self.gate_proj is not None:
            gate = self.gate_proj(hidden_states)
            attn_output = attn_output * torch.nn.functional.silu(gate)

        attn_output = self.o_proj(attn_output)

        return attn_output


class QwenCrossAttentionLayer(nn.Module):
    """
    A Qwen decoder layer augmented with cross-attention, following the MICE architecture.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine the attention class to use (Qwen2 or Qwen3)
        if hasattr(config, "model_type") and config.model_type == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import (
                Qwen3Attention,
                Qwen3RMSNorm,
                Qwen3MLP,
                Qwen3RotaryEmbedding,
            )

            self.input_layernorm = Qwen3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attn = Qwen3Attention(config, layer_idx)
            self.post_attention_layernorm = Qwen3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = Qwen3MLP(config)
            self.rotary_emb = Qwen3RotaryEmbedding(config)
        else:
            from transformers.models.qwen2.modeling_qwen2 import (
                Qwen2Attention,
                Qwen2RMSNorm,
                Qwen2MLP,
                Qwen2RotaryEmbedding,
            )

            self.input_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attn = Qwen2Attention(config, layer_idx)
            self.post_attention_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = Qwen2MLP(config)
            self.rotary_emb = Qwen2RotaryEmbedding(config)

        self.cross_attn = QwenCrossAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    ):
        # Prepare position embeddings for RoPE if not provided
        if position_embeddings is None and position_ids is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 1. Self-Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # 2. Cross-Attention Block (Mid-Fusion)
        if encoder_hidden_states is not None:
            residual = hidden_states
            # MICE: attend to document states using query states
            # We use the same pre-norm (input_layernorm) for query and doc
            hidden_states = self.cross_attn(
                self.input_layernorm(hidden_states),
                encoder_hidden_states=self.input_layernorm(encoder_hidden_states),
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states

        # 3. MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


def seed_qwen_cross_attn(src_layer: nn.Module, target_layer: QwenCrossAttentionLayer):
    """
    Seeds the cross-attention layer weights from the self-attention weights of a backbone layer.
    """
    print(f"Seeding cross-attention for layer {target_layer.layer_idx}...")
    with torch.no_grad():
        # Copy self-attention weights directly
        target_layer.self_attn.load_state_dict(src_layer.self_attn.state_dict())
        target_layer.input_layernorm.load_state_dict(
            src_layer.input_layernorm.state_dict()
        )
        target_layer.post_attention_layernorm.load_state_dict(
            src_layer.post_attention_layernorm.state_dict()
        )
        target_layer.mlp.load_state_dict(src_layer.mlp.state_dict())

        # Seed cross-attention from self-attention
        src_attn = src_layer.self_attn
        target_cross = target_layer.cross_attn

        # Qwen3: q_proj might be split into q_proj and gate_proj in the weights
        if hasattr(target_cross, "gate_proj") and target_cross.gate_proj is not None:
            # We assume q_proj in source contains [Q; Gate] if its dim is 2 * all_head_size
            all_head = target_cross.num_heads * target_cross.head_dim
            if src_attn.q_proj.weight.shape[0] == 2 * all_head:
                target_cross.q_proj.weight.copy_(src_attn.q_proj.weight[:all_head])
                target_cross.gate_proj.weight.copy_(src_attn.q_proj.weight[all_head:])
            else:
                # Fallback or different architecture
                target_cross.q_proj.weight.copy_(src_attn.q_proj.weight)
        else:
            target_cross.q_proj.weight.copy_(src_attn.q_proj.weight)
            if hasattr(src_attn.q_proj, "bias") and src_attn.q_proj.bias is not None:
                if target_cross.q_proj.bias is not None:
                    target_cross.q_proj.bias.copy_(src_attn.q_proj.bias)

        target_cross.k_proj.weight.copy_(src_attn.k_proj.weight)
        if hasattr(src_attn.k_proj, "bias") and src_attn.k_proj.bias is not None:
            if target_cross.k_proj.bias is not None:
                target_cross.k_proj.bias.copy_(src_attn.k_proj.bias)

        target_cross.v_proj.weight.copy_(src_attn.v_proj.weight)
        if hasattr(src_attn.v_proj, "bias") and src_attn.v_proj.bias is not None:
            if target_cross.v_proj.bias is not None:
                target_cross.v_proj.bias.copy_(src_attn.v_proj.bias)

        target_cross.o_proj.weight.copy_(src_attn.o_proj.weight)
        if hasattr(src_attn.o_proj, "bias") and src_attn.o_proj.bias is not None:
            if target_cross.o_proj.bias is not None:
                target_cross.o_proj.bias.copy_(src_attn.o_proj.bias)

        # Qwen3 specific: copy q_norm and k_norm weights
        if hasattr(src_attn, "q_norm") and not isinstance(
            target_cross.q_norm, nn.Identity
        ):
            target_cross.q_norm.load_state_dict(src_attn.q_norm.state_dict())
        if hasattr(src_attn, "k_norm") and not isinstance(
            target_cross.k_norm, nn.Identity
        ):
            target_cross.k_norm.load_state_dict(src_attn.k_norm.state_dict())


if __name__ == "__main__":
    # Test with a Qwen model
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Loading config for {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        print(f"Could not load {model_id}, trying Qwen/Qwen2.5-0.5B")
        model_id = "Qwen/Qwen2.5-0.5B"
        config = AutoConfig.from_pretrained(model_id)

    # 1. Build a dummy cross-attention layer
    layer_idx = 12
    mice_layer = QwenCrossAttentionLayer(config, layer_idx)

    # 2. Load backbone and seed weights
    print(f"Loading backbone {model_id} to seed weights...")
    backbone = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    src_layer = backbone.model.layers[layer_idx]

    seed_qwen_cross_attn(src_layer, mice_layer)
    print("Seeding successful.")

    # 3. Simple forward test
    batch_size = 2
    q_len = 10
    d_len = 20
    hidden_dim = config.hidden_size

    q_hidden = torch.randn(batch_size, q_len, hidden_dim)
    d_hidden = torch.randn(batch_size, d_len, hidden_dim)
    pos_ids = torch.arange(q_len).unsqueeze(0).expand(batch_size, q_len)

    # We can pass encoder_attention_mask if needed (e.g. for padding)
    # SDPA expects [batch, heads, q_len, kv_len] or [batch, q_len, kv_len]
    cross_mask = torch.ones(batch_size, 1, q_len, d_len, dtype=torch.bool)

    output = mice_layer(
        q_hidden,
        encoder_hidden_states=d_hidden,
        encoder_attention_mask=cross_mask,
        position_ids=pos_ids,
    )[0]

    print(f"Input shape: {q_hidden.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == q_hidden.shape
    print("Forward pass successful!")
