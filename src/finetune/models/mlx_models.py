"""
MLX model implementations for common architectures.
"""

import math
from pathlib import Path

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

from finetune.models.base import BaseModel, ModelConfig
from finetune.training.lora import LoRAConfig, apply_lora_to_model, get_lora_trainable_params

if MLX_AVAILABLE:

    class RMSNorm(nn.Module):
        """RMSNorm layer for MLX."""

        def __init__(self, dims: int, eps: float = 1e-6):
            super().__init__()
            self.weight = mx.ones((dims,))
            self.eps = eps

        def __call__(self, x):
            return mx.fast.rms_norm(x, self.weight, self.eps)

    class Attention(nn.Module):
        """Multi-head attention for MLX."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = config.num_key_value_heads or self.num_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.rope_theta = config.rope_theta

            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

            self.rope = nn.RoPE(self.head_dim, traditional=True, base=self.rope_theta)

        def __call__(
            self,
            x: mx.array,
            mask: mx.array | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> mx.array:
            B, L, D = x.shape

            queries = (
                self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            )
            keys = (
                self.k_proj(x)
                .reshape(B, L, self.num_key_value_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            values = (
                self.v_proj(x)
                .reshape(B, L, self.num_key_value_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

            # Apply RoPE
            queries = self.rope(queries)
            keys = self.rope(keys)

            # Handle cache for inference
            if cache is not None:
                key_cache, value_cache = cache
                keys = mx.concatenate([key_cache, keys], axis=2)
                values = mx.concatenate([value_cache, values], axis=2)

            # Repeat keys and values for GQA
            if self.num_key_value_groups > 1:
                keys = mx.repeat(keys, self.num_key_value_groups, axis=1)
                values = mx.repeat(values, self.num_key_value_groups, axis=1)

            # Scaled dot-product attention
            scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores + mask

            scores = mx.softmax(scores, axis=-1)
            output = mx.matmul(scores, values)

            # Reshape and project
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            output = self.o_proj(output)

            return output

    class MLP(nn.Module):
        """Feed-forward network for MLX."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            self.act = nn.silu if config.hidden_act == "silu" else nn.gelu

        def __call__(self, x):
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    class TransformerBlock(nn.Module):
        """Transformer block for MLX."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.self_attn = Attention(config)
            self.mlp = MLP(config)
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def __call__(
            self,
            x: mx.array,
            mask: mx.array | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> mx.array:
            # Self-attention with residual
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r

            # MLP with residual
            r = self.mlp(self.post_attention_layernorm(h))
            out = h + r

            return out

    class MLXLlamaModel(nn.Module, BaseModel):
        """Llama model implementation in MLX."""

        def __init__(self, config: ModelConfig):
            nn.Module.__init__(self)
            BaseModel.__init__(self, config)

            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            # Output projection
            if config.tie_word_embeddings:
                self.lm_head = None  # Will use embed_tokens.weight.T
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def forward(
            self,
            input_ids: mx.array,
            mask: mx.array | None = None,
            cache: list | None = None,
        ) -> mx.array:
            """Forward pass through the model."""
            # Token embeddings
            h = self.embed_tokens(input_ids)

            # Create causal mask if not provided
            if mask is None and h.shape[1] > 1:
                mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
                mask = mask.astype(h.dtype)

            # Pass through transformer layers
            for i, layer in enumerate(self.layers):
                layer_cache = cache[i] if cache else None
                h = layer(h, mask, layer_cache)

            # Final norm
            h = self.norm(h)

            # Output projection
            if self.lm_head is not None:
                logits = self.lm_head(h)
            else:
                logits = h @ self.embed_tokens.weight.T

            return logits

        def generate(
            self,
            input_ids: mx.array,
            max_length: int = 100,
            temperature: float = 1.0,
            top_p: float = 1.0,
            **kwargs,
        ) -> mx.array:
            """Generate text from the model."""
            cache = []

            # Initial forward pass
            logits = self.forward(input_ids, cache=cache)

            generated = [input_ids]

            for _ in range(max_length - input_ids.shape[1]):
                # Get next token logits
                next_logits = logits[:, -1, :] / temperature

                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits = mx.sort(next_logits, axis=-1)
                    cumsum = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
                    cutoff_idx = mx.argmax(cumsum > top_p, axis=-1)
                    cutoff = sorted_logits[0, cutoff_idx]
                    next_logits = mx.where(next_logits < cutoff, -float("inf"), next_logits)

                # Sample next token
                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs))
                next_token = next_token.reshape(1, 1)

                generated.append(next_token)

                # Forward pass with cache
                logits = self.forward(next_token, cache=cache)

            return mx.concatenate(generated, axis=1)

        def save(self, path: Path):
            """Save model to disk."""
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save config
            self.config.save(path / "config.json")

            # Save weights using MLX native format
            # Flatten nested parameter dicts
            weights = {}
            def flatten_params(params, prefix=""):
                for k, v in params.items():
                    if isinstance(v, dict):
                        flatten_params(v, prefix + k + ".")
                    elif isinstance(v, list):
                        # Handle lists of modules (like layers)
                        for i, item in enumerate(v):
                            if hasattr(item, 'parameters'):
                                flatten_params(dict(item.parameters()), f"{prefix}{k}.{i}.")
                    elif hasattr(v, 'size'):
                        weights[prefix + k] = v
            flatten_params(dict(self.parameters()))
            mx.savez(str(path / "model.npz"), **weights)

        def load(self, path: Path):
            """Load model from disk."""
            path = Path(path)

            # Load weights from MLX native format
            flat_weights = mx.load(str(path / "model.npz"))
            
            # Unflatten weights back to nested structure
            weights = {}
            for k, v in flat_weights.items():
                parts = k.split(".")
                d = weights
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = v
            
            self.update(weights)

        @property
        def num_parameters(self) -> int:
            """Get total number of parameters."""
            total = 0
            def count_params(params):
                nonlocal total
                for k, v in params.items():
                    if isinstance(v, dict):
                        count_params(v)
                    elif isinstance(v, list):
                        # Handle lists of modules (like layers)
                        for item in v:
                            if hasattr(item, 'parameters'):
                                count_params(dict(item.parameters()))
                    elif hasattr(v, 'size'):
                        total += v.size
            count_params(dict(self.parameters()))
            return total

        def add_lora(self, lora_config: LoRAConfig) -> None:
            """Add LoRA adapters to the model."""
            apply_lora_to_model(self, lora_config)

        def get_lora_params(self) -> tuple[list[mx.array], int, int]:
            """Get LoRA parameters for training."""
            return get_lora_trainable_params(self)

    class MLXGPTModel(nn.Module, BaseModel):
        """GPT-2 style model implementation in MLX."""

        def __init__(self, config: ModelConfig):
            nn.Module.__init__(self)
            BaseModel.__init__(self, config)

            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)  # Token embeddings
            self.wpe = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )  # Position embeddings

            # Transformer blocks (using LayerNorm instead of RMSNorm for GPT)
            self.blocks = []
            for _ in range(config.num_hidden_layers):
                self.blocks.append(self._make_gpt_block(config))

            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            # Output projection
            if config.tie_word_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def _make_gpt_block(self, config: ModelConfig) -> nn.Module:
            """Create a GPT-style transformer block."""
            # GPT uses LayerNorm and different architecture
            # This is simplified - full implementation would be more complex
            return TransformerBlock(config)  # Reusing for now

        def forward(self, input_ids: mx.array, **kwargs) -> mx.array:
            """Forward pass through GPT model."""
            B, L = input_ids.shape

            # Get token and position embeddings
            positions = mx.arange(L)
            token_embeddings = self.wte(input_ids)
            position_embeddings = self.wpe(positions)

            h = token_embeddings + position_embeddings

            # Pass through transformer blocks
            for block in self.blocks:
                h = block(h)

            # Final layer norm
            h = self.ln_f(h)

            # Output projection
            if self.lm_head is not None:
                logits = self.lm_head(h)
            else:
                logits = h @ self.wte.weight.T

            return logits

        def generate(self, input_ids: mx.array, max_length: int = 100, **kwargs) -> mx.array:
            """Generate text (simplified version)."""
            # Similar to Llama generate but without cache
            return MLXLlamaModel.generate(self, input_ids, max_length, **kwargs)

        def save(self, path: Path):
            """Save model."""
            MLXLlamaModel.save(self, path)

        def load(self, path: Path):
            """Load model."""
            MLXLlamaModel.load(self, path)

        @property
        def num_parameters(self) -> int:
            """Get total number of parameters."""
            total = 0
            def count_params(params):
                nonlocal total
                for k, v in params.items():
                    if isinstance(v, dict):
                        count_params(v)
                    elif isinstance(v, list):
                        # Handle lists of modules (like layers)
                        for item in v:
                            if hasattr(item, 'parameters'):
                                count_params(dict(item.parameters()))
                    elif hasattr(v, 'size'):
                        total += v.size
            count_params(dict(self.parameters()))
            return total

        def add_lora(self, lora_config: LoRAConfig) -> None:
            """Add LoRA adapters to the model."""
            apply_lora_to_model(self, lora_config)

        def get_lora_params(self) -> tuple[list[mx.array], int, int]:
            """Get LoRA parameters for training."""
            return get_lora_trainable_params(self)

    # Model registry when MLX is available
    MLX_MODEL_REGISTRY = {
        "llama": MLXLlamaModel,
        "llama2": MLXLlamaModel,
        "llama3": MLXLlamaModel,
        "mistral": MLXLlamaModel,  # Similar architecture
        "gpt2": MLXGPTModel,
        "gpt-j": MLXGPTModel,
        "gpt-neo": MLXGPTModel,
    }

    def get_mlx_model(config: ModelConfig) -> BaseModel:
        """Get MLX model based on config."""
        model_type = config.model_type.lower()

        # Find matching model class
        model_class = None
        for key, cls in MLX_MODEL_REGISTRY.items():
            if key in model_type:
                model_class = cls
                break

        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model_class(config)

else:
    # Stub when MLX is not available
    def get_mlx_model(config: ModelConfig) -> BaseModel:
        """Raise error when MLX is not available."""
        raise ImportError("MLX is not available. Please install MLX or use PyTorch backend.")
