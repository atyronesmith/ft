"""
MLX model implementations for common architectures.
"""

import math
from pathlib import Path

try:
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

    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full_key = f"{prefix}{k}." if prefix else f"{k}."
            if isinstance(v, dict):
                flat.update(flatten_params(v, full_key))
            elif hasattr(v, "shape"):
                flat[full_key.rstrip(".")] = v
            # else: skip non-array values
        return flat

    class RMSNorm(nn.Module):
        """RMSNorm layer for MLX - HuggingFace compatible implementation."""

        def __init__(self, dims: int, eps: float = 1e-6):
            super().__init__()
            self.weight = mx.ones((dims,))
            self.eps = eps

        def __call__(self, x):
            # Pure bfloat16 RMSNorm for memory efficiency (MLX examples pattern)
            # Keep everything in bfloat16 to avoid unnecessary conversions

            # Compute variance: mean of squares along last dimension
            # HF: hidden_states.pow(2).mean(-1, keepdim=True)
            variance = mx.mean(mx.square(x), axis=-1, keepdims=True)

            # Normalize: x * rsqrt(variance + eps)
            # HF: hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            normalized = x * mx.rsqrt(variance + self.eps)

            # Apply weight in same dtype (bfloat16)
            return self.weight * normalized

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
            offset: int = 0,  # Add offset parameter
        ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
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
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

            # Handle cache for inference
            if cache is not None:
                key_cache, value_cache = cache
                keys = mx.concatenate([key_cache, keys], axis=2)
                values = mx.concatenate([value_cache, values], axis=2)

            cache = (keys, values)

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

            return output, cache

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
            offset: int = 0,
        ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
            # Self-attention with residual
            r, cache = self.self_attn(self.input_layernorm(x), mask, cache, offset=offset)
            h = x + r

            # MLP with residual
            r = self.mlp(self.post_attention_layernorm(h))
            out = h + r

            return out, cache

    class LlamaModel(nn.Module):
        """Inner nested model matching MLX structure exactly.

        This class contains the core transformer components (layers, embeddings, norm)
        and matches the nested structure used in MLX examples.
        """

        def __init__(self, config: ModelConfig):
            super().__init__()

            # Core transformer components (nested under 'model' in MLX structure)
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            # Store config for reference
            self.num_hidden_layers = config.num_hidden_layers
            self.vocab_size = config.vocab_size

        def __call__(
            self,
            input_ids: mx.array,
            mask: mx.array | None = None,
            cache: list | None = None,
            offset: int = 0,
        ) -> tuple[mx.array, list]:
            """Forward pass through the nested model components."""
            # Token embeddings
            h = self.embed_tokens(input_ids)

            # Create causal mask if not provided
            if mask is None and h.shape[1] > 1:
                mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
                mask = mask.astype(h.dtype)

            # Pass through transformer layers
            if cache is None or len(cache) == 0:
                cache = [None] * len(self.layers)

            new_cache = []
            for i, layer in enumerate(self.layers):
                h, c = layer(h, mask, cache[i], offset=offset)
                new_cache.append(c)

            # Final norm
            h = self.norm(h)

            return h, new_cache

        def update(self, parameters: dict):
            """Custom update method to handle list-based layers in nested model."""
            # Separate layer parameters from top-level parameters
            layer_params = {}
            top_level_params = {}

            for key, value in parameters.items():
                if key == "layers":
                    layer_params = value
                else:
                    top_level_params[key] = value

            # Update top-level parameters (embed_tokens, norm) using the parent method
            if top_level_params:
                super().update(top_level_params)

            # Update each layer individually from the nested dictionary or list
            if layer_params:
                if isinstance(layer_params, dict):
                    # Normal case: dict mapping layer indices to weights
                    for i, layer_weights in layer_params.items():
                        try:
                            layer_index = int(i)
                            if layer_index < len(self.layers):
                                self.layers[layer_index].update(layer_weights)
                        except ValueError:
                            # Skip non-numeric keys (e.g., 'layers' metadata)
                            continue
                elif isinstance(layer_params, list):
                    # After LoRA: list of layer weight dicts
                    for layer_index, layer_weights in enumerate(layer_params):
                        if layer_index < len(self.layers) and isinstance(layer_weights, dict):
                            self.layers[layer_index].update(layer_weights)

        def parameters(self):
            """Return all parameters for the nested model, including layers."""
            # Start with parameters from the base Module (embed_tokens, norm)
            params = dict(super().parameters())

            # Manually add the parameters from each layer in the list
            for i, layer in enumerate(self.layers):
                params[f"layers.{i}"] = dict(layer.parameters())

            return params

    class MLXLlamaModel(nn.Module, BaseModel):
        """Llama model implementation in MLX with nested structure.

        This matches the MLX example structure exactly:
        - model: LlamaModel (contains layers, embed_tokens, norm)
        - lm_head: nn.Linear (at top level)

        Parameter names will be: model.layers.X.Y.Z and lm_head.weight
        """

        def __init__(self, config: ModelConfig):
            nn.Module.__init__(self)
            BaseModel.__init__(self, config)

            # Create nested model structure (MLX pattern)
            self.model = LlamaModel(config)

            # lm_head stays at top level (MLX pattern)
            if config.tie_word_embeddings:
                self.lm_head = None  # Will use embed_tokens.weight.T
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def forward(
            self,
            input_ids: mx.array,
            mask: mx.array | None = None,
            cache: list | None = None,
            offset: int = 0,
        ) -> tuple[mx.array, list]:
            """Forward pass through the model using nested structure."""
            # Pass through nested model components
            h, cache = self.model(input_ids, mask, cache, offset)

            # Output projection (at top level)
            if self.lm_head is not None:
                logits = self.lm_head(h)
            else:
                logits = h @ self.model.embed_tokens.weight.T

            return logits, cache

        def __call__(self, input_ids: mx.array, cache: list | None = None, **kwargs) -> tuple[mx.array, list]:
            """Call method to make the model callable like MLX examples."""
            return self.forward(input_ids, cache=cache, **kwargs)

        def generate(
            self,
            input_ids: mx.array,
            max_length: int = 100,
            temperature: float = 1.0,
            top_p: float = 1.0,
            eos_token_id: int | None = None,
            **kwargs,
        ) -> mx.array:
            """Generate text from the model."""
            cache = []
            current_offset = input_ids.shape[1]

            # Initial forward pass
            logits, cache = self.forward(input_ids, cache=cache, offset=0)

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
                next_token = mx.random.categorical(next_logits)
                next_token = next_token.reshape(1, 1)

                generated.append(next_token)

                # Stop if EOS is generated
                if eos_token_id is not None and int(next_token[0, 0]) == int(eos_token_id):
                    break

                # Forward pass with cache using the correct offset
                logits, cache = self.forward(next_token, cache=cache, offset=current_offset)
                current_offset += 1

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
                            if hasattr(item, "parameters"):
                                flatten_params(dict(item.parameters()), f"{prefix}{k}.{i}.")
                    elif hasattr(v, "size"):
                        weights[prefix + k] = v

            flatten_params(dict(self.parameters()))
            mx.save_safetensors(str(path / "model.safetensors"), weights, metadata={"format": "mlx"})

        def load(self, path: Path):
            """Load model from disk."""
            path = Path(path)

            # Load weights from MLX native format
            flat_weights = mx.load(str(path / "model.safetensors"))

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
            try:
                # Use the same method as the loader for consistency
                flat_params = flatten_params(self.parameters())
                return sum(p.size for p in flat_params.values() if hasattr(p, "size"))
            except Exception:
                # Fallback to simple count if flattening fails
                params = self.parameters()
                return sum(v.size for v in params.values() if hasattr(v, "size"))

        def add_lora(self, lora_config: LoRAConfig) -> None:
            """Add LoRA adapters to the model."""
            apply_lora_to_model(self, lora_config)

        def get_lora_params(self) -> tuple[list[mx.array], int, int]:
            """Get LoRA parameters for training."""
            return get_lora_trainable_params(self)

        def update(self, parameters: dict):
            """Custom update method to handle nested model structure."""
            # Separate model parameters from top-level parameters
            model_params = {}
            top_level_params = {}

            for key, value in parameters.items():
                if key == "model":
                    model_params = value
                else:
                    top_level_params[key] = value

            # Update top-level parameters (like lm_head) using the parent method
            if top_level_params:
                super().update(top_level_params)

            # Update nested model parameters
            if model_params:
                self.model.update(model_params)

        def parameters(self):
            """Return all parameters with nested structure (MLX-compatible naming)."""
            # Get top-level parameters (like lm_head)
            params = dict(super().parameters())

            # Add nested model parameters with 'model.' prefix
            params["model"] = self.model.parameters()

            return params

        def named_parameters(self):
            return [(name, param) for name, param in self.parameters().items()]

        # Backward compatibility properties
        @property
        def layers(self):
            """Backward compatibility: redirect to nested model layers."""
            return self.model.layers

        @property
        def embed_tokens(self):
            """Backward compatibility: redirect to nested model embed_tokens."""
            return self.model.embed_tokens

        @property
        def norm(self):
            """Backward compatibility: redirect to nested model norm."""
            return self.model.norm

    class GPTTransformerBlock(nn.Module):
        """GPT-2 style transformer block with proper parameter naming."""

        def __init__(self, config: ModelConfig):
            super().__init__()

            # Layer norms (GPT uses LayerNorm, not RMSNorm)
            self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            # Attention with combined qkv projection (like GPT-2)
            self.attn = GPTAttention(config)

            # MLP
            self.mlp = GPTMLP(config)

        def __call__(self, x: mx.array) -> mx.array:
            # GPT-2 style: prenorm
            h = x + self.attn(self.ln_1(x))
            h = h + self.mlp(self.ln_2(h))
            return h

    class GPTAttention(nn.Module):
        """GPT-2 style attention with combined qkv projection."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads

            # Combined q,k,v projection (like GPT-2 c_attn)
            self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
            self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        def __call__(self, x: mx.array) -> mx.array:
            B, L, D = x.shape

            # Project to q, k, v
            qkv = self.c_attn(x)
            q, k, v = mx.split(qkv, 3, axis=-1)

            # Reshape for multi-head attention
            q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            # Attention
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

            # Causal mask
            mask = mx.triu(mx.ones((L, L)), k=1) * -1e9
            scores = scores + mask

            attn = mx.softmax(scores, axis=-1)
            out = mx.matmul(attn, v)

            # Reshape and project
            out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
            return self.c_proj(out)

    class GPTMLP(nn.Module):
        """GPT-2 style MLP."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
            self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

        def __call__(self, x: mx.array) -> mx.array:
            return self.c_proj(nn.gelu(self.c_fc(x)))

    class MLXGPTModel(nn.Module, BaseModel):
        """GPT-2 style model implementation in MLX."""

        def __init__(self, config: ModelConfig):
            nn.Module.__init__(self)
            BaseModel.__init__(self, config)

            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)  # Token embeddings
            self.wpe = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )  # Position embeddings

            # Create transformer blocks using the same pattern as MLXLlamaModel
            # Use a simple list - MLX will handle parameter registration automatically
            self.layers = [GPTTransformerBlock(config) for _ in range(config.num_hidden_layers)]

            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            # Output projection
            if config.tie_word_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def _make_gpt_block(self, config: ModelConfig) -> nn.Module:
            """Create a GPT-style transformer block."""
            return GPTTransformerBlock(config)

        def forward(self, input_ids: mx.array, offset: int = 0, **kwargs) -> mx.array:
            """Forward pass through GPT model."""
            B, L = input_ids.shape

            # Get token and position embeddings
            positions = mx.arange(offset, offset + L)
            token_embeddings = self.wte(input_ids)
            position_embeddings = self.wpe(positions)

            h = token_embeddings + position_embeddings

            # Pass through transformer blocks
            for layer in self.layers:
                h = layer(h)

            # Final layer norm
            h = self.ln_f(h)

            # Output projection
            if self.lm_head is not None:
                logits = self.lm_head(h)
            else:
                logits = h @ self.wte.weight.T

            return logits

        def update(self, parameters: dict):
            """Custom update method to handle list-based layers."""
            # Handle top-level parameters normally
            top_level_params = {}
            layer_params = {}

            for key, value in parameters.items():
                if key.startswith("layers."):
                    # Extract layer index and parameter path
                    parts = key.split(".", 2)  # ['layers', '0', 'attn.c_attn.weight']
                    layer_idx = int(parts[1])
                    param_path = parts[2]

                    if layer_idx not in layer_params:
                        layer_params[layer_idx] = {}

                    # Rebuild nested structure for this layer
                    current = layer_params[layer_idx]
                    path_parts = param_path.split(".")
                    for part in path_parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[path_parts[-1]] = value
                else:
                    # Handle nested top-level parameters
                    if "." in key:
                        parts = key.split(".")
                        current = top_level_params
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        current[parts[-1]] = value
                    else:
                        top_level_params[key] = value

            # Update top-level parameters using parent method
            super().update(top_level_params)

            # Update each layer individually
            for layer_idx, layer_weights in layer_params.items():
                if layer_idx < len(self.layers):
                    self.layers[layer_idx].update(layer_weights)

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
            try:
                # Use the same method as the loader for consistency
                flat_params = flatten_params(self.parameters())
                return sum(p.size for p in flat_params.values() if hasattr(p, "size"))
            except Exception:
                # Fallback to recursive count if flattening fails
                total = 0

                def count_params(params):
                    nonlocal total
                    for v in params.items():
                        if isinstance(v, dict):
                            count_params(v)
                        elif isinstance(v, list):
                            # Handle lists of modules (like layers)
                            for item in v:
                                if hasattr(item, "parameters"):
                                    count_params(dict(item.parameters()))
                        elif hasattr(v, "size"):
                            total += v.size

                count_params(dict(self.parameters()))
                return total

        def add_lora(self, lora_config: LoRAConfig) -> None:
            """Add LoRA adapters to the model."""
            apply_lora_to_model(self, lora_config)

        def get_lora_params(self) -> tuple[list[mx.array], int, int]:
            """Get LoRA parameters for training."""
            return get_lora_trainable_params(self)

        def named_parameters(self):
            return [(name, param) for name, param in self.parameters().items()]

        def parameters(self):
            params = flatten_params(super().parameters())
            if hasattr(self, "layers") and isinstance(self.layers, list):
                for i, layer in enumerate(self.layers):
                    if isinstance(layer, nn.Module):
                        layer_params = flatten_params(layer.parameters())
                        for k, v in layer_params.items():
                            params[f"layers.{i}.{k}"] = v
            return params

    # Model registry for Hugging Face transformer architectures
    # Focused on decoder-only models (most popular for fine-tuning)
    MLX_MODEL_REGISTRY = {
        # Llama family (Meta's models + derivatives)
        "llama": MLXLlamaModel,
        "llama2": MLXLlamaModel,
        "llama3": MLXLlamaModel,
        "code_llama": MLXLlamaModel,
        # Mistral family (Mistral AI)
        "mistral": MLXLlamaModel,  # Same architecture as Llama
        "mixtral": MLXLlamaModel,  # MoE version, same base structure
        # Google models
        "gemma": MLXLlamaModel,  # Similar to Llama architecture
        # GPT family (OpenAI + derivatives)
        "gpt2": MLXGPTModel,
        "gpt-j": MLXGPTModel,
        "gpt-neo": MLXGPTModel,
        "gpt-neox": MLXGPTModel,
        # Microsoft models
        "phi": MLXGPTModel,  # Phi-2, Phi-3 series
        "dialoGPT": MLXGPTModel,  # Our current test case
        # Other popular decoder-only models
        "falcon": MLXLlamaModel,  # Similar architecture
        "qwen": MLXLlamaModel,  # Alibaba's model, similar architecture
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
