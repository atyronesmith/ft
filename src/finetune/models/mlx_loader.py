"""
MLX model loader for HuggingFace models.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None
    tree_map = None

if TYPE_CHECKING and MLX_AVAILABLE:
    # For type hints only
    import mlx.core as mx

from finetune.models.base import BaseModel, ModelConfig, ModelLoader
from finetune.models.mlx_models import get_mlx_model


class MLXModelLoader(ModelLoader):
    """Loader for MLX models from HuggingFace."""

    def __init__(self, cache_dir: Path | None = None):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Please install with: pip install mlx")

        self.cache_dir = cache_dir or Path.home() / ".cache" / "finetune" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_from_huggingface(
        self,
        model_id: str,
        revision: str | None = None,
        cache_dir: Path | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        tokenizer=None,
        **kwargs,
    ) -> BaseModel:
        """Load a model from HuggingFace Hub."""
        logger.info(f"Loading model from HuggingFace: {model_id}")

        cache_dir = cache_dir or self.cache_dir

        # Try to load from cache first
        cached_path = self._get_cached_path(model_id, revision, tokenizer)
        if cached_path.exists():
            logger.info(f"Loading from cache: {cached_path}")
            # Validate cached model is compatible with tokenizer
            if tokenizer:
                try:
                    with open(cached_path / "config.json", "r") as f:
                        cached_config = json.load(f)
                    cached_vocab_size = cached_config.get("vocab_size", 0)
                    if cached_vocab_size != len(tokenizer):
                        logger.warning(f"Cache vocab size mismatch: {cached_vocab_size} != {len(tokenizer)}, rebuilding...")
                        # Clear incompatible cache and rebuild
                        import shutil
                        shutil.rmtree(cached_path)
                    else:
                        return self.load_from_path(cached_path, tokenizer)
                except Exception as e:
                    logger.warning(f"Cache validation failed: {e}, rebuilding...")
                    import shutil
                    shutil.rmtree(cached_path)
            else:
                return self.load_from_path(cached_path, tokenizer)

        # Download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")

        # If not cached, download, convert, and cache
        else:
            logger.info(f"Downloading model {model_id} from HuggingFace Hub...")
            hf_path = snapshot_download(repo_id=model_id, local_dir=cached_path)
            return self._load_and_convert(Path(hf_path), cached_path, tokenizer)

    def _load_and_convert(self, hf_path: Path, model_path: Path, tokenizer: Any):
        """Load from HF, convert to MLX, and save."""
        # 1. Load the model using the transformers library, ensuring we use its
        # native floating-point precision to prevent weight corruption.
        logger.info("Loading full HuggingFace model to synchronize token embeddings...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(hf_path), trust_remote_code=True, dtype="auto"
        )

        # 2. Resize embedding matrix if tokenizer has extra tokens
        if tokenizer and len(tokenizer) > hf_model.config.vocab_size:
            logger.info(f"Resizing embedding matrix from {hf_model.config.vocab_size} to {len(tokenizer)}")
            hf_model.resize_token_embeddings(len(tokenizer))

        # 3. Convert and save weights
        self._convert_and_save_weights(hf_model, model_path)

        # 4. Load the converted model
        return self.load_from_path(model_path, tokenizer=tokenizer)

    def _convert_and_save_weights(self, hf_model: Any, model_path: Path):
        """Convert and save PyTorch weights to MLX-compatible NPZ format."""
        # Ensure the model path exists
        model_path.mkdir(parents=True, exist_ok=True)

        # Get PyTorch state dict and MLX model config
        source_weights = hf_model.state_dict()
        config = ModelConfig.from_huggingface(hf_model.config.to_dict())

        # Convert weights
        mlx_weights = self.convert_weights(source_weights, config)

        # Validate we have weights to save
        if not mlx_weights:
            raise ValueError("No weights were converted - this indicates a critical conversion failure")

        logger.info(f"Converted {len(mlx_weights)} weight tensors for saving")

        # Save weights and config
        np.savez(str(model_path / "weights.npz"), **mlx_weights)
        with open(model_path / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=4)

        logger.info(f"Saved model weights and config to {model_path}")

    def load_from_path(self, path: Path, tokenizer=None) -> BaseModel:
        """Load a model from local path."""
        path = Path(path)

        # Check if it's our converted MLX model format
        if (path / "weights.npz").exists() and (path / "config.json").exists():
            model = self._load_model_from_npz(path)
            mx.eval(model.parameters())  # Ensure weights are loaded
            return model

        # Otherwise, assume it's a raw HuggingFace model and convert it
        return self._load_and_convert(path, path, tokenizer)

    def _load_model_from_npz(self, model_path: Path) -> BaseModel:
        """Load model weights from NPZ file and config."""
        # Load config
        with open(model_path / "config.json", "r") as f:
            config = ModelConfig(**json.load(f))

        # Create MLX model
        model = get_mlx_model(config)

        # Load and set weights
        npz_weights = np.load(str(model_path / "weights.npz"))
        if len(npz_weights) == 0:
            raise ValueError(f"No weights found in {model_path / 'weights.npz'}")

        weights = {k: mx.array(v) for k, v in npz_weights.items()}
        nested_weights = self._unflatten_weights(weights)

        logger.info(f"Loading {len(weights)} weight tensors from NPZ file")
        model.update(nested_weights)

        # Critical: Ensure all weights are properly evaluated and loaded
        mx.eval(model.parameters())

        # Validate model has parameters using MLX flatten_params
        try:
            # Import flatten_params from mlx_models
            from finetune.models.mlx_models import flatten_params

            flat_params = flatten_params(model.parameters())
            param_count = sum(p.size for p in flat_params.values() if hasattr(p, 'size'))

            if param_count == 0:
                raise ValueError("Model loaded with 0 parameters - weight loading failed")

            logger.info(f"Model loaded successfully with {param_count:,} parameters")
        except Exception as e:
            logger.warning(f"Could not count parameters: {e}")
            # Don't fail loading, just warn
        return model

    def _load_pytorch_weights(self, path: Path) -> dict[str, Any]:
        """Load weights from Hugging Face standard formats.

        Prioritizes Safetensors (current standard) with fallback to PyTorch .bin format.
        """
        import torch
        from loguru import logger

        weights = {}

        # Priority 1: Safetensors (current Hugging Face standard)
        if (path / "model.safetensors").exists():
            logger.info("Loading weights from Safetensors format (current standard)")
            from safetensors import safe_open

            with safe_open(path / "model.safetensors", framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

            logger.info(f"Loaded {len(weights)} parameters from Safetensors")

        # Priority 2: PyTorch .bin (legacy format for compatibility)
        elif (path / "pytorch_model.bin").exists():
            logger.info("Loading weights from PyTorch .bin format (legacy compatibility)")
            weights = torch.load(path / "pytorch_model.bin", map_location="cpu")
            logger.info(f"Loaded {len(weights)} parameters from PyTorch .bin")

        elif (path / "model.pt").exists():
            # Load from .pt file
            weights = torch.load(path / "model.pt", map_location="cpu")

        else:
            # Try loading sharded model
            import glob

            pattern = str(path / "pytorch_model-*.bin")
            shards = glob.glob(pattern)

            if shards:
                for shard_path in shards:
                    shard_weights = torch.load(shard_path, map_location="cpu")
                    weights.update(shard_weights)
            else:
                raise FileNotFoundError(f"No model weights found in {path}")

        return weights

    def convert_weights(
        self, source_weights: dict, config: ModelConfig
    ) -> dict[str, mx.array]:
        """
        Convert PyTorch weights to MLX.

        This version uses a simple string replacement, which is robust under the
        assumption that the layer names in the MLX model directly correspond
        to the layer names in the Hugging Face model, with the exception of a
        'model.' prefix on the latter.
        """
        mlx_weights = {}
        for pt_name, pt_tensor in source_weights.items():
            mlx_name = pt_name
            if mlx_name.startswith("model."):
                mlx_name = mlx_name.replace("model.", "", 1)

            mlx_weights[mlx_name] = self._to_mlx(pt_tensor)

        return mlx_weights

    def _to_mlx(self, arr: torch.Tensor) -> mx.array:
        """Convert a PyTorch tensor to an MLX array."""
        # NumPy does not support bfloat16, so we must cast to a supported type first.
        return mx.array(arr.to(torch.float32).numpy())

    def _unflatten_weights(self, weights: dict[str, mx.array]) -> dict:
        """Helper to convert flat weight dict to nested dict for MLX."""
        nested = {}
        for key, value in weights.items():
            parts = key.split(".")
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return nested

    def _get_name_mapping(self, model_type: str) -> dict[str, str]:
        """Get weight name mapping for different model types."""

        if "llama" in model_type.lower():
            return {
                # Embeddings
                "model.embed_tokens.weight": "embed_tokens.weight",
                "lm_head.weight": "lm_head.weight",
                "model.norm.weight": "norm.weight",
                # Layers - using string replacement for layer indices
                # This is simplified - full implementation would handle all cases
            }

        elif "gpt" in model_type.lower():
            # GPT-2 / DialoGPT parameter mapping (matches actual PyTorch structure)
            mapping = {
                # Embeddings
                "transformer.wte.weight": "wte.weight",
                "transformer.wpe.weight": "wpe.weight",
                # Final layer norm
                "transformer.ln_f.weight": "ln_f.weight",
                "transformer.ln_f.bias": "ln_f.bias",
            }

            # Add layer-specific mappings for transformer blocks
            for i in range(50):  # Support up to 50 layers
                layer_prefix = f"transformer.h.{i}"
                mlx_prefix = f"layers.{i}"  # Use layers.0, layers.1, etc. format for MLX

                # Attention layers (keep GPT-2 naming)
                mapping[f"{layer_prefix}.attn.c_attn.weight"] = f"{mlx_prefix}.attn.c_attn.weight"
                mapping[f"{layer_prefix}.attn.c_attn.bias"] = f"{mlx_prefix}.attn.c_attn.bias"
                mapping[f"{layer_prefix}.attn.c_proj.weight"] = f"{mlx_prefix}.attn.c_proj.weight"
                mapping[f"{layer_prefix}.attn.c_proj.bias"] = f"{mlx_prefix}.attn.c_proj.bias"

                # MLP layers (keep GPT-2 naming)
                mapping[f"{layer_prefix}.mlp.c_fc.weight"] = f"{mlx_prefix}.mlp.c_fc.weight"
                mapping[f"{layer_prefix}.mlp.c_fc.bias"] = f"{mlx_prefix}.mlp.c_fc.bias"
                mapping[f"{layer_prefix}.mlp.c_proj.weight"] = f"{mlx_prefix}.mlp.c_proj.weight"
                mapping[f"{layer_prefix}.mlp.c_proj.bias"] = f"{mlx_prefix}.mlp.c_proj.bias"

                # Layer norms (keep GPT-2 naming)
                mapping[f"{layer_prefix}.ln_1.weight"] = f"{mlx_prefix}.ln_1.weight"
                mapping[f"{layer_prefix}.ln_1.bias"] = f"{mlx_prefix}.ln_1.bias"
                mapping[f"{layer_prefix}.ln_2.weight"] = f"{mlx_prefix}.ln_2.weight"
                mapping[f"{layer_prefix}.ln_2.bias"] = f"{mlx_prefix}.ln_2.bias"

            return mapping

        # Default: return as-is
        return {}

    def _should_skip_weight(self, name: str, config: ModelConfig, tokenizer_provided: bool = False) -> bool:
        """Check if weight should be skipped."""
        # Skip lm_head if it's tied to the embedding layer's weights
        if "lm_head.weight" in name and config.tie_word_embeddings and not tokenizer_provided:
            logger.debug(f"Skipping tied weight: {name}")
            return True

        # General skip patterns
        skip_patterns = [
            "rotary_emb",  # Computed dynamically
            "masked_bias",  # Not needed
            "attn.bias",  # Causal mask, computed dynamically
        ]

        return any(pattern in name for pattern in skip_patterns)

    def _handle_special_weights(self, name: str, weight: Any, model_type: str) -> Any:
        """Handle special weight conversions."""
        # FIXED: Removed transpose operation that was causing shape mismatch
        # Modern safetensors weights are already in the correct shape (out_features, in_features)
        # which is what MLX Linear layers expect. No transpose needed.
        return weight

    def _quantize_model(self, model: BaseModel, bits: int) -> BaseModel:
        """Apply quantization to the model."""
        nn.QuantizedLinear.quantize_module(model, bits)

    def _get_cached_path(self, model_id: str, revision: str | None, tokenizer: Any | None = None) -> Path: # CHANGED: Added tokenizer
        """Get cache path for a model, making it unique for different vocab sizes."""
        safe_model_id = model_id.replace("/", "--")
        if revision:
            safe_model_id = f"{safe_model_id}--{revision}"

        # CHANGED: This block is new. It makes the cache path unique to the vocab size.
        if tokenizer:
            vocab_size = len(tokenizer)
            safe_model_id = f"{safe_model_id}--vocab{vocab_size}"

        return self.cache_dir / safe_model_id

    def save_model(self, model: BaseModel, path: Path):
        """Save MLX model to disk."""
        model.save(path)
        logger.info(f"Model saved to {path}")

    def get_model_info(self, model: BaseModel) -> dict[str, Any]:
        """Get information about a model."""
        return {
            "type": model.config.model_type,
            "parameters": model.num_parameters,
            "memory_footprint_mb": model.memory_footprint / (1024**2),
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
        }
