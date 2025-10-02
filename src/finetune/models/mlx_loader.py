"""
MLX model loader for HuggingFace models.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import transformers
from loguru import logger

from finetune.models.base import BaseModel, ModelConfig, ModelLoader
from finetune.models.mlx_models import get_mlx_model

try:
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


class MLXModelLoader(ModelLoader):
    """Loader for MLX models from HuggingFace."""

    def __init__(self, cache_dir: Path):
        logger.info("Entering MLXModelLoader.__init__")
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Please install with: pip install mlx")

        # Cache directory is provided by ModelManager
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_from_huggingface(
        self,
        model_id: str,
        revision: str | None = None,
        cache_dir: Path | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        tokenizer_config: dict | None = None,
        **kwargs,
    ) -> BaseModel:
        logger.info("Entering MLXModelLoader.load_from_huggingface")
        """Load a model from HuggingFace Hub with caching."""
        logger.info(f"Loading model from HuggingFace: {model_id}")

        # Support Llama-based models (TinyLlama, Llama-2, etc.)
        supported_models = ["tinyllama", "llama"]
        if not any(model_type in model_id.lower() for model_type in supported_models):
            raise ValueError(f"Only Llama-based models are supported (TinyLlama, Llama-2, etc.). Got: {model_id}")

        cache_dir = cache_dir or self.cache_dir

        # Use HuggingFace's snapshot_download - it handles caching automatically
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "Please install huggingface_hub: pip install huggingface_hub"
            ) from e

        # snapshot_download automatically uses cache if available, downloads if not
        download_kwargs = {"repo_id": model_id, "allow_patterns": ["*.json", "*.safetensors", "tokenizer.model"]}
        if revision:
            download_kwargs["revision"] = revision

        hf_path = Path(snapshot_download(**download_kwargs))
        logger.info(f"Using model from: {hf_path}")

        # Load directly from HuggingFace cache path
        return self._load_hf_model_to_mlx(hf_path, tokenizer_config)

    def _load_hf_model_to_mlx(self, hf_path: Path, tokenizer_config=None) -> BaseModel:
        logger.info("Entering MLXModelLoader._load_hf_model_to_mlx")
        """Load HF model directly to MLX format (exactly like working example)."""
        if tokenizer_config is None:
            tokenizer_config = {}
        # 1. Load config (exactly like working example)
        with open(hf_path / "config.json") as f:
            config_dict = json.load(f)
            logger.info("config_dict valuesx:")
            for k, v in config_dict.items():
                logger.info(f"  {k}: {v}")

        config = ModelConfig.from_huggingface(config_dict)
        # 2. Load weights from safetensors (exactly like working example)
        import glob
        weight_files = glob.glob(str(hf_path / "*.safetensors"))
        if len(weight_files) == 0:
            raise FileNotFoundError(f"No safetensors found in {hf_path}")

        weights = {}
        assert mx is not None
        for wf in weight_files:
            # Load weights and convert to bfloat16 (like MLX LoRA examples)
            loaded_weights = mx.load(wf)
            # Convert all weights to bfloat16 for efficiency (matching MLX examples)
            converted_weights = {k: v.astype(mx.bfloat16) if hasattr(v, 'astype') else v
                               for k, v in loaded_weights.items()}
            weights.update(converted_weights.items())

        # 3. Create model and load weights with nested structure
        model = get_mlx_model(config)

        # Convert flat HuggingFace weights to nested MLX structure
        nested_weights = self._convert_to_nested_structure(weights, config)

        # Load weights using the model's update method
        model.update(nested_weights)
        mx.eval(model.parameters())
        logger.info("=" * 30 + " Model Parameters " + "=" * 28)

        def print_params(params, prefix=""):
            for name, value in params.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(value, dict):
                    print_params(value, full_name)
                elif hasattr(value, "shape"):
                    shape_str = str(value.shape)
                    dtype_str = str(value.dtype)
                    log_line = f"{full_name:<50} | Shape: {shape_str:<20} | Dtype: {dtype_str}"
                    logger.info(log_line)

        print_params(model.parameters())
        logger.info("=" * 80)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_path, **tokenizer_config
        )

        return model, tokenizer, config

    def load_from_path(self, path: Path, tokenizer_config: dict | None = None) -> tuple:
        logger.info("Entering MLXModelLoader.load_from_path")
        """Load a model from local path using simplified approach."""
        path = Path(path)

        # Check if this is a HuggingFace model directory with safetensors
        if (path / "config.json").exists():
            import glob
            safetensors_files = glob.glob(str(path / "*.safetensors"))
            if safetensors_files:
                # Use simplified loading (same as _load_hf_model_to_mlx)
                return self._load_hf_model_to_mlx(path, tokenizer_config)

        raise FileNotFoundError(f"No valid model found at {path}")

# Removed _get_cached_path and _cache_model_files - we now use HuggingFace cache directly

    def _load_pytorch_weights(self, path: Path) -> dict[str, Any]:
        """Load weights from Hugging Face standard formats.

        Prioritizes Safetensors (current standard) with fallback to PyTorch .bin format.
        """
        import torch
        from loguru import logger

        logger.info("Entering MLXModelLoader._load_pytorch_weights")

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

    def convert_weights(self, source_weights: dict, config: ModelConfig = None) -> "dict[str, mx.array]":
        logger.info("Entering MLXModelLoader.convert_weights")
        """Convert weights (not used in simplified approach - mx.load() handles conversion)."""
        # This method is required by the abstract base class but not used
        # in our simplified approach since mx.load() handles conversion automatically
        return source_weights

    def _unflatten_weights(self, weights: "dict[str, mx.array]") -> dict:
        logger.info("Entering MLXModelLoader._unflatten_weights")
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

    def _convert_to_nested_structure(self, weights: dict, config: ModelConfig) -> dict:
        """Convert flat HuggingFace weights to nested MLX structure."""
        nested_weights = {}

        for name, weight in weights.items():
            if name.startswith("model."):
                # HuggingFace: model.embed_tokens.weight → MLX: model.embed_tokens.weight
                # HuggingFace: model.layers.X.Y.Z → MLX: model.layers.X.Y.Z
                # Remove the "model." prefix from HuggingFace and add to nested structure
                model_weight_name = name[6:]  # Remove "model." prefix

                if "model" not in nested_weights:
                    nested_weights["model"] = {}

                # Split the path and create nested structure
                parts = model_weight_name.split(".")
                current = nested_weights["model"]

                for part in parts[:-1]:
                    if part not in current:
                        # Handle layer indices properly (convert to int for layers)
                        if part.isdigit() and "layers" in parts:
                            current["layers"] = current.get("layers", {})
                            current = current["layers"]
                            if int(part) not in current:
                                current[int(part)] = {}
                            current = current[int(part)]
                        else:
                            current[part] = {}
                            current = current[part]
                    else:
                        if part.isdigit() and "layers" in parts:
                            current = current["layers"][int(part)]
                        else:
                            current = current[part]

                current[parts[-1]] = weight

            elif name.startswith("lm_head."):
                # HuggingFace: lm_head.weight → MLX: lm_head.weight (top level)
                nested_weights[name] = weight
            else:
                # Other weights (rare case)
                nested_weights[name] = weight

        return nested_weights

    def _should_skip_weight(
        self, name: str, config: ModelConfig, tokenizer_provided: bool = False
    ) -> bool:
        logger.info("Entering MLXModelLoader._should_skip_weight")
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
        logger.info("Entering MLXModelLoader._handle_special_weights")
        """Handle special weight conversions."""
        # FIXED: Removed transpose operation that was causing shape mismatch
        # Modern safetensors weights are already in the correct shape (out_features, in_features)
        # which is what MLX Linear layers expect. No transpose needed.
        return weight

    def _quantize_model(self, model: BaseModel, bits: int) -> BaseModel:
        logger.info("Entering MLXModelLoader._quantize_model")
        """Apply quantization to the model."""
        nn.QuantizedLinear.quantize_module(model, bits)


    def save_model(self, model: BaseModel, path: Path):
        logger.info("Entering MLXModelLoader.save_model")
        """Save MLX model to disk."""
        model.save(path)
        logger.info(f"Model saved to {path}")

    def get_model_info(self, model: BaseModel) -> dict[str, Any]:
        logger.info("Entering MLXModelLoader.get_model_info")
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
