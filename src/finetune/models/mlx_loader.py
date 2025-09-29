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

    def __init__(self, cache_dir: Path | None = None):
        logger.info("Entering MLXModelLoader.__init__")
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
        tokenizer_config: dict | None = None,
        **kwargs,
    ) -> BaseModel:
        logger.info("Entering MLXModelLoader.load_from_huggingface")
        """Load a model from HuggingFace Hub with caching."""
        logger.info(f"Loading model from HuggingFace: {model_id}")

        # Only support TinyLlama for now (like working example)
        if "tinyllama" not in model_id.lower():
            raise ValueError(f"Only TinyLlama models are supported for now. Got: {model_id}")

        cache_dir = cache_dir or self.cache_dir

        # Check if model is already cached
        cached_path = self._get_cached_path(model_id, revision, tokenizer_config)
        if cached_path.exists() and (cached_path / "config.json").exists():
            logger.info(f"Loading from cache: {cached_path}")
            return self._load_hf_model_to_mlx(cached_path, tokenizer_config)

        # Download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "Please install huggingface_hub: pip install huggingface_hub"
            ) from e

        logger.info(f"Downloading model {model_id} from HuggingFace Hub...")
        # Download to HF cache location (revision=revision if specified)
        download_kwargs = {"repo_id": model_id, "allow_patterns": ["*.json", "*.safetensors", "tokenizer.model"]}
        if revision:
            download_kwargs["revision"] = revision

        hf_path = Path(snapshot_download(**download_kwargs))

        # Copy to our cache (simplified - just copy the necessary files)
        self._cache_model_files(hf_path, cached_path)
        print("hello I am here")

        # Load directly to MLX format in memory (exactly like working example)
        return self._load_hf_model_to_mlx(cached_path, tokenizer_config)

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

        # 3. Create model and load weights (exactly like working example)
        model = get_mlx_model(config)
        model.load_weights(list(weights.items()))
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

    def _get_cached_path(
        self, model_id: str, revision: str | None, tokenizer_config: dict | None = None
    ) -> Path:
        logger.info("Entering MLXModelLoader._get_cached_path")
        """Get cache path for a model."""
        safe_model_id = model_id.replace("/", "--")
        if revision:
            safe_model_id = f"{safe_model_id}--{revision}"
        return self.cache_dir / safe_model_id

    def _cache_model_files(self, hf_path: Path, cached_path: Path):
        logger.info("Entering MLXModelLoader._cache_model_files")
        """Copy necessary model files to our cache."""
        import shutil

        cached_path.mkdir(parents=True, exist_ok=True)

        # Copy config file
        if (hf_path / "config.json").exists():
            shutil.copy2(hf_path / "config.json", cached_path / "config.json")

        # Copy safetensors files
        import glob
        for safetensors_file in glob.glob(str(hf_path / "*.safetensors")):
            filename = Path(safetensors_file).name
            shutil.copy2(safetensors_file, cached_path / filename)

        # Copy tokenizer files
        for tokenizer_file in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer.model"]:
            if (hf_path / tokenizer_file).exists():
                shutil.copy2(hf_path / tokenizer_file, cached_path / tokenizer_file)

        logger.info(f"Cached model files to {cached_path}")

# Note: Redundant _load_model_from_safetensors removed - _load_hf_model_to_mlx does the same thing

    def _load_pytorch_weights(self, path: Path) -> dict[str, Any]:
        logger.info("Entering MLXModelLoader._load_pytorch_weights")
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

    def _get_name_mapping(self, model_type: str) -> dict[str, str]:
        logger.info("Entering MLXModelLoader._get_name_mapping")
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
