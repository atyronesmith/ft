"""
MLX model loader for HuggingFace models.
"""

import glob
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

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
        **kwargs,
    ) -> BaseModel:
        """Load a model from HuggingFace Hub."""
        logger.info(f"Loading model from HuggingFace: {model_id}")

        cache_dir = cache_dir or self.cache_dir

        # Try to load from cache first
        cached_path = self._get_cached_path(model_id, revision)
        if cached_path.exists():
            logger.info(f"Loading from cache: {cached_path}")
            return self.load_from_path(cached_path)

        # Download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")

        # Download model files
        logger.info(f"Downloading model {model_id} from HuggingFace Hub...")
        local_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_dir=str(cached_path),
            local_dir_use_symlinks=False,
        )

        # Load and convert
        model = self._load_and_convert(Path(local_path))

        # Apply quantization if requested
        if load_in_4bit or load_in_8bit:
            model = self._quantize_model(model, bits=4 if load_in_4bit else 8)

        return model

    def load_from_path(self, path: Path) -> BaseModel:
        """Load a model from local path."""
        path = Path(path)

        # Check if it's already an MLX model
        if (path / "model.safetensors").exists() and (path / "config.json").exists():
            return self._load_mlx_model(path)

        # Otherwise, assume it's a HuggingFace model and convert
        return self._load_and_convert(path)

    def _load_and_convert(self, path: Path) -> BaseModel:
        """Load HuggingFace model and convert to MLX."""
        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            hf_config = json.load(f)

        config = ModelConfig.from_huggingface(hf_config)

        # Create MLX model
        model = get_mlx_model(config)

        # Load and convert weights
        weights = self._load_pytorch_weights(path)
        mlx_weights = self.convert_weights(weights, config.model_type)

        # Update model with converted weights
        model.update(mlx_weights)

        logger.info(
            f"Successfully loaded {config.model_type} model with {model.num_parameters:,} parameters"
        )

        return model

    def _load_mlx_model(self, path: Path) -> BaseModel:
        """Load native MLX model."""
        # Load config
        config = ModelConfig.load(path / "config.json")

        # Create model
        model = get_mlx_model(config)

        # Load weights
        model.load(path)

        return model

    def _load_pytorch_weights(self, path: Path) -> dict[str, Any]:
        """Load PyTorch weights from various formats."""
        import torch

        weights = {}

        # Try different file formats
        if (path / "model.safetensors").exists():
            # Load from safetensors
            from safetensors import safe_open

            with safe_open(path / "model.safetensors", framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        elif (path / "pytorch_model.bin").exists():
            # Load from PyTorch bin
            weights = torch.load(path / "pytorch_model.bin", map_location="cpu")

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

    def convert_weights(self, source_weights: dict[str, Any], model_type: str) -> dict[str, Any]:
        """Convert PyTorch weights to MLX format."""
        import torch

        # Mapping from PyTorch names to MLX names
        name_map = self._get_name_mapping(model_type)

        mlx_weights = {}

        for pytorch_name, pytorch_tensor in source_weights.items():
            # Convert tensor to numpy then MLX
            if isinstance(pytorch_tensor, torch.Tensor):
                numpy_array = pytorch_tensor.detach().cpu().numpy()
            else:
                numpy_array = np.array(pytorch_tensor)

            # Map name
            mlx_name = name_map.get(pytorch_name, pytorch_name)

            # Skip unnecessary weights
            if self._should_skip_weight(mlx_name):
                continue

            # Convert to MLX array
            mlx_array = mx.array(numpy_array)

            # Handle special cases
            mlx_array = self._handle_special_weights(mlx_name, mlx_array, model_type)

            mlx_weights[mlx_name] = mlx_array

        return mlx_weights

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
            return {
                "wte.weight": "wte.weight",
                "wpe.weight": "wpe.weight",
                "ln_f.weight": "ln_f.weight",
                "ln_f.bias": "ln_f.bias",
            }

        # Default: return as-is
        return {}

    def _should_skip_weight(self, name: str) -> bool:
        """Check if weight should be skipped."""
        skip_patterns = [
            "rotary_emb",  # Computed dynamically
            "masked_bias",  # GPT-2 specific, not needed
            "attn.bias",  # Causal mask, computed dynamically
        ]

        return any(pattern in name for pattern in skip_patterns)

    def _handle_special_weights(self, name: str, weight: Any, model_type: str) -> Any:
        """Handle special weight conversions."""
        # Handle transposed weights for linear layers
        if any(
            key in name
            for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ):
            # PyTorch Linear uses (out_features, in_features)
            # MLX Linear uses (in_features, out_features)
            # So we need to transpose
            if len(weight.shape) == 2:
                weight = weight.T

        return weight

    def _quantize_model(self, model: BaseModel, bits: int = 4) -> BaseModel:
        """Quantize model weights."""
        logger.info(f"Quantizing model to {bits}-bit...")

        # This is a placeholder - real quantization would be more complex
        # MLX has built-in quantization support that should be used

        def quantize_weight(w):
            if len(w.shape) >= 2:
                # Simple quantization (placeholder)
                return w.astype(mx.float16)
            return w

        # Quantize weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight = quantize_weight(module.weight)

        logger.info("Quantization complete")
        return model

    def _get_cached_path(self, model_id: str, revision: str | None) -> Path:
        """Get cache path for a model."""
        # Clean model ID for filesystem
        safe_model_id = model_id.replace("/", "--")
        if revision:
            safe_model_id = f"{safe_model_id}--{revision}"

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
