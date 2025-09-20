"""
Model management and caching.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from finetune.backends.device import device_manager
from finetune.core.registry import ModelRegistry
from finetune.models.base import BaseModel
from finetune.models.mlx_loader import MLXModelLoader


class ModelManager:
    """Manages model loading, caching, and registry."""

    def __init__(self, cache_dir: Path | None = None, registry_db: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "finetune" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.registry = ModelRegistry(registry_db)
        self.backend = device_manager.get_optimal_backend()

        # Initialize appropriate loader
        if self.backend.name == "mlx":
            self.loader = MLXModelLoader(cache_dir=self.cache_dir)
        else:
            # Fallback to PyTorch loader
            from finetune.models.torch_loader import PyTorchModelLoader

            self.loader = PyTorchModelLoader(cache_dir=self.cache_dir)

    def load_model(
        self,
        model_id: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str | None = None,
        tokenizer=None,
        **kwargs,
    ) -> BaseModel:
        """Load a model from HuggingFace or local path."""
        logger.info(f"Loading model: {model_id}")

        # Check if it's a local path
        if Path(model_id).exists():
            model = self.loader.load_from_path(Path(model_id), tokenizer=tokenizer)
            model_name = Path(model_id).name
        else:
            # Load from HuggingFace
            model = self.loader.load_from_huggingface(
                model_id,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                tokenizer=tokenizer,
                **kwargs,
            )
            model_name = model_id

        # Register in database
        model_info = self.loader.get_model_info(model)
        model_db_id = self.registry.register_model(
            name=model_name,
            path=str(self.cache_dir / model_name.replace("/", "--")),
            source="huggingface" if "/" in model_id else "local",
            size_gb=model_info["memory_footprint_mb"] / 1024,
            parameters=model_info,
        )

        logger.info(f"Model loaded successfully: {model_info['parameters']:,} parameters")

        return model

    def list_models(self, source: str | None = None) -> list[dict[str, Any]]:
        """List available models."""
        # Get models from registry
        registered_models = self.registry.list_models()

        # Get cached models
        cached_models = self._list_cached_models()

        # Combine and deduplicate
        all_models = {}

        for model in registered_models:
            all_models[model["name"]] = {
                **model,
                "source": model.get("source", "unknown"),
                "cached": model["name"] in cached_models,
                "last_used": model.get("last_used"),
            }

        for model_name in cached_models:
            if model_name not in all_models:
                all_models[model_name] = {
                    "name": model_name,
                    "source": "cache",
                    "cached": True,
                    "path": str(self.cache_dir / model_name),
                }

        # Filter by source if specified
        if source:
            all_models = {k: v for k, v in all_models.items() if v.get("source") == source}

        return list(all_models.values())

    def _list_cached_models(self) -> list[str]:
        """List models in cache directory."""
        models = []

        for path in self.cache_dir.iterdir():
            if path.is_dir() and (path / "config.json").exists():
                models.append(path.name)

        return models

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from cache."""
        model_path = self.cache_dir / model_name.replace("/", "--")

        if model_path.exists():
            shutil.rmtree(model_path)
            logger.info(f"Deleted model: {model_name}")
            return True
        else:
            logger.warning(f"Model not found in cache: {model_name}")
            return False

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """Get detailed information about a model."""
        # Try to load from cache
        model_path = self.cache_dir / model_name.replace("/", "--")

        if not model_path.exists():
            return None

        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
        else:
            config_data = {}

        # Calculate size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

        # Get file list
        files = [str(f.relative_to(model_path)) for f in model_path.iterdir() if f.is_file()]

        return {
            "name": model_name,
            "path": str(model_path),
            "config": config_data,
            "size_gb": total_size / (1024**3),
            "files": files,
            "model_type": config_data.get("model_type", "unknown"),
            "vocab_size": config_data.get("vocab_size"),
            "hidden_size": config_data.get("hidden_size"),
            "num_layers": config_data.get("num_hidden_layers"),
            "num_heads": config_data.get("num_attention_heads"),
        }

    def convert_model(
        self, model_name: str, output_format: str, output_path: Path | None = None
    ) -> Path:
        """Convert model to different format."""
        logger.info(f"Converting {model_name} to {output_format}")

        # Load model
        model = self.load_model(model_name)

        output_path = output_path or self.cache_dir / f"{model_name}_{output_format}"

        if output_format == "gguf":
            # Convert to GGUF format for llama.cpp
            raise NotImplementedError("GGUF conversion not yet implemented")

        elif output_format == "onnx":
            # Convert to ONNX
            raise NotImplementedError("ONNX conversion not yet implemented")

        elif output_format == "coreml":
            # Convert to CoreML
            raise NotImplementedError("CoreML conversion not yet implemented")

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return output_path

    def estimate_memory_usage(
        self,
        model_name: str,
        batch_size: int = 1,
        sequence_length: int = 2048,
        training: bool = True,
    ) -> dict[str, float]:
        """Estimate memory usage for a model."""
        info = self.get_model_info(model_name)

        if not info:
            raise ValueError(f"Model not found: {model_name}")

        # Rough estimation
        vocab_size = info.get("vocab_size", 32000)
        hidden_size = info.get("hidden_size", 4096)
        num_layers = info.get("num_layers", 32)

        # Model parameters (assume FP16)
        param_count = vocab_size * hidden_size * 2  # Embeddings
        param_count += num_layers * hidden_size * hidden_size * 12  # Attention + MLP

        bytes_per_param = 2  # FP16
        model_memory = param_count * bytes_per_param

        # Activations (rough estimate)
        activation_memory = batch_size * sequence_length * hidden_size * num_layers * 4

        # Gradients (same as model for training)
        gradient_memory = model_memory if training else 0

        # Optimizer states (Adam needs 2x model size)
        optimizer_memory = model_memory * 2 if training else 0

        total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory

        return {
            "model_gb": model_memory / (1024**3),
            "activations_gb": activation_memory / (1024**3),
            "gradients_gb": gradient_memory / (1024**3),
            "optimizer_gb": optimizer_memory / (1024**3),
            "total_gb": total_memory / (1024**3),
        }


# Global model manager instance
model_manager = ModelManager()
