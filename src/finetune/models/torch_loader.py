"""
PyTorch model loader for fallback support.
"""

from pathlib import Path
from typing import Any

from loguru import logger

try:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from finetune.models.base import BaseModel, ModelConfig, ModelLoader


class PyTorchModel(BaseModel):
    """Wrapper for PyTorch/HuggingFace models."""

    def __init__(self, model: Any, config: ModelConfig, tokenizer: Any = None):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def forward(self, input_ids: Any, **kwargs) -> Any:
        """Forward pass through the model."""
        outputs = self.model(input_ids=input_ids, **kwargs)
        return outputs.logits if hasattr(outputs, "logits") else outputs

    def generate(self, input_ids: Any, max_length: int = 100, **kwargs) -> Any:
        """Generate text from the model."""
        return self.model.generate(input_ids=input_ids, max_length=max_length, **kwargs)

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / "config.json")

        # Save model
        self.model.save_pretrained(path)

        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)

    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        if (path / "tokenizer_config.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(path)

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def memory_footprint(self) -> int:
        """Estimate memory footprint in bytes."""
        # Calculate based on parameter dtypes
        total_bytes = 0
        for p in self.model.parameters():
            if p.dtype == torch.float32:
                bytes_per_param = 4
            elif p.dtype == torch.float16 or p.dtype == torch.bfloat16:
                bytes_per_param = 2
            elif p.dtype == torch.int8:
                bytes_per_param = 1
            else:
                bytes_per_param = 4  # Default

            total_bytes += p.numel() * bytes_per_param

        return total_bytes


class PyTorchModelLoader(ModelLoader):
    """Loader for PyTorch/HuggingFace models."""

    def __init__(self, cache_dir: Path):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Please install with: pip install torch transformers"
            )

        # Cache directory is provided by ModelManager
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

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
        logger.info(f"Loading model from HuggingFace: {model_id} (PyTorch backend)")

        cache_dir = cache_dir or self.cache_dir

        # Configure quantization
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            if not TORCH_AVAILABLE:
                logger.warning("Quantization requested but bitsandbytes not available")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

        # Load config
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, **kwargs
        )

        # Load model
        model_kwargs = {
            "revision": revision,
            "cache_dir": cache_dir,
            "torch_dtype": torch.float16 if self.device.type != "cpu" else torch.float32,
            "low_cpu_mem_usage": True,
            **kwargs,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Move to device if not using device_map
        if "device_map" not in model_kwargs and self.device.type != "cpu":
            model = model.to(self.device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        )

        # Create config
        model_config = ModelConfig.from_huggingface(config.to_dict())

        logger.info(
            f"Successfully loaded {model_config.model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        return PyTorchModel(model, model_config, tokenizer)

    def load_from_path(self, path: Path) -> BaseModel:
        """Load a model from local path."""
        path = Path(path)

        # Load config
        config = AutoConfig.from_pretrained(path)
        model_config = ModelConfig.from_huggingface(config.to_dict())

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        )

        if self.device.type != "cpu":
            model = model.to(self.device)

        # Load tokenizer if available
        tokenizer = None
        if (path / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(path)

        return PyTorchModel(model, model_config, tokenizer)

    def convert_weights(self, source_weights: dict[str, Any]) -> dict[str, Any]:
        """No conversion needed for PyTorch to PyTorch."""
        return source_weights

    def get_model_info(self, model: BaseModel) -> dict[str, Any]:
        """Get information about a model."""
        if not isinstance(model, PyTorchModel):
            return {}

        return {
            "type": model.config.model_type,
            "parameters": model.num_parameters,
            "memory_footprint_mb": model.memory_footprint / (1024**2),
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "device": str(model.device),
            "dtype": str(next(model.model.parameters()).dtype),
        }
