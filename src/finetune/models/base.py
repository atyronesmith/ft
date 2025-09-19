"""
Base classes for model handling.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_type: str  # llama, gpt2, mistral, etc.
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    use_cache: bool = True

    # Optional fields
    num_key_value_heads: int | None = None  # For GQA
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    hidden_act: str = "silu"

    @classmethod
    def from_huggingface(cls, hf_config: dict[str, Any]) -> "ModelConfig":
        """Create from HuggingFace config."""
        # Handle different naming conventions across model architectures
        model_type = hf_config.get("model_type", "unknown")

        # Map hidden_size field (varies by model type)
        if "hidden_size" in hf_config:
            hidden_size = hf_config["hidden_size"]
        elif "n_embd" in hf_config:  # GPT-2, DialoGPT
            hidden_size = hf_config["n_embd"]
        elif "d_model" in hf_config:  # T5, some other models
            hidden_size = hf_config["d_model"]
        else:
            raise ValueError(
                f"Could not find hidden_size field in config for model type: {model_type}"
            )

        # Map num_hidden_layers field
        if "num_hidden_layers" in hf_config:
            num_layers = hf_config["num_hidden_layers"]
        elif "n_layer" in hf_config:  # GPT-2, DialoGPT
            num_layers = hf_config["n_layer"]
        elif "num_layers" in hf_config:  # Some models
            num_layers = hf_config["num_layers"]
        else:
            raise ValueError(
                f"Could not find num_hidden_layers field in config for model type: {model_type}"
            )

        # Map num_attention_heads field
        if "num_attention_heads" in hf_config:
            num_heads = hf_config["num_attention_heads"]
        elif "n_head" in hf_config:  # GPT-2, DialoGPT
            num_heads = hf_config["n_head"]
        else:
            raise ValueError(
                f"Could not find num_attention_heads field in config for model type: {model_type}"
            )

        # Map max_position_embeddings field
        if "max_position_embeddings" in hf_config:
            max_pos_emb = hf_config["max_position_embeddings"]
        elif "n_positions" in hf_config:  # GPT-2, DialoGPT
            max_pos_emb = hf_config["n_positions"]
        elif "n_ctx" in hf_config:  # Alternative GPT-2 field
            max_pos_emb = hf_config["n_ctx"]
        else:
            max_pos_emb = 2048  # Default

        # Set model-specific defaults
        if model_type == "gpt2":
            # GPT-2 models (including DialoGPT) tie embeddings by default
            tie_word_embeddings = hf_config.get("tie_word_embeddings", True)
        else:
            tie_word_embeddings = hf_config.get("tie_word_embeddings", False)

        return cls(
            model_type=model_type,
            vocab_size=hf_config["vocab_size"],
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hf_config.get(
                "intermediate_size", hf_config.get("n_inner", hidden_size * 4)
            ),
            max_position_embeddings=max_pos_emb,
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            layer_norm_eps=hf_config.get(
                "layer_norm_eps", hf_config.get("layer_norm_epsilon", 1e-5)
            ),
            tie_word_embeddings=tie_word_embeddings,
            use_cache=hf_config.get("use_cache", True),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            rope_scaling=hf_config.get("rope_scaling"),
            hidden_act=hf_config.get("hidden_act", "silu"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        """Load configuration from file."""
        with open(path) as f:
            data = json.load(f)

        # Filter data to only include fields that our ModelConfig accepts
        # This allows loading configs with extra fields we don't support
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


class BaseModel(ABC):
    """Abstract base class for models."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def forward(self, input_ids: Any, **kwargs) -> Any:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def generate(self, input_ids: Any, max_length: int = 100, **kwargs) -> Any:
        """Generate text from the model."""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load model from disk."""
        pass

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        raise NotImplementedError

    @property
    def memory_footprint(self) -> int:
        """Estimate memory footprint in bytes."""
        # Rough estimate: params * bytes_per_param
        return self.num_parameters * 2  # Assuming FP16


class ModelLoader(ABC):
    """Abstract base class for model loaders."""

    @abstractmethod
    def load_from_huggingface(
        self, model_id: str, revision: str | None = None, cache_dir: Path | None = None, **kwargs
    ) -> BaseModel:
        """Load a model from HuggingFace Hub."""
        pass

    @abstractmethod
    def load_from_path(self, path: Path) -> BaseModel:
        """Load a model from local path."""
        pass

    @abstractmethod
    def convert_weights(self, source_weights: dict[str, Any]) -> dict[str, Any]:
        """Convert weights from source format to target format."""
        pass
