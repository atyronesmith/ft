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
        return cls(
            model_type=hf_config.get("model_type", "unknown"),
            vocab_size=hf_config["vocab_size"],
            hidden_size=hf_config["hidden_size"],
            num_hidden_layers=hf_config["num_hidden_layers"],
            num_attention_heads=hf_config["num_attention_heads"],
            intermediate_size=hf_config.get("intermediate_size", hf_config["hidden_size"] * 4),
            max_position_embeddings=hf_config.get("max_position_embeddings", 2048),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-5),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
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
        return cls(**data)


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
