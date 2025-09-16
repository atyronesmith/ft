"""
Core configuration classes for fine-tuning.

Defines configuration dataclasses for models, data, training parameters,
and validation logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import os


class ConfigError(Exception):
    """Raised when configuration is invalid or inconsistent."""
    pass


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    name: str
    cache_dir: Optional[str] = None
    load_in_4bit: bool = False
    torch_dtype: str = "auto"
    trust_remote_code: bool = False

    def __post_init__(self):
        """Validate model configuration."""
        if not self.name or not self.name.strip():
            raise ConfigError("Model name is required")

        # Expand cache directory path
        if self.cache_dir:
            self.cache_dir = str(Path(self.cache_dir).expanduser())


@dataclass
class DataConfig:
    """Configuration for data settings."""

    train_file: str
    validation_file: Optional[str] = None
    template: str = "alpaca"
    max_length: int = 2048
    validation_split: float = 0.1
    validate_files: bool = False

    def __post_init__(self):
        """Validate data configuration."""
        if not self.train_file or not self.train_file.strip():
            raise ConfigError("Training file is required")

        if self.validate_files:
            if not Path(self.train_file).exists():
                raise ConfigError(f"Training file does not exist: {self.train_file}")

            if self.validation_file and not Path(self.validation_file).exists():
                raise ConfigError(f"Validation file does not exist: {self.validation_file}")

        # Validate template name
        valid_templates = ["alpaca", "chatml", "llama"]
        if self.template not in valid_templates:
            raise ConfigError(f"Unknown template '{self.template}'. Valid options: {valid_templates}")

        if self.validation_split < 0 or self.validation_split > 1:
            raise ConfigError("Validation split must be between 0 and 1")


@dataclass
class LoRAConfig:
    """Configuration for LoRA settings."""

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self):
        """Validate LoRA configuration."""
        if self.r <= 0:
            raise ConfigError("LoRA rank must be positive")

        if self.alpha <= 0:
            raise ConfigError("LoRA alpha must be positive")

        if self.dropout < 0 or self.dropout > 1:
            raise ConfigError("Dropout must be between 0 and 1")

    @property
    def scaling(self) -> float:
        """Calculate LoRA scaling factor."""
        return self.alpha / self.r


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""

    learning_rate: float = 3e-4
    batch_size: int = 1
    epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    lr_scheduler: str = "linear"
    save_steps: int = 500
    eval_steps: int = 500

    def __post_init__(self):
        """Validate optimization configuration."""
        if self.learning_rate <= 0:
            raise ConfigError("Learning rate must be positive")

        if self.batch_size <= 0:
            raise ConfigError("Batch size must be positive")

        if self.epochs <= 0:
            raise ConfigError("Epochs must be positive")

        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("Gradient accumulation steps must be positive")

        valid_schedulers = ["linear", "cosine", "constant", "constant_with_warmup"]
        if self.lr_scheduler not in valid_schedulers:
            raise ConfigError(f"Unknown scheduler '{self.lr_scheduler}'. Valid options: {valid_schedulers}")


@dataclass
class TrainingConfig:
    """Main training configuration."""

    model: ModelConfig
    data: DataConfig
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output_dir: str = "./output"
    seed: int = 42

    def __post_init__(self):
        """Validate training configuration."""
        # Cross-validation between configs
        if self.optimization.learning_rate <= 0:
            raise ConfigError("Learning rate must be positive")

        if self.optimization.batch_size <= 0:
            raise ConfigError("Batch size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def _asdict_recursive(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, '__dict__'):
                        result[key] = _asdict_recursive(value)
                    elif isinstance(value, list):
                        result[key] = [_asdict_recursive(item) if hasattr(item, '__dict__') else item for item in value]
                    else:
                        result[key] = value
                return result
            return obj

        return _asdict_recursive(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        # Extract nested configs
        model_config = ModelConfig(**data.get("model", {}))
        data_config = DataConfig(**data.get("data", {}))
        lora_config = LoRAConfig(**data.get("lora", {}))
        optimization_config = OptimizationConfig(**data.get("optimization", {}))

        # Create main config
        config_kwargs = {
            "model": model_config,
            "data": data_config,
            "lora": lora_config,
            "optimization": optimization_config,
        }

        # Add any additional top-level fields
        for key, value in data.items():
            if key not in ["model", "data", "lora", "optimization"]:
                config_kwargs[key] = value

        return cls(**config_kwargs)