"""
Configuration management for FineTune.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model name or path")
    quantization: str | None = Field(None, description="Quantization type")
    device: str = Field("mps", description="Device to use")
    dtype: str = Field("float16", description="Data type")
    cache_dir: str = Field("~/Library/Application Support/FineTune/models")


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    path: str = Field(..., description="Dataset path")
    validation_path: str | None = None
    test_size: float = Field(0.1, ge=0.0, le=1.0)
    template: str = Field("alpaca")
    max_length: int = Field(2048, gt=0)
    padding: str = Field("longest")
    truncation: bool = True
    shuffle: bool = True
    seed: int = 42


class TrainingConfig(BaseModel):
    """Training configuration."""

    method: str = Field("lora", description="Training method")
    output_dir: str = "./checkpoints"
    num_epochs: int = Field(3, gt=0)
    batch_size: int = Field(4, gt=0)
    gradient_accumulation_steps: int = Field(4, gt=0)
    learning_rate: float = Field(2e-4, gt=0)
    warmup_steps: int = Field(100, ge=0)
    weight_decay: float = Field(0.01, ge=0)
    max_grad_norm: float = Field(1.0, gt=0)
    lr_scheduler: str = "cosine"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    fp16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42


class LoRAConfig(BaseModel):
    """LoRA configuration."""

    r: int = Field(16, gt=0, description="LoRA rank")
    alpha: int = Field(32, gt=0, description="LoRA alpha")
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    target_modules: list = ["q_proj", "v_proj", "k_proj", "o_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class Config:
    """Main configuration class."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or Path("config/train.yml")
        self.passwords_path = Path("passwords.yml")
        self._config = {}
        self._passwords = {}

        if self.config_path.exists():
            self.load()

    def load(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")

            # Load passwords if exists
            if self.passwords_path.exists():
                with open(self.passwords_path) as f:
                    self._passwords = yaml.safe_load(f)
                logger.info("Loaded passwords configuration")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        try:
            with open(save_path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(**self._config.get("model", {}))

    @property
    def dataset(self) -> DatasetConfig:
        """Get dataset configuration."""
        return DatasetConfig(**self._config.get("dataset", {}))

    @property
    def training(self) -> TrainingConfig:
        """Get training configuration."""
        return TrainingConfig(**self._config.get("training", {}))

    @property
    def lora(self) -> LoRAConfig:
        """Get LoRA configuration."""
        return LoRAConfig(**self._config.get("lora", {}))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def get_password(self, key: str) -> str | None:
        """Get password/secret by key."""
        keys = key.split(".")
        value = self._passwords
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return None
            else:
                return None
        return value

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
