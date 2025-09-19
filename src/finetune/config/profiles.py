"""
Configuration profiles for common use cases.

Provides predefined configurations for chat, instruction-following,
code generation, and other common fine-tuning scenarios.
"""

from copy import deepcopy

from .config import (
    ConfigError,
    TrainingConfig,
)


class ConfigProfile:
    """Predefined configuration profiles for common use cases."""

    _PROFILES = {
        "chat": {
            "data": {
                "template": "chatml",
                "max_length": 2048,
            },
            "lora": {
                "r": 8,
                "alpha": 16.0,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
            "optimization": {
                "learning_rate": 2e-4,
                "batch_size": 2,
                "epochs": 3,
                "warmup_steps": 100,
                "lr_scheduler": "cosine",
            },
        },
        "instruction": {
            "data": {
                "template": "alpaca",
                "max_length": 2048,
            },
            "lora": {
                "r": 16,
                "alpha": 32.0,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
            "optimization": {
                "learning_rate": 3e-4,
                "batch_size": 4,
                "epochs": 3,
                "warmup_steps": 200,
                "lr_scheduler": "linear",
            },
        },
        "code": {
            "data": {
                "template": "alpaca",
                "max_length": 4096,
            },
            "lora": {
                "r": 32,
                "alpha": 64.0,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "optimization": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 5,
                "warmup_steps": 500,
                "lr_scheduler": "cosine",
                "weight_decay": 0.01,
            },
        },
    }

    @classmethod
    def get_profile(cls, profile_name: str) -> TrainingConfig:
        """
        Get configuration profile by name.

        Args:
            profile_name: Name of the profile

        Returns:
            TrainingConfig with profile settings

        Raises:
            ConfigError: If profile doesn't exist
        """
        if profile_name not in cls._PROFILES:
            available = list(cls._PROFILES.keys())
            raise ConfigError(f"Unknown profile '{profile_name}'. Available profiles: {available}")

        profile_data = deepcopy(cls._PROFILES[profile_name])

        # Create a minimal config and apply profile
        base_config = {"model": {"name": "placeholder"}, "data": {"train_file": "placeholder"}}

        # Merge profile data
        for section, settings in profile_data.items():
            if section in base_config:
                base_config[section].update(settings)
            else:
                base_config[section] = settings

        return TrainingConfig.from_dict(base_config)

    @classmethod
    def list_profiles(cls) -> list[str]:
        """
        List available profile names.

        Returns:
            List of profile names
        """
        return list(cls._PROFILES.keys())

    @classmethod
    def apply_profile(cls, base_config: TrainingConfig, profile_name: str) -> TrainingConfig:
        """
        Apply profile settings to existing configuration.

        Args:
            base_config: Base configuration to modify
            profile_name: Profile to apply

        Returns:
            New TrainingConfig with profile applied

        Raises:
            ConfigError: If profile doesn't exist
        """
        if profile_name not in cls._PROFILES:
            available = list(cls._PROFILES.keys())
            raise ConfigError(f"Unknown profile '{profile_name}'. Available profiles: {available}")

        # Convert base config to dict
        config_dict = base_config.to_dict()
        profile_data = deepcopy(cls._PROFILES[profile_name])

        # Merge profile settings
        for section, settings in profile_data.items():
            if section in config_dict:
                config_dict[section].update(settings)
            else:
                config_dict[section] = settings

        return TrainingConfig.from_dict(config_dict)

    @classmethod
    def create_profile(cls, name: str, profile_data: dict) -> None:
        """
        Create a new custom profile.

        Args:
            name: Profile name
            profile_data: Profile configuration data
        """
        cls._PROFILES[name] = deepcopy(profile_data)
