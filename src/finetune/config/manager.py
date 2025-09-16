"""
Configuration manager for loading and saving configurations.

Handles YAML file operations and configuration validation.
"""

import yaml
from pathlib import Path
from typing import Union

from .config import TrainingConfig, ConfigError


class ConfigManager:
    """Manager for loading and saving training configurations."""

    def load_config(self, config_path: Union[str, Path]) -> TrainingConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            TrainingConfig instance

        Raises:
            ConfigError: If file doesn't exist or configuration is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML format: {e}")

        if not config_data:
            raise ConfigError("Configuration file is empty")

        try:
            return TrainingConfig.from_dict(config_data)
        except Exception as e:
            raise ConfigError(f"Invalid configuration: {e}")

    def save_config(self, config: TrainingConfig, config_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: TrainingConfig to save
            config_path: Path where to save configuration

        Raises:
            ConfigError: If unable to save configuration
        """
        config_path = Path(config_path)

        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_dict = config.to_dict()

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2
                )
        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {e}")

    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """
        Validate configuration file without loading.

        Args:
            config_path: Path to configuration file

        Returns:
            True if valid, False otherwise
        """
        try:
            self.load_config(config_path)
            return True
        except ConfigError:
            return False