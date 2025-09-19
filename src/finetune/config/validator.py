"""
Configuration validator for checking compatibility and optimization.

Provides validation, warnings, and recommendations for training configurations.
"""

import re
from typing import Any

from .config import TrainingConfig


class ConfigValidator:
    """Validator for training configurations."""

    def validate(self, config: TrainingConfig) -> list[str]:
        """
        Validate configuration and return warnings.

        Args:
            config: Configuration to validate

        Returns:
            List of warning messages
        """
        warnings = []

        # Check LoRA configuration
        warnings.extend(self._validate_lora_config(config))

        # Check optimization settings
        warnings.extend(self._validate_optimization_config(config))

        # Check model-specific settings
        warnings.extend(self._validate_model_config(config))

        # Check data configuration
        warnings.extend(self._validate_data_config(config))

        return warnings

    def _validate_lora_config(self, config: TrainingConfig) -> list[str]:
        """Validate LoRA configuration."""
        warnings = []

        if config.lora.r > 64:
            warnings.append(
                f"High LoRA rank ({config.lora.r}) may be unnecessary and slow training"
            )

        if config.lora.alpha / config.lora.r > 4.0:
            warnings.append(f"High LoRA scaling ({config.lora.scaling:.1f}) may cause instability")

        if config.lora.dropout > 0.3:
            warnings.append(f"High LoRA dropout ({config.lora.dropout}) may hurt performance")

        return warnings

    def _validate_optimization_config(self, config: TrainingConfig) -> list[str]:
        """Validate optimization configuration."""
        warnings = []

        # Check learning rate
        if config.optimization.learning_rate > 1e-3:
            warnings.append("High learning rate may cause training instability")

        if config.optimization.learning_rate < 1e-6:
            warnings.append("Very low learning rate may result in slow convergence")

        # Check batch size and gradient accumulation
        effective_batch_size = (
            config.optimization.batch_size * config.optimization.gradient_accumulation_steps
        )

        if effective_batch_size > 32:
            warnings.append(
                f"Large effective batch size ({effective_batch_size}) may hurt generalization"
            )

        if (
            config.optimization.batch_size > 8
            and config.optimization.gradient_accumulation_steps == 1
        ):
            warnings.append("Consider using gradient accumulation instead of large batch size")

        return warnings

    def _validate_model_config(self, config: TrainingConfig) -> list[str]:
        """Validate model configuration."""
        warnings = []

        # Check quantization with LoRA
        if config.model.load_in_4bit and config.lora.r > 32:
            warnings.append("High LoRA rank with 4-bit quantization may not provide benefits")

        # Check model name format
        if not self._is_valid_model_name(config.model.name):
            warnings.append("Model name should follow HuggingFace format (org/model-name)")

        return warnings

    def _validate_data_config(self, config: TrainingConfig) -> list[str]:
        """Validate data configuration."""
        warnings = []

        if config.data.max_length > 4096:
            warnings.append("Very long sequences may cause memory issues")

        if config.data.validation_split > 0.3:
            warnings.append("Large validation split may reduce training data significantly")

        return warnings

    def _is_valid_model_name(self, name: str) -> bool:
        """Check if model name follows HuggingFace convention."""
        if name == "placeholder":
            return True  # Allow placeholder for testing

        # HuggingFace format: org/model-name or just model-name
        pattern = r"^([a-zA-Z0-9._-]+/)?[a-zA-Z0-9._-]+$"
        return bool(re.match(pattern, name))

    def estimate_memory_usage(self, config: TrainingConfig) -> float:
        """
        Estimate memory usage in GB.

        Args:
            config: Configuration to analyze

        Returns:
            Estimated memory usage in GB
        """
        # Rough estimation based on model size and configuration
        base_memory = 0

        # Estimate based on model name
        model_name = config.model.name.lower()
        if "7b" in model_name:
            base_memory = 14.0  # 7B parameters ≈ 14GB in FP16
        elif "13b" in model_name:
            base_memory = 26.0
        elif "30b" in model_name or "33b" in model_name:
            base_memory = 60.0
        elif "65b" in model_name or "70b" in model_name:
            base_memory = 130.0
        else:
            base_memory = 8.0  # Default for smaller models

        # Adjust for quantization
        if config.model.load_in_4bit:
            base_memory *= 0.5

        # Add overhead for LoRA adapters (minimal)
        lora_overhead = 0.1 * (config.lora.r / 8.0)

        # Add gradient and optimizer memory
        training_overhead = base_memory * 0.5  # Conservative estimate

        # Multiply by batch size
        total_memory = (
            base_memory + lora_overhead + training_overhead
        ) * config.optimization.batch_size

        return round(total_memory, 1)

    def recommend_batch_size(self, config: TrainingConfig, available_memory_gb: float) -> int:
        """
        Recommend optimal batch size based on available memory.

        Args:
            config: Configuration to analyze
            available_memory_gb: Available GPU memory in GB

        Returns:
            Recommended batch size
        """
        # Start with batch size 1 and estimate memory
        test_config = config
        test_config.optimization.batch_size = 1

        memory_per_batch = self.estimate_memory_usage(test_config)

        # Calculate max batch size with 80% memory utilization
        max_batch_size = int((available_memory_gb * 0.8) / memory_per_batch)

        # Ensure minimum of 1 and reasonable maximum
        recommended_batch_size = max(1, min(max_batch_size, 16))

        return recommended_batch_size

    def check_compatibility(self, config: TrainingConfig) -> dict[str, Any]:
        """
        Check overall compatibility and provide recommendations.

        Args:
            config: Configuration to check

        Returns:
            Dictionary with compatibility information
        """
        warnings = self.validate(config)
        memory_estimate = self.estimate_memory_usage(config)

        return {
            "warnings": warnings,
            "memory_estimate_gb": memory_estimate,
            "is_valid": len([w for w in warnings if "error" in w.lower()]) == 0,
            "recommendations": self._generate_recommendations(config, warnings),
        }

    def _generate_recommendations(self, config: TrainingConfig, warnings: list[str]) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if config.lora.r < 8:
            recommendations.append("Consider increasing LoRA rank to 8-16 for better adaptation")

        if (
            config.optimization.batch_size == 1
            and config.optimization.gradient_accumulation_steps == 1
        ):
            recommendations.append(
                "Consider using gradient accumulation to increase effective batch size"
            )

        if config.optimization.epochs > 5:
            recommendations.append("Many epochs may lead to overfitting; consider early stopping")

        return recommendations
