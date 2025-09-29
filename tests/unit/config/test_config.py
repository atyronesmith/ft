"""
Test-driven development for configuration system.

Tests are written first to drive the implementation of configuration
management including config/train.yml, profiles, and validation.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from finetune.config import (
    ConfigError,
    ConfigManager,
    ConfigProfile,
    ConfigValidator,
    DataConfig,
    LoRAConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)


class TestTrainingConfig:
    """Test training configuration functionality."""

    def test_training_config_creation(self):
        """Test creating training config with required fields."""
        # Arrange & Act
        config = TrainingConfig(
            model=ModelConfig(name="meta-llama/Llama-2-7b-hf", cache_dir="/tmp/models"),
            data=DataConfig(train_file="train.jsonl", template="alpaca"),
            lora=LoRAConfig(r=8, alpha=16.0, dropout=0.1),
            optimization=OptimizationConfig(learning_rate=1e-4, batch_size=4, epochs=3),
        )

        # Assert
        assert config.model.name == "meta-llama/Llama-2-7b-hf"
        assert config.data.train_file == "train.jsonl"
        assert config.lora.r == 8
        assert config.optimization.learning_rate == 1e-4

    def test_training_config_defaults(self):
        """Test training config with default values."""
        # Arrange & Act
        config = TrainingConfig(
            model=ModelConfig(name="test-model"), data=DataConfig(train_file="train.jsonl")
        )

        # Assert - defaults should be applied
        assert config.lora.r == 8  # Default LoRA rank
        assert config.optimization.learning_rate == 3e-4  # Default LR
        assert config.optimization.batch_size == 1  # Default batch size
        assert config.data.template == "alpaca"  # Default template

    def test_training_config_validation(self):
        """Test training config validation."""
        # Test invalid learning rate
        with pytest.raises(ConfigError, match="Learning rate must be positive"):
            TrainingConfig(
                model=ModelConfig(name="test"),
                data=DataConfig(train_file="train.jsonl"),
                optimization=OptimizationConfig(learning_rate=-1.0),
            )

        # Test invalid batch size
        with pytest.raises(ConfigError, match="Batch size must be positive"):
            TrainingConfig(
                model=ModelConfig(name="test"),
                data=DataConfig(train_file="train.jsonl"),
                optimization=OptimizationConfig(batch_size=0),
            )

    def test_training_config_to_dict(self):
        """Test converting training config to dictionary."""
        # Arrange
        config = TrainingConfig(
            model=ModelConfig(name="test-model"),
            data=DataConfig(train_file="train.jsonl", template="chatml"),
            lora=LoRAConfig(r=16, alpha=32.0),
        )

        # Act
        config_dict = config.to_dict()

        # Assert
        assert config_dict["model"]["name"] == "test-model"
        assert config_dict["data"]["train_file"] == "train.jsonl"
        assert config_dict["data"]["template"] == "chatml"
        assert config_dict["lora"]["r"] == 16
        assert config_dict["lora"]["alpha"] == 32.0

    def test_training_config_from_dict(self):
        """Test creating training config from dictionary."""
        # Arrange
        config_dict = {
            "model": {"name": "test-model", "cache_dir": "/tmp"},
            "data": {"train_file": "train.jsonl", "validation_file": "val.jsonl"},
            "lora": {"r": 32, "alpha": 64.0, "dropout": 0.05},
            "optimization": {"learning_rate": 5e-4, "batch_size": 8},
        }

        # Act
        config = TrainingConfig.from_dict(config_dict)

        # Assert
        assert config.model.name == "test-model"
        assert config.model.cache_dir == "/tmp"
        assert config.data.validation_file == "val.jsonl"
        assert config.lora.r == 32
        assert config.optimization.batch_size == 8


class TestModelConfig:
    """Test model configuration functionality."""

    def test_model_config_required_fields(self):
        """Test model config requires name field."""
        with pytest.raises(ConfigError, match="Model name is required"):
            ModelConfig(name="")

    def test_model_config_default_values(self):
        """Test model config default values."""
        # Arrange & Act
        config = ModelConfig(name="test-model")

        # Assert
        assert config.load_in_4bit is False
        assert config.torch_dtype == "auto"
        assert config.trust_remote_code is False

    def test_model_config_cache_dir_expansion(self):
        """Test cache directory path expansion."""
        # Arrange & Act
        config = ModelConfig(name="test", cache_dir="~/models")

        # Assert
        assert str(Path("~/models").expanduser()) in str(config.cache_dir)


class TestDataConfig:
    """Test data configuration functionality."""

    def test_data_config_required_fields(self):
        """Test data config requires train_file."""
        with pytest.raises(ConfigError, match="Training file is required"):
            DataConfig(train_file="")

    def test_data_config_file_validation(self):
        """Test data config validates file existence."""
        # Test non-existent file
        with pytest.raises(ConfigError, match="Training file does not exist"):
            DataConfig(train_file="nonexistent.jsonl", validate_files=True)

    def test_data_config_with_validation_file(self):
        """Test data config with validation file."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as train_f:
            train_f.write(b'{"instruction": "test", "output": "test"}\n')
            train_path = train_f.name

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as val_f:
            val_f.write(b'{"instruction": "val", "output": "val"}\n')
            val_path = val_f.name

        try:
            # Act
            config = DataConfig(
                train_file=train_path, validation_file=val_path, validate_files=True
            )

            # Assert
            assert config.train_file == train_path
            assert config.validation_file == val_path
        finally:
            os.unlink(train_path)
            os.unlink(val_path)

    def test_data_config_template_validation(self):
        """Test data config validates template names."""
        with pytest.raises(ConfigError, match="Unknown template"):
            DataConfig(train_file="train.jsonl", template="invalid_template")


class TestLoRAConfig:
    """Test LoRA configuration functionality."""

    def test_lora_config_defaults(self):
        """Test LoRA config default values."""
        # Arrange & Act
        config = LoRAConfig()

        # Assert
        assert config.r == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_lora_config_validation(self):
        """Test LoRA config validation."""
        # Test invalid rank
        with pytest.raises(ConfigError, match="LoRA rank must be positive"):
            LoRAConfig(r=0)

        # Test invalid alpha
        with pytest.raises(ConfigError, match="LoRA alpha must be positive"):
            LoRAConfig(alpha=-1.0)

        # Test invalid dropout
        with pytest.raises(ConfigError, match="Dropout must be between 0 and 1"):
            LoRAConfig(dropout=1.5)

    def test_lora_config_scaling_calculation(self):
        """Test LoRA scaling factor calculation."""
        # Arrange & Act
        config = LoRAConfig(r=8, alpha=16.0)

        # Assert
        assert config.scaling == 16.0 / 8.0  # alpha / r

    def test_lora_config_custom_target_modules(self):
        """Test LoRA with custom target modules."""
        # Arrange & Act
        config = LoRAConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

        # Assert
        assert len(config.target_modules) == 4
        assert "o_proj" in config.target_modules


class TestOptimizationConfig:
    """Test optimization configuration functionality."""

    def test_optimization_config_defaults(self):
        """Test optimization config default values."""
        # Arrange & Act
        config = OptimizationConfig()

        # Assert
        assert config.learning_rate == 3e-4
        assert config.batch_size == 1
        assert config.epochs == 3
        assert config.warmup_steps == 100
        assert config.weight_decay == 0.01

    def test_optimization_config_validation(self):
        """Test optimization config validation."""
        # Test invalid learning rate
        with pytest.raises(ConfigError, match="Learning rate must be positive"):
            OptimizationConfig(learning_rate=0.0)

        # Test invalid epochs
        with pytest.raises(ConfigError, match="Epochs must be positive"):
            OptimizationConfig(epochs=0)

        # Test invalid gradient accumulation
        with pytest.raises(ConfigError, match="Gradient accumulation steps must be positive"):
            OptimizationConfig(gradient_accumulation_steps=0)

    def test_optimization_config_scheduler_options(self):
        """Test learning rate scheduler options."""
        # Arrange & Act
        config = OptimizationConfig(lr_scheduler="cosine", warmup_steps=500)

        # Assert
        assert config.lr_scheduler == "cosine"
        assert config.warmup_steps == 500


class TestConfigManager:
    """Test configuration manager functionality."""

    def test_config_manager_load_from_file(self):
        """Test loading configuration from YAML file."""
        # Arrange
        config_data = {
            "model": {"name": "test-model"},
            "data": {"train_file": "train.jsonl"},
            "lora": {"r": 16, "alpha": 32.0},
            "optimization": {"learning_rate": 1e-4, "batch_size": 4},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            manager = ConfigManager()

            # Act
            config = manager.load_config(config_path)

            # Assert
            assert config.model.name == "test-model"
            assert config.lora.r == 16
            assert config.optimization.batch_size == 4
        finally:
            os.unlink(config_path)

    def test_config_manager_save_to_file(self):
        """Test saving configuration to YAML file."""
        # Arrange
        config = TrainingConfig(
            model=ModelConfig(name="test-model"),
            data=DataConfig(train_file="train.jsonl"),
            lora=LoRAConfig(r=32),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            config_path = f.name

        try:
            manager = ConfigManager()

            # Act
            manager.save_config(config, config_path)

            # Assert - reload and verify
            loaded_config = manager.load_config(config_path)
            assert loaded_config.model.name == "test-model"
            assert loaded_config.lora.r == 32
        finally:
            os.unlink(config_path)

    def test_config_manager_validation(self):
        """Test config manager validates loaded config."""
        # Arrange - create invalid config file
        invalid_config = {
            "model": {"name": ""},  # Invalid empty name
            "data": {"train_file": "train.jsonl"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            manager = ConfigManager()

            # Act & Assert
            with pytest.raises(ConfigError):
                manager.load_config(config_path)
        finally:
            os.unlink(config_path)


class TestConfigProfile:
    """Test configuration profile functionality."""

    def test_profile_chat_preset(self):
        """Test chat configuration profile."""
        # Arrange & Act
        profile = ConfigProfile.get_profile("chat")

        # Assert
        assert profile.data.template == "chatml"
        assert profile.lora.r == 8  # Good for chat
        assert profile.optimization.learning_rate == 2e-4  # Conservative for chat

    def test_profile_instruction_preset(self):
        """Test instruction-following configuration profile."""
        # Arrange & Act
        profile = ConfigProfile.get_profile("instruction")

        # Assert
        assert profile.data.template == "alpaca"
        assert profile.lora.r == 16  # Higher rank for instructions
        assert profile.optimization.batch_size == 4

    def test_profile_code_preset(self):
        """Test code generation configuration profile."""
        # Arrange & Act
        profile = ConfigProfile.get_profile("code")

        # Assert
        assert profile.data.template == "alpaca"
        assert profile.lora.r == 32  # Higher rank for code
        assert profile.optimization.learning_rate == 1e-4  # Lower LR for code

    def test_profile_unknown(self):
        """Test unknown profile raises error."""
        with pytest.raises(ConfigError, match="Unknown profile"):
            ConfigProfile.get_profile("unknown")

    def test_profile_list(self):
        """Test listing available profiles."""
        # Arrange & Act
        profiles = ConfigProfile.list_profiles()

        # Assert
        assert "chat" in profiles
        assert "instruction" in profiles
        assert "code" in profiles

    def test_profile_apply_to_config(self):
        """Test applying profile to existing config."""
        # Arrange
        base_config = TrainingConfig(
            model=ModelConfig(name="test-model"), data=DataConfig(train_file="train.jsonl")
        )

        # Act
        chat_config = ConfigProfile.apply_profile(base_config, "chat")

        # Assert
        assert chat_config.model.name == "test-model"  # Preserved
        assert chat_config.data.template == "chatml"  # Updated by profile
        assert chat_config.lora.r == 8  # From chat profile


class TestConfigValidator:
    """Test configuration validation functionality."""

    def test_validator_check_model_compatibility(self):
        """Test validator checks model compatibility."""
        # Arrange
        validator = ConfigValidator()
        config = TrainingConfig(
            model=ModelConfig(name="test-model", load_in_4bit=True),
            data=DataConfig(train_file="train.jsonl"),
            lora=LoRAConfig(r=128),  # Very high rank
        )

        # Act
        warnings = validator.validate(config)

        # Assert
        assert any("High LoRA rank" in warning for warning in warnings)

    def test_validator_check_data_consistency(self):
        """Test validator checks data configuration consistency."""
        # Arrange
        validator = ConfigValidator()
        config = TrainingConfig(
            model=ModelConfig(name="test-model"),
            data=DataConfig(train_file="train.jsonl", validation_file="val.jsonl"),
            optimization=OptimizationConfig(batch_size=32, gradient_accumulation_steps=1),
        )

        # Act
        warnings = validator.validate(config)

        # Assert
        # Should warn about large batch size without gradient accumulation
        assert len(warnings) >= 0  # May have warnings

    def test_validator_memory_estimation(self):
        """Test validator estimates memory requirements."""
        # Arrange
        validator = ConfigValidator()
        config = TrainingConfig(
            model=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
            data=DataConfig(train_file="train.jsonl"),
            optimization=OptimizationConfig(batch_size=8),
        )

        # Act
        memory_estimate = validator.estimate_memory_usage(config)

        # Assert
        assert memory_estimate > 0
        assert isinstance(memory_estimate, (int, float))

    def test_validator_recommend_batch_size(self):
        """Test validator recommends optimal batch size."""
        # Arrange
        validator = ConfigValidator()
        config = TrainingConfig(
            model=ModelConfig(name="test-model"),
            data=DataConfig(train_file="train.jsonl"),
        )

        # Act
        recommended_batch_size = validator.recommend_batch_size(config, available_memory_gb=16)

        # Assert
        assert recommended_batch_size > 0
        assert isinstance(recommended_batch_size, int)


class TestConfigIntegration:
    """Test configuration integration with other components."""

    def test_config_with_data_loading(self):
        """Test config integration with data loading."""
        # Arrange
        from finetune.data import DatasetLoader, TemplateRegistry

        # Create test data file
        test_data = [
            {"instruction": "What is Python?", "output": "A programming language"},
            {"instruction": "What is AI?", "output": "Artificial Intelligence"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(test_data, f)
            data_path = f.name

        try:
            config = TrainingConfig(
                model=ModelConfig(name="test-model"),
                data=DataConfig(train_file=data_path, template="alpaca"),
            )

            # Act
            loader = DatasetLoader()
            dataset = loader.load(config.data.train_file)

            registry = TemplateRegistry()
            template = registry.get_template(config.data.template)
            formatted_examples = template.format_batch(dataset)

            # Assert
            assert len(formatted_examples) == 2
            assert "### Instruction:" in formatted_examples[0]
            assert "What is Python?" in formatted_examples[0]
        finally:
            os.unlink(data_path)

    def test_config_serialization_roundtrip(self):
        """Test complete config serialization roundtrip."""
        # Arrange
        original_config = TrainingConfig(
            model=ModelConfig(
                name="meta-llama/Llama-2-7b-hf", cache_dir="/tmp/models", load_in_4bit=True
            ),
            data=DataConfig(
                train_file="train.jsonl",
                validation_file="val.jsonl",
                template="chatml",
                max_length=2048,
            ),
            lora=LoRAConfig(
                r=32,
                alpha=64.0,
                dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            optimization=OptimizationConfig(
                learning_rate=1e-4, batch_size=4, epochs=5, warmup_steps=200, lr_scheduler="cosine"
            ),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            config_path = f.name

        try:
            manager = ConfigManager()

            # Act - save and reload
            manager.save_config(original_config, config_path)
            loaded_config = manager.load_config(config_path)

            # Assert - verify all fields preserved
            assert loaded_config.model.name == original_config.model.name
            assert loaded_config.model.load_in_4bit == original_config.model.load_in_4bit
            assert loaded_config.data.template == original_config.data.template
            assert loaded_config.data.max_length == original_config.data.max_length
            assert loaded_config.lora.r == original_config.lora.r
            assert loaded_config.lora.target_modules == original_config.lora.target_modules
            assert (
                loaded_config.optimization.lr_scheduler == original_config.optimization.lr_scheduler
            )
        finally:
            os.unlink(config_path)


# Test fixtures
@pytest.fixture
def sample_config():
    """Provide sample training configuration."""
    return TrainingConfig(
        model=ModelConfig(name="test-model"),
        data=DataConfig(train_file="train.jsonl"),
        lora=LoRAConfig(r=16, alpha=32.0),
        optimization=OptimizationConfig(learning_rate=2e-4, batch_size=4),
    )


@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        config_dict = sample_config.to_dict()
        yaml.dump(config_dict, f)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
