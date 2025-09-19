"""
Unit tests for base model classes.
"""

from pathlib import Path

import pytest
from finetune.models.base import BaseModel, ModelConfig, ModelLoader


class TestModelConfig:
    """Test ModelConfig class."""

    def test_init_with_required_fields(self):
        """Test initialization with required fields."""
        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )

        assert config.model_type == "test"
        assert config.vocab_size == 1000
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 4
        assert config.num_attention_heads == 8
        assert config.intermediate_size == 1024
        assert config.max_position_embeddings == 512

    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        config = ModelConfig(
            model_type="llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=11008,
            max_position_embeddings=2048,
            num_key_value_heads=8,  # Optional
            rope_theta=50000.0,  # Optional
            hidden_act="gelu",  # Optional
        )

        assert config.num_key_value_heads == 8
        assert config.rope_theta == 50000.0
        assert config.hidden_act == "gelu"

    def test_from_huggingface(self, sample_model_config):
        """Test creating config from HuggingFace format."""
        config = ModelConfig.from_huggingface(sample_model_config)

        assert config.model_type == sample_model_config["model_type"]
        assert config.vocab_size == sample_model_config["vocab_size"]
        assert config.hidden_size == sample_model_config["hidden_size"]
        assert config.num_hidden_layers == sample_model_config["num_hidden_layers"]
        assert config.num_attention_heads == sample_model_config["num_attention_heads"]

    def test_from_huggingface_with_missing_fields(self):
        """Test creating config from incomplete HuggingFace format."""
        minimal_config = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
        }

        config = ModelConfig.from_huggingface(minimal_config)

        # Check defaults are applied
        assert config.model_type == "unknown"
        assert config.intermediate_size == 256 * 4  # 4x hidden_size
        assert config.max_position_embeddings == 2048
        assert config.rms_norm_eps == 1e-6

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
            num_key_value_heads=4,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_type"] == "test"
        assert config_dict["vocab_size"] == 1000
        assert config_dict["num_key_value_heads"] == 4
        # None values should be excluded
        assert "rope_scaling" not in config_dict or config_dict["rope_scaling"] is None

    def test_save_and_load(self, temp_dir):
        """Test saving and loading config."""
        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )

        config_path = temp_dir / "config.json"
        config.save(config_path)

        assert config_path.exists()

        # Load and verify
        loaded_config = ModelConfig.load(config_path)

        assert loaded_config.model_type == config.model_type
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.hidden_size == config.hidden_size
        assert loaded_config.num_hidden_layers == config.num_hidden_layers

    def test_save_creates_parent_dirs(self, temp_dir):
        """Test that save creates parent directories if needed."""
        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )

        config_path = temp_dir / "nested" / "dir" / "config.json"
        config.save(config_path)

        assert config_path.exists()
        assert config_path.parent.exists()


class TestBaseModel:
    """Test BaseModel abstract class."""

    def test_base_model_is_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )

        with pytest.raises(TypeError):
            # Should fail because abstract methods not implemented
            BaseModel(config)

    def test_base_model_properties(self):
        """Test BaseModel properties with a concrete implementation."""

        class ConcreteModel(BaseModel):
            def forward(self, input_ids, **kwargs):
                return input_ids

            def generate(self, input_ids, max_length=100, **kwargs):
                return input_ids

            def save(self, path):
                pass

            def load(self, path):
                pass

            @property
            def num_parameters(self):
                return 1000000

        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )

        model = ConcreteModel(config)

        assert model.config == config
        assert model.num_parameters == 1000000
        assert model.memory_footprint == 1000000 * 2  # FP16 assumption


class TestModelLoader:
    """Test ModelLoader abstract class."""

    def test_model_loader_is_abstract(self):
        """Test that ModelLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Should fail because abstract methods not implemented
            ModelLoader()

    def test_model_loader_interface(self):
        """Test ModelLoader interface with a concrete implementation."""

        class ConcreteLoader(ModelLoader):
            def load_from_huggingface(self, model_id, revision=None, cache_dir=None, **kwargs):
                return f"Loaded {model_id}"

            def load_from_path(self, path):
                return f"Loaded from {path}"

            def convert_weights(self, source_weights):
                return source_weights

        loader = ConcreteLoader()

        # Test interface methods
        result = loader.load_from_huggingface("test/model")
        assert result == "Loaded test/model"

        result = loader.load_from_path(Path("/test/path"))
        assert result == "Loaded from /test/path"

        weights = {"test": "weight"}
        assert loader.convert_weights(weights) == weights
