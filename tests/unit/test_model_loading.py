"""
Tests for model loading functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from finetune.models.base import ModelConfig


def _mlx_available():
    """Check if MLX is available."""
    try:
        import mlx

        return True
    except ImportError:
        return False


def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


class TestModelConfig:
    """Test ModelConfig class."""

    def test_from_huggingface(self):
        """Test creating config from HuggingFace format."""
        hf_config = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "num_key_value_heads": 32,
        }

        config = ModelConfig.from_huggingface(hf_config)

        assert config.model_type == "llama"
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32

    def test_save_load(self):
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

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config.save(config_path)

            assert config_path.exists()

            loaded_config = ModelConfig.load(config_path)
            assert loaded_config.model_type == config.model_type
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.hidden_size == config.hidden_size


class TestMLXModelLoading:
    """Test MLX model loading."""

    @pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
    def test_mlx_model_creation(self):
        """Test creating MLX model from config."""
        from finetune.models.mlx_models import get_mlx_model

        config = ModelConfig(
            model_type="llama",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )

        model = get_mlx_model(config)

        assert model is not None
        assert model.config == config
        assert model.num_parameters > 0

    @pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
    def test_mlx_forward_pass(self):
        """Test forward pass through MLX model."""
        import mlx.core as mx
        from finetune.models.mlx_models import get_mlx_model

        config = ModelConfig(
            model_type="llama",
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=128,
        )

        model = get_mlx_model(config)

        # Create dummy input
        batch_size = 2
        seq_len = 10
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model.forward(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)


class TestPyTorchModelLoading:
    """Test PyTorch model loading."""

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_pytorch_model_wrapper(self):
        """Test PyTorchModel wrapper."""
        import torch
        from finetune.models.torch_loader import PyTorchModel

        # Create mock model
        mock_model = Mock()
        mock_model.parameters = Mock(side_effect=lambda: iter([torch.zeros(1)]))
        # Mock the model as callable (not forward method)
        mock_model.return_value = Mock(logits=torch.randn(2, 10, 100))

        config = ModelConfig(
            model_type="gpt2",
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=128,
        )

        model = PyTorchModel(mock_model, config)

        # Test forward
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model.forward(input_ids)

        assert logits is not None
        mock_model.assert_called_once_with(input_ids=input_ids)


class TestModelManager:
    """Test ModelManager functionality."""

    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        from finetune.models.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(
                cache_dir=Path(tmpdir) / "models", registry_db=Path(tmpdir) / "registry.db"
            )

            assert manager.cache_dir.exists()
            assert manager.registry is not None

    def test_list_cached_models(self):
        """Test listing cached models."""
        from finetune.models.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "models"

            # Create fake model directories
            model1_dir = cache_dir / "model1"
            model1_dir.mkdir(parents=True)
            (model1_dir / "config.json").write_text("{}")

            model2_dir = cache_dir / "model2"
            model2_dir.mkdir(parents=True)
            (model2_dir / "config.json").write_text("{}")

            manager = ModelManager(cache_dir=cache_dir)
            cached = manager._list_cached_models()

            assert len(cached) == 2
            assert "model1" in cached
            assert "model2" in cached

    def test_get_model_info(self):
        """Test getting model information."""
        from finetune.models.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "models"
            model_dir = cache_dir / "test-model"
            model_dir.mkdir(parents=True)

            # Create config
            config = {
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            }
            (model_dir / "config.json").write_text(json.dumps(config))

            # Create dummy weight file
            (model_dir / "model.bin").write_text("dummy")

            manager = ModelManager(cache_dir=cache_dir)
            info = manager.get_model_info("test-model")

            assert info is not None
            assert info["name"] == "test-model"
            assert info["model_type"] == "llama"
            assert info["vocab_size"] == 32000
            assert "config.json" in info["files"]

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        from finetune.models.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "models"
            model_dir = cache_dir / "test-model"
            model_dir.mkdir(parents=True)

            # Create config
            config = {
                "vocab_size": 1000,
                "hidden_size": 256,
                "num_hidden_layers": 4,
            }
            (model_dir / "config.json").write_text(json.dumps(config))

            manager = ModelManager(cache_dir=cache_dir)
            memory = manager.estimate_memory_usage(
                "test-model", batch_size=2, sequence_length=512, training=True
            )

            assert "model_gb" in memory
            assert "activations_gb" in memory
            assert "gradients_gb" in memory
            assert "optimizer_gb" in memory
            assert "total_gb" in memory
            assert memory["total_gb"] > 0
