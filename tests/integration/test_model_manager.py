"""
Integration tests for model manager.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from finetune.models.base import ModelConfig


class TestModelManager:
    """Integration tests for ModelManager."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create model manager with temp directories."""
        with patch("finetune.models.manager.device_manager") as mock_device:
            mock_backend = Mock()
            mock_backend.name = "mlx"
            mock_device.get_optimal_backend.return_value = mock_backend

            with patch("finetune.models.manager.MLXModelLoader"):
                from finetune.models.manager import ModelManager

                return ModelManager(
                    cache_dir=temp_dir / "cache", registry_db=temp_dir / "registry.db"
                )

    def test_manager_initialization(self, temp_dir):
        """Test ModelManager initialization."""
        with patch("finetune.models.manager.device_manager") as mock_device:
            mock_backend = Mock()
            mock_backend.name = "mlx"
            mock_device.get_optimal_backend.return_value = mock_backend

            with patch("finetune.models.manager.MLXModelLoader"):
                from finetune.models.manager import ModelManager

                manager = ModelManager(
                    cache_dir=temp_dir / "cache", registry_db=temp_dir / "registry.db"
                )

                assert manager.cache_dir == temp_dir / "cache"
                assert manager.cache_dir.exists()
                assert manager.registry is not None
                assert manager.loader is not None

    def test_manager_pytorch_fallback(self, temp_dir):
        """Test manager with PyTorch backend."""
        with patch("finetune.models.manager.device_manager") as mock_device:
            mock_backend = Mock()
            mock_backend.name = "pytorch"
            mock_device.get_optimal_backend.return_value = mock_backend

            with patch("finetune.models.manager.PyTorchModelLoader"):
                from finetune.models.manager import ModelManager

                manager = ModelManager()
                assert manager.loader is not None

    def test_list_cached_models(self, manager, temp_dir):
        """Test listing cached models."""
        # Create fake cached models
        cache_dir = temp_dir / "cache"

        model1_dir = cache_dir / "model1"
        model1_dir.mkdir(parents=True)
        (model1_dir / "config.json").write_text("{}")

        model2_dir = cache_dir / "model2"
        model2_dir.mkdir(parents=True)
        (model2_dir / "config.json").write_text("{}")

        # Directory without config should be ignored
        model3_dir = cache_dir / "not_a_model"
        model3_dir.mkdir(parents=True)

        cached = manager._list_cached_models()

        assert len(cached) == 2
        assert "model1" in cached
        assert "model2" in cached
        assert "not_a_model" not in cached

    def test_list_models(self, manager, temp_dir):
        """Test listing all models."""
        # Create cached model
        cache_dir = temp_dir / "cache"
        model_dir = cache_dir / "cached-model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        # Mock registry models
        manager.registry.list_models = Mock(
            return_value=[
                {
                    "name": "registered-model",
                    "source": "huggingface",
                    "size_gb": 1.5,
                }
            ]
        )

        models = manager.list_models()

        assert len(models) >= 2
        model_names = [m["name"] for m in models]
        assert "cached-model" in model_names
        assert "registered-model" in model_names

    def test_list_models_with_filter(self, manager):
        """Test listing models with source filter."""
        manager.registry.list_models = Mock(
            return_value=[
                {"name": "model1", "source": "huggingface"},
                {"name": "model2", "source": "local"},
                {"name": "model3", "source": "huggingface"},
            ]
        )
        manager._list_cached_models = Mock(return_value=[])

        # Filter by source
        hf_models = manager.list_models(source="huggingface")
        assert len(hf_models) == 2

        local_models = manager.list_models(source="local")
        assert len(local_models) == 1
        assert local_models[0]["name"] == "model2"

    def test_load_model_from_path(self, manager, temp_dir):
        """Test loading model from local path."""
        model_path = temp_dir / "local_model"
        model_path.mkdir()

        # Create config
        config = {
            "model_type": "test",
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
        }
        (model_path / "config.json").write_text(json.dumps(config))

        # Mock loader
        mock_model = Mock()
        mock_model.config = ModelConfig(
            **{
                **config,
                "intermediate_size": 1024,
                "max_position_embeddings": 512,
            }
        )
        manager.loader.load_from_path = Mock(return_value=mock_model)
        manager.loader.get_model_info = Mock(
            return_value={
                "parameters": 1000000,
                "memory_footprint_mb": 100,
            }
        )

        model = manager.load_model(str(model_path))

        assert model == mock_model
        manager.loader.load_from_path.assert_called_once()

        # Check that model was registered
        manager.registry.register_model.assert_called_once()

    def test_load_model_from_huggingface(self, manager):
        """Test loading model from HuggingFace."""
        # Mock loader
        mock_model = Mock()
        manager.loader.load_from_huggingface = Mock(return_value=mock_model)
        manager.loader.get_model_info = Mock(
            return_value={
                "parameters": 125000000,
                "memory_footprint_mb": 250,
            }
        )

        model = manager.load_model("gpt2", load_in_4bit=True)

        assert model == mock_model
        manager.loader.load_from_huggingface.assert_called_once_with(
            "gpt2",
            load_in_4bit=True,
            load_in_8bit=False,
        )

    def test_delete_model(self, manager, temp_dir):
        """Test deleting a model from cache."""
        # Create model in cache
        cache_dir = temp_dir / "cache"
        model_dir = cache_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").touch()
        (model_dir / "model.bin").touch()

        assert model_dir.exists()

        # Delete model
        success = manager.delete_model("test-model")

        assert success is True
        assert not model_dir.exists()

    def test_delete_nonexistent_model(self, manager):
        """Test deleting non-existent model."""
        success = manager.delete_model("nonexistent-model")
        assert success is False

    def test_get_model_info(self, manager, temp_dir):
        """Test getting model information."""
        # Create model in cache
        cache_dir = temp_dir / "cache"
        model_dir = cache_dir / "test-model"
        model_dir.mkdir(parents=True)

        config = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        # Create some weight files
        (model_dir / "model.bin").write_text("x" * 1000)
        (model_dir / "tokenizer.json").write_text("{}")

        info = manager.get_model_info("test-model")

        assert info is not None
        assert info["name"] == "test-model"
        assert info["model_type"] == "llama"
        assert info["vocab_size"] == 32000
        assert info["hidden_size"] == 4096
        assert info["num_layers"] == 32
        assert info["num_heads"] == 32
        assert "config.json" in info["files"]
        assert "model.bin" in info["files"]
        assert info["size_gb"] > 0

    def test_get_model_info_not_found(self, manager):
        """Test getting info for non-existent model."""
        info = manager.get_model_info("nonexistent-model")
        assert info is None

    def test_estimate_memory_usage(self, manager, temp_dir):
        """Test memory usage estimation."""
        # Create model config
        cache_dir = temp_dir / "cache"
        model_dir = cache_dir / "test-model"
        model_dir.mkdir(parents=True)

        config = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_hidden_layers": 4,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        memory = manager.estimate_memory_usage(
            "test-model", batch_size=2, sequence_length=512, training=True
        )

        assert "model_gb" in memory
        assert "activations_gb" in memory
        assert "gradients_gb" in memory
        assert "optimizer_gb" in memory
        assert "total_gb" in memory

        # Basic sanity checks
        assert memory["model_gb"] > 0
        assert memory["total_gb"] > memory["model_gb"]

        # Training should require more memory
        memory_inference = manager.estimate_memory_usage(
            "test-model", batch_size=2, sequence_length=512, training=False
        )

        assert memory["total_gb"] > memory_inference["total_gb"]
        assert memory_inference["gradients_gb"] == 0
        assert memory_inference["optimizer_gb"] == 0

    def test_estimate_memory_usage_not_found(self, manager):
        """Test memory estimation for non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            manager.estimate_memory_usage("nonexistent-model")

    def test_convert_model_not_implemented(self, manager):
        """Test model conversion (not implemented)."""
        manager.load_model = Mock(return_value=Mock())

        # Test GGUF conversion
        with pytest.raises(NotImplementedError, match="GGUF conversion"):
            manager.convert_model("test-model", "gguf")

        # Test ONNX conversion
        with pytest.raises(NotImplementedError, match="ONNX conversion"):
            manager.convert_model("test-model", "onnx")

        # Test CoreML conversion
        with pytest.raises(NotImplementedError, match="CoreML conversion"):
            manager.convert_model("test-model", "coreml")

    def test_convert_model_invalid_format(self, manager):
        """Test conversion with invalid format."""
        manager.load_model = Mock(return_value=Mock())

        with pytest.raises(ValueError, match="Unsupported output format"):
            manager.convert_model("test-model", "invalid_format")


class TestEndToEndModelLoading:
    """End-to-end tests for model loading pipeline."""

    @pytest.mark.slow
    @pytest.mark.requires_mlx
    def test_load_small_model_mlx(self):
        """Test loading a small model with MLX backend."""
        from finetune.models.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(cache_dir=Path(tmpdir) / "cache")

            # Create a tiny test model
            model_dir = Path(tmpdir) / "tiny_model"
            model_dir.mkdir()

            config = {
                "model_type": "gpt2",
                "vocab_size": 100,
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
            }
            (model_dir / "config.json").write_text(json.dumps(config))

            # Create minimal weights
            import mlx.core as mx

            weights = {
                "wte.weight": mx.random.normal((100, 32)),
                "wpe.weight": mx.random.normal((128, 32)),
                "ln_f.weight": mx.ones(32),
            }
            mx.save(str(model_dir / "model.safetensors"), weights)

            # Load model
            model = manager.load_model(str(model_dir))

            assert model is not None
            assert model.config.vocab_size == 100
            assert model.config.hidden_size == 32

    @pytest.mark.slow
    @pytest.mark.requires_torch
    @pytest.mark.requires_transformers
    def test_load_small_model_pytorch(self):
        """Test loading a small model with PyTorch backend."""
        from finetune.models.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Force PyTorch backend
            with patch("finetune.models.manager.device_manager") as mock_device:
                mock_backend = Mock()
                mock_backend.name = "pytorch"
                mock_device.get_optimal_backend.return_value = mock_backend

                manager = ModelManager(cache_dir=Path(tmpdir) / "cache")

                # This would typically download from HuggingFace
                # For testing, we'd mock this
                with patch.object(manager.loader, "load_from_huggingface") as mock_load:
                    mock_model = Mock()
                    mock_model.config = ModelConfig(
                        model_type="gpt2",
                        vocab_size=50257,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        max_position_embeddings=1024,
                    )
                    mock_load.return_value = mock_model

                    model = manager.load_model("gpt2")

                    assert model is not None
                    mock_load.assert_called_once()
