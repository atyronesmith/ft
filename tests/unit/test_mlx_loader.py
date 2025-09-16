"""
Unit tests for MLX model loader.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np

from finetune.models.base import ModelConfig


class TestMLXModelLoader:
    """Test MLX model loader."""

    @pytest.fixture
    def loader(self, temp_dir):
        """Create a loader instance."""
        with patch("finetune.models.mlx_loader.MLX_AVAILABLE", True):
            from finetune.models.mlx_loader import MLXModelLoader

            return MLXModelLoader(cache_dir=temp_dir / "cache")

    def test_loader_initialization(self, temp_dir):
        """Test loader initialization."""
        with patch("finetune.models.mlx_loader.MLX_AVAILABLE", True):
            from finetune.models.mlx_loader import MLXModelLoader

            loader = MLXModelLoader(cache_dir=temp_dir / "cache")
            assert loader.cache_dir == temp_dir / "cache"
            assert loader.cache_dir.exists()

    def test_loader_requires_mlx(self):
        """Test that loader requires MLX."""
        with patch("finetune.models.mlx_loader.MLX_AVAILABLE", False):
            from finetune.models.mlx_loader import MLXModelLoader

            with pytest.raises(ImportError, match="MLX is not installed"):
                MLXModelLoader()

    def test_get_cached_path(self, loader):
        """Test cache path generation."""
        # Test basic model ID
        path = loader._get_cached_path("gpt2", None)
        assert path == loader.cache_dir / "gpt2"

        # Test model ID with organization
        path = loader._get_cached_path("meta-llama/Llama-2-7b", None)
        assert path == loader.cache_dir / "meta-llama--Llama-2-7b"

        # Test with revision
        path = loader._get_cached_path("gpt2", "main")
        assert path == loader.cache_dir / "gpt2--main"

    def test_should_skip_weight(self, loader):
        """Test weight skipping logic."""
        # Should skip these
        assert loader._should_skip_weight("model.rotary_emb.inv_freq")
        assert loader._should_skip_weight("transformer.masked_bias")
        assert loader._should_skip_weight("attn.bias")

        # Should not skip these
        assert not loader._should_skip_weight("model.embed_tokens.weight")
        assert not loader._should_skip_weight("model.layers.0.self_attn.q_proj.weight")

        # Should skip lm_head.weight for tied embeddings (GPT-2 models)
        assert loader._should_skip_weight("lm_head.weight")

    def test_get_name_mapping_llama(self, loader):
        """Test name mapping for Llama models."""
        mapping = loader._get_name_mapping("llama")

        assert "model.embed_tokens.weight" in mapping
        assert mapping["model.embed_tokens.weight"] == "embed_tokens.weight"
        assert mapping["lm_head.weight"] == "lm_head.weight"
        assert mapping["model.norm.weight"] == "norm.weight"

    def test_get_name_mapping_gpt(self, loader):
        """Test name mapping for GPT models."""
        mapping = loader._get_name_mapping("gpt2")

        # Check PyTorch -> MLX mappings
        assert "transformer.wte.weight" in mapping
        assert mapping["transformer.wte.weight"] == "wte.weight"
        assert mapping["transformer.wpe.weight"] == "wpe.weight"
        assert mapping["transformer.ln_f.weight"] == "ln_f.weight"

    @patch("finetune.models.mlx_loader.mx")
    def test_handle_special_weights(self, mock_mx, loader):
        """Test special weight handling."""
        # Create mock weight
        mock_weight = MagicMock()
        mock_weight.shape = (128, 256)  # out_features, in_features for PyTorch
        mock_weight.T = MagicMock()  # Transposed version

        # Test linear layer weights get transposed
        result = loader._handle_special_weights("q_proj.weight", mock_weight, "llama")
        assert result == mock_weight.T

        result = loader._handle_special_weights("gate_proj.weight", mock_weight, "llama")
        assert result == mock_weight.T

        # Test other weights don't get transposed
        mock_weight.shape = (128,)  # 1D weight
        result = loader._handle_special_weights("norm.weight", mock_weight, "llama")
        assert result == mock_weight  # Not transposed

    @patch("finetune.models.mlx_loader.mx")
    def test_convert_weights(self, mock_mx, loader):
        """Test weight conversion from PyTorch to MLX."""
        # Create mock PyTorch tensors
        mock_tensor = Mock()
        mock_tensor.detach().cpu().numpy.return_value = np.array([[1, 2], [3, 4]])

        source_weights = {
            "model.embed_tokens.weight": mock_tensor,
            "model.layers.0.self_attn.q_proj.weight": mock_tensor,
            "model.rotary_emb.inv_freq": mock_tensor,  # Should be skipped
        }

        # Mock MLX array creation with shape attribute
        mock_array = Mock()
        mock_array.shape = (2, 2)
        mock_array.T = Mock()
        mock_mx.array.return_value = mock_array

        # Mock torch import with proper tensor type
        with patch.dict("sys.modules", {"torch": Mock()}):
            mock_torch = Mock()
            mock_torch.Tensor = type(mock_tensor)  # Use the actual mock tensor type
            import sys
            sys.modules["torch"] = mock_torch

            # Convert weights
            mlx_weights = loader.convert_weights(source_weights, "llama")

            # Check that rotary_emb was skipped
            assert "model.rotary_emb.inv_freq" not in mlx_weights

            # Check that other weights were converted
            assert "embed_tokens.weight" in mlx_weights

            # Verify MLX array creation was called
            assert mock_mx.array.called

    def test_load_pytorch_weights_safetensors(self, loader, temp_dir):
        """Test loading weights from safetensors format."""
        model_path = temp_dir / "model"
        model_path.mkdir()

        # Create mock safetensors file
        (model_path / "model.safetensors").touch()

        # Mock both torch and safetensors
        with patch.dict("sys.modules", {"torch": Mock(), "safetensors": Mock()}):
            with patch("safetensors.safe_open") as mock_safe_open:
                mock_file = MagicMock()
                mock_file.keys.return_value = ["weight1", "weight2"]
                mock_file.get_tensor.return_value = Mock()
                mock_safe_open.return_value.__enter__.return_value = mock_file

                weights = loader._load_pytorch_weights(model_path)

                assert "weight1" in weights
                assert "weight2" in weights
                mock_safe_open.assert_called_once()

    def test_load_pytorch_weights_bin(self, loader, temp_dir):
        """Test loading weights from PyTorch bin format."""
        model_path = temp_dir / "model"
        model_path.mkdir()

        # Create mock bin file
        (model_path / "pytorch_model.bin").touch()

        # Mock torch in sys.modules before import
        with patch.dict("sys.modules", {"torch": Mock()}) as mock_modules:
            mock_torch = mock_modules["torch"]
            mock_torch.load.return_value = {"weight1": Mock(), "weight2": Mock()}

            weights = loader._load_pytorch_weights(model_path)

            assert "weight1" in weights
            assert "weight2" in weights
            mock_torch.load.assert_called_once()

    def test_load_pytorch_weights_sharded(self, loader, temp_dir):
        """Test loading sharded model weights."""
        model_path = temp_dir / "model"
        model_path.mkdir()

        # Create mock sharded files
        (model_path / "pytorch_model-00001-of-00002.bin").touch()
        (model_path / "pytorch_model-00002-of-00002.bin").touch()

        with patch("finetune.models.mlx_loader.glob.glob") as mock_glob:
            mock_glob.return_value = [
                str(model_path / "pytorch_model-00001-of-00002.bin"),
                str(model_path / "pytorch_model-00002-of-00002.bin"),
            ]

            # Mock torch directly using patch
            with patch.dict("sys.modules", {"torch": Mock()}):
                import sys
                mock_torch = sys.modules["torch"]
                mock_torch.load.side_effect = [
                    {"weight1": Mock()},
                    {"weight2": Mock()},
                ]

                weights = loader._load_pytorch_weights(model_path)

                assert "weight1" in weights
                assert "weight2" in weights
                assert mock_torch.load.call_count == 2

    def test_load_pytorch_weights_not_found(self, loader, temp_dir):
        """Test error when no weights found."""
        model_path = temp_dir / "model"
        model_path.mkdir()

        with patch.dict("sys.modules", {"torch": Mock()}):
            with pytest.raises(FileNotFoundError, match="No model weights found"):
                loader._load_pytorch_weights(model_path)

    @patch("finetune.models.mlx_loader.get_mlx_model")
    def test_load_mlx_model(self, mock_get_model, loader, temp_dir):
        """Test loading native MLX model."""
        model_path = temp_dir / "model"
        model_path.mkdir()

        # Create config
        config = {
            "model_type": "llama",
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 512,
            "max_position_embeddings": 512,
        }
        with open(model_path / "mlx_config.json", "w") as f:
            json.dump(config, f)

        # Create mock weights file
        (model_path / "mlx_model.safetensors").touch()

        # Mock model
        mock_model = Mock()
        mock_model.load = Mock()
        mock_get_model.return_value = mock_model

        model = loader._load_mlx_model(model_path)

        assert model == mock_model
        mock_model.load.assert_called_once_with(model_path)

    @patch("finetune.models.mlx_loader.get_mlx_model")
    def test_load_from_path_mlx_format(self, mock_get_model, loader, mock_mlx_model):
        """Test loading from path with MLX format."""
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_model.load = Mock()

        model = loader.load_from_path(mock_mlx_model)

        assert model == mock_model
        mock_model.load.assert_called_once()

    @patch("finetune.models.mlx_loader.get_mlx_model")
    def test_load_and_convert(
        self, mock_get_model, loader, mock_huggingface_model, mock_pytorch_weights
    ):
        """Test loading and converting HuggingFace model."""
        mock_model = Mock()
        mock_model.update = Mock()
        mock_model.num_parameters = 1000000
        mock_get_model.return_value = mock_model

        with patch.object(loader, "_load_pytorch_weights", return_value=mock_pytorch_weights):
            with patch.object(loader, "convert_weights", return_value={}):
                model = loader._load_and_convert(mock_huggingface_model)

        assert model == mock_model
        mock_model.update.assert_called_once()

    @patch("finetune.models.mlx_loader.get_mlx_model")
    def test_load_from_huggingface(self, mock_get_model, loader, temp_dir):
        """Test loading model from HuggingFace."""
        # Mock huggingface_hub
        with patch.dict("sys.modules", {"huggingface_hub": Mock()}):
            with patch("huggingface_hub.snapshot_download") as mock_download:
                # Setup mocks
                mock_download.return_value = str(temp_dir / "downloaded")
                mock_model = Mock()
                mock_model.num_parameters = 1000000
                mock_get_model.return_value = mock_model

                # Create mock downloaded model
                downloaded_path = temp_dir / "downloaded"
                downloaded_path.mkdir()
                config = {
                    "model_type": "gpt2",
                    "vocab_size": 50257,
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                }
                with open(downloaded_path / "config.json", "w") as f:
                    json.dump(config, f)
                (downloaded_path / "pytorch_model.bin").touch()

                with patch.object(loader, "_load_and_convert", return_value=mock_model):
                    model = loader.load_from_huggingface("gpt2")

                assert model == mock_model
                mock_download.assert_called_once()

    @patch("finetune.models.mlx_loader.logger")
    def test_quantize_model_placeholder(self, mock_logger, loader):
        """Test quantization (placeholder implementation)."""
        mock_model = Mock()
        mock_model.named_modules.return_value = []

        quantized = loader._quantize_model(mock_model, bits=4)

        assert quantized == mock_model
        mock_logger.info.assert_any_call("Quantizing model to 4-bit...")
        mock_logger.info.assert_any_call("Quantization complete")

    def test_get_model_info(self, loader):
        """Test getting model information."""
        mock_model = Mock()
        mock_model.config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )
        mock_model.num_parameters = 1000000
        mock_model.memory_footprint = 2000000

        info = loader.get_model_info(mock_model)

        assert info["type"] == "test"
        assert info["parameters"] == 1000000
        assert info["memory_footprint_mb"] == 2000000 / (1024**2)
        assert info["vocab_size"] == 1000
        assert info["hidden_size"] == 256
        assert info["num_layers"] == 4
        assert info["num_heads"] == 8
