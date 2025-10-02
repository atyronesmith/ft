"""
Unit tests for MLX model loader.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
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
        from finetune.models.base import ModelConfig
        config = ModelConfig(
            model_type="llama",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
            tie_word_embeddings=True
        )

        # Should skip these
        assert loader._should_skip_weight("model.rotary_emb.inv_freq", config)
        assert loader._should_skip_weight("transformer.masked_bias", config)
        assert loader._should_skip_weight("attn.bias", config)

        # Should not skip these
        assert not loader._should_skip_weight("model.embed_tokens.weight", config)
        assert not loader._should_skip_weight("model.layers.0.self_attn.q_proj.weight", config)

        # Should skip lm_head.weight for tied embeddings when tie_word_embeddings=True
        assert loader._should_skip_weight("lm_head.weight", config)

    def test_convert_to_nested_structure(self, loader):
        """Test conversion of flat weights to nested structure."""
        from finetune.models.base import ModelConfig
        config = ModelConfig(
            model_type="llama",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512
        )

        # Mock weights with flat HuggingFace structure
        flat_weights = {
            "model.embed_tokens.weight": "embed_weight",
            "model.layers.0.self_attn.q_proj.weight": "q_weight",
            "model.layers.0.self_attn.k_proj.weight": "k_weight",
            "model.norm.weight": "norm_weight",
            "lm_head.weight": "lm_weight"
        }

        nested = loader._convert_to_nested_structure(flat_weights, config)

        # Check nested structure is created correctly
        assert "model" in nested
        assert "lm_head.weight" in nested
        assert nested["lm_head.weight"] == "lm_weight"

        # Check model structure
        model_params = nested["model"]
        assert "embed_tokens" in model_params
        assert "layers" in model_params
        assert "norm" in model_params

        # Check layer structure (layers creates nested structure)
        # The method creates: model.layers.layers.0.self_attn...
        layers_outer = model_params["layers"]
        layers_inner = layers_outer["layers"]
        assert 0 in layers_inner
        layer_0 = layers_inner[0]
        assert "self_attn" in layer_0
        assert "q_proj" in layer_0["self_attn"]
        assert layer_0["self_attn"]["q_proj"]["weight"] == "q_weight"

    def test_handle_special_weights(self, loader):
        """Test special weight handling."""
        # Create mock weight
        mock_weight = MagicMock()
        mock_weight.shape = (128, 256)

        # Current implementation just returns weight as-is (no transposition)
        result = loader._handle_special_weights("q_proj.weight", mock_weight, "llama")
        assert result == mock_weight

        result = loader._handle_special_weights("gate_proj.weight", mock_weight, "llama")
        assert result == mock_weight

        # Test 1D weights
        mock_weight.shape = (128,)
        result = loader._handle_special_weights("norm.weight", mock_weight, "llama")
        assert result == mock_weight

    def test_convert_weights(self, loader):
        """Test weight conversion from PyTorch to MLX."""
        # Current implementation just returns the source weights as-is
        # since mx.load() handles conversion automatically
        from finetune.models.base import ModelConfig
        config = ModelConfig(
            model_type="llama",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512
        )

        source_weights = {
            "model.embed_tokens.weight": "embed_weight",
            "model.layers.0.self_attn.q_proj.weight": "q_weight",
        }

        # Convert weights (should return as-is in current implementation)
        converted = loader.convert_weights(source_weights, config)
        assert converted == source_weights

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

        with patch("glob.glob") as mock_glob:
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

    def test_load_from_path_hf_format(self, loader, temp_dir):
        """Test loading from path with HuggingFace format."""
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
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)

        # Create mock safetensors file
        (model_path / "model.safetensors").touch()

        # Mock the _load_hf_model_to_mlx method
        with patch.object(loader, "_load_hf_model_to_mlx") as mock_load_hf:
            mock_result = ("mock_model", "mock_tokenizer", "mock_config")
            mock_load_hf.return_value = mock_result

            result = loader.load_from_path(model_path)

            assert result == mock_result
            mock_load_hf.assert_called_once_with(model_path, None)

    def test_load_from_huggingface(self, loader, temp_dir):
        """Test loading model from HuggingFace."""
        # Mock huggingface_hub
        with patch.dict("sys.modules", {"huggingface_hub": Mock()}):
            with patch("huggingface_hub.snapshot_download") as mock_download:
                # Setup mocks
                mock_download.return_value = str(temp_dir / "downloaded")

                # Create mock downloaded model
                downloaded_path = temp_dir / "downloaded"
                downloaded_path.mkdir()
                config = {
                    "model_type": "llama",  # Use supported model type
                    "vocab_size": 32000,
                    "hidden_size": 4096,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                }
                with open(downloaded_path / "config.json", "w") as f:
                    json.dump(config, f)
                (downloaded_path / "model.safetensors").touch()

                # Mock the _load_hf_model_to_mlx method
                with patch.object(loader, "_load_hf_model_to_mlx") as mock_load_hf:
                    mock_result = ("mock_model", "mock_tokenizer", "mock_config")
                    mock_load_hf.return_value = mock_result

                    result = loader.load_from_huggingface("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

                    assert result == mock_result
                    mock_download.assert_called_once()
                    mock_load_hf.assert_called()


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
