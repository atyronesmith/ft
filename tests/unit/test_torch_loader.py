"""
Unit tests for PyTorch model loader and fallback.
"""

from unittest.mock import Mock, PropertyMock, patch

import pytest
from finetune.models.base import ModelConfig


class TestPyTorchModel:
    """Test PyTorchModel wrapper class."""

    @pytest.fixture
    def mock_torch_model(self):
        """Create a mock PyTorch model."""
        model = Mock()
        # Make parameters() return a new iterator each time it's called
        mock_param = Mock(device="cpu", numel=Mock(return_value=1000))
        model.parameters = Mock(side_effect=lambda: iter([mock_param]))
        model.save_pretrained = Mock()
        return model

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return ModelConfig(
            model_type="gpt2",
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
        )

    @pytest.mark.requires_torch
    def test_pytorch_model_init(self, mock_torch_model, config):
        """Test PyTorchModel initialization."""
        from finetune.models.torch_loader import PyTorchModel

        model = PyTorchModel(mock_torch_model, config)

        assert model.model == mock_torch_model
        assert model.config == config
        assert model.tokenizer is None
        assert model.device == "cpu"

    @pytest.mark.requires_torch
    def test_pytorch_model_with_tokenizer(self, mock_torch_model, config):
        """Test PyTorchModel with tokenizer."""
        from finetune.models.torch_loader import PyTorchModel

        mock_tokenizer = Mock()
        model = PyTorchModel(mock_torch_model, config, tokenizer=mock_tokenizer)

        assert model.tokenizer == mock_tokenizer

    @pytest.mark.requires_torch
    def test_pytorch_model_forward(self, mock_torch_model, config):
        """Test forward pass."""
        from finetune.models.torch_loader import PyTorchModel

        mock_output = Mock(logits="test_logits")
        mock_torch_model.return_value = mock_output

        model = PyTorchModel(mock_torch_model, config)

        input_ids = Mock()
        logits = model.forward(input_ids)

        assert logits == "test_logits"
        mock_torch_model.assert_called_once_with(input_ids=input_ids)

    @pytest.mark.requires_torch
    def test_pytorch_model_generate(self, mock_torch_model, config):
        """Test text generation."""
        from finetune.models.torch_loader import PyTorchModel

        mock_torch_model.generate.return_value = "generated_text"

        model = PyTorchModel(mock_torch_model, config)

        input_ids = Mock()
        output = model.generate(input_ids, max_length=50)

        assert output == "generated_text"
        mock_torch_model.generate.assert_called_once_with(input_ids=input_ids, max_length=50)

    @pytest.mark.requires_torch
    def test_pytorch_model_save(self, mock_torch_model, config, temp_dir):
        """Test model saving."""
        from finetune.models.torch_loader import PyTorchModel

        mock_tokenizer = Mock()
        model = PyTorchModel(mock_torch_model, config, tokenizer=mock_tokenizer)

        save_path = temp_dir / "saved_model"
        model.save(save_path)

        # Check config was saved
        assert (save_path / "config.json").exists()

        # Check model save was called
        mock_torch_model.save_pretrained.assert_called_once_with(save_path)

        # Check tokenizer save was called
        mock_tokenizer.save_pretrained.assert_called_once_with(save_path)

    @pytest.mark.requires_torch
    def test_pytorch_model_load(self, mock_torch_model, config, temp_dir):
        """Test model loading."""
        from finetune.models.torch_loader import PyTorchModel

        # Create model path with tokenizer config
        model_path = temp_dir / "model"
        model_path.mkdir()
        (model_path / "tokenizer_config.json").touch()

        with patch("finetune.models.torch_loader.AutoModelForCausalLM") as mock_auto:
            with patch("finetune.models.torch_loader.AutoTokenizer") as mock_tokenizer:
                mock_auto.from_pretrained.return_value = Mock()
                mock_tokenizer.from_pretrained.return_value = Mock()

                model = PyTorchModel(mock_torch_model, config)
                model.load(model_path)

                mock_auto.from_pretrained.assert_called_once_with(model_path)
                mock_tokenizer.from_pretrained.assert_called_once_with(model_path)

    @pytest.mark.requires_torch
    def test_pytorch_model_num_parameters(self, config):
        """Test parameter counting."""
        import torch
        from finetune.models.torch_loader import PyTorchModel

        # Create mock model with real tensors
        mock_model = Mock()
        mock_model.parameters = Mock(
            side_effect=lambda: iter(
                [
                    torch.zeros(100, 200),  # 20,000 params
                    torch.zeros(50),  # 50 params
                    torch.zeros(10, 10, 10),  # 1,000 params
                ]
            )
        )

        model = PyTorchModel(mock_model, config)

        assert model.num_parameters == 21050

    @pytest.mark.requires_torch
    def test_pytorch_model_memory_footprint(self, config):
        """Test memory footprint calculation."""
        import torch
        from finetune.models.torch_loader import PyTorchModel

        # Create mock model with different dtypes
        mock_model = Mock()
        params = [
            Mock(numel=lambda: 1000, dtype=torch.float32),  # 4000 bytes
            Mock(numel=lambda: 1000, dtype=torch.float16),  # 2000 bytes
            Mock(numel=lambda: 1000, dtype=torch.int8),  # 1000 bytes
        ]
        mock_model.parameters = Mock(side_effect=lambda: iter(params))

        model = PyTorchModel(mock_model, config)

        assert model.memory_footprint == 7000


class TestPyTorchModelLoader:
    """Test PyTorchModelLoader class."""

    @pytest.fixture
    def loader(self, temp_dir):
        """Create loader instance."""
        with patch("finetune.models.torch_loader.TORCH_AVAILABLE", True):
            with patch("finetune.models.torch_loader.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = True
                mock_torch.device.return_value = Mock()

                from finetune.models.torch_loader import PyTorchModelLoader

                return PyTorchModelLoader(cache_dir=temp_dir / "cache")

    def test_loader_initialization_cuda(self, temp_dir):
        """Test loader initialization with CUDA."""
        with patch("finetune.models.torch_loader.TORCH_AVAILABLE", True):
            with patch("finetune.models.torch_loader.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.device.return_value = Mock()

                from finetune.models.torch_loader import PyTorchModelLoader

                loader = PyTorchModelLoader()

                mock_torch.device.assert_called_with("cuda")

    def test_loader_initialization_mps(self, temp_dir):
        """Test loader initialization with MPS."""
        with patch("finetune.models.torch_loader.TORCH_AVAILABLE", True):
            with patch("finetune.models.torch_loader.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = True
                mock_torch.device.return_value = Mock()

                from finetune.models.torch_loader import PyTorchModelLoader

                loader = PyTorchModelLoader()

                mock_torch.device.assert_called_with("mps")

    def test_loader_initialization_cpu(self, temp_dir):
        """Test loader initialization with CPU only."""
        with patch("finetune.models.torch_loader.TORCH_AVAILABLE", True):
            with patch("finetune.models.torch_loader.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False
                mock_torch.device.return_value = Mock()

                from finetune.models.torch_loader import PyTorchModelLoader

                loader = PyTorchModelLoader()

                mock_torch.device.assert_called_with("cpu")

    def test_loader_requires_torch(self):
        """Test that loader requires PyTorch."""
        with patch("finetune.models.torch_loader.TORCH_AVAILABLE", False):
            from finetune.models.torch_loader import PyTorchModelLoader

            with pytest.raises(ImportError, match="PyTorch is not installed"):
                PyTorchModelLoader()

    @patch("finetune.models.torch_loader.AutoConfig")
    @patch("finetune.models.torch_loader.AutoModelForCausalLM")
    @patch("finetune.models.torch_loader.AutoTokenizer")
    def test_load_from_huggingface(self, mock_tokenizer, mock_model, mock_config, loader):
        """Test loading model from HuggingFace."""
        # Setup mocks
        mock_config.from_pretrained.return_value = Mock(
            to_dict=lambda: {
                "model_type": "gpt2",
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
            }
        )
        mock_hf_model = Mock()
        mock_param = Mock(numel=lambda: 1000, device="cpu")
        mock_hf_model.parameters = Mock(side_effect=lambda: iter([mock_param]))
        # Mock the .to() method to return the same model with parameters
        mock_hf_model.to.return_value = mock_hf_model
        mock_model.from_pretrained.return_value = mock_hf_model
        mock_tokenizer.from_pretrained.return_value = Mock()

        model = loader.load_from_huggingface("gpt2")

        assert model is not None
        mock_config.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()

    @patch("finetune.models.torch_loader.AutoConfig")
    @patch("finetune.models.torch_loader.AutoModelForCausalLM")
    @patch("finetune.models.torch_loader.AutoTokenizer")
    @patch("finetune.models.torch_loader.BitsAndBytesConfig")
    def test_load_from_huggingface_quantized(
        self, mock_bnb, mock_tokenizer, mock_model, mock_config, loader
    ):
        """Test loading quantized model."""
        # Setup mocks
        mock_config.from_pretrained.return_value = Mock(
            to_dict=lambda: {
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            }
        )
        mock_empty_model = Mock()
        mock_empty_model.parameters = Mock(
            side_effect=lambda: iter([Mock(device="cpu", numel=lambda: 0)])
        )
        mock_empty_model.to.return_value = mock_empty_model
        mock_model.from_pretrained.return_value = mock_empty_model
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_bnb.return_value = Mock()

        model = loader.load_from_huggingface("meta-llama/Llama-2-7b", load_in_4bit=True)

        # Check that quantization config was created
        mock_bnb.assert_called_once()
        call_kwargs = mock_bnb.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["load_in_8bit"] is False

    @patch("finetune.models.torch_loader.AutoConfig")
    @patch("finetune.models.torch_loader.AutoModelForCausalLM")
    @patch("finetune.models.torch_loader.AutoTokenizer")
    def test_load_from_path(self, mock_tokenizer, mock_model, mock_config, loader, temp_dir):
        """Test loading model from local path."""
        model_path = temp_dir / "local_model"
        model_path.mkdir()
        (model_path / "tokenizer_config.json").touch()

        # Setup mocks
        mock_config.from_pretrained.return_value = Mock(
            to_dict=lambda: {
                "model_type": "gpt2",
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
            }
        )
        mock_empty_model = Mock()
        mock_empty_model.parameters = Mock(
            side_effect=lambda: iter([Mock(device="cpu", numel=lambda: 0)])
        )
        mock_empty_model.to.return_value = mock_empty_model
        mock_model.from_pretrained.return_value = mock_empty_model
        mock_tokenizer.from_pretrained.return_value = Mock()

        model = loader.load_from_path(model_path)

        assert model is not None
        mock_config.from_pretrained.assert_called_with(model_path)
        mock_model.from_pretrained.assert_called()
        mock_tokenizer.from_pretrained.assert_called_with(model_path)

    def test_convert_weights(self, loader):
        """Test weight conversion (no-op for PyTorch)."""
        weights = {"layer1": "weight1", "layer2": "weight2"}
        converted = loader.convert_weights(weights)

        assert converted == weights

    def test_get_model_info(self, loader):
        """Test getting model information."""
        from finetune.models.torch_loader import PyTorchModel

        mock_torch_model = Mock()
        mock_torch_model.parameters = Mock(
            side_effect=lambda: iter([Mock(dtype="float16", device="mps")])
        )

        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
        )

        model = PyTorchModel(mock_torch_model, config)
        # Mock the properties using PropertyMock
        with patch.object(
            PyTorchModel, "num_parameters", new_callable=PropertyMock
        ) as mock_num_params:
            with patch.object(
                PyTorchModel, "memory_footprint", new_callable=PropertyMock
            ) as mock_mem:
                mock_num_params.return_value = 1000000
                mock_mem.return_value = 2000000
                model.device = "mps"

                info = loader.get_model_info(model)

                assert info["type"] == "test"
                assert info["parameters"] == 1000000
                assert info["memory_footprint_mb"] == 2000000 / (1024**2)
                assert info["vocab_size"] == 1000
                assert info["hidden_size"] == 256
                assert info["num_layers"] == 4
                assert info["num_heads"] == 8
                assert info["device"] == "mps"

    def test_get_model_info_non_pytorch_model(self, loader):
        """Test get_model_info with non-PyTorch model."""
        mock_model = Mock()
        info = loader.get_model_info(mock_model)

        assert info == {}
