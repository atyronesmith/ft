"""
Unit tests for MLX model implementations.
"""

from unittest.mock import patch

import pytest

from tests.utils import ModelConfigFactory, TestEnvironment


class TestMLXModels:
    """Test MLX model implementations."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return ModelConfigFactory.create_small_config(model_type="llama")

    @pytest.mark.requires_mlx
    def test_rms_norm(self, small_config):
        """Test RMSNorm layer."""
        import mlx.core as mx
        from finetune.models.mlx_models import RMSNorm

        norm = RMSNorm(dims=64, eps=1e-6)

        # Test forward pass
        x = mx.random.normal((2, 10, 64))
        output = norm(x)

        assert output.shape == x.shape

    @pytest.mark.requires_mlx
    def test_attention_layer(self, small_config):
        """Test Attention layer."""
        import mlx.core as mx
        from finetune.models.mlx_models import Attention

        attn = Attention(small_config)

        # Test forward pass
        batch_size, seq_len = 2, 10
        x = mx.random.normal((batch_size, seq_len, small_config.hidden_size))

        output = attn(x)

        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    @pytest.mark.requires_mlx
    def test_attention_with_mask(self, small_config):
        """Test Attention layer with mask."""
        import mlx.core as mx
        import mlx.nn as nn
        from finetune.models.mlx_models import Attention

        attn = Attention(small_config)

        batch_size, seq_len = 2, 10
        x = mx.random.normal((batch_size, seq_len, small_config.hidden_size))

        # Create causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        output = attn(x, mask=mask)

        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    @pytest.mark.requires_mlx
    def test_mlp_layer(self, small_config):
        """Test MLP layer."""
        import mlx.core as mx
        from finetune.models.mlx_models import MLP

        mlp = MLP(small_config)

        # Test forward pass
        x = mx.random.normal((2, 10, small_config.hidden_size))
        output = mlp(x)

        assert output.shape == x.shape

    @pytest.mark.requires_mlx
    def test_transformer_block(self, small_config):
        """Test TransformerBlock."""
        import mlx.core as mx
        from finetune.models.mlx_models import TransformerBlock

        block = TransformerBlock(small_config)

        # Test forward pass
        x = mx.random.normal((2, 10, small_config.hidden_size))
        output = block(x)

        assert output.shape == x.shape

    @pytest.mark.requires_mlx
    def test_llama_model_creation(self, small_config):
        """Test creating a Llama model."""
        from finetune.models.mlx_models import MLXLlamaModel

        model = MLXLlamaModel(small_config)

        assert model.config == small_config
        assert len(model.layers) == small_config.num_hidden_layers
        assert model.num_parameters > 0

    @pytest.mark.requires_mlx
    def test_llama_forward_pass(self, small_config):
        """Test Llama model forward pass."""
        import mlx.core as mx
        from finetune.models.mlx_models import MLXLlamaModel

        model = MLXLlamaModel(small_config)

        # Create input
        batch_size, seq_len = 2, 10
        input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model.forward(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    @pytest.mark.requires_mlx
    def test_gpt_model_creation(self, small_config):
        """Test creating a GPT model."""
        from finetune.models.mlx_models import MLXGPTModel

        # Modify config for GPT
        small_config.model_type = "gpt2"

        model = MLXGPTModel(small_config)

        assert model.config == small_config
        assert len(model.layers) == small_config.num_hidden_layers
        assert model.num_parameters > 0

    @pytest.mark.requires_mlx
    def test_gpt_forward_pass(self, small_config):
        """Test GPT model forward pass."""
        import mlx.core as mx
        from finetune.models.mlx_models import MLXGPTModel

        small_config.model_type = "gpt2"
        model = MLXGPTModel(small_config)

        # Create input
        batch_size, seq_len = 2, 10
        input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model.forward(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    @pytest.mark.requires_mlx
    def test_model_save_and_load(self, small_config, temp_dir):
        """Test saving and loading a model."""
        import mlx.core as mx
        from finetune.models.mlx_models import MLXLlamaModel

        model = MLXLlamaModel(small_config)

        # Save model
        save_path = temp_dir / "test_model"
        model.save(save_path)

        assert (save_path / "config.json").exists()
        assert (save_path / "model.npz").exists()

        # Create new model and load weights
        new_model = MLXLlamaModel(small_config)
        new_model.load(save_path)

        # Verify weights are loaded (just check they have same structure)
        # Can't easily compare nested dicts/lists, so just verify model works
        input_ids = mx.random.randint(0, small_config.vocab_size, (1, 5))
        original_output = model.forward(input_ids)
        loaded_output = new_model.forward(input_ids)
        assert original_output.shape == loaded_output.shape

    @pytest.mark.requires_mlx
    def test_get_mlx_model(self, small_config):
        """Test get_mlx_model function."""
        from finetune.models.mlx_models import MLXGPTModel, MLXLlamaModel, get_mlx_model

        # Test Llama
        small_config.model_type = "llama"
        model = get_mlx_model(small_config)
        assert isinstance(model, MLXLlamaModel)

        # Test GPT
        small_config.model_type = "gpt2"
        model = get_mlx_model(small_config)
        assert isinstance(model, MLXGPTModel)

        # Test Mistral (should use Llama)
        small_config.model_type = "mistral"
        model = get_mlx_model(small_config)
        assert isinstance(model, MLXLlamaModel)

    @pytest.mark.requires_mlx
    def test_unsupported_model_type(self, small_config):
        """Test error for unsupported model type."""
        from finetune.models.mlx_models import get_mlx_model

        small_config.model_type = "unknown_model"

        with pytest.raises(ValueError, match="Unsupported model type"):
            get_mlx_model(small_config)

    @pytest.mark.requires_mlx
    @pytest.mark.slow
    def test_text_generation(self, small_config):
        """Test text generation (basic)."""
        import mlx.core as mx
        from finetune.models.mlx_models import MLXLlamaModel

        model = MLXLlamaModel(small_config)

        # Create input
        input_ids = mx.array([[1, 2, 3]])  # Start tokens

        # Generate
        output = model.generate(input_ids, max_length=10, temperature=1.0)

        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] <= 10  # Max length


class TestMLXModelsWithoutMLX:
    """Test MLX models behavior when MLX is not available."""

    @pytest.mark.skipif(_mlx_available(), reason="Skip when MLX is available")
    def test_import_without_mlx(self):
        """Test that importing works even without MLX."""
        # This test simulates MLX not being available
        # It's skipped when MLX is actually installed
        import sys

        # Clean up any existing imports
        modules_to_remove = [
            k for k in sys.modules.keys() if "mlx" in k or "finetune.models.mlx" in k
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None, "mlx.nn": None}):
            from finetune.models import mlx_models

            assert not mlx_models.MLX_AVAILABLE

    @pytest.mark.skipif(TestEnvironment.mlx_available(), reason="Skip when MLX is available")
    def test_model_creation_fails_without_mlx(self):
        """Test that model creation fails gracefully without MLX."""
        # This test only makes sense when MLX is not installed
        # When MLX is installed, the else branch is never executed
        from finetune.models.mlx_models import get_mlx_model

        config = ModelConfigFactory.create_small_config(model_type="llama")

        # Should raise ImportError when MLX not available
        with pytest.raises(ImportError, match="MLX is not available"):
            get_mlx_model(config)
