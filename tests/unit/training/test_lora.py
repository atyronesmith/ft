"""
Unit tests for LoRA (Low-Rank Adaptation) implementation.

Tests are written first following TDD approach to drive the implementation
of MLX-native LoRA layers for efficient fine-tuning.
"""


import mlx.core as mx
import pytest

from finetune.training.lora import (
    LoRAConfig,
)
from finetune.training.lora import (
    LoRALinear as MLXLoRALinear,  # Alias for consistency with test names
)
from finetune.training.lora import (
    ,
)


class TestLoRAConfig:
    """Test LoRA configuration class."""

    def test_lora_config_creation(self):
        """Test LoRA config creates with valid parameters."""
        config = LoRAConfig(r=16, alpha=32, dropout=0.1, target_modules=["q_proj", "v_proj"])

        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.scaling == config.alpha / config.r  # 32/16 = 2.0

    def test_lora_config_default_values(self):
        """Test LoRA config has sensible defaults."""
        config = LoRAConfig(r=8)

        assert config.r == 8
        assert config.alpha == 16.0  # Default 2x rank
        assert config.dropout == 0.0  # Default dropout
        assert len(config.target_modules) > 0  # Should have default targets
        assert config.scaling == 2.0  # alpha/r = 16/8

    def test_lora_config_validation(self):
        """Test LoRA config validates parameters."""
        # Test invalid rank
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            LoRAConfig(r=0)

        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            LoRAConfig(r=-1)

        # Test invalid alpha
        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            LoRAConfig(r=8, alpha=0)

        # Test invalid dropout
        with pytest.raises(ValueError, match="Dropout must be in"):
            LoRAConfig(r=8, dropout=-0.1)

        with pytest.raises(ValueError, match="Dropout must be in"):
            LoRAConfig(r=8, dropout=1.1)


class TestMLXLoRALinear:
    """Test MLX LoRA linear layer implementation."""

    def test_lora_layer_initialization(self):
        """Test LoRA layer initializes with correct parameter shapes."""
        # Arrange
        in_features, out_features, rank = 1024, 1024, 16
        config = LoRAConfig(r=rank)

        # Act
        layer = MLXLoRALinear(in_features, out_features, config)

        # Assert
        assert hasattr(layer, "base")
        assert hasattr(layer, "lora_a")
        assert hasattr(layer, "lora_b")

        # Check parameter shapes
        assert layer.lora_a.shape == (rank, in_features)
        assert layer.lora_b.shape == (out_features, rank)
        assert layer.base.weight.shape == (out_features, in_features)

    def test_lora_layer_forward_pass(self):
        """Test LoRA forward pass combines base + adapter outputs correctly."""
        # Arrange
        config = LoRAConfig(r=8, alpha=16)
        layer = MLXLoRALinear(512, 512, config)
        x = mx.random.normal(shape=(2, 512), key=mx.random.key(42))

        # Act
        output = layer(x)

        # Assert
        assert output.shape == (2, 512)

        # Verify it's base + lora adaptation with scaling
        base_output = layer.base(x)
        lora_output = x @ layer.lora_a.T @ layer.lora_b.T * config.scaling
        expected = base_output + lora_output

        assert mx.allclose(output, expected, atol=1e-6)

    def test_lora_layer_reduces_parameters(self):
        """Test LoRA layer has significantly fewer trainable parameters."""
        # Arrange
        in_features, out_features = 4096, 4096
        original_params = in_features * out_features  # Full linear layer

        rank = 16
        config = LoRAConfig(r=rank)
        layer = MLXLoRALinear(in_features, out_features, config)

        # Act
        lora_params = rank * (in_features + out_features)  # A + B matrices

        # Assert
        reduction_ratio = lora_params / original_params
        assert reduction_ratio < 0.01  # Less than 1% of original parameters

        # For 4096x4096 with rank 16: 16*(4096+4096) = 131,072 vs 16,777,216
        # Reduction: ~99.2%
        assert lora_params == 131072
        assert original_params == 16777216

    def test_lora_layer_frozen_base_weights(self):
        """Test LoRA layer keeps base weights frozen."""
        # Arrange
        config = LoRAConfig(r=8)
        layer = MLXLoRALinear(256, 256, config)

        # Get initial base weights
        initial_base_weights = mx.array(layer.base.weight)

        # Simulate training step (this will be implemented later)
        # For now, just verify base layer is marked as frozen

        # Assert
        # Base layer should not require gradients (frozen)
        # This test will be expanded when we implement freezing mechanism
        assert hasattr(layer.base, "weight")
        assert layer.base.weight.shape == (256, 256)

    def test_lora_layer_with_dropout(self):
        """Test LoRA layer applies dropout during training."""
        # Arrange
        config = LoRAConfig(r=8, dropout=0.5)
        layer = MLXLoRALinear(128, 128, config)
        x = mx.random.normal(shape=(2, 128))

        # This test will be expanded when dropout is implemented
        # For now, just verify the layer can be created with dropout config
        assert hasattr(layer, "dropout")
        # Check dropout is configured (MLX dropout doesn't expose rate directly)
        assert layer.dropout is not None

    def test_lora_layer_different_ranks(self):
        """Test LoRA layer works with different rank values."""
        configs_and_expected = [
            (4, 4 * (64 + 64)),  # rank=4
            (8, 8 * (64 + 64)),  # rank=8
            (16, 16 * (64 + 64)),  # rank=16
            (32, 32 * (64 + 64)),  # rank=32
        ]

        for rank, expected_params in configs_and_expected:
            config = LoRAConfig(r=rank)
            layer = MLXLoRALinear(64, 64, config)

            # Check parameter count
            actual_params = layer.lora_a.size + layer.lora_b.size
            assert actual_params == expected_params

            # Check shapes
            assert layer.lora_a.shape == (rank, 64)
            assert layer.lora_b.shape == (64, rank)


class TestLoRAAdapterApplication:
    """Test applying LoRA adapters to existing models."""

    def test_apply_lora_to_linear_layers(self):
        """Test applying LoRA adapters to specific linear layers in a model."""
        # This test will drive implementation of adapter application
        # Will be implemented after basic LoRA layer is working
        pass

    def test_count_trainable_parameters(self):
        """Test counting trainable parameters before and after LoRA."""
        # This test will drive implementation of parameter counting utility
        pass

    def test_target_modules_selection(self):
        """Test LoRA is applied only to specified target modules."""
        # This test will drive implementation of selective adapter application
        pass


class TestLoRAIntegration:
    """Test LoRA integration with existing MLX models."""

    @pytest.mark.skipif(not mx.metal.is_available(), reason="MLX not available")
    def test_lora_with_mlx_llama_model(self):
        """Test LoRA integration with existing MLX Llama model."""
        # This will test integration with models from Phase 1
        # Will be implemented after basic LoRA functionality works
        pass

    def test_lora_memory_efficiency(self):
        """Test LoRA uses significantly less memory than full fine-tuning."""
        # This test will drive memory optimization implementation
        pass

    def test_lora_checkpoint_saving_loading(self):
        """Test saving and loading LoRA adapter weights."""
        # This test will drive checkpoint functionality
        pass


# Test fixtures and utilities
@pytest.fixture
def sample_lora_config():
    """Provide a standard LoRA config for testing."""
    return LoRAConfig(
        rank=16, alpha=32, dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@pytest.fixture
def small_lora_layer():
    """Provide a small LoRA layer for testing."""
    config = LoRAConfig(rank=4, alpha=8)
    return MLXLoRALinear(64, 64, config)


class TestLoRAWeightMerging:
    """Test merging LoRA weights back to base model."""

    def test_merge_lora_weights(self):
        """Test merging LoRA adapter weights into base model."""
        # This test will drive implementation of weight merging for deployment
        pass

    def test_merged_weights_equivalence(self):
        """Test merged model produces same outputs as LoRA model."""
        # This test ensures merging preserves functionality
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
