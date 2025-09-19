"""Unit tests for LoRA implementation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from finetune.training.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora_to_model,
    get_lora_trainable_params,
    load_lora_weights,
    save_lora_weights,
)
from finetune.training.trainer import LoRATrainer, SimpleDataLoader, TrainingConfig


class TestLoRAConfig(unittest.TestCase):
    """Test LoRA configuration."""

    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        self.assertEqual(config.r, 8)
        self.assertEqual(config.alpha, 16.0)
        self.assertEqual(config.dropout, 0.0)
        self.assertFalse(config.use_rslora)
        self.assertFalse(config.use_dora)
        self.assertIsNotNone(config.target_modules)

    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(r=16, alpha=32.0, dropout=0.1, use_rslora=True)
        self.assertEqual(config.r, 16)
        self.assertEqual(config.alpha, 32.0)
        self.assertEqual(config.dropout, 0.1)
        self.assertTrue(config.use_rslora)

    def test_invalid_rank(self):
        """Test invalid rank raises error."""
        with self.assertRaises(ValueError):
            LoRAConfig(r=0)
        with self.assertRaises(ValueError):
            LoRAConfig(r=-1)

    def test_invalid_alpha(self):
        """Test invalid alpha raises error."""
        with self.assertRaises(ValueError):
            LoRAConfig(alpha=0)
        with self.assertRaises(ValueError):
            LoRAConfig(alpha=-1)

    def test_invalid_dropout(self):
        """Test invalid dropout raises error."""
        with self.assertRaises(ValueError):
            LoRAConfig(dropout=-0.1)
        with self.assertRaises(ValueError):
            LoRAConfig(dropout=1.0)

    def test_scaling_calculation(self):
        """Test LoRA scaling calculation."""
        # Standard scaling
        config = LoRAConfig(r=8, alpha=16.0, use_rslora=False)
        self.assertEqual(config.scaling, 2.0)

        # RSLoRA scaling
        config = LoRAConfig(r=4, alpha=16.0, use_rslora=True)
        self.assertEqual(config.scaling, 8.0)


@patch("finetune.training.lora.mx")
@patch("finetune.training.lora.nn")
class TestLoRALinear(unittest.TestCase):
    """Test LoRA linear layer."""

    def test_initialization(self, mock_nn, mock_mx):
        """Test LoRA linear layer initialization."""
        # Setup mocks
        mock_mx.random.normal.return_value = Mock(shape=(8, 768))
        mock_mx.zeros.return_value = Mock(shape=(768, 8))
        mock_mx.ones.return_value = Mock(shape=(768, 1))

        mock_linear = Mock()
        mock_nn.Linear.return_value = mock_linear

        config = LoRAConfig(r=8, dropout=0.1, use_dora=True)
        layer = LoRALinear(768, 768, config, bias=False)

        # Check initialization
        mock_nn.Linear.assert_called_once_with(768, 768, bias=False)
        mock_linear.freeze.assert_called_once()
        mock_mx.random.normal.assert_called_once()
        mock_mx.zeros.assert_called_once()
        mock_mx.ones.assert_called_once()  # For DoRA

    def test_forward_pass(self, mock_nn, mock_mx):
        """Test forward pass through LoRA layer."""
        # Setup mocks for array operations
        mock_array = Mock()
        mock_array.T = mock_array  # Transpose returns self
        mock_array.__matmul__ = Mock(return_value=mock_array)
        mock_array.__mul__ = Mock(return_value=mock_array)
        mock_array.__add__ = Mock(return_value=mock_array)

        mock_mx.random.normal.return_value = mock_array
        mock_mx.zeros.return_value = mock_array

        mock_linear = Mock()
        mock_linear.return_value = mock_array  # Base output
        mock_nn.Linear.return_value = mock_linear

        config = LoRAConfig(r=8)
        layer = LoRALinear(768, 768, config)
        layer.lora_a = mock_array
        layer.lora_b = mock_array

        # Forward pass
        result = layer(mock_array)

        # Check operations
        mock_linear.assert_called_with(mock_array)
        self.assertIsNotNone(result)

    def test_merged_weight(self, mock_nn, mock_mx):
        """Test getting merged weights."""
        # Setup mocks with proper operator overloading
        mock_result = Mock()
        mock_lora_a = Mock()
        mock_lora_b = Mock()
        mock_lora_b.__matmul__ = Mock(return_value=mock_result)
        mock_result.__mul__ = Mock(return_value=mock_result)

        mock_base_weight = Mock()
        mock_base_weight.__add__ = Mock(return_value=mock_base_weight)

        mock_linear = Mock()
        mock_linear.weight = mock_base_weight
        mock_nn.Linear.return_value = mock_linear

        config = LoRAConfig(r=8, alpha=16.0)
        layer = LoRALinear(768, 768, config)
        layer.lora_a = mock_lora_a
        layer.lora_b = mock_lora_b

        # Get merged weight
        merged = layer.merged_weight()

        # Should combine base weight with LoRA weights
        self.assertIsNotNone(merged)


@patch("finetune.training.lora.mx")
@patch("finetune.training.lora.nn")
class TestLoRAApplication(unittest.TestCase):
    """Test applying LoRA to models."""

    def test_apply_lora_to_model(self, mock_nn, mock_mx):
        """Test applying LoRA to a model."""

        # Create a proper mock Linear class for isinstance check
        class MockLinear:
            def __init__(self, in_features=None, out_features=None, bias=False):
                self.weight = Mock(shape=(out_features or 768, in_features or 768))
                self.bias = Mock() if bias else None
                self.freeze = Mock()  # Add freeze method

        # Set the mock Linear to be the MockLinear class
        mock_nn.Linear = MockLinear

        # Create mock model
        mock_model = Mock()
        mock_linear = MockLinear()

        # Setup named_modules
        mock_model.named_modules.return_value = [
            ("transformer.h.0.self_attn.q_proj", mock_linear),
            ("transformer.h.0.self_attn.k_proj", mock_linear),
        ]

        config = LoRAConfig(target_modules=["q_proj", "k_proj"])
        result = apply_lora_to_model(mock_model, config)

        # Check that model was modified
        self.assertEqual(result, mock_model)

    def test_get_lora_trainable_params(self, mock_nn, mock_mx):
        """Test getting trainable LoRA parameters."""
        # Create mock model with LoRA parameters
        mock_model = Mock()
        mock_param1 = Mock(size=1000)
        mock_param2 = Mock(size=2000)
        mock_param3 = Mock(size=3000)

        mock_model.named_parameters.return_value = [
            ("base.weight", mock_param1),
            ("lora_a", mock_param2),
            ("lora_b", mock_param3),
        ]

        params, trainable, total = get_lora_trainable_params(mock_model)

        # Check counts
        self.assertEqual(len(params), 2)  # lora_a and lora_b
        self.assertEqual(trainable, 5000)  # 2000 + 3000
        self.assertEqual(total, 6000)  # 1000 + 2000 + 3000


@patch("finetune.training.lora.mx")
class TestLoRASaveLoad(unittest.TestCase):
    """Test saving and loading LoRA weights."""

    def test_save_lora_weights(self, mock_mx):
        """Test saving LoRA weights."""
        # Create mock model with LoRA modules
        mock_model = Mock()
        mock_lora_linear = Mock(spec=LoRALinear)
        mock_lora_linear.lora_a = Mock()
        mock_lora_linear.lora_b = Mock()

        mock_model.named_modules.return_value = [
            ("layer1", mock_lora_linear),
            ("layer2", Mock()),  # Non-LoRA layer
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/lora_weights.npz"
            save_lora_weights(mock_model, path)

            # Check that mx.save was called
            mock_mx.save.assert_called_once()
            args = mock_mx.save.call_args[0]
            self.assertEqual(args[0], path)

    def test_load_lora_weights(self, mock_mx):
        """Test loading LoRA weights."""
        # Setup mock weights
        mock_weights = {
            "layer1.lora_a": Mock(),
            "layer1.lora_b": Mock(),
        }
        mock_mx.load.return_value = mock_weights

        # Create mock model
        mock_model = Mock()
        mock_lora_linear = Mock(spec=LoRALinear)
        mock_model.named_modules.return_value = [
            ("layer1", mock_lora_linear),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/lora_weights.npz"
            load_lora_weights(mock_model, path)

            # Check that weights were loaded
            mock_mx.load.assert_called_once_with(path)
            self.assertEqual(mock_lora_linear.lora_a, mock_weights["layer1.lora_a"])
            self.assertEqual(mock_lora_linear.lora_b, mock_weights["layer1.lora_b"])


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.warmup_steps, 100)

    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=8,
            output_dir="./custom_output",
        )
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.output_dir, "./custom_output")


@patch("finetune.training.trainer.optim")
@patch("finetune.training.trainer.mx")
class TestLoRATrainer(unittest.TestCase):
    """Test LoRA trainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.add_lora = Mock()
        self.mock_model.forward = Mock(return_value=Mock(shape=(2, 10, 50000)))
        self.mock_model.parameters = Mock(return_value=iter([]))

        self.lora_config = LoRAConfig()
        self.training_config = TrainingConfig(num_epochs=1)

    def test_initialization(self, mock_mx, mock_optim):
        """Test trainer initialization."""
        # Setup mocks
        mock_optim.AdamW.return_value = Mock()

        with patch("finetune.training.trainer.get_lora_trainable_params") as mock_get_params:
            mock_get_params.return_value = ([], 1000, 10000)

            trainer = LoRATrainer(
                self.mock_model,
                self.lora_config,
                self.training_config,
            )

            # Check initialization
            self.mock_model.add_lora.assert_called_once_with(self.lora_config)
            mock_optim.AdamW.assert_called_once()
            self.assertEqual(trainer.trainable_count, 1000)
            self.assertEqual(trainer.total_count, 10000)

    def test_compute_loss(self, mock_mx, mock_optim):
        """Test loss computation."""
        with patch("finetune.training.trainer.get_lora_trainable_params") as mock_get_params:
            mock_get_params.return_value = ([], 1000, 10000)

            with patch("finetune.training.trainer.nn") as mock_nn:
                mock_loss_fn = Mock(return_value=Mock())
                mock_nn.losses.cross_entropy = mock_loss_fn
                mock_mx.mean.return_value = 0.5

                trainer = LoRATrainer(
                    self.mock_model,
                    self.lora_config,
                    self.training_config,
                )

                # Create properly mockable batch data
                mock_input_ids = Mock(shape=(2, 10))
                mock_labels = Mock(shape=(2, 10))
                mock_labels.__getitem__ = Mock(return_value=mock_labels)
                mock_labels.reshape = Mock(return_value=mock_labels)

                batch = {
                    "input_ids": mock_input_ids,
                    "labels": mock_labels,
                }

                # Mock the forward output to be subscriptable
                forward_output = Mock()
                forward_output.__getitem__ = Mock(return_value=forward_output)
                forward_output.reshape = Mock(return_value=forward_output)
                forward_output.shape = [2, 10, 50000]
                self.mock_model.forward.return_value = forward_output

                loss = trainer.compute_loss(batch)

                # Check that loss was computed
                self.mock_model.forward.assert_called_once()
                mock_loss_fn.assert_called_once()
                self.assertEqual(loss, 0.5)

    def test_save_checkpoint(self, mock_mx, mock_optim):
        """Test saving checkpoints."""
        with patch("finetune.training.trainer.get_lora_trainable_params") as mock_get_params:
            mock_get_params.return_value = ([], 1000, 10000)

            with patch("finetune.training.trainer.save_lora_weights") as mock_save:
                trainer = LoRATrainer(
                    self.mock_model,
                    self.lora_config,
                    self.training_config,
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    trainer.output_dir = Path(tmpdir)
                    trainer.save_checkpoint()

                    # Check that weights were saved
                    mock_save.assert_called_once()
                    mock_mx.save.assert_called()


@patch("finetune.training.trainer.mx")
class TestSimpleDataLoader(unittest.TestCase):
    """Test simple data loader."""

    def test_initialization(self, mock_mx):
        """Test data loader initialization."""
        data = [{"input_ids": Mock()} for _ in range(10)]
        loader = SimpleDataLoader(data, batch_size=2)

        self.assertEqual(loader.batch_size, 2)
        self.assertEqual(len(loader.data), 10)
        self.assertTrue(loader.shuffle)

    def test_iteration(self, mock_mx):
        """Test iterating through data loader."""
        # Setup mocks
        mock_mx.arange.return_value = list(range(4))
        mock_mx.stack.return_value = Mock()

        data = [{"input_ids": Mock()} for _ in range(4)]
        loader = SimpleDataLoader(data, batch_size=2, shuffle=False)

        batches = list(loader)

        # Should have 2 batches
        self.assertEqual(len(batches), 2)

        # Each batch should have input_ids and labels
        for batch in batches:
            self.assertIn("input_ids", batch)
            self.assertIn("labels", batch)

    def test_length(self, mock_mx):
        """Test data loader length."""
        data = [{"input_ids": Mock()} for _ in range(10)]
        loader = SimpleDataLoader(data, batch_size=3)

        # 10 items / 3 batch size = 3 batches
        self.assertEqual(len(loader), 3)


if __name__ == "__main__":
    unittest.main()
