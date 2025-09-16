"""
Integration test for the complete end-to-end fine-tuning workflow.

Tests the integration of all Phase 2 components: configuration, data loading,
templates, LoRA training, and CLI commands.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path

from finetune.config import TrainingConfig, ModelConfig, DataConfig, LoRAConfig, OptimizationConfig
from finetune.training.workflow import FineTuningWorkflow, create_quick_workflow


class TestEndToEndWorkflow:
    """Test complete fine-tuning workflow integration."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample training dataset."""
        data = [
            {
                "instruction": "What is Python?",
                "output": "Python is a high-level programming language known for its simplicity and readability."
            },
            {
                "instruction": "What is machine learning?",
                "output": "Machine learning is a subset of AI that enables computers to learn from data."
            },
            {
                "instruction": "Explain neural networks",
                "output": "Neural networks are computing systems inspired by biological neural networks."
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def training_config(self, sample_dataset):
        """Create training configuration for testing."""
        return TrainingConfig(
            model=ModelConfig(name="test-model"),  # Use placeholder model for testing
            data=DataConfig(train_file=sample_dataset, template="alpaca", validation_split=0.0),  # No validation split for testing
            lora=LoRAConfig(r=4, alpha=8.0),  # Small LoRA for testing
            optimization=OptimizationConfig(
                learning_rate=1e-4,
                batch_size=1,
                epochs=1,  # Single epoch for testing
            ),
            output_dir="./test_output",
        )

    def test_workflow_initialization(self, training_config):
        """Test workflow can be initialized with configuration."""
        workflow = FineTuningWorkflow(training_config)

        assert workflow.config == training_config
        assert workflow.model_manager is not None
        assert workflow.dataset_loader is not None
        assert workflow.template_registry is not None

    def test_dataset_preparation(self, training_config):
        """Test dataset loading and template application."""
        workflow = FineTuningWorkflow(training_config)

        # Test dataset preparation
        workflow.prepare_dataset()

        assert workflow.train_dataset is not None
        assert len(workflow.train_dataset) == 3  # Our sample has 3 examples

        # Check that templates were applied
        first_example = workflow.train_dataset[0]
        assert "### Instruction:" in first_example
        assert "### Response:" in first_example
        assert "What is Python?" in first_example

    def test_quick_workflow_creation(self, sample_dataset):
        """Test quick workflow creation utility."""
        workflow = create_quick_workflow(
            model_name="test-model",
            data_file=sample_dataset,
            template="chatml",
            output_dir="./quick_test"
        )

        assert workflow.config.model.name == "test-model"
        assert workflow.config.data.train_file == sample_dataset
        assert workflow.config.data.template == "chatml"
        assert workflow.config.output_dir == "./quick_test"

    def test_tokenization(self, training_config):
        """Test dataset tokenization functionality."""
        workflow = FineTuningWorkflow(training_config)

        # Prepare dataset first
        workflow.prepare_dataset()

        # Test tokenization
        tokenized = workflow.tokenize_dataset(workflow.train_dataset[:2], max_length=50)

        assert len(tokenized) == 2
        assert all("input_ids" in item for item in tokenized)
        assert all("attention_mask" in item for item in tokenized)

    def test_configuration_integration(self, sample_dataset):
        """Test integration with configuration profiles."""
        from finetune.config import ConfigProfile

        # Create base config
        config = TrainingConfig(
            model=ModelConfig(name="test-model"),
            data=DataConfig(train_file=sample_dataset, validation_split=0.0),
        )

        # Apply chat profile
        chat_config = ConfigProfile.apply_profile(config, "chat")

        # Create workflow with profile
        workflow = FineTuningWorkflow(chat_config)

        assert workflow.config.data.template == "chatml"
        assert workflow.config.lora.r == 8  # Chat profile LoRA rank

    def test_workflow_error_handling(self):
        """Test workflow handles errors gracefully."""
        # Test with invalid model
        config = TrainingConfig(
            model=ModelConfig(name="nonexistent-model"),
            data=DataConfig(train_file="nonexistent.json"),
        )

        workflow = FineTuningWorkflow(config)

        # Should handle missing files gracefully
        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar
            workflow.prepare_dataset()

    def test_template_integration(self, sample_dataset):
        """Test integration with different template types."""
        templates_to_test = ["alpaca", "chatml", "llama"]

        for template in templates_to_test:
            config = TrainingConfig(
                model=ModelConfig(name="test-model"),
                data=DataConfig(train_file=sample_dataset, template=template, validation_split=0.0),
            )

            workflow = FineTuningWorkflow(config)
            workflow.prepare_dataset()

            assert workflow.train_dataset is not None
            assert len(workflow.train_dataset) == 3

            # Each template should produce different formatting
            first_example = workflow.train_dataset[0]
            if template == "alpaca":
                assert "### Instruction:" in first_example
            elif template == "chatml":
                assert "<|im_start|>user" in first_example
            elif template == "llama":
                assert "[INST]" in first_example

    def test_config_yaml_integration(self, sample_dataset):
        """Test integration with YAML configuration files."""
        from finetune.config import ConfigManager

        # Create config
        config = TrainingConfig(
            model=ModelConfig(name="test-model"),
            data=DataConfig(train_file=sample_dataset, template="alpaca"),
            lora=LoRAConfig(r=16, alpha=32.0),
        )

        # Save to temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config_path = f.name

        try:
            manager = ConfigManager()
            manager.save_config(config, config_path)

            # Load workflow from config file
            from finetune.training.workflow import create_training_workflow_from_config
            workflow = create_training_workflow_from_config(config_path)

            assert workflow.config.model.name == "test-model"
            assert workflow.config.lora.r == 16
            assert workflow.config.lora.alpha == 32.0

        finally:
            os.unlink(config_path)

    def test_validation_integration(self, sample_dataset):
        """Test integration with data validation."""
        config = TrainingConfig(
            model=ModelConfig(name="test-model"),
            data=DataConfig(train_file=sample_dataset, validation_split=0.0),
        )

        workflow = FineTuningWorkflow(config)

        # Dataset preparation should include validation
        workflow.prepare_dataset()

        # Should successfully validate and prepare data
        assert workflow.train_dataset is not None
        assert len(workflow.train_dataset) > 0

    def test_memory_estimation_integration(self, training_config):
        """Test integration with memory estimation."""
        from finetune.config import ConfigValidator

        validator = ConfigValidator()
        memory_estimate = validator.estimate_memory_usage(training_config)

        # Should return a reasonable memory estimate
        assert isinstance(memory_estimate, (int, float))
        assert memory_estimate > 0

    @pytest.mark.skip(reason="Requires actual model loading - mock for CI")
    def test_full_workflow_simulation(self, training_config):
        """Test complete workflow execution (simulation mode)."""
        workflow = FineTuningWorkflow(training_config)

        # This would test the full workflow but requires model loading
        # In a real test environment, we'd mock the model manager
        try:
            results = workflow.run_training()

            # Check results structure
            assert "training_loss" in results
            assert "global_step" in results
            assert "epoch" in results

        except Exception as e:
            # Expected in test environment without real models
            assert "test-model" in str(e) or "placeholder" in str(e)