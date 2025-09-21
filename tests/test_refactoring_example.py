"""
Example test file demonstrating the new common test utilities.

This shows how the refactored test utilities can be used across different test scenarios.
"""

import pytest

from tests.utils import (
    AssertionHelpers,
    DatasetFactory,
    FileHelper,
    ModelConfigFactory,
    TestEnvironment,
)


class TestRefactoredExamples:
    """Example tests demonstrating the common utilities."""

    def test_model_config_creation(self):
        """Test using the ModelConfigFactory."""
        # Create different types of configs easily
        small_config = ModelConfigFactory.create_small_config()
        sample_config = ModelConfigFactory.create_sample_config()
        gpt_config = ModelConfigFactory.create_gpt_config()

        # Use common assertions
        AssertionHelpers.assert_model_config_valid(small_config)
        AssertionHelpers.assert_model_config_valid(sample_config)
        AssertionHelpers.assert_model_config_valid(gpt_config)

        # Verify specific properties
        assert small_config.model_type == "llama"
        assert gpt_config.model_type == "gpt2"
        assert sample_config.vocab_size > small_config.vocab_size

    def test_dataset_creation_and_validation(self):
        """Test using the DatasetFactory."""
        # Create different types of datasets
        sample_data = DatasetFactory.create_sample_dataset(size=5)
        qa_data = DatasetFactory.create_qa_dataset(size=3)
        math_data = DatasetFactory.create_math_dataset(size=2)

        # Use common validation
        AssertionHelpers.assert_dataset_valid(sample_data)
        AssertionHelpers.assert_dataset_valid(qa_data)
        AssertionHelpers.assert_dataset_valid(math_data)

        # Verify expected sizes
        assert len(sample_data) == 5
        assert len(qa_data) == 3
        assert len(math_data) == 2

        # Verify content types
        assert "capital" in qa_data[0]["instruction"].lower()
        assert any(char.isdigit() for char in math_data[0]["instruction"])

    def test_temporary_file_creation(self):
        """Test using the FileHelper for temporary files."""
        # Create test data
        test_data = DatasetFactory.create_sample_dataset(size=3)

        # Create temporary files
        json_file = FileHelper.create_temp_json_file(test_data)
        jsonl_file = FileHelper.create_temp_jsonl_file(test_data)

        try:
            # Verify files exist and are accessible
            import json
            import os

            assert os.path.exists(json_file)
            assert os.path.exists(jsonl_file)

            # Load and verify content
            with open(json_file) as f:
                loaded_json = json.load(f)
            assert loaded_json == test_data

            with open(jsonl_file) as f:
                lines = f.readlines()
            assert len(lines) == len(test_data)

        finally:
            # Clean up
            import os

            os.unlink(json_file)
            os.unlink(jsonl_file)

    @TestEnvironment.skip_if_no_mlx()
    def test_environment_checks(self):
        """Test environment checking utilities."""
        # This test only runs if MLX is available
        assert TestEnvironment.mlx_available()

        # Print environment status
        TestEnvironment.verbose_print("MLX is available for testing")

    def test_loss_convergence_validation(self):
        """Test loss convergence assertion helper."""
        # Test with converging losses
        good_losses = [3.5, 3.2, 2.8, 2.5, 2.1, 1.9]
        AssertionHelpers.assert_loss_convergence(good_losses)

        # Test with non-converging losses (should fail)
        bad_losses = [2.0, 2.1, 2.2, 2.3, 2.4]
        with pytest.raises(AssertionError, match="Loss reduction"):
            AssertionHelpers.assert_loss_convergence(bad_losses)

    def test_mock_directory_creation(self, tmp_path):
        """Test mock directory creation utilities."""
        # Create mock model directories
        hf_model_dir = FileHelper.create_mock_model_directory(tmp_path, "llama")
        mlx_model_dir = FileHelper.create_mock_mlx_model_directory(tmp_path, "llama")

        # Verify structure
        assert (hf_model_dir / "config.json").exists()
        assert (hf_model_dir / "pytorch_model.bin").exists()
        assert (hf_model_dir / "tokenizer_config.json").exists()

        assert (mlx_model_dir / "mlx_config.json").exists()
        assert (mlx_model_dir / "mlx_model.safetensors").exists()

        # Verify configs are loadable
        import json

        with open(hf_model_dir / "config.json") as f:
            hf_config = json.load(f)
        with open(mlx_model_dir / "mlx_config.json") as f:
            mlx_config = json.load(f)

        assert hf_config["model_type"] == "llama"
        assert mlx_config["model_type"] == "llama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
