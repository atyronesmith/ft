"""
Unit tests for CLI commands.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from finetune.cli.app import app
from finetune.cli.utils import CLIError, format_number, format_size, validate_path
from typer.testing import CliRunner


class TestCLIApp:
    """Test main CLI application."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Fine-tune language models" in result.stdout

    def test_info_command(self, runner):
        """Test info command."""
        with patch("finetune.cli.app.device_manager") as mock_dm:
            mock_dm.device_info.to_dict.return_value = {
                "device_type": "apple_silicon",
                "chip_name": "Apple M4 Pro",
                "total_memory_gb": 64.0,
                "cpu_count": 12,
            }
            mock_dm.get_memory_info.return_value = {
                "available_gb": 32.0,
            }
            mock_dm.get_optimal_backend.return_value = Mock(
                get_device_info=lambda: {"backend": "mlx", "mlx_version": "0.10.0"}
            )

            result = runner.invoke(app, ["info"])
            assert result.exit_code == 0
            assert "System Information" in result.stdout
            assert "Apple M4 Pro" in result.stdout

    def test_init_command(self, runner, temp_dir):
        """Test init command."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert "Project initialized successfully" in result.stdout

            # Check directories were created
            assert Path("data").exists()
            assert Path("models").exists()
            assert Path("checkpoints").exists()
            assert Path("logs").exists()
            assert Path("configs").exists()


class TestTrainCommands:
    """Test training commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_train_start_missing_dataset(self, runner):
        """Test train start with missing dataset."""
        result = runner.invoke(app, ["train", "start", "gpt2", "missing.json"])
        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout

    @patch("finetune.cli.commands.train.create_training_workflow_from_config")
    @patch("finetune.cli.commands.train.FineTuningWorkflow")
    def test_train_start_with_valid_args(
        self, mock_workflow_class, mock_create_workflow, runner, temp_dir
    ):
        """Test train start with valid arguments."""
        # Mock workflow and its methods
        mock_workflow = Mock()
        mock_workflow.run_training.return_value = {"loss": 0.5, "perplexity": 1.6}
        mock_workflow_class.return_value = mock_workflow

        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create dummy dataset
            dataset_path = Path("data.json")
            dataset_path.write_text(json.dumps([{"instruction": "test", "output": "response"}]))

            result = runner.invoke(
                app,
                ["train", "start", "gpt2", str(dataset_path), "--epochs", "1", "--batch-size", "2"],
            )

            assert result.exit_code == 0
            assert "🚀 FineTune" in result.stdout

    @pytest.mark.skip(reason="stop command not yet implemented")
    def test_train_stop(self, runner):
        """Test train stop command."""
        result = runner.invoke(app, ["train", "stop"])
        assert result.exit_code == 0
        # Should handle gracefully even with no running jobs

    @pytest.mark.skip(reason="status command not yet implemented")
    def test_train_status(self, runner):
        """Test train status command."""
        result = runner.invoke(app, ["train", "status"])
        assert result.exit_code == 0


class TestModelCommands:
    """Test model management commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_models_list(self, runner):
        """Test models list command."""
        with patch("finetune.models.manager.model_manager") as mock_mm:
            mock_mm.list_models.return_value = [
                {
                    "name": "gpt2",
                    "source": "huggingface",
                    "cached": True,
                    "size_gb": 0.5,
                }
            ]

            result = runner.invoke(app, ["models", "list"])
            assert result.exit_code == 0
            assert "Available Models" in result.stdout

    def test_models_info(self, runner):
        """Test models info command."""
        with patch("finetune.models.manager.model_manager") as mock_mm:
            mock_mm.get_model_info.return_value = {
                "name": "gpt2",
                "path": "/path/to/model",
                "size_gb": 0.5,
                "model_type": "gpt2",
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "files": ["config.json", "model.bin"],
            }

            result = runner.invoke(app, ["models", "info", "gpt2"])
            assert result.exit_code == 0
            assert "Model Information" in result.stdout
            assert "gpt2" in result.stdout


class TestDatasetCommands:
    """Test dataset commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_dataset_validate(self, runner, temp_dir):
        """Test dataset validate command."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create valid dataset
            dataset_path = Path("valid.json")
            dataset_path.write_text(
                json.dumps(
                    [
                        {"text": "Example 1"},
                        {"text": "Example 2"},
                    ]
                )
            )

            result = runner.invoke(app, ["dataset", "validate", str(dataset_path)])
            assert result.exit_code == 0
            assert "Validation Passed" in result.stdout

    def test_dataset_validate_invalid(self, runner, temp_dir):
        """Test dataset validate with invalid data."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create invalid dataset
            dataset_path = Path("invalid.json")
            dataset_path.write_text("not json")

            result = runner.invoke(app, ["dataset", "validate", str(dataset_path)])
            assert result.exit_code == 0
            assert "Validation Failed" in result.stdout

    def test_dataset_split(self, runner, temp_dir):
        """Test dataset split command."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create dataset
            dataset_path = Path("data.json")
            dataset_path.write_text(json.dumps([{"text": f"Example {i}"} for i in range(10)]))

            result = runner.invoke(
                app,
                [
                    "dataset",
                    "split",
                    str(dataset_path),
                    "--train",
                    "0.8",
                    "--val",
                    "0.1",
                    "--test",
                    "0.1",
                ],
            )

            assert result.exit_code == 0
            assert "Splitting Dataset" in result.stdout
            assert Path("data/train.json").exists()
            assert Path("data/validation.json").exists()
            assert Path("data/test.json").exists()

    def test_dataset_stats(self, runner, temp_dir):
        """Test dataset stats command."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create dataset
            dataset_path = Path("data.json")
            dataset_path.write_text(
                json.dumps(
                    [
                        {"text": "Short"},
                        {"text": "A bit longer example"},
                        {"text": "The longest example in this dataset"},
                    ]
                )
            )

            result = runner.invoke(app, ["dataset", "stats", str(dataset_path)])
            assert result.exit_code == 0
            assert "Dataset Statistics" in result.stdout
            assert "Total Examples" in result.stdout

    def test_dataset_prepare(self, runner, temp_dir):
        """Test dataset prepare command."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create dataset with instruction/output format
            dataset_path = Path("raw.json")
            dataset_path.write_text(
                json.dumps(
                    [
                        {"instruction": "What is 2+2?", "output": "4"},
                        {"instruction": "Capital of France?", "output": "Paris"},
                    ]
                )
            )

            result = runner.invoke(
                app, ["dataset", "prepare", str(dataset_path), "--template", "alpaca"]
            )

            assert result.exit_code == 0
            assert "Preparing Dataset" in result.stdout
            assert Path("raw_prepared.json").exists()

            # Check prepared data
            with open("raw_prepared.json") as f:
                prepared = json.load(f)
            assert len(prepared) == 2
            assert "text" in prepared[0]
            assert "Instruction" in prepared[0]["text"]


class TestCLIUtils:
    """Test CLI utility functions."""

    def test_validate_path_exists(self, temp_dir):
        """Test path validation for existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        result = validate_path(test_file, must_exist=True)
        assert result == test_file

        with pytest.raises(CLIError):
            validate_path(temp_dir / "missing.txt", must_exist=True)

    def test_validate_path_file_or_dir(self, temp_dir):
        """Test path validation for file vs directory."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        test_dir = temp_dir / "subdir"
        test_dir.mkdir()

        # Test file validation
        validate_path(test_file, must_be_file=True)
        with pytest.raises(CLIError):
            validate_path(test_dir, must_be_file=True)

        # Test directory validation
        validate_path(test_dir, must_be_dir=True)
        with pytest.raises(CLIError):
            validate_path(test_file, must_be_dir=True)

    def test_format_size(self):
        """Test size formatting."""
        assert format_size(512) == "512.0 B"
        assert format_size(1024) == "1.0 KB"
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_format_number(self):
        """Test number formatting."""
        assert format_number(1000) == "1,000"
        assert format_number(1000000) == "1,000,000"
        assert format_number(1234567890) == "1,234,567,890"
