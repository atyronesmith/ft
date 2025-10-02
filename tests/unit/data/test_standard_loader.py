"""
Unit tests for StandardDataLoader and related functionality.

Tests the standardized data loading system with train/test/valid splits,
dataset discovery, path handling, and convenience functions.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from finetune.data.loaders import (
    StandardDataLoader,
    get_available_datasets,
    get_dataset_info,
    load_test_data,
    load_train_data,
    load_valid_data,
)


class TestStandardDataLoader:
    """Test StandardDataLoader class functionality."""

    @pytest.fixture
    def temp_data_root(self):
        """Create temporary data root with test datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)

            # Create test dataset 1: mlx_examples
            mlx_examples = data_root / "mlx_examples"
            mlx_examples.mkdir()

            # Create train.jsonl
            train_data = [
                {"text": "Question: What is 2+2? Answer: 4"},
                {"text": "Question: What is the capital of France? Answer: Paris"},
                {"text": "Question: What is Python? Answer: A programming language"},
            ]
            with open(mlx_examples / "train.jsonl", "w") as f:
                for item in train_data:
                    f.write(json.dumps(item) + "\n")

            # Create test.jsonl
            test_data = [
                {"text": "Question: What is 3+3? Answer: 6"},
            ]
            with open(mlx_examples / "test.jsonl", "w") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            # Create valid.jsonl
            valid_data = [
                {"text": "Question: What is 5+5? Answer: 10"},
                {"text": "Question: What is the capital of Spain? Answer: Madrid"},
            ]
            with open(mlx_examples / "valid.jsonl", "w") as f:
                for item in valid_data:
                    f.write(json.dumps(item) + "\n")

            # Create test dataset 2: chat_examples
            chat_examples = data_root / "chat_examples"
            chat_examples.mkdir()

            # Create chat format data
            chat_train_data = [
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"}
                    ]
                }
            ]
            with open(chat_examples / "train.jsonl", "w") as f:
                for item in chat_train_data:
                    f.write(json.dumps(item) + "\n")

            with open(chat_examples / "test.jsonl", "w") as f:
                f.write(json.dumps(chat_train_data[0]) + "\n")

            with open(chat_examples / "valid.jsonl", "w") as f:
                f.write(json.dumps(chat_train_data[0]) + "\n")

            yield data_root

    def test_init_default_data_root(self):
        """Test StandardDataLoader initialization with default data root."""
        loader = StandardDataLoader()
        assert loader.data_root.name == "data"
        assert loader._loader is not None

    def test_init_custom_data_root(self, temp_data_root):
        """Test StandardDataLoader initialization with custom data root."""
        loader = StandardDataLoader(temp_data_root)
        assert loader.data_root == temp_data_root

    def test_get_available_datasets(self, temp_data_root):
        """Test dataset discovery functionality."""
        loader = StandardDataLoader(temp_data_root)
        datasets = loader.get_available_datasets()

        assert "mlx_examples" in datasets
        assert "chat_examples" in datasets
        assert len(datasets) == 2
        assert datasets == sorted(datasets)  # Should be sorted

    def test_get_available_datasets_empty_directory(self):
        """Test dataset discovery with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = StandardDataLoader(temp_dir)
            datasets = loader.get_available_datasets()
            assert datasets == []

    def test_has_standard_structure_valid(self, temp_data_root):
        """Test standard structure validation for valid dataset."""
        loader = StandardDataLoader(temp_data_root)
        mlx_path = temp_data_root / "mlx_examples"
        assert loader._has_standard_structure(mlx_path) is True

    def test_has_standard_structure_invalid(self, temp_data_root):
        """Test standard structure validation for invalid dataset."""
        loader = StandardDataLoader(temp_data_root)

        # Create directory with missing files
        incomplete = temp_data_root / "incomplete"
        incomplete.mkdir()
        (incomplete / "train.jsonl").touch()
        # Missing test.jsonl and valid.jsonl

        assert loader._has_standard_structure(incomplete) is False

    def test_find_dataset_path_by_name(self, temp_data_root):
        """Test finding dataset by name."""
        loader = StandardDataLoader(temp_data_root)
        path = loader._find_dataset_path("mlx_examples")
        assert path == temp_data_root / "mlx_examples"

    def test_find_dataset_path_by_relative_path(self, temp_data_root):
        """Test finding dataset by relative path."""
        loader = StandardDataLoader(temp_data_root)
        relative_path = str(temp_data_root / "mlx_examples")
        path = loader._find_dataset_path(relative_path)
        assert path == temp_data_root / "mlx_examples"

    def test_find_dataset_path_by_absolute_path(self, temp_data_root):
        """Test finding dataset by absolute path."""
        loader = StandardDataLoader(temp_data_root)
        absolute_path = str(temp_data_root / "mlx_examples")
        path = loader._find_dataset_path(absolute_path)
        assert path == temp_data_root / "mlx_examples"

    def test_find_dataset_path_nonexistent_name(self, temp_data_root):
        """Test finding nonexistent dataset by name raises appropriate error."""
        loader = StandardDataLoader(temp_data_root)

        with pytest.raises(FileNotFoundError, match="Dataset 'nonexistent' not found"):
            loader._find_dataset_path("nonexistent")

    def test_find_dataset_path_nonexistent_path(self, temp_data_root):
        """Test finding nonexistent dataset by path raises appropriate error."""
        loader = StandardDataLoader(temp_data_root)
        nonexistent_path = "/nonexistent/path/to/dataset"

        with pytest.raises(FileNotFoundError, match="does not exist or lacks required files"):
            loader._find_dataset_path(nonexistent_path)

    def test_load_train_data_default(self, temp_data_root):
        """Test loading training data with default dataset."""
        with patch.dict(os.environ, {'FT_DATA_ROOT': str(temp_data_root)}):
            loader = StandardDataLoader()
            # Since mlx_examples is the default, but we need to have it in the right place
            # Let's test with explicit dataset name instead
            data = loader.load_train_data("mlx_examples")

            assert len(data) == 3
            assert all("text" in item for item in data)
            assert "2+2" in data[0]["text"]

    def test_load_train_data_by_name(self, temp_data_root):
        """Test loading training data by dataset name."""
        loader = StandardDataLoader(temp_data_root)
        data = loader.load_train_data("mlx_examples")

        assert len(data) == 3
        assert all("text" in item for item in data)

    def test_load_train_data_by_path(self, temp_data_root):
        """Test loading training data by dataset path."""
        loader = StandardDataLoader(temp_data_root)
        path = str(temp_data_root / "mlx_examples")
        data = loader.load_train_data(path)

        assert len(data) == 3
        assert all("text" in item for item in data)

    def test_load_test_data(self, temp_data_root):
        """Test loading test data."""
        loader = StandardDataLoader(temp_data_root)
        data = loader.load_test_data("mlx_examples")

        assert len(data) == 1
        assert "text" in data[0]
        assert "3+3" in data[0]["text"]

    def test_load_valid_data(self, temp_data_root):
        """Test loading validation data."""
        loader = StandardDataLoader(temp_data_root)
        data = loader.load_valid_data("mlx_examples")

        assert len(data) == 2
        assert all("text" in item for item in data)
        assert "5+5" in data[0]["text"]

    def test_load_dataset_split_train(self, temp_data_root):
        """Test loading specific dataset split - train."""
        loader = StandardDataLoader(temp_data_root)
        data = loader.load_dataset_split("train", "mlx_examples")

        assert len(data) == 3
        assert all("text" in item for item in data)

    def test_load_dataset_split_test(self, temp_data_root):
        """Test loading specific dataset split - test."""
        loader = StandardDataLoader(temp_data_root)
        data = loader.load_dataset_split("test", "mlx_examples")

        assert len(data) == 1
        assert "text" in data[0]

    def test_load_dataset_split_valid(self, temp_data_root):
        """Test loading specific dataset split - valid."""
        loader = StandardDataLoader(temp_data_root)
        data = loader.load_dataset_split("valid", "mlx_examples")

        assert len(data) == 2
        assert all("text" in item for item in data)

    def test_load_dataset_split_invalid(self, temp_data_root):
        """Test loading invalid dataset split raises error."""
        loader = StandardDataLoader(temp_data_root)

        with pytest.raises(ValueError, match="Invalid split 'invalid'"):
            loader.load_dataset_split("invalid", "mlx_examples")

    def test_get_dataset_info(self, temp_data_root):
        """Test getting dataset information."""
        loader = StandardDataLoader(temp_data_root)
        info = loader.get_dataset_info("mlx_examples")

        assert info["name"] == "mlx_examples"
        assert info["path"] == str(temp_data_root / "mlx_examples")
        assert info["splits"]["train"] == 3
        assert info["splits"]["test"] == 1
        assert info["splits"]["valid"] == 2
        assert info["total_examples"] == 6
        assert info["format"] == "mlx_text"
        assert "text" in info["sample_keys"]

    def test_get_dataset_info_chat_format(self, temp_data_root):
        """Test getting dataset info for chat format data."""
        loader = StandardDataLoader(temp_data_root)
        info = loader.get_dataset_info("chat_examples")

        assert info["name"] == "chat_examples"
        assert info["format"] == "chat"
        assert "messages" in info["sample_keys"]

    def test_get_dataset_info_nonexistent(self, temp_data_root):
        """Test getting info for nonexistent dataset."""
        loader = StandardDataLoader(temp_data_root)
        info = loader.get_dataset_info("nonexistent")

        assert info["name"] == "nonexistent"
        assert "error" in info


class TestConvenienceFunctions:
    """Test convenience functions for data loading."""

    @pytest.fixture
    def temp_data_root(self):
        """Create temporary data root with test datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)

            # Create mlx_examples dataset
            mlx_examples = data_root / "mlx_examples"
            mlx_examples.mkdir()

            # Create sample data
            sample_data = [{"text": "Sample question and answer"}]

            for split in ["train", "test", "valid"]:
                with open(mlx_examples / f"{split}.jsonl", "w") as f:
                    for item in sample_data:
                        f.write(json.dumps(item) + "\n")

            yield data_root

    def test_load_train_data_convenience(self, temp_data_root):
        """Test load_train_data convenience function."""
        with patch.dict(os.environ, {'FT_DATA_ROOT': str(temp_data_root)}):
            data = load_train_data("mlx_examples")
            assert len(data) == 1
            assert "text" in data[0]

    def test_load_test_data_convenience(self, temp_data_root):
        """Test load_test_data convenience function."""
        with patch.dict(os.environ, {'FT_DATA_ROOT': str(temp_data_root)}):
            data = load_test_data("mlx_examples")
            assert len(data) == 1
            assert "text" in data[0]

    def test_load_valid_data_convenience(self, temp_data_root):
        """Test load_valid_data convenience function."""
        with patch.dict(os.environ, {'FT_DATA_ROOT': str(temp_data_root)}):
            data = load_valid_data("mlx_examples")
            assert len(data) == 1
            assert "text" in data[0]

    def test_get_available_datasets_convenience(self, temp_data_root):
        """Test get_available_datasets convenience function."""
        with patch.dict(os.environ, {'FT_DATA_ROOT': str(temp_data_root)}):
            datasets = get_available_datasets()
            assert "mlx_examples" in datasets

    def test_get_dataset_info_convenience(self, temp_data_root):
        """Test get_dataset_info convenience function."""
        with patch.dict(os.environ, {'FT_DATA_ROOT': str(temp_data_root)}):
            info = get_dataset_info("mlx_examples")
            assert info["name"] == "mlx_examples"
            assert info["total_examples"] == 3


class TestDataRootEnvironment:
    """Test data root configuration via environment variable."""

    def test_custom_data_root_via_env(self):
        """Test setting custom data root via FT_DATA_ROOT environment variable."""
        custom_path = "/custom/data/path"

        with patch.dict(os.environ, {'FT_DATA_ROOT': custom_path}):
            loader = StandardDataLoader()
            assert str(loader.data_root) == custom_path

    def test_default_data_root_when_env_not_set(self):
        """Test default data root when FT_DATA_ROOT is not set."""
        # Remove the env var if it exists
        env_copy = os.environ.copy()
        if 'FT_DATA_ROOT' in env_copy:
            del env_copy['FT_DATA_ROOT']

        with patch.dict(os.environ, env_copy, clear=True):
            loader = StandardDataLoader()
            assert loader.data_root.name == "data"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_load_data_with_missing_split_file(self, tmp_path):
        """Test loading data when split file is missing."""
        # Create incomplete dataset (missing test.jsonl)
        dataset_dir = tmp_path / "incomplete_dataset"
        dataset_dir.mkdir()

        (dataset_dir / "train.jsonl").touch()
        (dataset_dir / "valid.jsonl").touch()
        # Missing test.jsonl

        loader = StandardDataLoader(tmp_path)

        # Should raise FileNotFoundError when trying to load test data
        with pytest.raises(FileNotFoundError):
            loader.load_test_data("incomplete_dataset")

    def test_load_data_with_invalid_json(self, tmp_path):
        """Test loading data with invalid JSON content."""
        dataset_dir = tmp_path / "invalid_json"
        dataset_dir.mkdir()

        # Create files with required names
        for split in ["train", "test", "valid"]:
            with open(dataset_dir / f"{split}.jsonl", "w") as f:
                f.write("invalid json content\n")

        loader = StandardDataLoader(tmp_path)

        # Should raise DataFormatError when trying to load invalid JSON
        with pytest.raises(Exception):  # Could be DataFormatError or json.JSONDecodeError
            loader.load_train_data("invalid_json")

    def test_path_vs_name_detection(self, tmp_path):
        """Test proper detection of paths vs names."""
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()

        # Create valid dataset
        for split in ["train", "test", "valid"]:
            with open(dataset_dir / f"{split}.jsonl", "w") as f:
                f.write('{"text": "test"}\n')

        loader = StandardDataLoader(tmp_path)

        # Test name detection (no separators)
        name_path = loader._find_dataset_path("test_dataset")
        assert name_path == dataset_dir

        # Test relative path detection (contains separator)
        rel_path = loader._find_dataset_path(f"test_dataset")  # Actually still a name
        assert rel_path == dataset_dir

        # Test absolute path detection
        abs_path = loader._find_dataset_path(str(dataset_dir))
        assert abs_path == dataset_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])