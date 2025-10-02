"""
Data loading functionality for various file formats.

Implements loaders for JSON, JSONL, and other data formats
with proper validation and error handling.

This module provides standardized data loading for the FineTune project,
supporting the conventional train/test/valid split structure.
"""

import json
from pathlib import Path
from typing import Any, Union

from loguru import logger

from .exceptions import DataFormatError


class JSONLoader:
    """Loader for JSON format datasets."""

    def load(self, file_path: Union[str, Path]) -> list[dict[str, Any]]:
        """
        Load dataset from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of dataset examples

        Raises:
            FileNotFoundError: If file doesn't exist
            DataFormatError: If JSON format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataFormatError(f"Invalid JSON format: {e}")

        return self._validate_structure(data)

    def _validate_structure(self, data: Any) -> list[dict[str, Any]]:
        """
        Validate and normalize JSON structure.

        Args:
            data: Loaded JSON data

        Returns:
            List of dictionaries

        Raises:
            DataFormatError: If structure is invalid
        """
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise DataFormatError("JSON data must be a dictionary or list of dictionaries")


class JSONLLoader:
    """Loader for JSONL (JSON Lines) format datasets."""

    def load(self, file_path: Union[str, Path]) -> list[dict[str, Any]]:
        """
        Load dataset from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of dataset examples

        Raises:
            FileNotFoundError: If file doesn't exist
            DataFormatError: If JSONL format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = []

        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    raise DataFormatError(f"Invalid JSON on line {line_num}: {e}")

        return data


class DatasetLoader:
    """Auto-detecting dataset loader for multiple formats."""

    def __init__(self):
        self._loaders = {
            "json": JSONLoader(),
            "jsonl": JSONLLoader(),
        }

    def load(self, file_path: Union[str, Path]) -> list[dict[str, Any]]:
        """
        Load dataset with automatic format detection.

        Args:
            file_path: Path to dataset file

        Returns:
            List of dataset examples

        Raises:
            FileNotFoundError: If file doesn't exist
            DataFormatError: If format is unsupported or invalid
        """
        file_path = Path(file_path)
        format_type = self._detect_format(file_path)

        loader = self._loaders[format_type]
        return loader.load(file_path)

    def _detect_format(self, file_path: Union[str, Path]) -> str:
        """
        Detect file format from extension.

        Args:
            file_path: Path to file

        Returns:
            Format type string

        Raises:
            DataFormatError: If format is unsupported
        """
        path_obj = Path(file_path)
        suffix = path_obj.suffix.lower()

        if suffix == ".json":
            return "json"
        elif suffix == ".jsonl":
            return "jsonl"
        else:
            raise DataFormatError(f"Unsupported file format: {suffix}")


class StandardDataLoader:
    """
    Standardized data loader for FineTune datasets.

    Provides consistent access to train/test/valid datasets with conventional directory structure.
    Each dataset directory should contain: train.jsonl, test.jsonl, valid.jsonl
    """

    def __init__(self, data_root: Union[str, Path] = None):
        """
        Initialize standardized data loader.

        Args:
            data_root: Root directory for datasets. Defaults to project data directory.
        """
        if data_root is None:
            # Default to project data directory
            import os
            self.data_root = Path(os.environ.get('FT_DATA_ROOT', Path(__file__).parent.parent.parent.parent / "data"))
        else:
            self.data_root = Path(data_root)

        self._loader = DatasetLoader()

    def get_available_datasets(self) -> list[str]:
        """
        Get list of available datasets.

        Returns:
            List of dataset names that have the standard train/test/valid structure
        """
        datasets = []

        # Check direct data root only
        if self.data_root.exists() and self.data_root.is_dir():
            for dataset_dir in self.data_root.iterdir():
                if dataset_dir.is_dir() and self._has_standard_structure(dataset_dir):
                    datasets.append(dataset_dir.name)

        return sorted(datasets)

    def _has_standard_structure(self, dataset_dir: Path) -> bool:
        """Check if directory has the standard train/test/valid structure."""
        required_files = ["train.jsonl", "test.jsonl", "valid.jsonl"]
        return all((dataset_dir / file).exists() for file in required_files)

    def _find_dataset_path(self, dataset_name: str) -> Path:
        """
        Find the full path to a dataset by name or path.

        Args:
            dataset_name: Either a dataset name (e.g., 'mlx_examples') or a full path

        Returns:
            Path to the dataset directory
        """
        dataset_path = Path(dataset_name)

        # If it's already an absolute path or a relative path with separators, use it directly
        if dataset_path.is_absolute() or '/' in dataset_name or '\\' in dataset_name:
            if dataset_path.exists() and self._has_standard_structure(dataset_path):
                return dataset_path
            else:
                raise FileNotFoundError(
                    f"Dataset path '{dataset_name}' does not exist or lacks required files (train.jsonl, test.jsonl, valid.jsonl)"
                )

        # Otherwise, treat it as a dataset name and look directly under data root
        dataset_path = self.data_root / dataset_name

        if dataset_path.exists() and self._has_standard_structure(dataset_path):
            return dataset_path

        # If not found, list available datasets for helpful error
        available = self.get_available_datasets()
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found. Available datasets: {available}"
        )

    def load_train_data(self, dataset_name: str = "mlx_examples") -> list[dict[str, Any]]:
        """
        Load training data from dataset.

        Args:
            dataset_name: Name of dataset directory or full path. Defaults to 'mlx_examples'

        Returns:
            List of training examples
        """
        dataset_path = self._find_dataset_path(dataset_name)
        train_file = dataset_path / "train.jsonl"
        logger.info(f"Loading training data from: {train_file}")
        return self._loader.load(train_file)

    def load_test_data(self, dataset_name: str = "mlx_examples") -> list[dict[str, Any]]:
        """
        Load test data for evaluation during training.

        Args:
            dataset_name: Name of dataset directory or full path. Defaults to 'mlx_examples'

        Returns:
            List of test examples for evaluation during training
        """
        dataset_path = self._find_dataset_path(dataset_name)
        test_file = dataset_path / "test.jsonl"
        logger.info(f"Loading test data from: {test_file}")
        return self._loader.load(test_file)

    def load_valid_data(self, dataset_name: str = "mlx_examples") -> list[dict[str, Any]]:
        """
        Load validation data for generation testing.

        Args:
            dataset_name: Name of dataset directory or full path. Defaults to 'mlx_examples'

        Returns:
            List of validation examples for generation testing
        """
        dataset_path = self._find_dataset_path(dataset_name)
        valid_file = dataset_path / "valid.jsonl"
        logger.info(f"Loading validation data from: {valid_file}")
        return self._loader.load(valid_file)

    def load_dataset_split(self, split: str, dataset_name: str = "mlx_examples") -> list[dict[str, Any]]:
        """
        Load specific split of dataset.

        Args:
            split: Data split ('train', 'test', or 'valid')
            dataset_name: Name of dataset directory or full path. Defaults to 'mlx_examples'

        Returns:
            List of examples for the specified split
        """
        if split == "train":
            return self.load_train_data(dataset_name)
        elif split == "test":
            return self.load_test_data(dataset_name)
        elif split == "valid":
            return self.load_valid_data(dataset_name)
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'test', or 'valid'")

    def get_dataset_info(self, dataset_name: str = "mlx_examples") -> dict[str, Any]:
        """
        Get information about a dataset.

        Args:
            dataset_name: Name of dataset directory or full path. Defaults to 'mlx_examples'

        Returns:
            Dictionary with dataset statistics
        """
        try:
            train_data = self.load_train_data(dataset_name)
            test_data = self.load_test_data(dataset_name)
            valid_data = self.load_valid_data(dataset_name)

            # Analyze data format
            sample = train_data[0] if train_data else {}
            has_messages = "messages" in sample
            has_text = "text" in sample

            return {
                "name": dataset_name,
                "path": str(self._find_dataset_path(dataset_name)),
                "splits": {
                    "train": len(train_data),
                    "test": len(test_data),
                    "valid": len(valid_data),
                },
                "total_examples": len(train_data) + len(test_data) + len(valid_data),
                "format": "chat" if has_messages else "mlx_text" if has_text else "unknown",
                "sample_keys": list(sample.keys()) if sample else [],
            }
        except Exception as e:
            return {
                "name": dataset_name,
                "error": str(e),
            }


# Convenience functions for backward compatibility and ease of use
def load_train_data(dataset_name: str = "mlx_examples") -> list[dict[str, Any]]:
    """Load training data from specified dataset."""
    loader = StandardDataLoader()
    return loader.load_train_data(dataset_name)


def load_test_data(dataset_name: str = "mlx_examples") -> list[dict[str, Any]]:
    """Load test data from specified dataset."""
    loader = StandardDataLoader()
    return loader.load_test_data(dataset_name)


def load_valid_data(dataset_name: str = "mlx_examples") -> list[dict[str, Any]]:
    """Load validation data from specified dataset."""
    loader = StandardDataLoader()
    return loader.load_valid_data(dataset_name)


def get_available_datasets() -> list[str]:
    """Get list of available datasets."""
    loader = StandardDataLoader()
    return loader.get_available_datasets()


def get_dataset_info(dataset_name: str = "mlx_examples") -> dict[str, Any]:
    """Get information about a dataset."""
    loader = StandardDataLoader()
    return loader.get_dataset_info(dataset_name)
