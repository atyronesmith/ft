"""
Data loading functionality for various file formats.

Implements loaders for JSON, JSONL, and other data formats
with proper validation and error handling.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union

from .exceptions import DataFormatError, DataValidationError


class JSONLoader:
    """Loader for JSON format datasets."""

    def load(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
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
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataFormatError(f"Invalid JSON format: {e}")

        return self._validate_structure(data)

    def _validate_structure(self, data: Any) -> List[Dict[str, Any]]:
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

    def load(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
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

        with open(file_path, 'r', encoding='utf-8') as f:
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
            'json': JSONLoader(),
            'jsonl': JSONLLoader(),
        }

    def load(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
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

        if suffix == '.json':
            return 'json'
        elif suffix == '.jsonl':
            return 'jsonl'
        else:
            raise DataFormatError(f"Unsupported file format: {suffix}")