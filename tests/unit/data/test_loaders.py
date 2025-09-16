"""
Test-driven development for data loading functionality.

Tests are written first to drive the implementation of data loaders
for various formats (JSON, JSONL, CSV) with proper validation.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from finetune.data import (
    JSONLoader,
    JSONLLoader,
    DatasetLoader,
    DataFormatError,
    DataValidationError,
    DatasetValidator,
)


class TestJSONLoader:
    """Test JSON dataset loading functionality."""

    def test_load_valid_json_list(self):
        """Test loading valid JSON with list of examples."""
        # Arrange
        data = [
            {"instruction": "What is Python?", "output": "A programming language"},
            {"instruction": "How to loop?", "output": "Use for loops"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = JSONLoader()

            # Act
            result = loader.load(temp_path)

            # Assert
            assert len(result) == 2
            assert result[0]["instruction"] == "What is Python?"
            assert result[0]["output"] == "A programming language"
            assert result[1]["instruction"] == "How to loop?"
            assert result[1]["output"] == "Use for loops"
        finally:
            os.unlink(temp_path)

    def test_load_valid_json_object(self):
        """Test loading JSON with single object converted to list."""
        # Arrange
        data = {"instruction": "Single example", "output": "Single response"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = JSONLoader()

            # Act
            result = loader.load(temp_path)

            # Assert
            assert len(result) == 1
            assert result[0]["instruction"] == "Single example"
            assert result[0]["output"] == "Single response"
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json_format(self):
        """Test loading invalid JSON raises appropriate error."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_path = f.name

        try:
            loader = JSONLoader()

            # Act & Assert
            with pytest.raises(DataFormatError, match="Invalid JSON format"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises FileNotFoundError."""
        loader = JSONLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.json")

    def test_validate_json_structure(self):
        """Test JSON structure validation."""
        # Arrange
        loader = JSONLoader()

        # Valid structures
        valid_list = [{"instruction": "test", "output": "test"}]
        valid_dict = {"instruction": "test", "output": "test"}

        # Invalid structures
        invalid_string = "not a dict or list"
        invalid_number = 42

        # Act & Assert
        assert loader._validate_structure(valid_list) == valid_list
        assert loader._validate_structure(valid_dict) == [valid_dict]

        with pytest.raises(DataFormatError, match="must be a dictionary or list"):
            loader._validate_structure(invalid_string)

        with pytest.raises(DataFormatError, match="must be a dictionary or list"):
            loader._validate_structure(invalid_number)


class TestJSONLLoader:
    """Test JSONL (JSON Lines) dataset loading functionality."""

    def test_load_valid_jsonl(self):
        """Test loading valid JSONL file."""
        # Arrange
        lines = [
            '{"instruction": "What is AI?", "output": "Artificial Intelligence"}\n',
            '{"instruction": "Python usage?", "output": "Programming language"}\n',
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.writelines(lines)
            temp_path = f.name

        try:
            loader = JSONLLoader()

            # Act
            result = loader.load(temp_path)

            # Assert
            assert len(result) == 2
            assert result[0]["instruction"] == "What is AI?"
            assert result[0]["output"] == "Artificial Intelligence"
            assert result[1]["instruction"] == "Python usage?"
            assert result[1]["output"] == "Programming language"
        finally:
            os.unlink(temp_path)

    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL with empty lines (should skip them)."""
        # Arrange
        lines = [
            '{"instruction": "First", "output": "Response 1"}\n',
            '\n',  # Empty line
            '{"instruction": "Second", "output": "Response 2"}\n',
            '   \n',  # Whitespace only
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.writelines(lines)
            temp_path = f.name

        try:
            loader = JSONLLoader()

            # Act
            result = loader.load(temp_path)

            # Assert
            assert len(result) == 2
            assert result[0]["instruction"] == "First"
            assert result[1]["instruction"] == "Second"
        finally:
            os.unlink(temp_path)

    def test_load_jsonl_with_invalid_line(self):
        """Test loading JSONL with invalid JSON line raises error."""
        # Arrange
        lines = [
            '{"instruction": "Valid", "output": "Valid response"}\n',
            '{"invalid": json line}\n',  # Invalid JSON
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.writelines(lines)
            temp_path = f.name

        try:
            loader = JSONLLoader()

            # Act & Assert
            with pytest.raises(DataFormatError, match="Invalid JSON on line 2"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_empty_jsonl_file(self):
        """Test loading empty JSONL file returns empty list."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            pass  # Create empty file
            temp_path = f.name

        try:
            loader = JSONLLoader()

            # Act
            result = loader.load(temp_path)

            # Assert
            assert result == []
        finally:
            os.unlink(temp_path)


class TestDatasetLoader:
    """Test automatic dataset loader that detects format."""

    def test_load_json_file(self):
        """Test automatic loading of JSON file."""
        # Arrange
        data = [{"instruction": "Test", "output": "Response"}]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = DatasetLoader()

            # Act
            result = loader.load(temp_path)

            # Assert
            assert len(result) == 1
            assert result[0]["instruction"] == "Test"
        finally:
            os.unlink(temp_path)

    def test_load_jsonl_file(self):
        """Test automatic loading of JSONL file."""
        # Arrange
        lines = ['{"instruction": "Test", "output": "Response"}\n']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.writelines(lines)
            temp_path = f.name

        try:
            loader = DatasetLoader()

            # Act
            result = loader.load(temp_path)

            # Assert
            assert len(result) == 1
            assert result[0]["instruction"] == "Test"
        finally:
            os.unlink(temp_path)

    def test_load_unsupported_format(self):
        """Test loading unsupported file format raises error."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Some text")
            temp_path = f.name

        try:
            loader = DatasetLoader()

            # Act & Assert
            with pytest.raises(DataFormatError, match="Unsupported file format"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_detect_format_json(self):
        """Test format detection for JSON files."""
        loader = DatasetLoader()

        assert loader._detect_format("dataset.json") == "json"
        assert loader._detect_format("path/to/data.json") == "json"

    def test_detect_format_jsonl(self):
        """Test format detection for JSONL files."""
        loader = DatasetLoader()

        assert loader._detect_format("dataset.jsonl") == "jsonl"
        assert loader._detect_format("path/to/data.jsonl") == "jsonl"

    def test_detect_format_unsupported(self):
        """Test format detection for unsupported files."""
        loader = DatasetLoader()

        with pytest.raises(DataFormatError, match="Unsupported file format"):
            loader._detect_format("dataset.txt")

        with pytest.raises(DataFormatError, match="Unsupported file format"):
            loader._detect_format("dataset.csv")


class TestDataValidation:
    """Test data validation functionality."""

    def test_validate_required_fields_present(self):
        """Test validation passes when required fields are present."""
        # Arrange
        validator = DatasetValidator(required_fields=["instruction", "output"])
        data = [
            {"instruction": "Test 1", "output": "Response 1"},
            {"instruction": "Test 2", "output": "Response 2"},
        ]

        # Act & Assert (should not raise)
        validator.validate(data)

    def test_validate_missing_required_field(self):
        """Test validation fails when required field is missing."""
        # Arrange
        validator = DatasetValidator(required_fields=["instruction", "output"])
        data = [
            {"instruction": "Test 1", "output": "Response 1"},
            {"instruction": "Test 2"},  # Missing 'output'
        ]

        # Act & Assert
        with pytest.raises(DataValidationError, match="Missing required field 'output' in item 1"):
            validator.validate(data)

    def test_validate_empty_field_value(self):
        """Test validation fails when required field is empty."""
        # Arrange
        validator = DatasetValidator(required_fields=["instruction", "output"])
        data = [
            {"instruction": "Test 1", "output": "Response 1"},
            {"instruction": "", "output": "Response 2"},  # Empty instruction
        ]

        # Act & Assert
        with pytest.raises(DataValidationError, match="Empty value for required field 'instruction' in item 1"):
            validator.validate(data)

    def test_validate_field_types(self):
        """Test validation of field types."""
        # Arrange
        validator = DatasetValidator(
            required_fields=["instruction", "output"],
            field_types={"instruction": str, "output": str}
        )

        valid_data = [{"instruction": "Test", "output": "Response"}]
        invalid_data = [{"instruction": 123, "output": "Response"}]  # Wrong type

        # Act & Assert
        validator.validate(valid_data)  # Should not raise

        with pytest.raises(DataValidationError, match="Field 'instruction' in item 0 must be of type str"):
            validator.validate(invalid_data)

    def test_validate_minimum_length(self):
        """Test validation of minimum dataset length."""
        # Arrange
        validator = DatasetValidator(required_fields=["instruction"], min_length=2)

        valid_data = [
            {"instruction": "Test 1"},
            {"instruction": "Test 2"},
        ]
        invalid_data = [{"instruction": "Test 1"}]  # Too short

        # Act & Assert
        validator.validate(valid_data)  # Should not raise

        with pytest.raises(DataValidationError, match="Dataset must contain at least 2 items"):
            validator.validate(invalid_data)

    def test_validation_summary(self):
        """Test validation returns summary statistics."""
        # Arrange
        validator = DatasetValidator(required_fields=["instruction", "output"])
        data = [
            {"instruction": "Short", "output": "OK"},
            {"instruction": "This is a longer instruction", "output": "Longer response here"},
        ]

        # Act
        summary = validator.get_summary(data)

        # Assert
        assert summary["total_items"] == 2
        assert summary["avg_instruction_length"] == (5 + 28) / 2  # Average character length
        assert summary["avg_output_length"] == (2 + 20) / 2
        assert "min_instruction_length" in summary
        assert "max_instruction_length" in summary


# Test fixtures
@pytest.fixture
def sample_dataset():
    """Provide sample dataset for testing."""
    return [
        {
            "instruction": "Explain what machine learning is",
            "output": "Machine learning is a method of data analysis that automates analytical model building."
        },
        {
            "instruction": "What is Python?",
            "output": "Python is a high-level programming language known for its simplicity and readability."
        },
        {
            "instruction": "How do neural networks work?",
            "output": "Neural networks are computing systems inspired by biological neural networks."
        }
    ]


@pytest.fixture
def temp_json_file(sample_dataset):
    """Create temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_dataset, f)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_jsonl_file(sample_dataset):
    """Create temporary JSONL file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_dataset:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])