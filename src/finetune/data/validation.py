"""
Data validation functionality for datasets.

Validates dataset structure, required fields, and data quality.
"""

from typing import Any, Optional


class DatasetValidator:
    """Validator for dataset structure and content."""

    def __init__(
        self,
        required_fields: list[str],
        field_types: Optional[dict[str, type]] = None,
        min_length: Optional[int] = None,
    ):
        """
        Initialize dataset validator.

        Args:
            required_fields: List of required field names
            field_types: Optional mapping of field names to expected types
            min_length: Optional minimum number of items in dataset
        """
        self.required_fields = required_fields
        self.field_types = field_types or {}
        self.min_length = min_length

    def validate(self, data: list[dict[str, Any]]) -> None:
        """
        Validate dataset structure and content.

        Args:
            data: Dataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        from .exceptions import DataValidationError

        # Check minimum length
        if self.min_length is not None and len(data) < self.min_length:
            raise DataValidationError(f"Dataset must contain at least {self.min_length} items")

        # Validate each item
        for i, item in enumerate(data):
            self._validate_item(item, i)

    def _validate_item(self, item: dict[str, Any], index: int) -> None:
        """
        Validate a single dataset item.

        Args:
            item: Item to validate
            index: Item index for error reporting

        Raises:
            DataValidationError: If validation fails
        """
        from .exceptions import DataValidationError

        # Check required fields are present
        for field in self.required_fields:
            if field not in item:
                raise DataValidationError(f"Missing required field '{field}' in item {index}")

            # Check field is not empty
            value = item[field]
            if isinstance(value, str) and not value.strip():
                raise DataValidationError(
                    f"Empty value for required field '{field}' in item {index}"
                )

            # Check field type
            if field in self.field_types:
                expected_type = self.field_types[field]
                if not isinstance(value, expected_type):
                    raise DataValidationError(
                        f"Field '{field}' in item {index} must be of type {expected_type.__name__}"
                    )

    def get_summary(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Get summary statistics for the dataset.

        Args:
            data: Dataset to analyze

        Returns:
            Dictionary with summary statistics
        """
        if not data:
            return {"total_items": 0}

        summary = {"total_items": len(data)}

        # Calculate length statistics for required fields
        for field in self.required_fields:
            field_lengths = []
            for item in data:
                if field in item and isinstance(item[field], str):
                    field_lengths.append(len(item[field]))

            if field_lengths:
                summary[f"avg_{field}_length"] = sum(field_lengths) / len(field_lengths)
                summary[f"min_{field}_length"] = min(field_lengths)
                summary[f"max_{field}_length"] = max(field_lengths)

        return summary
