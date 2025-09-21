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

        # Check if this is a messages format (chat conversations)
        if "messages" in item:
            self._validate_messages_format(item, index)
            return

        # Check if this is legacy instruction/output format
        if "instruction" in item or "output" in item:
            self._validate_instruction_format(item, index)
            return

        # If neither format is detected, check required fields normally
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

    def _validate_messages_format(self, item: dict[str, Any], index: int) -> None:
        """
        Validate a chat messages format item.

        Args:
            item: Item to validate
            index: Item index for error reporting

        Raises:
            DataValidationError: If validation fails
        """
        from .exceptions import DataValidationError

        messages = item.get("messages")
        if not isinstance(messages, list):
            raise DataValidationError(f"Field 'messages' in item {index} must be a list")

        if not messages:
            raise DataValidationError(f"Field 'messages' in item {index} cannot be empty")

        # Validate each message
        required_roles = {"user", "assistant"}
        found_roles = set()

        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise DataValidationError(
                    f"Message {msg_idx} in item {index} must be a dictionary"
                )

            # Check required fields in message
            if "role" not in message:
                raise DataValidationError(
                    f"Missing 'role' field in message {msg_idx} of item {index}"
                )

            if "content" not in message:
                raise DataValidationError(
                    f"Missing 'content' field in message {msg_idx} of item {index}"
                )

            role = message["role"]
            content = message["content"]

            # Validate role
            if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
                raise DataValidationError(
                    f"Invalid role '{role}' in message {msg_idx} of item {index}. "
                    f"Must be 'system', 'user', or 'assistant'"
                )

            # Validate content
            if not isinstance(content, str):
                raise DataValidationError(
                    f"Content in message {msg_idx} of item {index} must be a string"
                )

            if not content.strip():
                raise DataValidationError(
                    f"Empty content in message {msg_idx} of item {index}"
                )

            found_roles.add(role)

        # Ensure we have at least user and assistant messages
        missing_roles = required_roles - found_roles
        if missing_roles:
            raise DataValidationError(
                f"Item {index} missing required roles: {', '.join(missing_roles)}"
            )

    def _validate_instruction_format(self, item: dict[str, Any], index: int) -> None:
        """
        Validate a legacy instruction/output format item.

        Args:
            item: Item to validate
            index: Item index for error reporting

        Raises:
            DataValidationError: If validation fails
        """
        from .exceptions import DataValidationError

        # Check required fields for instruction format
        required_instruction_fields = ["instruction", "output"]
        for field in required_instruction_fields:
            if field not in item:
                raise DataValidationError(f"Missing required field '{field}' in item {index}")

            # Check field is not empty
            value = item[field]
            if isinstance(value, str) and not value.strip():
                raise DataValidationError(
                    f"Empty value for required field '{field}' in item {index}"
                )

            # Check field type
            if not isinstance(value, str):
                raise DataValidationError(
                    f"Field '{field}' in item {index} must be a string"
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

        # Detect format and calculate appropriate statistics
        messages_format_count = 0
        instruction_format_count = 0

        instruction_lengths = []
        output_lengths = []
        message_counts = []
        total_message_length = []

        for item in data:
            if "messages" in item:
                messages_format_count += 1
                messages = item.get("messages", [])
                message_counts.append(len(messages))

                # Calculate total conversation length
                total_length = sum(len(msg.get("content", "")) for msg in messages if isinstance(msg, dict))
                total_message_length.append(total_length)

            elif "instruction" in item or "output" in item:
                instruction_format_count += 1
                if "instruction" in item and isinstance(item["instruction"], str):
                    instruction_lengths.append(len(item["instruction"]))
                if "output" in item and isinstance(item["output"], str):
                    output_lengths.append(len(item["output"]))

        # Add format statistics
        summary["messages_format_count"] = messages_format_count
        summary["instruction_format_count"] = instruction_format_count

        # Add messages format statistics
        if message_counts:
            summary["avg_messages_per_conversation"] = sum(message_counts) / len(message_counts)
            summary["min_messages_per_conversation"] = min(message_counts)
            summary["max_messages_per_conversation"] = max(message_counts)

        if total_message_length:
            summary["avg_conversation_length"] = sum(total_message_length) / len(total_message_length)
            summary["min_conversation_length"] = min(total_message_length)
            summary["max_conversation_length"] = max(total_message_length)

        # Add instruction format statistics (for backward compatibility)
        if instruction_lengths:
            summary["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
            summary["min_instruction_length"] = min(instruction_lengths)
            summary["max_instruction_length"] = max(instruction_lengths)

        if output_lengths:
            summary["avg_output_length"] = sum(output_lengths) / len(output_lengths)
            summary["min_output_length"] = min(output_lengths)
            summary["max_output_length"] = max(output_lengths)

        return summary
