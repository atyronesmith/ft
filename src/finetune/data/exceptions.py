"""
Data processing exceptions.

Custom exceptions for data loading and validation errors.
"""


class DataFormatError(Exception):
    """Raised when data format is invalid or unsupported."""
    pass


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class TemplateError(Exception):
    """Raised when template formatting or configuration fails."""
    pass