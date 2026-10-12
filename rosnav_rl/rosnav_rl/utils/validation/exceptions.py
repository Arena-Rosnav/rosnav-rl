"""
Exception classes for validation framework.

This module defines custom exceptions used in the validation system.
"""


class MissingObservationError(ValueError):
    """Raised when required observations are missing from the observation space."""

    def __init__(self, message: str, missing_keys: list[str] | None = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []


__all__ = ["MissingObservationError"]
