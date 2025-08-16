"""
Unified Schema-Based Validation Framework

This package provides a universal validation solution for any component that follows
the RequiresProtocol pattern (has a 'requires' attribute).

Features:
- Works with observation spaces, generators, reward units, and any future components
- Rich error reporting with metadata extraction
- Performance optimizations with fast path checks
- Beautiful visual error messages with smart suggestions

The framework is split into logical modules:
- protocols: Core protocol definitions
- exceptions: Custom exception classes
- base: Base validation functionality
- validators: Specialized validators for different use cases
- functions: Convenience functions for common validation tasks
"""

# Core components
from .protocols import RequiresProtocol
from .exceptions import MissingObservationError
from .base import BaseSchemaValidator
from .validators import SchemaValidator, GeneratorSchemaValidator, ObservationValidator
from .functions import (
    validate_observation_spaces,
    validate_generators,
    validate_reward_units,
)

# Export all public components
__all__ = [
    # Core protocols and exceptions
    "RequiresProtocol",
    "MissingObservationError",
    # Validator classes
    "BaseSchemaValidator",
    "SchemaValidator",
    "GeneratorSchemaValidator",
    "ObservationValidator",  # Legacy alias
    # Convenience functions
    "validate_observation_spaces",
    "validate_generators",
    "validate_reward_units",
]
