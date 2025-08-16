"""
Protocol definitions for validation framework.

This module defines the core protocols used throughout the validation system.
"""

from typing import Dict, Any, Protocol, runtime_checkable


@runtime_checkable
class RequiresProtocol(Protocol):
    """Protocol for components that have observation requirements."""

    requires: Dict[str, Any]
    """
    'requires' is a mapping from logical input names (str) to string keys referencing
    the Annotated data type schema of required dependencies. This enables schema-based
    configuration and validation of generator inputs, decoupling logical names from
    concrete data source classes.
    Example:
        requires = {
            "front_laser": "LaserScanSchema",
            "odom": "OdometrySchema"
        }
    """


__all__ = ["RequiresProtocol"]
