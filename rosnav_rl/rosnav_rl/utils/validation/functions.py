"""
Convenience functions for component validation.

This module provides easy-to-use validation functions for specific component types.
"""

from typing import Any

from .protocols import RequiresProtocol
from .validators import GeneratorSchemaValidator, SchemaValidator


def validate_observation_spaces(
    observations: dict[str, Any], spaces: dict[str, RequiresProtocol]
) -> None:
    """Validate observation space requirements."""
    SchemaValidator.validate_requirements(observations, spaces, "Observation Space")


def validate_generators(
    observations: dict[str, Any], generators: dict[str, RequiresProtocol]
) -> None:
    """Validate generator requirements using the specialized generator validator."""
    GeneratorSchemaValidator.validate_root_generators(observations, generators)


def validate_reward_units(
    observations: dict[str, Any], reward_units: dict[str, RequiresProtocol]
) -> None:
    """Validate reward unit requirements."""
    SchemaValidator.validate_requirements(observations, reward_units, "Reward Unit")


__all__ = [
    "validate_observation_spaces",
    "validate_generators",
    "validate_reward_units",
]
