"""
Convenience functions for component validation.

This module provides easy-to-use validation functions for specific component types.
"""

from typing import Dict, Any

from .protocols import RequiresProtocol
from .validators import SchemaValidator, GeneratorSchemaValidator


def validate_observation_spaces(
    observations: Dict[str, Any], spaces: Dict[str, RequiresProtocol]
) -> None:
    """Validate observation space requirements."""
    SchemaValidator.validate_requirements(observations, spaces, "Observation Space")


def validate_generators(
    observations: Dict[str, Any], generators: Dict[str, RequiresProtocol]
) -> None:
    """Validate generator requirements using the specialized generator validator."""
    GeneratorSchemaValidator.validate_root_generators(observations, generators)


def validate_reward_units(
    observations: Dict[str, Any], reward_units: Dict[str, RequiresProtocol]
) -> None:
    """Validate reward unit requirements."""
    SchemaValidator.validate_requirements(observations, reward_units, "Reward Unit")


__all__ = [
    "validate_observation_spaces",
    "validate_generators",
    "validate_reward_units",
]
