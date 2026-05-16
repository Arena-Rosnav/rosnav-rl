"""
Specialized validators for different component types.

This module contains validators that extend the base functionality
for specific use cases like generators and observation spaces.
"""

from typing import Dict, Any

from .base import BaseSchemaValidator
from .protocols import RequiresProtocol
from .exceptions import MissingObservationError


class SchemaValidator(BaseSchemaValidator):
    """
    Standard validator for schema-based requirements.

    Works with any component that has a 'requires' attribute containing
    observation type mappings. Validates all requirements upfront.
    """

    pass


class GeneratorSchemaValidator(BaseSchemaValidator):
    """
    Specialized validator for generator dependencies with dependency resolution awareness.

    Handles the unique case where generators depend on other generators, creating
    dependency chains that need to be validated in the correct order.
    """

    @staticmethod
    def validate_configuration_dependencies(
        generators: Dict[str, RequiresProtocol], collectors: Dict[str, Any]
    ) -> None:
        """
        Validate that all generator dependencies exist in the configuration.
        This catches configuration errors early during initialization.

        Args:
            generators: Dictionary of generators with 'requires' attributes
            collectors: Dictionary of available collectors

        Raises:
            MissingObservationError: If any required dependencies are missing from config
        """
        all_available_keys = set(collectors.keys()) | set(generators.keys())

        missing_dependencies = {}
        for name, generator in generators.items():
            if not hasattr(generator, "requires"):
                continue

            missing_keys = []
            for dep_key in generator.requires.keys():
                if dep_key not in all_available_keys:
                    missing_keys.append(dep_key)

            if missing_keys:
                missing_dependencies[name] = {
                    "missing_keys": missing_keys,
                    "component": generator,
                }

        if missing_dependencies:
            error_msg = BaseSchemaValidator._format_error_message(
                {}, missing_dependencies, "Generator"
            )

            missing_keys_list = [
                key
                for info in missing_dependencies.values()
                for key in info["missing_keys"]
            ]

            raise MissingObservationError(
                f"Generator dependency configuration validation failed:\n{error_msg}",
                missing_keys=missing_keys_list,
            )

    @staticmethod
    def validate_root_generators(
        observations: Dict[str, Any], generators: Dict[str, RequiresProtocol]
    ) -> None:
        """
        Validate generators that only depend on collectors (root generators).
        These can be validated upfront since their dependencies should be available.

        Args:
            observations: Available observations (from collectors)
            generators: All generators
        """
        root_generators = {}

        for name, generator in generators.items():
            if not hasattr(generator, "requires"):
                continue

            # Check if all dependencies are collectors (not other generators)
            is_root_generator = True
            for dep_key in generator.requires.keys():
                if dep_key in generators:
                    is_root_generator = False
                    break

            if is_root_generator:
                root_generators[name] = generator

        if root_generators:
            BaseSchemaValidator.validate_requirements(
                observations, root_generators, "Generator"
            )

    @staticmethod
    def validate_single_generator(
        observations: Dict[str, Any], generator_name: str, generator: RequiresProtocol
    ) -> None:
        """
        Validate a single generator's requirements against current observation state.
        Used for just-in-time validation during dependency-resolved execution.

        Args:
            observations: Current observation state
            generator_name: Name of the generator being validated
            generator: Generator instance to validate

        Raises:
            MissingObservationError: If required dependencies are missing
        """
        if not hasattr(generator, "requires"):
            return

        missing_keys = []
        for key in generator.requires.keys():
            if key not in observations:
                missing_keys.append(key)

        if missing_keys:
            # Create a single-generator missing components dict for error formatting
            missing_components = {
                generator_name: {
                    "missing_keys": missing_keys,
                    "component": generator,
                }
            }

            error_msg = BaseSchemaValidator._format_error_message(
                observations, missing_components, "Generator"
            )

            raise MissingObservationError(
                f"Generator '{generator_name}' validation failed:\n{error_msg}",
                missing_keys=missing_keys,
            )


# Backward compatibility alias
ObservationValidator = SchemaValidator

__all__ = [
    "SchemaValidator",
    "GeneratorSchemaValidator",
    "ObservationValidator",  # Legacy alias
]
