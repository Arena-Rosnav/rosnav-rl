"""
Unified Schema-Based Validation Framework

This module provides a universal validation solution for any component that follows
the RequiresProtocol pattern (has a 'requires' attribute).

Features:
- Works with observation spaces, generators, reward units, and any future components
- Rich error reporting with metadata extraction
- Performance optimizations with fast path checks
- Beautiful visual error messages with smart suggestions
"""

from typing import Dict, Any, List, Type, Protocol, runtime_checkable
from difflib import get_close_matches

try:
    from ...observations.utils.types import DataSpec
except ImportError:
    # Fallback for direct execution or different import contexts
    try:
        from rosnav_rl.observations.utils.types import DataSpec
    except ImportError:
        # Create a minimal DataSpec for standalone usage
        class DataSpec:
            def __init__(
                self, description=None, shape=None, units=None, constraints=None
            ):
                self.description = description
                self.shape = shape
                self.units = units
                self.constraints = constraints


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


class MissingObservationError(ValueError):
    """Raised when required observations are missing from the observation space."""

    def __init__(self, message: str, missing_keys: List[str] = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []


class BaseSchemaValidator:
    """
    Base class for schema-based validators.

    Provides common validation infrastructure that can be extended
    for specific validation strategies.
    """

    @staticmethod
    def validate_requirements(
        observations: Dict[str, Any],
        components: Dict[str, RequiresProtocol],
        component_type: str = "component",
    ) -> None:
        """
        Validate that all component requirements are satisfied by available observations.

        Args:
            observations: Dictionary of available observations
            components: Dictionary of components with 'requires' attributes
            component_type: Type description for error messages

        Raises:
            MissingObservationError: If any required observations are missing
        """
        if not components:
            return

        all_missing = {}

        for comp_name, component in components.items():
            if not hasattr(component, "requires"):
                continue

            missing_keys = []
            for key in component.requires.keys():
                if key not in observations:
                    missing_keys.append(key)

            if missing_keys:
                all_missing[comp_name] = {
                    "missing_keys": missing_keys,
                    "component": component,
                }

        if all_missing:
            error_msg = BaseSchemaValidator._format_error_message(
                observations, all_missing, component_type
            )
            missing_keys_list = [
                key for info in all_missing.values() for key in info["missing_keys"]
            ]
            raise MissingObservationError(error_msg, missing_keys=missing_keys_list)

    @staticmethod
    def _format_error_message(
        observations: Dict[str, Any],
        missing_components: Dict[str, Dict],
        component_type: str,
    ) -> str:
        """Format a clean, readable error message with excellent information density."""

        total_missing = sum(
            len(info["missing_keys"]) for info in missing_components.values()
        )
        total_components = len(missing_components)

        lines = [
            f"🚨 Missing Observations for {component_type.title()}s",
            f"📊 Summary: {total_missing} missing requirements across {total_components} {component_type}(s)",
            "=" * 80,
            "",
        ]

        for comp_name, info in missing_components.items():
            component = info["component"]
            missing_keys = info["missing_keys"]

            # Component header
            lines.extend(
                [
                    f"🔧 {component_type.title()}: {comp_name}",
                    f"📋 Missing {len(missing_keys)} observation(s): {', '.join(missing_keys)}",
                    "─" * 70,
                ]
            )

            # Details for each missing key
            for i, key in enumerate(missing_keys):
                if hasattr(component, "requires") and key in component.requires:
                    obs_type = component.requires[key]
                    metadata = BaseSchemaValidator._extract_metadata(obs_type)

                    lines.append(f"  {i+1}. 🔍 '{key}'")

                    # Compact metadata display
                    details = []
                    if metadata.get("description"):
                        description = (
                            metadata["description"]
                            if metadata.get("description") is not None
                            else "N/A"
                        )
                        details.append(f"📝 Description: {description}")
                    if metadata.get("shape"):
                        shape = (
                            metadata["shape"]
                            if metadata.get("shape") is not None
                            else "N/A"
                        )
                        details.append(f"📐 Shape: {shape}")
                    if metadata.get("units"):
                        units = (
                            metadata["units"]
                            if metadata.get("units") is not None
                            else "N/A"
                        )
                        details.append(f"📏 Units: {units}")
                    if metadata.get("constraints"):
                        constraints = (
                            metadata["constraints"]
                            if metadata.get("constraints") is not None
                            else "N/A"
                        )
                        details.append(f"🔒 Constraints: {constraints}")
                    if metadata.get("source"):
                        source = (
                            metadata["source"]
                            if metadata.get("source") is not None
                            else "N/A"
                        )
                        details.append(f"🌐 Source: {source}")
                    if metadata.get("example"):
                        example = (
                            metadata["example"]
                            if metadata.get("example") is not None
                            else "N/A"
                        )
                        details.append(f"💡 Example: {example}")
                    for detail in details:
                        lines.append(f"     {detail}")

                    # Similar suggestions
                    suggestions = get_close_matches(
                        key, observations.keys(), n=3, cutoff=0.6
                    )
                    if suggestions:
                        lines.append(f"     💡 Similar: {', '.join(suggestions)}")

                    if i < len(missing_keys) - 1:  # Add spacing between items
                        lines.append("")

            lines.extend(["", ""])

        # Rest remains the same...
        lines.extend(["📋 Available Observations:", "-" * 40])

        if observations:
            for key, value in observations.items():
                obs_info = f"   ✅ '{key}'"
                obs_info += f" (type: {type(value)})"
                if hasattr(value, "shape"):
                    obs_info += f" (shape: {value.shape})"
                if hasattr(value, "dtype"):
                    obs_info += f" (dtype: {value.dtype})"
                lines.append(obs_info)
        else:
            lines.append("   ❌ No observations available")

        lines.extend(
            [
                "",
                "💡 Solution Steps:",
                "   1. Check observation key spelling and case sensitivity",
                "   2. Verify observation generation in ObservationManager",
                "   3. Ensure required sensors/data sources are configured",
                "   4. Check observation space configuration matches requirements",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _extract_metadata(obs_type: Type) -> Dict[str, Any]:
        """Extract metadata from observation type annotations."""
        metadata = {
            "description": None,
            "shape": None,
            "units": None,
            "constraints": None,
            "example": None,
            "source": None,
        }

        try:
            # Handle typing.Annotated types (Python 3.9+)
            if hasattr(obs_type, "__metadata__") and hasattr(obs_type, "__origin__"):
                # This is an Annotated type, extract the metadata
                for annotation in obs_type.__metadata__:
                    if isinstance(annotation, DataSpec):
                        metadata["description"] = annotation.description
                        metadata["shape"] = getattr(annotation, "shape", None)
                        metadata["units"] = getattr(annotation, "units", None)
                        metadata["constraints"] = getattr(
                            annotation, "constraints", None
                        )
                        metadata["source"] = getattr(annotation, "source", None)
                        metadata["example"] = getattr(annotation, "example", None)
                        break

                # If no DataSpec found, try to get basic info from the origin type
                if not metadata["description"] and hasattr(obs_type, "__origin__"):
                    origin_type = obs_type.__origin__
                    metadata["description"] = getattr(
                        origin_type, "__doc__", str(origin_type)
                    )

            # Handle typing_extensions.Annotated (Python < 3.9)
            elif hasattr(obs_type, "__args__") and hasattr(obs_type, "__metadata__"):
                for annotation in obs_type.__metadata__:
                    if isinstance(annotation, DataSpec):
                        metadata["description"] = annotation.description
                        metadata["shape"] = getattr(annotation, "shape", None)
                        metadata["units"] = getattr(annotation, "units", None)
                        metadata["constraints"] = getattr(
                            annotation, "constraints", None
                        )
                        metadata["source"] = getattr(annotation, "source", None)
                        metadata["example"] = getattr(annotation, "example", None)
                        break

            # Check if it's a DataSpec type with metadata directly
            elif hasattr(obs_type, "__annotations__"):
                # Try to get DataSpec metadata
                spec = getattr(obs_type, "_spec", None)
                if spec and isinstance(spec, DataSpec):
                    metadata["description"] = spec.description
                    metadata["shape"] = getattr(spec, "shape", None)
                    metadata["units"] = getattr(spec, "units", None)
                    metadata["constraints"] = getattr(spec, "constraints", None)
                    metadata["source"] = getattr(spec, "source", None)
                    metadata["example"] = getattr(spec, "example", None)

            # Fallback to basic type information
            if not metadata["description"]:
                # Try to get the actual type name instead of Annotated wrapper
                if hasattr(obs_type, "__origin__"):
                    type_name = getattr(
                        obs_type.__origin__, "__name__", str(obs_type.__origin__)
                    )
                elif hasattr(obs_type, "__name__"):
                    type_name = obs_type.__name__
                else:
                    type_name = str(obs_type)

                metadata["description"] = f"Data type: {type_name}"

        except Exception as e:
            # Safe fallback with error info for debugging
            metadata["description"] = (
                f"Type: {str(obs_type)} (metadata extraction failed: {e})"
            )

        return metadata


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


# Convenience functions for specific component types
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


# Backward compatibility alias
ObservationValidator = SchemaValidator

# Export all public components
__all__ = [
    "BaseSchemaValidator",
    "SchemaValidator",
    "GeneratorSchemaValidator",
    "RequiresProtocol",
    "MissingObservationError",
    "validate_observation_spaces",
    "validate_generators",
    "validate_reward_units",
    "ObservationValidator",  # Legacy alias
]
