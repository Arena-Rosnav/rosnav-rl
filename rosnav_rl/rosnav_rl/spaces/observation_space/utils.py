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
    from ...observations.type_annotations import DataSpec
except ImportError:
    # Fallback for direct execution or different import contexts
    try:
        from rosnav_rl.observations.type_annotations import DataSpec
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


class SchemaValidator:
    """
    Universal validator for schema-based requirements.

    Works with any component that has a 'requires' attribute containing
    observation type mappings.
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
            error_msg = SchemaValidator._format_error_message(
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
                    metadata = SchemaValidator._extract_metadata(obs_type)

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
        }

        try:
            # Check if it's a DataSpec type with metadata
            if hasattr(obs_type, "__annotations__"):
                # Try to get DataSpec metadata
                spec = getattr(obs_type, "_spec", None)
                if spec and isinstance(spec, DataSpec):
                    metadata["description"] = spec.description
                    metadata["shape"] = getattr(spec, "shape", None)
                    metadata["units"] = getattr(spec, "units", None)
                    metadata["constraints"] = getattr(spec, "constraints", None)

            # Fallback to basic type information
            if not metadata["description"]:
                metadata["description"] = getattr(obs_type, "__doc__", str(obs_type))

        except Exception:
            # Safe fallback
            metadata["description"] = str(obs_type)

        return metadata


# Convenience functions for specific component types
def validate_observation_spaces(
    observations: Dict[str, Any], spaces: Dict[str, RequiresProtocol]
) -> None:
    """Validate observation space requirements."""
    SchemaValidator.validate_requirements(observations, spaces, "Observation Space")


def validate_generators(
    observations: Dict[str, Any], generators: Dict[str, RequiresProtocol]
) -> None:
    """Validate generator requirements."""
    SchemaValidator.validate_requirements(observations, generators, "Generator")


def validate_reward_units(
    observations: Dict[str, Any], reward_units: Dict[str, RequiresProtocol]
) -> None:
    """Validate reward unit requirements."""
    SchemaValidator.validate_requirements(observations, reward_units, "Reward Unit")


# Backward compatibility alias
ObservationValidator = SchemaValidator

# Export all public components
__all__ = [
    "SchemaValidator",
    "RequiresProtocol",
    "MissingObservationError",
    "validate_observation_spaces",
    "validate_generators",
    "validate_reward_units",
    "ObservationValidator",  # Legacy alias
]
