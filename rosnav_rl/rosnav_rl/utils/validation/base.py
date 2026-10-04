"""
Base validation functionality.

This module contains the core validation infrastructure that can be extended
for specific validation strategies.
"""

from difflib import get_close_matches
from typing import Annotated, Any, get_args, get_origin

import numpy as np

from .exceptions import MissingObservationError
from .protocols import RequiresProtocol

try:
    from ...observations.utils.types import DataSpec
except ImportError:
    # Fallback for direct execution or different import contexts
    try:
        from rosnav_rl.observations.utils.types import DataSpec
    except ImportError:
        # Create a minimal DataSpec for standalone usage
        class DataSpec:
            source: str | None = None
            example: str | None = None

            def __init__(
                self,
                description: str | None = None,
                shape: str | None = None,
                units: str | None = None,
                constraints: str | None = None,
            ):
                self.description = description
                self.shape = shape
                self.units = units
                self.constraints = constraints


class BaseSchemaValidator:
    """
    Base class for schema-based validators.

    Provides common validation infrastructure that can be extended
    for specific validation strategies.
    """

    @staticmethod
    def validate_requirements(
        observations: dict[str, Any],
        components: dict[str, RequiresProtocol],
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
        observations: dict[str, Any],
        missing_components: dict[str, dict[str, Any]],
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
                if key in component.requires:
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
                if isinstance(value, (np.ndarray, np.generic)):
                    obs_info += f" (shape: {value.shape})"
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
    def _extract_metadata(obs_type: object) -> dict[str, Any]:
        """Extract metadata from observation type annotations."""
        metadata: dict[str, Any] = {
            "description": None,
            "shape": None,
            "units": None,
            "constraints": None,
            "example": None,
            "source": None,
        }

        try:
            # Handle typing.Annotated types (Python 3.9+)
            if get_origin(obs_type) is Annotated:
                # This is an Annotated type, extract the metadata
                for annotation in get_args(obs_type)[1:]:
                    if isinstance(annotation, DataSpec):
                        metadata["description"] = annotation.description
                        metadata["shape"] = annotation.shape
                        metadata["units"] = annotation.units
                        metadata["constraints"] = annotation.constraints
                        metadata["source"] = annotation.source
                        metadata["example"] = annotation.example
                        break

                # If no DataSpec found, try to get basic info from the origin type
                if not metadata["description"]:
                    metadata["description"] = get_args(obs_type)[0].__doc__

            # Fallback to basic type information
            if not metadata["description"]:
                # Try to get the actual type name instead of Annotated wrapper
                origin = get_args(obs_type)[0] if get_origin(obs_type) is Annotated else get_origin(obs_type)
                if origin is not None:
                    type_name = origin.__name__ if isinstance(origin, type) else str(origin)
                elif isinstance(obs_type, type):
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


__all__ = ["BaseSchemaValidator"]
