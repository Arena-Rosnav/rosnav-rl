"""Observation Space Categories

Enum definitions for categorizing observation spaces in a type-safe manner.
"""

from enum import Enum


class SpaceCategory(Enum):
    """Enum for observation space categories.

    This enum provides type-safe categorization of observation spaces,
    preventing string-based errors and making the codebase more maintainable.

    Categories:
        PERCEPTION: Sensor-based observations (laser, vision, etc.)
        NAVIGATION: Goal and path-related observations (goal distance/angle, etc.)
        DYNAMICS: Robot motion and action-related observations (last action, velocity, etc.)
        ENVIRONMENT: Environmental context observations (pedestrians, obstacles, etc.)
        LOCALIZATION: Robot pose and localization-related observations
        META: Episode and system metadata observations (step count, terminal flags, etc.)
        UNCATEGORIZED: Default category for spaces without explicit categorization
    """

    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    DYNAMICS = "dynamics"
    ENVIRONMENT = "environment"
    LOCALIZATION = "localization"
    META = "meta"
    UNCATEGORIZED = "uncategorized"

    def __str__(self) -> str:
        """Return the string value of the category."""
        return self.value

    @classmethod
    def from_string(cls, category_str: str) -> "SpaceCategory":
        """Create SpaceCategory from string, with fallback to UNCATEGORIZED.

        Args:
            category_str: String representation of the category

        Returns:
            SpaceCategory: Matching category enum or UNCATEGORIZED if not found
        """
        for category in cls:
            if category.value == category_str:
                return category
        return cls.UNCATEGORIZED

    @classmethod
    def get_all_categories(cls) -> list[str]:
        """Get list of all category string values.

        Returns:
            List of all category string values
        """
        return [category.value for category in cls]


# Convenience constants for backward compatibility
PERCEPTION = SpaceCategory.PERCEPTION
NAVIGATION = SpaceCategory.NAVIGATION
DYNAMICS = SpaceCategory.DYNAMICS
ENVIRONMENT = SpaceCategory.ENVIRONMENT
LOCALIZATION = SpaceCategory.LOCALIZATION
META = SpaceCategory.META
UNCATEGORIZED = SpaceCategory.UNCATEGORIZED
