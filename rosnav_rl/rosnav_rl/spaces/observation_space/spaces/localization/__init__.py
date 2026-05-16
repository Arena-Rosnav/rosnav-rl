"""Localization Spaces - Production Ready & Legacy

This module contains both production-ready and legacy localization observation spaces.

Production Spaces:
    - RobustOdometrySpace: Filtered odometry with EMA
    - PoseStabilizedSpace: Stabilized pose representation
    - LocalizationCombinedSpace: Combined pose and velocity

Legacy Spaces: (Currently none in base localization)
"""

# Advanced spaces
from .advanced_localization_spaces import (
    RobustOdometrySpace,
    PoseStabilizedSpace,
    LocalizationCombinedSpace,
)

__all__ = ["RobustOdometrySpace", "PoseStabilizedSpace", "LocalizationCombinedSpace"]
