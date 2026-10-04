"""Navigation Spaces - Production Ready & Legacy

This module contains both production-ready and legacy navigation observation spaces.

Production Spaces:
    - RobustGoalSpace: Reliable goal representation
    - MultiScaleGoalSpace: Multi-scale goal representation
    - SubgoalContextSpace: Subgoal with context

Legacy Spaces:
    - DistAngleToGoalSpace: Original distance/angle to goal
    - DistAngleToSubgoalSpace: Original distance/angle to subgoal
"""

from .advanced_navigation_spaces import (
    MultiScaleGoalSpace,
    RobustGoalSpace,
    SubgoalContextSpace,
)
from .basic_navigation_spaces import DistAngleToGoalSpace, DistAngleToSubgoalSpace

__all__ = [
    "DistAngleToGoalSpace",
    "DistAngleToSubgoalSpace",
    "RobustGoalSpace",
    "MultiScaleGoalSpace",
    "SubgoalContextSpace",
]
