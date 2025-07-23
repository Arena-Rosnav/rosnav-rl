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

# Basic spaces
from .basic_navigation_spaces import DistAngleToGoalSpace, DistAngleToSubgoalSpace

# Advanced spaces
from .advanced_navigation_spaces import (
    RobustGoalSpace,
    MultiScaleGoalSpace,
    SubgoalContextSpace,
)

__all__ = [
    "DistAngleToGoalSpace",
    "DistAngleToSubgoalSpace",
    "RobustGoalSpace",
    "MultiScaleGoalSpace",
    "SubgoalContextSpace",
]
