"""Dynamics Spaces - Production Ready & Legacy

This module contains both production-ready and legacy dynamics observation spaces.

Production Spaces:
    - MotionStateSpace: Motion state with stability metrics
    - KinematicStateSpace: Kinematic state representation
    - TrajectoryStateSpace: Trajectory state with history

Legacy Spaces:
    - LastActionSpace: Original last action space
    - SubgoalInRobotFrameSpace: Subgoal in robot frame
"""

from .advanced_dynamics_spaces import (
    KinematicStateSpace,
    MotionStateSpace,
    TrajectoryStateSpace,
)
from .basic_dynamics_spaces import LastActionSpace, SubgoalInRobotFrameSpace

__all__ = [
    "LastActionSpace",
    "SubgoalInRobotFrameSpace",
    "MotionStateSpace",
    "KinematicStateSpace",
    "TrajectoryStateSpace",
]
