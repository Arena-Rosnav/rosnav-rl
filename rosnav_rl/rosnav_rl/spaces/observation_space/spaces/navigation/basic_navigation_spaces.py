"""Legacy Navigation Spaces - Migrated from Base

Original goal and subgoal navigation spaces integrated into hierarchical architecture.
"""

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import (
    DistanceAngleMetrics,
)
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.NAVIGATION)
class DistAngleToGoalSpace(BaseObservationSpace):
    """Basic navigation observation space for distance and angle to goal.

    This space provides a minimal, production-ready representation of the robot's position relative to its navigation goal.
    It encodes the distance and angle to the goal as a 2D vector, supporting both continuous and discrete navigation tasks.

    Technical Specifications:
    - Distance to Goal: Euclidean distance from robot to goal position
    - Angle to Goal: Relative heading from robot to goal in robot-centric frame

    Configuration:
    - goal_max_dist: Maximum distance to goal (meters)

    Output Format: 2-dimensional vector [distance, angle] for navigation control and reward shaping.

    Applications: Goal-reaching, path planning, reward computation, and curriculum learning.
    """

    name = "DIST_ANGLE_TO_GOAL"

    # Schema-based requirements: defines the data sources needed from observations.yaml
    # Each key corresponds to a data source name, each value provides rich type metadata
    requires = {
        "dist_angle_to_goal": DistanceAngleMetrics,  # Distance and angle to navigation goal
    }

    def __init__(self, goal_max_dist: float = 30, *args, **kwargs) -> None:
        self._max_dist = goal_max_dist
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the goal observation.

        Returns:
            spaces.Space: The Gym space for the goal observation.
        """
        return spaces.Box(
            low=np.array([0, -np.pi]),
            high=np.array([self._max_dist, np.pi]),
            dtype=np.float32,
            shape=(2,),
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, dist_angle_to_goal: DistanceAngleMetrics, *args, **kwargs
    ) -> DistanceAngleMetrics:
        """Encode distance and angle to navigation goal for minimal navigation context.

        Args:
            dist_angle_to_goal (DistanceAngleMetrics): Distance and angle measurements to navigation goal
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [2.5, 0.785] (2.5m away, 45° to the right)

        Returns:
            DistanceAngleMetrics: Encoded navigation goal vector.
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Range: distance ∈ [0, max_dist], angle ∈ [-π, π]
                - Example: [2.5, 0.785] (2.5m away, 45° to the right)
        """
        return dist_angle_to_goal


@SpaceFactory.register(auto_name=True, category=SpaceCategory.NAVIGATION)
class DistAngleToSubgoalSpace(BaseObservationSpace):
    """Basic navigation observation space for distance and angle to subgoal.

    This space provides a minimal, production-ready representation of the robot's position relative to a navigation subgoal.
    It encodes the distance and angle to the subgoal as a 2D vector, supporting hierarchical and curriculum-based navigation tasks.

    Technical Specifications:
    - Distance to Subgoal: Euclidean distance from robot to subgoal position
    - Angle to Subgoal: Relative heading from robot to subgoal in robot-centric frame

    Configuration:
    - subgoal_max_dist: Maximum distance to subgoal (meters)

    Output Format: 2-dimensional vector [distance, angle] for subgoal navigation and reward shaping.

    Applications: Hierarchical navigation, curriculum learning, subgoal-based planning, and reward computation.
    """

    name = "DIST_ANGLE_TO_SUBGOAL"
    requires = {
        "dist_angle_to_subgoal": DistanceAngleMetrics,  # Distance and angle to navigation subgoal
    }

    def __init__(self, subgoal_max_dist: float = 30, *args, **kwargs) -> None:
        self._max_dist = subgoal_max_dist
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the subgoal observation.

        Returns:
            spaces.Space: The Gym space for the subgoal observation.
        """
        return spaces.Box(
            low=np.array([0, -np.pi]),
            high=np.array([self._max_dist, np.pi]),
            dtype=np.float32,
            shape=(2,),
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, dist_angle_to_subgoal: DistanceAngleMetrics, *args, **kwargs
    ) -> DistanceAngleMetrics:
        """Encode distance and angle to navigation subgoal for hierarchical navigation context.

        Args:
            dist_angle_to_subgoal (DistanceAngleMetrics): Distance and angle measurements to navigation subgoal
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [1.2, -0.524] (1.2m away, 30° to the left)

        Returns:
            DistanceAngleMetrics: Encoded navigation subgoal vector.
                - Shape: (2,) - [distance, angle]
                - Dtype: np.float32
                - Units: [meters, radians]
                - Range: distance ∈ [0, max_dist], angle ∈ [-π, π]
                - Example: [1.2, -0.524] (1.2m away, 30° to the left)
        """
        return dist_angle_to_subgoal
