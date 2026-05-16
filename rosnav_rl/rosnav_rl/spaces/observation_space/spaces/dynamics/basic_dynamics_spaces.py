"""Basic Dynamics Spaces

Standard dynamics-based observation spaces integrated into hierarchical architecture.
"""

from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import (
    RobotActionVector,
    SubgoalRelativePosition,
)
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.DYNAMICS)
class LastActionSpace(BaseObservationSpace):
    """
    Original observation space representing the last action taken by the agent.

    This space captures the previous action (linear and angular velocities) to provide
    the agent with information about its recent control decisions.
    """

    name = "LastActionSpace"
    requires = {
        "last_action": RobotActionVector,  # Last action taken by the agent
    }

    def __init__(
        self,
        min_linear_vel: float,
        max_linear_vel: float,
        min_angular_vel: float,
        max_angular_vel: float,
        min_translational_vel: float = 0.0,
        max_translational_vel: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        self._min_linear_vel = min_linear_vel
        self._max_linear_vel = max_linear_vel
        self._min_translational_vel = min_translational_vel
        self._max_translational_vel = max_translational_vel
        self._min_angular_vel = min_angular_vel
        self._max_angular_vel = max_angular_vel
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for the last action.

        Returns:
            spaces.Space: The Gym observation space.
        """
        # Check if translational velocity is used (holonomic robot)
        if self._min_translational_vel != 0.0 or self._max_translational_vel != 0.0:
            # 3D action space: [linear_vel, translational_vel, angular_vel]
            return spaces.Box(
                low=np.array(
                    [
                        self._min_linear_vel,
                        self._min_translational_vel,
                        self._min_angular_vel,
                    ]
                ),
                high=np.array(
                    [
                        self._max_linear_vel,
                        self._max_translational_vel,
                        self._max_angular_vel,
                    ]
                ),
                dtype=np.float32,
            )
        else:
            # 2D action space: [linear_vel, angular_vel] (differential drive)
            return spaces.Box(
                low=np.array([self._min_linear_vel, self._min_angular_vel]),
                high=np.array([self._max_linear_vel, self._max_angular_vel]),
                dtype=np.float32,
            )

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, last_action: RobotActionVector, *args, **kwargs
    ) -> RobotActionVector:
        """
        Encodes the last action observation.

        Args:
            last_action: Robot action command vector (last executed action)
                - Shape: (2,) or (3,)
                - Dtype: np.float32
                - Units: [m/s, rad/s] or [m/s, m/s, rad/s]
                - Format: 2D: [linear_vel, angular_vel], 3D: [linear_vel, translational_vel, angular_vel]
                - Example: [0.5, 0.2] (differential drive) or [0.5, 0.1, 0.2] (holonomic)

        Returns:
            RobotActionVector: (Normalized) Robot action command vector (last executed action).
                - Shape: (2,) or (3,)
                - Dtype: np.float32
                - Units: [m/s, rad/s] or [m/s, m/s, rad/s]
                - Format: 2D: [linear_vel, angular_vel], 3D: [linear_vel, translational_vel, angular_vel]
                - Example: [0.5, 0.2] (differential drive) or [0.5, 0.1, 0.2] (holonomic)
        """
        return last_action


@SpaceFactory.register(auto_name=True, category=SpaceCategory.DYNAMICS)
class SubgoalInRobotFrameSpace(BaseObservationSpace):
    """Original observation space representing the subgoal position in the robot's coordinate frame.

    This class defines a 2D observation space that contains the relative position (x, y) of
    the subgoal from the robot's perspective. The values are bounded by the maximum distance
    parameter.
    """

    name = "SubgoalInRobotFrameSpace"
    requires = {
        "subgoal_in_robot_frame": SubgoalRelativePosition,  # Subgoal position in robot's local coordinate frame
    }

    def __init__(self, subgoal_max_dist: float = 5, *args, **kwargs) -> None:
        self._max_dist = subgoal_max_dist
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the subgoal observation.

        Returns:
            spaces.Space: The Gym space for the subgoal observation.
        """
        return spaces.Box(
            low=np.array([-self._max_dist, -self._max_dist]),
            high=np.array([self._max_dist, self._max_dist]),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, subgoal_in_robot_frame: SubgoalRelativePosition, *args, **kwargs
    ) -> SubgoalRelativePosition:
        """
        Encodes the subgoal observation in robot frame.

        Args:
            subgoal_in_robot_frame: Subgoal position in robot's local coordinate frame
                - Shape: (2,) - [x, y] coordinates
                - Dtype: np.float32
                - Units: meters
                - Range: [-subgoal_max_dist, subgoal_max_dist] for both x and y
                - Coordinate frame: x=forward/backward, y=left/right from robot perspective
                - Example: [2.5, -1.3] (2.5m forward, 1.3m to the right)

        Returns:
            SubgoalRelativePosition: (Normalized) Subgoal position in robot's local coordinate frame.
                - Shape: (2,) - [x, y] coordinates
                - Dtype: np.float32
                - Units: meters
                - Range: [-subgoal_max_dist, subgoal_max_dist] for both x and y
                - Coordinate frame: x=forward/backward, y=left/right from robot perspective
                - Example: [2.5, -1.3] (2.5m forward, 1.3m to the right)
        """
        return subgoal_in_robot_frame
