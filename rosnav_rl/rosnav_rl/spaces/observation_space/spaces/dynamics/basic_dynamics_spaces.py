"""Basic Dynamics Spaces

Standard dynamics-based observation spaces integrated into hierarchical architecture.
"""

from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    LastActionCollector,
    SubgoalLocationInRobotFrameGenerator,
)
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("last_action", SpaceCategory.DYNAMICS)
class LastActionSpace(BaseObservationSpace):
    """
    Original observation space representing the last action taken by the agent.

    This space captures the previous action (linear and angular velocities) to provide
    the agent with information about its recent control decisions.
    """

    name = "LAST_ACTION"
    required_observation_units = [LastActionCollector]

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
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """
        Encodes the last action observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            Last action data.
        """
        return observation[LastActionCollector.name]


@SpaceFactory.register("subgoal_in_robot_frame")
class SubgoalInRobotFrameSpace(BaseObservationSpace):
    """Original observation space representing the subgoal position in the robot's coordinate frame.

    This class defines a 2D observation space that contains the relative position (x, y) of
    the subgoal from the robot's perspective. The values are bounded by the maximum distance
    parameter.
    """

    name = "SUBGOAL_IN_ROBOT_FRAME"
    required_observation_units = [SubgoalLocationInRobotFrameGenerator]

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
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """
        Encodes the subgoal observation in robot frame.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            Subgoal position in robot frame.
        """
        return observation[SubgoalLocationInRobotFrameGenerator.name]
