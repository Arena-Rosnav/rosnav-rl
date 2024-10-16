import numpy as np
import rospy

from gymnasium import spaces
from rl_utils.utils.observation_collector import (
    SubgoalLocationInRobotFrame,
    ObservationDict,
)

from ...observation_space_factory import SpaceFactory
from ...utils import stack_spaces
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("subgoal_in_robot_frame")
class SubgoalInRobotFrameSpace(BaseObservationSpace):
    """
    Represents the observation space for the goal in a navigation task.

    Args:
        goal_max_dist (float): The maximum distance to the goal. Defaults to 30.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _max_dist (float): The maximum distance to the goal.

    Methods:
        get_gym_space: Returns the Gym space for the goal observation.
        encode_observation: Encodes the goal observation.

    """

    name = "SUBGOAL_IN_ROBOT_FRAME"
    required_observations = [SubgoalLocationInRobotFrame]

    def __init__(
        self, ns, goal_max_dist: float = 5, *args, **kwargs
    ) -> None:
        self._ns = ns
        self._max_dist = goal_max_dist
        super().__init__(*args, **kwargs)

        rospy.set_param(f"{self._ns}/follow_subgoal", True)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the goal observation.

        Returns:
            spaces.Space: The Gym space for the goal observation.

        """
        return spaces.Box(
            low=np.array([[-self._max_dist, -self._max_dist]]),
            high=np.array([[self._max_dist, self._max_dist]]),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> SubgoalLocationInRobotFrame.data_class:
        """
        Encodes the goal observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            ndarray: The encoded goal observation.

        """
        return observation[SubgoalLocationInRobotFrame.name][np.newaxis, :]
