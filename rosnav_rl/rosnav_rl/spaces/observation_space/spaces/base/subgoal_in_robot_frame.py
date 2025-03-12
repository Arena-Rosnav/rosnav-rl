import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import SubgoalLocationInRobotFrameGenerator
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("subgoal_in_robot_frame")
class SubgoalInRobotFrameSpace(BaseObservationSpace):
    """Observation space representing the subgoal position in the robot's coordinate frame.

    This class defines a 2D observation space that contains the relative position (x, y) of
    the subgoal from the robot's perspective. The values are bounded by the maximum distance
    parameter.

    Attributes:
        name (str): Identifier for this observation space type.
        required_observation_units (list): List of required observation generators.
        _max_dist (float): Maximum distance in meters for the subgoal position.
    """
    name = "SUBGOAL_IN_ROBOT_FRAME"
    required_observation_units = [SubgoalLocationInRobotFrameGenerator]

    def __init__(self, subgoal_max_dist: float = 5, *args, **kwargs) -> None:
        self._max_dist = subgoal_max_dist
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the goal observation.

        Returns:
            spaces.Space: The Gym space for the goal observation.

        """
        return spaces.Box(
            low=np.array([-self._max_dist, -self._max_dist]),
            high=np.array([self._max_dist, self._max_dist]),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> SubgoalLocationInRobotFrameGenerator.data_class:
        """
        Encodes the goal observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            ndarray: The encoded goal observation.

        """
        return observation[SubgoalLocationInRobotFrameGenerator.name]
