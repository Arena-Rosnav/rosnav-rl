import numpy as np
from gymnasium import spaces


class BaseSpaceEncoder:
    """
    Base class for space encoders used in ROS navigation.
    """

    def __init__(
        self,
        laser_num_beams: int,
        laser_max_range: float,
        radius: float,
        holonomic: bool,
        actions: dict,
        action_space_discrete: bool,
        stacked: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize the BaseSpaceEncoder.

        Args:
            laser_num_beams (int): Number of laser beams.
            laser_max_range (float): Maximum range of the laser.
            radius (float): Radius of the robot.
            holonomic (bool): Flag indicating whether the robot is holonomic or not.
            actions (dict): Dictionary of available actions.
            action_space_discrete (bool): Flag indicating whether the action space is discrete or continuous.
            stacked (bool, optional): Flag indicating whether the observations are stacked. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._laser_num_beams = laser_num_beams
        self._laser_max_range = laser_max_range
        self._radius = radius
        self._is_holonomic = holonomic
        self._actions = actions
        self._is_action_space_discrete = action_space_discrete
        self._stacked = stacked

    @property
    def observation_space(self) -> spaces.Box:
        """
        Get the observation space.

        Returns:
            spaces.Box: The observation space.
        """
        raise NotImplementedError()

    @property
    def action_space(self) -> spaces.Box:
        """
        Get the action space.

        Returns:
            spaces.Box: The action space.
        """
        raise NotImplementedError()

    def decode_action(self, action) -> np.ndarray:
        """
        Decode the action.

        Args:
            action: The action to decode.

        Returns:
            np.ndarray: The decoded action.
        """
        raise NotImplementedError()

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        """
        Encode the observation.

        Args:
            observation (dict): The observation to encode.

        Returns:
            np.ndarray: The encoded observation.
        """
        raise NotImplementedError()
