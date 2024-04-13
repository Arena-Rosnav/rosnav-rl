import numpy as np
from gymnasium import spaces
from crowdsim_agents.utils import SemanticAttribute
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_vel_x")
class PedestrianVelXSpace(BaseFeatureMapSpace):
    """
    A feature map space representing the pedestrian velocity in the x-direction.

    Args:
        min_speed_x (float): The minimum speed in the x-direction.
        max_speed_x (float): The maximum speed in the x-direction.
        feature_map_size (int): The size of the feature map.
        roi_in_m (float): The region of interest in meters.
        flatten (bool, optional): Whether to flatten the feature map. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _map_size (int): The size of the feature map.
        _min_speed (float): The minimum speed in the x-direction.
        _max_speed (float): The maximum speed in the x-direction.

    Methods:
        get_gym_space(): Returns the Gym space corresponding to the feature map.
        encode_observation(observation, *args, **kwargs): Encodes the observation into a numpy array.

    """

    def __init__(
        self,
        min_speed_x: float,
        max_speed_x: float,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._map_size = feature_map_size
        self._min_speed = min_speed_x
        self._max_speed = max_speed_x
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs
        )

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space corresponding to the feature map.

        Returns:
            spaces.Space: The Gym space object.

        """
        return spaces.Box(
            low=self._min_speed,
            high=self._max_speed,
            shape=(self._map_size * self._map_size,),
            dtype=float,
        )

    def _get_semantic_map(
        self,
        relative_x_vel: np.ndarray = None,
        relative_pos: np.ndarray = None,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Generates a semantic map based on the relative x velocity and position of pedestrians.

        Args:
            relative_x_vel (np.ndarray): Array of relative x velocities of pedestrians.
            relative_pos (np.ndarray): Array of relative positions of pedestrians.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Semantic map representing the x velocity of pedestrians.
        """
        x_vel_map = np.zeros((self.feature_map_size, self.feature_map_size))

        if relative_x_vel is not None and relative_pos is not None:
            for vel_x, pos in zip(relative_x_vel, relative_pos):
                index = self._get_map_index(pos)
                if (
                    0 <= index[0] < self.feature_map_size
                    and 0 <= index[1] < self.feature_map_size
                ):
                    x_vel_map[index] = vel_x

        return x_vel_map

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        """
        Encodes the observation into a numpy array.

        Args:
            observation (dict): The observation dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: The encoded observation as a numpy array.

        """
        return self._get_semantic_map(
            observation[OBS_DICT_KEYS.SEMANTIC.RELATIVE_X_VEL.value],
            observation[OBS_DICT_KEYS.SEMANTIC.RELATIVE_LOCATION.value],
        ).flatten()
