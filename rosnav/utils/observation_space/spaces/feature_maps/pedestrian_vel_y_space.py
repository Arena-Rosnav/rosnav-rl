import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from .base_feature_map_space import BaseFeatureMapSpace

from pedsim_agents.utils import SemanticAttribute

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


@SpaceFactory.register("ped_vel_y")
class PedestrianVelYSpace(BaseFeatureMapSpace):
    """
    A feature map space representing the y-component of pedestrian velocity.

    Args:
        min_speed_y (float): The minimum y-component of pedestrian velocity.
        max_speed_y (float): The maximum y-component of pedestrian velocity.
        feature_map_size (int): The size of the feature map.
        roi_in_m (float): The region of interest in meters.
        flatten (bool, optional): Whether to flatten the feature map. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _map_size (int): The size of the feature map.
        _min_speed (float): The minimum y-component of pedestrian velocity.
        _max_speed (float): The maximum y-component of pedestrian velocity.
    """

    def __init__(
        self,
        min_speed_y: float,
        max_speed_y: float,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._map_size = feature_map_size
        self._min_speed = min_speed_y
        self._max_speed = max_speed_y
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs
        )

    def get_gym_space(self) -> spaces.Space:
        """
        Get the Gym space representation of the feature map.

        Returns:
            spaces.Space: The Gym space representing the feature map.
        """
        return spaces.Box(
            low=self._min_speed,
            high=self._max_speed,
            shape=(self._map_size * self._map_size,),
            dtype=float,
        )

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        """
        Encode the observation into a numpy array.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            np.ndarray: The encoded observation as a numpy array.
        """
        return self._get_semantic_map(
            observation[SemanticAttribute.PEDESTRIAN_VEL_X.value],
            observation[OBS_DICT_KEYS.ROBOT_POSE],
        )
