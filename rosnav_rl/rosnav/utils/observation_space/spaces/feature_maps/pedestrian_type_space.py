import numpy as np
from gymnasium import spaces
from crowdsim_agents.utils import SemanticAttribute
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_type")
class PedestrianTypeSpace(BaseFeatureMapSpace):
    """
    A class representing the observation space for pedestrian types in a feature map.

    Args:
        num_ped_types (int): The number of pedestrian types.
        feature_map_size (int): The size of the feature map.
        roi_in_m (float): The region of interest in meters.
        flatten (bool, optional): Whether to flatten the feature map. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _num_ped_types (int): The number of pedestrian types.

    Methods:
        get_gym_space: Returns the gym space for the observation.
        encode_observation: Encodes the observation into a numpy array.

    """

    def __init__(
        self,
        num_ped_types: int,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._num_ped_types = num_ped_types
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs
        )

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the gym space for the observation.

        Returns:
            spaces.Space: The gym space for the observation.

        """
        return spaces.Box(
            low=0,
            high=self._num_ped_types,
            shape=(self._feature_map_size * self._feature_map_size,),
            dtype=int,
        )

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
            observation[SemanticAttribute.PEDESTRIAN_TYPE.value],
            observation[OBS_DICT_KEYS.SEMANTIC.RELATIVE_LOCATION.value],
            observation[OBS_DICT_KEYS.ROBOT_POSE],
        ).flatten()
