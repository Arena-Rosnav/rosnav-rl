import numpy as np
import crowdsim_msgs.msg as pedsim_msgs
from gymnasium import spaces
from crowdsim_agents.utils import SemanticAttribute
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_social_state")
class PedestrianSocialStateSpace(BaseFeatureMapSpace):
    """
    Represents the observation space for pedestrian social state feature map.

    Args:
        feature_map_size (int): The size of the feature map.
        roi_in_m (float): The region of interest in meters.
        flatten (bool, optional): Whether to flatten the feature map. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _feature_map_size (int): The size of the feature map.
        _roi_in_m (float): The region of interest in meters.
        _flatten (bool): Whether to flatten the feature map.

    Methods:
        get_gym_space: Get the Gym space representation of the feature map.
        _get_semantic_map: Get the semantic map based on semantic data and relative position.
        encode_observation: Encode the observation into a numpy array.
    """

    def __init__(
        self,
        social_state_num: int,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._social_state_num = social_state_num
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
            low=0,
            high=self._social_state_num,
            shape=(self._feature_map_size * self._feature_map_size,),
            dtype=int,
        )

    def _get_semantic_map(
        self,
        semantic_data: pedsim_msgs.SemanticData,
        relative_pos: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Get the semantic map based on semantic data and relative position.

        Args:
            semantic_data (pedsim_msgs.SemanticData): The semantic data.
            relative_pos (np.ndarray): The relative position.

        Returns:
            np.ndarray: The semantic map.
        """
        social_state_map = np.zeros((self.feature_map_size, self.feature_map_size))
        social_states = list(
            map(
                lambda x: int(x.evidence) >> 8,
                semantic_data.points,
            )
        )

        if social_states is not None and relative_pos is not None:
            for state, pos in zip(social_states, relative_pos):
                index = self._get_map_index(pos)
                if (
                    0 <= index[0] < self.feature_map_size
                    and 0 <= index[1] < self.feature_map_size
                ):
                    social_state_map[index] = state

        return social_state_map

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        """
        Encode the observation into a numpy array.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            np.ndarray: The encoded observation as a numpy array.
        """
        return self._get_semantic_map(
            observation[OBS_DICT_KEYS.SEMANTIC.PEDESTRIAN_SOCIAL_STATE.value],
            observation[OBS_DICT_KEYS.SEMANTIC.RELATIVE_LOCATION.value],
        ).flatten()
