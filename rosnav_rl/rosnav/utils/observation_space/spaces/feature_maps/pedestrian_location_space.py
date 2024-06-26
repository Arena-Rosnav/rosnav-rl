import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace

from rl_utils.utils.observation_collector import (
    PedestrianLocationCollector,
    RobotPoseCollector,
    PedestrianRelativeLocation,
    ObservationDict,
)


@SpaceFactory.register("ped_location")
class PedestrianLocationSpace(BaseFeatureMapSpace):
    """
    A class representing the observation space for pedestrian locations.

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
        get_gym_space: Returns the gym space for the observation.
        encode_observation: Encodes the observation into a numpy array.
    """

    name = "PEDESTRIAN_LOCATION"
    required_observations = [
        PedestrianLocationCollector,
        PedestrianRelativeLocation,
        RobotPoseCollector,
    ]

    def __init__(
        self,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
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
            high=1,
            shape=(self._feature_map_size, self._feature_map_size),
            dtype=int,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
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
            observation[PedestrianLocationCollector.name],
            observation[PedestrianRelativeLocation.name],
            observation[RobotPoseCollector.name],
        )
