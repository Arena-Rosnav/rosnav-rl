import numpy as np
from gymnasium import spaces
from rl_utils.utils.observation_collector import (
    PedestrianRelativeLocation,
    PedestrianTypeCollector,
    RobotPoseCollector,
    ObservationDict,
)

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

    name = "PEDESTRIAN_TYPE"
    required_observations = [
        PedestrianTypeCollector,
        PedestrianRelativeLocation,
        RobotPoseCollector,
    ]

    def __init__(
        self,
        num_ped_types: int,
        feature_map_size: int,
        roi_in_m: float,
        *args,
        **kwargs
    ) -> None:
        """
        Initializes a new instance of the PedestrianTypeSpace class.

        Args:
            num_ped_types (int): The number of pedestrian types.
            feature_map_size (int): The size of the feature map.
            roi_in_m (float): The region of interest in meters.
            flatten (bool, optional): Whether to flatten the feature map. Defaults to True.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        self._num_ped_types = num_ped_types
        super().__init__(
            feature_map_size=feature_map_size, roi_in_m=roi_in_m, *args, **kwargs
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
            shape=(self._feature_map_size, self._feature_map_size),
            dtype=int,
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
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
            observation[PedestrianTypeCollector.name],
            observation[PedestrianRelativeLocation.name],
            observation[RobotPoseCollector.name],
        )
