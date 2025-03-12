import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    PedestrianRelativeLocationGenerator,
    PedestrianTypeCollector,
    RobotPoseCollector,
)
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_type")
class PedestrianTypeSpace(BaseFeatureMapSpace):
    """A space for representing pedestrian types as a feature map.
    
    This class inherits from BaseFeatureMapSpace and creates a feature map that encodes
    different pedestrian types in the robot's environment. It uses pedestrian type information,
    relative locations, and the robot's pose to generate a semantic map where each cell 
    represents the type of pedestrian (if any) at that location.
    
    Attributes:
        name (str): The name identifier for this space, set to "PEDESTRIAN_TYPE".
        required_observation_units (list): The observation collectors and generators required
            for this space to function, including pedestrian type information, pedestrian
            relative locations, and robot pose.
        background_value (int): The default value for cells with no pedestrian, set to -1.
    """

    name = "PEDESTRIAN_TYPE"
    required_observation_units = [
        PedestrianTypeCollector,
        PedestrianRelativeLocationGenerator,
        RobotPoseCollector,
    ]
    background_value = -1

    def __init__(
        self,
        ped_num_types: int,
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
        self._num_ped_types = ped_num_types
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
            low=-1,
            high=self._num_ped_types,
            shape=(1, self._feature_map_size, self._feature_map_size),
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
            observation[PedestrianRelativeLocationGenerator.name],
            observation[RobotPoseCollector.name],
        )
