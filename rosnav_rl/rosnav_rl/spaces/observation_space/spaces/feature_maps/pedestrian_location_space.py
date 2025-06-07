import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    PedestrianLocationGenerator,
    PedestrianRelativeLocationGenerator,
    RobotPoseCollector,
)
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_location")
class PedestrianLocationSpace(BaseFeatureMapSpace):
    """A feature map observation space for pedestrian locations.

    This class represents a feature map in the observation space that captures the locations of pedestrians
    in the environment relative to the robot's position. It creates a 2D grid representation where pedestrian
    locations are marked.

    Attributes:
        name (str): Identifier for this observation space ("PEDESTRIAN_LOCATION").
        required_observation_units (list): List of collector and generator classes required
            to build this observation space.

    Parameters:
        feature_map_size (int): The size of the feature map grid (N x N).
        roi_in_m (float): Region of interest in meters, representing the physical area covered by the feature map.
        flatten (bool, optional): If True, the resulting observation will be flattened to 1D. Defaults to False.
        *args: Additional positional arguments passed to the parent class.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    name = "PEDESTRIAN_LOCATION"
    required_observation_units = [
        PedestrianLocationGenerator,
        PedestrianRelativeLocationGenerator,
        RobotPoseCollector,
    ]

    def __init__(
        self,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = False,
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
            semantic_data=np.ones(
                shape=(len(observation[PedestrianLocationGenerator.name]),)
            ),
            poses=observation[PedestrianLocationGenerator.name],
            relative_poses=observation[PedestrianRelativeLocationGenerator.name],
            robot_pose=observation[RobotPoseCollector.name],
        )
