import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    PedestrianRelativeLocationGenerator,
    PedestrianRelativeVelYGenerator,
)
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_vel_y")
class PedestrianVelYSpace(BaseFeatureMapSpace):
    """A class for creating and processing feature maps based on the y-component of pedestrians' velocities.

    This class creates a 2D feature map where each cell represents a spatial location, and the value
    in that cell represents the y-component of a pedestrian's velocity if a pedestrian is present at
    that location. The feature map provides information about pedestrian movement in the y-direction
    (typically forward/backward movement in the robot's reference frame).

    Attributes:
        name (str): The name identifier for this space, "PEDESTRIAN_VEL_Y".
        required_observation_units (list): List of required observation generators needed for this space.
        ped_min_speed_y (float): Minimum y-velocity value for pedestrians (m/s).
        ped_max_speed_y (float): Maximum y-velocity value for pedestrians (m/s).
        feature_map_size (int): Size of the feature map (width and height in cells).
        roi_in_m (float): Region of interest in meters, defining the real-world area covered by the feature map.
        *args: Additional positional arguments passed to the parent class.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    name = "PEDESTRIAN_VEL_Y"
    required_observation_units = [
        PedestrianRelativeLocationGenerator,
        PedestrianRelativeVelYGenerator,
    ]

    def __init__(
        self,
        ped_min_speed_y: float,
        ped_max_speed_y: float,
        feature_map_size: int,
        roi_in_m: float,
        *args,
        **kwargs
    ) -> None:
        self._min_speed = ped_min_speed_y
        self._max_speed = ped_max_speed_y
        super().__init__(
            feature_map_size=feature_map_size, roi_in_m=roi_in_m, *args, **kwargs
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
            shape=(1, self._feature_map_size, self._feature_map_size),
            dtype=float,
        )

    def _get_semantic_map(
        self,
        relative_pos: PedestrianRelativeLocationGenerator.data_class = None,
        relative_y_vel: PedestrianRelativeVelYGenerator.data_class = None,
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
        y_vel_map = np.zeros((self.feature_map_size, self.feature_map_size))

        if relative_y_vel is not None and relative_pos is not None:
            for vel_y, pos in zip(relative_y_vel, relative_pos):
                index = self._get_map_index(pos)
                if (
                    0 <= index[0] < self.feature_map_size
                    and 0 <= index[1] < self.feature_map_size
                ):
                    y_vel_map[index] = vel_y

        return y_vel_map

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encode the observation into a numpy array.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            np.ndarray: The encoded observation as a numpy array.
        """
        return self._get_semantic_map(
            observation[PedestrianRelativeLocationGenerator.name],
            observation[PedestrianRelativeVelYGenerator.name],
        )
