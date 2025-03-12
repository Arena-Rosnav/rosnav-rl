import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    PedestrianRelativeLocationGenerator,
    PedestrianRelativeVelXGenerator,
)
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("ped_vel_x")
class PedestrianVelXSpace(BaseFeatureMapSpace):
    """A space for representing the feature map of pedestrian x-velocity.
    
    This class creates a spatial representation of pedestrians' x-velocity in the robot's
    environment. The feature map is a 2D grid where each cell represents a location,
    and the value in each cell represents the x-component of the velocity of a pedestrian
    at that location, if present.
    The feature map uses relative positions to place pedestrians on the grid and assigns
    their x-velocity to the corresponding cells. Empty cells (without pedestrians) have zero values.
    
    Attributes:
        name (str): The name identifier for this space ("PEDESTRIAN_VEL_X").
        required_observation_units (list): The observation generators required for this feature map:
            - PedestrianRelativeLocationGenerator: Provides relative positions of pedestrians
            - PedestrianRelativeVelXGenerator: Provides x-velocity component of pedestrians
    
    Parameters:
        ped_min_speed_x (float): The minimum possible x-velocity of pedestrians.
        ped_max_speed_x (float): The maximum possible x-velocity of pedestrians.
        feature_map_size (int): The size of the feature map (width and height in cells).
        roi_in_m (float): Region of interest in meters, representing the physical size of the area
                          covered by the feature map.
    """

    name = "PEDESTRIAN_VEL_X"
    required_observation_units = [
        PedestrianRelativeLocationGenerator,
        PedestrianRelativeVelXGenerator,
    ]

    def __init__(
        self,
        ped_min_speed_x: float,
        ped_max_speed_x: float,
        feature_map_size: int,
        roi_in_m: float,
        *args,
        **kwargs
    ) -> None:
        self._min_speed = ped_min_speed_x
        self._max_speed = ped_max_speed_x
        super().__init__(
            feature_map_size=feature_map_size, roi_in_m=roi_in_m, *args, **kwargs
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
            shape=(1, self._feature_map_size, self._feature_map_size),
            dtype=float,
        )

    def _get_semantic_map(
        self,
        relative_pos: PedestrianRelativeLocationGenerator.data_class = None,
        relative_x_vel: PedestrianRelativeVelXGenerator.data_class = None,
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
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encodes the observation into a numpy array.

        Args:
            observation (ObservationDict): The observation dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: The encoded observation as a numpy array.

        """
        return self._get_semantic_map(
            observation[PedestrianRelativeLocationGenerator.name],
            observation[PedestrianRelativeVelXGenerator.name],
        )
