import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace

from rl_utils.utils.observation_collector import (
    PedestrianRelativeLocation,
    PedestrianRelativeVelY,
    ObservationDict,
)


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

    name = "PEDESTRIAN_VEL_Y"
    required_observations = [PedestrianRelativeLocation, PedestrianRelativeVelY]

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
            shape=(self._feature_map_size, self._feature_map_size),
            dtype=float,
        )

    def _get_semantic_map(
        self,
        relative_pos: PedestrianRelativeLocation.data_class = None,
        relative_y_vel: PedestrianRelativeVelY.data_class = None,
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
            observation[PedestrianRelativeLocation.name],
            observation[PedestrianRelativeVelY.name],
        )
