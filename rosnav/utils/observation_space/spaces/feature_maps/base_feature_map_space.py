from typing import List

import numpy as np
from pedsim_msgs.msg import SemanticDatum

from ..base_observation_space import BaseObservationSpace


class BaseFeatureMapSpace(BaseObservationSpace):
    """
    Base class for feature map observation spaces.
    """

    def __init__(
        self,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        """
        Initialize the BaseFeatureMapSpace.

        Args:
            feature_map_size (int): The size of the feature map.
            roi_in_m (float): The region of interest in meters.
            flatten (bool, optional): Whether to flatten the feature map. Defaults to True.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._feature_map_size = feature_map_size
        self._roi_in_m = roi_in_m
        self._flatten = flatten

        self._space = self.get_gym_space()

    @property
    def feature_map_size(self):
        """
        Get the size of the feature map.

        Returns:
            int: The size of the feature map.
        """
        return self._feature_map_size

    def _get_map_index(self, position: tuple) -> tuple:
        """
        Get the map index for a given position.

        Args:
            position (tuple): The position coordinates.

        Returns:
            tuple: The map index.
        """
        x, y, *_ = position
        x = int((x / self._roi_in_m) * self._feature_map_size) + (
            self._feature_map_size // 2 - 1
        )
        y = int((y / self._roi_in_m) * self._feature_map_size) + (
            self._feature_map_size // 2 - 1
        )
        x = min(max(x, 0), self._feature_map_size - 1)
        y = min(max(y, 0), self._feature_map_size - 1)
        return x, y

    def _get_semantic_map(
        self, semantic_data: List[SemanticDatum], robot_pose
    ) -> np.ndarray:
        """
        Get the semantic map based on the semantic data and robot pose.

        Args:
            semantic_data (List[SemanticDatum]): The semantic data.
            robot_pose: The robot pose.

        Returns:
            np.ndarray: The semantic map.
        """
        pos_map = np.zeros((self._feature_map_size, self._feature_map_size))
        map_size = pos_map.shape[0]

        for data in semantic_data:
            relative_pos = BaseFeatureMapSpace.get_relative_pos(
                data.location, robot_pose
            )
            index = self._get_map_index(relative_pos)
            if 0 <= index[0] < map_size and 0 <= index[1] < map_size:
                pos_map[index] = data.evidence

        return pos_map

    @staticmethod
    def get_relative_pos(reference_frame, distant_frame) -> tuple:
        """
        Get the relative position between a reference frame and a distant frame.

        Args:
            reference_frame: The reference frame.
            distant_frame: The distant frame.

        Returns:
            tuple: The relative position.
        """
        return (
            distant_frame.x - reference_frame.x,
            distant_frame.y - reference_frame.y,
            distant_frame.z - reference_frame.z,
        )
