from abc import abstractmethod
from typing import List, Union

import numpy as np
import rospy
from rl_utils.utils.observation_collector import (
    RobotPoseCollector,
    SemanticLayerCollector,
    ObservationDict,
)
from rl_utils.utils.observation_collector.utils.semantic import (
    get_relative_pos_to_robot,
)

from ..base_observation_space import (
    BaseObservationSpace,
    ObservationCollector,
    ObservationGenerator,
)


class BaseFeatureMapSpace(BaseObservationSpace):
    """
    Base class for feature map observation spaces.
    """

    name: str
    required_observations: List[Union[ObservationCollector, ObservationGenerator]] = []

    def __init__(
        self,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = False,
        *args,
        **kwargs,
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

        super().__init__(*args, **kwargs)

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
            self._feature_map_size // 2
        )
        y = int((y / self._roi_in_m) * self._feature_map_size) + (
            self._feature_map_size // 2
        )
        return x, y

    def _get_semantic_map(
        self,
        semantic_data: SemanticLayerCollector.data_class,
        relative_pos: np.ndarray = None,
        robot_pose: RobotPoseCollector.data_class = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Get the semantic map based on the given semantic data, relative position, and robot pose.

        Args:
            semantic_data (pedsim_msgs.SemanticData): The semantic data containing information about the environment.
            relative_pos (np.ndarray, optional): The relative positions of the semantic data points to the robot. Defaults to None.
            robot_pose (Pose2D, optional): The pose of the robot. Defaults to None.

        Returns:
            np.ndarray: The semantic map.
        """

        pos_map = np.zeros((self._feature_map_size, self._feature_map_size))

        if relative_pos is None and len(semantic_data.points) == 0:
            return pos_map

        try:
            # If relative_pos is not provided, calculate it
            if relative_pos is None and len(semantic_data.points) > 0:
                ped_points = np.stack(
                    [
                        [frame.location.x, frame.location.y, 1]
                        for frame in semantic_data.points
                    ]
                )
                relative_pos = get_relative_pos_to_robot(robot_pose, ped_points)

            for data, pos in zip(
                semantic_data.points,
                relative_pos,
            ):
                index = self._get_map_index(pos)
                if (
                    0 <= index[0] < self.feature_map_size
                    and 0 <= index[1] < self.feature_map_size
                ):
                    pos_map[index] = data.evidence
        except Exception as e:
            rospy.logwarn(e)

        return pos_map

    @abstractmethod
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
        raise NotImplementedError
