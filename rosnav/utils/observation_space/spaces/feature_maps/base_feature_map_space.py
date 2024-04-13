from abc import abstractmethod
from typing import List

import numpy as np
import rospy

import crowdsim_msgs.msg as crowdsim_msgs
from geometry_msgs.msg import Point, Pose2D


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
        semantic_data: crowdsim_msgs.SemanticData,
        relative_pos: np.ndarray = None,
        robot_pose: Pose2D = None,
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
                relative_pos = BaseFeatureMapSpace.get_relative_pos_to_robot(
                    robot_pose, semantic_data.points
                )

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

    @staticmethod
    def get_relative_pos_to_robot(
        robot_pose: Pose2D, distant_frames: crowdsim_msgs.SemanticData
    ):
        robot_pose_array = np.array([robot_pose.x, robot_pose.y, robot_pose.theta])
        # homogeneous transformation matrix: map_T_robot
        map_T_robot = np.array(
            [
                [
                    np.cos(robot_pose_array[2]),
                    -np.sin(robot_pose_array[2]),
                    robot_pose_array[0],
                ],
                [
                    np.sin(robot_pose_array[2]),
                    np.cos(robot_pose_array[2]),
                    robot_pose_array[1],
                ],
                [0, 0, 1],
            ]
        )
        robot_T_map = np.linalg.inv(map_T_robot)

        ped_pos = np.stack(
            [[frame.location.x, frame.location.y, 1] for frame in distant_frames]
        )
        return np.matmul(robot_T_map, ped_pos.T)

    @abstractmethod
    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
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
