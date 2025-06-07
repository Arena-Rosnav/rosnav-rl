from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, List, Union

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    RobotPoseCollector,
)
from rosnav_rl.observations.utils.semantic import (
    get_relative_pos_to_robot,
)

from ..base_observation_space import (
    BaseObservationSpace,
    ObservationCollector,
    ObservationGenerator,
)

if TYPE_CHECKING:
    from rosnav_rl.utils.type_aliases import ObservationDict


class BaseFeatureMapSpace(BaseObservationSpace):
    """A base class for creating feature map observation spaces in robotics applications.

    This class provides the foundation for implementing feature map-based observation spaces,
    which are commonly used in robotics for representing spatial information about the environment.
    It handles the conversion between real-world coordinates and feature map indices, and provides
    methods for semantic map generation.

    Attributes:
        name (ClassVar[str]): The name of the feature map space.
        required_observation_units (ClassVar[List[Union[ObservationCollector, ObservationGenerator]]]):
            List of required observation units for this feature map space.
        background_value (ClassVar[int]): Default value for empty/background cells in the feature map.
        _feature_map_size (int): The size of the feature map (width and height).
        _roi_in_m (float): The region of interest in meters.
        _flatten (bool): Whether to flatten the feature map output.

    Example:
        ```python
        class CustomFeatureMap(BaseFeatureMapSpace):
            def __init__(self, feature_map_size=64, roi_in_m=10.0):
                super().__init__(feature_map_size, roi_in_m)
        ```

        - The feature map is square with dimensions (feature_map_size × feature_map_size)
        - The origin (0,0) is mapped to the center of the feature map
        - Real-world coordinates are scaled based on the ROI size and feature map size
    """

    name: ClassVar[str]
    required_observation_units: ClassVar[
        List[Union[ObservationCollector, ObservationGenerator]]
    ] = []
    background_value: ClassVar[int] = 0

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

    @abstractmethod
    def get_gym_space(self) -> spaces.Space:
        raise NotImplementedError()

    def _get_map_index(self, position: tuple) -> tuple:
        """
        Converts real-world coordinates to feature map indices.

        Args:
            position (tuple): A tuple containing at least (x,y) coordinates in meters,
                             additional elements in tuple are ignored.

        Returns:
            tuple: A tuple (x,y) containing the corresponding indices in the feature map.
                  Origin (0,0) is mapped to the center of the feature map.

        Note:
            The conversion is done by scaling the real-world coordinates based on the ROI size
            and feature map size, then shifting by half the feature map size to center the origin.
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
        semantic_data: np.ndarray,
        poses: np.ndarray = None,
        relative_poses: np.ndarray = None,
        robot_pose: RobotPoseCollector.data_class = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Creates a semantic feature map based on provided semantic data.

        This method generates a 2D grid map representing semantic information in the robot's environment.
        Each cell in the map can contain evidence values from semantic data points.

        Args:
            semantic_data: Collected semantic layer data containing points with locations and evidence values
            relative_pos: Optional array of positions relative to the robot. If None, will be calculated
                 from semantic_data and robot_pose
            robot_pose: Robot's current pose (position and orientation), used for calculating relative positions
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: A feature map of shape (1, feature_map_size, feature_map_size) with semantic evidence values
                   placed at corresponding grid cells. Cells with no data contain the background value.
        Note:
            - The method handles cases where no semantic data points are available
            - If a data point falls outside the map boundaries, it will be ignored
            - Any exceptions during processing are logged as warnings
        """
        assert (
            isinstance(semantic_data, np.ndarray)
            and semantic_data.ndim == 1
            and semantic_data.shape[0] == 1
        ), "Semantic data must be a 1D numpy array with shape (N,)"

        assert (
            poses or relative_poses
        ), "Either poses or relative_poses must be provided"

        pos_map = (
            np.zeros((1, self._feature_map_size, self._feature_map_size))
            + self.background_value
        )

        if relative_poses is None and len(semantic_data) == 0:
            return pos_map

        try:
            # If relative_pos is not provided, calculate it
            if relative_poses is None and len(semantic_data) > 0:
                relative_poses = get_relative_pos_to_robot(robot_pose, poses)

            for data, pos in zip(
                semantic_data,
                relative_poses,
            ):
                index = self._get_map_index(pos)
                if (
                    0 <= index[0] < self.feature_map_size
                    and 0 <= index[1] < self.feature_map_size
                ):
                    pos_map[0, index[0], index[1]] = data
        except Exception as e:
            print(f"Exception occurred while processing semantic data: {e}")

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
