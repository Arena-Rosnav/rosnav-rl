"""Legacy Environment Feature Maps - Migrated from Feature Maps

Original feature map spaces for pedestrian and environmental information
integrated into hierarchical architecture.
"""

from collections import deque

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    LaserCollector,
    PedestrianLocationGenerator,
    PedestrianRelativeLocationGenerator,
    PedestrianRelativeVelXGenerator,
    PedestrianRelativeVelYGenerator,
    PedestrianSocialStateGenerator,
    PedestrianTypeGenerator,
    RobotPoseCollector,
)
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from rosnav_rl.spaces.observation_space.spaces.feature_maps.base_feature_map_space import (
    BaseFeatureMapSpace,
)
from rosnav_rl.utils.type_aliases import ObservationDict


@SpaceFactory.register("ped_vel_x")
class PedestrianVelXSpace(BaseFeatureMapSpace):
    """A space for representing the feature map of pedestrian x-velocity.

    This class creates a spatial representation of pedestrians' x-velocity in the robot's
    environment. The feature map is a 2D grid where each cell represents a location,
    and the value in each cell represents the x-component of the velocity of a pedestrian
    at that location, if present.
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
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._ped_min_speed_x = ped_min_speed_x
        self._ped_max_speed_x = ped_max_speed_x
        self._feature_map_size = feature_map_size
        self._roi_in_m = roi_in_m
        self._flatten = flatten
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for the pedestrian x-velocity feature map.

        Returns:
            spaces.Space: Box space representing the feature map.
        """
        if self._flatten:
            shape = (self._feature_map_size * self._feature_map_size,)
        else:
            shape = (self._feature_map_size, self._feature_map_size)

        return spaces.Box(
            low=self._ped_min_speed_x,
            high=self._ped_max_speed_x,
            shape=shape,
            dtype=np.float32,
        )

    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encodes the pedestrian x-velocity feature map observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            np.ndarray: The encoded feature map.
        """
        # Initialize feature map with zeros
        feature_map = np.zeros(
            (self._feature_map_size, self._feature_map_size), dtype=np.float32
        )

        # Get pedestrian locations and velocities
        ped_locations = observation.get(
            PedestrianRelativeLocationGenerator.name, np.array([])
        )
        ped_vel_x = observation.get(PedestrianRelativeVelXGenerator.name, np.array([]))

        if len(ped_locations) > 0 and len(ped_vel_x) > 0:
            # Convert relative positions to grid indices
            grid_resolution = self._roi_in_m / self._feature_map_size
            center = self._feature_map_size // 2

            for location, vel_x in zip(ped_locations, ped_vel_x):
                # Convert location to grid coordinates
                grid_x = int(location[0] / grid_resolution + center)
                grid_y = int(location[1] / grid_resolution + center)

                # Check bounds
                if (
                    0 <= grid_x < self._feature_map_size
                    and 0 <= grid_y < self._feature_map_size
                ):
                    feature_map[grid_y, grid_x] = vel_x

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register("ped_vel_y")
class PedestrianVelYSpace(BaseFeatureMapSpace):
    """A space for representing the feature map of pedestrian y-velocity."""

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
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._ped_min_speed_y = ped_min_speed_y
        self._ped_max_speed_y = ped_max_speed_y
        self._feature_map_size = feature_map_size
        self._roi_in_m = roi_in_m
        self._flatten = flatten
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for the pedestrian y-velocity feature map.
        """
        if self._flatten:
            shape = (self._feature_map_size * self._feature_map_size,)
        else:
            shape = (self._feature_map_size, self._feature_map_size)

        return spaces.Box(
            low=self._ped_min_speed_y,
            high=self._ped_max_speed_y,
            shape=shape,
            dtype=np.float32,
        )

    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encodes the pedestrian y-velocity feature map observation.
        """
        feature_map = np.zeros(
            (self._feature_map_size, self._feature_map_size), dtype=np.float32
        )

        ped_locations = observation.get(
            PedestrianRelativeLocationGenerator.name, np.array([])
        )
        ped_vel_y = observation.get(PedestrianRelativeVelYGenerator.name, np.array([]))

        if len(ped_locations) > 0 and len(ped_vel_y) > 0:
            grid_resolution = self._roi_in_m / self._feature_map_size
            center = self._feature_map_size // 2

            for location, vel_y in zip(ped_locations, ped_vel_y):
                grid_x = int(location[0] / grid_resolution + center)
                grid_y = int(location[1] / grid_resolution + center)

                if (
                    0 <= grid_x < self._feature_map_size
                    and 0 <= grid_y < self._feature_map_size
                ):
                    feature_map[grid_y, grid_x] = vel_y

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register("stacked_laser_map")
class StackedLaserMapSpace(BaseFeatureMapSpace):
    """A feature map space that stacks laser scan data to create a 2D representation of the environment.

    This class processes laser scan data by maintaining a queue of consecutive scans and
    transforming them into a feature map representation. The resulting map provides
    spatial information about the robot's surroundings based on laser readings.
    """

    name = "STACKED_LASER_MAP"
    required_observation_units = [LaserCollector]

    def __init__(
        self,
        laser_stack_size: int,
        feature_map_size: int,
        roi_in_m: float,
        laser_max_range: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._laser_stack_size = laser_stack_size
        self._feature_map_size = feature_map_size
        self._roi_in_m = roi_in_m
        self._laser_max_range = laser_max_range
        self._flatten = flatten

        # Initialize laser scan stack
        self._laser_stack = deque(maxlen=laser_stack_size)

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for the stacked laser map.
        """
        if self._flatten:
            shape = (self._feature_map_size * self._feature_map_size,)
        else:
            shape = (self._feature_map_size, self._feature_map_size)

        return spaces.Box(
            low=0.0,
            high=self._roi_in_m,
            shape=shape,
            dtype=np.float32,
        )

    def _process_laser_scan(self, laser_scan: np.ndarray) -> np.ndarray:
        """Process a single laser scan into a feature map representation."""
        # Apply range limit
        laser_scan = np.clip(laser_scan, 0, self._laser_max_range)

        # Convert to feature map
        feature_map = np.zeros(
            (self._feature_map_size, self._feature_map_size), dtype=np.float32
        )

        # Simple approach: project laser points onto grid
        n_beams = len(laser_scan)
        angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)

        grid_resolution = self._roi_in_m / self._feature_map_size
        center = self._feature_map_size // 2

        for i, (distance, angle) in enumerate(zip(laser_scan, angles)):
            if distance > 0:  # Valid reading
                # Calculate Cartesian coordinates
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)

                # Convert to grid coordinates
                grid_x = int(x / grid_resolution + center)
                grid_y = int(y / grid_resolution + center)

                # Check bounds and set value
                if (
                    0 <= grid_x < self._feature_map_size
                    and 0 <= grid_y < self._feature_map_size
                ):
                    feature_map[grid_y, grid_x] = max(
                        feature_map[grid_y, grid_x], distance
                    )

        return feature_map

    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encodes the stacked laser map observation.
        """
        # Get current laser scan
        current_scan = observation.get(LaserCollector.name, np.array([]))

        if len(current_scan) > 0:
            # Process current scan
            current_map = self._process_laser_scan(current_scan)

            # Add to stack
            self._laser_stack.append(current_map)

        # If stack is not full, pad with zeros
        while len(self._laser_stack) < self._laser_stack_size:
            self._laser_stack.append(
                np.zeros(
                    (self._feature_map_size, self._feature_map_size), dtype=np.float32
                )
            )

        # Combine stacked maps (take maximum values)
        combined_map = np.maximum.reduce(list(self._laser_stack))

        return combined_map.flatten() if self._flatten else combined_map


@SpaceFactory.register("ped_location", SpaceCategory.ENVIRONMENT)
class PedestrianLocationSpace(BaseFeatureMapSpace):
    """A feature map space representing pedestrian locations in a grid.

    This class creates a 2D feature map where each cell indicates the presence
    or absence of pedestrians in that spatial region. The feature map is
    generated using pedestrian location data and relative positioning information.

    Attributes:
        name (str): Identifier for this space type, set to "PEDESTRIAN_LOCATION".
        required_observation_units (list): Units required for generating observations:
            - PedestrianLocationGenerator: Provides absolute positions of pedestrians
            - PedestrianRelativeLocationGenerator: Provides relative positions of pedestrians

    Parameters:
        feature_map_size (int): Size of the feature map (width and height in cells).
        roi_in_m (float): Region of interest in meters around the robot.
        *args: Variable length argument list passed to parent class.
        **kwargs: Arbitrary keyword arguments passed to parent class.

    The feature map is a binary representation where cells contain 1 if a pedestrian
    is present in that spatial location, and 0 otherwise.
    """

    name = "PEDESTRIAN_LOCATION"
    required_observation_units = [
        PedestrianLocationGenerator,
        PedestrianRelativeLocationGenerator,
    ]
    background_value = 0

    def get_gym_space(self) -> spaces.Space:
        """
        Get the Gym space representation of the feature map.

        Returns:
            spaces.Space: The Gym space representing the feature map.
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
        Encode the observation into a numpy array.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            np.ndarray: The encoded observation as a numpy array.
        """
        # Create binary data (1 for pedestrian presence)
        poses = observation[PedestrianLocationGenerator.name]
        binary_data = [1] * len(poses) if poses is not None else []

        return self._get_semantic_map(
            semantic_data=binary_data,
            poses=observation[PedestrianLocationGenerator.name],
            relative_poses=observation[PedestrianRelativeLocationGenerator.name],
            robot_pose=observation[PedestrianLocationGenerator.name],
        )


@SpaceFactory.register("ped_social_state", SpaceCategory.ENVIRONMENT)
class PedestrianSocialStateSpace(BaseFeatureMapSpace):
    """A feature map space representing pedestrian social states in a grid.

    This class creates a 2D feature map where each cell contains a social state value
    for pedestrians detected in that spatial region. The social state is an integer
    value extracted from semantic data.

    Attributes:
        name (str): Identifier for this space type, set to "PEDESTRIAN_SOCIAL_STATE".
        required_observation_units (list): Units required for generating observations:
            - PedestrianSocialStateCollector: Collects social state data from pedestrians
            - PedestrianRelativeLocationGenerator: Provides relative positions of pedestrians

    Parameters:
        ped_social_state_num (int): Number of possible pedestrian social states.
        feature_map_size (int): Size of the feature map (width and height in cells).
        roi_in_m (float): Region of interest in meters around the robot.
        *args: Variable length argument list passed to parent class.
        **kwargs: Arbitrary keyword arguments passed to parent class.

    The feature map encodes pedestrian social states as integer values in a grid,
    where each cell corresponds to a spatial location. The social state is extracted
    from the 'evidence' field of pedestrian data points by bit-shifting.
    """

    name = "PEDESTRIAN_SOCIAL_STATE"
    required_observation_units = [
        PedestrianLocationGenerator,
        PedestrianSocialStateGenerator,
        PedestrianRelativeLocationGenerator,
    ]
    background_value = -1

    def __init__(
        self,
        ped_social_state_num: int,
        feature_map_size: int,
        roi_in_m: float,
        *args,
        **kwargs
    ) -> None:
        self._social_state_num = ped_social_state_num
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
            low=0,
            high=self._social_state_num,
            shape=(1, self._feature_map_size, self._feature_map_size),
            dtype=int,
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encode the observation into a numpy array.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            np.ndarray: The encoded observation as a numpy array.
        """
        return self._get_semantic_map(
            observation[PedestrianSocialStateGenerator.name],
            poses=observation[PedestrianLocationGenerator.name],
            relative_poses=observation[PedestrianRelativeLocationGenerator.name],
            robot_pose=observation[PedestrianLocationGenerator.name],
        )


@SpaceFactory.register("ped_type", SpaceCategory.ENVIRONMENT)
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
        PedestrianTypeGenerator,
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
            ped_num_types (int): The number of pedestrian types.
            feature_map_size (int): The size of the feature map.
            roi_in_m (float): The region of interest in meters.
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
            semantic_data=observation[PedestrianTypeGenerator.name],
            poses=observation[PedestrianLocationGenerator.name],
            relative_poses=observation[PedestrianRelativeLocationGenerator.name],
            robot_pose=observation[RobotPoseCollector.name],
        )
