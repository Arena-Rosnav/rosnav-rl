"""Environment Feature Map Spaces - Schema-Based Architecture

Feature map spaces for pedestrian and environmental information with proper
type annotations and schema-based requirements.
"""

from abc import abstractmethod
from collections import deque
from typing import ClassVar, Union

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import (
    IsTerminal,
    LaserFeatureMap,
    LidarRanges,
    PedestrianFeatureMap,
    PedestrianRelativeLocations,
    PedestrianRelativeVelocities,
    PedestrianSocialStates,
    PedestrianTypeArray,
    PedestrianTypeFeatureMap,
    SemanticFeatureMap,
    SocialStateFeatureMap,
)
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)


class BaseFeatureMapSpace(BaseObservationSpace):
    """Abstract base class for 2D feature map observation spaces.

    Provides the foundation for implementing spatial feature map-based observation spaces, representing
    environmental or agent-centric information in a 2D grid format. Handles coordinate conversion,
    grid creation, and common feature map operations for semantic, social, and sensor-based maps.

    Technical Specifications:
    - Feature Map Size: Configurable grid resolution for spatial encoding
    - ROI: Region of interest in meters, centered on the robot
    - Flattening: Option to output flattened or 2D feature maps

    Output Format: 2D or 1D numpy arrays representing spatial features for downstream learning.

    Applications: Semantic mapping, pedestrian tracking, sensor fusion, and spatial reasoning.
    """

    background_value: ClassVar[int] = 0

    def __init__(
        self,
        feature_map_size: int = 80,
        roi_in_m: float = 20.0,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the BaseFeatureMapSpace.

        Args:
            feature_map_size: The size of the feature map (width and height).
            roi_in_m: The region of interest in meters around the robot.
            flatten: Whether to flatten the feature map output.
        """
        self._feature_map_size = feature_map_size
        self._roi_in_m = roi_in_m
        self._flatten = flatten
        super().__init__(*args, **kwargs)

    @property
    def feature_map_size(self) -> int:
        """Get the size of the feature map."""
        return self._feature_map_size

    def _get_map_index(self, position: Union[tuple, np.ndarray]) -> tuple:
        """Convert real-world coordinates to feature map indices.

        Args:
            position: (x, y) coordinates in meters.

        Returns:
            (x, y) indices in the feature map, with origin at center.
        """
        x, y = position[0], position[1]

        # Scale and center coordinates
        grid_resolution = self._roi_in_m / self._feature_map_size
        center = self._feature_map_size // 2

        grid_x = int(x / grid_resolution + center)
        grid_y = int(y / grid_resolution + center)

        return grid_x, grid_y

    def _create_feature_map(self) -> np.ndarray:
        """Create an empty feature map with background values."""
        if self._flatten:
            return np.full(
                (self._feature_map_size * self._feature_map_size,),
                self.background_value,
                dtype=np.float32,
            )
        else:
            return np.full(
                (self._feature_map_size, self._feature_map_size),
                self.background_value,
                dtype=np.float32,
            )

    def _get_feature_map_shape(self) -> tuple:
        """Get the shape for the feature map."""
        if self._flatten:
            return (self._feature_map_size * self._feature_map_size,)
        else:
            return (self._feature_map_size, self._feature_map_size)

    @abstractmethod
    def get_gym_space(self) -> spaces.Space:
        """Get the Gym space for this feature map."""
        raise NotImplementedError

    @abstractmethod
    def encode_observation(
        self, *args, **kwargs
    ) -> Union[PedestrianFeatureMap, LaserFeatureMap, SemanticFeatureMap]:
        """Encode observation into feature map."""
        raise NotImplementedError


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianVelXSpace(BaseFeatureMapSpace):
    """Pedestrian x-velocity feature map observation space.

    Encodes the x-component of pedestrian velocities in a spatial 2D grid centered on the robot.
    Each cell represents the x-velocity of a pedestrian at that location, enabling learning of
    flow patterns and dynamic obstacle prediction.

    Technical Specifications:
    - Pedestrian X-Velocity: Forward/backward velocity in robot-centric frame
    - Feature Map Size: Configurable grid resolution
    - ROI: Region of interest in meters
    - Flattening: Option to output flattened or 2D feature maps

    Output Format: 2D or 1D numpy array with x-velocity values per grid cell.

    Applications: Crowd flow prediction, dynamic obstacle avoidance, and social navigation.
    """

    name = "PEDESTRIAN_VEL_X"
    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
        "pedestrian_vel_x": PedestrianRelativeVelocities,
    }

    def __init__(
        self,
        ped_min_speed_x: float,
        ped_max_speed_x: float,
        feature_map_size: int = 80,
        roi_in_m: float = 20.0,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._ped_min_speed_x = ped_min_speed_x
        self._ped_max_speed_x = ped_max_speed_x
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def get_gym_space(self) -> spaces.Space:
        """Returns the Gym observation space for the pedestrian x-velocity feature map."""
        return spaces.Box(
            low=self._ped_min_speed_x,
            high=self._ped_max_speed_x,
            shape=self._get_feature_map_shape(),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        pedestrian_vel_x: PedestrianRelativeVelocities,
        *args,
        **kwargs,
    ) -> PedestrianFeatureMap:
        """Encodes the pedestrian x-velocity feature map observation.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian positions relative to robot
                - Shape: (N, 2) where N = number of pedestrians
                - Units: meters
                - Source: pedestrian detection/tracking system
                - Constraints: positions relative to robot base frame
                - Example: [[2.5, 1.0], [-1.5, 3.0]] (2 pedestrians relative to robot)
            pedestrian_vel_x (PedestrianRelativeVelocities): Pedestrian velocity x-components
                - Shape: (N,) where N = number of pedestrians
                - Units: meters/second
                - Source: pedestrian velocity estimation system
                - Constraints: velocities in robot-centric x-axis (forward/backward)
                - Example: [1.2, -0.8] (first ped moving forward, second backward)

        Returns:
            PedestrianFeatureMap: 2D feature map representing pedestrian x-velocities.
                - Shape: (feature_map_size, feature_map_size) or (feature_map_size²,) if flattened
                - Dtype: np.float32
                - Units: velocity values per grid cell
                - Range: [ped_min_speed_x, ped_max_speed_x]
                - Grid: robot-centric with robot at center
                - Example: 64x64 grid with pedestrian x-velocity values at corresponding locations
        """
        # Initialize feature map with zeros
        feature_map = np.zeros(
            (self._feature_map_size, self._feature_map_size), dtype=np.float32
        )

        # Process pedestrian data if available
        if len(pedestrian_relative_locations) > 0 and len(pedestrian_vel_x) > 0:
            for location, vel_x in zip(pedestrian_relative_locations, pedestrian_vel_x):
                # Convert location to grid coordinates
                grid_x, grid_y = self._get_map_index(location)

                # Check bounds and set velocity value
                if (
                    0 <= grid_x < self._feature_map_size
                    and 0 <= grid_y < self._feature_map_size
                ):
                    feature_map[grid_y, grid_x] = vel_x

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianVelYSpace(BaseFeatureMapSpace):
    """Pedestrian y-velocity feature map observation space.

    Encodes the y-component of pedestrian velocities in a spatial 2D grid centered on the robot.
    Each cell represents the y-velocity of a pedestrian at that location, enabling learning of
    lateral flow and social group movement.

    Technical Specifications:
    - Pedestrian Y-Velocity: Left/right velocity in robot-centric frame
    - Feature Map Size: Configurable grid resolution
    - ROI: Region of interest in meters
    - Flattening: Option to output flattened or 2D feature maps

    Output Format: 2D or 1D numpy array with y-velocity values per grid cell.

    Applications: Social group detection, lateral flow analysis, and crowd navigation.
    """

    name = "PEDESTRIAN_VEL_Y"
    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
        "pedestrian_vel_y": PedestrianRelativeVelocities,
    }

    def __init__(
        self,
        ped_min_speed_y: float,
        ped_max_speed_y: float,
        feature_map_size: int = 80,
        roi_in_m: float = 20.0,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._ped_min_speed_y = ped_min_speed_y
        self._ped_max_speed_y = ped_max_speed_y
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def get_gym_space(self) -> spaces.Space:
        """Returns the Gym observation space for the pedestrian y-velocity feature map."""
        return spaces.Box(
            low=self._ped_min_speed_y,
            high=self._ped_max_speed_y,
            shape=self._get_feature_map_shape(),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        pedestrian_vel_y: PedestrianRelativeVelocities,
        *args,
        **kwargs,
    ) -> PedestrianFeatureMap:
        """Encodes the pedestrian y-velocity feature map observation.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian positions relative to robot
                - Shape: (N, 2) where N = number of pedestrians
                - Units: meters
                - Source: pedestrian detection/tracking system
                - Constraints: positions relative to robot base frame
                - Example: [[2.5, 1.0], [-1.5, 3.0]] (2 pedestrians relative to robot)
            pedestrian_vel_y (PedestrianRelativeVelocities): Y-component of pedestrian velocities
                - Shape: (N,) where N = number of pedestrians
                - Units: meters/second
                - Source: pedestrian velocity estimation system
                - Constraints: velocities in robot-centric y-axis (left/right)
                - Example: [0.5, -1.2] (first ped moving left, second moving right)

        Returns:
            PedestrianFeatureMap: 2D feature map representing pedestrian y-velocities.
                - Shape: (feature_map_size, feature_map_size) or (feature_map_size²,) if flattened
                - Dtype: np.float32
                - Units: velocity values per grid cell
                - Range: [ped_min_speed_y, ped_max_speed_y]
                - Grid: robot-centric with robot at center
                - Example: 64x64 grid with pedestrian y-velocity values at corresponding locations
        """
        # Initialize feature map with zeros
        feature_map = np.zeros(
            (self._feature_map_size, self._feature_map_size), dtype=np.float32
        )

        # Process pedestrian data if available
        if len(pedestrian_relative_locations) > 0 and len(pedestrian_vel_y) > 0:
            for location, vel_y in zip(pedestrian_relative_locations, pedestrian_vel_y):
                # Convert location to grid coordinates
                grid_x, grid_y = self._get_map_index(location)

                # Check bounds and set velocity value
                if (
                    0 <= grid_x < self._feature_map_size
                    and 0 <= grid_y < self._feature_map_size
                ):
                    feature_map[grid_y, grid_x] = vel_y

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class StackedLaserMapSpace(BaseFeatureMapSpace):
    """Stacked laser scan feature map observation space for temporal environment representation.

    Maintains a queue of consecutive laser scans and transforms them into a 2D feature map,
    providing temporal and spatial information about the robot's surroundings for robust
    perception and dynamic environment modeling.

    Technical Specifications:
    - Laser Stack Size: Number of consecutive scans to stack
    - Feature Map Size: Configurable grid resolution
    - ROI: Region of interest in meters
    - Flattening: Option to output flattened or 2D feature maps

    Output Format: 2D or 1D numpy array with aggregated laser features per grid cell.

    Applications: Temporal perception, dynamic obstacle tracking, and SLAM.
    """

    name = "STACKED_LASER_MAP"
    requires = {"front_laser": LidarRanges, "is_terminal": IsTerminal}

    def __init__(
        self,
        laser_stack_size: int = 10,
        feature_map_size: int = 80,
        roi_in_m: float = 20.0,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._laser_queue = deque()
        self._laser_stack_size = laser_stack_size
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the gym space for the feature map.
        """
        shape = (
            (self._feature_map_size * self._feature_map_size,)
            if self._flatten
            else (1, self._feature_map_size, self._feature_map_size)
        )
        return spaces.Box(
            low=0,
            high=self._roi_in_m,
            shape=shape,
            dtype=np.float32,
        )

    def _reset_laser_stack(self, laser_scan: np.ndarray):
        """
        Resets the laser stack with zeros.
        """
        self._laser_queue = deque([np.zeros_like(laser_scan)] * self._laser_stack_size)

    def _build_laser_map(self, laser_queue: deque) -> np.ndarray:
        """Builds a laser map from a queue of laser scans."""
        # The reference implementation expects a fixed structure.
        # We assume laser_stack_size=10, feature_map_size=80, and laser scan length = 720
        # to match the logic.
        SEGMENT_SIZE = 9

        temp = np.array(laser_queue, dtype=np.float32).flatten()

        # Single reshape for all operations
        reshaped = temp.reshape(
            self._laser_stack_size, self._feature_map_size, SEGMENT_SIZE
        )

        # Pre-allocate output with matching dtype
        scan_avg = np.zeros(
            (2 * self._laser_stack_size, self._feature_map_size), dtype=np.float32
        )

        # Vectorized calculations using axis reduction
        scan_avg[::2] = reshaped.min(axis=2)  # Even rows: minima
        scan_avg[1::2] = reshaped.mean(axis=2)  # Odd rows: averages

        # Final transformations
        scan_avg = scan_avg.reshape(2 * self._laser_stack_size * self._feature_map_size)
        scan_avg_map = np.tile(scan_avg, (4, 1)).reshape(
            1, self._feature_map_size, self._feature_map_size
        )

        return scan_avg_map

    def _process_laser_scan(
        self, laser_scan: LidarRanges, done: IsTerminal
    ) -> np.ndarray:
        """Process laser scan data and build a stacked laser map."""
        if not isinstance(laser_scan, np.ndarray) or laser_scan.size == 0:
            laser_scan = np.zeros(
                (720,),
                dtype=np.float32,
            )

        if len(self._laser_queue) == 0 or done:
            self._reset_laser_stack(laser_scan)

        self._laser_queue.pop()
        self._laser_queue.appendleft(laser_scan)

        laser_map = self._build_laser_map(self._laser_queue)

        return laser_map

    def encode_observation(
        self, front_laser: LidarRanges, is_terminal: IsTerminal, *args, **kwargs
    ) -> LaserFeatureMap:
        """Encodes the stacked laser map observation.

        Args:
            front_laser (LidarRanges): Front-facing laser scanner for environment mapping
                - Shape: (n,) where n = number of laser beams
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, max_range], NaN replaced with max_range
                - Example: [0.5, 1.2, 3.4, ..., 2.1] (array of distance measurements)
            done (Done): Flag indicating if the episode has ended
                - Shape: scalar boolean
                - Units: boolean flag
                - Source: episode termination system
                - Constraints: True when episode terminates, False otherwise
                - Example: False (episode continues) or True (episode ended)

        Returns:
            LaserFeatureMap: Stacked laser scan feature map for temporal analysis.
                - Shape: (feature_map_size, feature_map_size) or (feature_map_size²,) if flattened
                - Dtype: np.float32
                - Units: aggregated laser distance features per grid cell
                - Range: [0.0, 1.0] normalized values
                - Grid: robot-centric 2D representation of environment structure
                - Example: 80x80 grid with temporal laser scan information
        """
        processed_map = self._process_laser_scan(front_laser, is_terminal)
        return processed_map.flatten() if self._flatten else processed_map


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianLocationSpace(BaseFeatureMapSpace):
    """Pedestrian location feature map observation space.

    Encodes the presence of pedestrians in a spatial 2D grid centered on the robot.
    Each cell indicates whether a pedestrian is present at that location, enabling spatial
    reasoning and crowd density estimation.

    Technical Specifications:
    - Pedestrian Locations: Binary presence indicator per grid cell
    - Feature Map Size: Configurable grid resolution
    - ROI: Region of interest in meters
    - Flattening: Option to output flattened or 2D feature maps

    Output Format: 2D or 1D numpy array with binary presence indicators.

    Applications: Crowd density mapping, social navigation, and pedestrian avoidance.
    """

    name = "PEDESTRIAN_LOCATION"
    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
    }

    def get_gym_space(self) -> spaces.Space:
        """Returns the Gym observation space for the pedestrian location map."""
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._get_feature_map_shape(),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        *args,
        **kwargs,
    ) -> PedestrianFeatureMap:
        """Encodes the pedestrian location observation.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Positions of pedestrians in robot frame
                - Shape: (n, 2) where n = number of pedestrians
                - Units: meters
                - Source: pedestrian detection system
                - Constraints: positions relative to robot base frame
                - Example: [[1.5, 2.0], [-0.5, 3.2]] (x, y coordinates)

        Returns:
            PedestrianFeatureMap: Binary grid map indicating pedestrian presence.
                - Shape: (feature_map_size, feature_map_size) or (feature_map_size²,) if flattened
                - Dtype: np.float32
                - Units: binary presence indicator
                - Range: [0.0, 1.0]
                - Grid: Binary map with 1.0 for pedestrian locations, 0.0 elsewhere
                - Example: 32x32 grid with sparse 1.0 values
        """
        # Create feature map
        feature_map = self._create_feature_map()

        # Process each pedestrian location
        for location in pedestrian_relative_locations:
            grid_x, grid_y = self._get_map_index(location)

            # Check bounds and mark pedestrian presence
            if (
                0 <= grid_x < self._feature_map_size
                and 0 <= grid_y < self._feature_map_size
            ):
                feature_map[grid_y, grid_x] = 1.0

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianSocialStateSpace(BaseFeatureMapSpace):
    """Pedestrian social state feature map observation space.

    Encodes social state information for pedestrians in a spatial 2D grid centered on the robot.
    Each cell contains an integer code representing the social state of a pedestrian at that location,
    supporting social behavior modeling and group interaction analysis.

    Technical Specifications:
    - Pedestrian Social States: Integer state codes per grid cell
    - Feature Map Size: Configurable grid resolution
    - ROI: Region of interest in meters
    - Flattening: Option to output flattened or 2D feature maps

    Output Format: 2D or 1D numpy array with social state codes per grid cell.

    Applications: Social behavior recognition, group interaction modeling, and crowd simulation.
    """

    name = "PEDESTRIAN_SOCIAL_STATE"
    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
        "pedestrian_social_states": PedestrianSocialStates,
    }

    def __init__(
        self,
        ped_social_state_num: int,
        feature_map_size: int = 80,
        roi_in_m: float = 20.0,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._social_state_num = ped_social_state_num
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def get_gym_space(self) -> spaces.Space:
        """Returns the Gym observation space for the pedestrian social state map."""
        return spaces.Box(
            low=0,
            high=self._social_state_num,
            shape=self._get_feature_map_shape(),
            dtype=np.int32,
        )

    def encode_observation(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        pedestrian_social_states: PedestrianSocialStates,
        *args,
        **kwargs,
    ) -> SocialStateFeatureMap:
        """Encodes the pedestrian social state observation.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Positions of pedestrians in robot frame
                - Shape: (n, 2) where n = number of pedestrians
                - Units: meters
                - Source: pedestrian detection system
                - Constraints: positions relative to robot base frame
                - Example: [[1.5, 2.0], [-0.5, 3.2]] (x, y coordinates)
            pedestrian_social_states (PedestrianSocialStates): Social state classifications for pedestrians
                - Shape: (n,) where n = number of pedestrians
                - Units: integer state codes
                - Source: social behavior recognition system
                - Constraints: state ∈ [0, max_social_states]
                - Example: [0, 2, 1, 3] (state codes for different pedestrians)

        Returns:
            SocialStateFeatureMap: Grid map encoding pedestrian social states.
                - Shape: (feature_map_size, feature_map_size) or (feature_map_size²,) if flattened
                - Dtype: np.int32
                - Units: social state codes
                - Range: [0, ped_social_state_num]
                - Grid: Integer map with state codes for pedestrians, 0 elsewhere
                - Example: 32x32 grid with sparse integer values
        """
        # Create feature map with background value
        feature_map = np.zeros(
            (self._feature_map_size, self._feature_map_size), dtype=np.int32
        )

        # Process each pedestrian's location and social state
        for location, social_state in zip(
            pedestrian_relative_locations, pedestrian_social_states
        ):
            grid_x, grid_y = self._get_map_index(location)

            # Check bounds and set social state value
            if (
                0 <= grid_x < self._feature_map_size
                and 0 <= grid_y < self._feature_map_size
            ):
                feature_map[grid_y, grid_x] = social_state

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianTypeSpace(BaseFeatureMapSpace):
    """Pedestrian type feature map observation space.

    Encodes type information for pedestrians in a spatial 2D grid centered on the robot.
    Each cell contains an integer code representing the type of pedestrian at that location,
    supporting semantic crowd analysis and heterogeneous group modeling.

    Technical Specifications:
    - Pedestrian Types: Integer type codes per grid cell
    - Feature Map Size: Configurable grid resolution
    - ROI: Region of interest in meters
    - Flattening: Option to output flattened or 2D feature maps

    Output Format: 2D or 1D numpy array with type codes per grid cell.

    Applications: Semantic crowd analysis, heterogeneous group modeling, and social navigation.
    """

    name = "PEDESTRIAN_TYPE"
    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
        "pedestrian_types": PedestrianTypeArray,
    }

    def __init__(
        self,
        ped_num_types: int,
        feature_map_size: int = 80,
        roi_in_m: float = 20.0,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._num_ped_types = ped_num_types
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def get_gym_space(self) -> spaces.Space:
        """Returns the Gym observation space for the pedestrian type map."""
        return spaces.Box(
            low=-1,
            high=self._num_ped_types,
            shape=self._get_feature_map_shape(),
            dtype=np.int32,
        )

    def encode_observation(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        pedestrian_types: PedestrianTypeArray,
        *args,
        **kwargs,
    ) -> PedestrianTypeFeatureMap:
        """Encodes the pedestrian type observation.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Positions of pedestrians in robot frame
                - Shape: (n, 2) where n = number of pedestrians
                - Units: meters
                - Source: pedestrian detection system
                - Constraints: positions relative to robot base frame
                - Example: [[1.5, 2.0], [-0.5, 3.2]] (x, y coordinates)
            pedestrian_types (PedestrianTypeArray): Type classifications for pedestrians
                - Shape: (n,) where n = number of pedestrians
                - Units: integer type codes
                - Source: pedestrian classification system
                - Constraints: type ∈ [0, max_pedestrian_types]
                - Example: [0, 1, 2, 0] (type codes for different pedestrians)

        Returns:
            PedestrianTypeFeatureMap: Grid map encoding pedestrian types.
                - Shape: (feature_map_size, feature_map_size) or (feature_map_size²,) if flattened
                - Dtype: np.int32
                - Units: pedestrian type codes
                - Range: [-1, ped_num_types]
                - Grid: Integer map with type codes for pedestrians, -1 for empty cells
                - Example: 32x32 grid with sparse integer values
        """
        # Create feature map with background value (-1 for empty cells)
        feature_map = np.full(
            (self._feature_map_size, self._feature_map_size), -1, dtype=np.int32
        )

        # Process each pedestrian's location and type
        for location, ped_type in zip(pedestrian_relative_locations, pedestrian_types):
            grid_x, grid_y = self._get_map_index(location)

            # Check bounds and set type value
            if (
                0 <= grid_x < self._feature_map_size
                and 0 <= grid_y < self._feature_map_size
            ):
                feature_map[grid_y, grid_x] = ped_type

        return feature_map.flatten() if self._flatten else feature_map
