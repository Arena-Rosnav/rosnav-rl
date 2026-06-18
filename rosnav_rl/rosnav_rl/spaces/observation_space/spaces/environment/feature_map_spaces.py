"""Environment Feature Map Spaces - Schema-Based Architecture

Feature map spaces for pedestrian and environmental information with proper
type annotations and schema-based requirements.
"""

from abc import abstractmethod
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

        # Pre-compute grid constants for vectorized coordinate conversion
        self._grid_resolution = roi_in_m / feature_map_size
        self._inv_grid_resolution = feature_map_size / roi_in_m 
        self._grid_center = feature_map_size // 2

        # Pre-allocate reusable feature map buffer
        self._feature_map_buffer = np.full(
            (feature_map_size, feature_map_size),
            self.background_value,
            dtype=np.float32,
        )

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
        grid_x = int(x * self._inv_grid_resolution + self._grid_center)
        grid_y = int(y * self._inv_grid_resolution + self._grid_center)
        return grid_x, grid_y

    def _get_map_indices_vectorized(
        self, positions: np.ndarray
    ) -> tuple:
        """Convert multiple real-world coordinates to grid indices (vectorized).

        Args:
            positions: (N, 2) array of [x, y] coordinates in meters.

        Returns:
            Tuple of (grid_x, grid_y, valid_mask) where valid_mask filters in-bounds indices.
        """
        grid_coords = (positions * self._inv_grid_resolution + self._grid_center).astype(np.intp)
        grid_x = grid_coords[:, 0]
        grid_y = grid_coords[:, 1]
        valid = (
            (grid_x >= 0) & (grid_x < self._feature_map_size)
            & (grid_y >= 0) & (grid_y < self._feature_map_size)
        )
        return grid_x, grid_y, valid

    def _create_feature_map(self) -> np.ndarray:
        """Create an empty feature map with background values.

        Always returns a 2D array for consistent indexing in encode_observation.
        Flattening is handled at the end of encode_observation, not here.
        Uses the pre-allocated buffer, resetting it to background value.
        """
        self._feature_map_buffer[:] = self.background_value
        return self._feature_map_buffer

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

    name = "PedestrianVelXSpace"
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

        # Process pedestrian data if available — vectorized
        if len(pedestrian_relative_locations) > 0 and len(pedestrian_vel_x) > 0:
            grid_x, grid_y, valid = self._get_map_indices_vectorized(
                pedestrian_relative_locations
            )
            feature_map[grid_y[valid], grid_x[valid]] = pedestrian_vel_x[valid]

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

    name = "PedestrianVelYSpace"
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

        # Process pedestrian data if available — vectorized
        if len(pedestrian_relative_locations) > 0 and len(pedestrian_vel_y) > 0:
            grid_x, grid_y, valid = self._get_map_indices_vectorized(
                pedestrian_relative_locations
            )
            feature_map[grid_y[valid], grid_x[valid]] = pedestrian_vel_y[valid]

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class StackedLaserMapSpace(BaseObservationSpace):
    """Stacked laser scan feature map for temporal environment representation.

    Maintains a ring buffer of consecutive laser scans and builds a 2D feature
    map by computing per-column min/mean statistics, providing both spatial and
    temporal information about the surroundings.

    The output is a (1, feature_map_size, feature_map_size) tensor normalized
    to [-1, 1] via MaxAbsScaler([0, laser_max_range]).

    This class inherits directly from BaseObservationSpace — not
    BaseFeatureMapSpace — because it uses a ring buffer rather than grid
    projection, so the grid-projection machinery in BaseFeatureMapSpace is
    irrelevant and would waste memory and add misleading API surface.
    """

    name = "StackedLaserMapSpace"
    requires = {"front_laser": LidarRanges, "is_terminal": IsTerminal}

    def __init__(
        self,
        laser_stack_size: int = 10,
        feature_map_size: int = 80,
        laser_max_range: float = 30.0,
        laser_num_beams: int = 720,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._laser_stack_size = laser_stack_size
        self._feature_map_size = feature_map_size
        self._flatten = flatten

        # Pre-compute derived constants (avoids per-step division)
        beams_per_col = max(1, laser_num_beams // feature_map_size)
        self._beams_per_col = beams_per_col
        self._usable_beams = feature_map_size * beams_per_col
        # MaxAbsScaler: [0, laser_max_range] → [-1, 1],  scale = 2 / max_range
        self._norm_scale = np.float32(2.0 / laser_max_range)

        # Eagerly allocate ring buffer — no nullable sentinel needed
        self._ring_buffer = np.zeros(
            (laser_stack_size, laser_num_beams), dtype=np.float32
        )
        self._ring_idx: int = 0

        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        """Reset ring buffer for a new episode."""
        self._ring_buffer[:] = 0.0
        self._ring_idx = 0

    def get_gym_space(self) -> spaces.Space:
        """Returns the gym space: float32 in [-1, 1]."""
        shape = (
            (self._feature_map_size * self._feature_map_size,)
            if self._flatten
            else (1, self._feature_map_size, self._feature_map_size)
        )
        return spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

    def encode_observation(
        self, front_laser: LidarRanges, is_terminal: IsTerminal, *args, **kwargs
    ) -> LaserFeatureMap:
        """Build the stacked laser map and normalize to [-1, 1].

        Steps (equivalent to the reference MaxAbsScaler formula):
          1. Sanitize input scan.
          2. Reset ring buffer on episode end.
          3. Insert scan into ring buffer (overwrites oldest slot).
          4. Read ring in insertion order; reshape to (stack, cols, beams_per_col).
          5. Compute per-column min/mean → (2*stack, cols) summary.
          6. Tile summary to fill feature_map_size × feature_map_size.
          7. Apply MaxAbsScaler: x * (2 / laser_max_range) - 1.

        Args:
            front_laser: Laser ranges, shape (n_beams,), units meters.
            is_terminal: True when the episode just ended.

        Returns:
            np.ndarray: shape (1, H, W) or (H*W,) if flatten=True, dtype float32,
                range [-1, 1].
        """
        # 1. Sanitize
        scan = front_laser
        if not isinstance(scan, np.ndarray) or scan.size == 0:
            scan = np.zeros(self._ring_buffer.shape[1], dtype=np.float32)
        elif scan.dtype != np.float32:
            scan = scan.astype(np.float32)

        # 2. Reset buffer if beam count changed or episode ended
        if is_terminal or scan.shape[0] != self._ring_buffer.shape[1]:
            n = scan.shape[0]
            if n != self._ring_buffer.shape[1]:
                # Beam count changed — reallocate and recompute constants
                self._ring_buffer = np.zeros(
                    (self._laser_stack_size, n), dtype=np.float32
                )
                self._beams_per_col = max(1, n // self._feature_map_size)
                self._usable_beams = self._feature_map_size * self._beams_per_col
            else:
                self._ring_buffer[:] = 0.0
            self._ring_idx = 0

        # 3. Insert (ring buffer: oldest entry overwritten)
        self._ring_buffer[self._ring_idx] = scan
        self._ring_idx = (self._ring_idx + 1) % self._laser_stack_size

        # 4. Read in insertion order (oldest → newest)
        ordered = np.roll(self._ring_buffer, -self._ring_idx, axis=0)
        reshaped = ordered[:, :self._usable_beams].reshape(
            self._laser_stack_size, self._feature_map_size, self._beams_per_col
        )

        # 5. Per-column min (even rows) and mean (odd rows)
        summary = np.empty(
            (self._laser_stack_size * 2, self._feature_map_size), dtype=np.float32
        )
        summary[::2] = reshaped.min(axis=2)
        summary[1::2] = reshaped.mean(axis=2)

        # 6. Tile flat summary to fill H × W
        flat = summary.ravel()           # 1600 for defaults
        target = self._feature_map_size * self._feature_map_size  # 6400
        reps = -(-target // len(flat))   # ceil division
        tiled = np.tile(flat, reps)[:target]

        # 7. MaxAbsScaler: [0, laser_max_range] → [-1, 1]
        result = (tiled * self._norm_scale - 1.0).reshape(
            1, self._feature_map_size, self._feature_map_size
        )
        return result.flatten() if self._flatten else result


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

    name = "PedestrianLocationSpace"
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

        # Process each pedestrian location — vectorized
        if len(pedestrian_relative_locations) > 0:
            grid_x, grid_y, valid = self._get_map_indices_vectorized(
                pedestrian_relative_locations
            )
            feature_map[grid_y[valid], grid_x[valid]] = 1.0

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

    name = "PedestrianSocialStateSpace"
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

        # Process each pedestrian's location and social state — vectorized
        if (
            len(pedestrian_relative_locations) > 0
            and len(pedestrian_social_states) > 0
        ):
            grid_x, grid_y, valid = self._get_map_indices_vectorized(
                pedestrian_relative_locations
            )
            feature_map[grid_y[valid], grid_x[valid]] = pedestrian_social_states[valid]

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

    name = "PedestrianTypeSpace"
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

        # Process each pedestrian's location and type — vectorized
        if len(pedestrian_relative_locations) > 0 and len(pedestrian_types) > 0:
            grid_x, grid_y, valid = self._get_map_indices_vectorized(
                pedestrian_relative_locations
            )
            feature_map[grid_y[valid], grid_x[valid]] = pedestrian_types[valid]

        return feature_map.flatten() if self._flatten else feature_map


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class LaserCartesianMapSpace(BaseFeatureMapSpace):
    """Robot-centric Cartesian laser obstacle map for spatially coherent CNN encoding.

    Projects each laser beam endpoint from polar (r, θ) to Cartesian (r·cosθ, r·sinθ)
    in the robot's coordinate frame, placing it into the same 2D grid used by
    PedestrianVelXSpace and PedestrianVelYSpace. All three CNN channels then share
    the same physical coordinate frame, enabling the ConvEncoder to learn cross-modal
    relationships between static obstacles and pedestrian motion patterns.

    Cell value encodes obstacle proximity: 1 − r/max_range ∈ [0, 1].
    Beams at max_range (no return) are excluded (left as 0 = free).

    Why this outperforms StackedLaserMapSpace for DreamerV3:
    - True 2D Cartesian structure: CNN 2D locality is physically meaningful in both axes.
    - Same coordinate frame as pedestrian maps: cross-channel spatial reasoning possible.
    - Sparse output (mostly zeros): MSE decoder loss drops from ~141 to ~3–8.
    - No ring buffer: RSSM's recurrent state (h_t) provides temporal context natively.

    Output: (feature_map_size, feature_map_size) float32 ∈ [0, 1].
    """

    name = "LaserCartesianMapSpace"
    requires = {"front_laser": LidarRanges}

    def __init__(
        self,
        laser_num_beams: int = 720,
        laser_max_range: float = 30.0,
        laser_angle_min: float = -np.pi,
        feature_map_size: int = 80,
        roi_in_m: float = 20.0,
        flatten: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the LaserCartesianMapSpace.

        Args:
            laser_num_beams: Number of beams in the laser scan (default 720 for Jackal).
            laser_max_range: Maximum valid range in meters; beams at or above this are excluded.
            laser_angle_min: Angle of beam 0 in radians; beams are spaced evenly over 2π.
            feature_map_size: Grid resolution — the grid is (feature_map_size × feature_map_size).
            roi_in_m: Total grid extent in meters (grid covers ±roi_in_m/2 around robot).
            flatten: If True, return a flat 1D array; otherwise return a 2D array.
        """
        self._laser_max_range = laser_max_range
        self._hit_threshold = np.float32(laser_max_range * 0.99)

        # Pre-compute per-beam trig values once — no per-step memory allocation
        angles = np.linspace(
            laser_angle_min,
            laser_angle_min + 2.0 * np.pi,
            laser_num_beams,
            endpoint=False,
            dtype=np.float32,
        )
        self._cos_a = np.cos(angles)
        self._sin_a = np.sin(angles)

        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def get_gym_space(self) -> spaces.Space:
        """Returns a Box in [0, 1] with the feature map shape."""
        return spaces.Box(
            low=0.0, high=1.0, shape=self._get_feature_map_shape(), dtype=np.float32
        )

    def encode_observation(
        self, front_laser: LidarRanges, *args, **kwargs
    ) -> LaserFeatureMap:
        """Project laser beam endpoints to a robot-centric Cartesian grid.

        Args:
            front_laser: Laser ranges array, shape (n_beams,), units meters.
                Invalid/no-return beams should be at or above laser_max_range.

        Returns:
            np.ndarray: shape (H, W) or (H*W,) if flatten=True, dtype float32, range [0, 1].
                High values indicate obstacles close to the robot; zero = free or out-of-range.
        """
        scan = np.asarray(front_laser, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self._laser_max_range, posinf=self._laser_max_range)
        scan = np.clip(scan, 0.0, self._laser_max_range)

        # Only beams that actually returned a hit (exclude max-range no-return beams)
        hit = scan < self._hit_threshold

        # Polar → Cartesian in robot frame: x = forward, y = left
        x = self._cos_a[hit] * scan[hit]
        y = self._sin_a[hit] * scan[hit]
        positions = np.stack([x, y], axis=1)

        grid_x, grid_y, valid = self._get_map_indices_vectorized(positions)
        # Proximity encoding: near obstacle → close to 1.0, far → close to 0.0
        values = 1.0 - scan[hit][valid] / self._laser_max_range

        feature_map = self._create_feature_map()  # zeros from pre-allocated buffer
        feature_map[grid_y[valid], grid_x[valid]] = values
        return feature_map.flatten() if self._flatten else feature_map
