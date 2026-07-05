"""
Environmental context and obstacle information with reliable processing.
"""

from typing import Dict, List, Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.observation_types import (
    LidarRanges,
    EnvironmentContextVector,
    SpatialAwarenessVector,
    ObstacleProximityVector,
)
from ...observation_space_factory import SpaceFactory
from ...space_categories import SpaceCategory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class EnvironmentContextSpace(BaseObservationSpace):
    """Advanced environment context observation space with obstacle density and free space analysis.

    This space provides robust environmental context by analyzing laser scan data to estimate obstacle density,
    free space, and overall environmental complexity. Designed for navigation agents requiring situational awareness
    and adaptive behavior in dynamic or cluttered environments.

    Technical Specifications:
    - Obstacle Density Estimation: Sector-based analysis for local and global density metrics
    - Free Space Analysis: Minimum and maximum clearance computation for navigation safety
    - Environment Classification: Simple metrics for open, cluttered, or constrained spaces

    Configuration:
    - max_range: Maximum valid laser range (meters)
    - density_sectors: Number of angular sectors for density computation
    - obstacle_threshold: Distance threshold for classifying obstacles (meters)

    Output Format: (density_sectors + 3)-dimensional normalized vector with sector densities, total density,
    minimum clearance, and maximum clearance for comprehensive environmental context.

    Applications: Adaptive navigation, environment classification, obstacle avoidance, and exploration.
    """

    name = "EnvironmentContextSpace"
    requires = {
        "front_laser": LidarRanges,  # Front-facing laser for environment context analysis
    }

    def __init__(
        self,
        max_range: float = 10.0,
        density_sectors: int = 8,
        obstacle_threshold: float = 0.5,
        *args,
        **kwargs
    ):
        """Initialize environment context space.

        Args:
            max_range: Maximum laser range to consider
            density_sectors: Number of sectors for density analysis
            obstacle_threshold: Distance threshold for obstacle detection
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_range = max_range
        self.density_sectors = density_sectors
        self.obstacle_threshold = obstacle_threshold

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for environment context."""
        # [obstacle_density_per_sector + overall_metrics]
        # overall_metrics: [total_density, min_clearance, max_clearance]
        dims = self.density_sectors + 3

        return spaces.Box(
            low=np.array([0.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def _analyze_laser_data(self, laser_data: np.ndarray) -> Dict[str, Any]:
        """Analyze laser data for environment features."""
        # Filter valid readings
        valid_mask = (laser_data > 0.0) & (laser_data < self.max_range)
        valid_readings = laser_data[valid_mask]

        if len(valid_readings) == 0:
            return {
                "sector_densities": np.zeros(self.density_sectors),
                "total_density": 0.0,
                "min_clearance": 1.0,
                "max_clearance": 1.0,
            }

        # Sector-wise density analysis
        readings_per_sector = len(laser_data) // self.density_sectors
        sector_densities = []

        for i in range(self.density_sectors):
            start_idx = i * readings_per_sector
            end_idx = (
                (i + 1) * readings_per_sector
                if i < self.density_sectors - 1
                else len(laser_data)
            )

            sector_data = laser_data[start_idx:end_idx]
            sector_valid = (sector_data > 0.0) & (sector_data < self.max_range)

            if np.any(sector_valid):
                # Density based on how many obstacles are close
                close_obstacles = np.sum(
                    sector_data[sector_valid] < self.obstacle_threshold
                )
                total_valid = np.sum(sector_valid)
                density = close_obstacles / max(total_valid, 1)
            else:
                density = 0.0

            sector_densities.append(density)

        # Overall metrics
        total_density = np.mean(sector_densities)
        min_clearance = np.min(valid_readings) / self.max_range
        max_clearance = np.max(valid_readings) / self.max_range

        return {
            "sector_densities": np.array(sector_densities),
            "total_density": total_density,
            "min_clearance": min_clearance,
            "max_clearance": max_clearance,
        }

    def encode_observation(
        self, front_laser: LidarRanges, *args, **kwargs
    ) -> ObstacleProximityVector:
        """Encode comprehensive environment context with obstacle density and clearance metrics.

        Processes preprocessed front-facing laser scan data to extract sector-wise obstacle densities,
        total density, and clearance metrics for robust environmental context representation.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges for environment analysis
                - Shape: (n,) where n = number of laser beams
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, max_range], NaN replaced with max_range
                - Example: [0.5, 1.2, 3.4, ..., 2.1] (used for obstacle density and free space analysis)

        Returns:
            ObstacleProximityVector: Environment context vector with sector densities and clearance metrics.
                - Shape: (density_sectors + 3,)
                - Dtype: np.float32
                - Elements: [sector_densities..., total_density, min_clearance, max_clearance]
                - Units: [normalized, normalized, normalized, normalized]
                - Range: all elements ∈ [0,1]
                - Example: [0.1, 0.2, 0.0, 0.3, 0.15, 0.05, 0.9, 0.7, 0.95, 0.2, 0.8, 0.6, 0.1] (sector densities + metrics)
        """
        # Analyze laser data
        analysis = self._analyze_laser_data(front_laser)

        # Combine all features
        result = list(analysis["sector_densities"])
        result.extend(
            [
                analysis["total_density"],
                analysis["min_clearance"],
                analysis["max_clearance"],
            ]
        )

        return np.array(result, dtype=np.float32)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class SpatialAwarenessSpace(BaseObservationSpace):
    """Advanced spatial awareness observation space with directional clearance and corridor detection.

    This space provides detailed spatial awareness by analyzing laser scan data for directional clearances,
    corridor detection, and spatial constraint assessment. Enables navigation agents to reason about free space,
    passageways, and environmental constraints for safe and efficient movement.

    Technical Specifications:
    - Directional Clearance: Cardinal direction clearance (front, left, right, back)
    - Corridor Detection: Parallel wall and passageway identification with directionality
    - Spatial Constraint: Quantitative measure of environmental restriction

    Configuration:
    - max_range: Maximum valid laser range (meters)
    - safety_distance: Clearance threshold for safe navigation (meters)
    - corridor_width_threshold: Minimum width to consider as a corridor (meters)

    Output Format: 7-dimensional normalized vector with directional clearances, corridor detection,
    corridor direction, and spatial constraint metrics.

    Applications: Path planning, corridor following, spatial reasoning, and constraint-aware navigation.
    """

    name = "SpatialAwarenessSpace"
    requires = {
        "front_laser": LidarRanges,  # Front-facing laser for spatial awareness analysis
    }

    def __init__(
        self,
        max_range: float = 10.0,
        safety_distance: float = 0.8,
        corridor_width_threshold: float = 1.5,
        *args,
        **kwargs
    ):
        """Initialize spatial awareness space.

        Args:
            max_range: Maximum laser range to consider
            safety_distance: Distance for safety clearance calculation
            corridor_width_threshold: Minimum width to consider as corridor
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_range = max_range
        self.safety_distance = safety_distance
        self.corridor_width_threshold = corridor_width_threshold

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for spatial awareness."""
        # [front_clear, left_clear, right_clear, back_clear,
        #  corridor_detected, corridor_direction, spatial_constraint]
        return spaces.Box(
            low=np.array([0.0] * 7), high=np.array([1.0] * 7), dtype=np.float32
        )

    def _compute_directional_clearance(
        self, laser_data: np.ndarray
    ) -> Dict[str, float]:
        """Compute clearance in cardinal directions."""
        n_rays = len(laser_data)

        # Define angular sectors (assuming 360° scan)
        front_sector = self._get_sector_indices(n_rays, -30, 30)
        left_sector = self._get_sector_indices(n_rays, 60, 120)
        right_sector = self._get_sector_indices(n_rays, -120, -60)
        back_sector = self._get_sector_indices(n_rays, 150, 210)

        clearances = {}
        for direction, sector_indices in [
            ("front", front_sector),
            ("left", left_sector),
            ("right", right_sector),
            ("back", back_sector),
        ]:
            sector_data = laser_data[sector_indices]
            valid_data = sector_data[
                (sector_data > 0.0) & (sector_data < self.max_range)
            ]

            if len(valid_data) > 0:
                min_distance = np.min(valid_data)
                clearance = min(min_distance / self.safety_distance, 1.0)
            else:
                clearance = 1.0  # No obstacles detected = full clearance

            clearances[direction] = clearance

        return clearances

    def _get_sector_indices(
        self, n_rays: int, start_angle: float, end_angle: float
    ) -> np.ndarray:
        """Get laser ray indices for angular sector."""
        # Convert angles to indices (assuming 360° scan starting from front)
        start_idx = int((start_angle + 180) / 360 * n_rays) % n_rays
        end_idx = int((end_angle + 180) / 360 * n_rays) % n_rays

        if start_idx <= end_idx:
            return np.arange(start_idx, end_idx + 1)
        else:
            # Wrap around case
            return np.concatenate(
                [np.arange(start_idx, n_rays), np.arange(0, end_idx + 1)]
            )

    def _detect_corridor(self, laser_data: np.ndarray) -> tuple:
        """Detect corridor and its direction.

        Returns:
            (corridor_detected, corridor_direction)
        """
        n_rays = len(laser_data)

        # Check for corridor patterns (parallel walls)
        # Sample left and right sides
        left_indices = self._get_sector_indices(n_rays, 45, 135)
        right_indices = self._get_sector_indices(n_rays, -135, -45)

        left_data = laser_data[left_indices]
        right_data = laser_data[right_indices]

        left_valid = left_data[(left_data > 0.0) & (left_data < self.max_range)]
        right_valid = right_data[(right_data > 0.0) & (right_data < self.max_range)]

        if len(left_valid) > 0 and len(right_valid) > 0:
            left_distance = np.median(left_valid)
            right_distance = np.median(right_valid)

            corridor_width = left_distance + right_distance

            # Must be wider than threshold but not so wide it's just open space
            max_corridor_width = self.corridor_width_threshold * 4.0
            if self.corridor_width_threshold < corridor_width <= max_corridor_width:
                # Determine corridor direction based on which side is farther
                if left_distance > right_distance:
                    corridor_direction = 0.75  # Left-leaning corridor
                elif right_distance > left_distance:
                    corridor_direction = 0.25  # Right-leaning corridor
                else:
                    corridor_direction = 0.5  # Centered corridor

                return (1.0, corridor_direction)

        return (0.0, 0.5)  # No corridor detected

    def _compute_spatial_constraint(self, clearances: Dict[str, float]) -> float:
        """Compute overall spatial constraint level."""
        # Constraint based on how restricted the space is
        min_clearance = min(clearances.values())
        avg_clearance = np.mean(list(clearances.values()))

        # Higher constraint when clearances are low
        constraint = 1.0 - (min_clearance * 0.6 + avg_clearance * 0.4)
        return np.clip(constraint, 0.0, 1.0)

    def encode_observation(
        self, front_laser: LidarRanges, *args, **kwargs
    ) -> SpatialAwarenessVector:
        """Encode comprehensive spatial awareness with directional clearance and corridor detection.

        Processes front-facing laser scan data to extract cardinal direction clearances, detect corridors,
        and assess spatial constraints for robust spatial reasoning and navigation.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges for spatial awareness
                - Shape: (n,) where n = number of laser beams
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, max_range], NaN replaced with max_range
                - Example: [0.5, 1.2, 3.4, ..., 2.1] (used for directional clearances and corridor detection)

        Returns:
            SpatialAwarenessVector: Spatial awareness vector with directional clearances, corridor detection, and constraint metrics.
                - Shape: (7,) - [front, left, right, back, corridor_detected, corridor_direction, spatial_constraint]
                - Dtype: np.float32
                - Elements: [clearances..., corridor_detected, corridor_direction, spatial_constraint]
                - Units: [normalized, normalized, normalized, normalized, boolean, normalized, normalized]
                - Range: all elements ∈ [0,1]
                - Example: [0.8, 0.6, 0.7, 0.9, 1.0, 0.5, 0.2] (clearances + corridor + constraint)
        """
        # Compute directional clearances
        clearances = self._compute_directional_clearance(front_laser)

        # Detect corridor
        corridor_detected, corridor_direction = self._detect_corridor(front_laser)

        # Compute spatial constraint
        spatial_constraint = self._compute_spatial_constraint(clearances)

        result = [
            clearances["front"],
            clearances["left"],
            clearances["right"],
            clearances["back"],
            corridor_detected,
            corridor_direction,
            spatial_constraint,
        ]

        return np.array(result, dtype=np.float32)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class ObstacleProximitySpace(BaseObservationSpace):
    """Advanced obstacle proximity observation space with multi-zone detection and risk assessment.

    This space provides robust obstacle proximity analysis by dividing the environment into configurable
    distance zones, tracking the closest obstacle, and assessing collision risk. Enables navigation agents
    to react to nearby obstacles and maintain safe distances.

    Technical Specifications:
    - Multi-Zone Proximity: Configurable distance thresholds for close, medium, and far zones
    - Closest Obstacle Tracking: Normalized distance to nearest detected obstacle
    - Risk Assessment: Quantitative risk level based on proximity to obstacles

    Configuration:
    - proximity_zones: List of distance thresholds for proximity zones (meters)
    - max_range: Maximum valid laser range (meters)
    - risk_threshold: Distance threshold for high risk (meters)

    Output Format: (len(proximity_zones) + 2)-dimensional normalized vector with zone occupancies,
    closest obstacle distance, and risk level.

    Applications: Collision avoidance, risk-aware navigation, and emergency stop behaviors.
    """

    name = "ObstacleProximitySpace"
    requires = {
        "front_laser": LidarRanges,  # Front-facing laser for obstacle proximity detection
    }

    def __init__(
        self,
        proximity_zones: List[float] = None,
        max_range: float = 10.0,
        risk_threshold: float = 1.0,
        *args,
        **kwargs
    ):
        """Initialize obstacle proximity space.

        Args:
            proximity_zones: Distance thresholds for zones [close, medium, far]
            max_range: Maximum laser range
            risk_threshold: Distance threshold for risk assessment
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.proximity_zones = proximity_zones or [0.5, 1.5, 3.0]
        self.max_range = max_range
        self.risk_threshold = risk_threshold

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for obstacle proximity."""
        # [zone_occupancies + closest_distance + risk_level]
        dims = len(self.proximity_zones) + 2

        return spaces.Box(
            low=np.array([0.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def encode_observation(self, front_laser: LidarRanges, *args, **kwargs) -> Any:
        """Encode advanced obstacle proximity with multi-zone detection and risk assessment.

        Processes preprocessed front-facing laser scan data to compute zone occupancies, closest obstacle
        distance, and risk level for robust proximity-based navigation and collision avoidance.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges for obstacle proximity analysis
                - Shape: (n,) where n = number of laser beams
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, max_range], NaN replaced with max_range
                - Example: [0.5, 1.2, 3.4, ..., 2.1] (used for proximity zone analysis)

        Returns:
            np.ndarray: Obstacle proximity vector with zone occupancies, closest distance, and risk level.
                - Shape: (len(proximity_zones) + 2,)
                - Dtype: np.float32
                - Elements: [zone_occupancies..., normalized_closest, risk_level]
                - Units: [normalized, normalized, normalized]
                - Range: all elements ∈ [0,1]
                - Example: [0.2, 0.5, 0.1, 0.3, 0.0] (zone occupancies + closest + risk)
        """
        # Filter valid readings
        valid_mask = (front_laser > 0.0) & (front_laser < self.max_range)
        valid_readings = front_laser[valid_mask]

        if len(valid_readings) == 0:
            # No valid readings
            result = [0.0] * len(self.proximity_zones) + [1.0, 0.0]
            return np.array(result, dtype=np.float32)

        # Compute zone occupancies
        zone_occupancies = []
        for zone_distance in self.proximity_zones:
            obstacles_in_zone = np.sum(valid_readings <= zone_distance)
            total_rays = len(front_laser)
            occupancy = obstacles_in_zone / total_rays
            zone_occupancies.append(occupancy)

        # Closest obstacle distance (normalized)
        closest_distance = np.min(valid_readings)
        normalized_closest = closest_distance / self.max_range

        # Risk level based on closest distance
        if closest_distance <= self.risk_threshold:
            risk_level = 1.0 - (closest_distance / self.risk_threshold)
        else:
            risk_level = 0.0

        return np.array(
            zone_occupancies + [normalized_closest, risk_level], dtype=np.float32
        )
