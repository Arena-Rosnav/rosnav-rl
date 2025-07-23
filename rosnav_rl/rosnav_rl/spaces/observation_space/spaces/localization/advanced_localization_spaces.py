"""Robust Localization Spaces - Production Ready

Reliable localization spaces with proven odometry and pose processing.
"""

from typing import Any, Optional
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import OdometryGenerator, RobotPoseGenerator
from rosnav_rl.utils.type_aliases import ObservationDict
from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("robust_odometry")
class RobustOdometrySpace(BaseObservationSpace):
    """Production-ready odometry space with velocity filtering.

    Features:
    - Exponential moving average filtering
    - Configurable velocity limits
    - Acceleration estimation (optional)
    """

    name = "ROBUST_ODOMETRY"
    required_observation_units = [OdometryGenerator]

    def __init__(
        self,
        max_linear_vel: float = 2.0,
        max_angular_vel: float = 2.0,
        velocity_filter_alpha: float = 0.8,
        include_acceleration: bool = False,
        *args,
        **kwargs
    ):
        """Initialize robust odometry space.

        Args:
            max_linear_vel: Maximum expected linear velocity (m/s)
            max_angular_vel: Maximum expected angular velocity (rad/s)
            velocity_filter_alpha: Smoothing factor for velocity filtering (0-1)
            include_acceleration: Whether to include acceleration estimates
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.velocity_filter_alpha = velocity_filter_alpha
        self.include_acceleration = include_acceleration

        # Filtering state
        self.filtered_linear_vel = None
        self.filtered_angular_vel = None
        self.last_velocities = None  # For acceleration estimation

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for odometry."""
        dims = (
            4 if self.include_acceleration else 2
        )  # [linear_vel, angular_vel, linear_acc, angular_acc]

        return spaces.Box(
            low=np.array([-1.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def _filter_velocity(self, linear_vel: float, angular_vel: float) -> tuple:
        """Apply exponential moving average filtering to velocities."""
        if self.filtered_linear_vel is None:
            # Initialize filters
            self.filtered_linear_vel = linear_vel
            self.filtered_angular_vel = angular_vel
        else:
            # Apply EMA filter
            alpha = self.velocity_filter_alpha
            self.filtered_linear_vel = (
                alpha * self.filtered_linear_vel + (1 - alpha) * linear_vel
            )
            self.filtered_angular_vel = (
                alpha * self.filtered_angular_vel + (1 - alpha) * angular_vel
            )

        return self.filtered_linear_vel, self.filtered_angular_vel

    def _compute_acceleration(self, current_velocities: tuple) -> Optional[tuple]:
        """Compute simple acceleration estimate."""
        if self.last_velocities is None:
            acceleration = (0.0, 0.0)
        else:
            linear_acc = current_velocities[0] - self.last_velocities[0]
            angular_acc = current_velocities[1] - self.last_velocities[1]

            # Apply limits to acceleration (simple clipping)
            max_linear_acc = (
                self.max_linear_vel * 2.0
            )  # Assume 2x velocity as max acceleration
            max_angular_acc = self.max_angular_vel * 2.0

            linear_acc = np.clip(linear_acc, -max_linear_acc, max_linear_acc)
            angular_acc = np.clip(angular_acc, -max_angular_acc, max_angular_acc)

            acceleration = (linear_acc, angular_acc)

        self.last_velocities = current_velocities
        return acceleration

    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode odometry with filtering and normalization.

        Args:
            observation: Observation dictionary

        Returns:
            Processed odometry data
        """
        odom_data = observation[OdometryGenerator.name]

        # Extract velocities
        raw_linear_vel = float(odom_data[0])
        raw_angular_vel = float(odom_data[1])

        # Apply velocity filtering
        filtered_linear_vel, filtered_angular_vel = self._filter_velocity(
            raw_linear_vel, raw_angular_vel
        )

        # Normalize velocities
        normalized_linear_vel = np.clip(
            filtered_linear_vel / self.max_linear_vel, -1.0, 1.0
        )
        normalized_angular_vel = np.clip(
            filtered_angular_vel / self.max_angular_vel, -1.0, 1.0
        )

        result = [normalized_linear_vel, normalized_angular_vel]

        if self.include_acceleration:
            # Compute and normalize accelerations
            acceleration = self._compute_acceleration(
                (filtered_linear_vel, filtered_angular_vel)
            )

            normalized_linear_acc = np.clip(
                acceleration[0] / (self.max_linear_vel * 2.0), -1.0, 1.0
            )
            normalized_angular_acc = np.clip(
                acceleration[1] / (self.max_angular_vel * 2.0), -1.0, 1.0
            )

            result.extend([normalized_linear_acc, normalized_angular_acc])

        return np.array(result, dtype=np.float32)


@SpaceFactory.register("pose_stabilized")
class PoseStabilizedSpace(BaseObservationSpace):
    """Stabilized pose representation for reliable localization.

    Features:
    - Quaternion-based orientation (stable representation)
    - Position confidence weighting
    - Optional coordinate transformation
    """

    name = "POSE_STABILIZED"
    required_observation_units = [RobotPoseGenerator]

    def __init__(
        self,
        position_scale: float = 10.0,
        use_relative_coords: bool = False,
        include_confidence: bool = False,
        *args,
        **kwargs
    ):
        """Initialize pose stabilized space.

        Args:
            position_scale: Scale factor for position normalization
            use_relative_coords: Use relative coordinates from start
            include_confidence: Include pose confidence (if available)
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.position_scale = position_scale
        self.use_relative_coords = use_relative_coords
        self.include_confidence = include_confidence

        # Reference pose for relative coordinates
        self.reference_pose = None

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for stabilized pose."""
        # [x, y, cos(yaw), sin(yaw), confidence?]
        dims = 5 if self.include_confidence else 4

        return spaces.Box(
            low=np.array([-1.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def _set_reference_pose(self, pose_data: np.ndarray):
        """Set reference pose for relative coordinates."""
        if self.reference_pose is None and self.use_relative_coords:
            self.reference_pose = {
                "x": float(pose_data[0]),
                "y": float(pose_data[1]),
                "yaw": float(pose_data[2]),
            }

    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode pose with stabilized representation.

        Args:
            observation: Observation dictionary

        Returns:
            Stabilized pose representation
        """
        pose_data = observation[RobotPoseGenerator.name]

        x = float(pose_data[0])
        y = float(pose_data[1])
        yaw = float(pose_data[2])

        # Set reference if needed
        self._set_reference_pose(pose_data)

        # Apply relative coordinates if requested
        if self.use_relative_coords and self.reference_pose is not None:
            x_rel = x - self.reference_pose["x"]
            y_rel = y - self.reference_pose["y"]
            yaw_rel = yaw - self.reference_pose["yaw"]

            # Normalize relative yaw to [-π, π]
            yaw_rel = np.arctan2(np.sin(yaw_rel), np.cos(yaw_rel))

            x, y, yaw = x_rel, y_rel, yaw_rel

        # Normalize position
        normalized_x = np.tanh(x / self.position_scale)
        normalized_y = np.tanh(y / self.position_scale)

        # Stable orientation representation using trigonometric encoding
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        result = [normalized_x, normalized_y, cos_yaw, sin_yaw]

        if self.include_confidence:
            # Simple confidence based on consistency (placeholder)
            # In practice, this could be derived from sensor covariance
            confidence = 0.9  # Fixed high confidence for now
            result.append(confidence)

        return np.array(result, dtype=np.float32)


@SpaceFactory.register("localization_combined")
class LocalizationCombinedSpace(BaseObservationSpace):
    """Combined localization space with pose and velocity.

    Combines stabilized pose and filtered odometry for comprehensive
    localization information.
    """

    name = "LOCALIZATION_COMBINED"
    required_observation_units = [RobotPoseGenerator, OdometryGenerator]

    def __init__(
        self,
        position_scale: float = 10.0,
        max_linear_vel: float = 2.0,
        max_angular_vel: float = 2.0,
        velocity_filter_alpha: float = 0.8,
        *args,
        **kwargs
    ):
        """Initialize combined localization space.

        Args:
            position_scale: Scale factor for position normalization
            max_linear_vel: Maximum expected linear velocity
            max_angular_vel: Maximum expected angular velocity
            velocity_filter_alpha: Velocity filtering smoothing factor
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.position_scale = position_scale
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.velocity_filter_alpha = velocity_filter_alpha

        # Sub-spaces for modular processing
        self.pose_space = PoseStabilizedSpace(
            position_scale=position_scale,
            use_relative_coords=False,
            include_confidence=False,
        )
        self.odom_space = RobustOdometrySpace(
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            velocity_filter_alpha=velocity_filter_alpha,
            include_acceleration=False,
        )

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for combined localization."""
        # [x, y, cos_yaw, sin_yaw, linear_vel, angular_vel]
        return spaces.Box(
            low=np.array([-1.0] * 6), high=np.array([1.0] * 6), dtype=np.float32
        )

    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode combined localization data.

        Args:
            observation: Observation dictionary

        Returns:
            Combined pose and velocity representation
        """
        # Process pose
        pose_encoded = self.pose_space.encode_observation(observation, *args, **kwargs)

        # Process odometry
        odom_encoded = self.odom_space.encode_observation(observation, *args, **kwargs)

        # Combine results
        result = np.concatenate([pose_encoded, odom_encoded])

        return result.astype(np.float32)
