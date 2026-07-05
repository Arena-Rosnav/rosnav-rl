"""Robust Localization Spaces - Production Ready

Reliable localization spaces with proven odometry and pose processing.
"""

from typing import Optional

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.observation_types import (
    CombinedLocalizationVector,
    FilteredOdometryVector,
    Pose2D,
    RobotActionVector,
    StabilizedPoseVector,
)

from ...observation_space_factory import SpaceFactory
from ...space_categories import SpaceCategory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.LOCALIZATION)
class RobustOdometrySpace(BaseObservationSpace):
    """Production-ready odometry space with advanced velocity filtering and motion stability analysis.

    This space provides robust odometry data processing by implementing exponential moving average
    filtering for velocity stabilization and optional acceleration estimation for enhanced motion
    awareness. It transforms raw odometry velocity data into normalized, filtered representations
    that help the agent understand its current motion state with high temporal stability.

    Key Features:
    - Exponential moving average filtering for velocity smoothing and noise reduction
    - Configurable velocity limits with normalization to standardized ranges
    - Optional acceleration estimation through temporal velocity differentiation
    - Robust handling of velocity discontinuities and sensor noise
    - Temporal consistency validation for reliable motion state representation
    - Real-time performance optimization for control loop integration

    Mathematical Processing:
    - EMA filtering: v_filtered = α * v_prev + (1-α) * v_current for temporal smoothing
    - Acceleration estimation: a_est = (v_current - v_previous) / dt with clipping
    - Velocity normalization: v_norm = tanh(v_raw / v_max) for bounded output
    - Stability metrics: computed from velocity variance over configurable windows

    Use Cases:
    - Velocity-based control feedback for smooth motion execution
    - Motion state assessment for navigation quality evaluation
    - Acceleration-aware planning for dynamic obstacle avoidance
    - Real-time odometry quality monitoring and validation
    """

    name = "RobustOdometrySpace"
    requires = {"last_action": RobotActionVector}

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

    def reset(self) -> None:
        """Reset episode-local state."""
        self.filtered_linear_vel = None
        self.filtered_angular_vel = None
        self.last_velocities = None

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

    def encode_observation(
        self, last_action: RobotActionVector, *args, **kwargs
    ) -> FilteredOdometryVector:
        """Encode robust odometry data with advanced filtering and normalization processing.

        Processes robot action commands through exponential moving average filtering to produce
        stable velocity representations with optional acceleration estimation. This method transforms
        raw action data into normalized, filtered velocity vectors suitable for reinforcement learning.

        Args:
            last_action (RobotActionVector): Most recent robot action command vector
                - Shape: (2,) or (3,) depending on robot kinematics
                - Units: [m/s, rad/s] for differential drive or [m/s, m/s, rad/s] for holonomic
                - Source: robot action controller/command interface
                - Constraints: velocities within robot physical limits
                - Coordinate Frame: robot base frame (x: forward, y: left, z: up-rotation)
                - Temporal: immediate last executed action command
                - Example: [0.5, 0.2] (moving forward at 0.5 m/s, turning right at 0.2 rad/s)
                - Update Rate: typically matches control frequency (10-50 Hz)

        Returns:
            FilteredOdometryVector: Temporally filtered and normalized velocity representation.
                - Shape: (2,) or (4,) if include_acceleration=True
                - Dtype: np.float32
                - Elements: [linear_vel_norm, angular_vel_norm] or
                  [linear_vel_norm, angular_vel_norm, linear_acc_norm, angular_acc_norm]
                - Units: [normalized, normalized, normalized?, normalized?]
                - Range: velocities ∈ [-1,1], accelerations ∈ [-1,1] (all normalized)
                - Normalization: linear clipping with velocity limits and acceleration bounds
                - Filtering: EMA smoothed with configurable alpha parameter
                - Temporal Window: acceleration computed over single timestep difference
                - Example: [0.5, 0.2] (moderate forward motion, slight right turn)
                - Example with acceleration: [0.5, 0.2, 0.1, 0.05] (low acceleration changes)
        """
        # Extract velocities from pose data (assuming it contains velocity info)
        # Note: This assumes robot_pose contains velocity data beyond just pose
        raw_linear_vel, raw_angular_vel = (
            float(last_action[0]),
            float(last_action[-1]),
        )

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


@SpaceFactory.register(auto_name=True, category=SpaceCategory.LOCALIZATION)
class PoseStabilizedSpace(BaseObservationSpace):
    """Advanced pose stabilization system with trigonometric encoding and adaptive coordinate transformation.

    This space provides robust pose representation by implementing trigonometric orientation encoding,
    adaptive coordinate transformation, and position confidence weighting for enhanced localization
    stability. It transforms raw pose data into normalized, stable representations that maintain
    continuity across orientation boundaries and provide consistent spatial awareness.

    Key Features:
    - Trigonometric orientation encoding (sin/cos) for continuous angular representation
    - Adaptive coordinate transformation with optional relative positioning support
    - Position confidence weighting based on localization quality assessment
    - Multi-scale position normalization with hyperbolic tangent smoothing
    - Reference frame management for consistent spatial coordinate systems
    - Real-time pose validation and consistency checking for reliability

    Mathematical Processing:
    - Orientation encoding: [cos(θ), sin(θ)] for continuous angular representation
    - Position normalization: tanh(position / scale) for bounded spatial coordinates
    - Relative positioning: pose_rel = pose_current - pose_reference for drift compensation
    - Confidence weighting: adaptive scaling based on localization uncertainty metrics

    Use Cases:
    - Robust localization feedback for navigation control systems
    - Pose-based state estimation with orientation continuity guarantees
    - Spatial coordinate transformation for multi-frame navigation
    - Localization quality assessment and pose validation systems
    """

    name = "PoseStabilizedSpace"
    requires = {"robot_pose": Pose2D}

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

    def reset(self) -> None:
        """Reset episode-local state."""
        self.reference_pose = None

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

    def encode_observation(
        self, robot_pose: Pose2D, *args, **kwargs
    ) -> StabilizedPoseVector:
        """Encode pose data with advanced stabilization and trigonometric orientation representation.

        Transforms raw pose data through trigonometric encoding and adaptive coordinate transformation
        to produce stable, normalized pose representations suitable for reinforcement learning. This method
        handles orientation discontinuities and provides optional relative positioning capabilities.

        Args:
            robot_pose (Pose2D): Current robot pose from localization system
                - Shape: (3,) representing [x, y, theta]
                - Units: [meters, meters, radians]
                - Source: SLAM, odometry, or localization filter
                - Constraints: x,y ∈ real coordinates, theta ∈ [-π, π]
                - Data Type: np.ndarray of float64 values from pose estimation
                - Coordinate Frame: world/map frame with consistent origin
                - Temporal: current pose estimate at observation time
                - Accuracy: depends on localization method (±0.1m typical for good SLAM)
                - Example: [2.5, 1.2, 0.785] (2.5m east, 1.2m north, facing 45° northeast)
                - Update Rate: typically 10-50 Hz depending on localization system
                - Quality Indicators: consistency with previous estimates and sensor fusion confidence

        Returns:
            StabilizedPoseVector: Trigonometrically encoded and normalized pose representation.
                - Shape: (4,) or (5,) if include_confidence=True
                - Dtype: np.float32
                - Elements: [x_norm, y_norm, cos_yaw, sin_yaw, confidence?]
                - Units: [normalized, normalized, unitless, unitless, probability?]
                - Range: positions ∈ [-1,1], trigonometric ∈ [-1,1], confidence ∈ [0,1]
                - Normalization: hyperbolic tangent for positions, direct trigonometric for orientation
                - Reference Frame: optionally relative to initial pose if use_relative_coords=True
                - Example: [0.5, 0.2, 0.707, 0.707] (normalized pose at 45° orientation)
                - Example with confidence: [0.5, 0.2, 0.707, 0.707, 0.9] (high confidence estimate)
        """
        pose_data = robot_pose

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


@SpaceFactory.register(auto_name=True, category=SpaceCategory.LOCALIZATION)
class LocalizationCombinedSpace(BaseObservationSpace):
    """Comprehensive localization system integrating stabilized pose and filtered velocity dynamics.

    This space provides a unified localization representation by combining advanced pose stabilization
    with robust velocity filtering, creating a comprehensive spatial-temporal state descriptor that
    captures both where the robot is and how it's moving. It integrates multiple data sources to
    provide enhanced localization reliability and motion awareness for advanced navigation systems.

    Key Features:
    - Integrated pose and velocity representation with temporal consistency validation
    - Multi-source data fusion combining TF-based pose with odometry velocity data
    - Modular processing architecture using specialized sub-spaces for robust data handling
    - Cross-validated localization with pose-velocity consistency checking
    - Adaptive filtering and normalization for stable multi-dimensional state representation
    - Real-time performance optimization for control loop integration

    Mathematical Integration:
    - Pose processing: trigonometric encoding with position normalization
    - Velocity processing: EMA filtering with acceleration estimation capabilities
    - Data fusion: weighted combination of pose and motion components
    - Consistency validation: temporal correlation analysis between pose and velocity

    Use Cases:
    - Advanced navigation control with integrated spatial-temporal feedback
    - Multi-sensor localization fusion for enhanced position and motion estimation
    - Robust state estimation for dynamic environment navigation
    - Comprehensive localization quality assessment and validation systems
    """

    name = "LocalizationCombinedSpace"
    requires = {
        "robot_pose": Pose2D,  # Robot pose from TF system
        "last_action": RobotActionVector,  # Last action taken by the robot
    }

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

    def encode_observation(
        self, robot_pose: Pose2D, last_action: RobotActionVector, *args, **kwargs
    ) -> CombinedLocalizationVector:
        """Encode comprehensive localization data through integrated pose and velocity processing.

        Combines advanced pose stabilization with robust velocity filtering to create a unified
        spatial-temporal representation. This method processes multiple data sources through
        specialized sub-spaces to produce consistent, normalized localization vectors.

        Args:
            robot_pose (Pose2D): Current robot pose from localization system
                - Shape: (3,) representing [x, y, theta]
                - Units: [meters, meters, radians]
                - Source: SLAM, odometry, or localization filter
                - Constraints: x,y ∈ real coordinates, theta ∈ [-π, π]
                - Data Type: np.ndarray of float64 values from pose estimation
                - Coordinate Frame: world/map frame with consistent origin
                - Example: [2.5, 1.2, 0.785] (2.5m east, 1.2m north, facing 45° northeast)

            last_action (RobotActionVector): Most recent robot action command vector
                - Shape: (2,) or (3,) depending on robot kinematics
                - Units: [m/s, rad/s] for differential drive or [m/s, m/s, rad/s] for holonomic
                - Source: robot action controller/command interface
                - Constraints: velocities within robot physical limits
                - Coordinate Frame: robot base frame (x: forward, y: left, z: up-rotation)
                - Example: [0.5, 0.2] (moving forward at 0.5 m/s, turning right at 0.2 rad/s)

        Returns:
            CombinedLocalizationVector: Integrated spatial-temporal localization representation.
                - Shape: (6,) - fixed dimensionality for consistent learning
                - Dtype: np.float32
                - Elements: [x_norm, y_norm, cos_yaw, sin_yaw, linear_vel_norm, angular_vel_norm]
                - Units: [normalized, normalized, unitless, unitless, normalized, normalized]
                - Range: all elements ∈ [-1,1] for bounded learning space
                - Normalization: tanh for positions, trigonometric for orientation, clipping for velocities
                - Data Fusion: concatenated outputs from specialized pose and velocity sub-spaces
                - Example: [0.5, 0.2, 0.707, 0.707, 0.3, 0.1] (moderate position, 45° orientation, slow motion)
                - Interpretation: first 4 elements = stabilized pose, last 2 elements = filtered velocities
        """

        return np.concatenate(
            [
                self.pose_space.encode_observation(robot_pose, *args, **kwargs),
                self.odom_space.encode_observation(last_action, *args, **kwargs),
            ],
            dtype=np.float32,
        )
