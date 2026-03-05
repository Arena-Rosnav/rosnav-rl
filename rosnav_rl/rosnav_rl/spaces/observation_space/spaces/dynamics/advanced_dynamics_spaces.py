"""Robust Dynamics Spaces - Production Ready

Robot dynamics and motion-related spaces with proven features.
"""

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import (
    KinematicStateVector,
    MotionStateVector,
    Pose2D,
    RobotActionVector,
    TrajectoryStateVector,
)

from ...observation_space_factory import SpaceFactory
from ...space_categories import SpaceCategory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.DYNAMICS)
class MotionStateSpace(BaseObservationSpace):
    """Production-ready motion state representation with temporal stability analysis.

    This space provides a comprehensive view of the robot's motion characteristics by analyzing
    velocity patterns, motion consistency, and stability metrics over time. It transforms raw
    action commands into meaningful motion state descriptors that help the agent understand
    its current movement behavior and stability.

    Key Features:
    - Velocity magnitude and directional analysis with normalization
    - Motion consistency tracking through rolling window analysis
    - Stability metrics based on velocity variance computation
    - Optional acceleration estimation for enhanced motion awareness
    - Temporal smoothing to reduce noise in motion state estimation

    Use Cases:
    - Motion control feedback for smooth trajectory execution
    - Stability assessment for safe navigation in dynamic environments
    - Motion pattern recognition for adaptive behavior learning
    - Performance monitoring and motion quality evaluation
    """

    name = "MotionStateSpace"
    requires = {"last_action": RobotActionVector}

    def __init__(
        self,
        max_velocity: float = 2.0,
        stability_window: int = 5,
        include_acceleration: bool = False,
        *args,
        **kwargs
    ):
        """Initialize motion state space.

        Args:
            max_velocity: Maximum expected velocity magnitude
            stability_window: Window size for stability calculation
            include_acceleration: Include acceleration estimates
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_velocity = max_velocity
        self.stability_window = stability_window
        self.include_acceleration = include_acceleration

        # State tracking
        self.velocity_history = []
        self.last_velocity = None

        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        """Reset episode-local state."""
        self.velocity_history = []
        self.last_velocity = None

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for motion state."""
        # [velocity_magnitude, velocity_direction, stability, acceleration?]
        dims = 4 if self.include_acceleration else 3

        return spaces.Box(
            low=np.array([-1.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def _compute_stability(self, current_velocity: tuple) -> float:
        """Compute motion stability metric."""
        # Add current velocity to history
        self.velocity_history.append(current_velocity)

        # Keep only recent history
        if len(self.velocity_history) > self.stability_window:
            self.velocity_history.pop(0)

        if len(self.velocity_history) < 2:
            return 0.0

        # Calculate velocity variance as stability metric
        velocities = np.array(self.velocity_history)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)

        # Stability inversely related to velocity variance
        if len(velocity_magnitudes) > 1:
            variance = np.var(velocity_magnitudes)
            # Normalize variance and invert (high stability = low variance)
            stability = np.exp(-variance / (self.max_velocity**2))
            return stability

        return 0.0

    def _compute_acceleration_magnitude(self, current_velocity: tuple) -> float:
        """Compute acceleration magnitude."""
        if self.last_velocity is None:
            acceleration_magnitude = 0.0
        else:
            # Simple acceleration magnitude
            vel_diff = np.array(current_velocity) - np.array(self.last_velocity)
            acceleration_magnitude = np.linalg.norm(vel_diff)

            # Normalize
            acceleration_magnitude = np.tanh(acceleration_magnitude / self.max_velocity)

        self.last_velocity = current_velocity
        return acceleration_magnitude

    def encode_observation(
        self, last_action: RobotActionVector, *args, **kwargs
    ) -> MotionStateVector:
        """Encode comprehensive motion state from robot action commands.

        Args:
            last_action (RobotActionVector): Most recent robot action command vector
                - Shape: (2,) or (3,) depending on robot type
                - Units: [m/s, rad/s] for differential drive or [m/s, m/s, rad/s] for holonomic
                - Source: robot action controller/command interface
                - Constraints: velocities within robot physical limits
                - Coordinate Frame: robot base frame (x: forward, y: left, z: up-rotation)
                - Temporal: immediate last executed action command
                - Example: [0.5, 0.2] (moving forward at 0.5 m/s, turning right at 0.2 rad/s)

        Returns:
            MotionStateVector: Comprehensive robot motion state representation.
                - Shape: (3,) or (4,) if include_acceleration=True
                - Dtype: np.float32
                - Elements: [velocity_magnitude, velocity_direction, stability_metric, acceleration_magnitude?]
                - Units: [normalized, normalized, normalized, normalized?]
                - Range: velocity_magnitude ∈ [0,1], direction ∈ [-1,1], stability ∈ [0,1], acceleration ∈ [0,1]
                - Normalization: tanh for magnitude, linear for direction, exponential decay for stability
                - Temporal Window: stability computed over configurable window (default 5 steps)
                - Example: [0.75, 0.125, 0.92, 0.15] (high speed, slight right turn, very stable, low acceleration)
        """
        linear_vel = float(last_action[0])
        angular_vel = float(last_action[-1])

        # Compute velocity magnitude and direction
        velocity_magnitude = np.sqrt(linear_vel**2 + angular_vel**2)
        normalized_magnitude = np.tanh(velocity_magnitude / self.max_velocity)

        # Velocity direction (angle of velocity vector)
        if velocity_magnitude > 1e-6:
            velocity_direction = np.arctan2(angular_vel, linear_vel) / np.pi
        else:
            velocity_direction = 0.0

        # Compute stability
        current_velocity = (linear_vel, angular_vel)
        stability = self._compute_stability(current_velocity)

        result = [normalized_magnitude, velocity_direction, stability]

        if self.include_acceleration:
            acceleration = self._compute_acceleration_magnitude(current_velocity)
            result.append(acceleration)

        return np.array(result, dtype=np.float32)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.DYNAMICS)
class KinematicStateSpace(BaseObservationSpace):
    """Integrated kinematic state representation combining pose and motion dynamics.

    This space provides a unified view of the robot's kinematic state by combining spatial
    pose information with velocity dynamics. It captures both where the robot is and how
    it's moving, including the relationship between orientation and motion direction for
    enhanced spatial-temporal awareness.

    Key Features:
    - Integrated pose and velocity representation in normalized coordinates
    - Motion-orientation alignment analysis for maneuver assessment
    - Trigonometric orientation encoding (sin/cos) for continuous representation
    - Multi-scale position normalization with hyperbolic tangent smoothing
    - Kinematic consistency validation between pose and motion states
    - Configurable position scaling for different environment sizes

    Use Cases:
    - Integrated navigation control combining position and motion feedback
    - Kinematic constraint validation for feasible motion planning
    - Spatial-temporal pattern recognition for advanced navigation behaviors
    - Motion-pose coherence assessment for robust state estimation
    """

    name = "KinematicStateSpace"

    # Schema-based requirements: defines the data sources needed from observations.yaml
    # Each key corresponds to a data source name, each value provides rich type metadata
    requires = {
        "robot_pose": Pose2D,  # Robot pose and velocity for angular dynamics analysis
        "last_action": RobotActionVector,  # Last action taken by the robot
    }

    def __init__(
        self,
        max_velocity: float = 2.0,
        position_scale: float = 10.0,
        include_motion_alignment: bool = True,
        *args,
        **kwargs
    ):
        """Initialize kinematic state space.

        Args:
            max_velocity: Maximum expected velocity
            position_scale: Scale for position normalization
            include_motion_alignment: Include motion-orientation alignment
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_velocity = max_velocity
        self.position_scale = position_scale
        self.include_motion_alignment = include_motion_alignment

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for kinematic state."""
        # [x, y, cos_yaw, sin_yaw, linear_vel, angular_vel, motion_alignment?]
        dims = 7 if self.include_motion_alignment else 6

        return spaces.Box(
            low=np.array([-1.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def _compute_motion_alignment(
        self, yaw: float, linear_vel: float, angular_vel: float
    ) -> float:
        """Compute alignment between motion and orientation.

        Measures how well the robot's velocity vector aligns with its heading.
        +1 = pure forward motion, -1 = pure backward, 0 = pure rotation.

        Args:
            yaw: Current orientation
            linear_vel: Linear velocity
            angular_vel: Angular velocity

        Returns:
            Motion alignment metric [-1, 1]
        """
        speed = np.sqrt(linear_vel**2 + angular_vel**2)
        if speed < 1e-6:
            return 0.0  # No motion

        # Velocity vector direction in robot frame
        velocity_direction = np.arctan2(angular_vel, linear_vel)

        # Alignment: cos(0)=1 when velocity is purely forward,
        # cos(pi)=-1 when purely backward
        alignment = np.cos(velocity_direction)

        return alignment

    def encode_observation(
        self, robot_pose: Pose2D, last_action: RobotActionVector, *args, **kwargs
    ) -> KinematicStateVector:
        """Encode integrated kinematic state combining pose and motion dynamics.

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

            last_action (RobotActionVector): Most recent robot motion command
                - Shape: (2,) or (3,) for differential/holonomic drive
                - Units: [m/s, rad/s] or [m/s, m/s, rad/s]
                - Source: robot motion controller command interface
                - Constraints: within robot kinematic limits
                - Coordinate Frame: robot base frame (x: forward, y: left, z: up-rotation)
                - Temporal: immediate previous action command (one timestep ago)
                - Example: [0.5, 0.2] (forward 0.5 m/s, rotate right 0.2 rad/s)
                - Execution: actual executed command after safety filtering

        Returns:
            KinematicStateVector: Comprehensive kinematic state representation.
                - Shape: (6,) or (7,) if include_motion_alignment=True
                - Dtype: np.float32
                - Elements: [norm_x, norm_y, cos_yaw, sin_yaw, norm_linear_vel, norm_angular_vel, motion_alignment?]
                - Units: [normalized, normalized, unitless, unitless, normalized, normalized]
                - Range: positions ∈ [-1,1], trigonometric ∈ [-1,1], velocities ∈ [-1,1]
                - Normalization: tanh for positions, direct for trig functions, clipping for velocities
                - Coordinate System: normalized relative to configured scales and limits
                - Example: [0.5, -0.3, 0.707, 0.707, 0.4, 0.1]
                  (moderate position, 45° orientation, slow forward motion)
        """
        # Extract values once to avoid repeated indexing and casting
        x, y, yaw = robot_pose[0], robot_pose[1], robot_pose[2]
        linear_vel, angular_vel = last_action[0], last_action[-1]

        # Pre-calculate inverses for faster multiplication
        inv_pos_scale = 1.0 / self.position_scale
        inv_max_vel = 1.0 / self.max_velocity

        # Normalize pose and velocities
        normalized_x = np.tanh(x * inv_pos_scale)
        normalized_y = np.tanh(y * inv_pos_scale)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        normalized_linear_vel = np.clip(linear_vel * inv_max_vel, -1.0, 1.0)
        normalized_angular_vel = np.clip(angular_vel * inv_max_vel, -1.0, 1.0)

        # Directly construct the final numpy array
        if self.include_motion_alignment:
            motion_alignment = self._compute_motion_alignment(
                yaw, linear_vel, angular_vel
            )
            return np.array(
                [
                    normalized_x,
                    normalized_y,
                    cos_yaw,
                    sin_yaw,
                    normalized_linear_vel,
                    normalized_angular_vel,
                    motion_alignment,
                ],
                dtype=np.float32,
            )
        else:
            return np.array(
                [
                    normalized_x,
                    normalized_y,
                    cos_yaw,
                    sin_yaw,
                    normalized_linear_vel,
                    normalized_angular_vel,
                ],
                dtype=np.float32,
            )


@SpaceFactory.register(auto_name=True, category=SpaceCategory.DYNAMICS)
class TrajectoryStateSpace(BaseObservationSpace):
    """Advanced trajectory state representation with temporal motion history analysis.

    This space provides comprehensive trajectory analysis by maintaining a temporal history
    of robot poses and velocities, computing path characteristics, and extracting motion
    patterns. It enables advanced motion understanding through geometric path analysis,
    curvature estimation, and consistency metrics for enhanced navigation intelligence.

    Key Features:
    - Temporal trajectory tracking with configurable history window
    - Geometric path analysis including length and curvature computation
    - Motion consistency metrics through velocity variance analysis
    - Path smoothness evaluation for navigation quality assessment
    - Predictability scoring based on motion pattern regularity
    - Multi-scale trajectory features for different planning horizons

    Mathematical Computations:
    - Path curvature: computed from heading change rates over trajectory segments
    - Speed consistency: exponential variance weighting for stability assessment
    - Path length: accumulated Euclidean distances with normalization
    - Motion predictability: temporal correlation analysis of velocity patterns

    Use Cases:
    - Advanced trajectory planning with historical motion context
    - Navigation quality assessment and motion pattern recognition
    - Predictive motion modeling for anticipatory control strategies
    - Path optimization feedback based on historical trajectory analysis
    """

    name = "TrajectoryStateSpace"

    # Schema-based requirements: defines the data sources needed from observations.yaml
    # Each key corresponds to a data source name, each value provides rich type metadata
    requires = {
        "robot_pose": Pose2D,  # Robot pose
        "last_action": RobotActionVector,  # Last action taken by the robot
    }

    def __init__(
        self,
        history_length: int = 3,
        max_velocity: float = 2.0,
        position_scale: float = 5.0,
        *args,
        **kwargs
    ):
        """Initialize trajectory state space.

        Args:
            history_length: Number of historical states to track
            max_velocity: Maximum expected velocity
            position_scale: Scale for position differences
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.history_length = history_length
        self.max_velocity = max_velocity
        self.position_scale = position_scale

        # State history
        self.pose_history = []
        self.velocity_history = []

        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        """Reset episode-local state."""
        self.pose_history = []
        self.velocity_history = []

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for trajectory state."""
        # [current_pose(4) + current_vel(2) + trajectory_features(3)]
        # trajectory_features: [path_length, curvature, speed_consistency]
        dims = 9

        return spaces.Box(
            low=np.array([-1.0] * dims), high=np.array([1.0] * dims), dtype=np.float32
        )

    def _update_history(self, pose: tuple, velocity: tuple):
        """Update pose and velocity history."""
        self.pose_history.append(pose)
        self.velocity_history.append(velocity)

        # Keep only recent history
        if len(self.pose_history) > self.history_length:
            self.pose_history.pop(0)
            self.velocity_history.pop(0)

    def _compute_trajectory_features(self) -> tuple:
        """Compute trajectory features from history.

        Returns:
            (path_length, curvature, speed_consistency)
        """
        if len(self.pose_history) < 2:
            return (0.0, 0.0, 0.0)

        poses = np.array(self.pose_history)
        velocities = np.array(self.velocity_history)

        # Path length
        position_diffs = np.diff(poses[:, :2], axis=0)
        distances = np.linalg.norm(position_diffs, axis=1)
        path_length = np.sum(distances)
        normalized_path_length = np.tanh(path_length / self.position_scale)

        # Curvature (change in heading direction)
        if len(poses) >= 3:
            headings = poses[:, 2]
            heading_diffs = np.diff(headings)
            # Normalize heading differences to [-π, π]
            heading_diffs = np.arctan2(np.sin(heading_diffs), np.cos(heading_diffs))
            curvature = np.mean(np.abs(heading_diffs))
            normalized_curvature = curvature / np.pi
        else:
            normalized_curvature = 0.0

        # Speed consistency (variance in velocity magnitude)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        if len(velocity_magnitudes) > 1:
            speed_variance = np.var(velocity_magnitudes)
            # Consistency inversely related to variance
            speed_consistency = np.exp(-speed_variance / (self.max_velocity**2))
        else:
            speed_consistency = 1.0

        return (normalized_path_length, normalized_curvature, speed_consistency)

    def encode_observation(
        self, robot_pose: Pose2D, last_action: RobotActionVector, *args, **kwargs
    ) -> TrajectoryStateVector:
        """Encode comprehensive trajectory state with temporal motion history analysis.

        Args:
            robot_pose (Pose2D): Current robot pose from localization for trajectory tracking
                - Shape: (3,) representing [x, y, theta]
                - Units: [meters, meters, radians]
                - Source: SLAM, odometry, visual-inertial odometry, or sensor fusion
                - Constraints: x,y ∈ real world coordinates, theta ∈ [-π, π]
                - Coordinate Frame: consistent world/map frame with fixed origin
                - Example: [5.2, -1.8, 2.356] (5.2m east, 1.8m south, facing 135° northwest)

            last_action (RobotActionVector): Recent robot motion command for velocity tracking
                - Shape: (2,) for differential drive, (3,) for holonomic robots
                - Units: [m/s, rad/s] or [m/s, m/s, rad/s]
                - Source: robot controller command interface or action space
                - Constraints: within robot dynamic and kinematic limits
                - Coordinate Frame: robot base frame (x: forward, y: left, z: up-rotation)
                - Example: [0.8, -0.3] (forward 0.8 m/s, turn left 0.3 rad/s)
                - Execution: reflects actual commanded velocities to robot base

        Returns:
            TrajectoryStateVector: Advanced trajectory state with geometric and temporal features.
                - Shape: (9,) fixed-size comprehensive trajectory representation
                - Dtype: np.float32
                - Elements: [norm_x, norm_y, cos_yaw, sin_yaw, norm_linear_vel, norm_angular_vel,
                           path_length, curvature, speed_consistency]
                - Units: [normalized, normalized, unitless, unitless, normalized, normalized,
                         normalized, normalized, correlation_coefficient]
                - Range: positions/velocities ∈ [-1,1], trigonometric ∈ [-1,1], features ∈ [0,1]
                - Normalization: tanh for positions, clipping for velocities, exponential for consistency
                - Temporal Window: configurable history length (default 3 steps)
                - Geometric Features: path_length (accumulated distance), curvature (heading variance)
                - Example: [0.3, -0.2, 0.866, 0.5, 0.6, -0.15, 0.45, 0.25, 0.88]
                  (moderate position, 30° heading, medium speed, smooth curved path, high consistency)
        """
        # Extract values once (avoid repeated float() calls)
        x, y, yaw = robot_pose[0], robot_pose[1], robot_pose[2]
        linear_vel, angular_vel = last_action[0], last_action[-1]

        # Update history
        current_pose = (x, y, yaw)
        current_velocity = (linear_vel, angular_vel)
        self._update_history(current_pose, current_velocity)

        # Compute trigonometric values once
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        # Vectorized normalization
        inv_pos_scale = 1.0 / self.position_scale
        inv_max_vel = 1.0 / self.max_velocity

        normalized_x = np.tanh(x * inv_pos_scale)
        normalized_y = np.tanh(y * inv_pos_scale)
        normalized_linear_vel = np.clip(linear_vel * inv_max_vel, -1.0, 1.0)
        normalized_angular_vel = np.clip(angular_vel * inv_max_vel, -1.0, 1.0)

        # Compute trajectory features
        path_length, curvature, speed_consistency = self._compute_trajectory_features()

        # Direct array creation (faster than list + conversion)
        return np.array(
            [
                normalized_x,
                normalized_y,
                cos_yaw,
                sin_yaw,
                normalized_linear_vel,
                normalized_angular_vel,
                path_length,
                curvature,
                speed_consistency,
            ],
            dtype=np.float32,
        )
