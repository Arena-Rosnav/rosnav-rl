"""Robust Dynamics Spaces - Production Ready

Robot dynamics and motion-related spaces with proven features.
"""

from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import OdometryGenerator, RobotPoseGenerator
from rosnav_rl.utils.type_aliases import ObservationDict
from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("motion_state")
class MotionStateSpace(BaseObservationSpace):
    """Production-ready motion state representation.
    
    Features:
    - Velocity magnitude and direction
    - Motion consistency tracking
    - Simple stability metrics
    """
    
    name = "MOTION_STATE"
    required_observation_units = [OdometryGenerator]
    
    def __init__(self,
                 max_velocity: float = 2.0,
                 stability_window: int = 5,
                 include_acceleration: bool = False,
                 *args, **kwargs):
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
    
    def get_gym_space(self) -> spaces.Space:
        """Return gym space for motion state."""
        # [velocity_magnitude, velocity_direction, stability, acceleration?]
        dims = 4 if self.include_acceleration else 3
        
        return spaces.Box(
            low=np.array([-1.0] * dims),
            high=np.array([1.0] * dims),
            dtype=np.float32
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
            stability = np.exp(-variance / (self.max_velocity ** 2))
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
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode motion state.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Motion state representation
        """
        odom_data = observation[OdometryGenerator.name]
        
        linear_vel = float(odom_data[0])
        angular_vel = float(odom_data[1])
        
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


@SpaceFactory.register("kinematic_state")
class KinematicStateSpace(BaseObservationSpace):
    """Kinematic state with pose and velocity integration.
    
    Features:
    - Combined position and velocity representation
    - Motion direction relative to orientation
    - Simple kinematic consistency checks
    """
    
    name = "KINEMATIC_STATE"
    required_observation_units = [RobotPoseGenerator, OdometryGenerator]
    
    def __init__(self,
                 max_velocity: float = 2.0,
                 position_scale: float = 10.0,
                 include_motion_alignment: bool = True,
                 *args, **kwargs):
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
            low=np.array([-1.0] * dims),
            high=np.array([1.0] * dims),
            dtype=np.float32
        )
    
    def _compute_motion_alignment(self, yaw: float, linear_vel: float, angular_vel: float) -> float:
        """Compute alignment between motion and orientation.
        
        Args:
            yaw: Current orientation
            linear_vel: Linear velocity
            angular_vel: Angular velocity
            
        Returns:
            Motion alignment metric [-1, 1]
        """
        if abs(linear_vel) < 1e-6:
            return 0.0  # No linear motion
        
        # Motion direction based on velocities
        if linear_vel > 0:
            motion_direction = yaw  # Forward motion
        else:
            motion_direction = yaw + np.pi  # Backward motion
        
        # Normalize motion direction
        motion_direction = np.arctan2(np.sin(motion_direction), np.cos(motion_direction))
        
        # Compute alignment (cosine of angle difference)
        alignment = np.cos(motion_direction - yaw)
        
        return alignment
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode kinematic state.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Kinematic state representation
        """
        pose_data = observation[RobotPoseGenerator.name]
        odom_data = observation[OdometryGenerator.name]
        
        # Extract pose
        x = float(pose_data[0])
        y = float(pose_data[1])
        yaw = float(pose_data[2])
        
        # Extract velocities
        linear_vel = float(odom_data[0])
        angular_vel = float(odom_data[1])
        
        # Normalize pose
        normalized_x = np.tanh(x / self.position_scale)
        normalized_y = np.tanh(y / self.position_scale)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Normalize velocities
        normalized_linear_vel = np.clip(linear_vel / self.max_velocity, -1.0, 1.0)
        normalized_angular_vel = np.clip(angular_vel / self.max_velocity, -1.0, 1.0)
        
        result = [normalized_x, normalized_y, cos_yaw, sin_yaw, 
                 normalized_linear_vel, normalized_angular_vel]
        
        if self.include_motion_alignment:
            motion_alignment = self._compute_motion_alignment(yaw, linear_vel, angular_vel)
            result.append(motion_alignment)
        
        return np.array(result, dtype=np.float32)


@SpaceFactory.register("trajectory_state")
class TrajectoryStateSpace(BaseObservationSpace):
    """Trajectory state with motion history.
    
    Features:
    - Recent motion trajectory
    - Path curvature estimation
    - Motion predictability metrics
    """
    
    name = "TRAJECTORY_STATE"
    required_observation_units = [RobotPoseGenerator, OdometryGenerator]
    
    def __init__(self,
                 history_length: int = 3,
                 max_velocity: float = 2.0,
                 position_scale: float = 5.0,
                 *args, **kwargs):
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
    
    def get_gym_space(self) -> spaces.Space:
        """Return gym space for trajectory state."""
        # [current_pose(4) + current_vel(2) + trajectory_features(3)]
        # trajectory_features: [path_length, curvature, speed_consistency]
        dims = 9
        
        return spaces.Box(
            low=np.array([-1.0] * dims),
            high=np.array([1.0] * dims),
            dtype=np.float32
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
            speed_consistency = np.exp(-speed_variance / (self.max_velocity ** 2))
        else:
            speed_consistency = 1.0
        
        return (normalized_path_length, normalized_curvature, speed_consistency)
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode trajectory state.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Trajectory state representation
        """
        pose_data = observation[RobotPoseGenerator.name]
        odom_data = observation[OdometryGenerator.name]
        
        # Current state
        x = float(pose_data[0])
        y = float(pose_data[1])
        yaw = float(pose_data[2])
        linear_vel = float(odom_data[0])
        angular_vel = float(odom_data[1])
        
        # Update history
        current_pose = (x, y, yaw)
        current_velocity = (linear_vel, angular_vel)
        self._update_history(current_pose, current_velocity)
        
        # Normalize current state
        normalized_x = np.tanh(x / self.position_scale)
        normalized_y = np.tanh(y / self.position_scale)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        normalized_linear_vel = np.clip(linear_vel / self.max_velocity, -1.0, 1.0)
        normalized_angular_vel = np.clip(angular_vel / self.max_velocity, -1.0, 1.0)
        
        # Compute trajectory features
        path_length, curvature, speed_consistency = self._compute_trajectory_features()
        
        result = [normalized_x, normalized_y, cos_yaw, sin_yaw,
                 normalized_linear_vel, normalized_angular_vel,
                 path_length, curvature, speed_consistency]
        
        return np.array(result, dtype=np.float32)
