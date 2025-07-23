"""Robust Meta Spaces - Production Ready

Meta-information and high-level context spaces with reliable features.
"""

from typing import Any, Dict
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import DistAngleToGoalGenerator, OdometryGenerator, LaserScanGenerator
from rosnav_rl.utils.type_aliases import ObservationDict
from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("mission_context")
class MissionContextSpace(BaseObservationSpace):
    """Production-ready mission context information.
    
    Features:
    - Progress tracking
    - Time awareness (normalized)
    - Simple mission phase detection
    """
    
    name = "MISSION_CONTEXT"
    required_observation_units = [DistAngleToGoalGenerator]
    
    def __init__(self,
                 max_mission_time: float = 300.0,  # 5 minutes
                 progress_smoothing: float = 0.9,
                 phase_distance_thresholds: list = None,
                 *args, **kwargs):
        """Initialize mission context space.
        
        Args:
            max_mission_time: Maximum expected mission duration (seconds)
            progress_smoothing: Smoothing factor for progress tracking
            phase_distance_thresholds: Distance thresholds for mission phases
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.max_mission_time = max_mission_time
        self.progress_smoothing = progress_smoothing
        self.phase_distance_thresholds = phase_distance_thresholds or [10.0, 3.0, 1.0]
        
        # State tracking
        self.mission_start_time = None
        self.initial_distance = None
        self.smoothed_progress = 0.0
        self.step_count = 0
        
        super().__init__(*args, **kwargs)
    
    def get_gym_space(self) -> spaces.Space:
        """Return gym space for mission context."""
        # [progress, time_normalized, mission_phase, urgency]
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
    
    def _initialize_mission(self, goal_distance: float):
        """Initialize mission tracking."""
        if self.initial_distance is None:
            self.initial_distance = max(goal_distance, 1.0)  # Avoid division by zero
            self.mission_start_time = self.step_count
    
    def _compute_progress(self, current_distance: float) -> float:
        """Compute mission progress."""
        if self.initial_distance is None:
            return 0.0
        
        # Raw progress based on distance reduction
        raw_progress = max(0.0, 1.0 - (current_distance / self.initial_distance))
        
        # Apply smoothing
        self.smoothed_progress = (
            self.progress_smoothing * self.smoothed_progress + 
            (1.0 - self.progress_smoothing) * raw_progress
        )
        
        return self.smoothed_progress
    
    def _determine_mission_phase(self, goal_distance: float) -> float:
        """Determine current mission phase based on distance to goal.
        
        Returns:
            Mission phase as normalized value [0, 1]
        """
        thresholds = sorted(self.phase_distance_thresholds, reverse=True)
        
        for i, threshold in enumerate(thresholds):
            if goal_distance > threshold:
                return i / len(thresholds)
        
        # Close to goal (final phase)
        return 1.0
    
    def _compute_urgency(self, progress: float, time_normalized: float) -> float:
        """Compute urgency based on progress and time."""
        # Urgency increases if time is passing but progress is slow
        expected_progress = time_normalized * 0.8  # Expected 80% progress by end of time
        
        if progress < expected_progress:
            urgency = min(1.0, (expected_progress - progress) * 2.0)
        else:
            urgency = 0.0
        
        return urgency
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode mission context.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Mission context representation
        """
        goal_data = observation[DistAngleToGoalGenerator.name]
        goal_distance = float(goal_data[0])
        
        self.step_count += 1
        
        # Initialize mission tracking
        self._initialize_mission(goal_distance)
        
        # Compute progress
        progress = self._compute_progress(goal_distance)
        
        # Time normalization
        elapsed_steps = self.step_count - (self.mission_start_time or 0)
        # Assume ~10 Hz, so max_time * 10 steps
        max_steps = self.max_mission_time * 10.0
        time_normalized = min(1.0, elapsed_steps / max_steps)
        
        # Mission phase
        mission_phase = self._determine_mission_phase(goal_distance)
        
        # Urgency
        urgency = self._compute_urgency(progress, time_normalized)
        
        result = [progress, time_normalized, mission_phase, urgency]
        
        return np.array(result, dtype=np.float32)


@SpaceFactory.register("performance_context")
class PerformanceContextSpace(BaseObservationSpace):
    """Performance context with efficiency metrics.
    
    Features:
    - Path efficiency tracking
    - Energy efficiency (velocity-based)
    - Smoothness metrics
    """
    
    name = "PERFORMANCE_CONTEXT"
    required_observation_units = [DistAngleToGoalGenerator, OdometryGenerator]
    
    def __init__(self,
                 efficiency_window: int = 50,
                 max_velocity: float = 2.0,
                 *args, **kwargs):
        """Initialize performance context space.
        
        Args:
            efficiency_window: Window size for efficiency calculations
            max_velocity: Maximum velocity for energy calculations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.efficiency_window = efficiency_window
        self.max_velocity = max_velocity
        
        # Performance tracking
        self.distance_history = []
        self.velocity_history = []
        self.cumulative_distance_traveled = 0.0
        self.last_position = None
        
        super().__init__(*args, **kwargs)
    
    def get_gym_space(self) -> spaces.Space:
        """Return gym space for performance context."""
        # [path_efficiency, energy_efficiency, smoothness, overall_performance]
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
    
    def _update_tracking(self, goal_distance: float, velocity: tuple):
        """Update performance tracking."""
        # Track goal distance
        self.distance_history.append(goal_distance)
        if len(self.distance_history) > self.efficiency_window:
            self.distance_history.pop(0)
        
        # Track velocity
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > self.efficiency_window:
            self.velocity_history.pop(0)
    
    def _compute_path_efficiency(self) -> float:
        """Compute path efficiency based on goal distance reduction."""
        if len(self.distance_history) < 2:
            return 0.5  # Neutral efficiency
        
        # Measure how consistently distance to goal decreases
        distance_changes = np.diff(self.distance_history)
        
        # Good efficiency = consistent distance reduction
        beneficial_steps = np.sum(distance_changes < 0)
        total_steps = len(distance_changes)
        
        efficiency = beneficial_steps / total_steps if total_steps > 0 else 0.5
        return efficiency
    
    def _compute_energy_efficiency(self) -> float:
        """Compute energy efficiency based on velocity usage."""
        if len(self.velocity_history) == 0:
            return 1.0  # No movement = perfect energy efficiency
        
        velocities = np.array(self.velocity_history)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Energy efficiency inversely related to average velocity usage
        avg_velocity_ratio = np.mean(velocity_magnitudes) / self.max_velocity
        energy_efficiency = 1.0 - avg_velocity_ratio
        
        return np.clip(energy_efficiency, 0.0, 1.0)
    
    def _compute_smoothness(self) -> float:
        """Compute motion smoothness."""
        if len(self.velocity_history) < 2:
            return 1.0  # No changes = perfect smoothness
        
        velocities = np.array(self.velocity_history)
        
        # Smoothness based on velocity consistency
        velocity_changes = np.diff(velocities, axis=0)
        acceleration_magnitudes = np.linalg.norm(velocity_changes, axis=1)
        
        # Lower acceleration variance = higher smoothness
        if len(acceleration_magnitudes) > 0:
            avg_acceleration = np.mean(acceleration_magnitudes)
            max_expected_acceleration = self.max_velocity * 0.5  # Assume reasonable acceleration
            
            smoothness = 1.0 - min(1.0, avg_acceleration / max_expected_acceleration)
        else:
            smoothness = 1.0
        
        return smoothness
    
    def _compute_overall_performance(self, path_eff: float, energy_eff: float, smoothness: float) -> float:
        """Compute overall performance score."""
        # Weighted combination of metrics
        weights = [0.5, 0.3, 0.2]  # Path efficiency most important
        overall = (weights[0] * path_eff + 
                  weights[1] * energy_eff + 
                  weights[2] * smoothness)
        
        return overall
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode performance context.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Performance context representation
        """
        goal_data = observation[DistAngleToGoalGenerator.name]
        odom_data = observation[OdometryGenerator.name]
        
        goal_distance = float(goal_data[0])
        velocity = (float(odom_data[0]), float(odom_data[1]))
        
        # Update tracking
        self._update_tracking(goal_distance, velocity)
        
        # Compute metrics
        path_efficiency = self._compute_path_efficiency()
        energy_efficiency = self._compute_energy_efficiency()
        smoothness = self._compute_smoothness()
        overall_performance = self._compute_overall_performance(
            path_efficiency, energy_efficiency, smoothness
        )
        
        result = [path_efficiency, energy_efficiency, smoothness, overall_performance]
        
        return np.array(result, dtype=np.float32)


@SpaceFactory.register("safety_context")
class SafetyContextSpace(BaseObservationSpace):
    """Safety context with risk assessment.
    
    Features:
    - Collision risk assessment
    - Safety margin tracking
    - Violation history
    """
    
    name = "SAFETY_CONTEXT"
    required_observation_units = [LaserScanGenerator, OdometryGenerator]
    
    def __init__(self,
                 safety_distance: float = 1.0,
                 critical_distance: float = 0.3,
                 risk_history_length: int = 20,
                 *args, **kwargs):
        """Initialize safety context space.
        
        Args:
            safety_distance: Safe distance threshold
            critical_distance: Critical distance threshold
            risk_history_length: Length of risk history to track
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.safety_distance = safety_distance
        self.critical_distance = critical_distance
        self.risk_history_length = risk_history_length
        
        # Safety tracking
        self.risk_history = []
        self.violation_count = 0
        self.total_steps = 0
        
        super().__init__(*args, **kwargs)
    
    def get_gym_space(self) -> spaces.Space:
        """Return gym space for safety context."""
        # [current_risk, avg_risk, violation_rate, safety_margin]
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
    
    def _compute_collision_risk(self, laser_data: np.ndarray, velocity: tuple) -> float:
        """Compute current collision risk."""
        # Filter valid laser readings
        valid_mask = laser_data > 0.0
        valid_readings = laser_data[valid_mask]
        
        if len(valid_readings) == 0:
            return 0.0  # No valid readings = no risk
        
        # Find closest obstacle
        min_distance = np.min(valid_readings)
        
        # Base risk on distance
        if min_distance <= self.critical_distance:
            distance_risk = 1.0
        elif min_distance <= self.safety_distance:
            # Linear risk increase as distance decreases
            distance_risk = (self.safety_distance - min_distance) / (
                self.safety_distance - self.critical_distance
            )
        else:
            distance_risk = 0.0
        
        # Modify risk based on velocity (higher velocity = higher risk)
        velocity_magnitude = np.linalg.norm(velocity)
        velocity_factor = min(1.5, 1.0 + velocity_magnitude * 0.5)
        
        total_risk = min(1.0, distance_risk * velocity_factor)
        
        return total_risk
    
    def _update_safety_tracking(self, current_risk: float):
        """Update safety tracking."""
        self.total_steps += 1
        
        # Update risk history
        self.risk_history.append(current_risk)
        if len(self.risk_history) > self.risk_history_length:
            self.risk_history.pop(0)
        
        # Count violations (high risk situations)
        if current_risk > 0.7:  # Threshold for violation
            self.violation_count += 1
    
    def _compute_safety_metrics(self, current_risk: float) -> Dict[str, float]:
        """Compute safety metrics."""
        # Average risk
        if len(self.risk_history) > 0:
            avg_risk = np.mean(self.risk_history)
        else:
            avg_risk = current_risk
        
        # Violation rate
        violation_rate = self.violation_count / max(self.total_steps, 1)
        
        # Safety margin (inverse of average risk)
        safety_margin = 1.0 - avg_risk
        
        return {
            'avg_risk': avg_risk,
            'violation_rate': violation_rate,
            'safety_margin': safety_margin
        }
    
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """Encode safety context.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Safety context representation
        """
        laser_data = observation[LaserScanGenerator.name]
        odom_data = observation[OdometryGenerator.name]
        
        velocity = (float(odom_data[0]), float(odom_data[1]))
        
        # Compute current collision risk
        current_risk = self._compute_collision_risk(laser_data, velocity)
        
        # Update tracking
        self._update_safety_tracking(current_risk)
        
        # Compute metrics
        safety_metrics = self._compute_safety_metrics(current_risk)
        
        result = [
            current_risk,
            safety_metrics['avg_risk'],
            safety_metrics['violation_rate'],
            safety_metrics['safety_margin']
        ]
        
        return np.array(result, dtype=np.float32)
