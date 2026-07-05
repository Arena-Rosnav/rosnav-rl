"""Robust Meta Spaces - Production Ready

Meta-information and high-level context spaces with reliable features.
"""

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.observation_types import (
    DistanceAngleMetrics,
    LidarRanges,
    MissionContextVector,
    PerformanceContextVector,
    RobotActionVector,
    SafetyContextVector,
)
from ...observation_space_factory import SpaceFactory
from ...space_categories import SpaceCategory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.META)
class MissionContextSpace(BaseObservationSpace):
    """Advanced mission context system with comprehensive progress tracking and temporal awareness.

    This space provides sophisticated mission management capabilities by tracking progress metrics,
    temporal context, and mission phase detection for enhanced navigation intelligence. It transforms
    goal-relative information into normalized mission context vectors that capture both immediate
    progress and long-term mission evolution patterns.

    Key Features:
    - Multi-dimensional progress tracking with exponential smoothing for temporal stability
    - Adaptive mission phase detection based on configurable distance thresholds
    - Temporal awareness with normalized time representation for deadline management
    - Urgency computation combining progress rate with temporal constraints
    - Mission state persistence across observation cycles for consistent tracking
    - Real-time performance optimization for control loop integration

    Mathematical Processing:
    - Progress computation: progress = 1 - (current_distance / initial_distance) with EMA smoothing
    - Phase detection: categorical assignment based on distance threshold comparison
    - Urgency calculation: urgency = max(0, expected_progress - actual_progress) * urgency_factor
    - Time normalization: time_norm = elapsed_time / max_mission_time with clamping

    Use Cases:
    - Mission-aware navigation with deadline constraints and progress monitoring
    - Adaptive behavior based on mission phase and temporal context
    - Performance assessment and mission completion prediction
    - Progress-based reward shaping for reinforcement learning optimization
    """

    name = "MissionContextSpace"
    requires = {"dist_angle_to_goal": DistanceAngleMetrics}

    def __init__(
        self,
        max_mission_time: float = 300.0,  # 5 minutes
        progress_smoothing: float = 0.9,
        phase_distance_thresholds: list = None,
        *args,
        **kwargs
    ):
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

    def reset(self) -> None:
        """Reset episode-local state."""
        self.mission_start_time = None
        self.initial_distance = None
        self.smoothed_progress = 0.0
        self.step_count = 0

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for mission context."""
        # [progress, time_normalized, mission_phase, urgency]
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
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
            self.progress_smoothing * self.smoothed_progress
            + (1.0 - self.progress_smoothing) * raw_progress
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
        expected_progress = (
            time_normalized * 0.8
        )  # Expected 80% progress by end of time

        if progress < expected_progress:
            urgency = min(1.0, (expected_progress - progress) * 2.0)
        else:
            urgency = 0.0

        return urgency

    def encode_observation(
        self, dist_angle_to_goal: DistanceAngleMetrics, *args, **kwargs
    ) -> MissionContextVector:
        """Encode comprehensive mission context with progress tracking and temporal awareness.

        Processes goal-relative distance information through sophisticated mission tracking algorithms
        to produce normalized context vectors capturing progress, temporal constraints, mission phases,
        and urgency metrics for enhanced navigation intelligence.

        Args:
            dist_angle_to_goal (DistanceAngleMetrics): Distance and angle measurements to navigation goal
                - Shape: (2,) - [distance, angle]
                - Units: [meters, radians]
                - Source: navigation planner with goal management system
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Coordinate Frame: robot-relative measurements in local frame
                - Temporal: current goal-relative measurements at observation time
                - Accuracy: depends on localization and goal management quality
                - Example: [2.5, 0.785] (2.5m away, 45° to the right of current heading)
                - Update Rate: typically matches navigation planning frequency (1-10 Hz)

        Returns:
            MissionContextVector: Comprehensive mission state representation with temporal context.
                - Shape: (4,) - fixed dimensionality for consistent learning
                - Dtype: np.float32
                - Elements: [progress, time_normalized, mission_phase, urgency]
                - Units: [normalized, normalized, normalized, normalized]
                - Range: all elements ∈ [0,1] for bounded learning space
                - Progress: exponentially smoothed progress based on distance reduction from initial
                - Time: normalized elapsed time relative to maximum mission duration
                - Phase: categorical mission phase based on distance thresholds
                - Urgency: computed urgency based on progress rate versus time constraints
                - Temporal Consistency: maintains state across observation cycles for smooth tracking
                - Example: [0.6, 0.3, 0.8, 0.2] (60% progress, 30% time elapsed, final phase, low urgency)
        """
        goal_distance = float(dist_angle_to_goal[0])

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


@SpaceFactory.register(auto_name=True, category=SpaceCategory.META)
class PerformanceContextSpace(BaseObservationSpace):
    """Advanced performance assessment system with comprehensive efficiency and smoothness analysis.

    This space provides sophisticated performance monitoring capabilities by tracking path efficiency,
    energy consumption patterns, and motion smoothness metrics over configurable temporal windows.
    It transforms goal-distance and velocity data into normalized performance vectors that enable
    comprehensive navigation quality assessment and optimization feedback.

    Key Features:
    - Multi-dimensional path efficiency tracking with goal distance reduction analysis
    - Energy efficiency monitoring based on velocity usage patterns and optimization
    - Motion smoothness assessment through acceleration variance and consistency metrics
    - Temporal windowing for robust statistical performance evaluation
    - Overall performance scoring with weighted combination of efficiency metrics
    - Real-time performance optimization with adaptive threshold management

    Mathematical Processing:
    - Path efficiency: beneficial_steps / total_steps where beneficial = distance reduction
    - Energy efficiency: 1 - (avg_velocity_usage / max_velocity) with normalization
    - Smoothness: 1 - (avg_acceleration_magnitude / max_expected_acceleration)
    - Overall performance: weighted_sum([path_eff, energy_eff, smoothness]) with configurable weights

    Use Cases:
    - Navigation performance monitoring and optimization feedback systems
    - Adaptive behavior assessment with efficiency-based reward shaping
    - Motion quality evaluation for trajectory planning and control systems
    - Performance benchmarking and comparative analysis across navigation strategies
    """

    name = "PerformanceContextSpace"
    requires = {
        "dist_angle_to_goal": DistanceAngleMetrics,
        "last_action": RobotActionVector,
    }

    def __init__(
        self, efficiency_window: int = 50, max_velocity: float = 2.0, *args, **kwargs
    ):
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

    def reset(self) -> None:
        """Reset episode-local state."""
        self.distance_history = []
        self.velocity_history = []
        self.cumulative_distance_traveled = 0.0
        self.last_position = None

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for performance context."""
        # [path_efficiency, energy_efficiency, smoothness, overall_performance]
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
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
            max_expected_acceleration = (
                self.max_velocity * 0.5
            )  # Assume reasonable acceleration

            smoothness = 1.0 - min(1.0, avg_acceleration / max_expected_acceleration)
        else:
            smoothness = 1.0

        return smoothness

    def _compute_overall_performance(
        self, path_eff: float, energy_eff: float, smoothness: float
    ) -> float:
        """Compute overall performance score."""
        # Weighted combination of metrics
        weights = [0.5, 0.3, 0.2]  # Path efficiency most important
        overall = (
            weights[0] * path_eff + weights[1] * energy_eff + weights[2] * smoothness
        )

        return overall

    def encode_observation(
        self,
        dist_angle_to_goal: DistanceAngleMetrics,
        last_action: RobotActionVector,
        *args,
        **kwargs
    ) -> PerformanceContextVector:
        """Encode comprehensive performance context with efficiency analysis and smoothness assessment.

        Processes goal-relative distance and robot action data through sophisticated performance tracking
        algorithms to produce normalized performance vectors capturing path efficiency, energy consumption,
        motion smoothness, and overall navigation quality metrics.

        Args:
            dist_angle_to_goal (DistanceAngleMetrics): Distance and angle measurements to navigation goal
                - Shape: (2,) - [distance, angle]
                - Units: [meters, radians]
                - Source: navigation planner with goal management system
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [2.5, 0.785] (2.5m away, efficiency based on distance reduction)

            last_action (RobotActionVector): Most recent robot action command vector
                - Shape: (2,) or (3,) depending on robot kinematics
                - Units: [m/s, rad/s] for differential drive or [m/s, m/s, rad/s] for holonomic
                - Source: robot action controller for velocity and smoothness analysis
                - Constraints: velocities within robot physical limits
                - Example: [0.5, 0.2] (velocity data for energy and smoothness metrics)

        Returns:
            PerformanceContextVector: Comprehensive performance assessment with efficiency metrics.
                - Shape: (4,) - fixed dimensionality for consistent learning
                - Dtype: np.float32
                - Elements: [path_efficiency, energy_efficiency, smoothness, overall_performance]
                - Units: [normalized, normalized, normalized, normalized]
                - Range: all elements ∈ [0,1] for bounded performance space
                - Path Efficiency: ratio of beneficial steps (distance reduction) to total steps
                - Energy Efficiency: inverse relationship to average velocity usage
                - Smoothness: inverse relationship to acceleration variance and motion consistency
                - Overall Performance: weighted combination of efficiency metrics with configurable weights
                - Temporal Window: computed over configurable history length for robust statistics
                - Example: [0.85, 0.7, 0.9, 0.82] (high path efficiency, good energy use, smooth motion)
        """
        goal_distance = float(dist_angle_to_goal[0])
        velocity = (float(last_action[0]), float(last_action[-1]))

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


@SpaceFactory.register(auto_name=True, category=SpaceCategory.META)
class SafetyContextSpace(BaseObservationSpace):
    """Advanced safety context observation space with collision risk assessment and violation tracking.

    This space provides comprehensive safety monitoring through real-time collision risk analysis,
    obstacle proximity assessment, and safety violation tracking. Designed for safety-critical
    navigation applications where collision avoidance is paramount.

    Technical Specifications:
    - Collision Risk Assessment: Multi-modal risk computation using LiDAR proximity and velocity analysis
    - Safety Margin Tracking: Continuous monitoring of safety buffer zones around robot
    - Violation History: Statistical tracking of safety violations for learning and adaptation
    - Risk Aggregation: Temporal smoothing of risk metrics for stable policy learning
    - Velocity-aware Safety: Dynamic risk adjustment based on robot motion characteristics

    Configuration:
    - safety_distance: Primary safety buffer distance threshold (meters)
    - critical_distance: Emergency collision threshold (meters)
    - risk_history_length: Temporal window for risk statistics (steps)

    Output Format: 4-dimensional normalized safety vector with collision risk, average risk,
    violation rate, and safety margin metrics for comprehensive safety assessment.

    Applications: Collision avoidance training, safety-aware navigation, risk assessment,
    emergency behavior learning, and safe exploration in unknown environments.
    """

    name = "SafetyContextSpace"
    requires = {
        "front_laser": LidarRanges,  # LiDAR sensor data for obstacle detection and proximity analysis
        "last_action": RobotActionVector,  # Robot odometry for velocity-based risk adjustment
    }

    def __init__(
        self,
        safety_distance: float = 1.0,
        critical_distance: float = 0.3,
        risk_history_length: int = 20,
        *args,
        **kwargs
    ):
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

    def reset(self) -> None:
        """Reset episode-local state."""
        self.risk_history = []
        self.violation_count = 0
        self.total_steps = 0

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for safety context."""
        # [current_risk, avg_risk, violation_rate, safety_margin]
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
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

    def _compute_safety_metrics(self, current_risk: float):
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
            "avg_risk": avg_risk,
            "violation_rate": violation_rate,
            "safety_margin": safety_margin,
        }

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, front_laser: LidarRanges, last_action: RobotActionVector, *args, **kwargs
    ) -> SafetyContextVector:
        """Encode comprehensive safety context with collision risk assessment and violation tracking.

        Processes LiDAR sensor data and robot odometry through sophisticated safety algorithms
        to produce normalized safety vectors capturing collision risk, safety margins, violation
        statistics, and velocity-aware risk assessment for safe navigation.

        Args:
            front_laser (LidarRanges): LiDAR sensor measurements for obstacle detection
                - Shape: (n_rays,) - typically 360 or 720 laser readings
                - Units: [meters] - distance measurements to obstacles
                - Source: LiDAR sensor with configurable angular resolution
                - Constraints: values ≥ 0, invalid readings marked as inf or negative
                - Temporal: current sensor frame for real-time collision assessment
                - Example: [3.2, 2.1, inf, 1.5, ...] (obstacle distances at various angles)

            odom (RobotActionVector): Robot odometry data for velocity-based risk adjustment
                - Shape: (3,) - [linear_x, linear_y, angular_z] velocity components
                - Units: [m/s, m/s, rad/s] - translational and rotational velocities
                - Source: robot odometry system with wheel encoders or visual odometry
                - Constraints: velocities within robot physical limits
                - Temporal: current velocity state for dynamic risk computation
                - Example: [0.5, 0.0, 0.3] (forward motion with slight rotation)

        Returns:
            SafetyContextVector: Comprehensive safety assessment with risk metrics and violation tracking.
                - Shape: (4,) - fixed dimensionality for consistent learning
                - Dtype: np.float32
                - Elements: [current_risk, avg_risk, violation_rate, safety_margin]
                - Units: [normalized, normalized, rate, normalized]
                - Range: all elements ∈ [0,1] for bounded safety space
                - Current Risk: immediate collision danger based on proximity and velocity
                - Average Risk: temporal smoothing over configurable history window
                - Violation Rate: frequency of safety threshold violations for learning
                - Safety Margin: available safety buffer (inverse of average risk)
                - Risk Computation: velocity-adjusted distance assessment with critical thresholds
                - Example: [0.2, 0.15, 0.05, 0.85] (low current risk, good safety history)
        """
        velocity = (float(last_action[0]), float(last_action[-1]))

        # Compute current collision risk
        current_risk = self._compute_collision_risk(front_laser, velocity)

        # Update tracking
        self._update_safety_tracking(current_risk)

        # Compute metrics
        safety_metrics = self._compute_safety_metrics(current_risk)

        return np.array(
            [
                current_risk,
                safety_metrics["avg_risk"],
                safety_metrics["violation_rate"],
                safety_metrics["safety_margin"],
            ],
            dtype=np.float32,
        )
