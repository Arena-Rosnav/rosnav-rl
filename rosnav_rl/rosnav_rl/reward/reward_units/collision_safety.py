"""Safe-distance and collision reward units."""

from typing import Any

import numpy as np

from rosnav_rl.observations.utils.types import (
    LidarRanges,
    SafetyStatus,
)
from rosnav_rl.cfg.parameters import AgentParameters

from ..constants import DEFAULTS, DONE_REASONS
from ..reward_function import RewardFunction
from ..utils import check_params
from .base_reward_units import RewardUnit
from .reward_unit_factory import RewardUnitFactory


@RewardUnitFactory.register("safe_distance")
class RewardSafeDistance(RewardUnit):
    """Reward unit for safe distance violation detection and collision avoidance enforcement.

    Applies negative reward when robot violates minimum safe distance to obstacles,
    as detected by laser safety monitoring systems. Promotes collision-free navigation
    by penalizing proximity violations and maintaining safe operational distances.

    Technical Specifications:
    - Safety Distance Monitoring: Laser-based proximity violation detection
    - Binary Penalty Application: Fixed negative reward per violation instance
    - Episode Information: Provides safety violation context for episode analysis

    Configuration:
    - reward: Negative reward value for safety distance violations (default: -0.1)

    Output Behavior: Applies fixed negative reward when safety distance is violated

    Applications: Collision avoidance training, safety constraint enforcement, and proximity awareness.
    """

    requires = {
        "laser_safety_violation": SafetyStatus,
    }

    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.SAFE_DISTANCE.REWARD,
        *args,
        **kwargs,
    ):
        """Initialize safe distance violation reward unit.

        Args:
            reward_function: The reward function object managing this unit
            reward: Negative reward value for safety distance violations (default: -0.1)
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        laser_safety_violation: SafetyStatus,
        *args: Any,
        **kwargs: Any,
    ):
        """Apply penalty for safe distance violations detected by laser safety monitoring.

        Monitors laser-based safety status and applies negative reward when robot
        violates minimum safe distance to obstacles in the environment.

        Args:
            laser_safety_violation (SafetyStatus): Safety check result for collision avoidance
                - Format: Boolean indicating safety distance violation
                - Units: Boolean (True = violation detected, False = safe)
                - Source: laser-based proximity monitoring
                - Constraints: True if within safety distance, False otherwise
                - Example: True (indicating robot is too close to obstacles)
        """
        if laser_safety_violation:
            self.add_reward(self._reward)
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)

@RewardUnitFactory.register("factored_safe_distance")
class RewardFactoredSafeDistance(RewardUnit):
    """Reward unit for factored safe distance violations with proportional penalty scaling.

    Applies distance-proportional negative reward when robot violates minimum safe distance,
    where penalty magnitude scales with proximity severity. Encourages maintaining safe
    margins by applying stronger penalties for closer obstacle approaches.

    Technical Specifications:
    - Proportional Penalty: Reward scales with distance violation magnitude
    - Safety Distance Integration: Uses robot safety parameters and radius
    - Laser-Based Detection: Minimum laser reading analysis for proximity measurement

    Configuration:
    - factor: Scaling factor for distance-proportional penalty (default: -0.5)

    Output Behavior: reward = factor × (safety_distance + radius - min_laser_distance)

    Applications: Smooth distance-based penalties, safety margin enforcement, and proximity awareness.
    """

    requires = {
        "laser_safety_violation": SafetyStatus,
        "front_laser": LidarRanges,
        "simulation_state_container": AgentParameters,
    }

    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        factor: float = -0.5,
        *args,
        **kwargs,
    ):
        """Initialize factored safe distance violation reward unit.

        Args:
            reward_function: The reward function object managing this unit
            factor: Scaling factor for distance-proportional penalty (default: -0.5)
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self._factor = factor
        super().__init__(reward_function, True, *args, **kwargs)

    def check_parameters(self, *args, **kwargs):
        if self._factor >= 0.0:
            warn_msg = (
                f"Reconsider this reward. "
                f"Positive factor may lead to unfavorable behaviors. "
                f"Current value: {self._factor}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        laser_safety_violation: SafetyStatus,
        front_laser: LidarRanges,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ):
        """Apply proportional penalty for safe distance violations based on proximity severity.

        Calculates distance-based penalty when safety violations occur, with stronger
        penalties for closer obstacle approaches using laser scan minimum readings.

        Args:
            laser_safety_violation (SafetyStatus): Safety check result for collision avoidance
                - Format: Boolean indicating safety distance violation
                - Units: Boolean (True = violation detected, False = safe)
                - Source: laser-based proximity monitoring
                - Constraints: True if within safety distance, False otherwise
                - Example: True (triggering proportional penalty calculation)

            front_laser (LidarRanges): Preprocessed laser scan ranges for proximity measurement
                - Shape: (n,) where n = number of laser beams
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, max_range], NaN replaced with max_range
                - Example: [0.3, 0.25, 0.4, ..., 2.1] (minimum used for penalty scaling)

            simulation_state_container (AgentParameters): Robot and environment configuration
                - Contains: robot safety distance, radius, and system parameters
                - Source: simulation environment
                - Used for: safety threshold calculation and penalty scaling
        """
        if laser_safety_violation:
            # Calculate minimum laser reading for distance-based penalty
            laser_min = np.min(front_laser)

            # Calculate proportional penalty based on safety margin violation
            safety_threshold = (
                simulation_state_container.safety_distance
                + simulation_state_container.robot_radius
            )
            distance_violation = safety_threshold - laser_min

            self.add_reward(self._factor * distance_violation)
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)

@RewardUnitFactory.register("collision")
class RewardCollision(RewardUnit):
    """Reward unit for collision detection and episode termination management.

    Detects robot collisions using both laser scan proximity analysis and collision monitor
    state, applying significant negative reward while terminating the episode. Integrates
    multiple collision detection sources for robust safety enforcement.

    Technical Specifications:
    - Dual Collision Detection: Laser scan proximity + nav2 collision monitor
    - Episode Termination: Automatic episode ending with failure state
    - Safety Integration: Configurable bumper zone for collision sensitivity

    Configuration:
    - reward: Large negative reward for collision events (default: -20.0)
    - bumper_zone: Distance threshold for collision detection (meters)

    Output Behavior: Large negative reward + episode termination on collision.

    Applications: Safety enforcement, collision avoidance training, and episode management.
    """

    requires = {
        "front_laser": LidarRanges,
        "collision_monitor": SafetyStatus,
        "simulation_state_container": AgentParameters,
    }

    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.COLLISION,
        "is_success": False,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.COLLISION.REWARD,
        bumper_zone: float = DEFAULTS.COLLISION.BUMPER_ZONE,
        *args,
        **kwargs,
    ):
        """Initialize collision detection reward unit with safety parameters.

        Args:
            reward_function: The reward function object managing this unit
            reward: Large negative reward value for collision detection (default: -20.0)
            bumper_zone: Distance threshold for collision detection in meters
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward
        self._bumper_zone = bumper_zone
        # Latch: once a collision is detected it stays True until reset() so a
        # brief collision_monitor_state pulse (True→False within one step) is
        # never silently missed.
        self._collision_latched: bool = False

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        front_laser: LidarRanges,
        simulation_state_container: AgentParameters,
        collision_monitor: SafetyStatus = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Detect collisions using laser scan data and collision monitor, applying penalties.

        Analyzes both laser scan readings and nav2 collision monitor state to detect when
        the robot is within collision zones, applying significant negative reward and
        terminating the episode. Uses dual detection for robust collision identification.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges for collision detection
                - Shape: (n,) where n = number of laser beams
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, max_range], NaN replaced with max_range
                - Example: [0.1, 0.15, 0.2, ..., 1.5] (close readings indicating collision)

            collision_monitor (SafetyStatus): Nav2 collision monitor state
                - Format: Boolean indicating collision detection by nav2
                - Units: Boolean (True = collision detected, False = clear)
                - Source: nav2 collision monitor
                - Constraints: True if collision detected, False otherwise
                - Example: True (indicating nav2 detected collision)

            simulation_state_container (AgentParameters): Container with task configuration
                - Contains: robot parameters, safety thresholds, episode state
                - Source: simulation environment
                - Used for: collision zone configuration and episode termination
        """
        # If no collision monitor signal yet, fallback to laser-based collision check
        if not collision_monitor:
            robot_radius = simulation_state_container.robot_radius
            collision_threshold = robot_radius + self._bumper_zone

            # Check for collision by finding minimum laser reading
            min_laser_distance = np.min(front_laser)
            collision_monitor = min_laser_distance <= collision_threshold

        # Latch on the first detected collision; stays True until reset() is called.
        # This prevents the common race condition where collision_monitor_state fires
        # and clears (True→False) faster than one training step is evaluated.
        if collision_monitor:
            self._collision_latched = True

        # Apply penalty and terminate episode whenever the latch is set
        if self._collision_latched:
            self.add_reward(self._reward)
            self.add_info(self.DONE_INFO)

    def reset(self) -> None:
        """Clear the collision latch at the start of each new episode."""
        self._collision_latched = False
