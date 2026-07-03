import random
from typing import Any, Callable, Dict
from warnings import warn
import numpy as np

from rosnav_rl.observations.utils.types import (
    DistanceAngleMetrics,
    GoalRelativePosition,
    LidarRanges,
    Pose2D,
    PedestrianRelativeLocations,
    PedestrianRelativeVelocities,
    PedestrianTypeMinDistances,
    RobotActionVector,
    SafetyStatus,
    SubgoalRelativePosition,
)
from rosnav_rl.cfg.parameters import AgentParameters

from ..constants import DEFAULTS, DONE_REASONS, REWARD_CONSTANTS
from ..reward_function import RewardFunction
from ..utils import check_params
from .base_reward_units import RewardUnit
from .reward_unit_factory import RewardUnitFactory

# UPDATE WHEN ADDING A NEW UNIT
__all__ = [
    "RewardGoalReached",
    "RewardSafeDistance",
    "RewardNoMovement",
    "RewardApproachGoal",
    "RewardCollision",
    "RewardDistanceTravelled",
    "RewardReverseDrive",
    "RewardAbruptVelocityChange",
    "RewardRootVelocityDifference",
    "RewardTwoFactorVelocityDifference",
    "RewardActiveHeadingDirection",
]


@RewardUnitFactory.register("goal_reached")
class RewardGoalReached(RewardUnit):
    """
    Reward unit for goal achievement detection and reward assignment.

    Provides positive reward when the robot reaches within the goal radius,
    and manages episode termination with success state information.
    """

    requires = {
        "dist_angle_to_goal": DistanceAngleMetrics,
        "dist_angle_to_subgoal": DistanceAngleMetrics,
        "simulation_state_container": AgentParameters,
    }

    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.SUCCESS,
        "is_success": True,
    }
    NOT_DONE_INFO = {"is_done": False}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.GOAL_REACHED.REWARD,
        _follow_subgoal: bool = False,
        _on_safe_dist_violation: bool = DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when the goal is reached.

        Args:
            reward_function (RewardFunction): The reward function object holding this unit.
            reward (float, optional): The reward value for reaching the goal.
                Defaults to DEFAULTS.GOAL_REACHED.REWARD.
            following_subgoal (bool, optional): Whether to check subgoal instead of main goal.
                Defaults to False.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation
                of safe distance. Defaults to DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward
        self._follow_subgoal = _follow_subgoal

    def check_parameters(self, *args, **kwargs):
        if self._reward < 0.0:
            warn_msg = (
                f"Reconsider this reward. "
                f"Negative rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        dist_angle_to_goal: GoalRelativePosition,
        dist_angle_to_subgoal: SubgoalRelativePosition,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculates the reward and updates the information when the goal is reached.

        Args:
            goal_in_robot_frame: Distance and angle to goal in robot frame.
            subgoal_in_robot_frame: Distance and angle to subgoal in robot frame.
            simulation_state_container: Container with task configuration.
        """
        # Choose which target to check based on _following_subgoal flag
        target_distance = (
            dist_angle_to_subgoal[0] if self._follow_subgoal else dist_angle_to_goal[0]
        )

        if target_distance < simulation_state_container.goal_radius:
            self.add_reward(self._reward)
            self.add_info(self.DONE_INFO)
        else:
            self.add_info(self.NOT_DONE_INFO)


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


@RewardUnitFactory.register("no_movement")
class RewardNoMovement(RewardUnit):
    """Reward unit for detecting robot inactivity and encouraging active movement behavior.

    Applies negative reward when the robot's linear velocity is below a minimum threshold,
    promoting exploration and task completion. Supports dynamic activation based on safety
    conditions and provides configurable sensitivity to motion detection.

    Technical Specifications:
    - Motion Detection: Linear velocity threshold-based activity monitoring
    - Safety Integration: Optional deactivation during safe distance violations
    - Tolerance Configuration: Adjustable sensitivity for micro-movements

    Configuration:
    - reward: Negative reward value for inactivity (default: -0.1)
    - _on_safe_dist_violation: Enable/disable during safety violations

    Output Behavior: Applies reward when linear velocity ≤ tolerance threshold.

    Applications: Exploration encouragement, deadlock prevention, and active navigation.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.NO_MOVEMENT.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Initialize no movement reward unit with motion detection parameters.

        Args:
            reward_function: The reward function object managing this unit
            reward: Negative reward value for inactivity detection (default: -0.1)
            _on_safe_dist_violation: Enable reward during safety violations (default: False)
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
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
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate reward for robot inactivity based on linear velocity threshold.

        Monitors robot's linear velocity and applies negative reward when movement
        is below the configured tolerance threshold, encouraging active behavior.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [0.01, 0.0, 0.0] (low linear velocity triggering reward)
        """
        # Check linear velocity magnitude against tolerance threshold
        linear_velocity_magnitude = abs(last_action[0])

        if linear_velocity_magnitude <= REWARD_CONSTANTS.NO_MOVEMENT_TOLERANCE:
            self.add_reward(self._reward)


@RewardUnitFactory.register("approach_goal")
class RewardApproachGoal(RewardUnit):
    """Reward unit for goal approach behavior with distance-based reward computation.

    Provides positive reward for approaching the goal and negative reward for moving away,
    creating a distance-based potential field that guides the robot toward successful
    task completion. Supports both goal and subgoal following with adaptive thresholds.

    Technical Specifications:
    - Distance-Based Reward: Potential field approach using position differences
    - Goal Switching Support: Configurable goal vs subgoal following
    - Adaptive Thresholds: Goal update detection to prevent reward noise

    Configuration:
    - pos_factor: Positive reward scaling for goal approach
    - neg_factor: Negative reward scaling for goal retreat (should be > pos_factor)
    - _goal_update_threshold: Minimum distance for detecting goal changes

    Output Behavior: reward = factor * (last_distance - current_distance)

    Applications: Goal-directed navigation, potential field guidance, and progress tracking.
    """

    requires = {
        "robot_pose": Pose2D,
        "goal_in_robot_frame": GoalRelativePosition,
        "subgoal_in_robot_frame": SubgoalRelativePosition,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        pos_factor: float = DEFAULTS.APPROACH_GOAL.POS_FACTOR,
        neg_factor: float = DEFAULTS.APPROACH_GOAL.NEG_FACTOR,
        _potential_based: bool = False,
        factor: float = DEFAULTS.APPROACH_GOAL.FACTOR,
        gamma: float = DEFAULTS.APPROACH_GOAL.GAMMA,
        _goal_update_threshold: float = DEFAULTS.APPROACH_GOAL._GOAL_UPDATE_THRESHOLD,
        _follow_subgoal: bool = False,
        _on_safe_dist_violation: bool = DEFAULTS.APPROACH_GOAL._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Initialize goal approach reward unit with distance-based parameters.

        Args:
            reward_function: The reward function object managing this unit
            pos_factor: Positive scaling factor for goal approach (default: 0.1)
            neg_factor: Negative scaling factor for goal retreat (default: 0.2)
            _potential_based: use potential-based reward shaping (PBRS, Ng et al. 1999)
                instead of the asymmetric pos/neg factor form. reward = factor *
                (last_distance - gamma * current_distance), i.e. F = gamma*Phi(s') -
                Phi(s) with Phi(s) = -distance_to_goal. Policy-invariant for any factor
                or gamma value, so it needs no pos/neg asymmetry to resist reward hacking.
            factor: symmetric scaling factor used when _potential_based is True
            gamma: discount used in the potential difference when _potential_based is
                True; must match the RL algorithm's discount factor for the PBRS
                policy-invariance guarantee to hold
            _goal_update_threshold: Minimum distance for goal change detection
            _follow_subgoal: Use subgoal instead of main goal for tracking
            _on_safe_dist_violation: Enable reward during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._pos_factor = pos_factor
        self._neg_factor = neg_factor
        self._potential_based = _potential_based
        self._factor = factor
        self._gamma = gamma
        self._goal_update_threshold_sq = _goal_update_threshold**2
        self._follow_subgoal = _follow_subgoal

        self.last_robot_pose = None
        self.last_goal_distance = None
        self._last_goal_world: np.ndarray = None  # world-frame target for jump detection

    def check_parameters(self, *args, **kwargs):
        if self._potential_based:
            return
        if self._pos_factor < 0 or self._neg_factor < 0:
            warn_msg = (
                f"Both factors should be positive. "
                f"Current values: [pos_factor={self._pos_factor}], "
                f"[neg_factor={self._neg_factor}]"
            )
            self._report_warning(warn_msg)
        if self._pos_factor >= self._neg_factor:
            warn_msg = (
                "'pos_factor' should be smaller than 'neg_factor' otherwise "
                "rotary trajectories will get rewarded. "
                f"Current values: [pos_factor={self._pos_factor}], "
                f"[neg_factor={self._neg_factor}]"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        robot_pose: Pose2D,
        goal_in_robot_frame: GoalRelativePosition,
        subgoal_in_robot_frame: SubgoalRelativePosition,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate reward based on goal approach behavior and distance changes.

        Computes potential field-based reward by comparing current and previous
        distances to the target, encouraging goal approach and penalizing retreat.

        Args:
            robot_pose (Pose2D): Robot pose in world coordinates
                - Shape: (3,)
                - Units: [meters, meters, radians]
                - Source: odometry or SLAM
                - Constraints: theta ∈ [-π, π]
                - Example: [2.5, 1.2, 0.785] (used for distance computation)

            goal_in_robot_frame (GoalRelativePosition): Main goal position in robot frame
                - Shape: (2,)
                - Units: meters
                - Constraints: x: forward/backward, y: left/right from robot
                - Example: [3.5, -1.2] (goal position relative to robot)

            subgoal_in_robot_frame (SubgoalRelativePosition): Subgoal position in robot frame
                - Shape: (2,)
                - Units: meters
                - Constraints: x: forward/backward, y: left/right from robot
                - Example: [1.5, 0.8] (intermediate waypoint relative to robot)
        """
        # Choose target based on configuration
        target_relative = (
            subgoal_in_robot_frame if self._follow_subgoal else goal_in_robot_frame
        )

        # Calculate current distance to target
        current_distance = np.sqrt(target_relative[0] ** 2 + target_relative[1] ** 2)

        # Reconstruct target position in world frame for goal-jump detection.
        # robot_pose is a structured array with fields "x", "y", "yaw".
        yaw = float(robot_pose["yaw"])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        current_goal_world = np.array([
            float(robot_pose["x"]) + cos_yaw * target_relative[0] - sin_yaw * target_relative[1],
            float(robot_pose["y"]) + sin_yaw * target_relative[0] + cos_yaw * target_relative[1],
        ], dtype=np.float64)

        # Detect goal/subgoal reassignment: if the target jumped further than the
        # threshold in world space, the generator updated it → skip this step's
        # reward to avoid a spurious bonus/penalty from the discontinuity.
        goal_jumped = (
            self._last_goal_world is not None
            and np.sum((current_goal_world - self._last_goal_world) ** 2)
            > self._goal_update_threshold_sq
        )

        # Apply reward only when we have a valid previous distance and the goal
        # has NOT just been reassigned.
        if self.last_goal_distance is not None and not goal_jumped:
            if self._potential_based:
                # F(s,a,s') = gamma*Phi(s') - Phi(s), Phi(s) = -distance_to_goal
                shaped = self.last_goal_distance - self._gamma * current_distance
                self.add_reward(self._factor * shaped)
            else:
                distance_change = self.last_goal_distance - current_distance
                factor = self._pos_factor if distance_change > 0 else self._neg_factor
                self.add_reward(factor * distance_change)

        # Update tracking variables
        self.last_robot_pose = robot_pose.copy()
        self.last_goal_distance = current_distance
        self._last_goal_world = current_goal_world

    def reset(self):
        """Reset internal state for new episode."""
        self.last_robot_pose = None
        self.last_goal_distance = None
        self._last_goal_world = None


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


@RewardUnitFactory.register("distance_travelled")
class RewardDistanceTravelled(RewardUnit):
    """Reward unit for distance traveled calculation with velocity-based energy consumption.

    Provides reward proportional to the robot's movement, encouraging efficient navigation
    while penalizing excessive energy consumption. Supports separate scaling for linear
    and angular velocities to balance speed and rotational behavior.

    Technical Specifications:
    - Energy Consumption Model: Velocity-scaled negative reward
    - Velocity Separation: Independent linear and angular scaling factors
    - Consumption Factor: Overall energy efficiency scaling

    Configuration:
    - consumption_factor: Overall energy consumption penalty scaling
    - lin_vel_scalar: Linear velocity importance weight
    - ang_vel_scalar: Angular velocity importance weight

    Output Behavior: reward = -factor * (linear_scaled + angular_scaled)

    Applications: Energy-efficient navigation, velocity regulation, and movement encouragement.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        consumption_factor: float = DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR,
        lin_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR,
        ang_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR,
        _on_safe_dist_violation: bool = DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Initialize distance traveled reward unit with velocity scaling parameters.

        Args:
            reward_function: The reward function object managing this unit
            consumption_factor: Overall energy consumption penalty factor (default: 0.01)
            lin_vel_scalar: Linear velocity scaling weight (default: 1.0)
            ang_vel_scalar: Angular velocity scaling weight (default: 0.1)
            _on_safe_dist_violation: Enable reward during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._factor = consumption_factor
        self._lin_vel_scalar = lin_vel_scalar
        self._ang_vel_scalar = ang_vel_scalar

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate reward based on robot movement with energy consumption modeling.

        Computes energy-based reward proportional to robot velocity, encouraging efficient
        movement while penalizing excessive energy consumption through configurable scaling.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [0.5, 0.0, 0.2] (forward motion with rotation)
        """
        # Extract velocity components
        linear_velocity = last_action[0]  # Forward/backward velocity
        angular_velocity = last_action[-1]  # Rotational velocity

        # Calculate scaled energy consumption (use abs to avoid rewarding reverse/CW spin)
        linear_energy = abs(linear_velocity) * self._lin_vel_scalar
        angular_energy = abs(angular_velocity) * self._ang_vel_scalar

        # Apply negative consumption factor (encouraging efficient movement)
        total_energy_reward = -(linear_energy + angular_energy) * self._factor

        self.add_reward(total_energy_reward)


@RewardUnitFactory.register("reverse_drive")
class RewardReverseDrive(RewardUnit):
    """Reward unit for penalizing reverse driving behavior to encourage forward navigation.

    Applies negative reward when the robot drives in reverse (negative linear velocity),
    discouraging backward movement and promoting efficient forward navigation patterns.
    Supports configurable velocity thresholds for fine-tuned reverse detection.

    Technical Specifications:
    - Reverse Detection: Negative linear velocity threshold monitoring
    - Threshold Control: Configurable sensitivity for reverse movement detection
    - Safety Integration: Optional activation during safe distance violations

    Configuration:
    - reward: Negative reward value for reverse driving (default: -0.1)
    - threshold: Velocity threshold for reverse detection (default: 0.0)
    - _on_safe_dist_violation: Enable penalty during safety violations

    Output Behavior: Applies reward when linear velocity < threshold < 0

    Applications: Forward navigation encouragement, behavior shaping, and exploration guidance.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.REVERSE_DRIVE.REWARD,
        threshold: float = None,
        _on_safe_dist_violation: bool = DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize reverse drive penalty unit with velocity threshold parameters.

        Args:
            reward_function: The reward function object managing this unit
            reward: Negative reward value for reverse driving (default: -0.1)
            threshold: Velocity threshold for reverse detection (default: 0.0)
            _on_safe_dist_violation: Enable penalty during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._reward = reward
        self._threshold = threshold if threshold else 0.0

    def check_parameters(self, *args, **kwargs):
        """Validate reward parameters and issue warnings for positive values."""
        if self._reward > 0.0:
            warn_msg = (
                f"Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply penalty for reverse driving based on linear velocity analysis.

        Monitors robot's linear velocity and applies negative reward when reverse
        movement is detected below the configured threshold.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [-0.2, 0.0, 0.1] (reverse motion triggering penalty)
        """
        linear_velocity = last_action[0]

        # Apply penalty if driving in reverse below threshold
        if linear_velocity < 0 and linear_velocity < self._threshold:
            self.add_reward(self._reward)


@RewardUnitFactory.register("factored_reverse_drive")
class RewardFactoredReverseDrive(RewardUnit):
    """Reward unit for factored reverse driving penalty with velocity-proportional scaling.

    Applies velocity-proportional negative reward for reverse driving, where the penalty
    scales with the magnitude of reverse velocity. This creates stronger discouragement
    for faster reverse movements while allowing fine-grained control.

    Technical Specifications:
    - Factored Penalty: Reward scales with reverse velocity magnitude
    - Threshold Control: Configurable sensitivity for reverse detection
    - Proportional Response: Stronger penalty for higher reverse speeds

    Configuration:
    - factor: Scaling factor for velocity-proportional penalty (default: -0.1)
    - threshold: Velocity threshold for reverse detection (default: 0.0)
    - _on_safe_dist_violation: Enable penalty during safety violations

    Output Behavior: reward = factor * linear_velocity (when reverse)

    Applications: Smooth velocity control, gradual behavior shaping, and nuanced movement penalties.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        factor: float = -0.1,
        threshold: float = None,
        _on_safe_dist_violation: bool = DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize factored reverse drive penalty unit with velocity scaling parameters.

        Args:
            reward_function: The reward function object managing this unit
            factor: Velocity scaling factor for proportional penalty (default: -0.1)
            threshold: Velocity threshold for reverse detection (default: 0.0)
            _on_safe_dist_violation: Enable penalty during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._factor = factor
        self._threshold = threshold if threshold else 0.0

    def check_parameters(self, *args, **kwargs):
        """Validate factor parameters and issue warnings for positive values."""
        if self._factor > 0.0:
            warn_msg = (
                f"Reconsider this factor. "
                f"Positive factors may lead to unfavorable behaviors. "
                f"Current value: {self._factor}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply velocity-proportional penalty for reverse driving behavior.

        Computes penalty that scales with the magnitude of reverse velocity,
        providing stronger discouragement for faster reverse movements.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [-0.3, 0.0, 0.0] (moderate reverse speed with proportional penalty)
        """
        linear_velocity = last_action[0]

        # Apply velocity-proportional penalty for reverse movement
        # Use abs so that factor=-0.1 * abs(velocity) is always negative (a penalty)
        if linear_velocity < 0 and linear_velocity < self._threshold:
            self.add_reward(self._factor * abs(linear_velocity))


@RewardUnitFactory.register("abrupt_velocity_change")
class RewardAbruptVelocityChange(RewardUnit):
    """Reward unit for penalizing abrupt velocity changes to encourage smooth navigation.

    Detects sudden changes in robot velocity between consecutive actions and applies
    negative rewards based on the magnitude of change. Promotes smooth, continuous
    movement patterns while penalizing jerky or erratic behavior.

    Technical Specifications:
    - Velocity Smoothness: Multi-dimensional velocity change detection
    - Dimensional Weighting: Configurable penalties per velocity dimension
    - Quartic Penalty: Non-linear penalty scaling (change^4) for strong smoothness

    Configuration:
    - vel_factors: Dictionary mapping dimension indices to penalty factors
    - _on_safe_dist_violation: Enable penalty during safety violations

    Output Behavior: reward = -sum((velocity_diff^4 / 100) * factor) per dimension

    Applications: Smooth motion control, jerk reduction, and stable navigation patterns.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        vel_factors: Dict[str, float] = DEFAULTS.ABRUPT_VEL_CHANGE.VEL_FACTORS,
        _on_safe_dist_violation: bool = DEFAULTS.ABRUPT_VEL_CHANGE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize abrupt velocity change penalty unit with dimensional weighting.

        Args:
            reward_function: The reward function object managing this unit
            vel_factors: Dictionary mapping dimension indices to penalty factors
            _on_safe_dist_violation: Enable penalty during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._vel_factors = vel_factors
        self.last_action = None

        self._vel_change_fcts = self._get_vel_change_fcts()

    def _get_vel_change_fcts(self):
        """Create velocity change penalty functions for each configured dimension."""
        return [
            self._prepare_reward_function(int(idx), factor)
            for idx, factor in self._vel_factors.items()
        ]

    def _prepare_reward_function(
        self, idx: int, factor: float
    ) -> Callable[[np.ndarray], None]:
        """Prepare dimension-specific velocity change penalty function.

        Args:
            idx: Velocity dimension index (0=linear_x, 1=linear_y, 2=angular_z)
            factor: Penalty scaling factor for this dimension

        Returns:
            Callable that applies quartic penalty for velocity changes
        """

        def vel_change_fct(action: np.ndarray):
            assert isinstance(self.last_action, np.ndarray)
            vel_diff = abs(action[idx] - self.last_action[idx])
            self.add_reward(-((vel_diff**4 / 100) * factor))

        return vel_change_fct

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate penalty for abrupt velocity changes across configured dimensions.

        Compares current velocity with previous action to detect abrupt changes,
        applying quartic penalties to encourage smooth, continuous motion.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [0.4, 0.0, 0.3] (compared with previous action for smoothness)
        """
        if self.last_action is not None:
            for reward_function in self._vel_change_fcts:
                reward_function(last_action)

        # Update tracking state
        self.last_action = last_action.copy()

    def reset(self):
        """Reset internal state for new episode."""
        self.last_action = None


@RewardUnitFactory.register("root_velocity_difference")
class RewardRootVelocityDifference(RewardUnit):
    """Reward unit for promoting consistent root velocity differences.

    Calculates the L2 norm of velocity differences between consecutive actions
    and provides positive rewards when velocity changes remain below a threshold,
    encouraging smooth and consistent motion patterns.

    Technical Specifications:
    - Root Velocity Metric: L2 norm of squared velocity differences
    - Threshold-Based Reward: Positive rewards for consistent motion under threshold
    - Consistency Promotion: Encourages predictable velocity command patterns

    Configuration:
    - k: Threshold value for velocity difference normalization
    - _on_safe_dist_violation: Enable reward during safety violations

    Output Behavior: reward = (1 - vel_diff) / k when vel_diff < k, else 0

    Applications: Smooth motion control, velocity consistency, and stable navigation.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        k: float = DEFAULTS.ROOT_VEL_DIFF.K,
        _on_safe_dist_violation: bool = DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize root velocity difference reward unit with threshold.

        Args:
            reward_function: The reward function object managing this unit
            k: Threshold value for velocity difference normalization
            _on_safe_dist_violation: Enable reward during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._k = k
        self.last_action = None

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate reward for consistent root velocity differences.

        Computes L2 norm of velocity changes and provides positive reward when
        changes remain below threshold, promoting smooth motion consistency.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [0.4, 0.0, 0.3] (evaluated for L2 norm consistency)
        """
        if self.last_action is not None:
            # Calculate L2 norm of velocity differences
            vel_diff = np.linalg.norm(last_action - self.last_action)

            # Provide positive reward for consistent motion under threshold
            if vel_diff < self._k:
                consistency_reward = (1 - vel_diff) / self._k
                self.add_reward(consistency_reward)

        # Update tracking state
        self.last_action = last_action.copy()

    def reset(self):
        """Reset internal state for new episode."""
        self.last_action = None


@RewardUnitFactory.register("two_factor_velocity_difference")
class RewardTwoFactorVelocityDifference(RewardUnit):
    """Reward unit for penalizing two-factor velocity differences with dimensional weighting.

    Calculates absolute differences between consecutive velocity commands and applies
    dimensional penalties using alpha (first dimension) and beta (last dimension)
    weights to encourage smooth control transitions.

    Technical Specifications:
    - Two-Factor Penalty: Separate weighting for first and last velocity dimensions
    - Absolute Difference: Direct penalty on velocity command changes
    - Dimensional Weighting: Configurable penalties for linear.x and angular.z

    Configuration:
    - alpha: Weight for absolute difference in first dimension (linear.x)
    - beta: Weight for absolute difference in last dimension (angular.z)
    - _on_safe_dist_violation: Enable penalty during safety violations

    Output Behavior: reward = -(diff[0] * alpha + diff[2] * beta)

    Applications: Smooth control transitions, dimensional penalty weighting, and stability.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        alpha: float = DEFAULTS.TWO_FACTOR_VEL_DIFF.ALPHA,
        beta: float = DEFAULTS.TWO_FACTOR_VEL_DIFF.BETA,
        _on_safe_dist_violation: bool = DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize two-factor velocity difference penalty unit.

        Args:
            reward_function: The reward function object managing this unit
            alpha: Weight for absolute difference in first dimension (linear.x)
            beta: Weight for absolute difference in last dimension (angular.z)
            _on_safe_dist_violation: Enable penalty during safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._alpha = alpha
        self._beta = beta
        self.last_action = None

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate penalty for two-factor velocity differences with dimensional weights.

        Computes absolute differences between current and previous velocity commands,
        applying separate penalties for first and last dimensions.

        Args:
            last_action (RobotActionVector): Robot velocity command in base frame
                - Shape: (3,)
                - Units: [meters/second, meters/second, radians/second]
                - Source: robot controller or action
                - Constraints: linear.x, linear.y, angular.z
                - Example: [0.4, 0.0, 0.3] (penalized for absolute changes in x and z)
        """
        if self.last_action is not None:
            # Calculate absolute differences for dimensional penalty
            diff = abs(last_action - self.last_action)

            # Apply two-factor penalty with dimensional weighting
            velocity_penalty = -(diff[0] * self._alpha + diff[-1] * self._beta)
            self.add_reward(velocity_penalty)

        # Update tracking state
        self.last_action = last_action.copy()

    def reset(self):
        """Reset internal state for new episode."""
        self.last_action = None


@RewardUnitFactory.register("active_heading_direction")
class RewardActiveHeadingDirection(RewardUnit):
    """Reward unit for active heading direction optimization with velocity obstacle avoidance.

    Calculates optimal heading direction considering goal orientation and pedestrian velocity
    obstacles (VO). Uses iterative sampling to find collision-free paths that minimize
    deviation from the desired goal direction while respecting pedestrian motion constraints.

    Technical Specifications:
    - Velocity Obstacle (VO) Avoidance: Considers pedestrian motion for trajectory planning
    - Iterative Theta Sampling: Finds optimal collision-free heading directions
    - Distance-Based Activation: Only applies VO constraints within specified pedestrian distance

    Configuration:
    - r_angle: Weight for angular deviation from desired goal direction
    - theta_m: Maximum allowable deviation threshold for penalty calculation
    - theta_min: Minimum deviation initialization value for iterative search
    - ped_min_dist: Distance threshold for pedestrian VO constraint activation
    - iters: Number of random sampling iterations for collision-free direction search

    Output Behavior: reward = r_angle × (theta_m - |deviation_from_optimal_direction|)

    Applications: Social navigation, velocity obstacle avoidance, and goal-directed motion planning.
    """

    requires = {
        "dist_angle_to_goal": DistanceAngleMetrics,
        "dist_angle_to_subgoal": DistanceAngleMetrics,
        "last_action": RobotActionVector,
        "pedestrian_relative_locations": PedestrianRelativeLocations,
        "pedestrian_relative_velocities": PedestrianRelativeVelocities,
        "simulation_state_container": AgentParameters,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        r_angle: float = 0.6,
        theta_m: float = np.pi / 6,
        theta_min: int = 1000,
        ped_min_dist: float = 8.0,
        iters: int = 60,
        _following_subgoal: bool = False,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize active heading direction reward unit.

        Args:
            reward_function: The reward function object managing this unit
            r_angle: Weight for angular deviation from desired goal direction
            theta_m: Maximum allowable deviation threshold for penalty calculation
            theta_min: Minimum deviation initialization for iterative search
            ped_min_dist: Distance threshold for pedestrian VO constraint activation
            iters: Number of random sampling iterations for collision-free direction search
            _following_subgoal: Whether to use subgoal instead of goal for direction calculation
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._r_angle = r_angle
        self._theta_m = theta_m
        self._theta_min = theta_min
        self._ped_min_dist = ped_min_dist
        self._iters = iters
        self._following_subgoal = _following_subgoal

    def __call__(
        self,
        dist_angle_to_goal: DistanceAngleMetrics,
        dist_angle_to_subgoal: DistanceAngleMetrics,
        last_action: RobotActionVector,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        pedestrian_relative_velocities: PedestrianRelativeVelocities,
        simulation_state_container: AgentParameters,
        *args,
        **kwargs,
    ) -> None:
        """Calculate reward based on active heading direction with velocity obstacle avoidance.

        Computes optimal heading direction considering goal orientation and pedestrian velocity
        obstacles. Uses iterative sampling to find collision-free directions that minimize
        deviation from desired goal while avoiding pedestrian motion conflicts.

        Args:
            dist_angle_to_goal (DistanceAngleMetrics): Distance and angle to navigation goal
                - Format: [distance, angle] in meters and radians
                - Units: [meters, radians]
                - Source: goal tracking system
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [5.2, 0.75] (5.2m away, 0.75 rad to the right)
            dist_angle_to_subgoal (DistanceAngleMetrics): Distance and angle to intermediate waypoint
                - Format: [distance, angle] in meters and radians
                - Units: [meters, radians]
                - Source: subgoal tracking system
                - Constraints: distance ≥ 0, angle ∈ [-π, π]
                - Example: [2.1, -0.3] (2.1m away, 0.3 rad to the left)
            last_action (RobotActionVector): Robot's most recent action command
                - Format: Tuple[float, float, float] for linear_x, linear_y, angular_z
                - Units: [m/s, m/s, rad/s]
                - Source: action execution system
                - Constraints: Forward velocity used for VO calculations
                - Example: (0.8, 0.0, 0.2) (forward motion for collision prediction)
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian positions in robot frame
                - Format: Array of [x, y] positions in robot reference frame
                - Units: meters
                - Source: pedestrian tracking system
                - Constraints: N×2 array, negative x = behind robot
                - Example: [[2.0, 1.5], [-1.0, 0.5]] (pedestrian positions for VO calculation)
            pedestrian_relative_velocities (PedestrianRelativeVelocities): Pedestrian velocities in robot frame
                - Format: Array of [vx, vy] velocities in robot reference frame
                - Units: meters/second
                - Source: pedestrian tracking system
                - Constraints: N×2 array matching pedestrian_relative_locations
                - Example: [[0.5, 0.2], [-0.3, 0.8]] (pedestrian velocities for VO prediction)
            simulation_state_container (AgentParameters): Robot and environment state
                - Contains: robot configuration, dimensions, and simulation parameters
                - Used for: robot radius in velocity obstacle calculations
        """
        # Select goal based on subgoal following preference
        dist_angle_to_target = (
            dist_angle_to_subgoal if self._following_subgoal else dist_angle_to_goal
        )

        if (
            pedestrian_relative_locations is None
            or pedestrian_relative_velocities is None
            or last_action is None
        ):
            return

        # Extract goal direction and robot forward velocity
        theta_pre = dist_angle_to_target[1]
        d_theta = theta_pre
        v_x = last_action[0]

        # Apply velocity obstacle avoidance if pedestrians present
        if len(pedestrian_relative_locations) > 0:
            d_theta = np.pi / 2  # Initialize with perpendicular direction
            theta_min = self._theta_min

            # Iterative sampling for collision-free direction
            for _ in range(self._iters):
                theta = random.uniform(-np.pi, np.pi)
                free = True

                # Check collision with each pedestrian using velocity obstacles
                for i, (ped_location, ped_velocity) in enumerate(
                    zip(pedestrian_relative_locations, pedestrian_relative_velocities)
                ):
                    p_x, p_y = ped_location[0], ped_location[1]
                    p_vx, p_vy = ped_velocity[0], ped_velocity[1]

                    ped_dis = np.linalg.norm([p_x, p_y])

                    # Apply VO constraints only within minimum distance
                    if ped_dis <= self._ped_min_dist:
                        ped_theta = np.arctan2(p_y, p_x)

                        # Calculate VO cone using robot radius estimation
                        vector = (
                            ped_dis**2
                            - (3 * simulation_state_container.robot_radius) ** 2
                        )
                        if vector < 0:
                            continue  # Robot too close to pedestrian, skip

                        vo_theta = np.arctan2(
                            3 * simulation_state_container.robot_radius,
                            np.sqrt(vector),
                        )
                        # Check if trajectory intersects with pedestrian's VO cone
                        theta_rp = np.arctan2(
                            v_x * np.sin(theta) - p_vy, v_x * np.cos(theta) - p_vx
                        )
                        if (ped_theta - vo_theta) <= theta_rp <= (ped_theta + vo_theta):
                            free = False
                            break

                # Update best direction if collision-free and closer to goal
                if free:
                    theta_diff = (theta - theta_pre) ** 2
                    if theta_diff < theta_min:
                        theta_min = theta_diff
                        d_theta = theta
        else:
            # No obstacles: use direct goal direction
            d_theta = theta_pre

        self.add_reward(self._r_angle * (self._theta_m - abs(d_theta)))

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("ped_type_safety_distance")
class RewardPedTypeSafetyDistance(RewardUnit):
    """Reward unit for pedestrian type-specific safety distance violation detection.

    Applies negative rewards when the robot violates minimum safety distance to specific
    pedestrian types or groups. Supports flexible type-reward mapping for different
    pedestrian behaviors and prioritized safety enforcement.

    Technical Specifications:
    - Type-Specific Safety: Different safety thresholds per pedestrian type/group
    - Distance Monitoring: Continuous distance tracking to pedestrian groups
    - Flexible Configuration: Dictionary-based type-reward mapping or single type mode

    Configuration:
    - type_reward_pairs: Dictionary mapping pedestrian types to reward values
    - ped_type: Single pedestrian type to monitor (fallback if no pairs provided)
    - reward: Default reward value for single type mode
    - safety_distance: Distance threshold for safety violation detection
    - _on_safe_dist_violation: Enable/disable during general safety violations

    Output Behavior: Applies type-specific reward when distance < safety_distance

    Applications: Social navigation, type-aware safety, and pedestrian behavior adaptation.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        type_reward_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.REWARD,
        safety_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific safety distance reward unit.

        Args:
            reward_function: The reward function object managing this unit
            type_reward_pairs: Dictionary mapping pedestrian types to reward values
            ped_type: Single pedestrian type to monitor (fallback mode)
            reward: Default reward value for single type mode
            safety_distance: Distance threshold for safety violation detection
            _on_safe_dist_violation: Enable/disable during general safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._reward = reward
        self._safety_distance = safety_distance
        self._type_reward_pairs = (
            type_reward_pairs
            if isinstance(type_reward_pairs, dict)
            else {ped_type: reward}
        )

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply type-specific safety distance violation penalties.

        Monitors minimum distances to pedestrian groups and applies configured
        penalties when safety thresholds are violated.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (type-specific distance monitoring)
        """
        if not pedestrian_distances:
            self._report_warning(f"No pedestrian type distances found.")
            return

        for ped_type, reward in self._type_reward_pairs.items():
            if ped_type not in pedestrian_distances:
                self._report_warning(f"Pedestrian type {ped_type} not found.")
                continue

            if pedestrian_distances[ped_type] < self._safety_distance:
                self.add_reward(reward)

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("ped_type_factored_safety_distance")
class RewardPedTypeFactoredSafetyDistance(RewardUnit):
    """Reward unit for pedestrian type-specific factored safety distance violations.

    Applies proportional negative rewards based on safety distance violations to specific
    pedestrian types. The penalty magnitude scales with the severity of the violation,
    providing fine-grained feedback for social navigation training.

    Technical Specifications:
    - Factored Distance Penalty: Reward = factor * (safety_distance - actual_distance)
    - Type-Specific Factors: Different scaling factors per pedestrian type/group
    - Proportional Feedback: Penalty magnitude reflects violation severity

    Configuration:
    - type_factor_pairs: Dictionary mapping pedestrian types to scaling factors
    - ped_type: Single pedestrian type to monitor (fallback if no pairs provided)
    - factor: Default scaling factor for single type mode
    - safety_distance: Distance threshold for safety violation detection
    - _on_safe_dist_violation: Enable/disable during general safety violations

    Output Behavior: reward = factor * (safety_distance - distance) when distance < safety_distance

    Applications: Smooth social navigation, fine-grained safety training, and type-aware penalty scaling.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        type_factor_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.TYPE,
        factor: float = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.FACTOR,
        safety_distance: float = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific factored safety distance reward unit.

        Args:
            reward_function: The reward function object managing this unit
            type_factor_pairs: Dictionary mapping pedestrian types to scaling factors
            ped_type: Single pedestrian type to monitor (fallback mode)
            factor: Default scaling factor for single type mode
            safety_distance: Distance threshold for safety violation detection
            _on_safe_dist_violation: Enable/disable during general safety violations
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._factor = factor
        self._safety_distance = safety_distance
        self._type_factor_pairs = (
            type_factor_pairs
            if isinstance(type_factor_pairs, dict)
            else {ped_type: factor}
        )

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply type-specific factored safety distance violation penalties.

        Calculates proportional penalties based on safety distance violations,
        with penalty magnitude reflecting violation severity and type-specific scaling.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (factored scaling per type)
        """
        if not pedestrian_distances:
            self._report_warning(
                "Won't apply reward unit. No pedestrian type distances found."
            )
            return

        for ped_type, factor in self._type_factor_pairs.items():
            if ped_type not in pedestrian_distances:
                self._report_warning(f"Pedestrian type {ped_type} not found.")
                continue

            # Apply proportional penalty based on safety distance violation
            if pedestrian_distances[ped_type] < self._safety_distance:
                violation_magnitude = (
                    self._safety_distance - pedestrian_distances[ped_type]
                )
                factored_penalty = factor * violation_magnitude
                self.add_reward(factored_penalty)

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("ped_type_collision")
class RewardPedTypeCollision(RewardUnit):
    """Reward unit for pedestrian type-specific collision detection and penalty application.

    Detects robot collisions with specific pedestrian types using proximity-based collision
    detection with configurable bumper zones. Applies significant negative rewards for
    type-specific collisions to promote safe social navigation behaviors.

    Technical Specifications:
    - Collision Detection: Distance-based collision with bumper zone consideration
    - Type-Specific Penalties: Different rewards per pedestrian type/group
    - Robot Radius Integration: Accounts for robot physical dimensions in collision detection

    Configuration:
    - type_reward_pairs: Dictionary mapping pedestrian types to collision penalties
    - ped_type: Single pedestrian type to monitor (fallback if no pairs provided)
    - reward: Default collision penalty for single type mode
    - bumper_zone: Additional collision buffer beyond robot radius

    Output Behavior: Applies type-specific penalty when distance ≤ (bumper_zone + robot_radius)

    Applications: Social safety enforcement, type-aware collision avoidance, and penalty differentiation.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
        "simulation_state_container": AgentParameters,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        type_reward_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.REWARD,
        bumper_zone: float = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.BUMPER_ZONE,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific collision detection reward unit.

        Args:
            reward_function: The reward function object managing this unit
            type_reward_pairs: Dictionary mapping pedestrian types to collision penalties
            ped_type: Single pedestrian type to monitor (fallback mode)
            reward: Default collision penalty for single type mode
            bumper_zone: Additional collision buffer beyond robot radius
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._type_reward_pairs = (
            type_reward_pairs
            if isinstance(type_reward_pairs, dict)
            else {ped_type: reward}
        )
        self._bumper_zone = bumper_zone

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Detect and penalize type-specific pedestrian collisions.

        Monitors minimum distances to pedestrian groups and applies collision penalties
        when robots collide with specific pedestrian types within the bumper zone.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (collision detection per type)
            simulation_state_container (AgentParameters): Robot and environment state
                - Contains: robot configuration, dimensions, and simulation parameters
                - Used for: robot radius in collision detection calculations
        """
        if not pedestrian_distances:
            self._report_warning(
                "Won't apply reward unit. No pedestrian type distances found."
            )
            return

        # Calculate collision threshold including robot dimensions
        collision_threshold = (
            self._bumper_zone + simulation_state_container.robot_radius
        )

        for ped_type, reward in self._type_reward_pairs.items():
            if ped_type not in pedestrian_distances:
                self._report_warning(f"Pedestrian type {ped_type} not found.")
                continue

            # Apply collision penalty if pedestrian within collision zone
            if pedestrian_distances[ped_type] <= collision_threshold:
                self.add_reward(reward)

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("ped_type_vel_constraint")
class RewardPedTypeVelocityConstraint(RewardUnit):
    """Reward unit for pedestrian type-specific velocity constraints and speed regulation.

    Monitors robot linear velocity in proximity to specific pedestrian types, applying
    velocity-proportional penalties to encourage speed reduction in social spaces.
    Promotes socially-aware navigation by constraining robot speed near pedestrians.

    Technical Specifications:
    - Velocity-Proportional Penalties: Reward scales with robot forward velocity
    - Type-Specific Activation: Only applies constraints near specified pedestrian types
    - Distance-Based Activation: Constraint activates within specified proximity distance

    Configuration:
    - ped_type: Specific pedestrian type to monitor for velocity constraints
    - penalty_factor: Velocity penalty scaling factor (negative for penalties)
    - active_distance: Distance threshold for constraint activation

    Output Behavior: Applies penalty = -penalty_factor × linear_velocity when pedestrian within active_distance

    Applications: Social speed regulation, pedestrian comfort zones, and velocity-aware navigation.
    """

    requires = {
        "pedestrian_distances": PedestrianTypeMinDistances,
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        penalty_factor: float = 0.05,
        active_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize pedestrian type-specific velocity constraint reward unit.

        Args:
            reward_function: The reward function object managing this unit
            ped_type: Specific pedestrian type to monitor for velocity constraints
            penalty_factor: Velocity penalty scaling factor (negative for penalties)
            active_distance: Distance threshold for constraint activation
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._penalty_factor = penalty_factor
        self._active_distance = active_distance

    def __call__(
        self,
        pedestrian_distances: PedestrianTypeMinDistances,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply velocity constraints when robot is near specific pedestrian types.

        Monitors robot linear velocity and applies proportional penalties when traveling
        too fast in proximity to specific pedestrian types within the active distance.

        Args:
            pedestrian_distances (PedestrianTypeMinDistances): Min distances to pedestrian groups
                - Format: Dict[Union[str, int], float]
                - Units: meters
                - Source: pedestrian distance tracking system
                - Constraints: Keys are group IDs, values are minimum distances ≥ 0
                - Example: {1: 2.5, 2: 4.1, 3: 1.8} (velocity constraints per type)
            last_action (ActionState): Robot's most recent action command
                - Format: Tuple[float, float, float] or similar action representation
                - Units: [m/s, m/s, rad/s] for linear_x, linear_y, angular_z
                - Source: action execution system
                - Constraints: action[0] represents forward linear velocity
                - Example: (0.5, 0.0, 0.2) (forward velocity used for penalty scaling)
        """
        if not pedestrian_distances:
            self._report_warning(
                "Won't apply reward unit. No pedestrian type distances found."
            )
            return

        if last_action is None:
            self._report_warning("Won't apply reward unit. No last action found.")
            return

        if self._type not in pedestrian_distances:
            self._report_warning(f"Pedestrian type {self._type} not found.")
            return

        # Apply velocity penalty when pedestrian within active distance
        if pedestrian_distances[self._type] < self._active_distance:
            self.add_reward(-self._penalty_factor * last_action[0])

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("angular_vel_constraint")
class RewardAngularVelocityConstraint(RewardUnit):
    """Reward unit for angular velocity constraints and rotational speed regulation.

    Monitors robot angular velocity and applies threshold-based penalties to discourage
    excessive rotational speeds. Promotes smooth navigation by constraining turning rates
    while maintaining directional control capabilities.

    Technical Specifications:
    - Threshold-Based Penalties: Only applies penalties when angular velocity exceeds threshold
    - Velocity-Proportional Scaling: Penalty magnitude scales with excess angular velocity
    - Rotational Speed Regulation: Encourages controlled turning behaviors

    Configuration:
    - penalty_factor: Angular velocity penalty scaling factor (typically negative)
    - threshold: Maximum allowed angular velocity before penalties apply

    Output Behavior: Applies penalty = penalty_factor × angular_velocity when |angular_velocity| > threshold

    Applications: Smooth motion control, turning speed regulation, and stability enhancement.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        penalty_factor: float = -0.05,
        threshold: float = None,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize angular velocity constraint reward unit.

        Args:
            reward_function: The reward function object managing this unit
            penalty_factor: Angular velocity penalty scaling factor (typically negative)
            threshold: Maximum allowed angular velocity before penalties apply
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._penalty_factor = penalty_factor
        self._threshold = threshold

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply angular velocity constraints and penalties for excessive rotation.

        Monitors robot angular velocity from the most recent action and applies
        threshold-based penalties when rotational speed exceeds configured limits.

        Args:
            last_action (RobotActionVector): Robot's most recent action command
                - Format: Tuple[float, float, float] or similar action representation
                - Units: [m/s, m/s, rad/s] for linear_x, linear_y, angular_z
                - Source: action execution system
                - Constraints: action[2] represents angular velocity in rad/s
                - Example: (0.5, 0.0, 0.8) (angular velocity used for constraint checking)
        """
        if last_action is None:
            return

        angular = abs(last_action[-1])

        # Apply penalty only if threshold is set and angular velocity exceeds it
        if self._threshold is not None and angular > self._threshold:
            self.add_reward(self._penalty_factor * angular)

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("max_steps_exceeded")
class RewardMaxStepsExceeded(RewardUnit):
    """
    A reward unit that penalizes the agent when the maximum number of steps is exceeded.

    Args:
        reward_function (RewardFunction): The reward function to which this unit belongs.
        penalty (float, optional): The penalty value to be applied when the maximum steps are exceeded. Defaults to 10.
        _on_safe_dist_violation (bool, optional): Whether to apply the penalty on safe distance violation. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _penalty (float): The penalty value to be applied when the maximum steps are exceeded.
        _steps (int): The current number of steps taken.

    Methods:
        __call__(*args, **kwargs): Updates the step count and applies the penalty if the maximum steps are exceeded.
        reset(): Resets the step count to zero.
    """

    requires = {
        "simulation_state_container": AgentParameters,
    }

    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.STEP_LIMIT,
        "is_success": 0,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        penalty: float = 10,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._penalty = penalty
        self._steps = 0

    def check_parameters(self, *args, **kwargs):
        if self._penalty < 0.0:
            warn_msg = (
                f"Reconsider this reward. "
                f"The penalty should be a positive value as it is going to be subtracted from the total reward."
                f"Current value: {self._penalty}"
            )
            self._report_warning(warn_msg)

    def __call__(
        self,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates the step count and applies the penalty if the maximum steps are exceeded.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._steps += 1
        if self._steps >= simulation_state_container.max_steps:
            self.add_reward(-self._penalty)
            self.add_info(self.DONE_INFO)

    def reset(self):
        """
        Resets the step count to zero.
        """
        self._steps = 0


@RewardUnitFactory.register("linear_vel_boost")
class RewardLinearVelBoost(RewardUnit):
    """Reward unit for linear velocity enhancement and forward speed encouragement.

    Monitors robot linear velocity and applies threshold-based rewards to encourage
    forward movement and efficient navigation. Promotes faster goal-directed motion
    by rewarding higher forward speeds above specified velocity thresholds.

    Technical Specifications:
    - Threshold-Based Rewards: Only applies rewards when linear velocity exceeds threshold
    - Velocity-Proportional Scaling: Reward magnitude scales with excess linear velocity
    - Forward Speed Enhancement: Encourages efficient forward motion behaviors

    Configuration:
    - reward_factor: Linear velocity reward scaling factor (typically positive for boosts)
    - threshold: Minimum linear velocity required before rewards apply

    Output Behavior: Applies reward = reward_factor × linear_velocity when linear_velocity > threshold

    Applications: Speed optimization, efficient navigation, and goal approach acceleration.
    """

    requires = {
        "last_action": RobotActionVector,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward_factor: float = 0.05,
        threshold: float = None,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize linear velocity boost reward unit.

        Args:
            reward_function: The reward function object managing this unit
            reward_factor: Linear velocity reward scaling factor (positive for speed boost)
            threshold: Minimum linear velocity required before rewards apply
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward_factor = reward_factor
        self._threshold = threshold

    def __call__(
        self,
        last_action: RobotActionVector,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply linear velocity boosts and rewards for efficient forward motion.

        Monitors robot linear velocity from the most recent action and applies
        threshold-based rewards when forward speed exceeds configured minimum levels.

        Args:
            last_action (RobotActionVector): Robot's most recent action command
                - Format: Tuple[float, float, float] or similar action representation
                - Units: [m/s, m/s, rad/s] for linear_x, linear_y, angular_z
                - Source: action execution system
                - Constraints: action[0] represents forward linear velocity in m/s
                - Example: (0.8, 0.0, 0.2) (linear velocity used for boost calculation)
        """
        if last_action is None:
            return

        linear = last_action[0]

        # Apply reward only if threshold is set and linear velocity exceeds it
        if self._threshold is not None and linear > self._threshold:
            self.add_reward(self._reward_factor * linear)

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("proxemic_intrusion")
class RewardProxemicIntrusion(RewardUnit):
    """Reward unit for asymmetric, heading-aware pedestrian personal-space intrusion.

    Penalizes the robot for entering a pedestrian's Gaussian comfort zone, with the zone
    stretched further ahead of the pedestrian's direction of travel than behind it (Kirby/SARL
    proxemics). This gives the policy a smooth gradient to route around a pedestrian's future
    path rather than just its current position, complementing the isotropic safety-distance
    units above.

    Technical Specifications:
    - Heading-Frame Rotation: Robot position relative to each pedestrian is rotated into that
      pedestrian's own heading frame (derived from its relative velocity).
    - Asymmetric Gaussian: Separate along-heading sigma ahead (sigma_front) vs. behind
      (sigma_back) the pedestrian; lateral spread uses sigma_side.
    - Stationary Fallback: Pedestrians with negligible speed use sigma_side for the along-axis
      too, since their heading is undefined.

    Configuration:
    - weight: Penalty scale applied to the summed per-pedestrian intrusion.
    - sigma_front / sigma_back / sigma_side: Comfort-zone spreads (meters) ahead, behind, and
      to the side of a pedestrian's heading.
    - activation_radius: Pedestrians farther than this (meters) are skipped entirely.
    - min_ped_speed: Speed (m/s) below which a pedestrian is treated as stationary.

    Output Behavior: reward = -weight * sum_i exp(-0.5 * [(d_along_i / sigma_i)^2 +
    (d_perp_i / sigma_side)^2]), sigma_i = sigma_front if d_along_i > 0 else sigma_back.

    Applications: Social navigation, proxemic comfort, path-anticipatory pedestrian avoidance.
    """

    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
        "pedestrian_relative_velocities": PedestrianRelativeVelocities,
        "simulation_state_container": AgentParameters,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        weight: float = 0.1,
        sigma_front: float = 1.2,
        sigma_back: float = 0.4,
        sigma_side: float = 0.6,
        activation_radius: float = 3.0,
        min_ped_speed: float = 0.15,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the proxemic intrusion reward unit.

        Args:
            reward_function: The reward function object holding this unit
            weight: Penalty scale applied to the summed per-pedestrian intrusion
            sigma_front: Comfort-zone spread (meters) ahead of a pedestrian's heading
            sigma_back: Comfort-zone spread (meters) behind a pedestrian's heading
            sigma_side: Comfort-zone spread (meters) lateral to a pedestrian's heading
            activation_radius: Pedestrians farther than this (meters) are skipped
            min_ped_speed: Speed (m/s) below which a pedestrian is treated as stationary
            _on_safe_dist_violation: Whether to apply penalty on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._weight = weight
        self._sigma_front = sigma_front
        self._sigma_back = sigma_back
        self._sigma_side = sigma_side
        self._activation_radius = activation_radius
        self._min_ped_speed = min_ped_speed

    def __call__(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        pedestrian_relative_velocities: PedestrianRelativeVelocities,
        simulation_state_container: AgentParameters,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Penalize the robot for intruding into pedestrians' asymmetric comfort zones.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian positions
                in the robot frame
                - Format: N×2 array of [x, y] positions in meters
                - Source: pedestrian tracking system
                - Example: [[2.0, 1.5], [-1.0, 0.5]]
            pedestrian_relative_velocities (PedestrianRelativeVelocities): Pedestrian velocities
                in the robot frame
                - Format: N×2 array of [vx, vy] velocities in meters/second
                - Source: pedestrian tracking system
                - Constraints: N×2 array matching pedestrian_relative_locations
                - Example: [[0.5, 0.2], [-0.3, 0.8]]
            simulation_state_container (AgentParameters): Robot and environment state
                - Unused here; part of the common schema surface for future robot-frame needs
        """
        if (
            pedestrian_relative_locations is None
            or pedestrian_relative_velocities is None
            or len(pedestrian_relative_locations) == 0
        ):
            return

        intrusion = 0.0
        for ped_location, ped_velocity in zip(
            pedestrian_relative_locations, pedestrian_relative_velocities
        ):
            px, py = ped_location[0], ped_location[1]
            dist = np.hypot(px, py)
            if dist > self._activation_radius:
                continue

            speed = np.hypot(ped_velocity[0], ped_velocity[1])
            if speed >= self._min_ped_speed:
                heading = np.arctan2(ped_velocity[1], ped_velocity[0])
            else:
                heading = None

            # Vector from pedestrian to robot, in robot-frame coordinates.
            rx, ry = -px, -py

            if heading is not None:
                cos_h, sin_h = np.cos(heading), np.sin(heading)
                d_along = rx * cos_h + ry * sin_h
                d_perp = -rx * sin_h + ry * cos_h
                sigma_along = self._sigma_front if d_along > 0 else self._sigma_back
            else:
                d_along, d_perp = rx, ry
                sigma_along = self._sigma_side

            intrusion += np.exp(
                -0.5
                * (
                    (d_along / sigma_along) ** 2
                    + (d_perp / self._sigma_side) ** 2
                )
            )

        if intrusion > 0.0:
            self.add_reward(-self._weight * intrusion)

    def reset(self):
        """Reset internal state for new episode."""
        pass


@RewardUnitFactory.register("social_potential")
class RewardSocialPotential(RewardUnit):
    """Reward unit for potential-based shaping toward crowd separation.

    Provides a dense, policy-invariant gradient (PBRS, Ng et al. 1999) that rewards
    increasing separation from the nearest pedestrian, complementing the cumulative
    comfort-zone cost of `proxemic_intrusion` with a telescoping gradient that helps
    the policy reach the low-cost region faster in imagination.

    Technical Specifications:
    - Potential: Phi(s) = min(nearest_ped_distance, clip_distance). Sign convention
      is the OPPOSITE of `approach_goal` (Phi = -distance_to_goal there): here Phi
      increases with pedestrian *separation*, since the objective is to move away
      from crowds rather than toward a goal. Do not "fix" this sign later.
    - Shaping: F = factor * (gamma * Phi(s') - Phi(s)). Per Ng et al. 1999 this
      leaves the optimal policy invariant for any factor/gamma — note this holds
      for the *optimal* policy, not per-step: with gamma < 1 a constant potential
      still yields a fixed per-step drift factor*(gamma-1)*Phi rather than exactly
      zero (same property already present in `approach_goal`'s PBRS branch).
    - Discontinuity Guard: skips shaping when the potential jumps by more than
      jump_threshold in one step (nearest-pedestrian identity switch, or a
      pedestrian entering/leaving clip_distance), avoiding spurious spikes.
    - Degenerate Cases: no pedestrians, or nearest pedestrian beyond clip_distance,
      both clip Phi(s) to clip_distance — identical, distance-independent drift in
      either case, so a far/absent pedestrian never adds a live gradient signal.

    Configuration:
    - factor: shaping scale.
    - gamma: discount used in the potential difference; must match the RL
      algorithm's discount factor for the PBRS policy-invariance guarantee.
    - clip_distance: potential saturates beyond this distance (meters).
    - jump_threshold: skip shaping if |Phi(s') - Phi(s)| exceeds this (meters).

    Applications: Dense crowd-separation gradient for imagination-horizon training,
    complementing `proxemic_intrusion`'s cumulative comfort-zone cost.
    """

    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        factor: float = DEFAULTS.SOCIAL_POTENTIAL.FACTOR,
        gamma: float = DEFAULTS.SOCIAL_POTENTIAL.GAMMA,
        clip_distance: float = DEFAULTS.SOCIAL_POTENTIAL.CLIP_DISTANCE,
        jump_threshold: float = DEFAULTS.SOCIAL_POTENTIAL.JUMP_THRESHOLD,
        _on_safe_dist_violation: bool = DEFAULTS.SOCIAL_POTENTIAL._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the social potential reward unit.

        Args:
            reward_function: The reward function object holding this unit
            factor: shaping scale applied to the potential difference
            gamma: discount used in the potential difference; must match the RL
                algorithm's discount factor for the PBRS policy-invariance guarantee
            clip_distance: potential saturates beyond this distance (meters)
            jump_threshold: skip shaping this step if the potential jumps more than
                this (meters), e.g. on a nearest-pedestrian identity switch
            _on_safe_dist_violation: Whether to apply shaping on safe distance violation
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._factor = factor
        self._gamma = gamma
        self._clip_distance = clip_distance
        self._jump_threshold = jump_threshold

        self.last_phi = None

    def __call__(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Shape reward toward increasing separation from the nearest pedestrian.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): Pedestrian
                positions in the robot frame
                - Format: N×2 array of [x, y] positions in meters
                - Source: pedestrian tracking system
                - Example: [[2.0, 1.5], [-1.0, 0.5]]
        """
        if pedestrian_relative_locations is None or len(pedestrian_relative_locations) == 0:
            nearest_dist = self._clip_distance
        else:
            nearest_dist = np.min(
                np.hypot(
                    pedestrian_relative_locations[:, 0],
                    pedestrian_relative_locations[:, 1],
                )
            )

        current_phi = min(float(nearest_dist), self._clip_distance)

        if self.last_phi is not None:
            phi_jumped = abs(current_phi - self.last_phi) > self._jump_threshold
            if not phi_jumped:
                shaped = self._gamma * current_phi - self.last_phi
                self.add_reward(self._factor * shaped)

        self.last_phi = current_phi

    def reset(self):
        """Reset internal state for new episode."""
        self.last_phi = None
