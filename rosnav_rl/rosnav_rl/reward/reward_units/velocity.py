"""Velocity-shaping reward units: movement, reversing, and velocity-change penalties."""

import random
from typing import Any, Callable, Dict

import numpy as np

from rosnav_rl.utils.observation_types import (
    DistanceAngleMetrics,
    PedestrianRelativeLocations,
    PedestrianRelativeVelocities,
    RobotActionVector,
)
from rosnav_rl.cfg.parameters import AgentParameters

from ..constants import DEFAULTS, REWARD_CONSTANTS
from ..reward_function import RewardFunction
from ..utils import check_params
from .base_reward_units import RewardUnit
from .reward_unit_factory import RewardUnitFactory


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
