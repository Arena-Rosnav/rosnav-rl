"""Goal-reaching and approach-progress reward units."""

from typing import Any

import numpy as np

from rosnav_rl.observations.utils.types import (
    DistanceAngleMetrics,
    GoalRelativePosition,
    Pose2D,
    SubgoalRelativePosition,
)
from rosnav_rl.cfg.parameters import AgentParameters

from ..constants import DEFAULTS, DONE_REASONS
from ..reward_function import RewardFunction
from ..utils import check_params
from .base_reward_units import RewardUnit
from .reward_unit_factory import RewardUnitFactory


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
