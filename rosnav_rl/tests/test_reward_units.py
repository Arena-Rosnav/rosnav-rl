"""Comprehensive tests for reward units.

Tests cover:
- Initialization and parameter validation
- __call__ behavior for every registered reward unit
- Reset behavior for stateful units
- Edge cases and boundary conditions
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# conftest.py ensures ROS stubs are available before importing rosnav_rl
from conftest import SimulationStateContainerStub, make_pose2d

from rosnav_rl.reward.reward_units.reward_units import (
    RewardGoalReached,
    RewardSafeDistance,
    RewardFactoredSafeDistance,
    RewardNoMovement,
    RewardApproachGoal,
    RewardCollision,
    RewardDistanceTravelled,
    RewardReverseDrive,
    RewardFactoredReverseDrive,
    RewardAbruptVelocityChange,
    RewardRootVelocityDifference,
    RewardTwoFactorVelocityDifference,
    RewardActiveHeadingDirection,
    RewardPedTypeSafetyDistance,
    RewardPedTypeFactoredSafetyDistance,
    RewardPedTypeCollision,
    RewardPedTypeVelocityConstraint,
    RewardAngularVelocityConstraint,
    RewardLinearVelBoost,
    RewardMaxStepsExceeded,
    RewardProxemicIntrusion,
    RewardSocialPotential,
    RewardTGRFDiscomfort,
)
from rosnav_rl.reward.constants import DEFAULTS, DONE_REASONS


# =====================================================================
#  Helper for extracting reward from mock
# =====================================================================

def total_reward(rf):
    """Get the total reward accumulated via add_reward calls."""
    return rf._reward_accum["total"]


def info(rf):
    """Get the info dict built up via add_info calls."""
    return rf.state.info


def reset_rf(rf):
    """Reset the mock reward function state."""
    rf._reset()


# =====================================================================
#  RewardGoalReached
# =====================================================================

class TestRewardGoalReached:
    def test_goal_reached_gives_reward_and_done(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardGoalReached(rf, reward=15.0)

        # distance < goal_radius => success
        dist_to_goal = np.array([0.1, 0.5])  # distance=0.1 < 0.3 goal_radius
        dist_to_subgoal = np.array([1.0, 0.3])

        unit(dist_angle_to_goal=dist_to_goal, dist_angle_to_subgoal=dist_to_subgoal,
             simulation_state_container=sim_state)

        assert total_reward(rf) == pytest.approx(15.0)
        assert info(rf)["is_done"] is True
        assert info(rf)["done_reason"] == DONE_REASONS.SUCCESS
        assert info(rf)["is_success"] is True

    def test_goal_not_reached(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardGoalReached(rf, reward=15.0)

        dist_to_goal = np.array([5.0, 0.5])  # far away
        dist_to_subgoal = np.array([3.0, 0.3])

        unit(dist_angle_to_goal=dist_to_goal, dist_angle_to_subgoal=dist_to_subgoal,
             simulation_state_container=sim_state)

        assert total_reward(rf) == 0.0
        assert info(rf)["is_done"] is False

    def test_follow_subgoal_mode(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardGoalReached(rf, reward=10.0, _follow_subgoal=True)

        # Goal is far, but subgoal is close
        dist_to_goal = np.array([10.0, 0.0])
        dist_to_subgoal = np.array([0.1, 0.0])  # within goal_radius

        unit(dist_angle_to_goal=dist_to_goal, dist_angle_to_subgoal=dist_to_subgoal,
             simulation_state_container=sim_state)

        assert total_reward(rf) == pytest.approx(10.0)
        assert info(rf)["is_done"] is True

    def test_exactly_at_goal_radius(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardGoalReached(rf, reward=15.0)

        # distance == goal_radius => NOT reached (strictly <)
        dist_to_goal = np.array([0.3, 0.0])
        dist_to_subgoal = np.array([1.0, 0.0])

        unit(dist_angle_to_goal=dist_to_goal, dist_angle_to_subgoal=dist_to_subgoal,
             simulation_state_container=sim_state)

        assert total_reward(rf) == 0.0
        assert info(rf)["is_done"] is False

    def test_check_parameters_warns_on_negative_reward(self, make_reward_function):
        rf = make_reward_function()
        with pytest.warns(UserWarning):
            RewardGoalReached(rf, reward=-5.0)


# =====================================================================
#  RewardSafeDistance
# =====================================================================

class TestRewardSafeDistance:
    def test_violation_gives_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSafeDistance(rf, reward=-0.15)

        unit(laser_safety_violation=True)

        assert total_reward(rf) == pytest.approx(-0.15)
        assert info(rf).get("safe_dist_violation") is True

    def test_no_violation_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSafeDistance(rf, reward=-0.15)

        unit(laser_safety_violation=False)

        assert total_reward(rf) == 0.0
        assert "safe_dist_violation" not in info(rf)

    def test_repeated_violations_accumulate(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSafeDistance(rf, reward=-0.1)

        for _ in range(3):
            unit(laser_safety_violation=True)

        assert total_reward(rf) == pytest.approx(-0.3)


# =====================================================================
#  RewardFactoredSafeDistance
# =====================================================================

class TestRewardFactoredSafeDistance:
    def test_proportional_penalty(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardFactoredSafeDistance(rf, factor=-0.5)

        # laser min=0.3, safety_distance=0.5, radius=0.3 => threshold=0.8
        # violation = 0.8 - 0.3 = 0.5; reward = -0.5 * 0.5 = -0.25
        laser = np.array([0.3, 1.0, 2.0, 1.5])

        unit(laser_safety_violation=True, front_laser=laser,
             simulation_state_container=sim_state)

        assert total_reward(rf) == pytest.approx(-0.25)

    def test_no_violation_no_penalty(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardFactoredSafeDistance(rf, factor=-0.5)

        laser = np.array([2.0, 3.0, 4.0])
        unit(laser_safety_violation=False, front_laser=laser,
             simulation_state_container=sim_state)

        assert total_reward(rf) == 0.0


# =====================================================================
#  RewardNoMovement
# =====================================================================

class TestRewardNoMovement:
    def test_stationary_robot_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardNoMovement(rf, reward=-0.01)

        action = np.array([0.0, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == pytest.approx(-0.01)

    def test_moving_robot_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardNoMovement(rf, reward=-0.01)

        action = np.array([0.5, 0.0, 0.2])
        unit(last_action=action)

        assert total_reward(rf) == 0.0

    def test_micro_movement_below_tolerance(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardNoMovement(rf, reward=-0.05)

        # Tolerance is 0.1
        action = np.array([0.05, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == pytest.approx(-0.05)

    def test_at_tolerance_boundary(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardNoMovement(rf, reward=-0.05)

        action = np.array([0.1, 0.0, 0.0])
        unit(last_action=action)

        # abs(0.1) <= 0.1 => penalized
        assert total_reward(rf) == pytest.approx(-0.05)


# =====================================================================
#  RewardApproachGoal
# =====================================================================

class TestRewardApproachGoal:
    def test_approaching_goal_positive_reward(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardApproachGoal(rf, pos_factor=0.3, neg_factor=0.5)

        # World goal fixed at (5, 0)
        # Step 1: robot at (0,0,0) => goal_in_robot_frame=(5,0), dist=5.0
        unit(robot_pose=make_pose2d(0.0, 0.0, 0.0),
             goal_in_robot_frame=np.array([5.0, 0.0]),
             subgoal_in_robot_frame=np.array([3.0, 0.0]))
        assert total_reward(rf) == 0.0  # No reward on first call

        # Step 2: robot advanced to (2,0,0) => goal_in_robot_frame=(3,0), dist=3.0
        # world goal = 2+3 = (5, 0) -- unchanged, no jump
        unit(robot_pose=make_pose2d(2.0, 0.0, 0.0),
             goal_in_robot_frame=np.array([3.0, 0.0]),
             subgoal_in_robot_frame=np.array([3.0, 0.0]))

        # distance changed from 5.0 to 3.0 => change=2.0 => reward = 0.3*2.0 = 0.6
        assert total_reward(rf) == pytest.approx(0.6)

    def test_moving_away_negative_reward(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardApproachGoal(rf, pos_factor=0.3, neg_factor=0.5)

        # World goal fixed at (3, 0)
        # Step 1: robot at (1,0,0) => goal_in_robot_frame = (2, 0), dist=2.0
        unit(robot_pose=make_pose2d(1.0, 0.0, 0.0),
             goal_in_robot_frame=np.array([2.0, 0.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))
        reset_rf(rf)

        # Step 2: robot backed up to (-2,0,0) => goal_in_robot_frame = (5, 0), dist=5.0
        # world goal = (-2)+5 = (3, 0) — unchanged, no jump
        unit(robot_pose=make_pose2d(-2.0, 0.0, 0.0),
             goal_in_robot_frame=np.array([5.0, 0.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))

        # distance changed from 2.0 to 5.0 => change=-3.0 => reward = 0.5*(-3.0) = -1.5
        assert total_reward(rf) == pytest.approx(-1.5)

    def test_reset_clears_state(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardApproachGoal(rf, pos_factor=0.3, neg_factor=0.5)

        robot_pose = make_pose2d(0.0, 0.0, 0.0)
        unit(robot_pose=robot_pose, goal_in_robot_frame=np.array([5.0, 0.0]),
             subgoal_in_robot_frame=np.array([3.0, 0.0]))

        unit.reset()

        assert unit.last_robot_pose is None
        assert unit.last_goal_distance is None
        assert unit._last_goal_world is None

    def test_follow_subgoal(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardApproachGoal(rf, pos_factor=0.3, neg_factor=0.5, _follow_subgoal=True)

        # World subgoal fixed at (5, 0)
        # Step 1: robot at (0,0,0) => subgoal_in_robot_frame = (5, 0), dist=5.0
        unit(robot_pose=make_pose2d(0.0, 0.0, 0.0),
             goal_in_robot_frame=np.array([10.0, 0.0]),
             subgoal_in_robot_frame=np.array([5.0, 0.0]))

        # Step 2: robot advanced to (3,0,0) => subgoal_in_robot_frame = (2, 0), dist=2.0
        # world subgoal = 3+2 = (5, 0) — unchanged
        unit(robot_pose=make_pose2d(3.0, 0.0, 0.0),
             goal_in_robot_frame=np.array([10.0, 0.0]),
             subgoal_in_robot_frame=np.array([2.0, 0.0]))

        # Distance: 5.0 -> 2.0, change = 3.0, reward = 0.3 * 3.0 = 0.9
        assert total_reward(rf) == pytest.approx(0.9)

    def test_diagonal_goal_distance(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardApproachGoal(rf, pos_factor=1.0, neg_factor=1.0)

        # World goal fixed at (3, 4)
        # Step 1: robot at (0,0,0) => goal_in_robot_frame = (3,4), dist=5.0
        unit(robot_pose=make_pose2d(0.0, 0.0, 0.0),
             goal_in_robot_frame=np.array([3.0, 4.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))

        # Step 2: robot at (3,4,0) => goal_in_robot_frame = (0,0), dist=0.0
        # world goal = (3+0, 4+0) = (3, 4) — unchanged
        unit(robot_pose=make_pose2d(3.0, 4.0, 0.0),
             goal_in_robot_frame=np.array([0.0, 0.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))

        # Distance: 5.0 -> 0.0, change = 5.0, reward = 1.0 * 5.0
        assert total_reward(rf) == pytest.approx(5.0)

    def test_goal_jump_skips_reward(self, make_reward_function):
        """When the goal teleports beyond the threshold, no reward is issued for that step."""
        rf = make_reward_function()
        # threshold=1.0 => threshold_sq=1.0
        unit = RewardApproachGoal(rf, pos_factor=0.3, neg_factor=0.5,
                                  _goal_update_threshold=1.0)

        # Robot at origin, facing right (yaw=0)
        robot_pose = make_pose2d(0.0, 0.0, 0.0)

        # Step 1: goal 5 m ahead in robot frame => world (5, 0)
        unit(robot_pose=robot_pose,
             goal_in_robot_frame=np.array([5.0, 0.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))
        assert total_reward(rf) == 0.0  # baseline step

        # Step 2: goal jumps to world (5, 10) — far outside threshold
        # In robot frame (yaw=0): same as world offset => (5, 10)
        unit(robot_pose=robot_pose,
             goal_in_robot_frame=np.array([5.0, 10.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))
        # Jump detected => reward must stay 0
        assert total_reward(rf) == pytest.approx(0.0)

    def test_small_goal_change_not_skipped(self, make_reward_function):
        """Small target movement within threshold should produce a normal reward."""
        rf = make_reward_function()
        unit = RewardApproachGoal(rf, pos_factor=0.3, neg_factor=0.5,
                                  _goal_update_threshold=5.0)  # generous threshold

        robot_pose = make_pose2d(0.0, 0.0, 0.0)

        # Step 1: goal 5 m ahead
        unit(robot_pose=robot_pose,
             goal_in_robot_frame=np.array([5.0, 0.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))

        # Step 2: goal 3 m ahead (robot advanced) — still within 5 m threshold
        unit(robot_pose=robot_pose,
             goal_in_robot_frame=np.array([3.0, 0.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))

        # distance 5 -> 3, change=2.0, factor=0.3 => reward=0.6
        assert total_reward(rf) == pytest.approx(0.6)

    def test_after_jump_next_step_resumes(self, make_reward_function):
        """After a skipped jump step, the following step should reward normally."""
        rf = make_reward_function()
        unit = RewardApproachGoal(rf, pos_factor=1.0, neg_factor=1.0,
                                  _goal_update_threshold=1.0)

        robot_pose = make_pose2d(0.0, 0.0, 0.0)

        # Step 1: baseline at 5 m
        unit(robot_pose=robot_pose,
             goal_in_robot_frame=np.array([5.0, 0.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))

        # Step 2: jump — skipped, but last_goal_distance now = sqrt(5^2+10^2)
        unit(robot_pose=robot_pose,
             goal_in_robot_frame=np.array([5.0, 10.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))
        assert total_reward(rf) == pytest.approx(0.0)

        # Step 3: small movement from the new reference point
        # goal in robot frame: (5, 9) => distance slightly less
        unit(robot_pose=robot_pose,
             goal_in_robot_frame=np.array([5.0, 9.0]),
             subgoal_in_robot_frame=np.array([1.0, 0.0]))
        # Should get a positive reward now (approaching from the new reference)
        assert total_reward(rf) > 0.0


# =====================================================================
#  RewardCollision
# =====================================================================

class TestRewardCollision:
    def test_collision_monitor_true_terminates(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardCollision(rf, reward=-10.0, bumper_zone=0.05)

        laser = np.array([1.0, 2.0, 3.0])
        unit(front_laser=laser, collision_monitor=True,
             simulation_state_container=sim_state)

        assert total_reward(rf) == pytest.approx(-10.0)
        assert info(rf)["is_done"] is True
        assert info(rf)["done_reason"] == DONE_REASONS.COLLISION

    def test_laser_fallback_collision(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardCollision(rf, reward=-10.0, bumper_zone=0.05)

        # radius=0.3, bumper_zone=0.05 => threshold=0.35
        # min laser = 0.2 < 0.35 => collision
        laser = np.array([0.2, 1.0, 2.0])
        unit(front_laser=laser, collision_monitor=False,
             simulation_state_container=sim_state)

        assert total_reward(rf) == pytest.approx(-10.0)
        assert info(rf)["is_done"] is True

    def test_no_collision(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardCollision(rf, reward=-10.0, bumper_zone=0.05)

        laser = np.array([1.0, 2.0, 3.0])
        unit(front_laser=laser, collision_monitor=False,
             simulation_state_container=sim_state)

        assert total_reward(rf) == 0.0
        assert "is_done" not in info(rf)

    def test_collision_monitor_none_falls_back(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardCollision(rf, reward=-10.0, bumper_zone=0.05)

        # collision_monitor=None => falls back to laser check
        # laser min = 0.1 < 0.35 => collision
        laser = np.array([0.1, 5.0])
        unit(front_laser=laser, collision_monitor=None,
             simulation_state_container=sim_state)

        assert total_reward(rf) == pytest.approx(-10.0)
        assert info(rf)["is_done"] is True


# =====================================================================
#  RewardDistanceTravelled
# =====================================================================

class TestRewardDistanceTravelled:
    def test_energy_consumption(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardDistanceTravelled(rf, consumption_factor=0.005,
                                        lin_vel_scalar=1.0, ang_vel_scalar=0.001)

        action = np.array([0.5, 0.0, 1.0])
        unit(last_action=action)

        # linear: abs(0.5)*1.0 = 0.5, angular: abs(1.0)*0.001 = 0.001
        # total: -(0.5 + 0.001) * 0.005 = -0.002505
        assert total_reward(rf) == pytest.approx(-0.002505)

    def test_zero_velocity(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardDistanceTravelled(rf)

        action = np.array([0.0, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == pytest.approx(0.0)

    def test_reverse_velocity_uses_abs(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardDistanceTravelled(rf, consumption_factor=1.0,
                                        lin_vel_scalar=1.0, ang_vel_scalar=0.0)

        action_fwd = np.array([0.5, 0.0, 0.0])
        action_rev = np.array([-0.5, 0.0, 0.0])

        unit(last_action=action_fwd)
        r_fwd = total_reward(rf)

        reset_rf(rf)
        unit(last_action=action_rev)
        r_rev = total_reward(rf)

        assert r_fwd == pytest.approx(r_rev)


# =====================================================================
#  RewardReverseDrive
# =====================================================================

class TestRewardReverseDrive:
    def test_reverse_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardReverseDrive(rf, reward=-0.01)

        action = np.array([-0.3, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == pytest.approx(-0.01)

    def test_forward_not_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardReverseDrive(rf, reward=-0.01)

        action = np.array([0.5, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == 0.0

    def test_zero_velocity_not_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardReverseDrive(rf, reward=-0.01)

        action = np.array([0.0, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == 0.0

    def test_threshold_respected(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardReverseDrive(rf, reward=-0.01, threshold=-0.5)

        # -0.3 is negative but > -0.5, so NOT below threshold
        action = np.array([-0.3, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == 0.0

        # -0.6 < -0.5 => penalty
        action2 = np.array([-0.6, 0.0, 0.0])
        unit(last_action=action2)

        assert total_reward(rf) == pytest.approx(-0.01)


# =====================================================================
#  RewardFactoredReverseDrive
# =====================================================================

class TestRewardFactoredReverseDrive:
    def test_proportional_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardFactoredReverseDrive(rf, factor=-0.1)

        action = np.array([-0.4, 0.0, 0.0])
        unit(last_action=action)

        # factor * abs(velocity) = -0.1 * 0.4 = -0.04
        assert total_reward(rf) == pytest.approx(-0.04)

    def test_forward_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardFactoredReverseDrive(rf, factor=-0.1)

        action = np.array([0.5, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == 0.0

    def test_threshold_respected(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardFactoredReverseDrive(rf, factor=-0.1, threshold=-0.2)

        action = np.array([-0.1, 0.0, 0.0])  # above threshold
        unit(last_action=action)
        assert total_reward(rf) == 0.0

        action2 = np.array([-0.3, 0.0, 0.0])  # below threshold
        unit(last_action=action2)
        assert total_reward(rf) == pytest.approx(-0.1 * 0.3)


# =====================================================================
#  RewardAbruptVelocityChange
# =====================================================================

class TestRewardAbruptVelocityChange:
    def test_first_step_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAbruptVelocityChange(rf)

        action = np.array([0.5, 0.0, 0.3])
        unit(last_action=action)

        assert total_reward(rf) == 0.0  # First step seeds state

    def test_identical_actions_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAbruptVelocityChange(rf)

        action = np.array([0.5, 0.0, 0.3])
        unit(last_action=action)
        reset_rf(rf)

        unit(last_action=action.copy())
        assert total_reward(rf) == 0.0

    def test_abrupt_change_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAbruptVelocityChange(rf,
                                           vel_factors={"0": 1.0, "2": 1.0})

        action1 = np.array([0.0, 0.0, 0.0])
        action2 = np.array([1.0, 0.0, 1.0])

        unit(last_action=action1)
        reset_rf(rf)
        unit(last_action=action2)

        # diff[0] = 1.0, diff[2] = 1.0
        # penalty_dim0 = -(1.0**4 / 100) * 1.0 = -0.01
        # penalty_dim2 = -(1.0**4 / 100) * 1.0 = -0.01
        assert total_reward(rf) == pytest.approx(-0.02)

    def test_reset_clears_state(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAbruptVelocityChange(rf)

        unit(last_action=np.array([1.0, 0.0, 0.5]))
        assert unit.last_action is not None

        unit.reset()
        assert unit.last_action is None

    def test_small_change_small_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAbruptVelocityChange(rf,
                                           vel_factors={"0": 1.0})

        unit(last_action=np.array([0.0, 0.0, 0.0]))
        reset_rf(rf)

        # Small change = 0.1; penalty = -(0.1^4 / 100) * 1.0 = -0.000001
        unit(last_action=np.array([0.1, 0.0, 0.0]))
        assert total_reward(rf) == pytest.approx(-0.000001, abs=1e-8)


# =====================================================================
#  RewardRootVelocityDifference
# =====================================================================

class TestRewardRootVelocityDifference:
    def test_first_step_no_reward(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardRootVelocityDifference(rf, k=500)

        unit(last_action=np.array([0.5, 0.0, 0.3]))
        assert total_reward(rf) == 0.0

    def test_consistent_velocity_rewarded(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardRootVelocityDifference(rf, k=500)

        action = np.array([0.5, 0.0, 0.3])
        unit(last_action=action)
        reset_rf(rf)

        # Identical velocity => diff=0 => reward = (1-0)/500 = 0.002
        unit(last_action=action.copy())
        assert total_reward(rf) == pytest.approx(1.0 / 500)

    def test_large_difference_no_reward(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardRootVelocityDifference(rf, k=0.1)

        unit(last_action=np.array([0.0, 0.0, 0.0]))
        reset_rf(rf)

        # L2 diff = sqrt(1.0 + 1.0) ~ 1.414 > 0.1
        unit(last_action=np.array([1.0, 0.0, 1.0]))
        assert total_reward(rf) == 0.0

    def test_reset_clears_state(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardRootVelocityDifference(rf)

        unit(last_action=np.array([1.0, 0.0, 0.0]))
        unit.reset()

        assert unit.last_action is None


# =====================================================================
#  RewardTwoFactorVelocityDifference
# =====================================================================

class TestRewardTwoFactorVelocityDifference:
    def test_first_step_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardTwoFactorVelocityDifference(rf, alpha=0.01, beta=0.025)

        unit(last_action=np.array([0.5, 0.0, 0.3]))
        assert total_reward(rf) == 0.0

    def test_velocity_difference_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardTwoFactorVelocityDifference(rf, alpha=1.0, beta=1.0)

        action1 = np.array([0.0, 0.0, 0.0])
        action2 = np.array([0.5, 0.0, 0.3])

        unit(last_action=action1)
        reset_rf(rf)
        unit(last_action=action2)

        # diff = abs([0.5, 0.0, 0.3]), reward = -(diff[0]*1.0 + diff[-1]*1.0) = -(0.5+0.3) = -0.8
        assert total_reward(rf) == pytest.approx(-0.8)

    def test_identical_actions_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardTwoFactorVelocityDifference(rf, alpha=1.0, beta=1.0)

        action = np.array([0.5, 0.0, 0.3])
        unit(last_action=action)
        reset_rf(rf)
        unit(last_action=action.copy())

        assert total_reward(rf) == 0.0

    def test_reset_clears_state(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardTwoFactorVelocityDifference(rf)

        unit(last_action=np.array([1.0, 0.0, 0.0]))
        unit.reset()
        assert unit.last_action is None


# =====================================================================
#  RewardAngularVelocityConstraint
# =====================================================================

class TestRewardAngularVelocityConstraint:
    def test_below_threshold_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAngularVelocityConstraint(rf, penalty_factor=-0.05, threshold=1.0)

        action = np.array([0.5, 0.0, 0.5])
        unit(last_action=action)
        assert total_reward(rf) == 0.0

    def test_above_threshold_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAngularVelocityConstraint(rf, penalty_factor=-0.05, threshold=1.0)

        action = np.array([0.5, 0.0, 1.5])
        unit(last_action=action)

        # penalty = -0.05 * 1.5 = -0.075
        assert total_reward(rf) == pytest.approx(-0.075)

    def test_no_threshold_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAngularVelocityConstraint(rf, penalty_factor=-0.05, threshold=None)

        action = np.array([0.5, 0.0, 5.0])
        unit(last_action=action)

        # No threshold => never triggers
        assert total_reward(rf) == 0.0

    def test_none_action_no_crash(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAngularVelocityConstraint(rf, penalty_factor=-0.05, threshold=1.0)
        unit(last_action=None)
        assert total_reward(rf) == 0.0

    def test_negative_angular_uses_abs(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardAngularVelocityConstraint(rf, penalty_factor=-0.05, threshold=1.0)

        action = np.array([0.0, 0.0, -1.5])
        unit(last_action=action)

        assert total_reward(rf) == pytest.approx(-0.05 * 1.5)


# =====================================================================
#  RewardLinearVelBoost
# =====================================================================

class TestRewardLinearVelBoost:
    def test_above_threshold_boosted(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardLinearVelBoost(rf, reward_factor=0.05, threshold=0.3)

        action = np.array([0.5, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == pytest.approx(0.05 * 0.5)

    def test_below_threshold_no_boost(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardLinearVelBoost(rf, reward_factor=0.05, threshold=0.3)

        action = np.array([0.2, 0.0, 0.0])
        unit(last_action=action)

        assert total_reward(rf) == 0.0

    def test_no_threshold_no_boost(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardLinearVelBoost(rf, reward_factor=0.05, threshold=None)

        action = np.array([10.0, 0.0, 0.0])
        unit(last_action=action)
        assert total_reward(rf) == 0.0

    def test_none_action_no_crash(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardLinearVelBoost(rf, reward_factor=0.05, threshold=0.3)
        unit(last_action=None)
        assert total_reward(rf) == 0.0


# =====================================================================
#  RewardMaxStepsExceeded
# =====================================================================

class TestRewardMaxStepsExceeded:
    def test_below_max_steps(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardMaxStepsExceeded(rf, penalty=10.0)

        for _ in range(10):
            unit(simulation_state_container=sim_state)

        assert total_reward(rf) == 0.0

    def test_at_max_steps_penalized(self, make_reward_function):
        rf = make_reward_function()
        sim = SimulationStateContainerStub(max_steps=5)
        unit = RewardMaxStepsExceeded(rf, penalty=10.0)

        for _ in range(5):
            unit(simulation_state_container=sim)

        assert total_reward(rf) == pytest.approx(-10.0)
        assert info(rf)["is_done"] is True
        assert info(rf)["done_reason"] == DONE_REASONS.STEP_LIMIT

    def test_reset_clears_steps(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardMaxStepsExceeded(rf, penalty=10.0)

        for _ in range(3):
            unit(simulation_state_container=sim_state)

        unit.reset()
        assert unit._steps == 0


# =====================================================================
#  RewardPedTypeSafetyDistance
# =====================================================================

class TestRewardPedTypeSafetyDistance:
    def test_violation_penalized(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeSafetyDistance(rf, ped_type=1, reward=-0.25,
                                           safety_distance=1.25)

        distances = {1: 0.5}
        unit(pedestrian_distances=distances)

        assert total_reward(rf) == pytest.approx(-0.25)

    def test_no_violation(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeSafetyDistance(rf, ped_type=1, reward=-0.25,
                                           safety_distance=1.25)

        distances = {1: 2.0}
        unit(pedestrian_distances=distances)

        assert total_reward(rf) == 0.0

    def test_empty_distances(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeSafetyDistance(rf, ped_type=1, reward=-0.25,
                                           safety_distance=1.25)

        unit(pedestrian_distances={})
        assert total_reward(rf) == 0.0

    def test_multiple_types(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeSafetyDistance(rf, type_reward_pairs={1: -0.25, 2: -0.5},
                                           safety_distance=1.0)

        distances = {1: 0.5, 2: 0.3}
        unit(pedestrian_distances=distances)

        assert total_reward(rf) == pytest.approx(-0.25 + -0.5)


# =====================================================================
#  RewardPedTypeFactoredSafetyDistance
# =====================================================================

class TestRewardPedTypeFactoredSafetyDistance:
    def test_proportional_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeFactoredSafetyDistance(rf, ped_type=1, factor=-0.5,
                                                    safety_distance=1.25)

        distances = {1: 0.75}  # violation = 1.25 - 0.75 = 0.5
        unit(pedestrian_distances=distances)

        assert total_reward(rf) == pytest.approx(-0.5 * 0.5)

    def test_no_violation(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeFactoredSafetyDistance(rf, ped_type=1, factor=-0.5,
                                                    safety_distance=1.0)

        distances = {1: 2.0}
        unit(pedestrian_distances=distances)

        assert total_reward(rf) == 0.0


# =====================================================================
#  RewardPedTypeCollision
# =====================================================================

class TestRewardPedTypeCollision:
    def test_collision_penalized(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardPedTypeCollision(rf, ped_type=1, reward=-10.0, bumper_zone=0.05)

        # collision threshold = 0.05 + 0.3 = 0.35
        distances = {1: 0.2}
        unit(pedestrian_distances=distances, simulation_state_container=sim_state)

        assert total_reward(rf) == pytest.approx(-10.0)

    def test_no_collision(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardPedTypeCollision(rf, ped_type=1, reward=-10.0, bumper_zone=0.05)

        distances = {1: 1.0}
        unit(pedestrian_distances=distances, simulation_state_container=sim_state)

        assert total_reward(rf) == 0.0


# =====================================================================
#  RewardPedTypeVelocityConstraint
# =====================================================================

class TestRewardPedTypeVelocityConstraint:
    def test_close_ped_velocity_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeVelocityConstraint(rf, ped_type=1, penalty_factor=0.05,
                                                active_distance=1.25)

        distances = {1: 0.5}
        action = np.array([0.8, 0.0, 0.0])

        unit(pedestrian_distances=distances, last_action=action)

        # penalty = -0.05 * 0.8 = -0.04
        assert total_reward(rf) == pytest.approx(-0.04)

    def test_far_ped_no_penalty(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardPedTypeVelocityConstraint(rf, ped_type=1, penalty_factor=0.05,
                                                active_distance=1.25)

        distances = {1: 5.0}
        action = np.array([0.8, 0.0, 0.0])

        unit(pedestrian_distances=distances, last_action=action)
        assert total_reward(rf) == 0.0


# =====================================================================
#  Cross-cutting concerns
# =====================================================================

class TestRewardUnitBase:
    """Test base class behaviors."""

    def test_add_reward_nan_rejected(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSafeDistance(rf, reward=-0.15)

        # NaN reward should be silently rejected
        unit.add_reward(float("nan"))
        assert total_reward(rf) == 0.0

    def test_add_reward_inf_rejected(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSafeDistance(rf, reward=-0.15)

        unit.add_reward(float("inf"))
        assert total_reward(rf) == 0.0

    def test_on_safe_dist_violation_flag(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardNoMovement(rf, reward=-0.01, _on_safe_dist_violation=False)
        assert unit.on_safe_dist_violation is False

        unit2 = RewardSafeDistance(rf, reward=-0.15)
        assert unit2.on_safe_dist_violation is True


# =====================================================================
#  RewardActiveHeadingDirection (basic tests)
# =====================================================================

class TestRewardActiveHeadingDirection:
    def test_no_pedestrians_uses_goal_direction(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardActiveHeadingDirection(rf, r_angle=0.6, theta_m=np.pi / 6)

        dist_to_goal = np.array([5.0, 0.1])
        dist_to_subgoal = np.array([2.0, 0.2])
        last_action = np.array([0.5, 0.0, 0.2])
        ped_locs = np.array([])
        ped_vels = np.array([])

        unit(dist_angle_to_goal=dist_to_goal,
             dist_angle_to_subgoal=dist_to_subgoal,
             last_action=last_action,
             pedestrian_relative_locations=ped_locs,
             pedestrian_relative_velocities=ped_vels,
             simulation_state_container=sim_state)

        # reward = 0.6 * (pi/6 - |0.1|) > 0
        expected = 0.6 * (np.pi / 6 - abs(0.1))
        assert total_reward(rf) == pytest.approx(expected)

    def test_none_inputs_no_crash(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardActiveHeadingDirection(rf)

        unit(dist_angle_to_goal=np.array([5.0, 0.1]),
             dist_angle_to_subgoal=np.array([2.0, 0.2]),
             last_action=None,
             pedestrian_relative_locations=None,
             pedestrian_relative_velocities=None,
             simulation_state_container=sim_state)

        assert total_reward(rf) == 0.0


class TestRewardProxemicIntrusion:
    def _call(self, unit, ped_loc, ped_vel, sim_state):
        unit(
            pedestrian_relative_locations=np.array([ped_loc]),
            pedestrian_relative_velocities=np.array([ped_vel]),
            simulation_state_container=sim_state,
        )

    def test_front_intrusion_penalized_more_than_side_or_behind(
        self, make_reward_function, sim_state
    ):
        # Pedestrian walking in +x (robot frame). pedestrian_relative_locations gives
        # the ped's position in the robot frame, so the robot's position relative to
        # the ped is the negation of that vector: placing the ped at (-d, 0) puts the
        # robot at (+d, 0) relative to the ped, i.e. directly ahead of its heading.
        # Front should incur the largest penalty since sigma_front > sigma_side >
        # sigma_back stretches the comfort zone furthest into the pedestrian's path.
        d = 0.8
        ped_vel = [1.0, 0.0]

        rf_front = make_reward_function()
        unit_front = RewardProxemicIntrusion(rf_front)
        self._call(unit_front, [-d, 0.0], ped_vel, sim_state)

        rf_side = make_reward_function()
        unit_side = RewardProxemicIntrusion(rf_side)
        self._call(unit_side, [0.0, d], ped_vel, sim_state)

        rf_back = make_reward_function()
        unit_back = RewardProxemicIntrusion(rf_back)
        self._call(unit_back, [d, 0.0], ped_vel, sim_state)

        penalty_front = -total_reward(rf_front)
        penalty_side = -total_reward(rf_side)
        penalty_back = -total_reward(rf_back)

        assert penalty_front > penalty_side > penalty_back > 0.0

    def test_pedestrian_outside_activation_radius_ignored(
        self, make_reward_function, sim_state
    ):
        rf = make_reward_function()
        unit = RewardProxemicIntrusion(rf, activation_radius=3.0)
        self._call(unit, [10.0, 0.0], [1.0, 0.0], sim_state)
        assert total_reward(rf) == 0.0

    def test_no_pedestrians_no_crash(self, make_reward_function, sim_state):
        rf = make_reward_function()
        unit = RewardProxemicIntrusion(rf)
        unit(
            pedestrian_relative_locations=np.array([]),
            pedestrian_relative_velocities=np.array([]),
            simulation_state_container=sim_state,
        )
        assert total_reward(rf) == 0.0

    def test_stationary_pedestrian_uses_isotropic_zone(
        self, make_reward_function, sim_state
    ):
        # A stationary pedestrian has no defined heading, so front/behind should be
        # penalized identically (both fall back to sigma_side).
        d = 0.8
        ped_vel = [0.0, 0.0]

        rf_front = make_reward_function()
        unit_front = RewardProxemicIntrusion(rf_front)
        self._call(unit_front, [d, 0.0], ped_vel, sim_state)

        rf_back = make_reward_function()
        unit_back = RewardProxemicIntrusion(rf_back)
        self._call(unit_back, [-d, 0.0], ped_vel, sim_state)

        assert total_reward(rf_front) == pytest.approx(total_reward(rf_back))


# =====================================================================
#  RewardSocialPotential
# =====================================================================

class TestRewardSocialPotential:
    def _call(self, unit, ped_locations):
        unit(pedestrian_relative_locations=np.array(ped_locations))

    def test_moving_away_gives_positive_reward(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSocialPotential(rf)

        self._call(unit, [[1.0, 0.0]])
        self._call(unit, [[1.3, 0.0]])

        assert total_reward(rf) > 0

    def test_moving_toward_gives_negative_reward(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSocialPotential(rf)

        self._call(unit, [[2.0, 0.0]])
        self._call(unit, [[1.7, 0.0]])

        assert total_reward(rf) < 0

    def test_no_pedestrians_gives_constant_drift_no_live_signal(
        self, make_reward_function
    ):
        # With gamma < 1 a constant potential yields a fixed per-step drift
        # (factor*(gamma-1)*clip_distance), not exactly zero — same property as
        # `approach_goal`'s PBRS branch. The invariant under test is that an
        # absent pedestrian contributes no *live* gradient beyond that drift.
        rf = make_reward_function()
        unit = RewardSocialPotential(rf)

        self._call(unit, np.empty((0, 2)))
        self._call(unit, np.empty((0, 2)))

        expected_drift = unit._factor * (unit._gamma - 1) * unit._clip_distance
        assert total_reward(rf) == pytest.approx(expected_drift)

    def test_nearest_pedestrian_switch_skips_shaping(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSocialPotential(rf, jump_threshold=0.5)

        self._call(unit, [[1.0, 0.0]])
        self._call(unit, [[2.5, 0.0]])

        assert total_reward(rf) == pytest.approx(0.0)

    def test_pedestrian_beyond_clip_distance_matches_no_pedestrian_drift(
        self, make_reward_function
    ):
        # A pedestrian beyond clip_distance should be indistinguishable from no
        # pedestrian at all — both clip Phi(s) to the same constant.
        rf_far_ped = make_reward_function()
        unit_far_ped = RewardSocialPotential(rf_far_ped, clip_distance=3.0)
        self._call(unit_far_ped, [[5.0, 0.0]])
        self._call(unit_far_ped, [[4.0, 0.0]])

        rf_no_ped = make_reward_function()
        unit_no_ped = RewardSocialPotential(rf_no_ped, clip_distance=3.0)
        self._call(unit_no_ped, np.empty((0, 2)))
        self._call(unit_no_ped, np.empty((0, 2)))

        assert total_reward(rf_far_ped) == pytest.approx(total_reward(rf_no_ped))

    def test_reset_clears_last_phi(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardSocialPotential(rf)

        self._call(unit, [[1.0, 0.0]])
        assert unit.last_phi is not None

        unit.reset()
        assert unit.last_phi is None


class TestRewardTGRFDiscomfort:
    def _call(self, unit, ped_locations):
        unit(pedestrian_relative_locations=np.array(ped_locations))

    def test_peak_penalty_at_zero_distance(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardTGRFDiscomfort(rf, weight=0.25, sigma=0.2, danger_zone_m=0.5)

        self._call(unit, [[0.0, 0.0]])

        assert total_reward(rf) == pytest.approx(-0.25)

    def test_exact_zero_beyond_danger_zone(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardTGRFDiscomfort(rf, weight=0.25, sigma=0.2, danger_zone_m=0.5)

        self._call(unit, [[0.5, 0.0]])
        self._call(unit, [[5.0, 0.0]])

        assert total_reward(rf) == pytest.approx(0.0)

    def test_penalty_decays_with_distance(self, make_reward_function):
        rf_near = make_reward_function()
        unit_near = RewardTGRFDiscomfort(rf_near, weight=0.25, sigma=0.2, danger_zone_m=0.5)
        self._call(unit_near, [[0.1, 0.0]])

        rf_far = make_reward_function()
        unit_far = RewardTGRFDiscomfort(rf_far, weight=0.25, sigma=0.2, danger_zone_m=0.5)
        self._call(unit_far, [[0.4, 0.0]])

        assert total_reward(rf_near) < total_reward(rf_far) < 0.0

    def test_nearest_pedestrian_selected(self, make_reward_function):
        rf_nearest = make_reward_function()
        unit_nearest = RewardTGRFDiscomfort(rf_nearest, weight=0.25, sigma=0.2, danger_zone_m=0.5)
        self._call(unit_nearest, [[0.1, 0.0]])

        rf_multi = make_reward_function()
        unit_multi = RewardTGRFDiscomfort(rf_multi, weight=0.25, sigma=0.2, danger_zone_m=0.5)
        self._call(unit_multi, [[0.1, 0.0], [5.0, 5.0]])

        assert total_reward(rf_multi) == pytest.approx(total_reward(rf_nearest))

    def test_no_pedestrians_no_op(self, make_reward_function):
        rf = make_reward_function()
        unit = RewardTGRFDiscomfort(rf)

        self._call(unit, np.empty((0, 2)))

        assert total_reward(rf) == pytest.approx(0.0)

    def test_check_parameters_warns_on_non_positive_weight(self, make_reward_function):
        rf = make_reward_function()

        with pytest.warns(UserWarning):
            RewardTGRFDiscomfort(rf, weight=-0.1)
