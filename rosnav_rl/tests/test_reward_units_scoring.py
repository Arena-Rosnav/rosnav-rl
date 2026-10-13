"""Reward unit scoring driven through a real RewardFunction holder."""

import random

import numpy as np
import pytest

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.utils.pose import Pose2DType
from rosnav_rl.reward.constants import DONE_REASONS
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.reward.reward_units.reward_units import (
    RewardAbruptVelocityChange,
    RewardActiveHeadingDirection,
    RewardAngularVelocityConstraint,
    RewardApproachGoal,
    RewardCollision,
    RewardDistanceTravelled,
    RewardFactoredReverseDrive,
    RewardFactoredSafeDistance,
    RewardGoalReached,
    RewardLinearVelBoost,
    RewardMaxStepsExceeded,
    RewardNoMovement,
    RewardPedTypeCollision,
    RewardPedTypeFactoredSafetyDistance,
    RewardPedTypeSafetyDistance,
    RewardPedTypeVelocityConstraint,
    RewardReverseDrive,
    RewardRootVelocityDifference,
    RewardSafeDistance,
    RewardTwoFactorVelocityDifference,
)
from rosnav_rl.utils.logging import ErrorSeverity, get_error_collector


@pytest.fixture
def holder():
    return RewardFunction({})


@pytest.fixture
def params():
    return AgentParameters(robot_radius=0.3, safety_distance=0.5, goal_radius=0.3, max_steps=3)


@pytest.fixture
def collector():
    errors = get_error_collector()
    previous = errors._min_severity
    errors.set_min_severity(ErrorSeverity.DEBUG)
    errors.clear()
    yield errors
    errors.clear()
    errors.set_min_severity(previous)


def warnings_of(errors, name):
    return [
        msg.message for msg in errors._errors if msg.component_name == name and msg.severity == ErrorSeverity.WARNING
    ]


def pose(x, y, yaw):
    return np.array((x, y, yaw), dtype=Pose2DType)


def score(holder):
    reward = holder.state.current_reward
    holder._reset_state()
    return reward


@pytest.mark.parametrize(
    ("unit_class", "kwargs"),
    [
        (RewardSafeDistance, {"reward": 0.2}),
        (RewardFactoredSafeDistance, {"factor": 0.0}),
        (RewardNoMovement, {"reward": 0.1}),
        (RewardCollision, {"reward": 5.0}),
        (RewardReverseDrive, {"reward": 0.3}),
        (RewardFactoredReverseDrive, {"factor": 0.4}),
        (RewardMaxStepsExceeded, {"penalty": -1.0}),
    ],
)
def test_sign_inverted_parameters_are_reported_as_one_warning(holder, collector, unit_class, kwargs):
    unit_class(holder, **kwargs)

    assert len(warnings_of(collector, unit_class.__name__)) == 1


@pytest.mark.parametrize(
    ("unit_class", "kwargs"),
    [
        (RewardSafeDistance, {"reward": -0.2}),
        (RewardFactoredSafeDistance, {"factor": -0.5}),
        (RewardNoMovement, {"reward": -0.1}),
        (RewardCollision, {"reward": -5.0}),
        (RewardReverseDrive, {"reward": -0.3}),
        (RewardFactoredReverseDrive, {"factor": -0.4}),
        (RewardMaxStepsExceeded, {"penalty": 1.0}),
        (RewardApproachGoal, {"pos_factor": 0.3, "neg_factor": 0.5}),
    ],
)
def test_well_signed_parameters_report_nothing(holder, collector, unit_class, kwargs):
    unit_class(holder, **kwargs)

    assert warnings_of(collector, unit_class.__name__) == []


def test_approach_goal_warns_on_negative_factor_and_on_pos_not_below_neg(holder, collector):
    RewardApproachGoal(holder, pos_factor=-0.1, neg_factor=-0.2)

    messages = warnings_of(collector, "RewardApproachGoal")
    assert len(messages) == 2
    assert any("Both factors should be positive" in m for m in messages)
    assert any("rotary trajectories" in m for m in messages)


def test_approach_goal_warns_only_on_ordering_when_both_positive(holder, collector):
    RewardApproachGoal(holder, pos_factor=0.5, neg_factor=0.5)

    messages = warnings_of(collector, "RewardApproachGoal")
    assert len(messages) == 1
    assert "rotary trajectories" in messages[0]


def test_goal_reached_writes_success_info_into_the_holder(holder, params):
    unit = RewardGoalReached(holder, reward=12.0)

    unit(np.array([0.1, 0.0]), np.array([5.0, 0.0]), params)

    assert holder.state.current_reward == pytest.approx(12.0)
    assert holder.state.info == {"is_done": True, "done_reason": DONE_REASONS.SUCCESS, "is_success": True}
    assert holder.state.reward_overview == {"RewardGoalReached": 12.0}


def test_factored_safe_distance_scales_with_the_margin_violation(holder, params):
    unit = RewardFactoredSafeDistance(holder, factor=-2.0)

    unit(True, np.array([1.0, 0.6, 2.0]), params)

    assert holder.state.info == {"safe_dist_violation": True}
    assert score(holder) == pytest.approx(-2.0 * (0.5 + 0.3 - 0.6))


def test_collision_latches_after_the_monitor_pulse_clears(holder, params):
    unit = RewardCollision(holder, reward=-10.0)
    clear = np.array([5.0, 5.0])

    unit(clear, params, collision_monitor=True)
    first = score(holder)
    unit(clear, params, collision_monitor=False)

    assert first == pytest.approx(-10.0)
    assert holder.state.info["done_reason"] == DONE_REASONS.COLLISION
    assert score(holder) == pytest.approx(-10.0)


def test_collision_reset_releases_the_latch(holder, params):
    unit = RewardCollision(holder, reward=-10.0)
    unit(np.array([0.2]), params)
    score(holder)

    unit.reset()
    unit(np.array([5.0]), params, collision_monitor=False)

    assert score(holder) == 0.0


def test_collision_laser_threshold_is_radius_plus_bumper_zone_inclusive(holder, params):
    unit = RewardCollision(holder, reward=-10.0, bumper_zone=0.05)

    unit(np.array([0.35, 3.0]), params)

    assert score(holder) == pytest.approx(-10.0)


def test_approach_goal_rewards_progress_with_pos_and_retreat_with_neg_factor(holder):
    unit = RewardApproachGoal(holder, pos_factor=0.3, neg_factor=0.5)
    far = np.array([4.0, 3.0])
    near = np.array([3.0, 0.0])

    unit(pose(0.0, 0.0, 0.0), far, far)
    assert score(holder) == 0.0
    unit(pose(2.0, 3.0, 0.0), np.array([2.0, 0.0]), np.array([2.0, 0.0]))
    assert score(holder) == pytest.approx(0.3 * (5.0 - 2.0))
    unit(pose(1.0, 3.0, 0.0), near, near)
    assert score(holder) == pytest.approx(0.5 * (2.0 - 3.0))


def test_approach_goal_jump_is_measured_in_the_world_frame_of_a_rotated_robot(holder):
    unit = RewardApproachGoal(holder, pos_factor=0.3, neg_factor=0.5, _goal_update_threshold=1.0)

    unit(pose(0.0, 0.0, np.pi / 2), np.array([2.0, 0.0]), np.array([2.0, 0.0]))
    unit(pose(0.0, 1.0, np.pi / 2), np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    assert score(holder) == pytest.approx(0.3 * 1.0)
    np.testing.assert_allclose(unit._last_goal_world, [0.0, 2.0], atol=1e-6)


def test_distance_travelled_weighs_linear_and_angular_magnitudes(holder):
    unit = RewardDistanceTravelled(holder, consumption_factor=0.5, lin_vel_scalar=2.0, ang_vel_scalar=0.1)

    unit(np.array([-0.4, 0.0, -1.0]))

    assert score(holder) == pytest.approx(-(0.4 * 2.0 + 1.0 * 0.1) * 0.5)


def test_reverse_drive_threshold_only_fires_below_it(holder):
    unit = RewardReverseDrive(holder, reward=-0.3, threshold=-0.2)

    unit(np.array([-0.1, 0.0, 0.0]))
    assert score(holder) == 0.0
    unit(np.array([-0.25, 0.0, 0.0]))
    assert score(holder) == pytest.approx(-0.3)


@pytest.mark.xfail(
    strict=True, reason="reward_units RewardReverseDrive default reward is positive so reversing is rewarded"
)
def test_reverse_drive_defaults_penalize_reversing(holder):
    unit = RewardReverseDrive(holder)

    unit(np.array([-0.3, 0.0, 0.0]))

    assert score(holder) < 0.0


def test_factored_reverse_drive_is_proportional_to_reverse_speed(holder):
    unit = RewardFactoredReverseDrive(holder, factor=-0.2)

    unit(np.array([-0.5, 0.0, 0.0]))

    assert score(holder) == pytest.approx(-0.1)


def test_abrupt_velocity_change_applies_quartic_penalty_per_dimension(holder):
    unit = RewardAbruptVelocityChange(holder, vel_factors={"0": 1.0, "2": 0.5})

    unit(np.array([0.0, 0.0, 0.0]))
    unit(np.array([2.0, 5.0, -1.0]))

    assert score(holder) == pytest.approx(-(2.0**4 / 100 * 1.0) - (1.0**4 / 100 * 0.5))


def test_abrupt_velocity_change_tracks_a_copy_of_the_action(holder):
    unit = RewardAbruptVelocityChange(holder, vel_factors={"0": 1.0})
    action = np.array([1.0, 0.0, 0.0])

    unit(action)
    action[0] = 3.0
    unit(np.array([1.0, 0.0, 0.0]))

    assert score(holder) == 0.0


def test_root_velocity_difference_rewards_small_l2_change(holder):
    unit = RewardRootVelocityDifference(holder, k=2.0)

    unit(np.array([0.0, 0.0, 0.0]))
    unit(np.array([0.3, 0.0, 0.4]))

    assert score(holder) == pytest.approx((1 - 0.5) / 2.0)


def test_two_factor_velocity_difference_weighs_first_and_last_dimension(holder):
    unit = RewardTwoFactorVelocityDifference(holder, alpha=0.1, beta=0.2)

    unit(np.array([1.0, 9.0, 1.0]))
    unit(np.array([0.5, 0.0, -1.0]))

    assert score(holder) == pytest.approx(-(0.5 * 0.1 + 2.0 * 0.2))


def test_ped_type_safety_distance_only_penalizes_violated_types(holder):
    unit = RewardPedTypeSafetyDistance(holder, type_reward_pairs={1: -0.5, 2: -1.0}, safety_distance=1.0)

    unit({1: 0.4, 2: 1.5})

    assert score(holder) == pytest.approx(-0.5)


def test_ped_type_safety_distance_warns_on_missing_type_and_empty_input(holder, collector):
    unit = RewardPedTypeSafetyDistance(holder, ped_type=7, reward=-0.5, safety_distance=1.0)

    unit({1: 0.1})
    unit({})

    assert score(holder) == 0.0
    assert warnings_of(collector, "RewardPedTypeSafetyDistance") == [
        "Pedestrian type 7 not found.",
        "No pedestrian type distances found.",
    ]


def test_ped_type_factored_safety_distance_scales_with_intrusion(holder):
    unit = RewardPedTypeFactoredSafetyDistance(holder, type_factor_pairs={1: -2.0, 3: -1.0}, safety_distance=1.0)

    unit({1: 0.25, 3: 1.2})

    assert score(holder) == pytest.approx(-2.0 * 0.75)


def test_ped_type_factored_safety_distance_warns_on_missing_type_and_empty_input(holder, collector):
    unit = RewardPedTypeFactoredSafetyDistance(holder, ped_type=4, factor=-1.0)

    unit({1: 0.1})
    unit({})

    assert score(holder) == 0.0
    assert warnings_of(collector, "RewardPedTypeFactoredSafetyDistance") == [
        "Pedestrian type 4 not found.",
        "Won't apply reward unit. No pedestrian type distances found.",
    ]


def test_ped_type_collision_threshold_includes_robot_radius(holder, params):
    unit = RewardPedTypeCollision(holder, type_reward_pairs={1: -10.0, 2: -4.0}, bumper_zone=0.1)

    unit({1: 0.4, 2: 0.41}, params)

    assert score(holder) == pytest.approx(-10.0)


def test_ped_type_collision_warns_on_missing_type_and_empty_input(holder, params, collector):
    unit = RewardPedTypeCollision(holder, ped_type=9, reward=-10.0)

    unit({1: 0.0}, params)
    unit({}, params)

    assert score(holder) == 0.0
    assert warnings_of(collector, "RewardPedTypeCollision") == [
        "Pedestrian type 9 not found.",
        "Won't apply reward unit. No pedestrian type distances found.",
    ]


def test_ped_type_velocity_constraint_penalizes_forward_speed_near_the_type(holder):
    unit = RewardPedTypeVelocityConstraint(holder, ped_type=2, penalty_factor=0.5, active_distance=2.0)

    unit({2: 1.0}, np.array([0.8, 0.0, 0.0]))
    near = score(holder)
    unit({2: 2.5}, np.array([0.8, 0.0, 0.0]))

    assert near == pytest.approx(-0.4)
    assert score(holder) == 0.0


def test_ped_type_velocity_constraint_warns_and_skips_on_missing_inputs(holder, collector):
    unit = RewardPedTypeVelocityConstraint(holder, ped_type=2)

    unit({}, np.array([1.0, 0.0, 0.0]))
    unit({2: 0.1}, None)
    unit({1: 0.1}, np.array([1.0, 0.0, 0.0]))

    assert score(holder) == 0.0
    assert warnings_of(collector, "RewardPedTypeVelocityConstraint") == [
        "Won't apply reward unit. No pedestrian type distances found.",
        "Won't apply reward unit. No last action found.",
        "Pedestrian type 2 not found.",
    ]


def test_angular_velocity_constraint_scales_with_absolute_rate_above_threshold(holder):
    unit = RewardAngularVelocityConstraint(holder, penalty_factor=-0.1, threshold=0.5)

    unit(np.array([0.0, 0.0, -0.8]))

    assert score(holder) == pytest.approx(-0.08)


def test_linear_vel_boost_scales_with_speed_above_threshold(holder):
    unit = RewardLinearVelBoost(holder, reward_factor=0.2, threshold=0.5)

    unit(np.array([1.5, 0.0, 0.0]))

    assert score(holder) == pytest.approx(0.3)


def test_max_steps_exceeded_terminates_on_the_limit_step(holder, params):
    unit = RewardMaxStepsExceeded(holder, penalty=4.0)

    unit(params)
    unit(params)
    assert score(holder) == 0.0
    unit(params)

    assert holder.state.current_reward == pytest.approx(-4.0)
    assert holder.state.info["done_reason"] == DONE_REASONS.STEP_LIMIT


@pytest.mark.parametrize(
    "unit_class",
    [
        RewardActiveHeadingDirection,
        RewardPedTypeSafetyDistance,
        RewardPedTypeFactoredSafetyDistance,
        RewardPedTypeVelocityConstraint,
        RewardAngularVelocityConstraint,
        RewardLinearVelBoost,
    ],
)
def test_stateless_units_reset_without_changing_parameters(holder, unit_class):
    unit = unit_class(holder)
    before = dict(vars(unit))

    unit.reset()

    assert vars(unit) == before


def test_ped_type_collision_reset_keeps_type_pairs(holder):
    unit = RewardPedTypeCollision(holder, type_reward_pairs={3: -2.0})

    unit.reset()

    assert unit._type_reward_pairs == {3: -2.0}


def test_active_heading_without_pedestrians_rewards_alignment_with_the_goal(holder, params):
    unit = RewardActiveHeadingDirection(holder, r_angle=0.6, theta_m=np.pi / 6)

    unit(
        np.array([3.0, 0.2]),
        np.array([1.0, -0.4]),
        np.array([0.5, 0.0, 0.0]),
        np.empty((0, 2)),
        np.empty((0, 2)),
        params,
    )

    assert score(holder) == pytest.approx(0.6 * (np.pi / 6 - 0.2))


def test_active_heading_follows_the_subgoal_angle_when_configured(holder, params):
    unit = RewardActiveHeadingDirection(holder, r_angle=1.0, theta_m=0.5, _following_subgoal=True)

    unit(
        np.array([3.0, 0.2]),
        np.array([1.0, -0.4]),
        np.array([0.5, 0.0, 0.0]),
        np.empty((0, 2)),
        np.empty((0, 2)),
        params,
    )

    assert score(holder) == pytest.approx(0.5 - 0.4)


def test_active_heading_skips_without_pedestrian_data(holder, params):
    unit = RewardActiveHeadingDirection(holder)

    unit(np.array([3.0, 0.2]), np.array([1.0, 0.0]), np.array([0.5, 0.0, 0.0]), None, np.empty((0, 2)), params)

    assert holder.state.reward_overview == {}


@pytest.mark.parametrize(
    ("location", "velocity"),
    [([20.0, 0.0], [-1.0, 0.0]), ([0.5, 0.0], [-1.0, 0.0]), ([0.0, 3.0], [0.0, 0.0])],
)
def test_active_heading_with_only_harmless_pedestrians_settles_near_the_goal_angle(holder, params, location, velocity):
    unit = RewardActiveHeadingDirection(holder, r_angle=1.0, theta_m=0.5, ped_min_dist=8.0, iters=4000)
    random.seed(3)

    unit(
        np.array([4.0, 0.3]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([location]),
        np.array([velocity]),
        params,
    )

    assert score(holder) == pytest.approx(0.5 - 0.3, abs=0.01)


def test_active_heading_blocked_everywhere_falls_back_to_a_right_angle(holder, params):
    unit = RewardActiveHeadingDirection(holder, r_angle=1.0, theta_m=0.5, iters=50)
    random.seed(5)

    unit(
        np.array([4.0, 0.3]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([[2.0, 1e-4]]),
        np.array([[-1.0, -0.01]]),
        params,
    )

    assert score(holder) == pytest.approx(0.5 - np.pi / 2)


def test_active_heading_blocks_a_pedestrian_approaching_from_behind(holder, params):
    unit = RewardActiveHeadingDirection(holder, r_angle=1.0, theta_m=0.5, iters=50)
    random.seed(5)

    unit(
        np.array([4.0, 0.3]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([[-2.0, 1e-4]]),
        np.array([[1.0, 0.01]]),
        params,
    )

    assert score(holder) == pytest.approx(0.5 - np.pi / 2)


def test_active_heading_avoids_the_cone_of_a_pedestrian_ahead(holder, params):
    unit = RewardActiveHeadingDirection(holder, r_angle=1.0, theta_m=0.5, iters=4000)
    random.seed(11)

    unit(
        np.array([4.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([[2.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        params,
    )

    vo_half_angle = np.arctan2(0.9, np.sqrt(4.0 - 0.81))
    assert score(holder) == pytest.approx(0.5 - vo_half_angle, abs=0.01)
