"""Goal and subgoal encodings on known distance and angle geometry."""

import numpy as np
import pytest

from rosnav_rl.spaces.observation_space.spaces.navigation.advanced_navigation_spaces import (
    MultiScaleGoalSpace,
    RobustGoalSpace,
    SubgoalContextSpace,
)
from rosnav_rl.utils.logging import ErrorSeverity, get_error_collector


def test_robust_goal_unknown_scaling_raises_on_encode():
    space = RobustGoalSpace(distance_scaling="cubic")

    with pytest.raises(ValueError, match="cubic"):
        space.encode_observation(np.array([3.0, 0.0]))


def test_robust_goal_unknown_scaling_falls_back_to_a_null_observation():
    collector = get_error_collector()
    collector.clear()
    space = RobustGoalSpace(distance_scaling="cubic")

    result = space.safe_encode_observation(np.array([3.0, 0.0]))

    reported = [(msg.component_name, msg.severity, msg.error_type) for msg in collector._errors]
    collector.clear()
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))
    assert reported == [("RobustGoalSpace", ErrorSeverity.ERROR, "ValueError")]


@pytest.mark.parametrize("scaling", ["tanh", "linear", "log"])
def test_robust_goal_encoding_at_the_max_distance(scaling):
    space = RobustGoalSpace(goal_max_dist=50.0, include_progress=False, distance_scaling=scaling)

    expected = {"tanh": np.tanh(1.0), "linear": 1.0, "log": 1.0}[scaling]
    np.testing.assert_allclose(space.encode_observation(np.array([50.0, -np.pi / 4])), [expected, -0.25], rtol=1e-6)


def test_robust_goal_log_scaling_stays_inside_the_space_beyond_max_distance():
    space = RobustGoalSpace(goal_max_dist=50.0, distance_scaling="log")

    assert space.space.contains(space.encode_observation(np.array([100.0, 0.0])))


def test_multi_scale_goal_tanh_per_scale_with_shared_angle():
    space = MultiScaleGoalSpace(distance_scales=[0.5, 1.0, 2.0], goal_max_dist=10.0)

    result = space.encode_observation(np.array([5.0, np.pi / 2]))

    assert space.space.shape == (6,)
    np.testing.assert_allclose(result, [np.tanh(1.0), 0.5, np.tanh(0.5), 0.5, np.tanh(0.25), 0.5], rtol=1e-6)


def test_subgoal_context_encodes_both_targets_and_their_relation():
    space = SubgoalContextSpace(subgoal_max_dist=10.0)

    result = space.encode_observation(np.array([2.0, np.pi / 2]), np.array([6.0, -np.pi / 4]))

    assert space.space.shape == (6,)
    np.testing.assert_allclose(
        result, [np.tanh(0.2), 0.5, np.tanh(0.6), -0.25, np.tanh(0.4), -0.75], rtol=1e-6, atol=1e-7
    )


def test_subgoal_context_opposite_directions_have_a_unit_angle_difference():
    space = SubgoalContextSpace(subgoal_max_dist=10.0)

    assert abs(space.encode_observation(np.array([1.0, np.pi / 2]), np.array([1.0, -np.pi / 2]))[5]) == pytest.approx(
        1.0
    )


def test_subgoal_context_without_relation_has_four_dims():
    space = SubgoalContextSpace(subgoal_max_dist=10.0, include_goal_subgoal_relation=False)

    result = space.encode_observation(np.array([10.0, 0.0]), np.array([0.0, np.pi]))

    assert space.space.shape == (4,)
    np.testing.assert_allclose(result, [np.tanh(1.0), 0.0, 0.0, 1.0], rtol=1e-6)


def test_subgoal_context_angle_difference_wraps_across_the_back():
    space = SubgoalContextSpace(subgoal_max_dist=10.0)

    result = space.encode_observation(np.array([1.0, 0.9 * np.pi]), np.array([1.0, -0.9 * np.pi]))

    assert result[5] == pytest.approx(0.2, abs=1e-5)
