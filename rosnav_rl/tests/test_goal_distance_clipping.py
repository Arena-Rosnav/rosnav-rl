import numpy as np
import pytest

from rosnav_rl.spaces.observation_space.spaces.navigation.basic_navigation_spaces import DistAngleToGoalSpace


def _space(normalize: bool, encoding: int = 2) -> DistAngleToGoalSpace:
    return DistAngleToGoalSpace(
        goal_max_dist=10.0, normalize=normalize, normalizer="max_abs", encoding_version=encoding
    )


def test_goal_distance_encoding_1_passes_a_far_goal_through():
    goal = np.array([25.0, 0.5], dtype=np.float32)

    np.testing.assert_array_equal(_space(normalize=False, encoding=1).encode_observation(goal), goal)


def test_goal_beyond_the_declared_distance_encodes_at_the_bound():
    space = _space(normalize=True)

    far = space.encode_observation(np.array([25.0, 0.5], dtype=np.float32))
    edge = space.encode_observation(np.array([10.0, 0.5], dtype=np.float32))

    np.testing.assert_array_equal(far, edge)
    assert np.abs(far).max() <= 1.0


def test_goal_beyond_the_declared_distance_is_clipped_without_normalization():
    space = _space(normalize=False)
    goal = np.array([25.0, 0.5], dtype=np.float32)

    encoded = space.encode_observation(goal)

    np.testing.assert_array_equal(encoded, np.array([10.0, 0.5], dtype=np.float32))
    np.testing.assert_array_equal(goal, np.array([25.0, 0.5], dtype=np.float32))


def test_goal_within_the_declared_distance_encodes_as_before():
    plain = _space(normalize=False)
    scaled = _space(normalize=True)
    goal = np.array([4.0, -1.0], dtype=np.float32)

    np.testing.assert_array_equal(plain.encode_observation(goal), goal)
    assert scaled.encode_observation(goal)[0] == pytest.approx(2.0 * 4.0 / 10.0 - 1.0)
