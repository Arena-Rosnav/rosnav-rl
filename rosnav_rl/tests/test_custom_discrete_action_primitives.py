"""Discrete action grids, navigation primitives and log-spaced action sets."""

import string

import numpy as np
import pytest

from rosnav_rl.utils.action_space.custom_discrete_action import (
    generate_discrete_action_dict,
    generate_exponential_actions,
    generate_navigational_actions,
)


def motions(actions):
    return [(a["linear"], a["angular"]) for a in actions]


def test_grid_appends_a_stop_action_when_the_ranges_exclude_zero():
    actions = generate_discrete_action_dict((0.25, 0.5), (0.5, 1.0), 2, 2)

    assert motions(actions) == [(0.25, 0.5), (0.25, 1.0), (0.5, 0.5), (0.5, 1.0), (0, 0)]


def test_grid_keeps_a_single_stop_when_zero_is_on_the_grid():
    actions = generate_discrete_action_dict((0.0, 1.0), (-1.0, 1.0), 2, 3)

    assert motions(actions).count((0.0, 0.0)) == 1
    assert len(actions) == 6


def test_action_names_are_twelve_distinct_lowercase_letters():
    for action in generate_discrete_action_dict((0.0, 1.0), (-1.0, 1.0), 3, 3):
        assert len(action["name"]) == 12
        assert len(set(action["name"])) == 12
        assert set(action["name"]) <= set(string.ascii_lowercase)


def test_grid_reaches_the_configured_velocity_limits():
    actions = generate_discrete_action_dict((0.0, 0.22), (-2.84, 2.84), 2, 3)

    assert max(a["linear"] for a in actions) == pytest.approx(0.22, rel=1e-6)
    assert max(a["angular"] for a in actions) == pytest.approx(2.84, rel=1e-6)


def test_navigational_primitives_scale_with_the_velocity_limits():
    actions = generate_navigational_actions((-0.4, 1.0), (-2.0, 2.0))

    np.testing.assert_allclose(
        motions(actions),
        [
            (0.0, 0.0),
            (0.35, 0.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (1.0, -0.5),
            (0.65, 1.3),
            (0.65, -1.3),
            (0.0, 2.0),
            (0.0, -2.0),
            (0.35, 2.0),
            (0.35, -2.0),
            (-0.2, 0.0),
        ],
    )
    assert {a["translational"] for a in actions} == {None}


def test_navigational_forward_speeds_start_from_a_positive_minimum():
    actions = generate_navigational_actions((0.2, 1.0), (-1.0, 1.0))

    np.testing.assert_allclose(motions(actions)[1], (0.48, 0.0))
    np.testing.assert_allclose(motions(actions)[5], (0.72, 0.65))


def test_grid_translational_values_keep_their_configured_endpoints():
    actions = generate_discrete_action_dict((0.0, 0.2), (-1.0, 1.0), 2, 2, (-2.84, 0.22), 2)

    assert sorted({a["translational"] for a in actions}) == [-2.84, 0.22]


def test_navigational_actions_are_unique_without_reverse():
    actions = generate_navigational_actions((0.0, 1.0), (-1.0, 1.0))

    assert len(set(motions(actions))) == len(actions)


def test_exponential_grid_is_log_spaced_with_stop_and_gentle_reverse():
    actions = generate_exponential_actions((-0.5, 1.0), (-1.0, 1.0), n_forward=3, n_angular=2)

    mid = np.sqrt(2.0) - 1.0
    angulars = [-1.0, -mid, 0.0, mid, 1.0]
    expected = [(lin, ang) for lin in (0.0, mid, 1.0) for ang in angulars] + [(-0.25, 0.0)]
    np.testing.assert_allclose(motions(actions), expected)


def test_exponential_grid_without_reverse_has_no_backward_action():
    actions = generate_exponential_actions((0.0, 1.0), (-1.0, 1.0), n_forward=2, n_angular=1)

    np.testing.assert_allclose(motions(actions), [(0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])


def test_exponential_grid_with_positive_minimum_adds_a_separate_stop():
    actions = generate_exponential_actions((0.5, 1.0), (-1.0, 1.0), n_forward=2, n_angular=1)

    assert motions(actions)[-1] == (0.0, 0.0)
    np.testing.assert_allclose(motions(actions)[0], (0.5, -1.0))
    assert len(actions) == 7
