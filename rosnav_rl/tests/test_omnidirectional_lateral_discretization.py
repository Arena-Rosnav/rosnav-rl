import numpy as np
import pytest

from rosnav_rl.cfg.action_spaces import (
    DifferentialDriveActionSpace,
    DiscretizationCfg,
    DiscretizationStrategy,
    OmnidirectionalActionSpace,
)

STRATEGIES = [s for s in DiscretizationStrategy if s != DiscretizationStrategy.ROBOT_DEFINED]


def _decoded(space) -> np.ndarray:
    return np.array([space.decode(np.array([i])) for i in range(space.get_gym_space().n)])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_omnidirectional_discretization_spans_the_lateral_range(strategy):
    cfg = DiscretizationCfg(strategy=strategy, buckets_linear=3, buckets_angular=3)
    space = OmnidirectionalActionSpace(linear_range_y=(-0.4, 0.4), discretization=cfg).resolve_discretization()

    commands = _decoded(space)

    np.testing.assert_allclose(sorted(set(commands[:, 1].round(6))), [-0.4, 0.0, 0.4])
    assert len(commands) == 27
    assert len({tuple(row) for row in commands.round(6)}) == 27


def test_omnidirectional_discretization_with_a_zero_lateral_range_stays_planar():
    cfg = DiscretizationCfg(strategy=DiscretizationStrategy.UNIFORM, buckets_linear=3, buckets_angular=3)
    space = OmnidirectionalActionSpace(linear_range_y=(0.0, 0.0), discretization=cfg).resolve_discretization()

    commands = _decoded(space)

    assert len(commands) == 9
    assert not commands[:, 1].any()


def test_omnidirectional_discretization_keeps_the_forward_and_turn_grid():
    cfg = DiscretizationCfg(strategy=DiscretizationStrategy.UNIFORM, buckets_linear=3, buckets_angular=5)
    space = OmnidirectionalActionSpace(
        linear_range_x=(-0.5, 1.0), linear_range_y=(-0.3, 0.3), angular_range=(-2.0, 2.0), discretization=cfg
    ).resolve_discretization()

    commands = _decoded(space)

    np.testing.assert_allclose(sorted(set(commands[:, 0].round(6))), [-0.5, 0.0, 0.25, 1.0])
    np.testing.assert_allclose(sorted(set(commands[:, 2].round(6))), [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_differential_drive_uniform_discretization_is_the_linear_angular_grid():
    cfg = DiscretizationCfg(strategy=DiscretizationStrategy.UNIFORM, buckets_linear=3, buckets_angular=4)
    space = DifferentialDriveActionSpace(
        linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0), discretization=cfg
    ).resolve_discretization()

    pairs = [(a["linear"], a["angular"]) for a in space.discrete_actions]

    expected = [(float(x), float(w)) for x in np.linspace(-0.5, 1.0, 3) for w in np.linspace(-1.0, 1.0, 4)]
    expected.append((0, 0))
    assert pairs == expected
    assert all(a["translational"] is None for a in space.discrete_actions)
