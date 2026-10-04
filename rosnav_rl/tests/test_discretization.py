from dataclasses import dataclass

import numpy as np

from rosnav_rl.cfg.action_spaces import (
    DifferentialDriveActionSpace,
    DiscretizationCfg,
    DiscretizationStrategy,
    OmnidirectionalActionSpace,
)


@dataclass
class _RobotAction:
    name: str
    linear: float
    angular: float
    lateral: float = 0.0


_ROBOT_ACTIONS = [
    _RobotAction("forward", 0.5, 0.0),
    _RobotAction("turn_left", 0.0, 1.0),
    _RobotAction("strafe_left", 0.0, 0.0, lateral=0.3),
]


def _robot_defined() -> DiscretizationCfg:
    return DiscretizationCfg(strategy=DiscretizationStrategy.ROBOT_DEFINED)


def test_differential_drive_robot_defined_actions_decode_to_linear_and_angular():
    space = DifferentialDriveActionSpace(discretization=_robot_defined()).resolve_discretization(_ROBOT_ACTIONS)

    assert space.get_gym_space().n == 3
    np.testing.assert_array_almost_equal(space.decode(np.array([0])), [0.5, 0.0, 0.0])
    np.testing.assert_array_almost_equal(space.decode(np.array([1])), [0.0, 0.0, 1.0])


def test_omnidirectional_robot_defined_actions_carry_lateral_into_linear_y():
    space = OmnidirectionalActionSpace(discretization=_robot_defined()).resolve_discretization(_ROBOT_ACTIONS)

    np.testing.assert_array_almost_equal(space.decode(np.array([2])), [0.0, 0.3, 0.0])


def test_omnidirectional_uniform_actions_all_decode():
    cfg = DiscretizationCfg(strategy=DiscretizationStrategy.UNIFORM, buckets_linear=3, buckets_angular=3)
    space = OmnidirectionalActionSpace(discretization=cfg).resolve_discretization()

    decoded = [space.decode(np.array([i])) for i in range(space.get_gym_space().n)]

    assert len(decoded) == len(space.discrete_actions)
    assert all(cmd.shape == (3,) and cmd[1] == 0.0 for cmd in decoded)
