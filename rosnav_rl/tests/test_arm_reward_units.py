"""Arm goal reward units read their tolerances from the unit params."""

import numpy as np
import pytest

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.reward.constants import DONE_REASONS
from rosnav_rl.reward.reward_function import RewardFunction

PARAMS = AgentParameters()


def _step(reward_function: RewardFunction, distance: float, rotation: float, **obs) -> tuple[float, dict]:
    obs_dict = {
        "arm_dist_angle_to_goal": np.array([distance, rotation], dtype=np.float32),
        "simulation_state_container": PARAMS,
        **obs,
    }
    return reward_function.get_reward(obs_dict=obs_dict, simulation_state_container=PARAMS)


def test_arm_goal_inside_the_default_tolerances_succeeds():
    reward, info = _step(RewardFunction({"arm_goal_reached": {"reward": 5.0}}), 0.04, 0.1)

    assert reward == pytest.approx(5.0)
    assert info["is_done"] is True
    assert info["done_reason"] == DONE_REASONS.SUCCESS


def test_arm_goal_radius_comes_from_the_unit_params():
    loose = RewardFunction({"arm_goal_reached": {"reward": 5.0, "goal_radius": 0.2}})
    tight = RewardFunction({"arm_goal_reached": {"reward": 5.0}})

    assert _step(loose, 0.1, 0.1)[1].get("is_done") is True
    assert _step(tight, 0.1, 0.1)[1].get("is_done", False) is False


def test_arm_angle_tolerance_comes_from_the_unit_params():
    loose = RewardFunction({"arm_goal_reached": {"reward": 5.0, "angle_tolerance": 0.5}})
    tight = RewardFunction({"arm_goal_reached": {"reward": 5.0}})

    assert _step(loose, 0.01, 0.3)[1].get("is_done") is True
    assert _step(tight, 0.01, 0.3)[1].get("is_done", False) is False


@pytest.mark.parametrize(
    "units",
    [
        ("collision", "arm_approach_goal", "arm_goal_reached"),
        ("arm_approach_goal", "arm_goal_reached", "collision"),
    ],
)
def test_a_collision_still_ends_the_episode_next_to_the_arm_units(units):
    reward_function = RewardFunction({name: {} for name in units})
    laser = np.full(360, 5.0, dtype=np.float32)

    _step(reward_function, 1.0, 1.0, front_laser=laser, collision_monitor=False)
    _, info = _step(reward_function, 0.9, 1.0, front_laser=laser, collision_monitor=True)

    assert info["is_done"] is True
    assert info["done_reason"] == DONE_REASONS.COLLISION


def test_arm_units_leave_the_done_flag_unset_while_the_goal_is_far():
    _, info = _step(RewardFunction({"arm_approach_goal": {}, "arm_goal_reached": {}}), 1.0, 1.0)

    assert "is_done" not in info
