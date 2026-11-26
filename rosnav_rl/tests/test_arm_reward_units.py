"""Arm goal reward units read their tolerances from the unit params."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.reward.constants import DONE_REASONS
from rosnav_rl.reward.reward_function import RewardFunction

PARAMS = AgentParameters()
TARGET = "cube"


def _pose3d(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _pose2d(x=0.0, y=0.0, yaw=0.0) -> np.ndarray:
    return np.array([x, y, yaw], dtype=np.float32)


def _grasp_state(*grasped: str) -> SimpleNamespace:
    return SimpleNamespace(other=set(grasped))


def _step(reward_function: RewardFunction, **obs) -> tuple[float, dict]:
    return reward_function.get_reward(
        obs_dict={"simulation_state_container": PARAMS, **obs},
        simulation_state_container=PARAMS,
    )


def _arm_reach(**overrides) -> RewardFunction:
    params = {
        "pos_factor": 1.0,
        "neg_factor": 1.0,
        "orient_factor": 0.0,
        "reach_radius": 10.0,
        "reach_blend": 1.0,
        "max_step_distance": 10.0,
        "max_step_angle": 10.0,
        "target": TARGET,
        **overrides,
    }
    return RewardFunction({"arm_reach": params})


def _reach_step(
    reward_function: RewardFunction,
    distance: float,
    angle: float,
    *,
    robot=(0.0, 0.0),
    target=(0.0, 0.0),
    grasped=(),
) -> tuple[float, dict]:
    return _step(
        reward_function,
        grasp_dist_angle=np.array([distance, angle], dtype=np.float32),
        target_pose=_pose3d(*target),
        robot_pose=_pose2d(*robot),
        grasp_state=_grasp_state(*grasped),
    )


def test_arm_reach_first_step_only_records_the_baseline():
    reward, info = _reach_step(_arm_reach(), 1.0, 0.5)

    assert reward == 0.0
    assert info == {}


def test_arm_reach_rewards_a_shorter_distance():
    rf = _arm_reach()
    _reach_step(rf, 1.0, 0.5)

    reward, _ = _reach_step(rf, 0.8, 0.5)

    assert reward == pytest.approx(0.2)


def test_arm_reach_penalises_a_longer_distance():
    rf = _arm_reach()
    _reach_step(rf, 0.8, 0.5)

    reward, _ = _reach_step(rf, 1.0, 0.5)

    assert reward == pytest.approx(-0.2)


def test_arm_reach_scales_angle_improvement_by_orient_factor():
    rf = _arm_reach(orient_factor=0.5)
    _reach_step(rf, 1.0, 0.9)

    reward, _ = _reach_step(rf, 1.0, 0.5)

    assert reward == pytest.approx(0.2)


def test_arm_reach_blends_by_robot_distance_to_target():
    rf = _arm_reach(reach_radius=1.0, reach_blend=0.5)
    _reach_step(rf, 1.0, 0.5, robot=(0.0, 0.0), target=(1.0, 0.0))
    reward, _ = _reach_step(rf, 0.8, 0.5, robot=(0.0, 0.0), target=(1.0, 0.0))

    assert reward == pytest.approx(0.1)


def test_arm_reach_is_silent_outside_the_reach_radius():
    rf = _arm_reach(reach_radius=1.0, reach_blend=0.5)
    _reach_step(rf, 1.0, 0.5, robot=(0.0, 0.0), target=(2.0, 0.0))

    reward, _ = _reach_step(rf, 0.8, 0.5, robot=(0.0, 0.0), target=(2.0, 0.0))

    assert reward == 0.0


def test_arm_reach_stops_once_the_target_is_grasped():
    rf = _arm_reach()
    _reach_step(rf, 1.0, 0.5)

    reward, _ = _reach_step(rf, 0.8, 0.5, grasped=(TARGET,))

    assert reward == 0.0


def _object_lifted(**overrides) -> RewardFunction:
    params = {
        "reward": 5.0,
        "lift_height": 0.1,
        "min_height": 0.0,
        "terminate": True,
        "target": TARGET,
        "acquire_steps": 1,
        "lift_steps": 2,
        **overrides,
    }
    return RewardFunction({"object_lifted": params})


def _lift_step(reward_function: RewardFunction, z: float, *, grasped=True) -> tuple[float, dict]:
    return _step(
        reward_function,
        target_pose=_pose3d(0.0, 0.0, z),
        grasp_state=_grasp_state(TARGET) if grasped else _grasp_state(),
    )


def test_object_lifted_awards_after_holding_the_lift():
    rf = _object_lifted()

    assert _lift_step(rf, 0.0)[0] == 0.0
    assert _lift_step(rf, 0.2)[0] == 0.0

    reward, info = _lift_step(rf, 0.2)

    assert reward == pytest.approx(5.0)
    assert info["is_done"] is True
    assert info["done_reason"] == DONE_REASONS.SUCCESS
    assert info["is_success"] is True


def test_object_lifted_needs_the_target_to_be_grasped():
    rf = _object_lifted()

    for z in (0.0, 0.2, 0.4, 0.6):
        reward, info = _lift_step(rf, z, grasped=False)
        assert reward == 0.0
        assert "is_done" not in info


def test_object_lifted_requires_acquire_steps_of_contact():
    rf = _object_lifted(acquire_steps=2, lift_steps=1)

    assert _lift_step(rf, 0.0)[0] == 0.0

    reward, info = _lift_step(rf, 0.2)

    assert reward == pytest.approx(5.0)
    assert info["is_done"] is True


def test_object_lifted_resets_the_contact_counter_on_release():
    rf = _object_lifted(acquire_steps=2, lift_steps=1)

    _lift_step(rf, 0.0)
    _lift_step(rf, 0.0, grasped=False)

    assert _lift_step(rf, 0.2)[0] == 0.0


def test_object_lifted_respects_the_lift_height():
    rf = _object_lifted(lift_height=0.2, lift_steps=1)

    assert _lift_step(rf, 0.0)[0] == 0.0
    assert _lift_step(rf, 0.1)[0] == 0.0

    reward, _ = _lift_step(rf, 0.2)

    assert reward == pytest.approx(5.0)


def test_object_lifted_respects_min_height():
    rf = _object_lifted(lift_height=0.0, min_height=0.5, lift_steps=1)

    assert _lift_step(rf, 0.0)[0] == 0.0
    assert _lift_step(rf, 0.3)[0] == 0.0

    reward, _ = _lift_step(rf, 0.6)

    assert reward == pytest.approx(5.0)


def test_object_lifted_does_not_set_done_when_terminate_is_false():
    rf = _object_lifted(terminate=False, lift_steps=1)

    _lift_step(rf, 0.0)
    reward, info = _lift_step(rf, 0.2)

    assert reward == pytest.approx(5.0)
    assert "is_done" not in info


def test_object_lifted_awards_only_once():
    rf = _object_lifted(lift_steps=1)

    _lift_step(rf, 0.0)
    assert _lift_step(rf, 0.2)[0] == pytest.approx(5.0)

    assert _lift_step(rf, 0.3)[0] == 0.0


def test_object_lifted_reset_rearms_the_award():
    rf = _object_lifted(lift_steps=1)

    _lift_step(rf, 0.0)
    assert _lift_step(rf, 0.2)[0] == pytest.approx(5.0)

    rf.reset()

    _lift_step(rf, 0.0)
    reward, _ = _lift_step(rf, 0.2)

    assert reward == pytest.approx(5.0)


@pytest.mark.parametrize(
    "units",
    [
        ("collision", "arm_reach"),
        ("arm_reach", "collision"),
    ],
)
def test_a_collision_still_ends_the_episode_next_to_the_arm_units(units):
    rf = RewardFunction({name: {} for name in units})
    laser = np.full(360, 5.0, dtype=np.float32)

    def step(collision_monitor: bool):
        return _step(
            rf,
            grasp_dist_angle=np.array([1.0, 0.5], dtype=np.float32),
            target_pose=_pose3d(),
            robot_pose=_pose2d(),
            grasp_state=_grasp_state(),
            front_laser=laser,
            collision_monitor=collision_monitor,
        )

    step(False)
    _, info = step(True)

    assert info["is_done"] is True
    assert info["done_reason"] == DONE_REASONS.COLLISION
