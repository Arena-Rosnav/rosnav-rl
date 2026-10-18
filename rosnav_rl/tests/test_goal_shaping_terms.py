"""The step cost charges every step, and approach shaping with equal factors pays a round trip back to zero."""

import numpy as np
import pytest

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.reward.reward_function import RewardFunction

Pose2DType = np.dtype([("x", np.float32), ("y", np.float32), ("yaw", np.float32)])


def _pose(x: float) -> np.ndarray:
    return np.array((x, 0.0, 0.0), dtype=Pose2DType)


def test_step_cost_charges_every_step_alike():
    parameters = AgentParameters()
    reward_function = RewardFunction(function_dict={"step_cost": {"reward": -0.01}})
    rewards = [
        reward_function.get_reward(
            obs_dict={"simulation_state_container": parameters}, simulation_state_container=parameters
        )[0]
        for _ in range(3)
    ]
    assert rewards == pytest.approx([-0.01, -0.01, -0.01])


def test_symmetric_approach_pays_progress_and_takes_it_back_on_retreat():
    parameters = AgentParameters()
    reward_function = RewardFunction(
        function_dict={"approach_goal": {"pos_factor": 1.0, "neg_factor": 1.0, "_follow_subgoal": False}}
    )
    goal_x = 5.0
    rewards = []
    for x in (0.0, 0.3, 0.6, 0.3, 0.0):
        relative = np.array([goal_x - x, 0.0])
        obs = {
            "robot_pose": _pose(x),
            "goal_in_robot_frame": relative,
            "subgoal_in_robot_frame": relative,
            "simulation_state_container": parameters,
        }
        rewards.append(reward_function.get_reward(obs_dict=obs, simulation_state_container=parameters)[0])
    assert rewards == pytest.approx([0.0, 0.3, 0.3, -0.3, -0.3], abs=1e-5)
    assert sum(rewards) == pytest.approx(0.0, abs=1e-5)
