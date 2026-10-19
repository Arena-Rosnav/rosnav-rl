"""Arm goal generators on real poses."""

import numpy as np

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.data_sources.generators import (
    ArmDistAngleToGoalGenerator,
    ArmGoalLocationInEndEffectorFrameGenerator,
)

PARAMS = AgentParameters()


def test_the_arm_goal_is_expressed_in_the_end_effector_frame():
    gen = ArmGoalLocationInEndEffectorFrameGenerator(name="arm_goal_in_ee_frame")
    quarter_turn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)

    relative = gen.get_observation({"end_effector_pose": quarter_turn}, simulation_state_container=PARAMS)

    np.testing.assert_allclose(relative[:3], [0.0, -0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(np.abs(relative[3:]), [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-6)


def test_distance_and_rotation_error_come_from_the_relative_goal():
    gen = ArmDistAngleToGoalGenerator(name="arm_dist_angle_to_goal")
    relative = np.array([0.3, 0.4, 0.0, 0.0, 0.0, np.sin(0.25), np.cos(0.25)], dtype=np.float32)

    metrics = gen.get_observation({"arm_goal_in_ee_frame": relative}, simulation_state_container=PARAMS)

    np.testing.assert_allclose(metrics, [0.5, 0.5], atol=1e-6)
