import json
import os
from typing import Tuple

import numpy as np
import rospkg
import rospy
import yaml

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gym import spaces
from rosnav.utils.constants import RosnavEncoder


def get_robot_yaml_path(robot_model: str = None) -> str:
    robot_model = rospy.get_param(os.path.join(rospy.get_namespace(), "robot_model"))

    simulation_setup_path = rospkg.RosPack().get_path("arena-simulation-setup")
    return os.path.join(
        simulation_setup_path, "robot", robot_model, f"model_params.yaml"
    )


def get_laser_from_robot_yaml(robot_model: str = None) -> Tuple[int, int, int, int]:
    robot_yaml_path = get_robot_yaml_path(robot_model)

    with open(robot_yaml_path, "r") as fd:
        robot_data = yaml.safe_load(fd)
        laser_data = robot_data["laser"]

        rospy.set_param(
            os.path.join(rospy.get_namespace(), "laser/num_beams"),
            laser_data["num_beams"],
        )

        return (
            laser_data["num_beams"],
            laser_data["angle"]["min"],
            laser_data["angle"]["max"],
            laser_data["angle"]["increment"],
        )


def get_observation_space_from_file(robot_model: str = None) -> Tuple[int, int]:
    robot_state_size, action_state_size = 2, rospy.get_param(
        rospy.get_namespace() + "action_state_size", 3
    )
    num_beams, _, _, _ = get_laser_from_robot_yaml(robot_model)

    num_beams = RosnavEncoder[get_robot_space_encoder()]["lasers_to_adapted"](num_beams)

    return num_beams, action_state_size + robot_state_size


def get_robot_space_encoder() -> str:
    return rospy.get_param("space_encoder", "DefaultEncoder")


def get_observation_space() -> Tuple[int, int]:
    observation_space = RosnavEncoder[get_robot_space_encoder()]

    return observation_space["lasers"], observation_space["meta"]


def stack_spaces(*ss) -> spaces.Box:
    low = []
    high = []

    for space in ss:
        low.extend(space.low.tolist())
        high.extend(space.high.tolist())

    return spaces.Box(np.array(low).flatten(), np.array(high).flatten())


def stack_stacked_spaces(*ss) -> spaces.Box:
    low = []
    high = []

    for space in ss:
        low.extend(space.low.tolist())
        high.extend(space.high.tolist())

    return spaces.Box(
        np.expand_dims(np.array(low), axis=0), np.expand_dims(np.array(high), axis=0)
    )


def load_json(file_path: str) -> dict:
    with open(file_path) as file:
        return json.load(file)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def make_mock_env(config: dict) -> DummyVecEnv:
    import rl_utils.envs.flatland_gym_env as flatland_gym_env

    def _init():
        return flatland_gym_env.FlatlandEnv(
            ns="",
            reward_fnc=config["reward_fnc"],
            is_action_space_discrete=config["discrete_action_space"],
            requires_task_manager=False,
        )

    return DummyVecEnv([_init])


def load_vec_normalize(path: str, config: dict) -> VecNormalize:
    return VecNormalize.load(path, make_mock_env(config))
