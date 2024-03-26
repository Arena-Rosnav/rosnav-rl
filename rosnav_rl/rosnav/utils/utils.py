import json
import os
from typing import Tuple

import numpy as np
import rospkg
import rospy
import torch
import yaml
from gymnasium import spaces
from rosnav.utils.constants import RosnavEncoder
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def get_robot_yaml_path(robot_model: str = None) -> str:
    robot_model = rospy.get_param(os.path.join(rospy.get_namespace(), "model"))

    simulation_setup_path = rospkg.RosPack().get_path("arena_simulation_setup")
    return os.path.join(
        simulation_setup_path, "entities", "robots", robot_model, f"model_params.yaml"
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


def get_actions_from_robot_yaml(robot_model: str = None):
    robot_yaml_path = get_robot_yaml_path(robot_model)

    with open(robot_yaml_path, "r") as fd:
        robot_data = yaml.safe_load(fd)
        action_data = robot_data["actions"]

    return action_data


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


def load_json(file_path: str) -> dict:
    with open(file_path) as file:
        return json.load(file)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def make_mock_env(ns: str, agent_description) -> DummyVecEnv:
    import rl_utils.envs.flatland_gymnasium_env as flatland_gym_env

    def _init():
        return flatland_gym_env.FlatlandEnv(
            ns=ns,
            agent_description=agent_description,
            reward_fnc=None,
            trigger_init=False,
        )

    return DummyVecEnv([_init])


def wrap_vec_framestack(env: DummyVecEnv, stack_size: int) -> VecFrameStack:
    return VecFrameStack(env, n_stack=stack_size, channels_order="first")


def load_vec_normalize(path: str, venv: VecEnv = None) -> VecNormalize:
    return VecNormalize.load(path, venv)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
