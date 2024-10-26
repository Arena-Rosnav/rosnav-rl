from datetime import datetime as dt
from typing import TYPE_CHECKING, Literal

import rospy

if TYPE_CHECKING:
    from rosnav_rl.cfg import FrameworkCfg, SB3_Model_Cfg


def generate_name(architecture_name: str, robot: str = None) -> str:
    start_time = dt.now().strftime("%Y_%m_%d__%H_%M_%S")
    robot_model = rospy.get_param("robot_model", "") if robot is None else robot
    agent_name = f"{robot_model}_{architecture_name}_{start_time}"
    return agent_name


def generate_sb3_agent_name(framework_cfg: "SB3_Model_Cfg", robot: str = None) -> str:
    return generate_name(framework_cfg.model.architecture_name, robot=robot)


def generate_agent_name(framework_cfg: "FrameworkCfg", robot: str = None) -> str:
    if framework_cfg.name == "stable_baselines3":
        return generate_sb3_agent_name(framework_cfg, robot=robot)
    else:
        raise ValueError(
            f"Framework {framework_cfg.name} not supported for name generation."
        )
