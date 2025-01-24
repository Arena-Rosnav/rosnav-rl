from datetime import datetime as dt
from typing import TYPE_CHECKING

import rospy

if TYPE_CHECKING:
    from rosnav_rl.cfg import FrameworkCfg


def generate_name(architecture_name: str, algorithm: str, robot: str = None) -> str:
    start_time = dt.now().strftime("%Y_%m_%d__%H_%M_%S")
    robot_model = rospy.get_param("robot_model", "") if robot is None else robot
    agent_name = f"{robot_model}_{algorithm}_{architecture_name}_{start_time}"
    return agent_name


def generate_sb3_agent_name(framework_cfg: "FrameworkCfg", robot: str = None) -> str:
    return generate_name(
        framework_cfg.algorithm.architecture_name,
        framework_cfg.algorithm.__algorithm_name__,
        robot=robot,
    )


def generate_agent_name(framework_cfg: "FrameworkCfg", robot: str = None) -> str:
    import rosnav_rl.utils.type_aliases as type_aliases

    if framework_cfg.__name__ == type_aliases.SupportedRLFrameworks.STABLE_BASELINES3:
        return generate_sb3_agent_name(framework_cfg, robot=robot)
    else:
        raise ValueError(
            f"Framework {framework_cfg.name} not supported for name generation."
        )
