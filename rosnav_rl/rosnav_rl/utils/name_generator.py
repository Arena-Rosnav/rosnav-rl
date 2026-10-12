from __future__ import annotations

from datetime import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg
    from rosnav_rl.model.stable_baselines3.cfg import StableBaselinesCfg


def _create_formatted_name(*args: str) -> str:
    """
    Create a formatted name by joining the input arguments and appending a timestamp.

    Args:
        *args (str): Variable length argument list of strings to be joined.

    Returns:
        str: A formatted string where the arguments are joined with underscores
             followed by the current timestamp in the format 'YYYY_MM_DD__HH_MM_SS'.

    Example:
        >>> _create_formatted_name("model", "v1")
        'model_v1_2023_04_15__14_30_22'
    """
    formatted_args = "_".join(str(a) for a in args if a is not None)
    return f"{formatted_args}_{dt.now().strftime('%Y_%m_%d__%H_%M_%S')}"


def generate_agent_name(framework_cfg: StableBaselinesCfg | DreamerV3Cfg, robot: str | None = None) -> str:
    import rosnav_rl.model.stable_baselines3.cfg as sb3_cfg

    robot_name = robot if robot else "[no_robot_specified]"
    if isinstance(framework_cfg, sb3_cfg.StableBaselinesCfg):
        return _create_formatted_name(
            framework_cfg.algorithm.architecture_name,
            framework_cfg.algorithm.parameters.algorithm_name,
            robot_name,
        )
    return _create_formatted_name(framework_cfg.name, robot_name)
