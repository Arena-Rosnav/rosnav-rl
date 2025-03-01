from typing import Optional, Union, Any

import rospy
from pydantic import BaseModel, model_validator, Field

from rosnav_rl.utils.name_generator import generate_agent_name

from .sb3_cfg.framework import StableBaselinesCfg
from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg

from .reward import RewardCfg
from .action_space.action_space import ActionSpaceCfg


class AgentCfg(BaseModel):
    name: Optional[str] = None
    robot: Optional[str] = None
    framework: Union[StableBaselinesCfg, DreamerV3Cfg]
    reward: Optional[RewardCfg] = None
    action_space: Optional[ActionSpaceCfg] = ActionSpaceCfg()

    @model_validator(mode="after")
    def check_name(self):
        if self.name is None and (
            hasattr(self.framework, "algorithm") or "algorithm" in self.framework
        ):
            self.name = generate_agent_name(self.framework, robot=self.robot)
        return self

    @model_validator(mode="after")
    def check_robot(self):
        if self.robot is None:
            self.robot = rospy.get_param("model")
        else:
            if rospy.get_param("model") != self.robot:
                raise ValueError(
                    "Robot model in config does not match the one in ROS params."
                )
        return self
