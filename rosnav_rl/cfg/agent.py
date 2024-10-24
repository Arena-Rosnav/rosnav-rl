from datetime import datetime as dt
from typing import Optional

import rospy
from pydantic import BaseModel, model_validator

from .reward import RewardCfg
from .stable_baselines3.ppo import ActionSpaceCfg, PPO_Policy_Cfg


class AgentCfg(BaseModel):
    name: Optional[str] = None
    robot: Optional[str] = None
    policy: PPO_Policy_Cfg
    reward: Optional[RewardCfg] = None
    action_space: Optional[ActionSpaceCfg] = ActionSpaceCfg()

    @model_validator(mode="after")
    def check_name(self):
        if self.name is None:

            def generate_agent_name(architecture_name: str):
                START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M_%S")
                robot_model = (
                    rospy.get_param("robot_model", "")
                    if self.robot is None
                    else self.robot
                )
                agent_name = f"{robot_model}_{architecture_name}_{START_TIME}"
                return agent_name

            self.name = generate_agent_name(self.policy.architecture_name)

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
