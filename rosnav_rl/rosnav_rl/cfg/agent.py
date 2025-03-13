from typing import Optional, Union

import rospy
from pydantic import BaseModel, Discriminator, Field, model_validator
from typing_extensions import Annotated

from rosnav_rl.utils.name_generator import generate_agent_name

from ..model.dreamerv3.cfg import DreamerV3Cfg
from ..model.stable_baselines3.cfg import StableBaselinesCfg
from .action_space import ActionSpaceCfg
from .framework import FrameworkCfg
from .reward import RewardCfg


class AgentCfg(BaseModel):
    """
    Configuration class for RL agents in ROS navigation.

    This class represents the configuration of an RL agent, including its name, 
    robot model, framework, reward function, and action space settings.

    Attributes:
        name (Optional[str]): Name of the agent. If None, it will be auto-generated
            based on the framework and robot.
        robot (Optional[str]): Robot model name. If None, it will be fetched from
            ROS parameters.
        framework (Union[StableBaselinesCfg, DreamerV3Cfg]): The RL framework
            configuration to use for this agent.
        reward (Optional[RewardCfg]): Configuration for the reward function.
        action_space (Optional[ActionSpaceCfg]): Configuration for the action space.
            Defaults to an empty ActionSpaceCfg.

    Validators:
        check_name: Automatically generates an agent name if one isn't provided and
            the framework information is available.
        check_robot: Ensures the robot model is set and matches the one in ROS params.
            Raises ValueError if there's a mismatch.
    """
    name: Optional[str] = Field(None, description="Unique name of the agent.")
    robot: Optional[str] = Field(None, description="Robot model name.", example="jackal")
    framework: Annotated[
        Union[StableBaselinesCfg, DreamerV3Cfg], 
        Discriminator(discriminator="name")
    ]
    reward: Optional[RewardCfg] = None
    action_space: Optional[ActionSpaceCfg] = ActionSpaceCfg()

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
    
    @model_validator(mode="after")
    def check_name(self):
        if self.name is None:
            self.name = generate_agent_name(self.framework, robot=self.robot)
        return self

    
