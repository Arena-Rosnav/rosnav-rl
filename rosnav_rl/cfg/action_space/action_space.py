from typing import Optional

from pydantic import BaseModel
from rosnav_rl.utils.action_space.custom_discrete_action import (
    generate_discrete_action_dict,
)


class DiscreteFromBoxActionSpaceCfg(BaseModel):

    buckets_linear_vel: int
    buckets_angular_vel: int

    def generate_discrete_from_box_dict(
        self, linear_range: tuple, angular_range: tuple
    ) -> dict:
        """
        Generate a discrete action dictionary based on the given linear and angular ranges.

        Args:
            linear_range (tuple): The linear velocity range depending on the robot.
            angular_range (tuple): The angular velocity range depending on the robot.

        Returns:
            list: A list of discrete actions.
        """
        return generate_discrete_action_dict(
            linear_range,
            angular_range,
            self.buckets_linear_vel,
            self.buckets_angular_vel,
        )


class ActionSpaceCfg(BaseModel):
    is_discrete: Optional[bool] = False
    custom_discretization: Optional[DiscreteFromBoxActionSpaceCfg] = (
        DiscreteFromBoxActionSpaceCfg(buckets_angular_vel=16, buckets_linear_vel=12)
    )
