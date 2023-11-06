import numpy as np
from gymnasium import spaces

from ..utils.utils import stack_stacked_spaces
from .default_encoder import DefaultEncoder
from .encoder_factory import BaseSpaceEncoderFactory

"""
    This encoder offers a robot specific observation and action space
    Different actions spaces for holonomic and non holonomic robots

    Observation space:   Laser Scan, Goal, Current Vel 
    Action space: X Vel, (Y Vel), Angular Vel

"""


@BaseSpaceEncoderFactory.register("StackedEncoder")
class StackedSpaceEncoder(DefaultEncoder):
    def get_observation_space(self):
        return stack_stacked_spaces(
            spaces.Box(
                low=0,
                high=self._laser_max_range,
                shape=(self._laser_num_beams,),
                dtype=np.float32,
            ),
            spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(2,),
                dtype=np.float32,  # linear vel
            ),
            spaces.Box(
                low=-4.0,
                high=4.0,
                shape=(1,),
                dtype=np.float32,  # angular vel
            ),
        )

    def encode_observation(self, observation, structure):
        return np.expand_dims(super().encode_observation(observation, structure), 0)

    def decode_action(self, action):
        action_to_parse = action[0] if action.ndim == 2 else action
        return super().decode_action(action_to_parse)
