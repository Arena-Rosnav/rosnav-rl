import math

import numpy as np
import rospy
from gym import spaces

from ..utils.utils import stack_spaces
from .base_space_encoder import BaseSpaceEncoder
from .encoder_factory import BaseSpaceEncoderFactory
from .default_encoder import DefaultEncoder

"""

    TODO
    This encoder offers a robot specific observation and action space
    Different actions spaces for holonomic and non holonomic robots

    Observation space:   Laser Scan, Goal, Current Vel 
    Action space: X Vel, (Y Vel), Angular Vel

"""


@BaseSpaceEncoderFactory.register("ReducedLaserEncoder")
class ReducedLaserEncoder(DefaultEncoder):
    def __init__(self, *args):
        super().__init__(*args)
        self._reduced_num_laser_beams = rospy.get_param(
            "laser/reduced_num_laser_beams", self._laser_num_beams
        )

    def encode_observation(self, observation: dict, structure: list) -> np.ndarray:
        if "laser_scan" in structure:
            observation["laser_scan"] = ReducedLaserEncoder.reduce_laserbeams(
                observation["laser_scan"], self._reduced_num_laser_beams
            )

        return super().encode_observation(observation, structure)

    def get_observation_space(self):
        return stack_spaces(
            spaces.Box(
                low=0,
                high=self._laser_max_range,
                shape=(self._reduced_num_laser_beams,),
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

    @staticmethod
    def reduce_laserbeams(laserbeams: np.ndarray, x: int) -> np.ndarray:
        if x >= len(laserbeams):
            return laserbeams
        indices = np.round(np.linspace(0, len(laserbeams) - 1, x)).astype(int)[:x]
        return laserbeams[indices]
