import math

import numpy as np
import rospy
from gymnasium import spaces

from ..utils.observation_space.observation_space_manager import ObservationSpaceManager
from .base_space_encoder import BaseSpaceEncoder
from .default_encoder import DefaultEncoder
from .encoder_factory import BaseSpaceEncoderFactory

"""

    TODO
    This encoder offers a robot specific observation and action space
    Different actions spaces for holonomic and non holonomic robots

    Observation space:   Laser Scan, Goal, Current Vel 
    Action space: X Vel, (Y Vel), Angular Vel

"""


@BaseSpaceEncoderFactory.register("ReducedLaserEncoder")
class ReducedLaserEncoder(DefaultEncoder):
    def __init__(self, observation_kwargs: dict = None, *args, **kwargs):
        self._reduced_num_laser_beams = rospy.get_param(
            "laser/reduced_num_laser_beams", self._laser_num_beams
        )
        observation_kwargs["laser_num_beams"] = self._reduced_num_laser_beams
        super().__init__(observation_kwargs=observation_kwargs, **kwargs)

    def encode_observation(self, observation: dict, structure: list) -> np.ndarray:
        observation["laser_scan"] = ReducedLaserEncoder.reduce_laserbeams(
            observation["laser_scan"], self._reduced_num_laser_beams
        )

        return super().encode_observation(observation, structure)

    @staticmethod
    def reduce_laserbeams(laserbeams: np.ndarray, x: int) -> np.ndarray:
        if x >= len(laserbeams):
            return laserbeams
        indices = np.round(np.linspace(0, len(laserbeams) - 1, x)).astype(int)[:x]
        if type(laserbeams) == tuple:
            laserbeams = np.array(laserbeams)
        return laserbeams[indices]
