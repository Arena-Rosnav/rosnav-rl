import math

import numpy as np
import rospy
from gymnasium import spaces

from ..utils.observation_space.observation_space_manager import ObservationSpaceManager
from ..utils.observation_space.space_index import SPACE_FACTORY_KEYS
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
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
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
        return ObservationSpaceManager(
            [
                SPACE_FACTORY_KEYS.LASER.name,
                SPACE_FACTORY_KEYS.GOAL.name,
                SPACE_FACTORY_KEYS.LAST_ACTION.name,
            ],
            enable_frame_stacking=self._stacked,
            space_kwargs={
                "laser_num_beams": self._reduced_num_laser_beams,
                "laser_max_range": self._laser_max_range,
                "goal_max_dist": 20,
                "min_linear_vel": -2.0,
                "max_linear_vel": -2.0,
                "min_angular_vel": -4.0,
                "max_angular_vel": 4.0,
            },
        ).unified_observation_space

    @staticmethod
    def reduce_laserbeams(laserbeams: np.ndarray, x: int) -> np.ndarray:
        if x >= len(laserbeams):
            return laserbeams
        indices = np.round(np.linspace(0, len(laserbeams) - 1, x)).astype(int)[:x]
        if type(laserbeams) == tuple:
            laserbeams = np.array(laserbeams)
        return laserbeams[indices]
