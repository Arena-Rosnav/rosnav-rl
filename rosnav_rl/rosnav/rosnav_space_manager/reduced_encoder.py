from gym import spaces
import numpy as np
import math

from ..utils.utils import stack_spaces
from .encoder_factory import BaseSpaceEncoderFactory
from .base_space_encoder import BaseSpaceEncoder

from ..utils.constants import REDUCTION_FACTOR, RosnavEncoder

"""

    TODO
    This encoder offers a robot specific observation and action space
    Different actions spaces for holonomic and non holonomic robots

    Observation space:   Laser Scan, Goal, Current Vel 
    Action space: X Vel, (Y Vel), Angular Vel

"""


@BaseSpaceEncoderFactory.register("ReducedEncoder")
class ReducedEncoder(BaseSpaceEncoder):
    def __init__(self, *args):
        super().__init__(*args)

        rest = self._laser_num_beams % REDUCTION_FACTOR
        self._laser_append_amount = 0 if rest == 0 else REDUCTION_FACTOR - rest

    def decode_action(self, action):
        if self._is_action_space_discrete:
            return self._translate_disc_action(action)
        return self._extend_action_array(action)

    def _extend_action_array(self, action: np.ndarray) -> np.ndarray:
        if self._is_holonomic:
            assert (
                self._is_holonomic and len(action) == 3
            ), "Robot is holonomic but action with only two freedoms of movement provided"

            return action
        else:
            assert (
                not self._is_holonomic and len(action) == 2
            ), "Robot is non-holonomic but action with more than two freedoms of movement provided"
            return np.array([action[0], 0, action[1]])

    def _translate_disc_action(self, action):
        assert not self._is_holonomic, "Discrete action space currently not supported for holonomic robots"
        
        return np.array([
            self._actions[action]["linear"], 
            self._actions[action]["linear"]
        ])

    def encode_observation(self, observation, structure):
        # rho, theta = observation["goal_in_robot_frame"]
        # scan = observation["laser_scan"]
        # last_action = observation["last_action"]

        new_obs_space = []

        for name in structure:
            data = observation[name]

            if name == "laser_scan":
                data = np.pad(data, [(0, self._laser_append_amount)], constant_values=self._laser_max_range)

                data = np.array_split(data, len(data) / REDUCTION_FACTOR)
                data = [min(d) for d in data]
            
            new_obs_space.append(data)

        return np.hstack(new_obs_space)

    def get_observation_space(self):
        return stack_spaces(
            spaces.Box(
                low=0,
                high=self._laser_max_range,
                shape=(math.floor(self._laser_num_beams / REDUCTION_FACTOR),),
                dtype=np.float32,
            ),
            spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),
            spaces.Box(
                low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
            ),
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

    def get_action_space(self):
        if self._is_action_space_discrete:
                # self._discrete_actions is a list, each element is a dict with the keys ["name", 'linear','angular']
                return spaces.Discrete(len(self._actions))

        linear_range = self._actions["linear_range"]
        angular_range = self._actions["angular_range"]

        if not self._is_holonomic:
            return spaces.Box(
                low=np.array([linear_range[0], angular_range[0]]),
                high=np.array([linear_range[1], angular_range[1]]),
                dtype=np.float32,
            )

        linear_range_x, linear_range_y = (
            linear_range["x"],
            linear_range["y"],
        )
        
        return spaces.Box(
            low=np.array(
                [
                    linear_range_x[0],
                    linear_range_y[0],
                    angular_range[0],
                ]
            ),
            high=np.array(
                [
                    linear_range_x[1],
                    linear_range_y[1],
                    angular_range[1],
                ]
            ),
            dtype=np.float32,
        )

