import numpy as np
from gymnasium import spaces
from pedsim_agents.utils import SemanticAttribute
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


class OBS_SPACES:
    BASIC = {
        OBS_DICT_KEYS.LAST_ACTION: (
            spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(2,),
                dtype=np.float32,
            ),
            spaces.Box(
                low=-4.0,
                high=4.0,
                shape=(1,),
                dtype=np.float32,
            ),
        ),
        OBS_DICT_KEYS.GOAL: (
            spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        ),
    }

    FEATURE_MAP = {
        SemanticAttribute.IS_PEDESTRIAN: lambda map_size: spaces.Box(
            low=0,
            high=1,
            shape=(map_size * map_size,),
            dtype=int,
        ),
        SemanticAttribute.PEDESTRIAN_TYPE: lambda map_size: spaces.Box(
            low=0,
            high=5,
            shape=(map_size * map_size,),
            dtype=int,
        ),
        SemanticAttribute.PEDESTRIAN_VEL_X: lambda map_size: spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(map_size * map_size,),
            dtype=np.float32,
        ),
        SemanticAttribute.PEDESTRIAN_VEL_Y: lambda map_size: spaces.Box(
            low=-6.0,
            high=6.0,
            shape=(map_size * map_size,),
            dtype=np.float32,
        ),
    }
