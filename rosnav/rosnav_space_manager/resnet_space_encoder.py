from typing import List

from pedsim_msgs.msg import SemanticDatum

from ..utils.observation_space.space_index import SPACE_INDEX
from .default_encoder import DefaultEncoder
from .encoder_factory import BaseSpaceEncoderFactory


@BaseSpaceEncoderFactory.register("SemanticResNetSpaceEncoder")
class SemanticResNetSpaceEncoder(DefaultEncoder):
    default_observation_list = [
        SPACE_INDEX.STACKED_LASER_MAP,
        SPACE_INDEX.PEDESTRIAN_LOCATION,
        SPACE_INDEX.PEDESTRIAN_TYPE,
        SPACE_INDEX.GOAL,
    ]

    def __init__(
        self,
        observation_list: List[SPACE_INDEX] = None,
        observation_kwargs: dict = None,
        *args,
        **kwargs
    ):
        if not observation_list:
            observation_list = self.default_observation_list

        if not observation_kwargs:
            observation_kwargs = {
                "roi_in_m": 20,
                "feature_map_size": 80,
                "laser_stack_size": 10,
                **observation_kwargs,
            }

        super().__init__(
            observation_list=observation_list,
            observation_kwargs=observation_kwargs,
            **kwargs
        )
