from typing import List

from pedsim_msgs.msg import SemanticDatum

from ..utils.observation_space.space_index import SPACE_INDEX
from .default_encoder import DefaultEncoder
from .encoder_factory import BaseSpaceEncoderFactory


@BaseSpaceEncoderFactory.register("SemanticResNetSpaceEncoder")
class SemanticResNetSpaceEncoder(DefaultEncoder):
    """
    Encoder class for semantic ResNet space encoding.

    Args:
        observation_list (List[SPACE_INDEX], optional): List of observation spaces to encode. Defaults to None.
        observation_kwargs (dict, optional): Additional keyword arguments for observations. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        DEFAULT_OBS_LIST (List[SPACE_INDEX]): Default list of observation spaces to encode.

    """

    DEFAULT_OBS_LIST = [
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
        """
        Initializes the SemanticResNetSpaceEncoder.

        Args:
            observation_list (List[SPACE_INDEX], optional): List of observation spaces to encode. Defaults to None.
            observation_kwargs (dict, optional): Additional keyword arguments for observations. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        if not observation_list:
            observation_list = self.DEFAULT_OBS_LIST

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
