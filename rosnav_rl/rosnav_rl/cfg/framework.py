from typing import Union, ClassVar
from abc import ABC

from pydantic import BaseModel
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks


class FrameworkCfg(BaseModel, ABC):
    """
    Abstract base class to define the configuration for a reinforcement learning framework.

    This class establishes the common blueprint for all RL framework configurations.
    Each framework implementation should inherit from this class and define its specific
    configuration parameters.

    Attributes:
        __name__ (ClassVar[Union[str, SupportedRLFrameworks]]): Class variable storing 
            the name or enum value identifying the RL framework.
    """
    name: Union[str, SupportedRLFrameworks] = ""
