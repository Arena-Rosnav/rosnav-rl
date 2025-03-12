from typing import Any, Dict, Optional
from pydantic import BaseModel

RewardUnitDict = Dict[str, Any]
RewardFunctionDict = Dict[str, RewardUnitDict]


class RewardCfg(BaseModel):
    """
    Configuration class for reward in RL-based navigation.

    This class encapsulates the configuration for reward functions and their arguments
    in a reinforcement learning navigation setup.

    Attributes:
        reward_function_dict (RewardFunctionDict): Dictionary containing the reward functions 
                                                  to be used during training.
        reward_unit_kwargs (dict, optional): Additional keyword arguments for reward unit 
                                            configuration. Defaults to None.
        verbose (bool, optional): Flag to enable verbose logging of reward calculations. 
                                 Defaults to False.
    """
    reward_function_dict: RewardFunctionDict
    reward_unit_kwargs: Optional[dict] = None
    verbose: Optional[bool] = False
