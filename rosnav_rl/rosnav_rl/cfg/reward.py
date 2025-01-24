from typing import Any, Dict, Optional
from pydantic import BaseModel

RewardUnitDict = Dict[str, Any]
RewardFunctionDict = Dict[str, RewardUnitDict]


class RewardCfg(BaseModel):
    reward_function_dict: RewardFunctionDict
    reward_unit_kwargs: Optional[dict] = None
    verbose: Optional[bool] = False
