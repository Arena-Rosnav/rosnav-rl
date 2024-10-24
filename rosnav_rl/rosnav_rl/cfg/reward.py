from typing import Any, Dict, Optional
from pydantic import BaseModel

RewardUnitDict = Dict[str, Any]
RewardFunctionDict = Dict[str, RewardUnitDict]


class RewardCfg(BaseModel):
    file_name: str
    reward_unit_kwargs: Optional[dict] = None
    verbose: Optional[bool] = False
    _reward_dict: Optional[RewardFunctionDict] = None
