from typing import Any, Dict, Optional
from rosnav_rl.reward.utils import load_rew_fnc
from pydantic import BaseModel, field_validator

RewardUnitDict = Dict[str, Any]
RewardFunctionDict = Dict[str, RewardUnitDict]


class RewardCfg(BaseModel):
    file_name: str
    reward_unit_kwargs: Optional[dict] = None
    verbose: Optional[bool] = False
    _reward_dict: Optional[RewardFunctionDict] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str):
        try:
            load_rew_fnc(value)
        except Exception as e:
            raise ValueError(
                f"Invalid reward function name: {value}. Error: {e}"
            ) from e
        return value

    def load_reward_fnc(self):
        self._reward_dict = load_rew_fnc(self.file_name)
