from typing import Literal, Optional

from pydantic import BaseModel


class FrameworkCfg(BaseModel):
    name: Literal["stable_baselines3"] = "stable_baselines3"
    model: BaseModel = None
    algorithm: Optional[BaseModel] = None
