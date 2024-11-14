from typing import Optional, Union

from pydantic import BaseModel
from rl_utils.utils.type_alias.enums import RLFramework
from .sb3_cfg.base import BaseAlgorithmCfg


class FrameworkCfg(BaseModel):
    name: Union[str, RLFramework]
    algorithm: BaseAlgorithmCfg
