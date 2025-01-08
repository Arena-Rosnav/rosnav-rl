from typing import Union
from abc import ABC

from pydantic import BaseModel
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks


class FrameworkCfg(BaseModel, ABC):
    name: Union[str, SupportedRLFrameworks]
    algorithm: BaseModel
