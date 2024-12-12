from typing import Optional, Union
from abc import ABC

from pydantic import BaseModel
from rl_utils.utils.type_alias.enums import RLFramework


class FrameworkCfg(BaseModel, ABC):
    name: Union[str, RLFramework]
    algorithm: BaseModel
