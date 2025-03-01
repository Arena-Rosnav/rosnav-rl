from typing import Union, ClassVar
from abc import ABC

from pydantic import BaseModel
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks


class FrameworkCfg(BaseModel, ABC):
    __name__: ClassVar[Union[str, SupportedRLFrameworks]]
