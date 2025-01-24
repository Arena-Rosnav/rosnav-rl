import enum
from typing import Type, ClassVar

from ..collectors import BaseUnit

__all__ = ["DoneObservation", "DONE_REASONS"]


class DONE_REASONS(enum.Enum):
    STEP_LIMIT = enum.auto()
    COLLISION = enum.auto()
    SUCCESS = enum.auto()


class DoneObservation(BaseUnit):
    name: ClassVar[str] = "done"
    data_class: Type[bool] = bool
