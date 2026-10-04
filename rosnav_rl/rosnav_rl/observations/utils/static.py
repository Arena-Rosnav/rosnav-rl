import enum
from typing import ClassVar


class DONE_REASONS(enum.Enum):
    STEP_LIMIT = enum.auto()
    COLLISION = enum.auto()
    SUCCESS = enum.auto()


class IsDone:
    name: ClassVar[str] = "is_done"
    data_class: type[bool] = bool
