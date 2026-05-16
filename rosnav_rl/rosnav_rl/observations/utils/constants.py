import enum

MAX_WAIT = 5  # in seconds
SLEEP = 0.05  # in seconds


class DONE_REASONS(enum.Enum):
    STEP_LIMIT = enum.auto()
    COLLISION = enum.auto()
    SUCCESS = enum.auto()
