from typing import Any, Callable, Optional

from pydantic import AfterValidator
from rosnav_rl.utils.model.learning_rate_schedules import load_lr_schedule
from typing_extensions import Annotated


def validate_learning_rate_scheduler_type(value):
    # check if value can be imported from learning_rate_schedules
    try:
        load_lr_schedule(value, {})
    except Exception as e:
        raise ValueError(f"Invalid learning rate scheduler type: {value}. Error: {e}")


SchedulerType = Annotated[str, AfterValidator(validate_learning_rate_scheduler_type)]


class LearningRateSchedulerCfg(BaseModel):
    type: SchedulerType = "linear"
    kwargs: dict = {"initial_value": 0.001, "final_value": 0.0001}
    _callable: Optional[Callable[[Any], Callable[[float], float]]] = None

    def __init__(self, **data):
        super().__init__(**data)
        # self._callable = load_lr_schedule(self.type, self.kwargs)
