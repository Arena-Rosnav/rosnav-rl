from typing import Any, Callable, Optional

from pydantic import BaseModel, model_validator
from rosnav_rl.utils.stable_baselines3.model.learning_rate_schedules import (
    load_lr_schedule,
)


class LearningRateSchedulerCfg(BaseModel):
    type: str = "linear"
    kwargs: dict = {"initial_value": 0.001, "final_value": 0.0001}
    callable: Optional[Callable[[Any], Callable[[float], float]]] = None

    @model_validator(mode="after")
    def load_scheduler(self):
        self.callable = load_lr_schedule(self.type, self.kwargs)
        return self
