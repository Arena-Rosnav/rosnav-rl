from pydantic import BaseModel


class LearningRateSchedulerCfg(BaseModel):
    type: str = "linear"
    kwargs: dict = {"initial_value": 0.001, "final_value": 0.0001}
