from typing import Literal

from pydantic import BaseModel


class StopTrainingOnThreshCfg(BaseModel):
    threshold_type: Literal["rew", "succ"] = "succ"
    threshold: float = 0.9
    verbose: int = 1


class PeriodicEvaluationCfg(BaseModel):
    n_eval_episodes: int = 40
    eval_freq: int = 20000
    max_num_moves_per_eps: int = 250


class CheckpointCfg(BaseModel):
    save_freq: int = 250000
    save_path: str | None = None  # if None will be saved in the same directory as the model
    name_prefix: str = "model"
    save_replay_buffer: bool = True
    save_vecnormalize: bool = True


class CallbacksCfg(BaseModel):
    periodic_evaluation: PeriodicEvaluationCfg | None = PeriodicEvaluationCfg()
    checkpoint: CheckpointCfg | None = None
    stop_training_on_threshold: StopTrainingOnThreshCfg | None = StopTrainingOnThreshCfg()
