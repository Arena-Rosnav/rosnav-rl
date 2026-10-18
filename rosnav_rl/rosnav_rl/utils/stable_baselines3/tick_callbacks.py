"""StableBaselines3 callbacks driven by the training tick count instead of evaluation results."""

from pathlib import Path
from typing import Any

from rclpy.node import Node
from stable_baselines3.common.callbacks import BaseCallback

from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase, StageInput


def stage_at(stage_ticks: list[int], ticks: int) -> int:
    """Index of the stage in force at `ticks`, stage i starting at stage_ticks[i]."""
    return max(0, sum(1 for start in stage_ticks if start <= ticks) - 1)


class TickScheduledCurriculum(CurriculumBase, BaseCallback):
    """Applies curriculum stage i once training reaches stage_ticks[i] control ticks, never on performance."""

    def __init__(
        self,
        node: Node,
        train_stages: StageInput,
        stage_ticks: list[int],
        ticks_per_step: int,
        num_envs: int,
        parameter_node_template: str,
        timeout: float = 5.0,
        verbose: int = 0,
        tm_dict: dict[str, Any] | None = None,
    ):
        self.stage_ticks = list(stage_ticks)
        self.ticks_per_step = ticks_per_step
        CurriculumBase.__init__(
            self,
            node=node,
            train_stages=train_stages,
            threshold_type="succ",
            upper_threshold=float("inf"),
            lower_threshold=float("-inf"),
            num_envs=num_envs,
            parameter_node_template=parameter_node_template,
            timeout=timeout,
            starting_stage=0,
            verbose=verbose,
            tm_dict=tm_dict,
        )
        BaseCallback.__init__(self, verbose=verbose)
        if (
            len(self.stage_ticks) != self.max_index
            or self.stage_ticks[0] != 0
            or self.stage_ticks != sorted(self.stage_ticks)
        ):
            raise ValueError("stage_ticks needs one ascending start tick per stage, the first at 0")
        print("[CURRICULUM] stage 0 at tick 0", flush=True)

    def get_current_performance(self) -> float | None:
        return None

    def reset_performance_tracking(self) -> None:
        pass

    def _on_step(self) -> bool:
        ticks = self.num_timesteps * self.ticks_per_step
        while self.curriculum_index < stage_at(self.stage_ticks, ticks):
            self.advance_curriculum()
            print(f"[CURRICULUM] stage {self.curriculum_index} at tick {ticks}", flush=True)
        return True


class TickCheckpointCallback(BaseCallback):
    """Saves the model as tick_<T>.zip the first time training reaches each of the given control ticks."""

    def __init__(self, save_dir: str | Path, ticks: list[int], ticks_per_step: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_dir = Path(save_dir)
        self.pending = sorted(ticks)
        self.ticks_per_step = ticks_per_step

    def _on_training_start(self) -> None:
        if self.num_timesteps:
            self.pending = [t for t in self.pending if t > self.num_timesteps * self.ticks_per_step]

    def _on_step(self) -> bool:
        ticks = self.num_timesteps * self.ticks_per_step
        while self.pending and ticks >= self.pending[0]:
            target = self.pending.pop(0)
            self.model.save(self.save_dir / f"tick_{target}")
            print(f"[CHECKPOINT] tick_{target} saved at tick {ticks}", flush=True)
        return True
