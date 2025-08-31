"""StableBaselines3-specific curriculum learning callback."""

from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from rclpy.node import Node

from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase


class StagedTrainCallback(CurriculumBase, BaseCallback):
    """StableBaselines3 curriculum learning callback.

    This class combines the StableBaselines3 BaseCallback interface with the
    framework-agnostic CurriculumBase for curriculum learning functionality.

    It integrates curriculum progression with SB3's evaluation cycle and
    provides automatic parameter updates to ROS2 task generator nodes.
    """

    def __init__(
        self,
        node: Node,
        train_stages: dict,
        threshold_type: str,
        upper_threshold: float,
        lower_threshold: float,
        num_envs: int,
        verbose: int = 0,
    ):
        """Initialize the staged training callback.

        Args:
            node: ROS2 node for parameter communication
            train_stages: Dict mapping parameter names to lists of stage values
            threshold_type: Type of threshold ('rew' for reward, 'succ' for success)
            upper_threshold: Threshold to advance to next stage
            lower_threshold: Threshold to retreat to previous stage
            num_envs: Number of environments
            verbose: Verbosity level
        """
        # Initialize CurriculumBase first to satisfy its required args and
        # avoid BaseCallback.__init__ accidentally invoking CurriculumBase.__init__
        CurriculumBase.__init__(
            self,
            node=node,
            train_stages=train_stages,
            threshold_type=threshold_type,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            num_envs=num_envs,
            verbose=verbose,
        )

        BaseCallback.__init__(self, verbose=verbose)

    def get_current_performance(self) -> Optional[float]:
        """Get current performance from the parent EvalCallback.

        Returns:
            Current performance metric or None if not available
        """
        if not hasattr(self, "parent") or self.parent is None:
            return None

        eval_callback = self.parent

        if self.threshold_type == "rew":
            return eval_callback.best_mean_reward
        elif self.threshold_type == "succ":
            return getattr(eval_callback, "last_success_rate", 0.0)
        else:
            return None

    def reset_performance_tracking(self) -> None:
        """Reset performance tracking in the parent EvalCallback."""
        if not hasattr(self, "parent") or self.parent is None:
            return

        eval_callback = self.parent

        if self.threshold_type == "rew":
            eval_callback.best_mean_reward = float("-inf")
        elif self.threshold_type == "succ":
            if hasattr(eval_callback, "last_success_rate"):
                eval_callback.last_success_rate = 0.0

    def _on_step(self) -> bool:
        """Called on each training step. Checks thresholds and updates curriculum."""
        return self.check_thresholds_and_update()
