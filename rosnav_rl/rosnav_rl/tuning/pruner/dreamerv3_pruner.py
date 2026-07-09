"""DreamerV3-specific Optuna trial pruner.

``DreamerV3TrialPruner`` mirrors the ``DreamerV3Curriculum`` adapter:
it extends the framework-agnostic ``TrialPrunerBase`` and bridges the
DreamerV3 training loop via the ``after_eval_fn`` hook parameter.

Usage::

    from rosnav_rl.tuning.dreamerv3_pruner import DreamerV3TrialPruner

    pruner = DreamerV3TrialPruner(trial, metric="eval_return")
    model.train(
        train_envs=...,
        eval_envs=...,
        after_eval_fn=pruner.after_eval_hook,
    )
    best = pruner.best_metric

When the DreamerV3 ``helper.train()`` loop finishes an evaluation phase it
calls ``after_eval_fn({"eval_return": ..., "eval_success_rate": ..., ...})``.
``after_eval_hook`` computes the configured *metric* from that dict and
delegates to :meth:`TrialPrunerBase.report_metric`, which handles Optuna
reporting and pruning.

Supported *metric* values:

* ``"eval_return"`` (default) — raw DreamerV3 return, back-compat.
* ``"success_composite"`` — ``eval_success_rate + 0.001 * tanh(eval_return / 50)``.
  Success rate is the paper's headline metric; the bounded (<0.001) return
  term only breaks ties below SR's own resolution, it never dominates it.
* ``"context_pred_gap"`` — Social-Dreamer context-identifiability signal
  (Stage A objective). Any other key present in the eval payload dict is
  also accepted and read directly.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, Optional

from .pruner_base import TrialPrunerBase

if TYPE_CHECKING:
    from ..cfg import HealthPruneCfg

logger = logging.getLogger(__name__)


class DreamerV3TrialPruner(TrialPrunerBase):
    """Optuna trial pruner for the DreamerV3 training pipeline.

    Receives metric values via :meth:`after_eval_hook` (injected into the
    DreamerV3 training loop) and reports them to Optuna.

    Args:
        trial: The current Optuna trial object.
        metric: Which value from the eval payload dict to report to Optuna.
            See module docstring for supported values.
        health_prune: Optional context-collapse early-kill config. Only
            takes effect when ``social_context_enabled`` is also True.
        social_context_enabled: Whether this trial's config has
            ``social.context.enabled=True``. Health pruning is a no-op
            without a context bottleneck to collapse.
        verbose: Verbosity level.

    Attributes:
        best_metric: Best metric value seen during this trial.
    """

    def __init__(
        self,
        trial: "optuna.trial.Trial",
        metric: str = "eval_return",
        health_prune: Optional["HealthPruneCfg"] = None,
        social_context_enabled: bool = False,
        verbose: int = 0,
    ):
        super().__init__(trial=trial, metric=metric, verbose=verbose)
        self._last_value: Optional[float] = None
        self._health_prune = health_prune
        self._social_context_enabled = social_context_enabled
        self._collapsed_streak = 0

    # ── TrialPrunerBase interface ─────────────────────────────────────────

    def read_metric(self) -> Optional[float]:
        """Return the last value received via :meth:`after_eval_hook`.

        Returns ``None`` before the first evaluation.
        """
        return self._last_value

    # ── Metric selection ────────────────────────────────────────────────

    def _compute_metric(self, metrics: Dict[str, float]) -> float:
        if self._metric == "success_composite":
            eval_return = metrics.get("eval_return", 0.0)
            eval_success_rate = metrics.get("eval_success_rate", 0.0)
            return eval_success_rate + 0.001 * math.tanh(eval_return / 50.0)
        return metrics.get(self._metric, float("-inf"))

    def _check_health_prune(self, metrics: Dict[str, float]) -> None:
        """Kill trials whose social context has collapsed (no live `b`).

        No-op unless both ``health_prune.enabled`` and
        ``social_context_enabled`` are set — pruning on a signal that
        doesn't exist for this trial's config would be meaningless.
        """
        hp = self._health_prune
        if hp is None or not hp.enabled or not self._social_context_enabled:
            return
        if metrics.get("step", 0) < hp.after_steps:
            self._collapsed_streak = 0
            return

        if metrics.get("context_pred_gap", float("inf")) <= hp.min_gap:
            self._collapsed_streak += 1
        else:
            self._collapsed_streak = 0

        if self._collapsed_streak >= hp.patience:
            import optuna

            raise optuna.TrialPruned(
                f"Trial {self._trial.number} health-pruned: context_pred_gap "
                f"<= {hp.min_gap} for {self._collapsed_streak} consecutive "
                f"evals after step {hp.after_steps} (collapsed social context)."
            )

    # ── DreamerV3 hook ────────────────────────────────────────────────────

    def after_eval_hook(self, metrics: "Dict[str, float]") -> None:
        """Called by ``helper.train()`` after every evaluation phase.

        Computes the configured metric from *metrics*, checks the
        context-collapse health prune, and reports the metric to Optuna.
        If either pruner decides to stop this trial, ``optuna.TrialPruned``
        is raised (which should propagate out of the training loop).

        Args:
            metrics: Eval payload dict — at least ``eval_return`` and
                ``eval_success_rate``, plus DreamerV3's world-model
                diagnostics (``context_pred_gap``, ``step``, ...).

        Raises:
            optuna.TrialPruned: If the trial should be stopped early.
        """
        self._check_health_prune(metrics)

        value = self._compute_metric(metrics)
        self._last_value = value
        self.report_metric(value)
