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
calls ``after_eval_fn(eval_return)``.  ``after_eval_hook`` stores the value
and delegates to :meth:`TrialPrunerBase.report_metric` which handles Optuna
reporting and pruning.

The *metric* argument is kept for display/logging only — the actual value
is always the one passed to ``after_eval_hook``.
"""

from __future__ import annotations

import logging
from typing import Optional

from .pruner_base import TrialPrunerBase

logger = logging.getLogger(__name__)


class DreamerV3TrialPruner(TrialPrunerBase):
    """Optuna trial pruner for the DreamerV3 training pipeline.

    Receives metric values via :meth:`after_eval_hook` (injected into the
    DreamerV3 training loop) and reports them to Optuna.

    Args:
        trial: The current Optuna trial object.
        metric: Metric name used for logging. The actual numeric value
            is supplied by ``after_eval_hook``.
        verbose: Verbosity level.

    Attributes:
        best_metric: Best ``eval_return`` seen during this trial.
    """

    def __init__(
        self,
        trial: "optuna.trial.Trial",
        metric: str = "eval_return",
        verbose: int = 0,
    ):
        super().__init__(trial=trial, metric=metric, verbose=verbose)
        self._last_value: Optional[float] = None

    # ── TrialPrunerBase interface ─────────────────────────────────────────

    def read_metric(self) -> Optional[float]:
        """Return the last value received via :meth:`after_eval_hook`.

        Returns ``None`` before the first evaluation.
        """
        return self._last_value

    # ── DreamerV3 hook ────────────────────────────────────────────────────

    def after_eval_hook(self, metrics: "Dict[str, float]") -> None:
        """Called by ``helper.train()`` after every evaluation phase.

        Stores the ``eval_return`` from *metrics* and immediately reports it
        to Optuna.  If the pruner decides to stop this trial,
        ``optuna.TrialPruned`` is raised (which should propagate out of the
        training loop).

        Args:
            metrics: Dict with at least ``eval_return`` (and ``eval_success_rate``).

        Raises:
            optuna.TrialPruned: If the trial should be stopped early.
        """
        eval_return = metrics.get("eval_return", float("-inf"))
        self._last_value = eval_return
        self.report_metric(eval_return)
