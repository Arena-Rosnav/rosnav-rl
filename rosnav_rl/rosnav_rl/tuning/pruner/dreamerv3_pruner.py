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
calls ``after_eval_fn(metrics)``.  ``after_eval_hook`` reads the *metric*
key from that dict and delegates to :meth:`TrialPrunerBase.report_metric`
which handles Optuna reporting and pruning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .pruner_base import TrialPrunerBase

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


class DreamerV3TrialPruner(TrialPrunerBase):
    """Optuna trial pruner for the DreamerV3 training pipeline.

    Receives metric values via :meth:`after_eval_hook` (injected into the
    DreamerV3 training loop) and reports them to Optuna.

    Args:
        trial: The current Optuna trial object.
        metric: Key of the ``after_eval_hook`` metrics dict to report.
        verbose: Verbosity level.

    Attributes:
        best_metric: Best *metric* value seen during this trial.
    """

    def __init__(
        self,
        trial: optuna.trial.Trial,
        metric: str = "eval_return",
        verbose: int = 0,
    ):
        super().__init__(trial=trial, metric=metric, verbose=verbose)
        self._last_value: float | None = None

    # ── TrialPrunerBase interface ─────────────────────────────────────────

    def read_metric(self) -> float | None:
        """Return the last value received via :meth:`after_eval_hook`.

        Returns ``None`` before the first evaluation.
        """
        return self._last_value

    # ── DreamerV3 hook ────────────────────────────────────────────────────

    def after_eval_hook(self, metrics: dict[str, float]) -> None:
        """Called by ``helper.train()`` after every evaluation phase.

        Stores the configured metric from *metrics* and immediately reports it
        to Optuna.  If the pruner decides to stop this trial,
        ``optuna.TrialPruned`` is raised (which should propagate out of the
        training loop).

        Args:
            metrics: Dict containing the configured metric key.

        Raises:
            optuna.TrialPruned: If the trial should be stopped early.
            KeyError: If *metrics* lacks the configured metric.
        """
        value = metrics[self._metric]
        self._last_value = value
        self.report_metric(value)
