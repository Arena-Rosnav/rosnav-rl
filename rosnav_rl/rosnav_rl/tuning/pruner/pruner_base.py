"""Framework-agnostic Optuna trial pruner base class.

``TrialPrunerBase`` mirrors the ``CurriculumBase`` pattern: it extracts the
Optuna reporting / pruning logic into a reusable abstract base class.
Framework-specific subclasses only need to implement :meth:`read_metric` to
bridge their metric source.

Concrete adapters
-----------------
* ``SB3TrialPruner``        — reads from Stable-Baselines 3 logger
* ``DreamerV3TrialPruner``  — receives metrics via ``after_eval_hook``

Usage::

    class MyFrameworkPruner(TrialPrunerBase):
        def read_metric(self) -> float | None:
            return self._my_framework.get_latest_reward()

    pruner = MyFrameworkPruner(trial, metric="mean_reward")
    # … during training …
    pruner.check_and_report()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class TrialPrunerBase(ABC):
    """Framework-agnostic Optuna trial pruner.

    Handles metric tracking, reporting to Optuna, and pruning decisions.
    Subclasses must implement :meth:`read_metric` to extract the current
    metric value from their specific RL framework.

    Args:
        trial: The current Optuna trial object.
        metric: Metric name for logging/display purposes.
        verbose: Verbosity level (0=quiet, >=1=info logs).

    Attributes:
        best_metric: Best metric value observed during the trial.
            After training, use this as the trial's return value.
    """

    def __init__(
        self,
        trial: "optuna.trial.Trial",
        metric: str = "mean_reward",
        verbose: int = 0,
    ):
        self._trial = trial
        self._metric = metric
        self._report_step = 0
        self.verbose = verbose

        self.best_metric: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────

    def report_metric(self, value: float) -> bool:
        """Report *value* to Optuna and check whether the trial should be pruned.

        Updates :attr:`best_metric`, calls ``trial.report()``, and raises
        ``optuna.TrialPruned`` if the pruner determines this trial is
        unpromising.

        Args:
            value: The metric value to report.

        Returns:
            ``True`` to continue training.

        Raises:
            optuna.TrialPruned: If the trial is pruned.
        """
        import optuna  # lazy — only needed when actually tuning

        # Track best
        if self.best_metric is None or value > self.best_metric:
            self.best_metric = value

        # Report to Optuna
        self._trial.report(value, step=self._report_step)
        self._report_step += 1

        if self.verbose >= 1:
            logger.info(
                "Trial %d step %d: %s = %.4f (best = %.4f)",
                self._trial.number,
                self._report_step,
                self._metric,
                value,
                self.best_metric,
            )

        # Check for pruning
        if self._trial.should_prune():
            if self.verbose >= 1:
                logger.info(
                    "Trial %d pruned at step %d.",
                    self._trial.number,
                    self._report_step,
                )
            raise optuna.TrialPruned(
                f"Trial {self._trial.number} pruned at step {self._report_step} "
                f"({self._metric} = {value:.4f})"
            )

        return True

    def check_and_report(self) -> bool:
        """Convenience: read the current metric and report it.

        If :meth:`read_metric` returns ``None`` (no data yet), this method
        silently returns ``True`` (keep training).

        Returns:
            ``True`` to continue training.

        Raises:
            optuna.TrialPruned: If the trial is pruned.
        """
        value = self.read_metric()
        if value is None:
            return True
        return self.report_metric(value)

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def read_metric(self) -> Optional[float]:
        """Return the current metric value from the RL framework.

        Implementations should return ``None`` when no data is available yet
        (e.g. before the first evaluation).

        Returns:
            The metric value, or ``None``.
        """
        raise NotImplementedError
