"""Stable Baselines 3 Optuna trial pruner.

``SB3TrialPruner`` combines the framework-agnostic :class:`TrialPrunerBase`
with SB3's ``BaseCallback`` so that Optuna can prune unpromising trials
during SB3 training — mirroring the ``StagedTrainCallback`` pattern.

Usage::

    from rosnav_rl.tuning.sb3_pruner import SB3TrialPruner

    pruner = SB3TrialPruner(trial, metric="mean_reward")
    model.learn(total_timesteps=500_000, callback=[eval_cb, pruner])
    best_reward = pruner.best_metric
"""

from __future__ import annotations

import logging
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from .pruner_base import TrialPrunerBase

logger = logging.getLogger(__name__)

# SB3 logger keys for common metric aliases
_SB3_METRIC_MAP = {
    "mean_reward": "ep_rew_mean",
    "mean_ep_length": "ep_len_mean",
}


class SB3TrialPruner(TrialPrunerBase, BaseCallback):
    """Optuna trial pruner for Stable-Baselines 3.

    Inherits from both :class:`TrialPrunerBase` (Optuna reporting logic) and
    SB3's :class:`BaseCallback` (training hook interface), mirroring the
    ``CurriculumBase`` / ``StagedTrainCallback`` dual-inheritance pattern.

    After every rollout (or every *report_freq* steps) the callback reads
    the requested metric from SB3's logger and delegates to
    :meth:`~TrialPrunerBase.report_metric`.

    Args:
        trial: The current Optuna trial object.
        metric: Metric name to report.  Common aliases:

            - ``"mean_reward"`` → SB3 key ``rollout/ep_rew_mean``
            - ``"mean_ep_length"`` → SB3 key ``rollout/ep_len_mean``
            - Any other string is looked up verbatim with the prefixes
              ``rollout/``, ``train/``, ``eval/``, and bare.
        report_freq: Report to Optuna every *report_freq* calls to
            ``_on_step``.  ``0`` (default) reports once per rollout end.
        verbose: Verbosity level.

    Attributes:
        best_metric (float | None): Best metric value seen in this trial.
            After training, pass this as the trial's objective value.
    """

    def __init__(
        self,
        trial: "optuna.trial.Trial",
        metric: str = "mean_reward",
        report_freq: int = 0,
        verbose: int = 0,
    ):
        TrialPrunerBase.__init__(self, trial=trial, metric=metric, verbose=verbose)
        BaseCallback.__init__(self, verbose=verbose)
        self._report_freq = report_freq

    # ── TrialPrunerBase interface ─────────────────────────────────────────

    def read_metric(self) -> Optional[float]:
        """Extract the metric from the SB3 logger.

        Searches ``model.logger.name_to_value`` with common prefixes
        (``rollout/``, ``train/``, ``eval/``, bare).

        Returns:
            Metric value, or ``None`` if unavailable.
        """
        try:
            model = self.model  # type: ignore[union-attr]
        except AttributeError:
            return None

        if model is None:
            return None

        logger_key = _SB3_METRIC_MAP.get(self._metric, self._metric)

        try:
            name_to_value = model.logger.name_to_value  # type: ignore[union-attr]
            for prefix in ("rollout/", "train/", "eval/", ""):
                full_key = f"{prefix}{logger_key}"
                if full_key in name_to_value:
                    return float(name_to_value[full_key])
        except AttributeError:
            pass

        return None

    # ── SB3 callback hooks ────────────────────────────────────────────────

    def _on_step(self) -> bool:
        if self._report_freq > 0 and self.n_calls % self._report_freq == 0:
            return self.check_and_report()
        return True

    def _on_rollout_end(self) -> None:
        if self._report_freq == 0:
            self.check_and_report()
