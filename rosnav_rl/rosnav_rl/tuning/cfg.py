"""Top-level tuning configuration model.

``TuningCfg`` wraps everything needed for a hyperparameter search:

* **base_config** — path to the base ``TrainingCfg`` YAML.
* **search_space** — which parameters to tune and their ranges.
* **study settings** — Optuna study name, direction, storage, pruner.
* **trial settings** — number of trials, optional reduced timesteps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

from .search_space import SearchParam, SearchSpace


class PrunerCfg(BaseModel):
    """Configuration for the Optuna pruner.

    The pruner decides whether a trial should be stopped early based on
    intermediate metric reports from :class:`~rosnav_rl.tuning.sb3_pruner.SB3TrialPruner`.
    """

    type: Literal["median", "hyperband", "percentile", "none"] = Field(
        "median",
        description="Pruner algorithm. 'none' disables pruning.",
    )
    n_startup_trials: int = Field(
        5,
        description="Number of trials to complete before pruning kicks in.",
    )
    n_warmup_steps: int = Field(
        10,
        description=(
            "Number of intermediate reports to collect per trial before "
            "the pruner may prune."
        ),
    )
    percentile: float = Field(
        50.0,
        description="For 'percentile' pruner — keep trials above this percentile.",
    )


class TuningCfg(BaseModel):
    """Full configuration for an Optuna hyperparameter tuning run.

    Example YAML::

        base_config: configs/training/sb_training_config.yaml

        study_name: ppo_tuning
        n_trials: 50
        direction: maximize
        metric: mean_reward
        trial_timesteps: 500000

        storage: "sqlite:///tuning_results.db"

        pruner:
          type: median
          n_startup_trials: 5
          n_warmup_steps: 10

        agents_dir: /tmp/tuning_agents

        search_space:
          agent_cfg.framework.algorithm.parameters.learning_rate:
            type: float
            low: 1.0e-5
            high: 1.0e-3
            log: true
          agent_cfg.framework.algorithm.parameters.gamma:
            type: float
            low: 0.9
            high: 0.9999
          agent_cfg.framework.algorithm.parameters.n_steps:
            type: int
            low: 128
            high: 4096
            step: 128
    """

    # ── Base config ───────────────────────────────────────────────────────
    base_config: Path = Field(
        ...,
        description="Path to the base TrainingCfg YAML to use as template.",
    )

    # ── Optuna study settings ─────────────────────────────────────────────
    study_name: str = Field(
        "rosnav_tuning",
        description="Optuna study name (used for storage and dashboard).",
    )
    n_trials: int = Field(
        50,
        ge=1,
        description="Total number of Optuna trials to run.",
    )
    direction: Literal["maximize", "minimize"] = Field(
        "maximize",
        description="Optimization direction for the target metric.",
    )
    metric: str = Field(
        "mean_reward",
        description=(
            "Name of the metric to optimize. Must match the key reported "
            "by the training callback (e.g. 'mean_reward', 'success_rate')."
        ),
    )

    # ── Trial settings ────────────────────────────────────────────────────
    trial_timesteps: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Override total_timesteps per trial for faster exploration. "
            "If None, uses the value from the base config."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────
    agents_dir: Optional[Path] = Field(
        None,
        description=(
            "Custom base directory for trial agent artifacts. "
            "Each trial creates a sub-directory named "
            "'<study_name>_trial_<number>'. "
            "If None, uses the default agents directory."
        ),
    )

    # ── Persistence ───────────────────────────────────────────────────────
    storage: Optional[str] = Field(
        None,
        description=(
            "Optuna storage URL for persistent studies. "
            "Examples: 'sqlite:///tuning.db', 'postgresql://...'. "
            "If None, uses in-memory storage (results lost on exit)."
        ),
    )

    # ── Pruner ────────────────────────────────────────────────────────────
    pruner: PrunerCfg = Field(
        default_factory=PrunerCfg,
        description="Early stopping / pruning configuration.",
    )

    # ── Search space ──────────────────────────────────────────────────────
    search_space: Dict[str, SearchParam] = Field(
        ...,
        description=(
            "Dot-notation config paths mapped to search-space parameters. "
            "Any field in the base TrainingCfg can be tuned."
        ),
    )
