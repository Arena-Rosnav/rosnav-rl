"""
Hyperparameter Tuning Infrastructure
=====================================

Optuna-based hyperparameter tuning that integrates with the rosnav-rl Pydantic
configuration system.  Search spaces use **dot-notation config paths** so that
*any* field in a ``TrainingCfg`` (or derived config) can be tuned without
special handling.

Quick start::

    from rosnav_rl.tuning import TuningCfg, suggest_params, apply_params

Modules
-------
search_space
    Pydantic models for individual search-space parameters
    (``FloatParam``, ``IntParam``, ``CategoricalParam``).
cfg
    ``TuningCfg`` — top-level configuration model that wraps a base
    training config, search space, Optuna study settings, and pruner.
sampler
    ``suggest_params`` — convert an Optuna trial + search space into
    a concrete parameter dict.
    ``apply_params`` — overlay sampled values onto a nested config dict
    using dot-notation paths.
pruner_base
    ``TrialPrunerBase`` — abstract base class for framework-agnostic
    Optuna trial pruning.  Mirrors the ``CurriculumBase`` pattern.
sb3_pruner
    ``SB3TrialPruner`` — SB3 callback that reports intermediate training
    metrics to Optuna and supports early pruning.
dreamerv3_pruner
    ``DreamerV3TrialPruner`` — DreamerV3 adapter that hooks into the
    training loop via ``after_eval_fn``.
"""

from .cfg import TuningCfg
from .pruner.dreamerv3_pruner import DreamerV3TrialPruner
from .pruner.pruner_base import TrialPrunerBase
from .sampler import apply_params, suggest_params
from .pruner.sb3_pruner import SB3TrialPruner
from .search_space import CategoricalParam, FloatParam, IntParam, SearchParam

__all__ = [
    "TuningCfg",
    "suggest_params",
    "apply_params",
    "FloatParam",
    "IntParam",
    "CategoricalParam",
    "SearchParam",
    "TrialPrunerBase",
    "SB3TrialPruner",
    "DreamerV3TrialPruner",
]

