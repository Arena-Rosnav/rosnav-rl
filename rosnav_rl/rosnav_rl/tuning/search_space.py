"""Search-space parameter models for Optuna hyperparameter tuning.

Each model describes a single dimension and maps directly to an Optuna
``trial.suggest_*`` call.  The discriminated union ``SearchParam`` lets
YAML / JSON configs use a ``type`` field to select the variant::

    search_space:
      learning_rate:
        type: float
        low: 1.0e-5
        high: 1.0e-3
        log: true
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class FloatParam(BaseModel):
    """Continuous float search dimension.

    Maps to ``trial.suggest_float(name, low, high, log=log, step=step)``.
    """

    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False
    step: float | None = None


class IntParam(BaseModel):
    """Integer search dimension.

    Maps to ``trial.suggest_int(name, low, high, log=log, step=step)``.
    """

    type: Literal["int"] = "int"
    low: int
    high: int
    log: bool = False
    step: int = 1


class CategoricalParam(BaseModel):
    """Categorical search dimension.

    Maps to ``trial.suggest_categorical(name, choices)``.
    """

    type: Literal["categorical"] = "categorical"
    choices: list[Any]


SearchParam = Annotated[
    FloatParam | IntParam | CategoricalParam,
    Field(discriminator="type"),
]
"""Discriminated union of all search-space parameter types."""

SearchSpace = dict[str, SearchParam]
"""Mapping from dot-notation config paths to search-space parameters.

Example::

    {
        "agent_cfg.framework.algorithm.parameters.learning_rate": FloatParam(low=1e-5, high=1e-3, log=True),
        "agent_cfg.framework.algorithm.parameters.n_steps": IntParam(low=128, high=4096, step=128),
        "agent_cfg.framework.algorithm.parameters.gamma": FloatParam(low=0.9, high=0.9999),
    }
"""
