"""Sampling utilities for Optuna hyperparameter tuning.

Two complementary functions:

* ``suggest_params`` — use an Optuna *trial* to draw concrete values from a
  search space.
* ``apply_params`` — overlay the drawn values onto a nested config dictionary
  using dot-notation paths.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from .search_space import CategoricalParam, FloatParam, IntParam, SearchSpace


def suggest_params(trial: "optuna.trial.Trial", search_space: SearchSpace) -> Dict[str, Any]:
    """Sample hyperparameters from *search_space* using an Optuna trial.

    Args:
        trial: An active Optuna trial object.
        search_space: Mapping of dot-notation config paths to
            ``FloatParam`` / ``IntParam`` / ``CategoricalParam``.

    Returns:
        A flat dictionary mapping the **same** dot-notation keys to the
        sampled values.

    Example::

        params = suggest_params(trial, {
            "agent_cfg.framework.algorithm.parameters.learning_rate":
                FloatParam(low=1e-5, high=1e-3, log=True),
        })
        # params == {"agent_cfg.framework.algorithm.parameters.learning_rate": 0.000342}
    """
    params: Dict[str, Any] = {}
    for path, param in search_space.items():
        if isinstance(param, FloatParam):
            value = trial.suggest_float(
                path,
                param.low,
                param.high,
                log=param.log,
                step=param.step,
            )
        elif isinstance(param, IntParam):
            value = trial.suggest_int(
                path,
                param.low,
                param.high,
                log=param.log,
                step=param.step,
            )
        elif isinstance(param, CategoricalParam):
            value = trial.suggest_categorical(path, param.choices)
        else:
            raise TypeError(f"Unsupported search-space parameter type: {type(param)}")
        params[path] = value
    return params


def apply_params(config_dict: dict, params: Dict[str, Any]) -> dict:
    """Deep-copy *config_dict* and set values at dot-notation paths.

    Args:
        config_dict: Nested dictionary (e.g. from
            ``TrainingCfg.model_dump(mode="json")``).
        params: Flat dict with dot-notation keys from :func:`suggest_params`.

    Returns:
        A **new** dictionary with the overridden values.

    Raises:
        KeyError: If an intermediate key in the dot-notation path does not
            exist in *config_dict*.

    Example::

        base = {"agent_cfg": {"framework": {"algorithm": {"parameters": {"learning_rate": 3e-4}}}}}
        updated = apply_params(base, {
            "agent_cfg.framework.algorithm.parameters.learning_rate": 1e-4,
        })
        # updated["agent_cfg"]["framework"]["algorithm"]["parameters"]["learning_rate"] == 1e-4
    """
    result = copy.deepcopy(config_dict)
    for path, value in params.items():
        keys = path.split(".")
        obj = result
        for key in keys[:-1]:
            if key not in obj:
                raise KeyError(
                    f"Key '{key}' not found while traversing path '{path}'. "
                    f"Available keys: {list(obj.keys())}"
                )
            obj = obj[key]

        final_key = keys[-1]
        if final_key not in obj:
            raise KeyError(
                f"Final key '{final_key}' not found in path '{path}'. "
                f"Available keys: {list(obj.keys())}"
            )
        obj[final_key] = value
    return result
