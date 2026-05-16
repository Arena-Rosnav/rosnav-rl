"""Tests for the rosnav_rl.tuning module.

Tests cover:
- Search-space model construction (FloatParam, IntParam, CategoricalParam)
- TuningCfg validation
- suggest_params with a mock Optuna trial
- apply_params dot-notation overriding
- TrialPrunerBase abstract interface
- SB3TrialPruner instantiation and metric reading
- DreamerV3TrialPruner hook and metric reporting
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from rosnav_rl.tuning.cfg import TuningCfg
from rosnav_rl.tuning.pruner.dreamerv3_pruner import DreamerV3TrialPruner
from rosnav_rl.tuning.pruner.pruner_base import TrialPrunerBase
from rosnav_rl.tuning.sampler import apply_params, suggest_params
from rosnav_rl.tuning.pruner.sb3_pruner import SB3TrialPruner
from rosnav_rl.tuning.search_space import (
    CategoricalParam,
    FloatParam,
    IntParam,
    SearchParam,
    SearchSpace,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_search_space() -> SearchSpace:
    return {
        "agent_cfg.framework.algorithm.parameters.learning_rate": FloatParam(
            low=1e-5, high=1e-3, log=True
        ),
        "agent_cfg.framework.algorithm.parameters.n_steps": IntParam(
            low=128, high=4096, step=128
        ),
        "agent_cfg.framework.algorithm.parameters.gamma": FloatParam(
            low=0.9, high=0.9999
        ),
        "agent_cfg.framework.algorithm.parameters.batch_size": CategoricalParam(
            choices=[64, 128, 256, 512]
        ),
    }


@pytest.fixture
def nested_config() -> dict:
    """A minimal nested config dict mimicking TrainingCfg.model_dump()."""
    return {
        "agent_cfg": {
            "name": "test_agent",
            "framework": {
                "algorithm": {
                    "parameters": {
                        "learning_rate": 3e-4,
                        "n_steps": 2048,
                        "gamma": 0.99,
                        "batch_size": 64,
                        "n_epochs": 10,
                    }
                }
            },
            "reward": {
                "reward_function_dict": {
                    "goal_reached": {"reward": 15.0},
                    "collision": {"reward": -10.0},
                }
            },
        },
        "arena_cfg": {},
        "resume": False,
    }


# ── Search Space Models ──────────────────────────────────────────────────────


class TestSearchSpaceModels:
    def test_float_param_defaults(self):
        p = FloatParam(low=0.0, high=1.0)
        assert p.type == "float"
        assert p.log is False
        assert p.step is None

    def test_float_param_with_log(self):
        p = FloatParam(low=1e-5, high=1e-3, log=True)
        assert p.log is True

    def test_float_param_with_step(self):
        p = FloatParam(low=0.0, high=1.0, step=0.1)
        assert p.step == 0.1

    def test_int_param_defaults(self):
        p = IntParam(low=1, high=100)
        assert p.type == "int"
        assert p.step == 1
        assert p.log is False

    def test_int_param_with_step(self):
        p = IntParam(low=128, high=4096, step=128)
        assert p.step == 128

    def test_categorical_param(self):
        p = CategoricalParam(choices=["a", "b", "c"])
        assert p.type == "categorical"
        assert p.choices == ["a", "b", "c"]

    def test_categorical_param_numeric(self):
        p = CategoricalParam(choices=[64, 128, 256])
        assert p.choices == [64, 128, 256]

    def test_search_param_from_dict_float(self):
        """Discriminated union dispatch via type field."""
        from pydantic import TypeAdapter
        from rosnav_rl.tuning.search_space import SearchParam

        adapter = TypeAdapter(SearchParam)
        result = adapter.validate_python({"type": "float", "low": 0.0, "high": 1.0})
        assert isinstance(result, FloatParam)

    def test_search_param_from_dict_int(self):
        from pydantic import TypeAdapter
        from rosnav_rl.tuning.search_space import SearchParam

        adapter = TypeAdapter(SearchParam)
        result = adapter.validate_python({"type": "int", "low": 1, "high": 100})
        assert isinstance(result, IntParam)

    def test_search_param_from_dict_categorical(self):
        from pydantic import TypeAdapter
        from rosnav_rl.tuning.search_space import SearchParam

        adapter = TypeAdapter(SearchParam)
        result = adapter.validate_python(
            {"type": "categorical", "choices": [1, 2, 3]}
        )
        assert isinstance(result, CategoricalParam)


# ── TuningCfg ─────────────────────────────────────────────────────────────────


class TestTuningCfg:
    def test_minimal_valid(self, sample_search_space):
        cfg = TuningCfg(
            base_config=Path("training_config.yaml"),
            search_space=sample_search_space,
        )
        assert cfg.study_name == "rosnav_tuning"
        assert cfg.n_trials == 50
        assert cfg.direction == "maximize"
        assert cfg.metric == "mean_reward"
        assert cfg.trial_timesteps is None
        assert cfg.storage is None
        assert cfg.pruner.type == "median"

    def test_full_config(self, sample_search_space):
        cfg = TuningCfg(
            base_config=Path("/some/path.yaml"),
            study_name="my_study",
            n_trials=100,
            direction="minimize",
            metric="success_rate",
            trial_timesteps=500_000,
            storage="sqlite:///test.db",
            agents_dir=Path("/tmp/tuning"),
            search_space=sample_search_space,
        )
        assert cfg.study_name == "my_study"
        assert cfg.n_trials == 100
        assert cfg.direction == "minimize"
        assert cfg.trial_timesteps == 500_000

    def test_from_dict(self, sample_search_space):
        """Validate from a plain dict (as from YAML loading)."""
        data = {
            "base_config": "config.yaml",
            "study_name": "test",
            "n_trials": 10,
            "search_space": {
                "agent_cfg.framework.algorithm.parameters.learning_rate": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 1e-3,
                    "log": True,
                }
            },
        }
        cfg = TuningCfg.model_validate(data)
        assert cfg.n_trials == 10
        lr_param = cfg.search_space[
            "agent_cfg.framework.algorithm.parameters.learning_rate"
        ]
        assert isinstance(lr_param, FloatParam)
        assert lr_param.log is True

    def test_pruner_config(self):
        from rosnav_rl.tuning.cfg import PrunerCfg

        pruner = PrunerCfg(type="hyperband", n_startup_trials=3)
        assert pruner.type == "hyperband"
        assert pruner.n_startup_trials == 3

    def test_serialization_roundtrip(self, sample_search_space):
        cfg = TuningCfg(
            base_config=Path("config.yaml"),
            search_space=sample_search_space,
        )
        dumped = cfg.model_dump(mode="json")
        restored = TuningCfg.model_validate(dumped)
        assert restored.study_name == cfg.study_name
        assert len(restored.search_space) == len(cfg.search_space)


# ── suggest_params ────────────────────────────────────────────────────────────


class TestSuggestParams:
    def _make_mock_trial(self, values: Dict[str, Any]) -> MagicMock:
        """Create a mock Optuna trial that returns specified values."""
        trial = MagicMock()

        def suggest_float(name, low, high, **kwargs):
            return values[name]

        def suggest_int(name, low, high, **kwargs):
            return values[name]

        def suggest_categorical(name, choices):
            return values[name]

        trial.suggest_float = suggest_float
        trial.suggest_int = suggest_int
        trial.suggest_categorical = suggest_categorical
        return trial

    def test_basic_suggest(self, sample_search_space):
        expected = {
            "agent_cfg.framework.algorithm.parameters.learning_rate": 1e-4,
            "agent_cfg.framework.algorithm.parameters.n_steps": 512,
            "agent_cfg.framework.algorithm.parameters.gamma": 0.995,
            "agent_cfg.framework.algorithm.parameters.batch_size": 128,
        }
        trial = self._make_mock_trial(expected)
        result = suggest_params(trial, sample_search_space)
        assert result == expected

    def test_empty_search_space(self):
        trial = MagicMock()
        result = suggest_params(trial, {})
        assert result == {}

    def test_all_param_types_called(self, sample_search_space):
        """Verify correct Optuna methods are called for each param type."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        trial.suggest_int.return_value = 256
        trial.suggest_categorical.return_value = 128

        suggest_params(trial, sample_search_space)

        # FloatParam fields should call suggest_float
        assert trial.suggest_float.call_count == 2  # learning_rate + gamma
        # IntParam should call suggest_int
        assert trial.suggest_int.call_count == 1  # n_steps
        # CategoricalParam should call suggest_categorical
        assert trial.suggest_categorical.call_count == 1  # batch_size


# ── apply_params ──────────────────────────────────────────────────────────────


class TestApplyParams:
    def test_basic_override(self, nested_config):
        params = {
            "agent_cfg.framework.algorithm.parameters.learning_rate": 1e-4,
        }
        result = apply_params(nested_config, params)
        assert (
            result["agent_cfg"]["framework"]["algorithm"]["parameters"][
                "learning_rate"
            ]
            == 1e-4
        )
        # Original should be unchanged
        assert (
            nested_config["agent_cfg"]["framework"]["algorithm"]["parameters"][
                "learning_rate"
            ]
            == 3e-4
        )

    def test_multiple_overrides(self, nested_config):
        params = {
            "agent_cfg.framework.algorithm.parameters.learning_rate": 1e-4,
            "agent_cfg.framework.algorithm.parameters.n_steps": 512,
            "agent_cfg.framework.algorithm.parameters.gamma": 0.999,
        }
        result = apply_params(nested_config, params)
        assert (
            result["agent_cfg"]["framework"]["algorithm"]["parameters"][
                "learning_rate"
            ]
            == 1e-4
        )
        assert (
            result["agent_cfg"]["framework"]["algorithm"]["parameters"]["n_steps"]
            == 512
        )
        assert (
            result["agent_cfg"]["framework"]["algorithm"]["parameters"]["gamma"]
            == 0.999
        )

    def test_nested_reward_override(self, nested_config):
        params = {
            "agent_cfg.reward.reward_function_dict.goal_reached.reward": 20.0,
        }
        result = apply_params(nested_config, params)
        assert (
            result["agent_cfg"]["reward"]["reward_function_dict"]["goal_reached"][
                "reward"
            ]
            == 20.0
        )

    def test_deep_copy(self, nested_config):
        """Ensure apply_params does not mutate the original config."""
        original = copy.deepcopy(nested_config)
        apply_params(nested_config, {"agent_cfg.name": "modified"})
        assert nested_config == original

    def test_top_level_override(self, nested_config):
        result = apply_params(nested_config, {"resume": True})
        assert result["resume"] is True

    def test_missing_intermediate_key_raises(self, nested_config):
        with pytest.raises(KeyError, match="nonexistent"):
            apply_params(
                nested_config, {"agent_cfg.nonexistent.key": "value"}
            )

    def test_missing_final_key_raises(self, nested_config):
        with pytest.raises(KeyError, match="nonexistent_param"):
            apply_params(
                nested_config,
                {
                    "agent_cfg.framework.algorithm.parameters.nonexistent_param": 42
                },
            )

    def test_empty_params(self, nested_config):
        result = apply_params(nested_config, {})
        assert result == nested_config

    def test_override_preserves_siblings(self, nested_config):
        """Changing one param shouldn't affect sibling values."""
        params = {
            "agent_cfg.framework.algorithm.parameters.learning_rate": 1e-4,
        }
        result = apply_params(nested_config, params)
        # n_epochs should still be 10
        assert (
            result["agent_cfg"]["framework"]["algorithm"]["parameters"]["n_epochs"]
            == 10
        )


# ── TrialPrunerBase ───────────────────────────────────────────────────────────


class _DummyPruner(TrialPrunerBase):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, trial, metric="test_metric", verbose=0):
        super().__init__(trial=trial, metric=metric, verbose=verbose)
        self._value = None

    def set_value(self, v):
        self._value = v

    def read_metric(self):
        return self._value


class TestTrialPrunerBase:
    def _make_trial(self, should_prune=False):
        trial = MagicMock()
        trial.number = 0
        trial.should_prune.return_value = should_prune
        return trial

    def test_cannot_instantiate_abc(self):
        """TrialPrunerBase is abstract — direct instantiation must fail."""
        trial = MagicMock()
        with pytest.raises(TypeError):
            TrialPrunerBase(trial)

    def test_best_metric_starts_none(self):
        trial = self._make_trial()
        pruner = _DummyPruner(trial)
        assert pruner.best_metric is None

    def test_report_metric_updates_best(self):
        trial = self._make_trial()
        pruner = _DummyPruner(trial)

        pruner.report_metric(5.0)
        assert pruner.best_metric == 5.0

        pruner.report_metric(3.0)
        assert pruner.best_metric == 5.0  # doesn't decrease

        pruner.report_metric(7.0)
        assert pruner.best_metric == 7.0

    def test_report_metric_calls_trial_report(self):
        trial = self._make_trial()
        pruner = _DummyPruner(trial)

        pruner.report_metric(42.0)
        trial.report.assert_called_once_with(42.0, step=0)

        pruner.report_metric(43.0)
        assert trial.report.call_count == 2
        trial.report.assert_called_with(43.0, step=1)

    def test_report_metric_increments_step(self):
        trial = self._make_trial()
        pruner = _DummyPruner(trial)

        for i in range(5):
            pruner.report_metric(float(i))

        assert pruner._report_step == 5

    def test_report_metric_raises_on_prune(self):
        import optuna

        trial = self._make_trial(should_prune=True)
        pruner = _DummyPruner(trial)

        with pytest.raises(optuna.TrialPruned):
            pruner.report_metric(1.0)

    def test_check_and_report_skips_none(self):
        trial = self._make_trial()
        pruner = _DummyPruner(trial)
        # value is None by default
        result = pruner.check_and_report()
        assert result is True
        trial.report.assert_not_called()

    def test_check_and_report_reports_value(self):
        trial = self._make_trial()
        pruner = _DummyPruner(trial)
        pruner.set_value(10.0)

        result = pruner.check_and_report()
        assert result is True
        trial.report.assert_called_once_with(10.0, step=0)
        assert pruner.best_metric == 10.0


# ── SB3TrialPruner ───────────────────────────────────────────────────────────


class TestSB3TrialPruner:
    def test_instantiation(self):
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial, metric="mean_reward", verbose=1)
        assert pruner.best_metric is None
        assert pruner._metric == "mean_reward"

    def test_is_trial_pruner_base(self):
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial)
        assert isinstance(pruner, TrialPrunerBase)

    def test_read_metric_returns_none_without_model(self):
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial)
        assert pruner.read_metric() is None

    def test_read_metric_from_sb3_logger(self):
        """Simulate SB3 logger with name_to_value dict."""
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial, metric="mean_reward")

        # Mock SB3 model + logger
        mock_model = MagicMock()
        mock_model.logger.name_to_value = {"rollout/ep_rew_mean": 42.5}
        pruner.model = mock_model

        value = pruner.read_metric()
        assert value == 42.5

    def test_read_metric_with_eval_prefix(self):
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial, metric="mean_reward")

        mock_model = MagicMock()
        mock_model.logger.name_to_value = {"eval/ep_rew_mean": 99.0}
        pruner.model = mock_model

        assert pruner.read_metric() == 99.0

    def test_read_metric_custom_key(self):
        """A custom metric key should be looked up directly."""
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial, metric="my_custom_metric")

        mock_model = MagicMock()
        mock_model.logger.name_to_value = {"train/my_custom_metric": 7.7}
        pruner.model = mock_model

        assert pruner.read_metric() == 7.7

    def test_read_metric_bare_key(self):
        """Metric without any prefix should also be found."""
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial, metric="success_rate")

        mock_model = MagicMock()
        mock_model.logger.name_to_value = {"success_rate": 0.85}
        pruner.model = mock_model

        assert pruner.read_metric() == 0.85

    def test_read_metric_returns_none_when_key_missing(self):
        trial = MagicMock()
        trial.number = 0
        pruner = SB3TrialPruner(trial, metric="nonexistent")

        mock_model = MagicMock()
        mock_model.logger.name_to_value = {"rollout/ep_rew_mean": 1.0}
        pruner.model = mock_model

        assert pruner.read_metric() is None

    def test_on_rollout_end_reports(self):
        """_on_rollout_end triggers check_and_report when report_freq=0."""
        import optuna

        trial = MagicMock()
        trial.number = 0
        trial.should_prune.return_value = False
        pruner = SB3TrialPruner(trial, metric="mean_reward", report_freq=0)

        mock_model = MagicMock()
        mock_model.logger.name_to_value = {"rollout/ep_rew_mean": 55.0}
        pruner.model = mock_model

        pruner._on_rollout_end()

        trial.report.assert_called_once_with(55.0, step=0)
        assert pruner.best_metric == 55.0


# ── DreamerV3TrialPruner ──────────────────────────────────────────────────────


class TestDreamerV3TrialPruner:
    def _make_trial(self, should_prune=False):
        trial = MagicMock()
        trial.number = 0
        trial.should_prune.return_value = should_prune
        return trial

    def test_instantiation(self):
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial, metric="eval_return")
        assert pruner.best_metric is None
        assert pruner._metric == "eval_return"

    def test_is_trial_pruner_base(self):
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)
        assert isinstance(pruner, TrialPrunerBase)

    def test_read_metric_none_before_hook(self):
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)
        assert pruner.read_metric() is None

    def test_after_eval_hook_stores_value(self):
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)

        pruner.after_eval_hook(123.4)
        assert pruner.read_metric() == 123.4

    def test_after_eval_hook_reports_to_optuna(self):
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)

        pruner.after_eval_hook(10.0)
        trial.report.assert_called_once_with(10.0, step=0)

        pruner.after_eval_hook(20.0)
        trial.report.assert_called_with(20.0, step=1)

    def test_after_eval_hook_tracks_best(self):
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)

        pruner.after_eval_hook(5.0)
        assert pruner.best_metric == 5.0

        pruner.after_eval_hook(3.0)
        assert pruner.best_metric == 5.0

        pruner.after_eval_hook(8.0)
        assert pruner.best_metric == 8.0

    def test_after_eval_hook_prunes(self):
        import optuna

        trial = self._make_trial(should_prune=True)
        pruner = DreamerV3TrialPruner(trial)

        with pytest.raises(optuna.TrialPruned):
            pruner.after_eval_hook(1.0)

    def test_multiple_evals_increment_step(self):
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)

        for i in range(5):
            pruner.after_eval_hook(float(i))

        assert pruner._report_step == 5
        # Last call should have step=4
        trial.report.assert_called_with(4.0, step=4)

    def test_negative_returns(self):
        """Pruner handles negative eval_return correctly."""
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)

        pruner.after_eval_hook(-10.0)
        assert pruner.best_metric == -10.0

        pruner.after_eval_hook(-5.0)
        assert pruner.best_metric == -5.0  # -5 > -10

    def test_check_and_report_after_hook(self):
        """check_and_report returns the stored hook value."""
        trial = self._make_trial()
        pruner = DreamerV3TrialPruner(trial)

        pruner.after_eval_hook(42.0)
        # Manually call check_and_report — should use the stored value
        result = pruner.check_and_report()
        assert result is True
        # report called twice: once from after_eval_hook, once from check_and_report
        assert trial.report.call_count == 2
