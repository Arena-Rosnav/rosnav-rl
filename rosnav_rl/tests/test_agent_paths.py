"""Tests for resolving a saved agent's observations config."""

from __future__ import annotations

import pytest

from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.utils.agent_paths import resolve_observations_config_path


def _spec(observations_config: str | None) -> AgentConfig:
    return AgentConfig(
        name="observations_config_agent",
        observations_config=observations_config,
        framework=StableBaselinesCfg(algorithm=PPO_Cfg(architecture_name="test_arch", parameters=PPO_Algorithm_Cfg())),
    )


def test_unset_observations_config_resolves_to_packaged_default():
    path = resolve_observations_config_path(_spec(None))

    assert path.name == "observations.yaml"
    assert path.parent.name == "observations"
    assert path.is_file()


def test_observations_config_resolves_to_the_training_file(tmp_path):
    config = tmp_path / "custom_observations.yaml"
    config.write_text("datasources: {}\n")

    assert resolve_observations_config_path(_spec(str(config))) == config


def test_missing_observations_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_observations_config_path(_spec(str(tmp_path / "missing_observations.yaml")))
