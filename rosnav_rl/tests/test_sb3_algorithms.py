"""Test that RL agents can be created and initialised for every supported
Stable Baselines 3 algorithm configuration.

Each parametrized test:
  1. Validates the algorithm-specific Pydantic config (defaults + overrides).
  2. Wraps it in an AgentConfig (with the matching architecture name).
  3. Instantiates an RL_Agent and calls initialize_model() with a DummyVecEnv.

Registered test architectures (TEST_AGENT_*) are minimal stubs that re-use
the EXTRACTOR_5 feature extractor and the same observation spaces as AGENT_1
but set the correct algorithm_class for each SB3 algorithm.

Dict-parsing tests verify that configs can be round-tripped through plain
Python dicts (as produced by YAML loaders) and that StableBaselinesCfg can
identify the correct algorithm class automatically from the ``type`` field.
"""

from __future__ import annotations

from typing import Type
from unittest.mock import patch

import numpy as np
import pytest
import torch.nn as nn
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib import CrossQ, RecurrentPPO, TQC, TRPO

import rosnav_rl.spaces.observation_space.spaces as spaces
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import (
    A2C_Algorithm_Cfg,
    A2C_Cfg,
    CrossQ_Algorithm_Cfg,
    CrossQ_Cfg,
    PPO_Algorithm_Cfg,
    PPO_Cfg,
    SAC_Algorithm_Cfg,
    SAC_Cfg,
    TD3_Algorithm_Cfg,
    TD3_Cfg,
    TQC_Algorithm_Cfg,
    TQC_Cfg,
    TRPO_Algorithm_Cfg,
    TRPO_Cfg,
)
from rosnav_rl.model.stable_baselines3.cfg.framework import StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.policy.agent_factory import AgentFactory
from rosnav_rl.model.stable_baselines3.policy.base_policy import (
    StableBaselinesPolicyDescription,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.classic import (
    EXTRACTOR_5,
)
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env

# ── Minimal shared observation-space kwargs (same as AGENT_1 / AGENT_2) ──────

_OBS_KWARGS = {
    "normalize": True,
    "goal_max_dist": 10,
    "subgoal_max_dist": 10,
    "reduced_num_beams": 360,
}

_OBS_SPACES = [
    spaces.perception.ReducedLaserScanSpace,
    spaces.navigation.DistAngleToSubgoalSpace,
    spaces.dynamics.LastActionSpace,
]

_EXTRACTOR_KWARGS = dict(features_dim=256)
_NET_ARCH = dict(pi=[64, 64], vf=[64, 64])


# ── Register minimal test architecture stubs ──────────────────────────────────
# PPO and RecurrentPPO already have AGENT_1 / AGENT_2 registered in paper.py.
# We register thin stubs for every other algorithm so the factory can look them
# up by name during RL_Agent initialisation.


@AgentFactory.register("TEST_AGENT_A2C")
class _TestAgentA2C(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = A2C
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = _OBS_SPACES
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = _EXTRACTOR_KWARGS
    net_arch = _NET_ARCH
    activation_fn = nn.ReLU


@AgentFactory.register("TEST_AGENT_TRPO")
class _TestAgentTRPO(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = TRPO
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = _OBS_SPACES
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = _EXTRACTOR_KWARGS
    net_arch = _NET_ARCH
    activation_fn = nn.ReLU


@AgentFactory.register("TEST_AGENT_SAC")
class _TestAgentSAC(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = SAC
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = _OBS_SPACES
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = _EXTRACTOR_KWARGS
    net_arch = dict(pi=[64, 64], qf=[64, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("TEST_AGENT_TD3")
class _TestAgentTD3(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = TD3
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = _OBS_SPACES
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = _EXTRACTOR_KWARGS
    net_arch = dict(pi=[64, 64], qf=[64, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("TEST_AGENT_TQC")
class _TestAgentTQC(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = TQC
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = _OBS_SPACES
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = _EXTRACTOR_KWARGS
    net_arch = dict(pi=[64, 64], qf=[64, 64])
    activation_fn = nn.ReLU


@AgentFactory.register("TEST_AGENT_CROSSQ")
class _TestAgentCrossQ(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = CrossQ
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = _OBS_SPACES
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = _EXTRACTOR_KWARGS
    net_arch = dict(pi=[64, 64], qf=[64, 64])
    activation_fn = nn.ReLU


# ── Common fixture ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def agent_state() -> AgentConfig:
    """A minimal AgentConfig for testing SB3 algorithms."""
    return AgentConfig(
        name="test_fixture_agent",
        action_space=DifferentialDriveActionSpace(
            linear_range=(-0.5, 0.5),
            angular_range=(-1.0, 1.0),
        ),
        parameters=AgentParameters(
            laser_num_beams=360,
            laser_max_range=30.0,
            normalize=True,
        ),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(
                architecture_name="AGENT_1",
                parameters=PPO_Algorithm_Cfg(),
            )
        ),
    )


# ── Parametrize cases: (algorithm_cfg, architecture_name) ────────────────────

_ALGORITHM_CASES = [
    pytest.param(
        PPO_Cfg(
            architecture_name="AGENT_1",
            parameters=PPO_Algorithm_Cfg(total_batch_size=64, batch_size=64),
        ),
        "AGENT_1",
        id="PPO",
    ),
    pytest.param(
        A2C_Cfg(
            architecture_name="TEST_AGENT_A2C",
            parameters=A2C_Algorithm_Cfg(total_batch_size=64, batch_size=64),
        ),
        "TEST_AGENT_A2C",
        id="A2C",
    ),
    pytest.param(
        TRPO_Cfg(
            architecture_name="TEST_AGENT_TRPO",
            parameters=TRPO_Algorithm_Cfg(total_batch_size=64, batch_size=64),
        ),
        "TEST_AGENT_TRPO",
        id="TRPO",
    ),
    pytest.param(
        SAC_Cfg(
            architecture_name="TEST_AGENT_SAC",
            parameters=SAC_Algorithm_Cfg(buffer_size=1000, learning_starts=100),
        ),
        "TEST_AGENT_SAC",
        id="SAC",
    ),
    pytest.param(
        TD3_Cfg(
            architecture_name="TEST_AGENT_TD3",
            parameters=TD3_Algorithm_Cfg(buffer_size=1000, learning_starts=100),
        ),
        "TEST_AGENT_TD3",
        id="TD3",
    ),
    pytest.param(
        TQC_Cfg(
            architecture_name="TEST_AGENT_TQC",
            parameters=TQC_Algorithm_Cfg(buffer_size=1000, learning_starts=100),
        ),
        "TEST_AGENT_TQC",
        id="TQC",
    ),
    pytest.param(
        CrossQ_Cfg(
            architecture_name="TEST_AGENT_CROSSQ",
            parameters=CrossQ_Algorithm_Cfg(buffer_size=1000, learning_starts=100),
        ),
        "TEST_AGENT_CROSSQ",
        id="CrossQ",
    ),
]

# CrossQ (sb3-contrib) only registers MlpPolicy, so MultiInputPolicy — which is
# required for dict observation spaces — is unavailable.  initialize_model() is
# therefore expected to fail for CrossQ until sb3-contrib adds MultiInputPolicy.
_CROSSQ_XFAIL = pytest.mark.xfail(
    reason=(
        "sb3-contrib CrossQ only registers MlpPolicy, not MultiInputPolicy. "
        "Initialising CrossQ with multi-observation spaces is unsupported. "
        "This will pass once sb3-contrib adds a MultiInputPolicy for CrossQ."
    ),
    strict=True,
)

_ALGORITHM_CASES_INIT_MODEL = [
    *_ALGORITHM_CASES[:-1],  # all except CrossQ
    pytest.param(
        CrossQ_Cfg(
            architecture_name="TEST_AGENT_CROSSQ",
            parameters=CrossQ_Algorithm_Cfg(buffer_size=1000, learning_starts=100),
        ),
        "TEST_AGENT_CROSSQ",
        id="CrossQ",
        marks=_CROSSQ_XFAIL,
    ),
]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSB3AlgorithmConfigs:
    """Unit tests for the Pydantic algorithm configuration classes."""

    def test_ppo_cfg_defaults(self):
        cfg = PPO_Algorithm_Cfg()
        assert cfg.clip_range == 0.2
        assert cfg.gamma == 0.99
        assert cfg.batch_size > 0

    def test_a2c_cfg_defaults(self):
        cfg = A2C_Algorithm_Cfg()
        assert cfg.n_epochs == 1
        assert cfg.use_rms_prop is True

    def test_trpo_cfg_defaults(self):
        cfg = TRPO_Algorithm_Cfg()
        assert cfg.n_epochs == 1
        assert cfg.target_kl == pytest.approx(0.01)

    def test_sac_cfg_defaults(self):
        cfg = SAC_Algorithm_Cfg()
        assert cfg.ent_coef == "auto"
        assert cfg.target_entropy == "auto"

    def test_td3_cfg_defaults(self):
        cfg = TD3_Algorithm_Cfg()
        assert cfg.policy_delay == 2

    def test_tqc_cfg_defaults(self):
        cfg = TQC_Algorithm_Cfg()
        assert cfg.ent_coef == "auto"
        assert cfg.top_quantiles_to_drop_per_net == 2

    def test_crossq_cfg_defaults(self):
        cfg = CrossQ_Algorithm_Cfg()
        assert cfg.policy_delay == 1

    @pytest.mark.parametrize("algo_cfg,arch_name", _ALGORITHM_CASES)
    def test_algorithm_cfg_wrapped_in_framework_cfg(self, algo_cfg, arch_name):
        """The algorithm cfg must round-trip through the StableBaselinesCfg union."""
        framework_cfg = StableBaselinesCfg(algorithm=algo_cfg)
        assert framework_cfg.algorithm.architecture_name == arch_name


class TestSB3AgentCreation:
    """Integration tests: full RL_Agent instantiation + model setup per algorithm."""

    @pytest.mark.parametrize("algo_cfg,arch_name", _ALGORITHM_CASES)
    def test_agent_can_be_created(self, algo_cfg, arch_name, agent_state):
        """RL_Agent should be constructible for every SB3 algorithm config."""
        spec = agent_state.model_copy(
            update={
                "name": f"test_agent_{arch_name.lower()}",
                "framework": StableBaselinesCfg(algorithm=algo_cfg),
            }
        )
        agent = RL_Agent(spec)
        assert agent.model is not None
        assert agent.space_manager is not None

    @pytest.mark.parametrize("algo_cfg,arch_name", _ALGORITHM_CASES_INIT_MODEL)
    def test_agent_initialize_model(self, algo_cfg, arch_name, agent_state):
        """initialize_model() must successfully build the SB3 model with a mock env."""
        spec = agent_state.model_copy(
            update={
                "name": f"test_init_{arch_name.lower()}",
                "framework": StableBaselinesCfg(algorithm=algo_cfg),
            }
        )
        agent = RL_Agent(spec)

        mock_env = make_mock_env(
            ns="",
            space_manager=agent.space_manager,
            stack_size=agent.model._policy_description.stack_size,
        )
        agent.initialize_model(mock_env)

        assert agent.model._model is not None, (
            f"SB3 model was not created for algorithm {arch_name}"
        )


# ── Dict-parsing / auto-discovery tests ──────────────────────────────────────

# Mapping from algorithm (type tag) → (cfg class, param cfg class, unique param overrides)
_DICT_CASES = [
    pytest.param(
        "PPO",
        PPO_Cfg,
        PPO_Algorithm_Cfg,
        "AGENT_1",
        {"clip_range": 0.25, "gamma": 0.98, "total_batch_size": 64, "batch_size": 64},
        id="dict-PPO",
    ),
    pytest.param(
        "A2C",
        A2C_Cfg,
        A2C_Algorithm_Cfg,
        "TEST_AGENT_A2C",
        {"use_rms_prop": False, "gamma": 0.95, "total_batch_size": 64, "batch_size": 64},
        id="dict-A2C",
    ),
    pytest.param(
        "TRPO",
        TRPO_Cfg,
        TRPO_Algorithm_Cfg,
        "TEST_AGENT_TRPO",
        {"target_kl": 0.005, "n_critic_updates": 5, "total_batch_size": 64, "batch_size": 64},
        id="dict-TRPO",
    ),
    pytest.param(
        "SAC",
        SAC_Cfg,
        SAC_Algorithm_Cfg,
        "TEST_AGENT_SAC",
        {"buffer_size": 5000, "learning_starts": 200, "gamma": 0.98},
        id="dict-SAC",
    ),
    pytest.param(
        "TD3",
        TD3_Cfg,
        TD3_Algorithm_Cfg,
        "TEST_AGENT_TD3",
        {"buffer_size": 5000, "learning_starts": 200, "policy_delay": 3},
        id="dict-TD3",
    ),
    pytest.param(
        "TQC",
        TQC_Cfg,
        TQC_Algorithm_Cfg,
        "TEST_AGENT_TQC",
        {"buffer_size": 5000, "learning_starts": 200, "top_quantiles_to_drop_per_net": 1},
        id="dict-TQC",
    ),
    pytest.param(
        "CrossQ",
        CrossQ_Cfg,
        CrossQ_Algorithm_Cfg,
        "TEST_AGENT_CROSSQ",
        {"buffer_size": 5000, "learning_starts": 200, "policy_delay": 2},
        id="dict-CrossQ",
    ),
]


class TestSB3DictParsing:
    """Verify that algorithm configs round-trip through plain dicts (YAML-style)
    and that ``StableBaselinesCfg`` identifies the algorithm automatically via the
    ``type`` discriminator field.
    """

    @pytest.mark.parametrize(
        "type_tag,cfg_cls,param_cls,arch_name,param_overrides", _DICT_CASES
    )
    def test_algorithm_params_from_dict(
        self, type_tag, cfg_cls, param_cls, arch_name, param_overrides
    ):
        """Algorithm parameter config parses correctly from a plain dict."""
        cfg = param_cls.model_validate(param_overrides)
        assert isinstance(cfg, param_cls)
        # spot-check one representative override per parametrize case
        for key, val in param_overrides.items():
            assert getattr(cfg, key) == val, (
                f"{param_cls.__name__}.{key} expected {val}, got {getattr(cfg, key)}"
            )

    @pytest.mark.parametrize(
        "type_tag,cfg_cls,param_cls,arch_name,param_overrides", _DICT_CASES
    )
    def test_algorithm_cfg_from_dict(
        self, type_tag, cfg_cls, param_cls, arch_name, param_overrides
    ):
        """Full *_Cfg envelope parses correctly from a plain dict including ``type``."""
        raw = {
            "type": type_tag,
            "architecture_name": arch_name,
            "parameters": param_overrides,
        }
        cfg = cfg_cls.model_validate(raw)
        assert isinstance(cfg, cfg_cls)
        assert isinstance(cfg.parameters, param_cls)
        assert cfg.architecture_name == arch_name
        assert cfg.type == type_tag

    @pytest.mark.parametrize(
        "type_tag,cfg_cls,param_cls,arch_name,param_overrides", _DICT_CASES
    )
    def test_stable_baselines_cfg_auto_discovers_algorithm(
        self, type_tag, cfg_cls, param_cls, arch_name, param_overrides
    ):
        """StableBaselinesCfg selects the correct *_Cfg subclass from a raw dict
        using the ``type`` discriminator — no explicit Python type required.
        """
        raw = {
            "algorithm": {
                "type": type_tag,
                "architecture_name": arch_name,
                "parameters": param_overrides,
            }
        }
        framework_cfg = StableBaselinesCfg.model_validate(raw)
        assert isinstance(framework_cfg.algorithm, cfg_cls), (
            f"Expected {cfg_cls.__name__}, got {type(framework_cfg.algorithm).__name__}"
        )
        assert isinstance(framework_cfg.algorithm.parameters, param_cls)

    @pytest.mark.parametrize(
        "type_tag,cfg_cls,param_cls,arch_name,param_overrides", _DICT_CASES
    )
    def test_agent_cfg_from_full_dict(
        self, type_tag, cfg_cls, param_cls, arch_name, param_overrides, agent_state
    ):
        """Complete AgentConfig → RL_Agent pipeline works when the entire config
        originates from a nested dict (simulating YAML file loading).
        """
        raw_agent_spec = {
            "name": f"dict_test_{type_tag.lower()}",
            "action_space": {"type": "differential_drive"},
            "framework": {
                "name": "stable_baselines3",
                "algorithm": {
                    "type": type_tag,
                    "architecture_name": arch_name,
                    "parameters": param_overrides,
                },
            },
        }
        spec = AgentConfig.model_validate(raw_agent_spec)
        assert isinstance(spec.framework.algorithm, cfg_cls)
        assert isinstance(spec.framework.algorithm.parameters, param_cls)

        agent = RL_Agent(spec)
        assert agent.model is not None
        assert agent.space_manager is not None


class TestSB3ModelStateIsolation:
    """Regression tests for P0.1/P0.2 (audit 2026-07-04): ``StableBaselinesModel``
    used to share one mutable ``StableBaselinesModelState`` across every instance,
    and ``reset()`` crashed on ndarray ``last_observation`` values.
    """

    def test_state_is_not_shared_between_instances(self, agent_state):
        spec_a = agent_state.model_copy(update={"name": "state_iso_a"})
        spec_b = agent_state.model_copy(update={"name": "state_iso_b"})
        agent_a = RL_Agent(spec_a)
        agent_b = RL_Agent(spec_b)

        agent_a.model._StableBaselinesModel__state.last_observation = np.array(
            [1.0, 2.0, 3.0]
        )

        assert agent_b.model._StableBaselinesModel__state.last_observation is None
        assert (
            agent_a.model._StableBaselinesModel__state
            is not agent_b.model._StableBaselinesModel__state
        )

    def test_reset_does_not_raise_on_ndarray_last_observation(self, agent_state):
        spec = agent_state.model_copy(update={"name": "state_reset_ndarray"})
        agent = RL_Agent(spec)

        agent.model._StableBaselinesModel__state.last_observation = np.array(
            [1.0, 2.0, 3.0]
        )

        # Must not raise "truth value of an array is ambiguous".
        agent.model.reset()


class TestRLAgentResetCascade:
    """Regression test for P0.3 (audit 2026-07-04): episode reset must clear
    stateful observation-space buffers, not just the model's internal state.
    """

    def test_agent_reset_cascades_to_model_and_spaces(self, agent_state):
        spec = agent_state.model_copy(update={"name": "reset_cascade"})
        agent = RL_Agent(spec)

        with patch.object(agent.model, "reset") as mock_model_reset, patch.object(
            agent.space_manager, "reset_spaces"
        ) as mock_space_reset:
            agent.reset()

        mock_model_reset.assert_called_once()
        mock_space_reset.assert_called_once()

