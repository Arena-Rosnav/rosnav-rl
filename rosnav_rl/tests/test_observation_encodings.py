import numpy as np
import pytest

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.spaces import BaseSpaceManager, SpaceFactory
from rosnav_rl.spaces.observation_space.spaces.navigation.basic_navigation_spaces import DistAngleToGoalSpace

FAR_GOAL = {"dist_angle_to_goal": np.array([25.0, 0.5], dtype=np.float32)}


def _spec(encodings: dict[str, int]) -> AgentConfig:
    return AgentConfig(
        name="encodings",
        action_space=DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5)),
        parameters=AgentParameters(goal_max_dist=10.0, normalize=False),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_5", parameters=PPO_Algorithm_Cfg(device="cpu"))
        ),
        observation_encodings=encodings,
    )


def _goal_distance(encodings: dict[str, int]) -> float:
    manager = BaseSpaceManager(_spec(encodings), [DistAngleToGoalSpace])
    return float(np.asarray(manager.encode_observation(FAR_GOAL)).ravel()[0])


def test_current_encodings_name_every_space_past_version_1():
    assert SpaceFactory.current_encodings() == {
        "DistAngleToGoalSpace": 2,
        "LaserCartesianMapSpace": 2,
        "StackedLaserMapSpace": 2,
    }


def test_an_agent_without_recorded_encodings_reads_version_1():
    assert _goal_distance({}) == pytest.approx(25.0)


def test_recorded_encodings_reach_the_spaces():
    assert _goal_distance({"DistAngleToGoalSpace": 2}) == pytest.approx(10.0)


def test_an_encoding_newer_than_the_space_is_refused():
    with pytest.raises(ValueError, match="encodings 1 to 2"):
        DistAngleToGoalSpace(encoding_version=3)


def test_encodings_survive_the_saved_agent_config():
    spec = _spec(SpaceFactory.current_encodings())

    assert AgentConfig.model_validate(spec.model_dump()).observation_encodings == SpaceFactory.current_encodings()
