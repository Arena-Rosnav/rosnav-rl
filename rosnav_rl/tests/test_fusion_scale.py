"""The AGENT_5 fusion head meets the goal and last action with the pooled map features at unit norm, whatever the map magnitude."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from rosnav_rl.model.stable_baselines3.policy.feature_extractors.resnet.resnet import DRL_VO_ROSNAV_EXTRACTOR

SPACE = gym.spaces.Dict(
    {
        "StackedLaserMapSpace": gym.spaces.Box(0.0, 1.0, (1, 80, 80)),
        "PedestrianVelXSpace": gym.spaces.Box(-1.0, 1.0, (80, 80)),
        "PedestrianVelYSpace": gym.spaces.Box(-1.0, 1.0, (80, 80)),
        "DistAngleToSubgoalSpace": gym.spaces.Box(-1.0, 1.0, (2,)),
        "LastActionSpace": gym.spaces.Box(-1.0, 1.0, (3,)),
    }
)


@pytest.mark.parametrize("map_gain", [1.0, 100.0])
def test_head_sees_unit_norm_map_features_next_to_unchanged_goal_fields(map_gain: float):
    torch.manual_seed(0)
    extractor = DRL_VO_ROSNAV_EXTRACTOR(SPACE, features_dim=512, width_per_group=64, stack_size=1).eval()
    rng = np.random.default_rng(0)
    obs = {
        key: torch.as_tensor(np.stack([space.sample() for _ in range(4)]), dtype=torch.float32)
        for key, space in SPACE.spaces.items()
    }
    obs["StackedLaserMapSpace"] = obs["StackedLaserMapSpace"] * map_gain
    obs["DistAngleToSubgoalSpace"] = torch.as_tensor(rng.uniform(-1.0, 1.0, (4, 2)), dtype=torch.float32)
    head_inputs = []
    extractor.linear_fc[1].register_forward_hook(lambda module, inputs, output: head_inputs.append(inputs[0]))
    with torch.no_grad():
        extractor(obs)
    head_in = head_inputs[0]
    maps = head_in[:, :-5]
    assert torch.linalg.vector_norm(maps, dim=1).numpy() == pytest.approx(np.ones(4), abs=1e-3)
    assert torch.equal(head_in[:, -5:-3], obs["DistAngleToSubgoalSpace"])
    assert torch.equal(head_in[:, -3:], obs["LastActionSpace"])
