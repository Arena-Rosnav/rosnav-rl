"""Tests for the extractor that reads the predicted crowd energy change."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rosnav_rl.model.stable_baselines3.policy.feature_extractors.ham_extractor import DRL_VO_HAM_EXTRACTOR

K = 6


def _space():
    return spaces.Dict(
        {
            "StackedLaserMapSpace": spaces.Box(-1, 1, (1, 80, 80), np.float32),
            "PedestrianVelXSpace": spaces.Box(-3, 3, (80, 80), np.float32),
            "PedestrianVelYSpace": spaces.Box(-3, 3, (80, 80), np.float32),
            "DistAngleToSubgoalSpace": spaces.Box(-10, 10, (2,), np.float32),
            "LastActionSpace": spaces.Box(np.array([-0.2, -1.5], np.float32), np.array([1.0, 1.5], np.float32)),
            "CrowdSetSpace": spaces.Box(-12, 12, (K, 5), np.float32),
        }
    )


def _obs(crowd=None, action=None, batch=2):
    torch.manual_seed(0)
    if crowd is None:
        crowd = torch.zeros(batch, K, 5)
        crowd[:, :3, :2] = torch.randn(batch, 3, 2) * 3
        crowd[:, :3, 2:4] = torch.randn(batch, 3, 2) * 0.5
        crowd[:, :3, 4] = 1
    return {
        "StackedLaserMapSpace": torch.zeros(batch, 1, 80, 80),
        "PedestrianVelXSpace": torch.zeros(batch, 80, 80),
        "PedestrianVelYSpace": torch.zeros(batch, 80, 80),
        "DistAngleToSubgoalSpace": torch.zeros(batch, 2),
        "LastActionSpace": torch.zeros(batch, 2) if action is None else action,
        "CrowdSetSpace": crowd,
    }


def _extractor(variant):
    torch.manual_seed(0)
    return DRL_VO_HAM_EXTRACTOR(_space(), features_dim=64, stack_size=1, width_per_group=64, variant=variant).eval()


@pytest.mark.parametrize("variant", ["ham", "plain"])
def test_output_shape(variant):
    assert _extractor(variant)(_obs()).shape == (2, 64)


@pytest.mark.parametrize("variant", ["ham", "plain"])
def test_crowd_changes_the_features(variant):
    ex = _extractor(variant)
    a = ex(_obs())
    b = ex(_obs(crowd=torch.zeros(2, K, 5)))

    assert not torch.allclose(a, b)


def test_normalized_action_is_converted_to_m_per_s():
    ex = _extractor("ham")
    v = ex._robot_velocity(torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]))

    torch.testing.assert_close(v[:, 0], torch.tensor([1.0, -0.2, 0.4]))
    assert (v[:, 1] == 0).all()


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        DRL_VO_HAM_EXTRACTOR(_space(), features_dim=64, stack_size=1, variant="nope")


def test_gradients_reach_the_predictor_through_the_extractor():
    ex = _extractor("ham").train()
    ex(_obs()).sum().backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ex.predictor.parameters())
