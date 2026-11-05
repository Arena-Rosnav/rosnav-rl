"""Tests for the AGENT_5-compatible crowd-energy feature extractor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rosnav_rl.model.stable_baselines3.policy.feature_extractors.ham_crowd_predictor import count_parameters
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.ham_extractor import DRL_VO_HAM_EXTRACTOR


def _space():
    return spaces.Dict(
        {
            "StackedLaserMapSpace": spaces.Box(-1, 1, (1, 80, 80), np.float32),
            # The velocity maps are emitted in physical m/s by their encoders.
            "PedestrianVelXSpace": spaces.Box(-3, 3, (80, 80), np.float32),
            "PedestrianVelYSpace": spaces.Box(-3, 3, (80, 80), np.float32),
            "PedestrianTypeSpace": spaces.Box(-1, 3, (80, 80), np.float32),
            "PedestrianSocialStateSpace": spaces.Box(-1, 3, (80, 80), np.float32),
            "DistAngleToSubgoalSpace": spaces.Box(-10, 10, (2,), np.float32),
            "LastActionSpace": spaces.Box(
                np.array([-0.2, -1.5], np.float32), np.array([1.0, 1.5], np.float32)
            ),
        }
    )


def _obs(batch=2, include_pedestrian=True):
    torch.manual_seed(0)
    obs = {
        "StackedLaserMapSpace": torch.zeros(batch, 1, 80, 80),
        "PedestrianVelXSpace": torch.zeros(batch, 80, 80),
        "PedestrianVelYSpace": torch.zeros(batch, 80, 80),
        # PedestrianTypeSpace emits raw type codes: -1 is empty and 0 is occupied.
        "PedestrianTypeSpace": torch.full((batch, 80, 80), -1.0),
        "PedestrianSocialStateSpace": torch.zeros(batch, 80, 80),
        "DistAngleToSubgoalSpace": torch.zeros(batch, 2),
        "LastActionSpace": torch.zeros(batch, 2),
    }
    if include_pedestrian:
        y, x = 40, 44
        obs["PedestrianTypeSpace"][:, y, x] = 0.0  # physical type 0, occupied
        obs["PedestrianVelXSpace"][:, y, x] = 0.5  # physical +0.5 m/s
        obs["PedestrianVelYSpace"][:, y, x] = 0.0
    return obs


def _extractor(variant):
    torch.manual_seed(0)
    return DRL_VO_HAM_EXTRACTOR(
        _space(), features_dim=64, stack_size=1, width_per_group=64, variant=variant, map_roi_m=24.0
    ).eval()


@pytest.mark.parametrize("variant", ["ham", "plain"])
def test_output_shape_and_finite_values(variant):
    output = _extractor(variant)(_obs())
    assert output.shape == (2, 64)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("variant", ["ham", "plain"])
def test_reconstructed_crowd_uses_existing_agent5_maps(variant):
    ex = _extractor(variant)
    crowd = ex.crowd_from_feature_maps(_obs())
    assert crowd.shape == (2, 24, 5)
    torch.testing.assert_close(crowd[:, 0, 0], torch.tensor([1.35, 1.35]))
    torch.testing.assert_close(crowd[:, 0, 1], torch.tensor([0.15, 0.15]))
    torch.testing.assert_close(crowd[:, 0, 2], torch.tensor([0.5, 0.5]))
    assert torch.all(crowd[:, 0, 4] == 1)
    assert torch.all(crowd[:, 1:, 4] == 0)


@pytest.mark.parametrize("variant", ["ham", "plain"])
def test_crowd_changes_the_features(variant):
    ex = _extractor(variant)
    with_pedestrian = ex(_obs(include_pedestrian=True))
    empty = ex(_obs(include_pedestrian=False))
    assert not torch.allclose(with_pedestrian, empty)


def test_stationary_pedestrians_are_not_dropped():
    ex = _extractor("ham")
    obs = _obs()
    obs["PedestrianVelXSpace"][:, 40, 44] = 0.0
    crowd = ex.crowd_from_feature_maps(obs)
    assert torch.all(crowd[:, 0, 4] == 1)
    assert torch.all(crowd[:, 0, 2] == 0)


def test_normalized_action_is_converted_to_m_per_s():
    ex = _extractor("ham")
    v = ex._robot_velocity(torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]))
    torch.testing.assert_close(v[:, 0], torch.tensor([1.0, -0.2, 0.4]))
    assert (v[:, 1] == 0).all()
    assert (v[:, 2] == 0).all()  # second differential-drive component is yaw rate only when nonzero
    turn = ex._robot_velocity(torch.tensor([[0.0, 1.0], [0.0, -1.0]]))
    torch.testing.assert_close(turn[:, 2], torch.tensor([1.5, -1.5]))


def test_hamiltonian_and_plain_predictors_have_equal_parameter_count():
    ex = _extractor("ham")
    plain = _extractor("plain")
    assert count_parameters(ex.predictor) == count_parameters(plain.predictor)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        DRL_VO_HAM_EXTRACTOR(_space(), features_dim=64, stack_size=1, variant="nope")


def test_gradients_reach_the_predictor_through_the_extractor():
    ex = _extractor("ham").train()
    ex(_obs()).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ex.predictor.parameters())
