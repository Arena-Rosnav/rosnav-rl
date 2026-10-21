"""Tests for the Hamiltonian crowd predictor and its unstructured control."""

from __future__ import annotations

import torch

from rosnav_rl.model.stable_baselines3.policy.feature_extractors.ham_crowd_predictor import (
    HamCrowdPredictor,
    PlainCrowdPredictor,
    count_parameters,
    matched_pair,
)


def _crowd(batch=3, k=5, real=3):
    torch.manual_seed(0)
    crowd = torch.zeros(batch, k, 5)
    crowd[:, :real, :2] = torch.randn(batch, real, 2) * 3
    crowd[:, :real, 2:4] = torch.randn(batch, real, 2) * 0.5
    crowd[:, :real, 4] = 1
    return crowd, torch.zeros(batch, 2)


def test_shapes_and_padding_stay_zero():
    crowd, rv = _crowd()
    for net in (HamCrowdPredictor(16), PlainCrowdPredictor(16)):
        state, energy, delta = net(crowd, rv)
        assert state.shape == (3, 5, 4) and energy.shape == (3,) and delta.shape == (3,)
        assert torch.isfinite(state).all()
        assert (state[:, 3:] == 0).all()


def test_dissipation_is_nonnegative_and_removes_kinetic_energy_without_potential():
    net = HamCrowdPredictor(16)
    for p in net.potential.parameters():
        torch.nn.init.zeros_(p)
    crowd, rv = _crowd()
    _, _, delta = net(crowd, rv)

    assert (delta <= 1e-5).all()


def test_parameter_counts_are_matched():
    ham, plain = matched_pair(64)

    gap = abs(count_parameters(ham) - count_parameters(plain)) / count_parameters(ham)
    assert gap < 0.01


def test_gradients_reach_the_parameters():
    net = HamCrowdPredictor(16)
    net.train()
    crowd, rv = _crowd()
    state, _, _ = net(crowd, rv)
    state.pow(2).sum().backward()

    assert all(p.grad is not None for p in net.parameters())
