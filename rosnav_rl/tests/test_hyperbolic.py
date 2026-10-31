"""Unit tests for Poincaré ball operations, Möbius neural network layers,
Hyperbolic policy, and AGENT_5 hyperbolic / Euclidean twin architecture.
"""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from rosnav_rl.model.stable_baselines3.policy.agent_factory import AgentFactory
from rosnav_rl.model.stable_baselines3.policy.constants import POLICY_TYPE
from rosnav_rl.model.stable_baselines3.policy.hyperbolic import (
    HyperbolicActorCriticPolicy,
    HyperbolicMlpExtractor,
    HyperbolicPPO,
    MobiusActivation,
    MobiusLinear,
    _project,
    artanh,
    expmap0,
    logmap0,
    mobius_add,
    mobius_matvec,
)

# ==============================================================================
# 1. Mathematical Primitives Tests
# ==============================================================================


def test_artanh_correctness_and_clamping():
    """Verify artanh computes accurate values and does not overflow near +/- 1."""
    inputs = torch.tensor([0.0, 0.5, -0.5, 0.999999, -0.999999, 2.0, -2.0])
    res = artanh(inputs)

    assert not torch.isnan(res).any(), "artanh produced NaN values"
    assert not torch.isinf(res).any(), "artanh produced infinite values"

    # Known mathematical identity: tanh(artanh(x)) == x for |x| < 1
    test_val = torch.tensor([0.25, 0.5, -0.75])
    assert torch.allclose(torch.tanh(artanh(test_val)), test_val, atol=1e-5)


def test_project_boundary():
    """Verify points outside the ball are projected safely inside maxnorm."""
    c = 1.0
    maxnorm = (1.0 - 1e-3) / (c**0.5)

    inside = torch.tensor([[0.1, 0.2]])
    assert torch.allclose(_project(inside, c=c), inside)

    outside = torch.tensor([[5.0, 5.0]])
    projected = _project(outside, c=c)
    norm = projected.norm(dim=-1).item()
    assert norm <= maxnorm + 1e-6, f"Projected norm {norm} exceeds maxnorm {maxnorm}"


def test_expmap0_logmap0_roundtrip():
    """Verify expmap0 and logmap0 are inverse isomorphisms near the origin."""
    torch.manual_seed(42)
    u = torch.randn(20, 16) * 0.1
    c = 1.0

    p = expmap0(u, c=c)
    assert (p.norm(dim=-1) < 1.0).all(), "Points mapped outside Poincaré ball"

    u_rec = logmap0(p, c=c)
    assert torch.allclose(u, u_rec, atol=1e-4), f"Max roundtrip diff: {(u - u_rec).abs().max()}"


def test_mobius_add_identity():
    """Verify x ⊕ 0 = x and 0 ⊕ y = y."""
    torch.manual_seed(42)
    x = expmap0(torch.randn(10, 8) * 0.2, c=1.0)
    zeros = torch.zeros_like(x)

    assert torch.allclose(mobius_add(x, zeros, c=1.0), x, atol=1e-5)
    assert torch.allclose(mobius_add(zeros, x, c=1.0), x, atol=1e-5)


def test_mobius_matvec_zero_safety():
    """Verify mobius_matvec handles zero inputs without causing NaN gradients."""
    weight = torch.randn(16, 8, requires_grad=True)
    zero_x = torch.zeros(4, 8, requires_grad=True)

    out = mobius_matvec(weight, zero_x, c=1.0)
    assert not torch.isnan(out).any(), "Zero vector input produced NaN output"
    assert torch.allclose(out, torch.zeros(4, 16)), "Zero input did not yield zero output"

    loss = out.sum()
    loss.backward()
    assert not torch.isnan(weight.grad).any(), "NaN in weight gradients when input was zero"


# ==============================================================================
# 2. PyTorch Modules Tests
# ==============================================================================


def test_mobius_linear_layer():
    """Verify MobiusLinear initialization, forward pass, and autograd backward."""
    layer = MobiusLinear(in_features=32, out_features=16, c=1.0, bias=True)
    x = expmap0(torch.randn(8, 32) * 0.1, c=1.0)

    out = layer(x)
    assert out.shape == (8, 16)
    assert (out.norm(dim=-1) < 1.0).all()

    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None and not torch.isnan(layer.weight.grad).any()
    assert layer.bias.grad is not None and not torch.isnan(layer.bias.grad).any()


def test_mobius_activation():
    """Verify MobiusActivation applies activation through the logarithmic map."""
    act = MobiusActivation(activation_fn=torch.relu, c=1.0)
    x = expmap0(torch.tensor([[-0.5, 0.5]]), c=1.0)

    out = act(x)
    u = logmap0(out, c=1.0)
    assert u[0, 0].item() == pytest.approx(0.0, abs=1e-4)
    assert u[0, 1].item() > 0.0


# ==============================================================================
# 3. MlpExtractor and Matched Parameter Count (Hyp2Nav Head Config: 49,536 params)
# ==============================================================================


def test_hyperbolic_mlp_extractor_capacity_match():
    """Verify AGENT_5 Hyperbolic head parameter count matches Euclidean twin with Hyp2Nav head config."""
    feature_dim = 512
    net_arch = dict(pi=[32, 128, 32], vf=[32, 128, 32])

    hyp_extractor = HyperbolicMlpExtractor(
        feature_dim=feature_dim,
        net_arch=net_arch,
        activation_fn=nn.ReLU,
        c=1.0,
    )

    # 1. Check latent output dimensions
    assert hyp_extractor.latent_dim_pi == 32
    assert hyp_extractor.latent_dim_vf == 32

    # 2. Check forward pass shapes
    features = torch.randn(4, feature_dim, requires_grad=True)
    pi_lat, vf_lat = hyp_extractor(features)
    assert pi_lat.shape == (4, 32)
    assert vf_lat.shape == (4, 32)

    # 3. Check forward_actor and forward_critic
    assert hyp_extractor.forward_actor(features).shape == (4, 32)
    assert hyp_extractor.forward_critic(features).shape == (4, 32)

    # 4. Count parameters in hyperbolic heads
    hyp_params = sum(p.numel() for p in hyp_extractor.parameters() if p.requires_grad)

    # Manual calculation from Hyp2Nav head specification [32, 128, 32]:
    # pi hidden 1: 512 * 32 + 32   = 16,416
    # pi hidden 2: 32 * 128 + 128  =  4,224
    # pi hidden 3: 128 * 32 + 32   =  4,128
    # Subtotal pi: 16,416 + 4,224 + 4,128 = 24,768
    # vf hidden 1: 512 * 32 + 32   = 16,416
    # vf hidden 2: 32 * 128 + 128  =  4,224
    # vf hidden 3: 128 * 32 + 32   =  4,128
    # Subtotal vf: 24,768
    # Total = 24,768 + 24,768 = 49,536 (~50K)
    expected_params = 49536
    assert hyp_params == expected_params, f"Hyperbolic params {hyp_params} != {expected_params}"

    # 5. Compare with standard Euclidean Twin layers
    euc_pi = nn.Sequential(
        nn.Linear(512, 32),
        nn.ReLU(),
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
    )
    euc_vf = nn.Sequential(
        nn.Linear(512, 32),
        nn.ReLU(),
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
    )
    euc_params = sum(p.numel() for m in [euc_pi, euc_vf] for p in m.parameters() if p.requires_grad)
    assert euc_params == expected_params, f"Euclidean twin params {euc_params} != {expected_params}"
    assert hyp_params == euc_params, "Hyperbolic and Euclidean Twin parameter counts do not match 1:1!"


# ==============================================================================
# 4. Stable-Baselines3 Policy & Algorithm Integration Tests
# ==============================================================================


def test_hyperbolic_policy_and_ppo():
    """Verify HyperbolicActorCriticPolicy and HyperbolicPPO can be instantiated and run."""
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=float)})
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

    policy = HyperbolicActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 0.0003,
        net_arch=dict(pi=[32, 128, 32], vf=[32, 128, 32]),
    )

    batch_obs = {"obs": torch.randn(5, 16)}
    actions, values, log_prob = policy(batch_obs)

    assert actions.shape == (5, 2)
    assert values.shape == (5, 1)
    assert log_prob.shape == (5,)

    assert HyperbolicPPO.policy_aliases["MultiInputPolicy"] == HyperbolicActorCriticPolicy
    assert POLICY_TYPE[HyperbolicPPO] == "MultiInputPolicy"


def test_agent_factory_hyperbolic_and_twin_registration():
    """Verify AGENT_5_HYPERBOLIC and AGENT_5_EUCLIDEAN_TWIN are registered in AgentFactory."""
    hyp_agent = AgentFactory.instantiate("AGENT_5_HYPERBOLIC")
    euc_agent = AgentFactory.instantiate("AGENT_5_EUCLIDEAN_TWIN")

    assert hyp_agent.algorithm_class == HyperbolicPPO
    assert euc_agent.algorithm_class == PPO
    assert hyp_agent.net_arch == euc_agent.net_arch == {"pi": [32, 128, 32], "vf": [32, 128, 32]}
