from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy, MultiInputActorCriticPolicy

__all__ = [
    "artanh",
    "expmap0",
    "logmap0",
    "mobius_add",
    "mobius_matvec",
    "mobius_pointwise_fn",
    "MobiusLinear",
    "MobiusActivation",
    "HyperbolicMlpExtractor",
    "HyperbolicActorCriticPolicy",
    "HyperbolicPPO",
]


# ==============================================================================
# Mathematical Primitives on the Poincaré Ball (Plain PyTorch)
# ==============================================================================


def artanh(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse hyperbolic tangent with boundary clamping for numerical stability.

    Computes 0.5 * (ln(1 + x) - ln(1 - x)) while clamping inputs to [-1 + eps, 1 - eps].
    """
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _project(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Safe projection onto the Poincaré ball with radius < (1 - 1e-3) / sqrt(c).

    Ensures points stay strictly inside the ball to prevent boundary singularities.
    """
    c = abs(c)
    sqrt_c = c**0.5
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), eps)
    maxnorm = (1.0 - 1e-3) / sqrt_c
    cond = norm > maxnorm
    projected = (x / norm) * maxnorm
    return torch.where(cond, projected, x)


def expmap0(u: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map from tangent space at origin to the Poincaré ball.

    Maps Euclidean tangent vectors u at 0 to points in the Poincaré ball D_c^n.
    """
    c = abs(c)
    sqrt_c = c**0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), eps)
    gamma = torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return _project(gamma, c=c, eps=eps)


def logmap0(y: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Logarithmic map from the Poincaré ball to the tangent space at origin.

    Inverse of expmap0, projecting points on D_c^n back to Euclidean tangent space.
    """
    c = abs(c)
    sqrt_c = c**0.5
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), eps)
    return (y / y_norm / sqrt_c) * artanh(sqrt_c * y_norm, eps=eps)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Möbius vector addition in the Poincaré ball.

    Computes x ⊕_c y in gyrovector space with safe projection.
    """
    c = abs(c)
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c**2) * x2 * y2
    return _project(num / (denom + 1e-5), c=c)


def mobius_matvec(M: torch.Tensor, x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Möbius matrix-vector multiplication with safe autograd division.

    Computes M ⊗_c x = (1 / sqrt_c) * tanh((||Mx|| / ||x||) * artanh(sqrt_c * ||x||)) * (Mx / ||Mx||).
    Clamps norms to prevent NaN gradients when Mx == 0 or x == 0.
    """
    c = abs(c)
    sqrt_c = c**0.5
    x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), eps)
    mx = x @ M.t()
    mx_norm = torch.clamp_min(mx.norm(dim=-1, keepdim=True, p=2), eps)
    res_c = torch.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm, eps=eps)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).all(dim=-1, keepdim=True)
    res = torch.where(cond, torch.zeros_like(res_c), res_c)
    return _project(res, c=c, eps=eps)


def mobius_pointwise_fn(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    c: float = 1.0,
) -> torch.Tensor:
    """Pointwise non-linearity through the logarithmic map: Exp_0(fn(Log_0(x))).

    Pulls coordinates into tangent space at origin via Log_0, applies Euclidean non-linearity,
    and maps back to the Poincaré ball via Exp_0.
    """
    return expmap0(fn(logmap0(x, c=c)), c=c)


# ==============================================================================
# PyTorch Modules for Hyperbolic Neural Networks
# ==============================================================================


class MobiusLinear(nn.Module):
    """Hyperbolic linear layer operating on the Poincaré ball: y = (M ⊗_c x) ⊕_c Exp^c_0(b)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        c: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = abs(c)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights matching standard PyTorch nn.Linear conventions."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying Möbius matrix-vector product followed by Möbius bias addition."""
        mv = mobius_matvec(self.weight, x, c=self.c)
        if self.bias is None:
            return mv
        b_hyp = expmap0(self.bias, c=self.c)
        return mobius_add(mv, b_hyp, c=self.c)


class MobiusActivation(nn.Module):
    """Pointwise activation wrapper operating through the logarithmic map."""

    def __init__(
        self,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        c: float = 1.0,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.c = abs(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the non-linearity in tangent space at origin."""
        return mobius_pointwise_fn(x, self.activation_fn, c=self.c)


class HyperbolicMlpExtractor(nn.Module):
    """MlpExtractor with hyperbolic policy and value heads on the Poincaré ball.

    Maps Euclidean backbone representations into hyperbolic space using expmap0,
    evaluates policy and value streams with MobiusLinear and MobiusActivation layers,
    and projects back into tangent space via logmap0 for SB3 action and value linear layers.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module] = nn.ReLU,
        c: float = 1.0,
    ) -> None:
        super().__init__()
        self.c = abs(c)

        if isinstance(net_arch, dict):
            pi_layers = list(net_arch.get("pi", [32, 128, 32]))
            vf_layers = list(net_arch.get("vf", [32, 128, 32]))
        elif isinstance(net_arch, list):
            pi_layers = list(net_arch)
            vf_layers = list(net_arch)
        else:
            pi_layers = [32, 128, 32]
            vf_layers = [32, 128, 32]

        self.latent_dim_pi = pi_layers[-1] if len(pi_layers) > 0 else feature_dim
        self.latent_dim_vf = vf_layers[-1] if len(vf_layers) > 0 else feature_dim

        # Policy stream (pi)
        policy_modules: list[nn.Module] = []
        in_dim = feature_dim
        for h in pi_layers:
            policy_modules.append(MobiusLinear(in_dim, h, c=self.c))
            policy_modules.append(MobiusActivation(activation_fn(), c=self.c))
            in_dim = h
        self.policy_net = nn.Sequential(*policy_modules)

        # Value stream (vf)
        value_modules: list[nn.Module] = []
        in_dim = feature_dim
        for h in vf_layers:
            value_modules.append(MobiusLinear(in_dim, h, c=self.c))
            value_modules.append(MobiusActivation(activation_fn(), c=self.c))
            in_dim = h
        self.value_net = nn.Sequential(*value_modules)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both policy and value networks.

        Returns:
            Tuple of (latent_pi, latent_vf) in tangent space.
        """
        hyp_features = expmap0(features, c=self.c)
        hyp_pi = self.policy_net(hyp_features)
        hyp_vf = self.value_net(hyp_features)
        return logmap0(hyp_pi, c=self.c), logmap0(hyp_vf, c=self.c)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for the policy network returning latent_pi in tangent space."""
        hyp_features = expmap0(features, c=self.c)
        return logmap0(self.policy_net(hyp_features), c=self.c)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for the value network returning latent_vf in tangent space."""
        hyp_features = expmap0(features, c=self.c)
        return logmap0(self.value_net(hyp_features), c=self.c)


# ==============================================================================
# Stable-Baselines3 Policy & Algorithm Classes
# ==============================================================================


class HyperbolicActorCriticPolicy(MultiInputActorCriticPolicy):
    """Subclass of SB3 MultiInputActorCriticPolicy overriding _build_mlp_extractor with HyperbolicMlpExtractor."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HyperbolicMlpExtractor(
            feature_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            c=1.0,
        )


class HyperbolicPPO(PPO):
    """PPO algorithm variant configured with HyperbolicActorCriticPolicy as its MultiInputPolicy."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        **PPO.policy_aliases,
        "MultiInputPolicy": HyperbolicActorCriticPolicy,
    }
