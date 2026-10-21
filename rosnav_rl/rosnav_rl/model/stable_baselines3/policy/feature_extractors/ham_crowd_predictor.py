"""Crowd predictors over the padded crowd set: a port-Hamiltonian one and an unstructured control.

Both map a crowd set (B, K, 5) = [x, y, vx, vy, mask] and a candidate robot velocity (B, 2) to the crowd state one
second ahead (B, K, 4) and a scalar kinetic-plus-potential energy per sample. Positions are in the robot frame at the
start of the horizon, the robot moves at the candidate velocity.
"""

from __future__ import annotations

import torch
from torch import nn

HORIZON_STEPS = 10
DT = 0.1


def _mlp(inp: int, hidden: int, out: int, depth: int = 2, last_bias: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = inp
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.Tanh()]
        width = hidden
    layers.append(nn.Linear(width, out, bias=last_bias))
    return nn.Sequential(*layers)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class HamCrowdPredictor(nn.Module):
    """H = sum_i 1/2 |v_i|^2 + U, integrated with symplectic Euler and a dissipation that can only remove energy.

    U sums a learned pairwise term between each pedestrian and the robot, so U depends on the pedestrian positions,
    the robot position and the candidate robot velocity. Dissipation is -D v with D = (network output)^2 >= 0.
    """

    def __init__(
        self, hidden: int = 64, horizon_steps: int = HORIZON_STEPS, dt: float = DT, diss_hidden: int | None = None
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.dt = dt
        # pairwise potential input: relative position (2) and candidate robot velocity (2)
        self.potential = _mlp(4, hidden, 1, last_bias=False)
        # dissipation coefficient from the pedestrian speed
        self.dissipation = _mlp(1, diss_hidden or max(hidden // 4, 4), 1, depth=1)

    def potential_energy(self, pos: torch.Tensor, mask: torch.Tensor, robot_pos: torch.Tensor, robot_vel: torch.Tensor):
        rel = pos - robot_pos[:, None, :]
        vel = robot_vel[:, None, :].expand_as(rel)
        u = self.potential(torch.cat([rel, vel], dim=-1)).squeeze(-1)
        return (u * mask).sum(dim=-1)

    def energy(self, pos, vel, mask, robot_pos, robot_vel) -> torch.Tensor:
        kinetic = 0.5 * ((vel**2).sum(dim=-1) * mask).sum(dim=-1)
        return kinetic + self.potential_energy(pos, mask, robot_pos, robot_vel)

    def _force(self, pos, mask, robot_pos, robot_vel) -> torch.Tensor:
        with torch.enable_grad():
            pos_g = pos if pos.requires_grad else pos.detach().requires_grad_(True)
            u = self.potential_energy(pos_g, mask, robot_pos, robot_vel)
            (grad,) = torch.autograd.grad(u.sum(), pos_g, create_graph=self.training)
        return -grad

    def forward(self, crowd: torch.Tensor, robot_vel: torch.Tensor):
        pos, vel, mask = crowd[..., 0:2], crowd[..., 2:4], crowd[..., 4]
        m = mask[..., None]
        robot_pos = torch.zeros_like(robot_vel)
        energy0 = self.energy(pos, vel, mask, robot_pos, robot_vel)
        for _ in range(self.horizon_steps):
            force = self._force(pos, mask, robot_pos, robot_vel)
            damping = self.dissipation(vel.norm(dim=-1, keepdim=True)) ** 2
            vel = (vel + self.dt * (force - damping * vel)) * m
            pos = pos + self.dt * vel
            robot_pos = robot_pos + self.dt * robot_vel
        energy1 = self.energy(pos, vel, mask, robot_pos, robot_vel)
        state = torch.cat([pos, vel], dim=-1) * m
        return state, energy1, energy1 - energy0


class PlainCrowdPredictor(nn.Module):
    """Unstructured next-state network: per-pedestrian residual rolled out for the same horizon, no energy, no sign rule."""

    def __init__(self, hidden: int = 64, horizon_steps: int = HORIZON_STEPS, dt: float = DT, depth: int = 2) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.step = _mlp(7, hidden, 4, depth=depth)

    def energy(self, vel, mask) -> torch.Tensor:
        return 0.5 * ((vel**2).sum(dim=-1) * mask).sum(dim=-1)

    def forward(self, crowd: torch.Tensor, robot_vel: torch.Tensor):
        pos, vel, mask = crowd[..., 0:2], crowd[..., 2:4], crowd[..., 4]
        m = mask[..., None]
        robot_pos = torch.zeros_like(robot_vel)
        energy0 = self.energy(vel, mask)
        for _ in range(self.horizon_steps):
            rel = pos - robot_pos[:, None, :]
            inp = torch.cat([rel, vel, robot_vel[:, None, :].expand_as(rel), torch.ones_like(rel[..., :1])], dim=-1)
            delta = self.step(inp) * m
            pos = pos + self.dt * (vel + delta[..., 0:2])
            vel = vel + self.dt * delta[..., 2:4]
            robot_pos = robot_pos + self.dt * robot_vel
        energy1 = self.energy(vel, mask)
        state = torch.cat([pos, vel], dim=-1) * m
        return state, energy1, energy1 - energy0


def matched_plain_predictor(target: nn.Module, **kwargs) -> PlainCrowdPredictor:
    """The plain predictor (searching width and depth) whose parameter count is closest to `target`."""
    best, best_gap = None, None
    goal = count_parameters(target)
    for depth in (2, 3):
        for hidden in range(4, 512):
            candidate = PlainCrowdPredictor(hidden=hidden, depth=depth, **kwargs)
            gap = abs(count_parameters(candidate) - goal)
            if best_gap is None or gap < best_gap:
                best, best_gap = candidate, gap
    return best


def matched_pair(hidden: int = 64, **kwargs) -> tuple[HamCrowdPredictor, PlainCrowdPredictor]:
    """A Hamiltonian predictor and a plain one with (nearly) the same parameter count, tuned through the dissipation width."""
    base = HamCrowdPredictor(hidden=hidden, **kwargs)
    plain = matched_plain_predictor(base)
    goal = count_parameters(plain)
    best, best_gap = base, abs(count_parameters(base) - goal)
    for diss in range(2, 128):
        cand = HamCrowdPredictor(hidden=hidden, diss_hidden=diss, **kwargs)
        gap = abs(count_parameters(cand) - goal)
        if gap < best_gap:
            best, best_gap = cand, gap
    return best, plain
