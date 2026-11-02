"""Crowd predictors over the padded crowd set: a port-Hamiltonian one and an unstructured control.

Both map a crowd set (B, K, 5) = [x, y, vx, vy, mask] and a robot command to the crowd state one second ahead
(B, K, 4) and a scalar energy proxy per sample. Commands may be planar [vx, vy] or body-frame [vx, vy, yaw_rate].
All predicted positions are expressed in the robot frame at the start of the horizon.
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


def _split_robot_command(command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body-frame planar velocity (B,2) and yaw rate (B,1).

    Two-component inputs are legacy/inertial planar velocities with zero yaw rate;
    three-component inputs are body-frame [vx, vy, yaw_rate] commands.
    """
    if command.ndim != 2 or command.shape[-1] not in (2, 3):
        raise ValueError(f"robot command must have shape (B,2) or (B,3), got {tuple(command.shape)}")
    if command.shape[-1] == 2:
        return command, torch.zeros_like(command[:, :1])
    return command[:, :2], command[:, 2:3]


def _rotate_body_velocity(linear: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw[:, 0]), torch.sin(yaw[:, 0])
    vx, vy = linear[:, 0], linear[:, 1]
    return torch.stack((c * vx - s * vy, s * vx + c * vy), dim=-1)


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
        # A learned positive gain gives the dissipation channel one scalar of
        # flexibility while preserving the nonnegative sign constraint.
        self.dissipation_gain_logit = nn.Parameter(torch.tensor(0.0))

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
        robot_linear, robot_omega = _split_robot_command(robot_vel)
        robot_pos = torch.zeros_like(robot_linear)
        robot_yaw = torch.zeros_like(robot_omega)
        robot_vel_world = _rotate_body_velocity(robot_linear, robot_yaw)
        energy0 = self.energy(pos, vel, mask, robot_pos, robot_vel_world)
        for _ in range(self.horizon_steps):
            robot_vel_world = _rotate_body_velocity(robot_linear, robot_yaw)
            force = self._force(pos, mask, robot_pos, robot_vel_world)
            damping = torch.nn.functional.softplus(self.dissipation_gain_logit) * (
                self.dissipation(vel.norm(dim=-1, keepdim=True)) ** 2
            )
            # Kick then drift is symplectic Euler for the pedestrian state. Robot
            # translation follows its current heading; yaw advances by omega*dt.
            vel = (vel + self.dt * (force - damping * vel)) * m
            pos = pos + self.dt * vel
            robot_pos = robot_pos + self.dt * robot_vel_world
            robot_yaw = robot_yaw + self.dt * robot_omega
        robot_vel_world = _rotate_body_velocity(robot_linear, robot_yaw)
        energy1 = self.energy(pos, vel, mask, robot_pos, robot_vel_world)
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
        robot_linear, robot_omega = _split_robot_command(robot_vel)
        robot_pos = torch.zeros_like(robot_linear)
        robot_yaw = torch.zeros_like(robot_omega)
        energy0 = self.energy(vel, mask)
        for _ in range(self.horizon_steps):
            robot_vel_world = _rotate_body_velocity(robot_linear, robot_yaw)
            rel = pos - robot_pos[:, None, :]
            inp = torch.cat(
                [rel, vel, robot_vel_world[:, None, :].expand_as(rel), torch.ones_like(rel[..., :1])], dim=-1
            )
            delta = self.step(inp) * m
            pos = pos + self.dt * (vel + delta[..., 0:2])
            vel = vel + self.dt * delta[..., 2:4]
            robot_pos = robot_pos + self.dt * robot_vel_world
            robot_yaw = robot_yaw + self.dt * robot_omega
        energy1 = self.energy(vel, mask)
        state = torch.cat([pos, vel], dim=-1) * m
        return state, energy1, energy1 - energy0


def _ham_parameter_count(hidden: int, diss_hidden: int) -> int:
    # potential MLP: 4->h->h->1 (last bias disabled); dissipation MLP: 1->d->1
    # plus the learned positive dissipation gain.
    return hidden * hidden + 7 * hidden + 3 * diss_hidden + 2


def _plain_parameter_count(hidden: int, depth: int) -> int:
    # MLP: 7->h, then (depth-1) h->h layers, then h->4.
    return (depth - 1) * hidden * hidden + (depth + 11) * hidden + 4


def matched_plain_predictor(target: nn.Module, **kwargs) -> PlainCrowdPredictor:
    """Construct the nearest-width/depth plain control without instantiating every candidate."""
    goal = count_parameters(target)
    _, best_hidden, best_depth = min(
        (
            abs(_plain_parameter_count(hidden, depth) - goal),
            hidden,
            depth,
        )
        for depth in (2, 3)
        for hidden in range(4, 512)
    )
    return PlainCrowdPredictor(hidden=best_hidden, depth=best_depth, **kwargs)


def matched_pair(hidden: int = 64, **kwargs) -> tuple[HamCrowdPredictor, PlainCrowdPredictor]:
    """Return exactly parameter-matched predictors, with minimal construction overhead."""
    base = HamCrowdPredictor(hidden=hidden, **kwargs)
    plain = matched_plain_predictor(base)
    target = count_parameters(plain)
    best_diss = min(range(2, 128), key=lambda d: abs(_ham_parameter_count(hidden, d) - target))
    ham = HamCrowdPredictor(hidden=hidden, diss_hidden=best_diss, **kwargs)
    return ham, plain
