import torch
import torch.nn as nn


class KoopmanLinearHead(nn.Module):
    """Linear operator predicting the next latent state via matrices A and B."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.A: nn.Linear = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B: nn.Linear = nn.Linear(action_dim, latent_dim, bias=False)
        nn.init.eye_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight, gain=0.01)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        return self.A(z_t) + self.B(a_t)


class MLPControlHead(nn.Module):
    """Two-layer non-linear latent predictor serving as the baseline control."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, a_t], dim=-1))
