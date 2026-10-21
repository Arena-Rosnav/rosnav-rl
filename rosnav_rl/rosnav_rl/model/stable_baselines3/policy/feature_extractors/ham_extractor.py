"""DRL_VO_ROSNAV extractor whose head also reads the predicted crowd energy change under two robot actions."""

from __future__ import annotations

from typing import Any

import torch

from rosnav_rl.spaces.observation_space.spaces.dynamics import LastActionSpace
from rosnav_rl.spaces.observation_space.spaces.environment import CrowdSetSpace

from .ham_crowd_predictor import HamCrowdPredictor, matched_pair
from .resnet.resnet import DRL_VO_ROSNAV_EXTRACTOR

N_SCALARS = 2


class DRL_VO_HAM_EXTRACTOR(DRL_VO_ROSNAV_EXTRACTOR):
    """Appends [energy change under the last action, energy change under stop] to the robot state of the head.

    variant "ham" uses the port-Hamiltonian predictor, "plain" the unstructured control of the same size.
    """

    def __init__(
        self,
        *args: Any,
        variant: str = "ham",
        actions_normalized: bool = True,
        predictor_hidden: int = 64,
        **kwargs: Any,
    ):
        if variant not in ("ham", "plain"):
            raise ValueError(f"variant must be 'ham' or 'plain', got {variant!r}")
        self._variant = variant
        self._actions_normalized = actions_normalized
        self._predictor_hidden = predictor_hidden
        super().__init__(*args, **kwargs)

    def _get_input_sizes(self):
        super()._get_input_sizes()
        if self._last_action_size == 0:
            raise ValueError(f"{type(self).__name__} needs {LastActionSpace.name}")
        if CrowdSetSpace.name not in self._observation_space.spaces:
            raise ValueError(f"{type(self).__name__} needs {CrowdSetSpace.name}")
        action_space = self._observation_space[LastActionSpace.name]
        self._action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self._action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self._last_action_size += N_SCALARS

    def _setup_network(self, *args: Any, **kwargs: Any):
        super()._setup_network(*args, **kwargs)
        ham, plain = matched_pair(self._predictor_hidden)
        self.predictor = ham if self._variant == "ham" else plain
        self.register_buffer("action_low", self._action_low)
        self.register_buffer("action_high", self._action_high)

    def _robot_velocity(self, action: torch.Tensor) -> torch.Tensor:
        """Robot (vx, vy) in its own frame from the last action, in m/s."""
        if self._actions_normalized:
            action = (action + 1.0) * (self.action_high - self.action_low) / 2.0 + self.action_low
        vx = action[:, 0]
        vy = action[:, 1] if action.shape[-1] == 3 else torch.zeros_like(vx)
        return torch.stack([vx, vy], dim=-1)

    def _predicted_energy_change(self, crowd: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return self.predictor(crowd, velocity)[2]

    def _get_input(self, observations) -> dict[str, torch.Tensor | None]:
        inputs = super()._get_input(observations)
        action = inputs["last_action"]
        crowd = observations[CrowdSetSpace.name][..., :5]
        own = self._predicted_energy_change(crowd, self._robot_velocity(action))
        stop = self._predicted_energy_change(crowd, torch.zeros_like(own)[:, None].expand(-1, 2))
        inputs["last_action"] = torch.cat([action, own[:, None], stop[:, None]], dim=1)
        return inputs
