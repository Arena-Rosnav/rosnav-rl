"""AGENT_5-compatible extractor augmented with predicted crowd energy changes.

The public Gym observation deliberately stays identical to AGENT_5. The predictor
reconstructs a bounded pedestrian set from AGENT_5's existing spatial type and
velocity maps, then appends two scalars to the robot-state branch: predicted
crowd-energy change under the previous robot action and under a stop action.
"""

from __future__ import annotations

from typing import Any

import torch

from rosnav_rl.spaces.observation_space.spaces.dynamics import LastActionSpace
from rosnav_rl.spaces.observation_space.spaces.environment import (
    PedestrianTypeSpace,
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace,
)

from .ham_crowd_predictor import matched_pair
from .resnet.resnet import DRL_VO_ROSNAV_SCALAR_EXTRACTOR

N_SCALARS = 2


class DRL_VO_HAM_EXTRACTOR(DRL_VO_ROSNAV_SCALAR_EXTRACTOR):
    """Append own-action and stop-action crowd-energy changes to the last-action features.

    AGENT_5's pedestrian type/velocity map encoders currently emit physical/raw
    values (they do not apply ``BaseObservationSpace.apply_normalization``), while
    ``LastActionSpace`` does normalize its vector. ``action_normalizer`` therefore
    applies only to the last-action vector. Reconstructed pedestrian positions are
    cell centres, so spatial quantisation is at most half a cell diagonal.
    """

    SCALAR_SPACES = ()
    _VALID_NORMALIZERS = {"max_abs", "min_max", "standard", "identity", "none"}

    def __init__(
        self,
        *args: Any,
        variant: str = "ham",
        actions_normalized: bool = True,
        action_normalizer: str = "max_abs",
        map_roi_m: float = 24.0,
        crowd_range_m: float = 12.0,
        max_pedestrians: int = 24,
        predictor_hidden: int = 64,
        **kwargs: Any,
    ):
        if variant not in ("ham", "plain"):
            raise ValueError(f"variant must be 'ham' or 'plain', got {variant!r}")
        if action_normalizer not in self._VALID_NORMALIZERS:
            raise ValueError(f"Unsupported action_normalizer={action_normalizer!r}")
        if map_roi_m <= 0 or crowd_range_m <= 0 or max_pedestrians <= 0:
            raise ValueError("map_roi_m, crowd_range_m and max_pedestrians must be positive")
        self._variant = variant
        self._actions_normalized = actions_normalized
        self._action_normalizer = action_normalizer
        self._map_roi_m = float(map_roi_m)
        self._crowd_range_m = float(crowd_range_m)
        self._max_pedestrians = int(max_pedestrians)
        self._predictor_hidden = predictor_hidden
        super().__init__(*args, **kwargs)

    def _get_input_sizes(self):
        super()._get_input_sizes()
        if self._last_action_size == 0:
            raise ValueError(f"{type(self).__name__} needs {LastActionSpace.name}")
        required = (PedestrianTypeSpace.name, PedestrianVelXSpace.name, PedestrianVelYSpace.name)
        missing = [key for key in required if key not in self._observation_space.spaces]
        if missing:
            raise ValueError(f"{type(self).__name__} requires AGENT_5 feature maps: {missing}")
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

    @staticmethod
    def _as_map(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim == 4 and value.shape[1] == 1:
            value = value[:, 0]
        elif value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(f"{name} must have shape (B,H,W), got {tuple(value.shape)}")
        return value.float()

    def crowd_from_feature_maps(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Rebuild a (B,K,5) physical-unit crowd set from AGENT_5's existing maps.

        Type-map values mark occupancy independently of velocity, so stationary
        pedestrians are retained. Empty cells and pedestrians outside 12 m are
        masked out; if there are more than K pedestrians, nearest ones are kept.
        """
        types = self._as_map(observations[PedestrianTypeSpace.name], PedestrianTypeSpace.name)
        vx_map = self._as_map(observations[PedestrianVelXSpace.name], PedestrianVelXSpace.name)
        vy_map = self._as_map(observations[PedestrianVelYSpace.name], PedestrianVelYSpace.name)
        if types.shape != vx_map.shape or types.shape != vy_map.shape:
            raise ValueError("AGENT_5 pedestrian type/velocity maps must have identical shapes")
        batch, height, width = types.shape
        device, dtype = types.device, types.dtype

        # These three AGENT_5 map encoders do not call the normalization
        # decorator: type-map empty cells are raw -1, and velocity-map values
        # are already m/s. Do not apply LastActionSpace's inverse scaler here.
        occupied = types >= 0.0  # raw physical type codes are 0..N; empty is -1
        vx = vx_map
        vy = vy_map

        cell_x = (torch.arange(width, device=device, dtype=dtype) + 0.5 - width / 2.0) * (self._map_roi_m / width)
        cell_y = (torch.arange(height, device=device, dtype=dtype) + 0.5 - height / 2.0) * (self._map_roi_m / height)
        grid_x = cell_x.view(1, 1, width).expand(batch, height, width)
        grid_y = cell_y.view(1, height, 1).expand(batch, height, width)
        distance = torch.sqrt(grid_x.square() + grid_y.square())
        valid = occupied & (distance <= self._crowd_range_m)
        sortable = torch.where(valid, distance, torch.full_like(distance, float("inf"))).flatten(1)

        k = min(self._max_pedestrians, height * width)
        indices = torch.argsort(sortable, dim=1, stable=True)[:, :k]
        flat_valid = valid.flatten(1).gather(1, indices)
        flat_x = grid_x.flatten(1).gather(1, indices)
        flat_y = grid_y.flatten(1).gather(1, indices)
        flat_vx = vx.flatten(1).gather(1, indices)
        flat_vy = vy.flatten(1).gather(1, indices)
        mask = flat_valid.to(dtype=dtype)
        crowd = torch.stack((flat_x, flat_y, flat_vx, flat_vy, mask), dim=-1)
        crowd = crowd * mask.unsqueeze(-1)
        if k < self._max_pedestrians:
            crowd = torch.nn.functional.pad(crowd, (0, 0, 0, self._max_pedestrians - k))
        return crowd

    def _robot_velocity(self, action: torch.Tensor) -> torch.Tensor:
        """Convert LastActionSpace to physical [vx, vy, yaw_rate] robot command."""
        if self._actions_normalized:
            low = self.action_low.to(device=action.device, dtype=action.dtype)
            high = self.action_high.to(device=action.device, dtype=action.dtype)
            if self._action_normalizer == "max_abs":
                action = (action + 1.0) * (high - low) / 2.0 + low
            elif self._action_normalizer == "min_max":
                action = action * (high - low) + low
            elif self._action_normalizer == "standard":
                action = action * ((high - low) / 4.0) + ((high + low) / 2.0)
        vx = action[:, 0]
        # Differential drive is [linear_x, angular_z]; holonomic is [vx, vy, yaw_rate].
        if action.shape[-1] == 2:
            vy = torch.zeros_like(vx)
            yaw_rate = action[:, 1]
        elif action.shape[-1] == 3:
            vy = action[:, 1]
            yaw_rate = action[:, 2]
        else:
            raise ValueError(f"Expected 2D differential-drive or 3D holonomic action, got {tuple(action.shape)}")
        return torch.stack((vx, vy, yaw_rate), dim=-1)

    def _get_input(self, observations):
        inputs = super()._get_input(observations)
        action = inputs["last_action"]
        crowd = self.crowd_from_feature_maps(observations)
        robot_command = self._robot_velocity(action)
        own = self.predictor(crowd, robot_command)[2]
        stop = self.predictor(crowd, torch.zeros_like(robot_command))[2]
        inputs["last_action"] = torch.cat((action, own.unsqueeze(1), stop.unsqueeze(1)), dim=1)
        return inputs
