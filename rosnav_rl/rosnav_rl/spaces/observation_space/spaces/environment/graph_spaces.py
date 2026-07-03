"""Graph-based pedestrian observation spaces for Social-Dreamer (M1.2).

These spaces expose the padded pedestrian node-set and validity mask produced
by PedestrianGraphNodeGenerator as flat Box spaces consumable by DreamerV3's
MLP encoder.
"""

from typing import ClassVar, Dict, Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import (
    PedestrianGraphNodes,
    PedestrianNodeMask,
)
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianNodeSetSpace(BaseObservationSpace):
    """Fixed-size padded pedestrian graph node-set in robot frame.

    Shape: (max_peds, node_feat_dim + 1) — last column is validity flag (1=real, 0=pad).
    Consumed by the DreamerV3 MLP encoder via the ``peds_nodes`` obs key.
    """

    name: ClassVar[str] = "PedestrianNodeSetSpace"
    requires: ClassVar[Dict[str, Any]] = {
        "peds_nodes": PedestrianGraphNodes,
    }

    def __init__(
        self,
        max_peds: int = 8,
        node_feat_dim: int = 5,
        *args,
        **kwargs,
    ) -> None:
        self._max_peds = max_peds
        self._node_feat_dim = node_feat_dim
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        # Flat shape so MultiEncoder sums elements correctly: sum((N*(F+1),)) = N*(F+1).
        # GAT reshapes back to (N, F+1) from the decoded output — not from this raw tensor.
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._max_peds * (self._node_feat_dim + 1),),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        peds_nodes: PedestrianGraphNodes,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.asarray(peds_nodes, dtype=np.float32).flatten()


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianMaskSpace(BaseObservationSpace):
    """Validity mask for the padded pedestrian node-set (1=real ped, 0=padding).

    Shape: (max_peds,).  Used by GAT to mask out padded nodes during attention.
    """

    name: ClassVar[str] = "PedestrianMaskSpace"
    requires: ClassVar[Dict[str, Any]] = {
        "peds_mask": PedestrianNodeMask,
    }

    def __init__(
        self,
        max_peds: int = 8,
        *args,
        **kwargs,
    ) -> None:
        self._max_peds = max_peds
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._max_peds,),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        peds_mask: PedestrianNodeMask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.asarray(peds_mask, dtype=np.float32)
