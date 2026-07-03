from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

if TYPE_CHECKING:
    from rosnav_rl.observations.data_sources import (
        Collector,
        Generator,
    )
    from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
        BaseObservationSpace,
    )
    from rosnav_rl.spaces.observation_space.spaces.environment.feature_map_spaces import (
        BaseFeatureMapSpace,
    )


ObservationName = str
ObservationDict = Dict[ObservationName, Any]
ObservationSpaceUnit = Union["BaseObservationSpace", "BaseFeatureMapSpace"]

ObservationCollector = TypeVar("ObservationCollector", bound="Collector")
ObservationGenerator = TypeVar("ObservationGenerator", bound="Generator")

ObservationSpaceList = TypeVar(
    "ObservationSpaceList", bound=List[Type[ObservationSpaceUnit]]
)  # List containing classes of observation spaces to define a model's observation space
ObservationSpaceKwargs = TypeVar(
    "ObservationSpaceKwargs", bound=Dict[str, Any]
)  # Keyword arguments for observation spaces defining a model's observation space attributes
