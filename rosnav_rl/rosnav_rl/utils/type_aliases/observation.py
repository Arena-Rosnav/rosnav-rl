from __future__ import annotations

from typing import Any, Dict, TypeVar, Union, TYPE_CHECKING, Type, List

from rosnav_rl.observations import (
    ObservationCollectorUnit,
    ObservationGeneratorUnit,
)

if TYPE_CHECKING:
    from rosnav_rl.spaces import BaseFeatureMapSpace, BaseObservationSpace


ObservationName = str
ObservationDict = Dict[ObservationName, Any]
ObservationSpaceUnit = Union["BaseObservationSpace", "BaseFeatureMapSpace"]

ObservationCollector = TypeVar("ObservationCollector", bound=ObservationCollectorUnit)
ObservationGenerator = TypeVar("ObservationGenerator", bound=ObservationGeneratorUnit)

ObservationSpaceList = TypeVar(
    "ObservationSpaceList", bound=List[Type[ObservationSpaceUnit]]
)  # List containing classes of observation spaces to define a model's observation space
ObservationSpaceKwargs = TypeVar(
    "ObservationSpaceKwargs", bound=Dict[str, Any]
)  # Keyword arguments for observation spaces defining a model's observation space attributes
