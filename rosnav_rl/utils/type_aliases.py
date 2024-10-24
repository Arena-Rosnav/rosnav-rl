from typing import TYPE_CHECKING, Any, Dict, TypeVar, Union

if TYPE_CHECKING:
    from rosnav_rl.spaces import BaseFeatureMapSpace, BaseObservationSpace

ObservationDict = Dict[str, Any]
ObservationSpaceUnit = Union["BaseObservationSpace", "BaseFeatureMapSpace"]

ObservationCollector = TypeVar(
    "ObservationCollector"
)  # bound=ObservationCollectorUnit)
ObservationGenerator = TypeVar(
    "ObservationGenerator"
)  # bound=ObservationGeneratorUnit)
