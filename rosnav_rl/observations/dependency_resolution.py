from typing import TYPE_CHECKING, List, Set, Type, Union

from ..observations import (
    ObservationCollectorUnit,
    ObservationGeneratorUnit,
)

if TYPE_CHECKING:
    from ..reward.reward_units.base_reward_units import RewardUnit
    from ..spaces import BaseObservationSpace


def get_required_observation_units(
    list_of_units: Union[List["BaseObservationSpace"], List["RewardUnit"]]
) -> list:
    """Extracts and returns a unique list of required observation units from a list of units.
    
    This function iterates through each unit in the provided list, collecting all required
    observation units and their dependencies recursively.
    
    Args:
        list_of_units (Union[List["BaseObservationSpace"], List["RewardUnit"]]): A list of 
            units that have required_observation_units attributes. These can be either
            observation space units or reward units.
    
    Returns:
        list: A unique list of required observation unit classes without duplicates.
    """
    observations = []
    for unit in list_of_units:
        for observation_cls in unit.required_observation_units:
            observations.extend(retrieve_unique_unit(observation_cls, True))
    return list(set(observations))


def retrieve_unique_unit(
    unit: Union[ObservationCollectorUnit, ObservationGeneratorUnit],
    include_generators: bool = False,
) -> Set[Type[ObservationCollectorUnit]]:
    """
    Recursively retrieves a set of unique ObservationCollectorUnit types required by the given unit.
    
    This function traverses through the dependency tree of an observation unit to identify all collector units
    it depends on. It can optionally include generator units in the result as well.
    
    Args:
        unit: The observation unit to analyze. Can be either a collector or generator unit.
        include_generators: If True, ObservationGeneratorUnit types will also be included in the result.
                           Defaults to False.
    
    Returns:
        A set of unique ObservationCollectorUnit types (and optionally ObservationGeneratorUnit types)
        that the input unit depends on.
    """
    is_collector = issubclass(unit, ObservationCollectorUnit)
    collectors = [unit] if include_generators and not is_collector else []

    if is_collector:
        collectors.append(unit)
        return collectors

    for observations in unit.requires:
        if observations in collectors:
            continue

        if include_generators and issubclass(observations, ObservationGeneratorUnit):
            collectors.append(observations)
        collectors.extend(retrieve_unique_unit(observations, include_generators))
    return set(collectors)


def explore_dependency_hierarchy(
    list_of_units: List[Union[ObservationCollectorUnit, ObservationGeneratorUnit]]
):
    """
    Traverses the hierarchy of observation units and returns a dictionary
    that represents the depth of each unit in the hierarchy.

    Args:
        list_of_units (List[ObservationGeneric]): A list of observation units.

    Returns:
        dict: A dictionary where the keys are observation units and the values
        are their depths in the hierarchy.
    """

    def traverse_hierarchy(
        unit: Union[ObservationCollectorUnit, ObservationGeneratorUnit],
        hierarchy_dict: dict,
        recursion_depth: int = 0,
    ) -> int:
        if unit not in hierarchy_dict:
            hierarchy_dict[unit] = 0

        if issubclass(unit, ObservationCollectorUnit):
            pass
        else:
            for observation in unit.requires:
                depth = traverse_hierarchy(
                    observation, hierarchy_dict, recursion_depth + 1
                )
                if depth > hierarchy_dict.get(observation, 0):
                    hierarchy_dict[observation] = depth
        return recursion_depth

    hierarchy_dict = {}

    for unit in list_of_units:
        traverse_hierarchy(unit, hierarchy_dict)

    hierarchy_dict = dict(
        sorted(hierarchy_dict.items(), key=lambda item: item[1], reverse=True)
    )

    return hierarchy_dict


# if __name__ == "__main__":
#     dist = retrieve_unique_unit(DistAngleToGoal, True)
#     social_state = retrieve_unique_unit(PedestrianRelativeLocationGenerator, True)
#     # set out of dist and social_state
#     set_out = set(dist) - set(social_state)
#     print(set_out)

#     c = explore_dependency_hierarchy(
#         [
#             PedestrianRelativeVelXGenerator,
#             PedestrianRelativeLocationGenerator,
#             DistAngleToGoal,
#         ]
#     )
