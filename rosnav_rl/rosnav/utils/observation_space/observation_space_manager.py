from typing import Any, Dict, List, Type, Union

from gymnasium import spaces
from rl_utils.utils.observation_collector import ObservationDict
from rosnav.utils.observation_space import EncodedObservationDict

from .spaces.base_observation_space import BaseObservationSpace


class ObservationSpaceManager:
    """
    A class that manages the observation spaces for a given namespace.

    Args:
        ns (str): The namespace for the observation spaces.
        space_list (List[Type[BaseObservationSpace]]): A list of observation space classes.
        space_kwargs (Dict[str, Any]): Additional keyword arguments to be passed to the observation space classes.

    Attributes:
        space_list: The list of observation space classes.
        observation_space: The combined observation space.

    Methods:
        __getitem__: Get the observation space for a given space name or observation space class.
        encode_observation: Encode the observation using the observation spaces.
        get_space_container: Get the observation space container for a given space name or observation space class.
    """

    def __init__(
        self,
        ns: str,
        space_list: List[Type[BaseObservationSpace]],
        space_kwargs: Dict[str, Any],
    ) -> None:
        self._ns = ns
        self._space_cls_list = space_list
        self._space_kwargs = space_kwargs

        self._space_containers: Dict[str, BaseObservationSpace] = (
            self._instantiate_spaces()
        )
        self._observation_space = spaces.Dict(
            {name: space.space for name, space in self._space_containers.items()}
        )

    @property
    def space_list(self):
        return self._space_cls_list

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def _instantiate_spaces(self):
        return {
            space_cls.name: space_cls(ns=self._ns, **self._space_kwargs)
            for space_cls in self._space_cls_list
        }

    def __getitem__(
        self, space: Union[str, BaseObservationSpace]
    ) -> BaseObservationSpace:
        """
        Retrieve the observation space with the given name or instance.

        Parameters:
            space (Union[str, BaseObservationSpace]): The name or instance of the observation space.

        Returns:
            spaces.Box: The observation space.
        """
        space_name = space.name if issubclass(space, BaseObservationSpace) else space
        return self._space_containers[space_name.upper()]

    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        """
        Encode the observation using the observation spaces.

        Args:
            observation (ObservationDict): The observation to be encoded.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            EncodedObservation: The encoded observation.
        """
        return {
            name: space.encode_observation(observation, **kwargs)
            for name, space in self._space_containers.items()
        }
