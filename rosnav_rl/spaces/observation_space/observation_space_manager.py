from typing import Any, Dict, List, Type, Union

from gymnasium import spaces

from rosnav_rl.spaces.observation_space import EncodedObservationDict
from rosnav_rl.utils.space import extract_init_arguments
from rosnav_rl.utils.type_aliases import ObservationDict

from .spaces.base_observation_space import BaseObservationSpace


class ObservationSpaceManager:
    """
    A class that manages the observation spaces for a given namespace.

    Args:
        space_list (List[Type[BaseObservationSpace]]): A list of observation space classes.
        space_kwargs (Dict[str, Any]): Additional keyword arguments to be passed to the observation space classes.

    Attributes:
        space_list: The list of observation space classes.
        observation_space: The combined observation space.
    """

    def __init__(
        self, space_list: List[Type[BaseObservationSpace]], space_kwargs: Dict[str, Any]
    ) -> None:
        self._space_cls_list = space_list
        self._space_kwargs = space_kwargs
        self._space_containers: Dict[str, BaseObservationSpace] = {}

        self._initialize_spaces()
        self._observation_space = self._create_combined_observation_space()

    def _initialize_spaces(self) -> None:
        """Initialize the individual observation spaces."""
        for space_cls in self._space_cls_list:
            try:
                self._space_containers[space_cls.name] = space_cls(**self._space_kwargs)
            except TypeError as e:
                raise TypeError(
                    f"Error initializing the observation space '{space_cls.name}'. "
                    f"Ensure all required arguments are passed. Error: {e}"
                )

    def _create_combined_observation_space(self) -> spaces.Dict:
        """Create a combined observation space from individual spaces."""
        return spaces.Dict(
            {name: space.space for name, space in self._space_containers.items()}
        )

    def __getitem__(
        self, space: Union[str, BaseObservationSpace, Type[BaseObservationSpace]]
    ) -> BaseObservationSpace:
        """
        Retrieve the observation space with the given name or instance.

        Parameters:
            space (Union[str, BaseObservationSpace]): The name or instance of the observation space.

        Returns:
            BaseObservationSpace: The requested observation space.
        """
        space_name = self._get_space_name(space)
        return self._space_containers[space_name.upper()]

    def __contains__(self, space: Union[str, BaseObservationSpace]) -> bool:
        """
        Check if the observation space with the given name or instance exists.

        Parameters:
            space (Union[str, BaseObservationSpace]): The name or instance of the observation space.

        Returns:
            bool: Whether the observation space exists.
        """
        space_name = self._get_space_name(space)
        return space_name.upper() in self._space_containers

    def _get_space_name(self, space: Union[str, BaseObservationSpace]) -> str:
        """Extract the name from the provided space."""
        return space.name if isinstance(space, BaseObservationSpace) else str(space)

    def __iter__(self):
        """Iterate over the observation space containers."""
        return iter(self._space_containers.values())

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
            EncodedObservationDict: The encoded observation.
        """
        return {
            name: space.encode_observation(observation, **kwargs)
            for name, space in self._space_containers.items()
        }

    @property
    def space_list(self) -> List[Type[BaseObservationSpace]]:
        """Return the list of observation spaces."""
        return self._space_cls_list

    @property
    def observation_space(self) -> spaces.Dict:
        """Return the combined observation space."""
        return self._observation_space

    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration details for the manager."""
        return {
            "space": self.observation_space,
            "params": {
                name: space.config for name, space in self._space_containers.items()
            },
        }

    @property
    def space_keywords(self) -> Dict[str, Dict[str, str]]:
        """Return initialization arguments for each observation space."""
        return extract_init_arguments(self._space_cls_list)
