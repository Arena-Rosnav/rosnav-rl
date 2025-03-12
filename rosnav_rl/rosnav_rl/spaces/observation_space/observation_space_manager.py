from typing import Any, Dict, List, Type, Union

from gymnasium import spaces

from rosnav_rl.utils.space import extract_init_arguments
from rosnav_rl.utils.type_aliases import (
    EncodedObservationDict,
    ObservationDict,
    ObservationSpaceList,
)

from .spaces.base_observation_space import BaseObservationSpace


class ObservationSpaceManager:
    """ObservationSpaceManager manages multiple observation spaces for reinforcement learning.

    This class creates, manages, and combines different observation spaces into a unified
    observation space for an RL agent. It provides functionality to access individual spaces,
    encode observations, and retrieve configuration details.

    Attributes:
        _space_cls_list (ObservationSpaceList): List of observation space classes to initialize.
        _space_kwargs (Dict[str, Any]): Arguments to initialize observation spaces.
        _space_containers (Dict[str, BaseObservationSpace]): Dictionary of initialized observation spaces.
        _observation_space (spaces.Dict): Combined gym observation space from all individual spaces.
        
    Example:
        ```
        space_list = [LaserScanSpace, GoalSpace]
        space_kwargs = {"robot_state_size": 4, "laser_scan_size": 720}
        
        obs_manager = ObservationSpaceManager(space_list, space_kwargs)
        encoded_obs = obs_manager.encode_observation(observation_dict)
        ```
    """
    def __init__(
        self, space_list: ObservationSpaceList, space_kwargs: Dict[str, Any]
    ) -> None:
        """
        Initialize the ObservationSpaceManager with a list of observation spaces.

        This class manages multiple observation spaces and combines them into a single
        observation space.

        Args:
            space_list (ObservationSpaceList): List of observation space classes.
            space_kwargs (Dict[str, Any]): Dictionary of keyword arguments for initializing
                the observation spaces.

        Attributes:
            _space_cls_list (ObservationSpaceList): List of observation space classes.
            _space_kwargs (Dict[str, Any]): Dictionary of keyword arguments.
            _space_containers (Dict[str, BaseObservationSpace]): Dictionary mapping space names
                to their respective observation space instances.
            _observation_space: The combined observation space created from all individual spaces.
        """
        self._space_cls_list = space_list
        self._space_kwargs = space_kwargs
        self._space_containers: Dict[str, BaseObservationSpace] = {}

        self._initialize_spaces()
        self._observation_space = self._create_combined_observation_space()

    def _initialize_spaces(self) -> None:
        """Initialize all observation space containers based on the provided space classes.
        
        This method instantiates each observation space class in the _space_cls_list
        and stores the instances in the _space_containers dictionary, using the
        class name as the key. It passes any keyword arguments stored in _space_kwargs
        to the constructor of each space class.
        
        Raises:
            TypeError: If any space class constructor is missing required arguments
                      or receives incompatible arguments.
        """
        
        for space_cls in self._space_cls_list:
            try:
                self._space_containers[space_cls.name] = space_cls(**self._space_kwargs)
            except TypeError as e:
                raise TypeError(
                    f"Error initializing the observation space '{space_cls.name}'. "
                    f"Ensure all required arguments are passed. Error: {e}"
                )

    def _create_combined_observation_space(self) -> spaces.Dict:
        """Creates a combined observation space by merging individual observation spaces.
        
        This method combines all registered observation spaces from the _space_containers
        dictionary into a single Dict space, where each key corresponds to the name of
        an observation space container, and the value is the actual space object.
        
        Returns:
            spaces.Dict: A dictionary space containing all registered observation spaces.
        """
        
        return spaces.Dict(
            {name: space.space for name, space in self._space_containers.items()}
        )

    def __getitem__(
        self, space: Union[str, Type[BaseObservationSpace]]
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
        """Encodes the observation for all spaces in the manager.
        
        This method applies the encoding function of each space container to the 
        observation dictionary and returns a dictionary with the encoded observations.
        
        Args:
            observation (ObservationDict): The original observation dictionary to be encoded.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments passed to each space container's encode_observation method.
            
        Returns:
            EncodedObservationDict: A dictionary where each key is a space name and each value is the
                                   encoded observation for that space.
        """
        return {
            name: space.encode_observation(observation, **kwargs)
            for name, space in self._space_containers.items()
        }

    @property
    def space_list(self) -> ObservationSpaceList:
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
