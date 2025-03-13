from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import rospy
from typing_extensions import Self

from ..observations import (
    BaseUnit,
    ObservationCollectorUnit,
    ObservationGeneratorUnit,
)
from ..observations.collectors.base_collector import (
    SimulationNotCompatibleError,
)
from ..states import SimulationStateContainer
from ..utils.rostopic import Namespace, Topic
from .dependency_resolution import explore_dependency_hierarchy
from .generic_observation import GenericObservation

if TYPE_CHECKING:
    from rosnav_rl.utils.type_aliases import ObservationDict


def map_crowdsim_topics(topic: Union[str, Topic], manager: ObservationManager) -> str:
    _topic = Topic(topic)
    if manager._is_single_env:
        return _topic.name
    return str(_topic)


class ObservationHealth:
    """Tracks the health status of an observation source."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_update_time = rospy.Time(0)
        self.update_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.latest_error = None
        
    def record_update(self):
        """Record a successful observation update."""
        self.last_update_time = rospy.Time.now()
        self.update_count += 1
        self.consecutive_errors = 0
        
    def record_error(self, error: Exception):
        """Record an observation error."""
        self.error_count += 1
        self.consecutive_errors += 1
        self.latest_error = error
        
    @property
    def is_healthy(self) -> bool:
        """Check if the observation source is healthy."""
        # Health criteria: received at least one update and no consecutive errors
        return self.update_count > 0 and self.consecutive_errors < 3
        
    @property
    def time_since_update(self) -> rospy.Duration:
        """Get time elapsed since the last update."""
        return rospy.Time.now() - self.last_update_time


class ObservationManager:
    """Manages the collection and generation of observations for reinforcement learning agents in ROS.

    This class is responsible for:
    - Initializing observation collector and generator units
    - Setting up ROS subscribers for collecting observations from topics
    - Managing the state of observations (stale vs. up-to-date)
    - Providing methods to retrieve and process observations

    The manager works with a hierarchy of observation units (collectors and generators) and handles
    namespace-specific topics and subscriptions. It can be configured to wait for observations
    to be updated before providing them to the agent.

    Attributes:
        _ns (Namespace): The namespace object for ROS topics.
        _obs_structur (List[BaseUnit]): Structure of observation units.
        _simulation_state_container (SimulationStateContainer): Container for simulation state.
        _collectors (Dict[str, ObservationCollectorUnit]): Dictionary of collector units.
        _generators (Dict[str, ObservationGeneratorUnit]): Dictionary of generator units.
        _collectable_observations (Dict[str, GenericObservation]): Observations collected from topics.
        _subscribers (Dict[str, rospy.Subscriber]): ROS subscribers for observations.
        _default_topic_mappings (Dict[Union[str, Topic], Callable]): Default mappings for topics.
    """

    _ns: Namespace
    _obs_structur: List[BaseUnit]
    _simulation_state_container: SimulationStateContainer
    _collectors: Dict[str, ObservationCollectorUnit]
    _generators: Dict[str, ObservationGeneratorUnit]
    _collectable_observations: Dict[str, "GenericObservation"]
    _subscribers: Dict[str, rospy.Subscriber]
    _health_monitors: Dict[str, ObservationHealth]

    _default_topic_mappings: Dict[Union[str, Topic], Callable] = {
        ".*crowdsim.*": map_crowdsim_topics
    }

    def __init__(
        self,
        ns: Namespace,
        obs_structur: List[BaseUnit],
        simulation_state_container: SimulationStateContainer,
        topic_mappings: Dict[str, Callable[[Union[str, Topic], Self], str]] = None,
        is_single_env: Optional[bool] = False,
        obs_unit_kwargs: Optional[dict] = None,
        wait_for_obs: Optional[bool] = True,
        obs_timeout: Optional[float] = 10.0,
    ) -> None:
        """Initialize the ObservationManager.
        
        The ObservationManager handles the collection and management of observations
        from various sources within a ROS environment.
        
        Args:
            ns (Namespace): The ROS namespace.
            obs_structur (List[BaseUnit]): List of observation units defining the structure of observations.
            simulation_state_container (SimulationStateContainer): Container holding the simulation state.
            topic_mappings (Dict[str, Callable[[Union[str, Topic], Self], str]], optional): 
                Dictionary mapping observation types to functions that return topic names. Defaults to None.
            is_single_env (bool, optional): Whether this is a single environment setup. 
                Will be set to True if "sim" is in namespace. Defaults to False.
            obs_unit_kwargs (dict, optional): Additional keyword arguments for observation units. Defaults to None.
            wait_for_obs (bool, optional): Whether to wait for observations before proceeding. Defaults to True.
            obs_timeout (float, optional): Timeout in seconds when waiting for observations. Defaults to 10.0.
            
        Returns:
            None
        """
        self._ns = ns
        self._simulation_state_container = simulation_state_container
        self._obs_structur = list(explore_dependency_hierarchy(obs_structur).keys())

        self._topic_mappings = topic_mappings or {}
        self._topic_mappings.update(self._default_topic_mappings)

        obs_unit_kwargs = obs_unit_kwargs or {}
        self._collectable_observations = {}
        self._subscribers = {}
        self._health_monitors = {}

        self._wait_for_obs = wait_for_obs
        self._obs_timeout = obs_timeout
        self._is_single_env = is_single_env or "sim" in ns

        obs_unit_kwargs.update(
            {"ns": self._ns, "simulation_state_container": simulation_state_container}
        )

        self._inititialize_units(obs_unit_kwargs=obs_unit_kwargs)
        self._init_units()

    def _inititialize_units(self, obs_unit_kwargs: dict) -> None:
        """
        Initialize all observation units.
        """
        _collector_cls = [
            unit
            for unit in self._obs_structur
            if issubclass(unit, ObservationCollectorUnit)
        ]
        _generator_cls = [
            unit
            for unit in self._obs_structur
            if issubclass(unit, ObservationGeneratorUnit)
        ]

        self._collectors = {}
        for collector_class in _collector_cls:
            try:
                self._collectors[collector_class.name] = collector_class(
                    **obs_unit_kwargs
                )
            except SimulationNotCompatibleError as e:
                rospy.logwarn(e)
                pass

        self._generators = {
            generator_class.name: generator_class(**obs_unit_kwargs)
            for generator_class in _generator_cls
        }

    def _init_units(self) -> None:
        """
        Initializes the observation units by creating instances of `GenericObservation` for each collector
        and setting up ROS subscribers for each observation.

        This method performs the following steps:
        1. Imports the `GenericObservation` class from `rosnav_rl.observations.generic_observation`.
        2. Iterates over the `_collectors` dictionary.
        3. For each collector, creates an instance of `GenericObservation` with the initial message and preprocessing function.
        4. Stores the `GenericObservation` instance in the `_collectable_observations` dictionary.
        5. Sets up a ROS subscriber for each collector's topic, using the appropriate topic name and message data class.

        Note:
            If the collector's topic contains "crowdsim_agents" and the environment is single, the topic name is used directly.
            Otherwise, the topic name is generated using the `get_topic` function.

        Args:
            None

        Returns:
            None
        """
        import rosnav_rl.observations.generic_observation as gen_obs

        for collector in self._collectors.values():
            try:
                observation_container = gen_obs.GenericObservation(
                    initial_msg=collector.msg_data_class(),
                    process_fnc=collector.safe_preprocess,
                )

                self._collectable_observations[collector.name] = observation_container
                self._health_monitors[collector.name] = ObservationHealth(collector.name)

                # Get the mapping function for the topic
                _map_functions = [
                    mapping
                    for pattern, mapping in self._topic_mappings.items()
                    if re.match(pattern, str(collector.topic))
                ]

                _topic = (
                    self._ns(
                        collector.topic
                    )  # in this case, the namespace contains /*simulation ns*/*agent ns*/
                    if collector.is_topic_agent_specific
                    else self._ns.simulation_ns(
                        collector.topic
                    )  # in this case, the namespace contains /*simulation ns*/
                )

                for func in _map_functions:
                    _topic = func(_topic, self)

                # Create callback that updates health monitoring
                def callback_with_health_monitoring(msg, collector_name=collector.name):
                    try:
                        self._collectable_observations[collector_name].update(msg)
                        self._health_monitors[collector_name].record_update()
                    except Exception as e:
                        self._health_monitors[collector_name].record_error(e)
                        rospy.logerr(f"Error updating observation '{collector_name}': {e}")
                
                self._subscribers[collector.name] = rospy.Subscriber(
                    str(_topic),
                    collector.msg_data_class,
                    callback_with_health_monitoring,
                )
            except Exception as e:
                rospy.logerr(f"Failed to initialize collector '{collector.name}': {e}")

    def _invalidate_observations(self) -> None:
        """
        Invalidates all collectable observations.

        This method iterates through all the collectable observations and calls
        the `invalidate` method on each collector to mark them as invalid.
        """
        for collector in self._collectable_observations.values():
            collector.invalidate()

    def _wait_for_observation(self, collector_name: str) -> bool:
        """
        Waits for a message from the specified observation collector unit.

        This method waits for a message from the ROS topic associated with the given
        collector name. If a message is not received within the specified timeout,
        a warning is logged indicating that the observation may be stale.

        Args:
            collector_name (str): The name of the observation collector to wait for.

        Returns:
            bool: True if the observation was received, False if timed out.
        """
        try:
            rospy.wait_for_message(
                self._subscribers[collector_name].name,
                self._collectors[collector_name].msg_data_class,
                timeout=self._obs_timeout,
            )
            return True
        except rospy.ROSException:
            rospy.logwarn(
                f"Waiting for observation '{collector_name}' timed out. The observation may be stale."
            )
            return False
        except Exception as e:
            rospy.logerr(f"Error waiting for observation '{collector_name}': {e}")
            return False

    def _get_collectable_observations(
        self, obs_dict: "ObservationDict"
    ) -> "ObservationDict":
        """
        Retrieve and return all collectable observations.

        This method iterates through the collectable observations and checks if any
        of them are stale. If an observation is stale and up-to-date data is required,
        it waits for the observation to be updated. All observations are then added
        to the provided observation dictionary.

        Args:
            obs_dict (ObservationDict): The dictionary to populate with observations.

        Returns:
            ObservationDict: The updated dictionary containing all collectable observations.
        """
        # Retrieve all observations from the collectors
        for name, observation in self._collectable_observations.items():
            if observation.stale and self._collectors[name].up_to_date_required:
                rospy.logdebug_throttle(1, f"Observation '{name}' IS STALE.")
                if self._wait_for_obs:
                    self._wait_for_observation(name)
            obs_dict[name] = observation.value

        self._invalidate_observations()
        return obs_dict

    def _get_generatable_observations(
        self,
        obs_dict: "ObservationDict",
        simulation_state_container: SimulationStateContainer,
    ) -> "ObservationDict":
        """
        Generates observations using the available generators and updates the observation dictionary.

        Args:
            obs_dict (ObservationDict): The dictionary containing current observations.
            simulation_state_container (SimulationStateContainer): The container holding the current state of the simulation.

        Returns:
            ObservationDict: The updated observation dictionary with generated observations.

        Raises:
            KeyError: If a generator fails to generate an observation, a warning is logged and the observation is set to None.
        """
        # Generate observations from the generators
        for generator in self._generators.values():
            try:
                obs_dict[generator.name] = generator.safe_generate(
                    obs_dict=obs_dict,
                    simulation_state_container=simulation_state_container,
                )
            except KeyError as e:
                rospy.logwarn_once(
                    f"{e} \n Could not generate observation for '{generator.name}'."
                )
                obs_dict[generator.name] = None

        return obs_dict

    def get_observations(
        self,
        *args,
        **extra_observations: "ObservationDict",
    ) -> "ObservationDict":
        """
        Collects and returns a dictionary of observations.

        This method gathers observations from various sources, including
        collectable and generatable observations, and combines them with
        any additional observations provided.

        Args:
            simulation_state_container (Optional[SimulationStateContainer]):
                An optional container holding the state of the simulation.
            *args: Additional positional arguments.
            **extra_observations (ObservationDict):
                Additional observations to be included in the final dictionary.

        Returns:
            ObservationDict: A dictionary containing all collected observations.
        """
        obs_dict = {}

        self._get_collectable_observations(obs_dict)
        self._get_generatable_observations(
            obs_dict=obs_dict,
            simulation_state_container=self._simulation_state_container,
        )

        for key, val in extra_observations.items():
            obs_dict[key] = val

        return obs_dict

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the health status of all observation collectors.
        
        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping collector names to their health status.
        """
        result = {}
        for name, monitor in self._health_monitors.items():
            result[name] = {
                "is_healthy": monitor.is_healthy,
                "update_count": monitor.update_count,
                "error_count": monitor.error_count,
                "consecutive_errors": monitor.consecutive_errors,
                "time_since_update": monitor.time_since_update.to_sec(),
                "latest_error": str(monitor.latest_error) if monitor.latest_error else None,
            }
        return result
