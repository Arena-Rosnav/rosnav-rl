from abc import ABC, abstractmethod
from typing import Any, List

from ..reward_function import RewardFunction
from rl_utils.utils.observation_collector import TypeObservationGeneric

from rl_utils.state_container import StateContainer


class RewardUnit(ABC):
    """
    RewardUnit is an abstract base class for defining reward units in a reinforcement learning context.
    It provides methods for initializing the unit, adding rewards and information to the episode,
    checking parameters, and resetting the unit state. Derived classes must implement the __call__ method.

    Attributes:
        required_observation_units (List[TypeObservationGeneric]): List of required observations for the reward unit.


    Methods:
        robot_radius: Returns the robot's radius from the reward function.
        on_safe_dist_violation: Returns whether the unit is applied on safe distance violation.
        add_reward(value: float): Adds the given value to the episode's reward.
        add_info(info: dict): Adds the given information to the episode's info dict.
        check_parameters(*args, **kwargs): Checks the parsed unit parameters and sends a warning if parameters were chosen inappropriately.
        reset(): Resets the unit state after each episode.
        __call__(*args: Any, **kwargs: Any) -> Any: Abstract method to alter the reward and possibly the info dict. Must be overridden in derived classes.
    """

    required_observation_units: List[TypeObservationGeneric]

    def __init__(
        self,
        reward_function: RewardFunction,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs
    ) -> None:
        """Initializes the RewardUnit.

        Args:
            reward_function (RewardFunction): The RewardFunction instance holding this unit.
            _on_safe_dist_violation (bool, optional): Whether the unit is applied on safe distance violation. Defaults to True.
        """
        self._reward_function = reward_function
        self._on_safe_dist_violation = _on_safe_dist_violation

    @property
    def on_safe_dist_violation(self):
        return self._on_safe_dist_violation

    def add_reward(self, value: float):
        """Adds the given value to the episode's reward.

        Args:
            value (float): _description_
        """
        self._reward_function.add_reward(value=value, called_by=self.__class__.__name__)

    def add_info(self, info: dict):
        """Adds the given information to the episode's info dict.

        Args:
            info (dict): _description_
        """
        self._reward_function.add_info(info=info)

    def check_parameters(self, *args, **kwargs):
        """Method to check the parsed unit parameters. Send warning if params were chosen inappropriately."""
        pass

    def reset(self):
        """Method to reset the unit state after each episode."""
        pass

    @abstractmethod
    def __call__(
        self, state_container: StateContainer, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Placeholder method for calling the reward unit. It should alter the reward and possibly the info dict.
        This method should be overridden in the derived classes.
        """
        raise NotImplementedError()


# class GlobalplanRewardUnit(RewardUnit, ABC):
#     """
#     Base Globalplan Reward Unit

#     This class represents a globalplan reward unit and is a subclass of RewardUnit.
#     It provides methods for calculating the distance to a global plan and resetting the unit.

#     Attributes:
#         _kdtree: cKDTree

#     Internal State Information:
#         curr_dist_to_path (float): Current distance to path.
#     """

#     _kdtree: cKDTree

#     def __init__(
#         self,
#         reward_function: "RewardFunction",
#         _on_safe_dist_violation: bool = True,
#         *args,
#         **kwargs
#     ) -> None:
#         super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
#         self._kdtree = None
#         self._reward_function.add_internal_state_info("curr_dist_to_path", None)

#     @property
#     def curr_dist_to_path(self) -> float:
#         return self._reward_function.get_internal_state_info("curr_dist_to_path")

#     @curr_dist_to_path.setter
#     def curr_dist_to_path(self, value: float) -> None:
#         self._reward_function.add_internal_state_info("curr_dist_to_path", value)

#     def __call__(
#         self, global_plan: np.ndarray, robot_pose, *args: Any, **kwargs: Any
#     ) -> Any:
#         if (
#             not self.curr_dist_to_path
#             and isinstance(global_plan, np.ndarray)
#             and len(global_plan) > 0
#         ):
#             self.curr_dist_to_path = self.get_dist_to_globalplan(
#                 global_plan, robot_pose
#             )

#     def get_dist_to_globalplan(self, global_plan: np.ndarray, robot_pose):
#         if self._kdtree is None:
#             self._kdtree = cKDTree(global_plan)

#         dist, _ = self._kdtree.query([robot_pose["x"], robot_pose["y"]])
#         return dist

#     def reset(self):
#         self._kdtree = None
#         self.curr_dist_to_path = None
