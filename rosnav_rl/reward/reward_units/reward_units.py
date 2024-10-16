import random
from typing import Any, Callable, Dict
from warnings import warn

import numpy as np
import rospy
from rl_utils.state_container import SimulationStateContainer
from rl_utils.utils.observation_collector import *
from rl_utils.utils.observation_collector.constants import DONE_REASONS

from ..constants import DEFAULTS, REWARD_CONSTANTS
from ..reward_function import RewardFunction
from ..utils import check_params
from .base_reward_units import RewardUnit
from .reward_unit_factory import RewardUnitFactory

# UPDATE WHEN ADDING A NEW UNIT
__all__ = [
    "RewardGoalReached",
    "RewardSafeDistance",
    "RewardNoMovement",
    "RewardApproachGoal",
    "RewardCollision",
    "RewardDistanceTravelled",
    "RewardReverseDrive",
    "RewardAbruptVelocityChange",
    "RewardRootVelocityDifference",
    "RewardTwoFactorVelocityDifference",
    "RewardActiveHeadingDirection",
]


@RewardUnitFactory.register("goal_reached")
class RewardGoalReached(RewardUnit):
    required_observation_units = [DistAngleToGoal, DistAngleToSubgoal]
    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.SUCCESS.name,
        "is_success": True,
    }
    NOT_DONE_INFO = {"is_done": False}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.GOAL_REACHED.REWARD,
        _following_subgoal: bool = False,
        _on_safe_dist_violation: bool = DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when the goal is reached.

        Args:
            reward_function (RewardFunction): The reward function object holding this unit.
            reward (float, optional): The reward value for reaching the goal. Defaults to DEFAULTS.GOAL_REACHED.REWARD.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward
        self._goal_key = (
            DistAngleToSubgoal.name if _following_subgoal else DistAngleToGoal.name
        )

    def check_parameters(self, *args, **kwargs):
        if self._reward < 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Negative rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculates the reward and updates the information when the goal is reached.

        Args:
            distance_to_goal (float): Distance to the goal in m.
        """
        if obs_dict[self._goal_key][0] < simulation_state_container.task.goal_radius:
            self.add_reward(self._reward)
            self.add_info(self.DONE_INFO)
        else:
            self.add_info(self.NOT_DONE_INFO)


@RewardUnitFactory.register("safe_distance")
class RewardSafeDistance(RewardUnit):
    required_observation_units = [
        LaserSafeDistanceGenerator,
        PedSafeDistCollector,
        ObsSafeDistCollector,
    ]
    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.SAFE_DISTANCE.REWARD,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when violating the safe distance.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for violating the safe distance. Defaults to DEFAULTS.SAFE_DISTANCE.REWARD.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def check_safe_dist_violation(self, obs_dict: ObservationDict) -> bool:
        return (
            obs_dict[LaserSafeDistanceGenerator.name]
            or obs_dict.get(PedSafeDistCollector.name, False)
            or obs_dict.get(ObsSafeDistCollector.name, False)
        )

    def __call__(
        self,
        obs_dict: ObservationDict,
        *args: Any,
        **kwargs: Any,
    ):
        if self.check_safe_dist_violation(obs_dict):
            self.add_reward(self._reward)
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)


@RewardUnitFactory.register("factored_safe_distance")
class RewardFactoredSafeDistance(RewardUnit):
    required_observation_units = [
        LaserSafeDistanceGenerator,
        PedSafeDistCollector,
        ObsSafeDistCollector,
    ]
    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        factor: float = -0.5,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when violating the safe distance.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for violating the safe distance. Defaults to DEFAULTS.SAFE_DISTANCE.REWARD.
        """
        self._factor = factor
        super().__init__(reward_function, True, *args, **kwargs)

    def check_parameters(self, *args, **kwargs):
        if self._factor >= 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive factor may lead to unfavorable behaviors. "
                f"Current value: {self._factor}"
            )
            warn(warn_msg)

    def check_safe_dist_violation(self, obs_dict: ObservationDict) -> bool:
        return (
            obs_dict[LaserSafeDistanceGenerator.name]
            or obs_dict.get(PedSafeDistCollector.name, False)
            or obs_dict.get(ObsSafeDistCollector.name, False)
        )

    def __call__(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        *args: Any,
        **kwargs: Any,
    ):
        if self.check_safe_dist_violation(obs_dict=obs_dict):
            laser_min = (
                obs_dict[LaserCollector.name].min()
                if FullRangeLaserCollector.name not in obs_dict
                else obs_dict[FullRangeLaserCollector.name].min()
            )
            self.add_reward(
                self._factor
                * (
                    simulation_state_container.robot.safety_distance
                    + simulation_state_container.robot.radius
                    - laser_min
                )
            )
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)


@RewardUnitFactory.register("no_movement")
class RewardNoMovement(RewardUnit):
    required_observation_units = [LastActionCollector]

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.NO_MOVEMENT.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when there is no movement.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for no movement. Defaults to DEFAULTS.NO_MOVEMENT.REWARD.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any):
        action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )
        if (
            action is not None
            and abs(action[0]) <= REWARD_CONSTANTS.NO_MOVEMENT_TOLERANCE
        ):
            self.add_reward(self._reward)


@RewardUnitFactory.register("approach_goal")
class RewardApproachGoal(RewardUnit):
    required_observation_units = [
        GoalCollector,
        SubgoalCollector,
        RobotPoseCollector,
    ]

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        pos_factor: float = DEFAULTS.APPROACH_GOAL.POS_FACTOR,
        neg_factor: float = DEFAULTS.APPROACH_GOAL.NEG_FACTOR,
        _goal_update_threshold: float = DEFAULTS.APPROACH_GOAL._GOAL_UPDATE_THRESHOLD,
        _follow_subgoal: bool = False,
        _on_safe_dist_violation: bool = DEFAULTS.APPROACH_GOAL._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when approaching the goal.

        Args:
            reward_function (RewardFunction): The reward function object.
            pos_factor (float, optional): Positive factor for approaching the goal. Defaults to DEFAULTS.APPROACH_GOAL.POS_FACTOR.
            neg_factor (float, optional): Negative factor for distancing from the goal. Defaults to DEFAULTS.APPROACH_GOAL.NEG_FACTOR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.APPROACH_GOAL._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._pos_factor = pos_factor
        self._neg_factor = neg_factor
        self._goal_update_threshold = _goal_update_threshold

        self._goal_key = (
            SubgoalCollector.name if _follow_subgoal else GoalCollector.name
        )

        self.euclidean_distance = lambda a, b: np.sqrt(
            (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2
        )

        self.last_robot_pose = None
        self.last_goal_pose = None

    def check_parameters(self, *args, **kwargs):
        if self._pos_factor < 0 or self._neg_factor < 0:
            warn_msg = (
                f"[{self.__class__.__name__}] Both factors should be positive. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)
        if self._pos_factor >= self._neg_factor:
            warn_msg = (
                "'pos_factor' should be smaller than 'neg_factor' otherwise rotary trajectories will get rewarded. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)

    def __call__(self, obs_dict: ObservationDict, *args, **kwargs):
        if (
            self.last_robot_pose is not None and self.last_goal_pose is not None
        ):  # and not _inter_has_replanned:

            goal_location_diff = self.euclidean_distance(
                obs_dict[self._goal_key], self.last_goal_pose
            )
            if goal_location_diff >= self._goal_update_threshold:
                self.last_goal_pose: Union[
                    GoalCollector.data_class, SubgoalCollector.data_class
                ] = obs_dict[self._goal_key]
                self.last_robot_pose: RobotPoseCollector.data_class = obs_dict[
                    RobotPoseCollector.name
                ]
                return

            robot_pose: RobotPoseCollector.data_class = obs_dict[
                RobotPoseCollector.name
            ]

            last_goal_dist = self.euclidean_distance(
                self.last_goal_pose, self.last_robot_pose
            )
            curr_goal_dist = self.euclidean_distance(self.last_goal_pose, robot_pose)

            term = last_goal_dist - curr_goal_dist
            w = self._pos_factor if term > 0 else self._neg_factor
            self.add_reward(w * term)

        self.last_robot_pose: RobotPoseCollector.data_class = obs_dict[
            RobotPoseCollector.name
        ]
        self.last_goal_pose: Union[
            GoalCollector.data_class, SubgoalCollector.data_class
        ] = obs_dict[self._goal_key]

    def reset(self):
        self.last_robot_pose = None
        self.last_goal_pose = None


@RewardUnitFactory.register("collision")
class RewardCollision(RewardUnit):
    required_observation_units = [LaserCollector]
    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.COLLISION.name,
        "is_success": False,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.COLLISION.REWARD,
        bumper_zone: float = DEFAULTS.COLLISION.BUMPER_ZONE,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when a collision is detected.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for reaching the goal. Defaults to DEFAULTS.COLLISION.REWARD.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward
        self._bumper_zone = bumper_zone

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # quick Unity-specific check
        if obs_dict.get(CollisionCollector.name, False):
            self.add_reward(self._reward)
            self.add_info(self.DONE_INFO)
            return

        coll_in_blind_spot = False
        crash_radius = simulation_state_container.robot.radius + self._bumper_zone
        if len(obs_dict.get(FullRangeLaserCollector.name, [])) > 0:
            coll_in_blind_spot = obs_dict[FullRangeLaserCollector.name].min() <= (
                crash_radius
            )

        laser_min = (
            obs_dict[LaserCollector.name].min()
            if not FullRangeLaserCollector.name in obs_dict
            else obs_dict[FullRangeLaserCollector.name].min()
        )

        if laser_min <= crash_radius or coll_in_blind_spot:
            self.add_reward(self._reward)
            self.add_info(self.DONE_INFO)


@RewardUnitFactory.register("distance_travelled")
class RewardDistanceTravelled(RewardUnit):
    required_observation_units = [LastActionCollector]

    def __init__(
        self,
        reward_function: RewardFunction,
        consumption_factor: float = DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR,
        lin_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR,
        ang_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR,
        _on_safe_dist_violation: bool = DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward for the distance travelled.

        Args:
            reward_function (RewardFunction): The reward function object.
            consumption_factor (float, optional): Negative consumption factor. Defaults to DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR.
            lin_vel_scalar (float, optional): Scalar for the linear velocity. Defaults to DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR.
            ang_vel_scalar (float, optional): Scalar for the angular velocity. Defaults to DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._factor = consumption_factor
        self._lin_vel_scalar = lin_vel_scalar
        self._ang_vel_scalar = ang_vel_scalar

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any) -> Any:
        action = obs_dict.get(LastActionCollector.name, None)

        if action is None:
            return

        lin_vel, ang_vel = action[0], action[-1]
        reward = (
            (lin_vel * self._lin_vel_scalar) + (ang_vel * self._ang_vel_scalar)
        ) * -self._factor
        self.add_reward(reward)


@RewardUnitFactory.register("reverse_drive")
class RewardReverseDrive(RewardUnit):
    """
    A reward unit that provides a reward for driving in reverse.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        reward (float, optional): The reward value for driving in reverse. Defaults to DEFAULTS.REVERSE_DRIVE.REWARD.
        _on_safe_dist_violation (bool, optional): Whether to penalize for violating safe distance. Defaults to DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _reward (float): The reward value for driving in reverse.

    Methods:
        check_parameters: Checks if the reward value is positive and issues a warning if it is.
        __call__: Adds the reward value to the total reward if the action is not None and the first element of the action is less than 0.

    """

    required_observation_units = [LastActionCollector]

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.REVERSE_DRIVE.REWARD,
        threshold: float = None,
        _on_safe_dist_violation: bool = DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._reward = reward
        self._threshold = threshold if threshold else 0.0

    def check_parameters(self, *args, **kwargs):
        """
        Checks if the reward value is positive and issues a warning if it is.
        """
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, obs_dict: ObservationDict, *args, **kwargs):
        """
        Adds the reward value to the total reward if the action is not None and the first element of the action is less than 0.

        Args:
            action (np.ndarray): The action taken.

        """
        action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )
        if action is not None and action[0] < 0 and action[0] < self._threshold:
            self.add_reward(self._reward)


@RewardUnitFactory.register("factored_reverse_drive")
class RewardFactoredReverseDrive(RewardUnit):
    """
    A reward unit that provides a reward for driving in reverse.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        reward (float, optional): The reward value for driving in reverse. Defaults to DEFAULTS.REVERSE_DRIVE.REWARD.
        _on_safe_dist_violation (bool, optional): Whether to penalize for violating safe distance. Defaults to DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _reward (float): The reward value for driving in reverse.

    Methods:
        check_parameters: Checks if the reward value is positive and issues a warning if it is.
        __call__: Adds the reward value to the total reward if the action is not None and the first element of the action is less than 0.

    """

    required_observation_units = [LastActionCollector]

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        factor: float = -0.1,
        threshold: float = None,
        _on_safe_dist_violation: bool = DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._factor = factor
        self._threshold = threshold if threshold else 0.0

    def check_parameters(self, *args, **kwargs):
        """
        Checks if the reward value is positive and issues a warning if it is.
        """
        if self._factor < 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Negative rewards may lead to unfavorable behaviors. "
                f"Current value: {self._factor}"
            )
            warn(warn_msg)

    def __call__(self, obs_dict: ObservationDict, *args, **kwargs):
        """
        Adds the reward value to the total reward if the action is not None and the first element of the action is less than 0.

        Args:
            action (np.ndarray): The action taken.

        """
        action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )
        if action is not None and action[0] < 0 and action[0] < self._threshold:
            self.add_reward(self._factor * action[0])


@RewardUnitFactory.register("abrupt_velocity_change")
class RewardAbruptVelocityChange(RewardUnit):
    """
    A reward unit that penalizes abrupt changes in velocity.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        vel_factors (Dict[str, float], optional): Velocity factors for each dimension. Defaults to DEFAULTS.ABRUPT_VEL_CHANGE.VEL_FACTORS.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize abrupt velocity changes on safe distance violation. Defaults to DEFAULTS.ABRUPT_VEL_CHANGE._ON_SAFE_DIST_VIOLATION.

    Attributes:
        _vel_factors (Dict[str, float]): Velocity factors for each dimension.
        last_action (np.ndarray): The last action taken.
        _vel_change_fcts (List[Callable[[np.ndarray], None]]): List of velocity change functions.

    Methods:
        _get_vel_change_fcts(): Returns a list of velocity change functions.
        _prepare_reward_function(idx: int, factor: float) -> Callable[[np.ndarray], None]: Prepares a reward function for a specific dimension.
        __call__(action: np.ndarray, *args, **kwargs): Calculates the reward based on the action taken.
        reset(): Resets the last action to None.
    """

    required_observation_units = [LastActionCollector]

    def __init__(
        self,
        reward_function: RewardFunction,
        vel_factors: Dict[str, float] = DEFAULTS.ABRUPT_VEL_CHANGE.VEL_FACTORS,
        _on_safe_dist_violation: bool = DEFAULTS.ABRUPT_VEL_CHANGE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._vel_factors = vel_factors
        self.last_action = None

        self._vel_change_fcts = self._get_vel_change_fcts()

    def _get_vel_change_fcts(self):
        return [
            self._prepare_reward_function(int(idx), factor)
            for idx, factor in self._vel_factors.items()
        ]

    def _prepare_reward_function(
        self, idx: int, factor: float
    ) -> Callable[[np.ndarray], None]:
        def vel_change_fct(action: np.ndarray):
            assert isinstance(self.last_action, np.ndarray)
            vel_diff = abs(action[idx] - self.last_action[idx])
            self.add_reward(-((vel_diff**4 / 100) * factor))

        return vel_change_fct

    def __call__(self, obs_dict: ObservationDict, *args, **kwargs):
        action = obs_dict.get(LastActionCollector.name, None)

        if self.last_action is not None:
            for rew_fct in self._vel_change_fcts:
                rew_fct(action)
        self.last_action = action

    def reset(self):
        self.last_action = None


@RewardUnitFactory.register("root_velocity_difference")
class RewardRootVelocityDifference(RewardUnit):
    """
    A reward unit that calculates the difference in root velocity between consecutive actions.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        k (float, optional): The scaling factor for the velocity difference. Defaults to DEFAULTS.ROOT_VEL_DIFF.K.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize for violating safe distance.
            Defaults to DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _k (float): The scaling factor for the velocity difference.
        last_action (numpy.ndarray): The last action taken.

    Methods:
        __call__(self, action: np.ndarray, *args, **kwargs): Calculates the reward based on the velocity difference between
            the current action and the last action.
        reset(self): Resets the last action to None.
    """

    required_observation_units = [LastActionCollector]

    def __init__(
        self,
        reward_function: RewardFunction,
        k: float = DEFAULTS.ROOT_VEL_DIFF.K,
        _on_safe_dist_violation: bool = DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._k = k
        self.last_action = None

    def __call__(self, obs_dict: ObservationDict, *args, **kwargs):
        """
        Calculates and adds the reward based on the given action.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            None
        """
        action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )

        if self.last_action is not None:
            vel_diff = np.linalg.norm((action - self.last_action) ** 2)
            if vel_diff < self._k:
                self.add_reward((1 - vel_diff) / self._k)
        self.last_action = action

    def reset(self):
        self.last_action = None


@RewardUnitFactory.register("two_factor_velocity_difference")
class RewardTwoFactorVelocityDifference(RewardUnit):
    """
    A reward unit that calculates the difference in velocity between consecutive actions
    and penalizes the agent based on the squared difference.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        alpha (float, optional): The weight for the squared difference in the first dimension of the action. Defaults to DEFAULTS.ROOT_VEL_DIFF.K.
        beta (float, optional): The weight for the squared difference in the last dimension of the action. Defaults to DEFAULTS.ROOT_VEL_DIFF.K.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize the agent on safe distance violation. Defaults to DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    required_observation_units = [LastActionCollector]

    def __init__(
        self,
        reward_function: RewardFunction,
        alpha: float = DEFAULTS.TWO_FACTOR_VEL_DIFF.ALPHA,
        beta: float = DEFAULTS.TWO_FACTOR_VEL_DIFF.BETA,
        _on_safe_dist_violation: bool = DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._alpha = alpha
        self._beta = beta
        self.last_action = None

    def __call__(self, obs_dict: ObservationDict, *args, **kwargs):
        """
        Calculates and adds the reward based on the difference between the current action and the last action.

        Args:
            action (np.ndarray): The current action.

        Returns:
            None
        """
        action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )

        if self.last_action is not None:
            diff = abs((action - self.last_action))
            reward = -(diff[0] * self._alpha + diff[-1] * self._beta)
            self.add_reward(reward)
        self.last_action = action

    def reset(self):
        self.last_action = None


@RewardUnitFactory.register("active_heading_direction")
class RewardActiveHeadingDirection(RewardUnit):
    """
    Reward unit that calculates the reward based on the active heading direction of the robot.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        r_angle (float, optional): Weight for difference between max deviation of heading direction and desired heading direction. Defaults to 0.6.
        theta_m (float, optional): Maximum allowable deviation of the heading direction. Defaults to np.pi/6.
        theta_min (int, optional): Minimum allowable deviation of the heading direction. Defaults to 1000.
        ped_min_dist (float, optional): Minimum distance to pedestrians. Defaults to 8.0.
        iters (int, optional): Number of iterations to find a reachable available theta. Defaults to 60.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize the reward on safe distance violation. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _r_angle (float): Desired heading direction in the robot's local frame.
        _theta_m (float): Maximum allowable deviation of the heading direction.
        _theta_min (int): Minimum allowable deviation of the heading direction.
        _ped_min_dist (float): Minimum application distance to pedestrians.
        _iters (int): Number of iterations to find a reachable available theta.
    """

    required_observation_units = [
        DistAngleToGoal,
        DistAngleToSubgoal,
        LastActionCollector,
        PedestrianRelativeLocation,
        PedestrianRelativeVelX,
        PedestrianRelativeVelY,
    ]

    def __init__(
        self,
        reward_function: RewardFunction,
        r_angle: float = 0.6,
        theta_m: float = np.pi / 6,
        theta_min: int = 1000,
        ped_min_dist: float = 8.0,
        iters: int = 60,
        _following_subgoal: bool = False,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._r_angle = r_angle
        self._theta_m = theta_m
        self._theta_min = theta_min
        self._ped_min_dist = ped_min_dist
        self._iters = iters

        self._goal_key = (
            DistAngleToSubgoal.name if _following_subgoal else DistAngleToGoal.name
        )

    def __call__(
        self,
        # dist_angle_to_goal: np.ndarray,
        # action: np.ndarray,
        # relative_location: np.ndarray,
        # relative_x_vel: np.ndarray,
        # relative_y_vel: np.ndarray,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        *args,
        **kwargs,
    ) -> float:
        """
        Calculates the reward based on the active heading direction of the robot.

        Args:
            dist_angle_to_goal (np.ndarray): The goal position in the robot's frame of reference.
            action (np.ndarray): The last action taken by the robot.
            relative_location (np.ndarray): The relative location of the pedestrians.
            relative_x_vel (np.ndarray): The relative x-velocity of the pedestrians.
            relative_y_vel (np.ndarray): The relative y-velocity of the pedestrians.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: The calculated reward based on the active heading direction.
        """
        dist_angle_to_goal: Union[
            DistAngleToGoal.data_class, DistAngleToSubgoal.name
        ] = obs_dict[self._goal_key]
        action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )
        relative_location: PedestrianRelativeLocation.data_class = obs_dict.get(
            PedestrianRelativeLocation.name, None
        )
        relative_x_vel: PedestrianRelativeVelX.data_class = obs_dict.get(
            PedestrianRelativeVelX.name, None
        )
        relative_y_vel: PedestrianRelativeVelY.data_class = obs_dict.get(
            PedestrianRelativeVelY.name, None
        )

        if (
            relative_location is None
            or relative_x_vel is None
            or relative_y_vel is None
        ):
            return 0.0

        # prefer goal theta:
        theta_pre = dist_angle_to_goal[1]
        d_theta = theta_pre

        v_x = action[0]

        # get the pedestrian's position:
        if len(relative_location) != 0:  # tracker results
            d_theta = np.pi / 2  # theta_pre
            theta_min = self._theta_min
            for _ in range(self._iters):
                theta = random.uniform(-np.pi, np.pi)
                free = True
                for ped_location, ped_x_vel, ped_y_vel in zip(
                    relative_location, relative_x_vel, relative_y_vel
                ):
                    p_x = ped_location[0]
                    p_y = ped_location[1]
                    p_vx = ped_x_vel
                    p_vy = ped_y_vel

                    ped_dis = np.linalg.norm([p_x, p_y])

                    if ped_dis <= self._ped_min_dist:
                        ped_theta = np.arctan2(p_y, p_x)

                        # 3*robot_radius:= estimation for sum of the pedestrian radius and the robot radius
                        vector = (
                            ped_dis**2
                            - (3 * simulation_state_container.robot.radius) ** 2
                        )
                        if vector < 0:
                            continue  # in this case the robot likely crashed into the pedestrian, disregard this pedestrian

                        vo_theta = np.arctan2(
                            3 * simulation_state_container.robot.radius,
                            np.sqrt(vector),
                        )
                        # Check if the robot's trajectory intersects with the pedestrian's VO cone
                        theta_rp = np.arctan2(
                            v_x * np.sin(theta) - p_vy, v_x * np.cos(theta) - p_vx
                        )
                        if theta_rp >= (ped_theta - vo_theta) and theta_rp <= (
                            ped_theta + vo_theta
                        ):
                            free = False
                            break

                # Find the reachable available theta that minimizes the difference from the goal theta
                if free:
                    theta_diff = (theta - theta_pre) ** 2
                    if theta_diff < theta_min:
                        theta_min = theta_diff
                        d_theta = theta

        else:  # no obstacles:
            d_theta = theta_pre

        return self._r_angle * (self._theta_m - abs(d_theta))


@RewardUnitFactory.register("ped_safe_distance")
class RewardPedSafeDistance(RewardUnit):
    required_observation_units = [PedSafeDistCollector]
    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.PED_SAFE_DISTANCE.REWARD,
        safe_dist: float = DEFAULTS.PED_SAFE_DISTANCE.SAFE_DIST,
        *args,
        **kwargs,
    ):
        """Unity-specific class for calculating the reward when violating the ped-specific safe
        distance.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for violating the safe distance. Defaults to
                DEFAULTS.PED_SAFE_DISTANCE.REWARD.
            safe_dist (bool, optional): Safety distance which should not be passed. The value should
                not include the radius of the robot body. Defaults to
                DEFAULTS.PED_SAFE_DISTANCE.SAFE_DIST.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any):
        if obs_dict[PedSafeDistCollector.name]:
            self.add_reward(self._reward)
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)


@RewardUnitFactory.register("obs_safe_distance")
class RewardObsSafeDistance(RewardUnit):
    required_observation_units = [ObsSafeDistCollector]
    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.OBS_SAFE_DISTANCE.REWARD,
        *args,
        **kwargs,
    ):
        """Unity-specific class for calculating the reward when violating the obs-specific safe
        distance.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for violating the safe distance. Defaults to
                DEFAULTS.OBS_SAFE_DISTANCE.REWARD.
            safe_dist (bool, optional): Safety distance which should not be passed. The value should
                not include the radius of the robot body. Defaults to
                DEFAULTS.OBS_SAFE_DISTANCE.SAFE_DIST.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any):
        if (
            ObsSafeDistCollector.name in obs_dict
            and obs_dict[ObsSafeDistCollector.name]
        ):
            self.add_reward(self._reward)
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)


@RewardUnitFactory.register("ped_type_safety_distance")
class RewardPedTypeSafetyDistance(RewardUnit):
    """
    RewardPedTypeDistance is a reward unit that provides a reward based on the distance between the agent and a specific pedestrian type.

    Args:
        reward_function (RewardFunction): The reward function to which this reward unit belongs.
        ped_type (int, optional): The type of pedestrian to consider. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE.
        reward (float, optional): The reward value to be added if the distance to the pedestrian type is less than the safety distance. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.REWARD.
        safety_distance (float, optional): The safety distance threshold. If the distance to the pedestrian type is less than this value, the reward is added. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE.
        _on_safe_dist_violation (bool, optional): A flag indicating whether to trigger a violation event when the safety distance is violated. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _type (int): The type of pedestrian to consider.
        _reward (float): The reward value to be added if the distance to the pedestrian type is less than the safety distance.
        _safety_distance (float): The safety distance threshold.

    Methods:
        __call__(*args, **kwargs): Calculates the reward based on the distance to the pedestrian type.
        reset(): Resets the reward unit.
    """

    required_observation_units = [PedestrianDistanceGenerator]

    def __init__(
        self,
        reward_function: RewardFunction,
        type_reward_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.REWARD,
        safety_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._reward = reward
        self._safety_distance = safety_distance
        self._type_reward_pairs = (
            type_reward_pairs
            if isinstance(type_reward_pairs, dict)
            else {ped_type: reward}
        )

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any) -> None:
        ped_type_min_distances = obs_dict[PedestrianDistanceGenerator.name]

        for ped_type, reward in self._type_reward_pairs.items():
            if ped_type not in ped_type_min_distances:
                rospy.logwarn_throttle(
                    60,
                    f"[{rospy.get_name()}, {self.__class__.__name__}] Pedestrian type {ped_type} not found.",
                )
                continue

            if ped_type_min_distances[ped_type] < self._safety_distance:
                self.add_reward(reward)

    def reset(self):
        pass


@RewardUnitFactory.register("ped_type_factored_safety_distance")
class RewardPedTypeFactoredSafetyDistance(RewardUnit):

    required_observation_units = [PedestrianDistanceGenerator]

    def __init__(
        self,
        reward_function: RewardFunction,
        type_factor_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.TYPE,
        factor: float = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.FACTOR,
        safety_distance: float = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFICE_FACTORED_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._factor = factor
        self._safety_distance = safety_distance
        self._type_factor_pairs = (
            type_factor_pairs
            if isinstance(type_factor_pairs, dict)
            else {ped_type: factor}
        )

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any) -> None:
        ped_type_min_distances = obs_dict[PedestrianDistanceGenerator.name]

        for ped_type, factor in self._type_factor_pairs.items():
            if ped_type not in ped_type_min_distances:
                rospy.logwarn_throttle(
                    60,
                    f"[{rospy.get_name()}, {self.__class__.__name__}] Pedestrian type {ped_type} not found.",
                )
                continue

            if ped_type_min_distances[ped_type] < self._safety_distance:
                self.add_reward(
                    factor * (self._safety_distance - ped_type_min_distances[ped_type])
                )

    def reset(self):
        pass


@RewardUnitFactory.register("ped_type_collision")
class RewardPedTypeCollision(RewardUnit):
    """
    RewardPedTypeCollision is a reward unit that provides a reward when the robot collides with a specific pedestrian type.

    Args:
        reward_function (RewardFunction): The reward function to which this reward unit belongs.
        ped_type (int, optional): The specific pedestrian type to check for collision. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.TYPE.
        reward (float, optional): The reward value to be added when a collision with the specific pedestrian type occurs. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.REWARD.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _type (int): The specific pedestrian type to check for collision.
        _reward (float): The reward value to be added when a collision with the specific pedestrian type occurs.
    """

    required_observation_units = [PedestrianDistanceGenerator]

    def __init__(
        self,
        reward_function: RewardFunction,
        type_reward_pairs: Dict[int, float] = None,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.REWARD,
        bumper_zone: float = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.BUMPER_ZONE,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, True, *args, **kwargs)
        self._type_reward_pairs = (
            type_reward_pairs
            if isinstance(type_reward_pairs, dict)
            else {ped_type: reward}
        )
        self._bumper_zone = bumper_zone

    def __call__(
        self,
        obs_dict: ObservationDict,
        simulation_state_container: SimulationStateContainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Checks if the robot has collided with the specific pedestrian type and adds the reward if a collision occurs.

        Args:
            *args: Variable length argument list.
            **observation_dict: Arbitrary keyword arguments.
        """
        ped_type_min_distances = obs_dict[PedestrianDistanceGenerator.name]

        for ped_type, reward in self._type_reward_pairs.items():
            if ped_type not in ped_type_min_distances:
                rospy.logwarn_throttle(
                    60,
                    f"[{rospy.get_name()}, {self.__class__.__name__}] Pedestrian type {ped_type} not found.",
                )
                continue

            if ped_type_min_distances[ped_type] <= (
                self._bumper_zone + simulation_state_container.robot.radius
            ):
                self.add_reward(reward)

    def reset(self):
        pass


@RewardUnitFactory.register("ped_type_vel_constraint")
class RewardPedTypeVelocityConstraint(RewardUnit):

    required_observation_units = [
        PedestrianDistanceGenerator,
        RobotPoseCollector,
        LastActionCollector,
    ]

    def __init__(
        self,
        reward_function: RewardFunction,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        penalty_factor: float = 0.05,
        active_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._penalty_factor = penalty_factor
        self._active_distance = active_distance

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any) -> None:
        ped_type_min_distances = obs_dict.get(PedestrianDistanceGenerator.name, None)
        action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )

        if action is None:
            return

        if ped_type_min_distances is None:
            return

        if self._type not in ped_type_min_distances:
            rospy.logwarn_throttle(
                60,
                f"[{rospy.get_name()}, {self.__class__.__name__}] Pedestrian type {self._type} not found.",
            )
            return

        if ped_type_min_distances[self._type] < self._active_distance:
            self.add_reward(-self._penalty_factor * action[0])

    def reset(self):
        pass


@RewardUnitFactory.register("angular_vel_constraint")
class RewardAngularVelocityConstraint(RewardUnit):
    required_observation_units = [LastActionCollector]

    def __init__(
        self,
        reward_function: RewardFunction,
        penalty_factor: float = -0.05,
        threshold: float = None,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._penalty_factor = penalty_factor
        self._threshold = threshold

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any) -> None:
        last_action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )
        angular = abs(last_action[2]) if last_action is not None else 0.0

        if self._threshold and angular > self._threshold:
            self.add_reward(self._penalty_factor * angular)

    def reset(self):
        pass


@RewardUnitFactory.register("max_steps_exceeded")
class RewardMaxStepsExceeded(RewardUnit):
    """
    A reward unit that penalizes the agent when the maximum number of steps is exceeded.

    Args:
        reward_function (RewardFunction): The reward function to which this unit belongs.
        penalty (float, optional): The penalty value to be applied when the maximum steps are exceeded. Defaults to 10.
        _on_safe_dist_violation (bool, optional): Whether to apply the penalty on safe distance violation. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _penalty (float): The penalty value to be applied when the maximum steps are exceeded.
        _steps (int): The current number of steps taken.

    Methods:
        __call__(*args, **kwargs): Updates the step count and applies the penalty if the maximum steps are exceeded.
        reset(): Resets the step count to zero.
    """

    required_observation_units = []

    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.STEP_LIMIT.name,
        "is_success": 0,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        penalty: float = 10,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._penalty = penalty
        self._steps = 0

    def check_parameters(self, *args, **kwargs):
        if self._penalty < 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"The penalty should be a positive value as it is going to be subtracted from the total reward."
                f"Current value: {self._penalty}"
            )
            warn(warn_msg)

    def __call__(
        self,
        simulation_state_container: SimulationStateContainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates the step count and applies the penalty if the maximum steps are exceeded.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._steps += 1
        if self._steps >= simulation_state_container.task.max_steps:
            self.add_reward(-self._penalty)
            self.add_info(self.DONE_INFO)

    def reset(self):
        """
        Resets the step count to zero.
        """
        self._steps = 0


@RewardUnitFactory.register("linear_vel_boost")
class RewardLinearVelBoost(RewardUnit):
    required_observation_units = [LastActionCollector]

    def __init__(
        self,
        reward_function: RewardFunction,
        reward_factor: float = -0.05,
        threshold: float = None,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward_factor = reward_factor
        self._threshold = threshold

    def __call__(self, obs_dict: ObservationDict, *args: Any, **kwargs: Any) -> None:
        last_action: LastActionCollector.data_class = obs_dict.get(
            LastActionCollector.name, None
        )
        linear = last_action[0] if last_action is not None else 0.0

        if self._threshold and linear > self._threshold:
            self.add_reward(self._reward_factor * linear)

    def reset(self):
        pass
