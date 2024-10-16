from typing import Any, Dict, List

from rl_utils.state_container import SimulationStateContainer
from rosnav_rl.spaces.observation_space import BaseObservationSpace
from rosnav_rl.utils.agent_state import (
    ActionSpaceState,
    AgentStateContainer,
    ObservationSpaceState,
)

from .base_space_manager import BaseSpaceManager


class RosnavSpaceManager(BaseSpaceManager):

    def __init__(
        self,
        simulation_state_container: SimulationStateContainer,
        observation_space_list: List[BaseObservationSpace] = None,
        action_space_kwargs: Dict[str, Any] = None,
        observation_space_kwargs: Dict[str, Any] = None,
    ):

        observation_space_kwargs = observation_space_kwargs or {}
        action_space_kwargs = action_space_kwargs or {}

        agent_state_container: AgentStateContainer = AgentStateContainer(
            action_space=ActionSpaceState(
                actions=simulation_state_container.robot.action_state.actions,
                is_discrete=action_space_kwargs["is_discrete"],
                is_holonomic=simulation_state_container.robot.action_state.is_holonomic,
            ),
            observation_space=ObservationSpaceState(
                laser_max_range=simulation_state_container.robot.laser_state.laser_max_range,
                laser_num_beams=simulation_state_container.robot.laser_state.laser_num_beams,
                ped_num_types=simulation_state_container.task.semantic_state.num_ped_types,
                ped_min_speed_x=simulation_state_container.task.semantic_state.ped_min_speed_x,
                ped_max_speed_x=simulation_state_container.task.semantic_state.ped_max_speed_x,
                ped_min_speed_y=simulation_state_container.task.semantic_state.ped_min_speed_y,
                ped_max_speed_y=simulation_state_container.task.semantic_state.ped_max_speed_y,
                ped_social_state_num=simulation_state_container.task.semantic_state.social_state_num,
                min_linear_vel=simulation_state_container.robot.action_state.velocity_state.min_linear_vel,
                max_linear_vel=simulation_state_container.robot.action_state.velocity_state.max_linear_vel,
                min_translational_vel=simulation_state_container.robot.action_state.velocity_state.min_translational_vel,
                max_translational_vel=simulation_state_container.robot.action_state.velocity_state.max_translational_vel,
                min_angular_vel=simulation_state_container.robot.action_state.velocity_state.min_angular_vel,
                max_angular_vel=simulation_state_container.robot.action_state.velocity_state.max_angular_vel,
            ),
        )

        super().__init__(
            agent_state_container=agent_state_container,
            action_space_kwargs=action_space_kwargs,
            observation_space_list=observation_space_list,
            observation_space_kwargs=observation_space_kwargs,
        )

    # @property
    # def required_observations(self):
    #     return get_required_observations(self._observation_space_manager.space_list)
