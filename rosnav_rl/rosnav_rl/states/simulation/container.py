from dataclasses import dataclass, field

from rosnav_rl.states.agent import (
    ActionSpaceState,
    AgentStateContainer,
    ObservationSpaceState,
)

# from ..distributor import StateDistributor
from .states import RobotState, TaskState


@dataclass(frozen=False)
class SimulationStateContainer:
    """
    A container class for holding the state of the simulation, including the robot state and task state.

    Attributes:
        robot (RobotState): The state of the robot in the simulation.
        task (TaskState): The state of the task in the simulation.

    Methods:
        distribute():
            Distributes the state using the StateDistributor.

        to_agent_state_container() -> AgentStateContainer:
            Converts the simulation state to an agent state container, which includes the action space and observation space states.
    """

    robot: RobotState = field(default_factory=RobotState)
    task: TaskState = field(default_factory=TaskState)

    # def distribute(self):
    #     StateDistributor(self).distribute()

    def to_agent_state_container(self) -> AgentStateContainer:
        return AgentStateContainer(
            action_space=ActionSpaceState(
                actions=self.robot.action_state.actions,
                is_discrete=self.robot.action_state.is_discrete,
                is_holonomic=self.robot.action_state.is_holonomic,
            ),
            observation_space=ObservationSpaceState(
                laser_max_range=self.robot.laser_state.laser_max_range,
                laser_num_beams=self.robot.laser_state.laser_num_beams,
                ped_num_types=self.task.semantic_state.num_ped_types,
                ped_min_speed_x=self.task.semantic_state.ped_min_speed_x,
                ped_max_speed_x=self.task.semantic_state.ped_max_speed_x,
                ped_min_speed_y=self.task.semantic_state.ped_min_speed_y,
                ped_max_speed_y=self.task.semantic_state.ped_max_speed_y,
                ped_social_state_num=self.task.semantic_state.social_state_num,
                min_linear_vel=self.robot.action_state.velocity_state.min_linear_vel,
                max_linear_vel=self.robot.action_state.velocity_state.max_linear_vel,
                min_translational_vel=self.robot.action_state.velocity_state.min_translational_vel,
                max_translational_vel=self.robot.action_state.velocity_state.max_translational_vel,
                min_angular_vel=self.robot.action_state.velocity_state.min_angular_vel,
                max_angular_vel=self.robot.action_state.velocity_state.max_angular_vel,
            ),
        )
