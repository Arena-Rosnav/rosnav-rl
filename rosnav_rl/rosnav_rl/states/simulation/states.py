from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LaserState:
    attach_full_range_laser: bool = field(default=False)
    laser_num_beams: int = field(default=0)
    laser_max_range: float = field(default=0.0)


@dataclass(frozen=True)
class VelocityState:
    min_linear_vel: float = 0.0
    max_linear_vel: float = 0.0
    min_angular_vel: float = 0.0
    max_angular_vel: float = 0.0
    min_translational_vel: Optional[float] = None
    max_translational_vel: Optional[float] = None


@dataclass(frozen=True)
class ActionState:
    is_discrete: bool = False
    actions: list = field(default_factory=list)
    velocity_state: VelocityState = field(default_factory=VelocityState)
    is_holonomic: Optional[bool] = False


@dataclass(frozen=True)
class RobotState:
    radius: float = 0.0
    safety_distance: float = 0.0
    action_state: ActionState = field(default_factory=ActionState)
    laser_state: LaserState = field(default_factory=LaserState)


@dataclass(frozen=True)
class SemanticState:
    num_ped_types: int = 0
    ped_min_speed_x: float = 0.0
    ped_max_speed_x: float = 0.0
    ped_min_speed_y: float = 0.0
    ped_max_speed_y: float = 0.0
    social_state_num: int = 0


@dataclass(frozen=True)
class TaskModuleState:
    tm_robots: str = None
    tm_obstacles: str = None
    tm_modules: str = None


@dataclass(frozen=False)
class TaskState:
    goal_radius: float = 0.0
    max_steps: int = 0
    semantic_state: SemanticState = field(default_factory=SemanticState)
    task_modules: TaskModuleState = field(default_factory=TaskModuleState)
