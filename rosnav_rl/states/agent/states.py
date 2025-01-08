from dataclasses import dataclass


# STATE NAMES HAVE TO RESEMBLE THE ARGUMENT NAMES IN THE ROSNAV_RL SPACES
@dataclass(frozen=True)
class ActionSpaceState:
    actions: list
    is_discrete: bool
    is_holonomic: bool


@dataclass(frozen=True)
class ObservationSpaceState:
    laser_num_beams: int
    laser_max_range: float
    min_linear_vel: float
    max_linear_vel: float
    min_translational_vel: float
    max_translational_vel: float
    min_angular_vel: float
    max_angular_vel: float
    ped_num_types: int
    ped_min_speed_x: float
    ped_max_speed_x: float
    ped_min_speed_y: float
    ped_max_speed_y: float
    ped_social_state_num: int
