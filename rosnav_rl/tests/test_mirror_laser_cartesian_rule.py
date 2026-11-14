import numpy as np
import pytest

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.spaces.observation_space.spaces.environment.feature_map_spaces import LaserCartesianMapSpace
from rosnav_rl.spaces.space_manager import MirrorSpaceManager
from rosnav_rl.spaces.space_manager.mirror_space_manager import mirrored_beams


def _spec(beams: int, angle_min: float, increment: float | None, encoding: int = 2) -> AgentConfig:
    return AgentConfig(
        name="mirror_cartesian",
        observation_encodings={"LaserCartesianMapSpace": encoding},
        action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
        parameters=AgentParameters(
            laser_num_beams=beams,
            laser_angle_min=angle_min,
            laser_angle_increment=increment,
            robot_radius=0.4,
            min_translational_vel=0.0,
            max_translational_vel=0.0,
        ),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_5", parameters=PPO_Algorithm_Cfg(device="cpu"))
        ),
        mirror="canonical",
    )


def _manager(spec: AgentConfig, **kwargs) -> MirrorSpaceManager:
    return MirrorSpaceManager(spec, [LaserCartesianMapSpace], kwargs)


def test_jackal_full_circle_cartesian_map_is_mirrorable():
    _manager(_spec(720, -np.pi, 2 * np.pi / 720))


def test_symmetric_partial_scan_cartesian_map_is_mirrorable():
    _manager(_spec(541, -0.75 * np.pi, np.deg2rad(0.5)))


def test_encoding_1_cartesian_map_mirrors_its_full_circle_beams():
    _manager(_spec(720, -np.pi, np.deg2rad(0.25), encoding=1))
    with pytest.raises(ValueError, match="no beam at the mirror angle"):
        _manager(_spec(720, -np.pi, np.deg2rad(0.25), encoding=2))


def test_asymmetric_scan_cartesian_map_is_refused():
    with pytest.raises(ValueError, match="no beam at the mirror angle"):
        _manager(_spec(541, -0.5 * np.pi, np.deg2rad(0.5)))


def test_odd_feature_map_size_cartesian_map_is_refused():
    with pytest.raises(ValueError, match="odd feature_map_size"):
        _manager(_spec(720, -np.pi, 2 * np.pi / 720), feature_map_size=79)


def test_row_flip_equals_the_map_of_the_mirrored_scan_on_an_even_grid():
    beams, angle_min, increment = 541, -0.75 * np.pi, np.deg2rad(0.5)
    kwargs = {
        "laser_num_beams": beams,
        "laser_angle_min": angle_min,
        "laser_angle_increment": increment,
        "encoding_version": 2,
    }
    scan = np.full(beams, 8.0, dtype=np.float32)
    mirrored = scan[mirrored_beams(beams, angle_min, increment)]

    plain = LaserCartesianMapSpace(**kwargs).encode_observation(scan)
    reflected = LaserCartesianMapSpace(**kwargs).encode_observation(mirrored)

    assert np.count_nonzero(plain) > 100
    assert np.count_nonzero(np.flipud(plain) != reflected) <= 4
