"""Tests for mirror canonicalization of observations and actions."""

from __future__ import annotations

import numpy as np
import pytest

from rosnav_rl.cfg.action_spaces import (
    CompositeActionSpace,
    DifferentialDriveActionSpace,
    HumanoidActionSpace,
    ManipulatorActionSpace,
)
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.spaces.space_manager import BaseSpaceManager, MirrorSpaceManager
from rosnav_rl.spaces.space_manager.mirror_space_manager import mirrored_beams

BEAMS = 720
ANGLE_MIN = -np.pi
INCREMENT = 2 * np.pi / BEAMS
PEDESTRIAN_MAPS = ("PedestrianVelXSpace", "PedestrianVelYSpace", "PedestrianTypeSpace", "PedestrianSocialStateSpace")


DRIVE = DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0))


def _spec(
    mirror: str, architecture: str = "AGENT_5", seed: int | None = None, action_space=DRIVE, **parameters
) -> AgentConfig:
    return AgentConfig(
        name=f"mirror_{mirror}",
        action_space=action_space,
        parameters=AgentParameters(
            laser_num_beams=BEAMS,
            laser_angle_min=ANGLE_MIN,
            laser_angle_increment=INCREMENT,
            robot_radius=0.4,
            min_translational_vel=0.0,
            max_translational_vel=0.0,
            **parameters,
        ),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name=architecture, parameters=PPO_Algorithm_Cfg(device="cpu", seed=seed))
        ),
        mirror=mirror,
    )


def _scene(sign: float = 1.0) -> dict:
    positions = np.array([[1.37, -1.13 * sign], [4.11, 2.63 * sign], [-3.29, 0.87 * sign]], dtype=np.float32)
    velocities = np.array([[0.4, 0.3 * sign], [-0.6, -0.2 * sign], [0.1, 0.5 * sign]], dtype=np.float32)
    scan = np.linspace(1.0, 9.0, BEAMS, dtype=np.float32)
    return {
        "front_laser": scan if sign > 0 else scan[mirrored_beams(BEAMS, ANGLE_MIN, INCREMENT)],
        "is_terminal": False,
        "pedestrian_relative_locations": positions,
        "pedestrian_relative_velocities": velocities,
        "pedestrian_vel_x": velocities[:, 0],
        "pedestrian_vel_y": velocities[:, 1],
        "pedestrian_types": np.zeros(3, dtype=np.int32),
        "pedestrian_social_states": np.array([1, 2, 0], dtype=np.int32),
        "dist_angle_to_subgoal": np.array([3.0, 0.6 * sign], dtype=np.float32),
        "last_action": np.array([0.5, 0.3 * sign], dtype=np.float32),
        "crowd_ttc": np.array([2.0, 5.0, 1.0, 2.0], dtype=np.float32),
    }


def test_canonical_manager_is_selected_by_the_agent_config():
    assert type(RL_Agent(_spec("none")).space_manager) is BaseSpaceManager
    assert isinstance(RL_Agent(_spec("canonical")).space_manager, MirrorSpaceManager)


def test_nearest_pedestrian_on_the_right_reads_as_the_reflected_scene():
    mirrored = RL_Agent(_spec("canonical", "AGENT_5_TTC")).space_manager
    plain = RL_Agent(_spec("none", "AGENT_5_TTC")).space_manager

    encoded = mirrored.encode_observation(_scene())
    expected = plain.encode_observation(_scene(sign=-1.0))

    assert mirrored.reflected
    assert encoded.keys() == expected.keys()
    for name in encoded:
        np.testing.assert_allclose(encoded[name], expected[name], atol=1e-6, err_msg=name)
        assert encoded[name].flags["C_CONTIGUOUS"]


def test_nearest_pedestrian_on_the_left_passes_through():
    mirrored = RL_Agent(_spec("canonical")).space_manager
    plain = RL_Agent(_spec("none")).space_manager

    encoded = mirrored.encode_observation(_scene(sign=-1.0))
    expected = plain.encode_observation(_scene(sign=-1.0))

    assert not mirrored.reflected
    for name in encoded:
        np.testing.assert_array_equal(encoded[name], expected[name])


def test_reflected_decisions_turn_and_strafe_the_other_way():
    mirrored = RL_Agent(_spec("canonical")).space_manager
    mirrored.encode_observation(_scene())

    np.testing.assert_allclose(mirrored.decode_action(np.array([0.7, 0.4])), [0.7, 0.0, -0.4])

    mirrored.encode_observation(_scene(sign=-1.0))

    np.testing.assert_allclose(mirrored.decode_action(np.array([0.7, 0.4])), [0.7, 0.0, 0.4])


def test_reflected_split_decisions_turn_the_drive_the_other_way():
    mirrored = RL_Agent(_spec("canonical")).space_manager
    mirrored.encode_observation(_scene())

    [(sub, command)] = mirrored.decode_split_action(np.array([0.7, 0.4]))

    assert sub.type == "differential_drive"
    np.testing.assert_allclose(command, [0.7, 0.0, -0.4])

    mirrored.encode_observation(_scene(sign=-1.0))

    np.testing.assert_allclose(mirrored.decode_split_action(np.array([0.7, 0.4]))[0][1], [0.7, 0.0, 0.4])


def test_reflected_split_decisions_leave_non_drive_slices_alone():
    humanoid = HumanoidActionSpace(locomotion_dof=3)
    mirrored = RL_Agent(
        _spec("canonical", action_space=CompositeActionSpace(action_spaces=[DRIVE, humanoid]))
    ).space_manager
    mirrored.encode_observation(_scene())

    drive, rest = mirrored.decode_split_action(np.array([0.7, 0.4, 0.1, 0.2, 0.3]))

    np.testing.assert_allclose(drive[1], [0.7, 0.0, -0.4])
    assert rest[0] == humanoid
    np.testing.assert_allclose(rest[1], [0.1, 0.2, 0.3])


def test_arm_actions_are_refused():
    arm = ManipulatorActionSpace(
        joint_names=["shoulder"], position_limits=[(-1.0, 1.0)], controller="arm_controller", tip_link="tool0"
    )

    with pytest.raises(ValueError, match="arm actions"):
        RL_Agent(_spec("canonical", action_space=CompositeActionSpace(action_spaces=[DRIVE, arm])))


def test_random_mode_reflects_whole_episodes_about_half_the_time():
    mirrored = RL_Agent(_spec("random", seed=3)).space_manager
    flags = []
    for _ in range(400):
        mirrored.reset_spaces()
        first = (mirrored.encode_observation(_scene()), mirrored.reflected)[1]
        second = (mirrored.encode_observation(_scene(sign=-1.0)), mirrored.reflected)[1]
        assert first == second
        flags.append(first)

    assert 0.4 < np.mean(flags) < 0.6


def test_random_mode_repeats_its_episode_flips_under_the_same_seed():
    def flips() -> list[bool]:
        manager = RL_Agent(_spec("random", seed=11)).space_manager
        out = []
        for _ in range(20):
            manager.reset_spaces()
            manager.encode_observation(_scene())
            out.append(manager.reflected)
        return out

    assert flips() == flips()


def test_a_sign_breaking_normalizer_is_refused():
    with pytest.raises(ValueError, match="sign-preserving"):
        RL_Agent(_spec("canonical", normalizer="min_max"))


def test_asymmetric_turn_bounds_are_refused():
    with pytest.raises(ValueError, match="not symmetric"):
        RL_Agent(_spec("canonical", min_angular_vel=-0.5))


def test_a_space_without_a_reflection_rule_is_refused():
    with pytest.raises(ValueError, match="no reflection rule"):
        RL_Agent(_spec("canonical", architecture="AGENT_1"))


def test_mirror_partners_pair_beams_by_angle():
    partner = mirrored_beams(BEAMS, ANGLE_MIN, INCREMENT)
    angles = ANGLE_MIN + INCREMENT * np.arange(BEAMS)
    np.testing.assert_allclose(np.cos(angles[partner]), np.cos(angles), atol=1e-9)
    np.testing.assert_allclose(np.sin(angles[partner]), -np.sin(angles), atol=1e-9)
    assert partner[0] == 0 and partner[BEAMS // 2] == BEAMS // 2 and partner[1] == BEAMS - 1


def test_reflected_laser_history_matches_the_mirrored_scans():
    mirrored = RL_Agent(_spec("canonical")).space_manager
    plain = RL_Agent(_spec("none")).space_manager
    partner = mirrored_beams(BEAMS, ANGLE_MIN, INCREMENT)
    rng = np.random.default_rng(5)
    for _ in range(4):
        scene = _scene()
        scene["front_laser"] = rng.uniform(0.3, 9.0, BEAMS).astype(np.float32)
        mirror = _scene(sign=-1.0)
        mirror["front_laser"] = scene["front_laser"][partner]
        encoded = mirrored.encode_observation(scene)
        expected = plain.encode_observation(mirror)
        assert mirrored.reflected
        np.testing.assert_array_equal(encoded["StackedLaserMapSpace"], expected["StackedLaserMapSpace"])


@pytest.mark.parametrize("outside", [10.01, 10.2])
def test_a_scene_mirrored_across_the_forward_axis_encodes_to_the_row_flipped_pedestrian_maps(outside: float):
    plain = RL_Agent(_spec("none")).space_manager
    other = RL_Agent(_spec("none")).space_manager
    scene = _scene()
    scene["pedestrian_relative_locations"] = np.array([[1.0, -outside], [2.0, outside], [3.0, 0.3]], dtype=np.float32)
    mirror = dict(scene)
    mirror["pedestrian_relative_locations"] = scene["pedestrian_relative_locations"] * np.array(
        [1.0, -1.0], dtype=np.float32
    )
    mirror["pedestrian_vel_y"] = -scene["pedestrian_vel_y"]
    encoded = plain.encode_observation(scene)
    expected = other.encode_observation(mirror)
    for name in PEDESTRIAN_MAPS:
        flipped = np.flip(encoded[name], axis=-2) * (-1.0 if name == "PedestrianVelYSpace" else 1.0)
        np.testing.assert_array_equal(flipped, expected[name], err_msg=name)
