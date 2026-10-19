"""Tests for the AgentConfig container and typed action space hierarchy."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from rosnav_rl.cfg.action_spaces import (
    ActionSpaceSpec,
    BaseActionSpace,
    CompositeActionSpace,
    DifferentialDriveActionSpace,
    HumanoidActionSpace,
    ManipulatorActionSpace,
    OmnidirectionalActionSpace,
)
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.model.stable_baselines3.cfg import StableBaselinesCfg, PPO_Cfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.spaces.action_space.action_space_manager import ActionSpaceManager


# ============================================================
# Helpers
# ============================================================


def _make_sb3_framework():
    return StableBaselinesCfg(
        algorithm=PPO_Cfg(
            architecture_name="test_arch",
            parameters=PPO_Algorithm_Cfg(),
        )
    )


# ============================================================
# DifferentialDriveActionSpace
# ============================================================


class TestDifferentialDrive:
    def test_continuous_space(self):
        spec = DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0))
        assert spec.num_dof == 2
        assert not spec.is_discrete
        space = spec.get_gym_space()
        assert isinstance(space, spaces.Box)
        assert space.shape == (2,)
        np.testing.assert_array_almost_equal(space.low, [-0.5, -1.0])
        np.testing.assert_array_almost_equal(space.high, [1.0, 1.0])

    def test_continuous_decode(self):
        spec = DifferentialDriveActionSpace()
        action = np.array([0.3, -0.5])
        cmd = spec.decode(action)
        np.testing.assert_array_almost_equal(cmd, [0.3, 0.0, -0.5])

    def test_discrete_space(self):
        actions = [
            {"name": "fwd", "linear": 0.3, "angular": 0.0},
            {"name": "turn", "linear": 0.0, "angular": 0.5},
        ]
        spec = DifferentialDriveActionSpace(discrete_actions=actions)
        assert spec.is_discrete
        space = spec.get_gym_space()
        assert isinstance(space, spaces.Discrete)
        assert space.n == 2

    def test_discrete_decode(self):
        actions = [
            {"name": "fwd", "linear": 0.3, "angular": 0.0},
            {"name": "turn", "linear": 0.0, "angular": 0.5},
        ]
        spec = DifferentialDriveActionSpace(discrete_actions=actions)
        cmd = spec.decode(np.array([1]))
        np.testing.assert_array_almost_equal(cmd, [0.0, 0.0, 0.5])

    def test_to_discrete(self):
        spec = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.0, 1.0))
        discrete = spec.to_discrete(buckets_linear=3, buckets_angular=4)
        assert discrete.is_discrete
        assert len(discrete.discrete_actions) > 0

    def test_legacy_round_trip(self):
        spec = DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0))
        legacy = spec._to_legacy_actions()
        assert legacy == {"linear_range": [-0.5, 1.0], "angular_range": [-1.0, 1.0]}

    def test_legacy_round_trip_discrete(self):
        actions = [{"linear": 0.3, "angular": 0.0}]
        spec = DifferentialDriveActionSpace(discrete_actions=actions)
        assert spec._to_legacy_actions() == actions


# ============================================================
# OmnidirectionalActionSpace
# ============================================================


class TestOmnidirectional:
    def test_continuous_space(self):
        spec = OmnidirectionalActionSpace()
        assert spec.num_dof == 3
        assert not spec.is_discrete
        space = spec.get_gym_space()
        assert isinstance(space, spaces.Box)
        assert space.shape == (3,)

    def test_continuous_decode(self):
        spec = OmnidirectionalActionSpace()
        action = np.array([0.1, -0.2, 0.5])
        cmd = spec.decode(action)
        np.testing.assert_array_almost_equal(cmd, [0.1, -0.2, 0.5])

    def test_discrete_decode(self):
        actions = [{"linear_x": 0.3, "linear_y": -0.1, "angular": 0.2}]
        spec = OmnidirectionalActionSpace(discrete_actions=actions)
        cmd = spec.decode(np.array([0]))
        np.testing.assert_array_almost_equal(cmd, [0.3, -0.1, 0.2])

    def test_legacy_actions_continuous(self):
        spec = OmnidirectionalActionSpace(
            linear_range_x=(-1.0, 1.0), linear_range_y=(-0.5, 0.5), angular_range=(-2.0, 2.0)
        )
        legacy = spec._to_legacy_actions()
        assert legacy["linear_range"]["x"] == [-1.0, 1.0]
        assert legacy["linear_range"]["y"] == [-0.5, 0.5]
        assert legacy["angular_range"] == [-2.0, 2.0]


# ============================================================
# ManipulatorActionSpace
# ============================================================


@dataclass
class _Arm:
    chain: list[str]
    controller: str = "arm_controller"
    tip_link: str = "tool0"
    joint_limits: dict[str, dict[str, object]] = field(default_factory=dict)
    named_poses: dict[str, dict[str, float]] = field(default_factory=dict)


def _manipulator(position_limits):
    return ManipulatorActionSpace(
        joint_names=[f"joint_{i}" for i in range(len(position_limits))],
        position_limits=position_limits,
        controller="arm_controller",
        tip_link="tool0",
    )


class TestManipulator:
    def test_space(self):
        spec = _manipulator([(-1, 1), (-2, 2), (-0.5, 0.5)])
        assert spec.num_dof == 3
        space = spec.get_gym_space()
        assert isinstance(space, spaces.Box)
        assert space.shape == (3,)
        np.testing.assert_array_almost_equal(space.low, [-1, -1, -1])
        np.testing.assert_array_almost_equal(space.high, [1, 1, 1])

    def test_decode(self):
        spec = _manipulator([(-1, 1), (-2, 2)])
        action = np.array([0.5, -1.0])
        np.testing.assert_array_almost_equal(spec.decode(action), [0.5, -2.0])

    def test_decode_clips_to_the_joint_limits(self):
        spec = _manipulator([(0.0, 1.0), (-2, 2)])
        np.testing.assert_array_almost_equal(spec.decode(np.array([3.0, -7.0])), [1.0, -2.0])

    def test_legacy_actions_carry_the_position_limits(self):
        spec = _manipulator([(-1, 1), (-2, 2)])
        assert spec._to_legacy_actions() == {"joint_limits": [[-1, 1], [-2, 2]]}

    def test_from_config_reads_chain_controller_and_limits(self):
        arm = _Arm(
            chain=["shoulder", "elbow"],
            controller="ur_arm_controller",
            joint_limits={
                "shoulder": {"has_position_limits": True, "min_position": -1.5, "max_position": 1.5},
                "elbow": {"has_position_limits": False},
            },
        )

        spec = ManipulatorActionSpace.from_config(arm)

        assert spec.joint_names == ["shoulder", "elbow"]
        assert spec.controller == "ur_arm_controller"
        assert spec.tip_link == "tool0"
        assert spec.position_limits == [(-1.5, 1.5), (-6.283185307179586, 6.283185307179586)]
        assert spec.stow_positions is None

    def test_from_config_matches_limits_by_joint_suffix(self):
        arm = _Arm(
            chain=["arm_shoulder"],
            joint_limits={"shoulder": {"has_position_limits": True, "min_position": -0.5, "max_position": 0.25}},
        )

        assert ManipulatorActionSpace.from_config(arm).position_limits == [(-0.5, 0.25)]

    def test_from_config_orders_the_stow_pose_by_the_chain(self):
        arm = _Arm(chain=["shoulder", "elbow"], named_poses={"stow": {"elbow": 2.0, "shoulder": -1.0, "wrist": 0.5}})

        assert ManipulatorActionSpace.from_config(arm).stow_positions == [-1.0, 2.0]

    def test_from_config_drops_a_stow_pose_missing_a_chain_joint(self):
        arm = _Arm(chain=["shoulder", "elbow"], named_poses={"stow": {"shoulder": -1.0}})

        assert ManipulatorActionSpace.from_config(arm).stow_positions is None

    def test_from_config_ignores_other_named_poses(self):
        arm = _Arm(chain=["shoulder"], named_poses={"home": {"shoulder": 0.0}})

        assert ManipulatorActionSpace.from_config(arm).stow_positions is None


# ============================================================
# HumanoidActionSpace
# ============================================================


class TestHumanoid:
    def test_space_locomotion_only(self):
        spec = HumanoidActionSpace(locomotion_dof=6)
        assert spec.num_dof == 6
        space = spec.get_gym_space()
        assert space.shape == (6,)

    def test_space_with_upper_body(self):
        spec = HumanoidActionSpace(
            locomotion_dof=6,
            upper_body_joint_limits=[(-1, 1), (-1, 1)],
        )
        assert spec.num_dof == 8
        space = spec.get_gym_space()
        assert space.shape == (8,)


# ============================================================
# CompositeActionSpace
# ============================================================


class TestComposite:
    def _parts(self):
        return (
            DifferentialDriveActionSpace(linear_range=(-1.0, 2.0), angular_range=(-3.0, 3.0)),
            HumanoidActionSpace(locomotion_dof=2, upper_body_joint_limits=[(-0.5, 0.5)]),
        )

    def test_nested_composites_flatten_into_one_list(self):
        drive, humanoid = self._parts()
        spec = CompositeActionSpace(action_spaces=[CompositeActionSpace(action_spaces=[drive]), humanoid])

        assert spec.action_spaces == [drive, humanoid]
        assert spec.num_dof == 5
        assert spec.get_gym_space().shape == (5,)

    def test_nested_composites_flatten_from_yaml_data(self):
        spec = CompositeActionSpace.model_validate(
            {
                "action_spaces": [
                    {"type": "composite", "action_spaces": [{"type": "differential_drive"}]},
                    {"type": "humanoid", "locomotion_dof": 2},
                ]
            }
        )

        assert [type(s) for s in spec.action_spaces] == [DifferentialDriveActionSpace, HumanoidActionSpace]

    def test_discrete_sub_spaces_are_refused(self):
        drive = DifferentialDriveActionSpace().to_discrete(buckets_linear=2, buckets_angular=2)

        with pytest.raises(ValueError, match="continuous sub-spaces only"):
            CompositeActionSpace(action_spaces=[drive])

    def test_discrete_sub_spaces_are_refused_inside_nested_composites(self):
        drive = DifferentialDriveActionSpace().to_discrete(buckets_linear=2, buckets_angular=2)

        with pytest.raises(ValueError, match="continuous sub-spaces only"):
            CompositeActionSpace.model_validate(
                {"action_spaces": [{"type": "composite", "action_spaces": [drive.model_dump()]}]}
            )

    def test_decode_split_pairs_each_sub_space_with_its_slice(self):
        drive, humanoid = self._parts()
        spec = CompositeActionSpace(action_spaces=[drive, humanoid])

        split = spec.decode_split(np.array([0.5, -1.0, 0.1, 0.2, 0.3]))

        assert [s for s, _ in split] == [drive, humanoid]
        assert split[0][0] is drive
        np.testing.assert_allclose(split[0][1], [0.5, 0.0, -1.0])
        np.testing.assert_allclose(split[1][1], [0.1, 0.2, 0.3])

    def test_decode_split_rejects_a_wrong_action_length(self):
        spec = CompositeActionSpace(action_spaces=list(self._parts()))

        with pytest.raises(ValueError, match="expected 5"):
            spec.decode_split(np.zeros(4))

    def test_a_plain_spec_splits_into_itself(self):
        drive, _ = self._parts()

        split = ActionSpaceManager(drive).decode_split_action(np.array([0.5, -1.0]))

        assert len(split) == 1
        assert split[0][0] is drive
        np.testing.assert_allclose(split[0][1], [0.5, 0.0, -1.0])

    def test_the_manager_splits_a_composite_by_sub_space(self):
        drive, humanoid = self._parts()

        split = ActionSpaceManager(CompositeActionSpace(action_spaces=[drive, humanoid])).decode_split_action(
            np.array([0.5, -1.0, 0.1, 0.2, 0.3])
        )

        assert [s for s, _ in split] == [drive, humanoid]


# ============================================================
# ActionSpaceSpec discriminated union
# ============================================================


class TestDiscriminatedUnion:
    @pytest.mark.parametrize(
        "data, expected_type",
        [
            ({"type": "differential_drive"}, DifferentialDriveActionSpace),
            ({"type": "omnidirectional"}, OmnidirectionalActionSpace),
            (
                {
                    "type": "arm",
                    "joint_names": ["joint_0"],
                    "position_limits": [(-1, 1)],
                    "controller": "arm_controller",
                    "tip_link": "tool0",
                },
                ManipulatorActionSpace,
            ),
            ({"type": "humanoid"}, HumanoidActionSpace),
        ],
    )
    def test_discriminator(self, data, expected_type):
        from pydantic import TypeAdapter

        ta = TypeAdapter(ActionSpaceSpec)
        spec = ta.validate_python(data)
        assert isinstance(spec, expected_type)


# ============================================================
# AgentParameters
# ============================================================


class TestAgentParameters:
    def test_defaults(self):
        params = AgentParameters()
        assert params.laser_num_beams == 720
        assert params.normalize is True
        assert params.normalizer == "max_abs"
        assert params.robot_radius == 0.3
        assert params.max_steps == 600

    def test_frozen(self):
        params = AgentParameters()
        with pytest.raises(Exception):
            params.laser_num_beams = 100

    def test_observation_kwargs_excludes_env_fields(self):
        params = AgentParameters()
        kwargs = params.observation_kwargs()
        # observation fields are included
        assert "laser_num_beams" in kwargs
        assert "normalize" in kwargs
        assert "normalizer" in kwargs
        # reward fields are excluded
        assert "robot_radius" not in kwargs
        assert "safety_distance" not in kwargs
        assert "goal_radius" not in kwargs
        assert "max_steps" not in kwargs

    def test_normalizer_custom(self):
        params = AgentParameters(normalizer="min_max")
        kwargs = params.observation_kwargs()
        assert kwargs["normalizer"] == "min_max"

    def test_custom_values_round_trip(self):
        params = AgentParameters(laser_num_beams=360, robot_radius=0.15, max_steps=300)
        assert params.laser_num_beams == 360
        assert params.robot_radius == 0.15
        assert params.max_steps == 300


# ============================================================
# ActionSpaceManager with typed specs
# ============================================================


class TestActionSpaceManagerSpec:
    def test_from_spec_continuous(self):
        spec = DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0))
        mgr = ActionSpaceManager(spec=spec)
        assert isinstance(mgr.action_space, spaces.Box)
        cmd = mgr.decode_action(np.array([0.3, -0.5]))
        np.testing.assert_array_almost_equal(cmd, [0.3, 0.0, -0.5])

    def test_from_spec_discrete(self):
        actions = [
            {"name": "fwd", "linear": 0.3, "angular": 0.0},
            {"name": "spin", "linear": 0.0, "angular": 0.5},
        ]
        spec = DifferentialDriveActionSpace(discrete_actions=actions)
        mgr = ActionSpaceManager(spec=spec)
        assert isinstance(mgr.action_space, spaces.Discrete)
        cmd = mgr.decode_action(np.array([0]))
        np.testing.assert_array_almost_equal(cmd, [0.3, 0.0, 0.0])

    def test_config_is_serializable(self):
        spec = DifferentialDriveActionSpace()
        mgr = ActionSpaceManager(spec=spec)
        config = mgr.config
        assert isinstance(config, dict)
        assert "type" in config


# ============================================================
# AgentConfig
# ============================================================


class TestAgentConfig:
    def _make_spec(self, **overrides):
        defaults = dict(
            name="test_agent",
            robot="turtlebot3",
            action_space=DifferentialDriveActionSpace(linear_range=(0.0, 0.22), angular_range=(-2.84, 2.84)),
            parameters=AgentParameters(laser_num_beams=360, laser_max_range=3.5),
            framework=_make_sb3_framework(),
        )
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def test_basic_construction(self):
        spec = self._make_spec()
        assert spec.name == "test_agent"
        assert spec.robot == "turtlebot3"
        assert isinstance(spec.action_space, DifferentialDriveActionSpace)

    def test_auto_name(self):
        spec = self._make_spec(name=None)
        assert spec.name is not None
        assert len(spec.name) > 0

    def test_to_dict_round_trip(self):
        spec = self._make_spec()
        d = spec.to_dict()
        spec2 = AgentConfig.from_dict(d)
        assert spec2.name == spec.name
        assert spec2.robot == spec.robot
        assert isinstance(spec2.action_space, DifferentialDriveActionSpace)
        assert spec2.parameters.laser_num_beams == 360

    def test_yaml_round_trip(self, tmp_path):
        spec = self._make_spec()
        p = tmp_path / "agent.yaml"
        spec.to_yaml(p)
        spec2 = AgentConfig.from_yaml(p)
        assert spec2.name == spec.name
        assert spec2.robot == spec.robot
        assert isinstance(spec2.action_space, DifferentialDriveActionSpace)
        assert spec2.action_space.linear_range == (0.0, 0.22)

    def test_parameters_defaults(self):
        spec = self._make_spec()
        assert spec.parameters.robot_radius == 0.3
        assert spec.parameters.safety_distance == 1.0
        assert spec.parameters.goal_radius == 0.5
        assert spec.parameters.max_steps == 600

    def test_parameters_custom(self):
        params = AgentParameters(robot_radius=0.15, safety_distance=0.5, goal_radius=0.3, max_steps=300)
        spec = self._make_spec(parameters=params)
        assert spec.parameters.robot_radius == 0.15
        assert spec.parameters.max_steps == 300

    def test_parameters_frozen(self):
        params = AgentParameters()
        with pytest.raises(Exception):
            params.robot_radius = 0.5

    def test_parameters_yaml_round_trip(self, tmp_path):
        params = AgentParameters(robot_radius=0.2, safety_distance=0.8, goal_radius=0.4, max_steps=400)
        spec = self._make_spec(parameters=params)
        p = tmp_path / "agent_params.yaml"
        spec.to_yaml(p)
        spec2 = AgentConfig.from_yaml(p)
        assert spec2.parameters.robot_radius == 0.2
        assert spec2.parameters.safety_distance == 0.8
        assert spec2.parameters.goal_radius == 0.4
        assert spec2.parameters.max_steps == 400

    def test_is_holonomic_diff_drive(self):
        spec = self._make_spec()
        assert not spec.action_space.is_holonomic

    def test_is_holonomic_omni(self):
        spec = self._make_spec(action_space=OmnidirectionalActionSpace())
        assert spec.action_space.is_holonomic

    def test_is_discrete_continuous(self):
        spec = self._make_spec()
        assert not spec.action_space.is_discrete

    def test_is_discrete_with_actions(self):
        actions = [{"linear": 0.3, "angular": 0.0}]
        spec = self._make_spec(action_space=DifferentialDriveActionSpace(discrete_actions=actions))
        assert spec.action_space.is_discrete


def test_inference_acts_on_the_mean_action_unless_the_saved_config_asks_to_sample():
    sampled = AgentConfig(framework=_make_sb3_framework(), deterministic_inference=False)

    assert AgentConfig(framework=_make_sb3_framework()).deterministic_inference is True
    assert AgentConfig.model_validate(sampled.model_dump()).deterministic_inference is False
