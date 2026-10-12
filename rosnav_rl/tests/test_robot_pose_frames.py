"""Tests for pointing RobotPoseTFGenerator datasources at a robot's frames."""

from __future__ import annotations

from rosnav_rl.observations.factory.factory import set_robot_pose_frames


def _config() -> dict:
    return {
        "datasources": {
            "robot_pose_from_tf": {"type": "RobotPoseTFGenerator", "params": {}},
            "front_laser": {"type": "sensor_msgs/LaserScan", "params": {"topic": "lidar"}},
        }
    }


def test_tf_prefix_with_trailing_slash_resolves_base_and_odom():
    config = _config()

    set_robot_pose_frames(config, "env_0/jackal/", "base_link")

    params = config["datasources"]["robot_pose_from_tf"]["params"]
    assert params == {"source_frame": "env_0/jackal/base_link", "target_frame": "env_0/jackal/odom"}


def test_split_training_source_frame_round_trips():
    config = _config()

    set_robot_pose_frames(config, "env_0/jackal", "base_link")

    assert config["datasources"]["robot_pose_from_tf"]["params"]["source_frame"] == "env_0/jackal/base_link"


def test_generator_without_params_block_gets_frames():
    config = {"datasources": {"pose": {"type": "RobotPoseTFGenerator"}}}

    set_robot_pose_frames(config, "env_1/burger/", "base_footprint")

    assert config["datasources"]["pose"]["params"]["source_frame"] == "env_1/burger/base_footprint"


def test_other_datasources_are_untouched():
    config = _config()

    set_robot_pose_frames(config, "env_0/jackal/", "base_link")

    assert config["datasources"]["front_laser"]["params"] == {"topic": "lidar"}
