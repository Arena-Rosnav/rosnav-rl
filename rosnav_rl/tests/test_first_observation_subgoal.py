"""The first observation of an episode carries the subgoal of the new episode, through the bundled observation pipeline on a pose and goal recorded at an episode boundary of the flatsim training stack."""

import importlib.resources
import json
import math
import pathlib
import threading
import time

import geometry_msgs.msg
import numpy as np
import pytest
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.core.manager import ObservationManager
from rosnav_rl.observations.factory.factory import set_pedestrians_topic, set_robot_pose_frames

RECORDED = json.loads((pathlib.Path(__file__).parent / "data" / "first_observation_subgoal.json").read_text())
NAMESPACE = "first_subgoal_test/env_0/jackal"


@pytest.fixture
def pipeline():
    if not rclpy.ok():
        rclpy.init()
    node = Node("first_subgoal_test")
    config = yaml.safe_load((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())
    set_robot_pose_frames(config, "env_0/jackal", "base_link")
    set_pedestrians_topic(config, NAMESPACE)
    manager = ObservationManager.from_config(
        config=config, node=node, ns=NAMESPACE, simulation_state_container=AgentParameters(), wait_for_obs=False
    )

    x, y, yaw = RECORDED["robot_pose"]
    pose = geometry_msgs.msg.TransformStamped()
    pose.header.frame_id = "env_0/jackal/odom"
    pose.child_frame_id = "env_0/jackal/base_link"
    pose.transform.translation.x, pose.transform.translation.y = x, y
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    pose.transform.rotation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    StaticTransformBroadcaster(node).sendTransform(pose)

    goal_pub = node.create_publisher(geometry_msgs.msg.PoseStamped, f"/{NAMESPACE}/goal_pose", 1)
    cmd_vel_pub = node.create_publisher(geometry_msgs.msg.Twist, f"/{NAMESPACE}/cmd_vel", 1)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    yield manager, goal_pub, cmd_vel_pub
    executor.shutdown()
    node.destroy_node()


def _await(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_first_observation_subgoal_points_from_the_new_pose_at_the_new_goal(pipeline):
    manager, goal_pub, cmd_vel_pub = pipeline
    goal = geometry_msgs.msg.PoseStamped()
    goal.header.frame_id = "map"
    goal.pose.position.x, goal.pose.position.y = RECORDED["goal_pose"][:2]
    goal.pose.orientation.w = 1.0
    _await(lambda: goal_pub.get_subscription_count() > 0 and cmd_vel_pub.get_subscription_count() > 0)
    goal_pub.publish(goal)
    cmd_vel_pub.publish(geometry_msgs.msg.Twist())

    def observe() -> dict:
        return manager.get_observations(is_terminal=False, is_first=True)

    _await(
        lambda: (
            math.isclose(float(observe()["goal_pose"]["x"]), RECORDED["goal_pose"][0], abs_tol=1e-3)
            and math.isclose(float(observe()["robot_pose"]["x"]), RECORDED["robot_pose"][0], abs_tol=1e-3)
        )
    )
    obs = observe()

    robot, new_goal, previous = (
        np.asarray(RECORDED[key][:2]) for key in ("robot_pose", "goal_pose", "previous_subgoal")
    )
    subgoal = np.array([float(obs["subgoal_pose"]["x"]), float(obs["subgoal_pose"]["y"])])
    heading = (new_goal - robot) / np.linalg.norm(new_goal - robot)
    assert subgoal == pytest.approx(robot + 0.5 * heading, abs=1e-3)
    assert np.linalg.norm(subgoal - previous) > 1.0
    assert float(obs["dist_angle_to_subgoal"][0]) == pytest.approx(0.5, abs=1e-3)
