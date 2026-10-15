"""In training mode the subgoal lookahead follows the speed the training loop commands on cmd_vel."""

import threading
import time

import geometry_msgs.msg
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from rosnav_rl.inference.arena_inference_node import ArenaInferenceNode

NAMESPACE = "lookahead_test"


@pytest.fixture
def stack():
    if not rclpy.ok():
        rclpy.init()
    node = Node("rosnav_rl_inference", namespace=NAMESPACE, parameter_overrides=[Parameter("train_mode", value=True)])
    ArenaInferenceNode(node)
    driver = Node("lookahead_driver", namespace=NAMESPACE)
    pose = geometry_msgs.msg.TransformStamped()
    pose.header.frame_id = "map"
    pose.child_frame_id = "base_link"
    pose.transform.rotation.w = 1.0
    broadcaster = StaticTransformBroadcaster(driver)
    broadcaster.sendTransform(pose)
    subgoals: list[geometry_msgs.msg.PoseStamped] = []
    driver.create_subscription(geometry_msgs.msg.PoseStamped, "subgoal", subgoals.append, 10)
    goal_pub = driver.create_publisher(geometry_msgs.msg.PoseStamped, "goal_pose", 1)
    cmd_vel_pub = driver.create_publisher(geometry_msgs.msg.Twist, "cmd_vel", 1)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(driver)
    threading.Thread(target=executor.spin, daemon=True).start()
    yield subgoals, goal_pub, cmd_vel_pub
    executor.shutdown()
    driver.destroy_node()
    node.destroy_node()


def _await(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_lookahead_grows_with_the_commanded_speed_in_training_mode(stack):
    subgoals, goal_pub, cmd_vel_pub = stack
    goal = geometry_msgs.msg.PoseStamped()
    goal.header.frame_id = "map"
    goal.pose.position.x = 10.0
    goal.pose.orientation.w = 1.0
    _await(lambda: goal_pub.get_subscription_count() > 0 and cmd_vel_pub.get_subscription_count() > 0)

    goal_pub.publish(goal)
    _await(lambda: bool(subgoals))
    assert subgoals[-1].pose.position.x == pytest.approx(0.5)

    cmd_vel_pub.publish(geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=1.0)))
    _await(lambda: subgoals[-1].pose.position.x > 1.0)
    assert subgoals[-1].pose.position.x == pytest.approx(1.5)
    assert subgoals[-1].pose.position.y == pytest.approx(0.0)
