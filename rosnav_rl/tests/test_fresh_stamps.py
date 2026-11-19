"""The fresh-data streams of an observation manager, with their resolved topics and latest stamps, are what the lockstep gate registers."""

import threading
import time

import builtin_interfaces.msg
import geometry_msgs.msg
import pytest
import rclpy
import sensor_msgs.msg
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.core.manager import ObservationManager

NAMESPACE = "fresh_stamps_test"
CONFIG = {
    "aliases": {},
    "datasources": {
        "front_laser": {"type": "sensor_msgs/LaserScan", "params": {"topic": "lidar", "up_to_date_required": True}},
        "goal_pose": {
            "type": "geometry_msgs/PoseStamped",
            "params": {"topic": "goal_pose", "up_to_date_required": False},
        },
    },
}


@pytest.fixture
def manager():
    if not rclpy.ok():
        rclpy.init()
    node = Node("fresh_stamps_test")
    manager = ObservationManager.from_config(
        config=CONFIG, node=node, ns=NAMESPACE, simulation_state_container=AgentParameters(), wait_for_obs=False
    )
    publisher = node.create_publisher(sensor_msgs.msg.LaserScan, f"/{NAMESPACE}/lidar", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    deadline = time.monotonic() + 30.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    yield manager, publisher
    executor.shutdown()
    node.destroy_node()


def test_fresh_stamps_lists_stamped_fresh_streams_with_their_latest_stamp(manager):
    observations, publisher = manager
    assert observations.fresh_stamps() == {f"/{NAMESPACE}/lidar": (sensor_msgs.msg.LaserScan, None)}
    scan = sensor_msgs.msg.LaserScan(ranges=[1.0, 2.0])
    scan.header.stamp = builtin_interfaces.msg.Time(sec=12, nanosec=300_000_000)
    publisher.publish(scan)
    deadline = time.monotonic() + 10.0
    while observations.fresh_stamps()[f"/{NAMESPACE}/lidar"][1] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    msg_type, stamp = observations.fresh_stamps()[f"/{NAMESPACE}/lidar"]
    assert msg_type is sensor_msgs.msg.LaserScan
    assert stamp is not None and stamp.nanoseconds == 12_300_000_000
    assert geometry_msgs.msg.PoseStamped not in {msg_type for msg_type, _ in observations.fresh_stamps().values()}
