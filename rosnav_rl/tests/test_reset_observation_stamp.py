"""The first observation of an episode waits for a scan stamped at or after the episode start, on a scan pair recorded at an episode boundary of the flatsim training stack."""

import array
import json
import pathlib
import threading
import time

import builtin_interfaces.msg
import numpy as np
import pytest
import rclpy
import sensor_msgs.msg
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.core.manager import ObservationManager

RECORDED = json.loads((pathlib.Path(__file__).parent / "data" / "reset_boundary.json").read_text())
NAMESPACE = "reset_stamp_test"
CONFIG = {
    "aliases": {},
    "datasources": {
        "front_laser": {"type": "sensor_msgs/LaserScan", "params": {"topic": "lidar", "up_to_date_required": True}}
    },
}


def _scan(recorded: dict) -> sensor_msgs.msg.LaserScan:
    msg = sensor_msgs.msg.LaserScan(
        angle_min=recorded["angle_min"],
        angle_max=recorded["angle_max"],
        angle_increment=recorded["angle_increment"],
        range_min=recorded["range_min"],
        range_max=recorded["range_max"],
        ranges=recorded["ranges"],
    )
    msg.header.frame_id = recorded["frame_id"]
    msg.header.stamp = builtin_interfaces.msg.Time(sec=recorded["stamp"][0], nanosec=recorded["stamp"][1])
    return msg


@pytest.fixture
def pipeline():
    if not rclpy.ok():
        rclpy.init()
    node = Node("reset_stamp_test")
    manager = ObservationManager.from_config(
        config=CONFIG, node=node, ns=NAMESPACE, simulation_state_container=AgentParameters(), wait_for_obs=True
    )
    publisher = node.create_publisher(sensor_msgs.msg.LaserScan, f"/{NAMESPACE}/lidar", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    deadline = time.monotonic() + 10.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    yield manager, publisher
    executor.shutdown()
    node.destroy_node()


def test_first_observation_of_an_episode_waits_for_the_scan_of_the_new_pose(pipeline):
    manager, publisher = pipeline
    terminal, reset = _scan(RECORDED["terminal"]), _scan(RECORDED["reset"])
    start = Time.from_msg(builtin_interfaces.msg.Time(sec=RECORDED["start_time"][0], nanosec=RECORDED["start_time"][1]))

    publisher.publish(terminal)
    assert manager.wait_for_stamp(Time.from_msg(terminal.header.stamp), 5.0)
    assert not manager.wait_for_stamp(start, 0.3)

    waited: list[bool] = []
    waiter = threading.Thread(target=lambda: waited.append(manager.wait_for_stamp(start, 5.0)))
    waiter.start()
    time.sleep(0.1)
    assert not waited
    publisher.publish(reset)
    waiter.join()

    assert waited == [True]
    observed = np.asarray(manager.get_observations()["front_laser"])
    assert observed.min() == pytest.approx(np.nanmin(RECORDED["reset"]["ranges"]))
    assert observed.min() > np.nanmin(RECORDED["terminal"]["ranges"]) + 0.5


def test_a_reached_stamp_already_carries_its_scan(pipeline):
    manager, _publisher = pipeline
    collector = manager._collectors["front_laser"]
    collector.update(_scan(RECORDED["terminal"]))
    reset = _scan(RECORDED["reset"])
    reset.ranges = array.array("f", np.full(20_000_000, 2.0, np.float32).tobytes())

    updater = threading.Thread(target=collector.update, args=(reset,))
    updater.start()
    assert manager.wait_for_stamp(Time.from_msg(reset.header.stamp), 5.0)
    observed = np.asarray(manager.get_observations()["front_laser"])
    updater.join()

    assert len(observed) == len(reset.ranges)
