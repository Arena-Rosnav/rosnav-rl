"""Sensor stall detection of the observation collector: only consecutive missed updates count."""

import pytest
import rclpy
import sensor_msgs.msg
from rclpy.node import Node

from rosnav_rl.observations.data_sources.collectors import LaserScanCollector
from rosnav_rl.observations.strategies.collector import CollectorManager
from rosnav_rl.observations.strategies.waiting import WaitingStrategy


@pytest.fixture
def node():
    if not rclpy.ok():
        rclpy.init()
    node = Node("sensor_stall_test")
    yield node
    node.destroy_node()


def _laser() -> LaserScanCollector:
    laser = LaserScanCollector("front_laser", topic="scan", up_to_date_required=True)
    laser.timeout = 0.01
    return laser


def _scan() -> sensor_msgs.msg.LaserScan:
    return sensor_msgs.msg.LaserScan(ranges=[1.0, 2.0], range_max=5.0)


def test_fresh_collections_between_missed_updates_reset_the_stall_count(node):
    manager = CollectorManager(node)
    laser = _laser()
    for _ in range(3 * WaitingStrategy._STALL_THRESHOLD):
        manager.collect_observations({"front_laser": laser}, {})
        laser.update(_scan())
        manager.collect_observations({"front_laser": laser}, {})


def test_consecutive_missed_updates_raise_a_stall(node):
    manager = CollectorManager(node)
    laser = _laser()
    for _ in range(WaitingStrategy._STALL_THRESHOLD - 1):
        manager.collect_observations({"front_laser": laser}, {})
    with pytest.raises(RuntimeError, match="Sensor stall: 'front_laser' timed out 10 consecutive times"):
        manager.collect_observations({"front_laser": laser}, {})
