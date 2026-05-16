"""
Subscription Manager for ROS2 observation collectors.

Handles the setup and management of ROS subscriptions for different types of collectors,
including individual subscriptions and synchronized message filter subscriptions.
"""

from __future__ import annotations

import threading
from functools import partial
from typing import Dict, List

import message_filters
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription

from rosnav_rl.utils.rostopic import Namespace
from ..data_sources.base import Collector


class SubscriptionManager:
    """Manages ROS2 subscriptions for observation collectors."""

    def __init__(
        self,
        node: Node,
        ns: Namespace,
        enable_synchronization: bool = True,
        sync_tolerance_seconds: float = 0.1,
        buffer_size: int = 50,
        qos_profile: QoSProfile = 10,
    ):
        """
        Initialize the subscription manager.

        Args:
            node: ROS2 node for creating subscriptions
            ns: Namespace for ROS topics
            enable_synchronization: Whether to enable temporal synchronization
            sync_tolerance_seconds: Tolerance for synchronization timestamps
            buffer_size: Size of temporal buffer for each observation stream
            qos_profile: Default QoS profile for subscribers
        """
        self._node = node
        self._logger = node.get_logger()
        self._ns = ns
        self._enable_synchronization = enable_synchronization
        self._sync_tolerance = sync_tolerance_seconds
        self._buffer_size = buffer_size
        self._qos_profile = qos_profile
        self._lock = threading.Lock()

        # Subscription tracking
        self._subscribers: Dict[str, Subscription] = {}
        self._message_filter_synchronizer = None
        self._sync_collector_names: List[str] = []

    def setup_collectors(
        self,
        collectors: Dict[str, Collector],
        observation_callback,
        synchronized_callback,
    ) -> None:
        """
        Set up ROS2 subscribers for all collectors.

        Args:
            collectors: Dictionary of collectors to set up subscriptions for
            observation_callback: Callback for individual collector messages
            synchronized_callback: Callback for synchronized messages
        """
        # Determine which collectors need synchronization
        sync_collectors = {}
        if self._enable_synchronization:
            sync_collectors = {
                name: collector
                for name, collector in collectors.items()
                if collector.up_to_date_required
            }

        self._sync_collector_names = list(sync_collectors.keys())
        non_sync_collector_names = [
            name for name in collectors if name not in self._sync_collector_names
        ]

        # Setup for non-synchronized collectors
        for collector_name in non_sync_collector_names:
            collector = collectors[collector_name]
            self._setup_individual_collector(
                collector_name, collector, observation_callback
            )

        # Setup for synchronized collectors using message_filters
        if sync_collectors:
            self._setup_synchronized_collectors(sync_collectors, synchronized_callback)

    def _setup_individual_collector(
        self, name: str, collector: Collector, observation_callback
    ) -> None:
        """Set up a single collector with its own subscription."""
        # Set QoS profile if the collector doesn't have one
        if collector.qos_profile is None:
            collector.set_qos_profile(self._qos_profile)

        callback = partial(observation_callback, collector=collector)

        topic = (
            str(self._ns(collector.topic))
            if collector.topic[0] != "/"
            else collector.topic
        )

        self._subscribers[name] = self._node.create_subscription(
            collector.message_type,
            topic,
            callback,
            qos_profile=collector.qos_profile,
        )
        self._logger.debug(f"Subscribed to {topic} for collector '{name}'")

    def _setup_synchronized_collectors(
        self, sync_collectors: Dict[str, Collector], synchronized_callback
    ) -> None:
        """Set up synchronized collectors using message_filters."""
        subscribers = []

        for name, collector in sync_collectors.items():
            if collector.qos_profile is None:
                collector.set_qos_profile(self._qos_profile)

            topic = (
                str(self._ns(collector.topic))
                if collector.topic[0] != "/"
                else collector.topic
            )

            subscriber = message_filters.Subscriber(
                self._node,
                collector.message_type,
                topic,
                qos_profile=collector.qos_profile,
            )
            subscribers.append(subscriber)
            self._logger.debug(f"Preparing {topic} for synchronized collector '{name}'")

        self._message_filter_synchronizer = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=self._buffer_size,
            slop=self._sync_tolerance,
        )
        self._message_filter_synchronizer.registerCallback(synchronized_callback)
        self._logger.info(f"Using message_filters for: {self._sync_collector_names}")

    def shutdown(self) -> None:
        """Clean up subscriptions and resources."""
        for name, subscription in self._subscribers.items():
            self._node.destroy_subscription(subscription)
        self._logger.info("SubscriptionManager shutdown complete")

    @property
    def sync_collector_names(self) -> List[str]:
        """Get the names of synchronized collectors."""
        return self._sync_collector_names.copy()

    @property
    def lock(self) -> threading.Lock:
        """Get the thread lock for synchronized operations."""
        return self._lock
