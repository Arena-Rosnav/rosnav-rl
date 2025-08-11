"""
Waiting Strategy for ROS2 observation collectors.

Handles efficient waiting and polling for collector updates using ROS simulation time.
"""

from __future__ import annotations

import time
from typing import List

from rclpy.duration import Duration
from rclpy.node import Node

from ..data_sources.base import Collector


class WaitingStrategy:
    """Handles waiting for collector updates with simulation time awareness."""

    def __init__(self, node: Node):
        """
        Initialize the waiting strategy.

        Args:
            node: ROS2 node for accessing simulation time
        """
        self._node = node
        self._logger = node.get_logger()

    def wait_for_collectors(self, collectors: List[str], collector_dict: dict) -> int:
        """
        Wait for multiple collectors to update.

        Args:
            collectors: List of collector names to wait for
            collector_dict: Dictionary mapping collector names to Collector instances

        Returns:
            Number of collectors that successfully updated
        """
        if not collectors:
            return 0

        self._logger.debug(f"Waiting for updates from: {collectors}")

        # Sequential waiting with intelligent timeout management
        success_count = 0
        for name in collectors:
            if self._wait_for_observation(name, collector_dict):
                success_count += 1
            # Continue trying other collectors even if one times out

        self._logger.debug(
            f"Successfully updated {success_count}/{len(collectors)} stale collectors"
        )
        return success_count

    def _wait_for_observation(self, collector_name: str, collector_dict: dict) -> bool:
        """
        Wait for a specific collector to receive new data using ROS simulation time.

        Args:
            collector_name: Name of the collector to wait for
            collector_dict: Dictionary mapping collector names to Collector instances

        Returns:
            True if collector updated successfully, False if timeout or error
        """
        if collector_name not in collector_dict:
            self._logger.error(f"Cannot wait for unknown collector '{collector_name}'")
            return False

        collector = collector_dict[collector_name]
        timeout = getattr(collector, "timeout", 0.1)

        try:
            self._poll_for_update_with_timeout(collector, timeout)
            self._logger.debug(f"Collector '{collector_name}' updated successfully")
            return True
        except TimeoutError:
            self._logger.warn(
                f"Timeout waiting for '{collector_name}' after {timeout}s"
            )
            return False
        except Exception as e:
            self._logger.error(f"Exception while waiting for '{collector_name}': {e}")
            return False

    def _poll_for_update_with_timeout(
        self, collector: Collector, timeout: float
    ) -> None:
        """
        Efficiently poll for a collector update with timeout using ROS simulation time.

        Args:
            collector: The collector to poll for updates
            timeout: Maximum time to wait in seconds

        Raises:
            TimeoutError: If the timeout is exceeded
        """
        # Use ROS node's clock to respect simulation time
        start_time = self._node.get_clock().now()
        timeout_duration = Duration(seconds=timeout)

        # Adaptive polling intervals - start fast, slow down over time
        sleep_interval = 0.001  # Start with 1ms
        max_sleep_interval = 0.01  # Cap at 10ms
        sleep_increase_factor = 1.5

        while collector.stale:
            current_time = self._node.get_clock().now()
            elapsed = current_time - start_time

            if elapsed >= timeout_duration:
                raise TimeoutError(
                    f"Timeout after {timeout}s waiting for collector update"
                )

            # Adaptive sleep - start aggressive, become more conservative
            time.sleep(sleep_interval)
            sleep_interval = min(
                sleep_interval * sleep_increase_factor, max_sleep_interval
            )
