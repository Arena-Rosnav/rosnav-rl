"""
Waiting Strategy for ROS2 observation collectors.

Handles efficient waiting and polling for collector updates using ROS simulation time.
"""

from __future__ import annotations

import time
from typing import List

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
        Poll for a collector update using wall-clock time for both sleeping and
        timeout tracking.

        Using wall-clock time (time.monotonic) consistently avoids the mismatch
        that occurred when time.sleep() advanced wall-clock time but the timeout
        was measured against ROS simulation time — which can be paused, slow, or
        ahead of real time. Wall-clock is the correct measure here because we are
        waiting for a real network message to arrive.

        This method must not be called from inside a ROS executor callback because
        time.sleep() will block the executor thread. Call it only from the
        training-loop / step thread.

        Args:
            collector: The collector to poll for updates
            timeout: Maximum wall-clock seconds to wait

        Raises:
            TimeoutError: If the timeout is exceeded
        """
        start = time.monotonic()

        # Adaptive polling intervals - start fast, slow down over time
        sleep_interval = 0.001  # Start with 1ms
        max_sleep_interval = 0.01  # Cap at 10ms
        sleep_increase_factor = 1.5

        while collector.stale:
            elapsed = time.monotonic() - start

            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timeout after {timeout}s waiting for collector update"
                )

            # Yield briefly so the OS can deliver incoming messages
            time.sleep(sleep_interval)
            sleep_interval = min(
                sleep_interval * sleep_increase_factor, max_sleep_interval
            )
