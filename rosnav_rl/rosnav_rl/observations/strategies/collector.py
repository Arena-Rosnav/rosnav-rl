"""
Collection Strategy for ROS2 observation collectors.

Handles the logic for collecting observations from collectors and determining
which ones need fresh data updates.
"""

from __future__ import annotations

from typing import Any, Dict

from rclpy.node import Node

from ..data_sources.base import Collector
from .waiting import WaitingStrategy


class CollectorManager:
    """Handles observation collection logic for collectors."""

    def __init__(self, node: Node, wait_for_obs: bool = True):
        """
        Initialize the collection strategy.

        Args:
            node: ROS2 node for logging
            wait_for_obs: Whether to wait for stale observations
        """
        self._node = node
        self._logger = node.get_logger()
        self._wait_for_obs = wait_for_obs
        self._waiting_strategy = WaitingStrategy(node)

    def collect_observations(
        self, collectors: Dict[str, Collector], obs_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect observations from all collectors, waiting for required updates.

        Args:
            collectors: Dictionary of collectors to collect from
            obs_dict: Existing observation dictionary to update

        Returns:
            Updated observation dictionary with collector data
        """
        self._logger.debug(f"Collecting observations from {len(collectors)} collectors")

        # Check which collectors need fresh data
        stale_collectors = []
        for name, collector in collectors.items():
            if collector.stale and collector.up_to_date_required:
                self._logger.debug(
                    f"Collector '{name}' has been stale for {collector.age:.2f} seconds."
                )
                if self._wait_for_obs:
                    stale_collectors.append(name)
            # Always get the current value (stale or not)
            obs_dict[name] = collector.get_observation()

        # Wait for stale collectors if needed
        if stale_collectors:
            self._waiting_strategy.wait_for_collectors(stale_collectors, collectors)

            # Update observation values after waiting
            for name in stale_collectors:
                obs_dict[name] = collectors[name].get_observation()

        # Mark all collectors as stale for next collection cycle
        self._invalidate_observations(collectors)

        return obs_dict

    def _invalidate_observations(self, collectors: Dict[str, Collector]) -> None:
        """Mark all collectors as stale to ensure fresh data on next collection."""
        for collector in collectors.values():
            collector.stale = True
