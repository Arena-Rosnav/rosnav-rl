from __future__ import annotations

import pytest
import rclpy
from rclpy.node import Node

from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase


class _Curriculum(CurriculumBase):
    def get_current_performance(self):
        return None

    def reset_performance_tracking(self):
        pass


@pytest.fixture(scope="module")
def node():
    if not rclpy.ok():
        rclpy.init()
    n = Node("curriculum_default_template_test")
    yield n
    n.destroy_node()


@pytest.mark.parametrize(
    ("num_envs", "expected"),
    [
        (1, ["/arena/env_0/arena_env_node"]),
        (2, ["/arena/env_0/arena_env_node", "/arena/env_1/arena_env_node"]),
    ],
)
def test_default_parameter_node_template_targets_the_registry_nodes(node, num_envs, expected):
    curriculum = _Curriculum(node, [{}], "succ", 0.9, 0.3, num_envs=num_envs)
    assert list(curriculum.parameter_clients) == expected
    assert list(curriculum.task_mode_clients) == expected
