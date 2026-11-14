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
    n = Node("curriculum_env_nodes_test")
    yield n
    n.destroy_node()


@pytest.mark.parametrize(
    "env_nodes",
    [
        ["/arena/env_3/arena_env_node", "/arena/env_7/arena_env_node"],
        ["/arena/env_5/arena_env_node"],
    ],
)
def test_clients_target_exactly_the_given_env_nodes(node, env_nodes):
    curriculum = _Curriculum(node, [{}], "succ", 0.9, 0.3, env_nodes=env_nodes)
    assert list(curriculum.parameter_clients) == env_nodes
    assert list(curriculum.task_mode_clients) == env_nodes
    assert [c.srv_name for c in curriculum.parameter_clients.values()] == [f"{n}/set_parameters" for n in env_nodes]
    assert [c.srv_name for c in curriculum.task_mode_clients.values()] == [
        f"{n}/config/queue_episode" for n in env_nodes
    ]


def test_empty_env_nodes_rejected(node):
    with pytest.raises(ValueError):
        _Curriculum(node, [{}], "succ", 0.9, 0.3, env_nodes=[])
