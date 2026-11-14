"""The action server resets on its env's task_reset and plays a chunked policy out one command per call."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rosnav_rl_msgs.srv import GetCommand
from std_msgs.msg import Int16

from rosnav_rl.action_server.base_server import ActionServer
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Algorithm_Cfg, PPO_Cfg
from rosnav_rl.model.stable_baselines3.cfg.framework import StableBaselinesCfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env

BEAMS = 360
CHUNK = 3
ENV_NODE = "/reset_env/arena_env_node"


def _agent() -> RL_Agent:
    agent = RL_Agent(
        AgentConfig(
            name="reset_agent",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=BEAMS, laser_max_range=30.0, normalize=True),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(
                    architecture_name="AGENT_1",
                    parameters=PPO_Algorithm_Cfg(total_batch_size=64, batch_size=64, device="cpu"),
                )
            ),
            action_chunk=CHUNK,
            deterministic_inference=True,
        )
    )
    agent.initialize_model(
        make_mock_env(ns="", space_manager=agent.space_manager, stack_size=agent.model._policy_description.stack_size)
    )
    return agent


class _Observations:
    def get_observations(self) -> dict:
        return {
            "front_laser": np.full(BEAMS, 2.0, dtype=np.float32),
            "dist_angle_to_subgoal": np.array([3.0, 0.3], dtype=np.float32),
            "last_action": np.zeros(3, dtype=np.float32),
            "is_terminal": False,
        }


class _Server(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        return _agent()

    def _initialize_observation_collector(self) -> _Observations:
        return _Observations()


@pytest.fixture(scope="module")
def server():
    if not rclpy.ok():
        rclpy.init()
    server = _Server(agent_name="reset_agent", namespace="reset_ns", arena_env_node=ENV_NODE)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    yield server
    server.stop()
    thread.join(timeout=10.0)
    server.node.destroy_node()


@pytest.fixture(scope="module")
def client():
    node = rclpy.create_node("reset_client")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    yield node, executor
    executor.shutdown()
    node.destroy_node()


def _command(node, executor) -> list[float]:
    service = node.create_client(GetCommand, "/reset_ns/get_command")
    assert service.wait_for_service(timeout_sec=60.0)
    future = service.call_async(GetCommand.Request())
    executor.spin_until_future_complete(future, timeout_sec=30.0)
    return list(future.result().action)


def _reset(node, executor, server: _Server) -> None:
    publisher = node.create_publisher(Int16, f"{ENV_NODE}/task_reset", 1)
    deadline = time.monotonic() + 30.0
    while publisher.get_subscription_count() == 0:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    publisher.publish(Int16(data=0))
    while server._queued_commands:
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_a_chunked_policy_answers_every_call_with_a_full_command(server, client):
    node, executor = client

    commands = [_command(node, executor) for _ in range(CHUNK)]

    assert [len(command) for command in commands] == [3] * CHUNK


def test_task_reset_of_the_env_node_drops_the_queued_chunk_and_the_stacked_state(server, client):
    node, executor = client
    _reset(node, executor, server)
    first = _command(node, executor)
    _command(node, executor)

    _reset(node, executor, server)

    assert _command(node, executor) == pytest.approx(first)
