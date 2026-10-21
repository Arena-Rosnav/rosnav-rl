"""Under lockstep the inference node holds a hard channel at its control period and beats once per tick."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import rclpy
import yaml
from arena_runtime_msgs.msg import LockstepHeartbeat, LockstepRegistration
from arena_runtime_msgs.srv import LockstepRegister
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.inference.arena_inference_node import ArenaInferenceNode
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env

ENV = "/beat_test_env"
ROBOT_NS = f"{ENV}/jackal"
CONTROL_RATE = 20.0


def _saved_agent(directory: Path) -> Path:
    spec = AgentConfig(
        name="beat_agent",
        action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.5, 1.5)),
        parameters=AgentParameters(laser_num_beams=360, robot_radius=0.4),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_5", parameters=PPO_Algorithm_Cfg(device="cpu", seed=3))
        ),
        normalized_actions=True,
    )
    agent = RL_Agent(spec)
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    agent.model.model.save(directory / "best_model")
    (directory / "training_config.yaml").write_text(
        yaml.safe_dump({"arena_cfg": {}, "agent_config": spec.model_dump(mode="json")})
    )
    return directory


def _await(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.02)


@pytest.fixture
def lockstep(tmp_path: Path):
    if not rclpy.ok():
        rclpy.init()
    registrations: list[LockstepRegistration] = []
    heartbeats: list[LockstepHeartbeat] = []

    def register(request: LockstepRegister.Request, response: LockstepRegister.Response) -> LockstepRegister.Response:
        registrations.append(request.registration)
        response.success = True
        return response

    runtime = Node("beat_test_runtime")
    runtime.create_service(LockstepRegister, "/arena/sim_lifecycle/lockstep/register", register)
    runtime.create_subscription(LockstepHeartbeat, f"{ROBOT_NS}/lockstep/rosnav", heartbeats.append, 50)
    node = Node(
        "rosnav_rl_inference",
        namespace=ROBOT_NS,
        parameter_overrides=[
            Parameter("agent", value=str(_saved_agent(tmp_path))),
            Parameter("task_generator_node", value=f"{ENV}/task_generator_node"),
            Parameter("control_rate", value=CONTROL_RATE),
        ],
    )
    inference = ArenaInferenceNode(node)
    executor = MultiThreadedExecutor()
    executor.add_node(runtime)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    yield inference, executor, registrations, heartbeats
    executor.shutdown()
    node.destroy_node()
    runtime.destroy_node()


def _run(executor: MultiThreadedExecutor, coroutine) -> None:
    task = executor.create_task(coroutine)
    _await(task.done)
    task.result()


def test_inference_holds_a_hard_channel_at_its_control_period_and_beats_every_tick(lockstep):
    inference, executor, registrations, heartbeats = lockstep

    _run(executor, inference.start())

    assert len(registrations) == 1
    registration = registrations[0]
    assert (registration.env, registration.caller) == (ENV, f"{ROBOT_NS}/lockstep/rosnav")
    [channel] = registration.channels
    assert (channel.name, channel.topic, channel.hard) == ("rosnav/jackal", f"{ROBOT_NS}/lockstep/rosnav", True)
    assert channel.period_s == pytest.approx(1.0 / CONTROL_RATE)

    count = len(heartbeats)
    _await(lambda: len(heartbeats) >= count + 5)
    stamps = [hb.header.stamp.sec + hb.header.stamp.nanosec * 1e-9 for hb in heartbeats[count:]]
    assert stamps == sorted(stamps)


def test_inference_releases_its_channel_on_stop(lockstep):
    inference, executor, registrations, _heartbeats = lockstep

    _run(executor, inference.start())
    _run(executor, inference.stop())

    assert [len(r.channels) for r in registrations] == [1, 0]
