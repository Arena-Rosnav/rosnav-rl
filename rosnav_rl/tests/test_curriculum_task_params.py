"""A curriculum stage sets task.episode node parameters on the task_generator under their own name, and queues its modes and task params, under the namespace the env reports for each mode, through QueueEpisode."""

import threading

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from diagnostic_msgs.msg import KeyValue
from arena_runtime_msgs.srv import QueryTaskModes, QueueEpisode

from rosnav_rl.utils.stable_baselines3.staged_train_callback import StagedTrainCallback
from rosnav_rl.utils.stable_baselines3.tick_callbacks import TickScheduledCurriculum

TOLERANCE = "task.episode.goto_pose.tolerance.radius"
QUEUE_SERVICE = "/curriculum_test/env_0/arena_env_node/config/queue_episode"
QUERY_SERVICE = "/curriculum_test/env_0/arena_env_node/query/task_modes"


@pytest.fixture
def task_generator():
    if not rclpy.ok():
        rclpy.init()
    target = Node("arena_env_node", namespace="/curriculum_test/env_0", allow_undeclared_parameters=True)
    target.declare_parameter(TOLERANCE, 1.0)
    queued: list[QueueEpisode.Request] = []

    def queue_episode(request: QueueEpisode.Request, response: QueueEpisode.Response) -> QueueEpisode.Response:
        queued.append(request)
        response.success = True
        return response

    def query_task_modes(request: QueryTaskModes.Request, response: QueryTaskModes.Response) -> QueryTaskModes.Response:
        response.namespaces = [KeyValue(key="obstacles/random", value="task.rnd")]
        return response

    target.create_service(QueueEpisode, QUEUE_SERVICE, queue_episode)
    target.create_service(QueryTaskModes, QUERY_SERVICE, query_task_modes)
    trainer = Node("curriculum_test_trainer")
    executor = MultiThreadedExecutor()
    executor.add_node(target)
    executor.add_node(trainer)
    threading.Thread(target=executor.spin, daemon=True).start()
    yield target, trainer, queued
    executor.shutdown()
    trainer.destroy_node()
    target.destroy_node()


def test_stage_zero_sets_the_episode_tolerance_on_the_task_generator(task_generator):
    target, trainer, _ = task_generator
    stage = {"task.random.static.n": [0, 0], TOLERANCE: 1.5}
    StagedTrainCallback(
        node=trainer,
        train_stages=[stage, {**stage, TOLERANCE: 1.2}],
        threshold_type="succ",
        upper_threshold=0.9,
        lower_threshold=0.3,
        num_envs=1,
        parameter_node_template="/curriculum_test/env_{i}/arena_env_node",
        timeout=30.0,
    )
    assert target.get_parameter(TOLERANCE).value == pytest.approx(1.5)


def test_tick_scheduled_stage_zero_sets_the_episode_tolerance_on_the_task_generator(task_generator):
    target, trainer, _ = task_generator
    stage = {"task.random.static.n": [0, 0], TOLERANCE: 1.5}
    TickScheduledCurriculum(
        node=trainer,
        train_stages=[stage, {**stage, TOLERANCE: 1.2}],
        stage_ticks=[0, 50000],
        ticks_per_step=1,
        num_envs=1,
        parameter_node_template="/curriculum_test/env_{i}/arena_env_node",
        timeout=30.0,
    )
    assert target.get_parameter(TOLERANCE).value == pytest.approx(1.5)


def test_stage_modes_and_task_params_reach_queue_episode_as_modes_and_namespaced_names(task_generator):
    _, trainer, queued = task_generator
    StagedTrainCallback(
        node=trainer,
        train_stages=[{"task.random.static.n": [0, 2], TOLERANCE: 1.5}],
        threshold_type="succ",
        upper_threshold=0.9,
        lower_threshold=0.3,
        num_envs=1,
        parameter_node_template="/curriculum_test/env_{i}/arena_env_node",
        timeout=30.0,
        modes={"robots": "random", "modules": ["staged", "sounds"]},
    )
    assert [(pair.key, pair.value) for pair in queued[0].modes] == [("robots", "random"), ("modules", "staged,sounds")]
    assert [(param.name, list(param.value.integer_array_value)) for param in queued[1].params] == [
        ("task.rnd.static.n", [0, 2])
    ]
    assert not queued[1].modes
