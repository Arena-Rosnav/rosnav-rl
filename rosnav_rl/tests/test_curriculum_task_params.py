"""A curriculum stage sets task.episode node parameters on the task_generator under their own name, next to the task-mode keys."""

import threading

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from rosnav_rl.utils.stable_baselines3.staged_train_callback import StagedTrainCallback

TOLERANCE = "task.episode.goto_pose.tolerance.radius"


@pytest.fixture
def task_generator():
    if not rclpy.ok():
        rclpy.init()
    target = Node("task_generator_node", namespace="/curriculum_test/env_0", allow_undeclared_parameters=True)
    target.declare_parameter(TOLERANCE, 1.0)
    trainer = Node("curriculum_test_trainer")
    executor = MultiThreadedExecutor()
    executor.add_node(target)
    executor.add_node(trainer)
    threading.Thread(target=executor.spin, daemon=True).start()
    yield target, trainer
    executor.shutdown()
    trainer.destroy_node()
    target.destroy_node()


def test_stage_zero_sets_the_episode_tolerance_on_the_task_generator(task_generator):
    target, trainer = task_generator
    stage = {"task.random.static.n": [0, 0], TOLERANCE: 1.5}
    StagedTrainCallback(
        node=trainer,
        train_stages=[stage, {**stage, TOLERANCE: 1.2}],
        threshold_type="succ",
        upper_threshold=0.9,
        lower_threshold=0.3,
        num_envs=1,
        parameter_node_template="/curriculum_test/env_{i}/task_generator_node",
        timeout=5.0,
    )
    assert target.get_parameter(TOLERANCE).value == pytest.approx(1.5)
