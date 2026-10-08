"""Tests for the rosnav_rl action server mechanisms.

Tests are grouped into three layers:

  1. Path-resolution tests  — verify _find_agents_dir() / _resolve_agent_dir()
  2. Service-mechanism tests — verify the GetCommand ROS2 service wiring using a
                               mock agent and observation collector
  3. Load-pipeline smoke test — verify that ArenaActionServer._initialize_agent()
                                can load a model saved by create_test_agent.py
                                (only runs when a test agent exists on disk)

Run with::

    pytest tests/test_action_server.py -v

Prerequisites (for smoke test):
    python3 scripts/create_test_agent.py   # creates agents/test_agent/
"""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _get_agents_dir() -> Path:
    """Use the same resolution logic as the action server itself."""
    from rosnav_rl.action_server.arena_server import _find_agents_dir

    return _find_agents_dir()


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401

        return True
    except ImportError:
        return False


def _ros_running() -> bool:
    """Return True only if a ROS2 context can be initialised."""
    if not _rclpy_available():
        return False
    try:
        import rclpy

        if not rclpy.ok():
            rclpy.init(args=["--ros-args"])
        return rclpy.ok()
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Path-resolution tests (no ROS2 needed)
# ─────────────────────────────────────────────────────────────────────────────


class TestFindAgentsDir:
    """Unit tests for _find_agents_dir() and _resolve_agent_dir()."""

    def test_env_var_override(self, tmp_path):
        """ROSNAV_AGENTS_DIR env var should be preferred over all other paths."""
        from rosnav_rl.action_server.arena_server import _find_agents_dir

        with patch.dict(os.environ, {"ROSNAV_AGENTS_DIR": str(tmp_path)}):
            result = _find_agents_dir()

        assert result == tmp_path

    def test_env_var_nonexistent_falls_back(self, tmp_path):
        """If the env-var path doesn't exist the function falls back to other
        strategies.  It should NOT return the nonexistent env-var path when
        another valid directory can be found."""
        from rosnav_rl.action_server.arena_server import _find_agents_dir

        fake_path = tmp_path / "nonexistent_agents"
        with patch.dict(os.environ, {"ROSNAV_AGENTS_DIR": str(fake_path)}):
            result = _find_agents_dir()

        # Either a fallback dir was found (which is not fake_path) …
        if result.exists():
            assert result != fake_path, "Should have used a fallback, not the nonexistent env-var path"
        else:
            # … or no fallback exists, in which case fake_path is returned as best guess
            assert result == fake_path

    def test_resolve_agent_dir_found(self, tmp_path):
        """_resolve_agent_dir returns the correct path when the agent exists."""
        from rosnav_rl.action_server.arena_server import _resolve_agent_dir

        agent_name = "my_test_agent"
        agent_dir = tmp_path / agent_name
        agent_dir.mkdir()

        with patch.dict(os.environ, {"ROSNAV_AGENTS_DIR": str(tmp_path)}):
            result = _resolve_agent_dir(agent_name)

        assert result == agent_dir

    def test_resolve_agent_dir_not_found(self, tmp_path):
        """_resolve_agent_dir raises FileNotFoundError if agent is missing."""
        from rosnav_rl.action_server.arena_server import _resolve_agent_dir

        with patch.dict(os.environ, {"ROSNAV_AGENTS_DIR": str(tmp_path)}):
            with pytest.raises(FileNotFoundError, match="not found"):
                _resolve_agent_dir("ghost_agent")

    def test_resolve_agent_dir_lists_available(self, tmp_path):
        """Error message for a missing agent should list available agents."""
        from rosnav_rl.action_server.arena_server import _resolve_agent_dir

        (tmp_path / "agent_a").mkdir()
        (tmp_path / "agent_b").mkdir()

        with patch.dict(os.environ, {"ROSNAV_AGENTS_DIR": str(tmp_path)}):
            with pytest.raises(FileNotFoundError) as exc_info:
                _resolve_agent_dir("ghost")

        msg = str(exc_info.value)
        assert "agent_a" in msg or "agent_b" in msg


# ─────────────────────────────────────────────────────────────────────────────
# 2. Service-mechanism tests (needs ROS2)
# ─────────────────────────────────────────────────────────────────────────────

ros2 = pytest.mark.skipif(not _ros_running(), reason="ROS2 not available")


_NUM_BEAMS = 360


def _raw_observation() -> Dict[str, Any]:
    return {
        "front_laser": np.full(_NUM_BEAMS, 2.0, dtype=np.float32),
        "dist_angle_to_subgoal": np.array([3.0, 0.3], dtype=np.float32),
        "last_action": np.zeros(3, dtype=np.float32),
        "is_terminal": False,
    }


def _agent_1():
    from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
    from rosnav_rl.cfg.agent import AgentConfig
    from rosnav_rl.cfg.parameters import AgentParameters
    from rosnav_rl.model.stable_baselines3.cfg import PPO_Algorithm_Cfg, PPO_Cfg
    from rosnav_rl.model.stable_baselines3.cfg.framework import StableBaselinesCfg
    from rosnav_rl.rl_agent import RL_Agent
    from rosnav_rl.utils.utils import make_mock_env

    agent = RL_Agent(
        AgentConfig(
            name="action_server_agent",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=_NUM_BEAMS, laser_max_range=30.0, normalize=True),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(
                    architecture_name="AGENT_1",
                    parameters=PPO_Algorithm_Cfg(total_batch_size=64, batch_size=64),
                )
            ),
        )
    )
    agent.initialize_model(
        make_mock_env(
            ns="",
            space_manager=agent.space_manager,
            stack_size=agent.model._policy_description.stack_size,
        )
    )
    return agent


class FixedObservationCollector:
    def get_observations(self) -> Dict[str, Any]:
        return _raw_observation()


@ros2
class TestGetCommandService:
    """Integration tests for the GetCommand ROS2 service using a real AGENT_1 policy."""

    @pytest.fixture(autouse=True)
    def _spin(self):
        """Start the action server node in a background thread; tear it down after."""
        import rclpy
        from rosnav_rl.action_server.base_server import ActionServer

        agent = _agent_1()

        class ConcreteServer(ActionServer):
            def _initialize_agent(self_inner):
                return agent

            def _initialize_observation_collector(self_inner):
                return FixedObservationCollector()

        if not rclpy.ok():
            rclpy.init()

        self.server = ConcreteServer(agent_name="action_server_agent", namespace="test_ns")

        self._spin_thread = threading.Thread(target=self.server.start, daemon=True)
        self._spin_thread.start()
        time.sleep(0.5)
        yield

        self.server.node.destroy_node()

    def test_get_command_returns_the_decoded_agent_action(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rosnav_rl_msgs.srv import GetCommand

        client_node = rclpy.create_node("test_client")
        client = client_node.create_client(GetCommand, "/test_ns/get_command")
        executor = SingleThreadedExecutor()
        executor.add_node(client_node)

        assert client.wait_for_service(timeout_sec=5.0), "GetCommand service not available within 5 s"

        future = client.call_async(GetCommand.Request())
        executor.spin_until_future_complete(future, timeout_sec=5.0)

        response = future.result()
        assert response is not None, "Service call timed out"
        expected = self.server.agent.get_action(_raw_observation())
        assert list(response.action) == pytest.approx(expected.tolist())
        assert response.action_type == self.server.agent.space_manager.action_space_manager.spec.type

        executor.shutdown()
        client_node.destroy_node()

    def test_get_command_before_agent_init_returns_empty_action(self):
        from rosnav_rl_msgs.srv import GetCommand

        saved_agent = self.server.agent
        self.server.agent = None

        request = GetCommand.Request()
        response = GetCommand.Response()
        result = self.server._ActionServer__handle_next_action_srv(request, response)

        assert list(result.action) == []
        assert result.action_type == ""

        self.server.agent = saved_agent


# ─────────────────────────────────────────────────────────────────────────────
# 3. Load-pipeline smoke test (requires create_test_agent.py to have been run)
# ─────────────────────────────────────────────────────────────────────────────

TEST_AGENT_DIR = _get_agents_dir() / "test_agent"
requires_test_agent = pytest.mark.skipif(
    not TEST_AGENT_DIR.is_dir(),
    reason=("Test agent not found. Run: python3 scripts/create_test_agent.py --agent-name test_agent"),
)


@requires_test_agent
class TestLoadPipeline:
    """Smoke tests that load the model saved by create_test_agent.py."""

    def test_training_config_exists(self):
        assert (TEST_AGENT_DIR / "training_config.yaml").exists()

    def test_model_zip_exists(self):
        assert (TEST_AGENT_DIR / "best_model.zip").exists()

    def test_training_config_loads(self):
        """TrainingCfg must be reconstructable from the saved YAML."""
        from arena_training.arena_rosnav_rl.cfg.train import TrainingCfg
        from rosnav_rl.utils.utils import load_yaml

        cfg = TrainingCfg.model_validate(load_yaml(TEST_AGENT_DIR / "training_config.yaml"))
        assert cfg.agent_cfg.name == "test_agent"

    @ros2
    def test_arena_server_initializes_agent(self):
        """ArenaActionServer._initialize_agent() must load the saved model."""
        import rclpy
        from rosnav_rl.action_server.arena_server import ArenaActionServer

        if not rclpy.ok():
            rclpy.init()

        with patch.dict(os.environ, {"ROSNAV_AGENTS_DIR": str(_get_agents_dir())}):
            server = ArenaActionServer(
                agent_name="test_agent",
                namespace="smoke_test",
            )
            server._initialize_ros()
            agent = server._initialize_agent()

        assert agent is not None

        # Verify the model weights were actually loaded (not still uninitialised).
        # StableBaselinesModel.load() sets self._model; is_model_initialized
        # becomes True only after a successful load.
        assert agent.model.is_model_initialized, "Model was not loaded — is_model_initialized is still False"

        # Sanity-check the action + observation spaces match AGENT_3 specs.
        obs_space = agent.observation_space
        act_space = agent.action_space
        assert "ReducedLaserScanSpace" in obs_space.spaces
        assert "DistAngleToSubgoalSpace" in obs_space.spaces
        assert "LastActionSpace" in obs_space.spaces
        # Non-holonomic jackal → 2D action space
        assert act_space.shape == (2,)

        server.node.destroy_node()
