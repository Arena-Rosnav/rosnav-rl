import importlib.resources
import os
from pathlib import Path

import yaml

from rosnav_rl.observations.factory.factory import (
    create_observation_manager_from_config,
)
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.utils.utils import load_yaml

from .base_server import ActionServer, ObservationCollector


def _find_agents_dir() -> Path:
    """Find the agents directory by searching known locations."""
    candidates = []

    # 1. Environment variable override
    env_path = os.environ.get("ROSNAV_AGENTS_DIR")
    if env_path:
        candidates.append(Path(env_path))

    # 2. Try to find via ament_index (arena_training package share)
    try:
        from ament_index_python.packages import get_package_share_directory
        at_share = Path(get_package_share_directory("arena_training"))
        # The share dir is in install tree; agents are in source tree.
        # Walk up from share to find the source tree.
        # install/<pkg>/share/<pkg> -> 4 levels up is the workspace root
        ws_root = at_share.parents[3]
        candidates.append(ws_root / "src" / "Arena" / "arena_training" / "agents")
    except Exception:
        pass

    # 3. Navigate from this file's resolved path
    this_file = Path(__file__).resolve()
    # When in source tree via symlink: .../arena_training/deps/rosnav_rl/rosnav_rl/rosnav_rl/action_server/arena_server.py
    # Walk up looking for an "arena_training" directory that contains "agents"
    for parent in this_file.parents:
        if parent.name == "arena_training" and (parent / "agents").is_dir():
            candidates.append(parent / "agents")
            break

    # 4. Try workspace-relative paths using COLCON_PREFIX_PATH or AMENT_PREFIX_PATH
    for env_var in ("COLCON_PREFIX_PATH", "AMENT_PREFIX_PATH"):
        prefix_path = os.environ.get(env_var, "")
        if prefix_path:
            # Take first path, go up to workspace root
            first_prefix = Path(prefix_path.split(":")[0])
            ws_root = first_prefix.parent  # install/ -> ws_root
            candidates.append(ws_root / "src" / "Arena" / "arena_training" / "agents")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Return best guess even if it doesn't exist yet
    return candidates[0] if candidates else Path("/agents")


def _resolve_agent_dir(agent_name: str) -> Path:
    """Resolve the agent directory."""
    agents_dir = _find_agents_dir()
    agent_dir = agents_dir / agent_name

    if agent_dir.is_dir():
        return agent_dir

    raise FileNotFoundError(
        f"Agent '{agent_name}' not found at: {agent_dir}\n"
        f"Available agents: {[d.name for d in agents_dir.iterdir() if d.is_dir()] if agents_dir.is_dir() else '(agents dir not found)'}\n"
        f"Set ROSNAV_AGENTS_DIR environment variable to override."
    )



class ArenaActionServer(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        """Initialize and return an RL_Agent from a saved training config.

        Loads the ``AgentConfig`` from the training config and stores
        ``spec.parameters`` as ``agent_parameters`` for the observation pipeline.
        """
        from arena_training.arena_rosnav_rl.cfg.train import TrainingCfg

        model_dir = _resolve_agent_dir(self.agent_name)
        self.logger.info(f"Loading agent from: {model_dir}")

        training_cfg = TrainingCfg.model_validate(
            load_yaml(model_dir / "training_config.yaml")
        )

        spec = training_cfg.agent_config
        self.agent_parameters: AgentParameters = spec.parameters

        agent = RL_Agent(spec)
        agent.load_model(path=model_dir / "best_model.zip")
        return agent

    def _initialize_observation_collector(self) -> ObservationCollector:
        """
        Initializes and returns an ObservationCollector instance.

        Uses the ObservationManager.from_config() factory with the bundled
        observations.yaml config to set up ROS2 topic subscribers for
        collecting sensor data needed by the RL agent.

        Returns:
            ObservationCollector: Configured ObservationManager instance.
        """
        # Try to use agent-specific observation config if saved, otherwise use default
        agent_dir = _resolve_agent_dir(self.agent_name)
        obs_config_path = agent_dir / "observations.yaml"

        if not obs_config_path.exists():
            obs_config_path = str(
                importlib.resources.files("rosnav_rl")
                / "observations"
                / "observations.yaml"
            )

        with open(obs_config_path, "r") as f:
            config = yaml.safe_load(f)

        obs_manager = create_observation_manager_from_config(
            config=config,
            node=self.node,
            ns=str(self.namespace),
            simulation_state_container=self.agent_parameters,
            wait_for_obs=False,
        )

        return obs_manager
