import importlib.resources
import os
from pathlib import Path

import yaml

from rosnav_rl.observations.factory.factory import (
    create_observation_manager_from_config,
)
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.states import AgentStateContainer, SimulationStateContainer
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


def _get_arena_states(training_cfg) -> SimulationStateContainer:
    """Build SimulationStateContainer from the saved training config.

    This replaces the old `from tools.states import get_arena_states` which
    depended on arena_training internals. We construct the state container
    directly from rosnav_rl's own state classes.
    """
    import rosnav_rl.states.simulation as simulation_states

    arena_cfg = training_cfg.arena_cfg
    agent_cfg = training_cfg.agent_cfg

    robot_cfg = arena_cfg.robot
    robot_desc = robot_cfg.robot_description

    robot_state = simulation_states.RobotState(
        radius=robot_desc.robot_radius,
        safety_distance=arena_cfg.general.safety_distance,
        action_state=simulation_states.ActionState(
            is_discrete=agent_cfg.action_space.is_discrete,
            actions=(
                robot_desc.actions.discrete
                if agent_cfg.action_space.is_discrete
                else robot_desc.actions.continuous.model_dump()
            ),
            is_holonomic=robot_desc.is_holonomic,
            velocity_state=simulation_states.VelocityState(
                min_linear_vel=-2.0,
                max_linear_vel=2.0,
                min_translational_vel=-2.0,
                max_translational_vel=2.0,
                min_angular_vel=-4.0,
                max_angular_vel=4.0,
            ),
        ),
        laser_state=simulation_states.LaserState(
            attach_full_range_laser=robot_cfg.attach_full_range_laser,
            laser_max_range=robot_desc.laser.range,
            laser_num_beams=robot_desc.laser.num_beams,
        ),
    )

    task_state = simulation_states.TaskState(
        goal_radius=arena_cfg.general.goal_radius,
        max_steps=arena_cfg.general.max_num_moves_per_eps,
        semantic_state=simulation_states.SemanticState(
            num_ped_types=5,
            ped_min_speed_x=-5.0,
            ped_max_speed_x=5.0,
            ped_min_speed_y=-5.0,
            ped_max_speed_y=5.0,
            social_state_num=99,
        ),
        task_modules=simulation_states.TaskModuleState(
            tm_robots=arena_cfg.task.tm_robots,
            tm_obstacles=arena_cfg.task.tm_obstacles,
            tm_modules=arena_cfg.task.tm_modules,
        ),
    )

    return simulation_states.SimulationStateContainer(
        robot=robot_state, task=task_state
    )


class ArenaActionServer(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        """
        Initializes and returns an RL_Agent instance.

        Loads the training config and model checkpoint from the agent directory,
        reconstructs the simulation state, and creates an RL_Agent ready for inference.

        Returns:
            RL_Agent: An initialized agent with loaded model weights.
        """
        from arena_training.arena_rosnav_rl.cfg.train import TrainingCfg

        model_dir = _resolve_agent_dir(self.agent_name)
        self.logger.info(f"Loading agent from: {model_dir}")

        training_cfg = TrainingCfg.model_validate(
            load_yaml(model_dir / "training_config.yaml")
        )

        self.simulation_state_container: SimulationStateContainer = (
            _get_arena_states(training_cfg)
        )
        agent_state_container: AgentStateContainer = (
            self.simulation_state_container.to_agent_state_container()
        )

        agent = RL_Agent(
            agent_cfg=training_cfg.agent_cfg,
            agent_state_container=agent_state_container,
        )
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
            simulation_state_container=self.simulation_state_container,
            wait_for_obs=False,
        )

        return obs_manager
