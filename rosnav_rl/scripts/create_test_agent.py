#!/usr/bin/env python3
"""Create a minimal test agent in the agents directory so the action server can
load it without needing a completed training run.

Usage
-----
  python3 scripts/create_test_agent.py [--agent-name <name>] [--config <path>]

The script:
  1. Loads a TrainingCfg from an existing arena_bringup config YAML.
  2. Populates an AgentParameters from the robot description.
  3. Instantiates an RL_Agent from the AgentConfig which sets up the observation
     and action spaces.
  4. Calls setup_model() with a mock VecEnv to create the PPO model with random
     initial weights.
  5. Saves training_config.yaml + best_model.zip to
     Arena/arena_training/agents/<agent_name>/.
"""

import argparse
import os
import sys
from pathlib import Path

# ── path bootstrap ────────────────────────────────────────────────────────────
# Locate the Arena workspace root robustly regardless of whether this script
# is run directly from the repo or via the arena_training/deps/rosnav_rl symlink.


def _find_arena_root() -> Path:
    """Walk up from arena_training package if importable, else try typical paths."""
    # Strategy 1: ament_index → install tree → workspace root → src/Arena
    try:
        from ament_index_python.packages import get_package_share_directory

        at_share = Path(get_package_share_directory("arena_training"))
        ws_root = at_share.parents[3]  # install/<pkg>/share/<pkg> -> ws
        arena = ws_root / "src" / "Arena"
        if arena.is_dir():
            return arena
    except Exception:
        pass

    # Strategy 2: walk COLCON/AMENT_PREFIX_PATH
    for env_var in ("COLCON_PREFIX_PATH", "AMENT_PREFIX_PATH"):
        prefix = os.environ.get(env_var, "")
        if prefix:
            arena = Path(prefix.split(":")[0]).parent / "src" / "Arena"
            if arena.is_dir():
                return arena

    # Strategy 3: walk up from __file__ looking for src/Arena marker
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "Arena"
        if (candidate / "arena_training").is_dir():
            return candidate
        # Also check for direct Arena directory
        if (parent / "arena_training").is_dir() and (parent / "arena_bringup").is_dir():
            return parent

    # Fallback
    return Path("/home/le/arena5_ws/src/Arena")


ARENA_ROOT = _find_arena_root()

DEFAULT_CONFIG = (
    ARENA_ROOT / "arena_training" / "configs" / "sb_training_config.yaml"
)
_DEFAULT_AGENTS_DIR = ARENA_ROOT / "arena_training" / "agents"


def _resolve_agents_dir(cli_agents_dir: Path | None = None) -> Path:
    """Resolve agents directory with fallback chain: CLI > env var > default."""
    if cli_agents_dir is not None:
        return Path(cli_agents_dir).expanduser().resolve()
    env_dir = os.environ.get("ROSNAV_AGENTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return _DEFAULT_AGENTS_DIR

# ── helpers ───────────────────────────────────────────────────────────────────


def _populate_agent_spec(training_cfg):
    """Derive action_space, observation and environment from the robot description."""
    from rosnav_rl.cfg.action_spaces import (  # noqa: PLC0415
        DifferentialDriveActionSpace,
        OmnidirectionalActionSpace,
    )
    from rosnav_rl.cfg.parameters import AgentParameters  # noqa: PLC0415

    robot_desc = training_cfg.arena_cfg.robot.robot_description
    general = training_cfg.arena_cfg.general
    cont = robot_desc.actions.continuous

    if robot_desc.is_holonomic:
        action_space = OmnidirectionalActionSpace(
            linear_range_x=tuple(cont.linear_range),
            linear_range_y=tuple(cont.linear_range),
            angular_range=tuple(cont.angular_range),
        )
    else:
        action_space = DifferentialDriveActionSpace(
            linear_range=tuple(cont.linear_range),
            angular_range=tuple(cont.angular_range),
        )

    # Carry over the discretization config from the agent_config YAML, then
    # resolve it immediately using the robot's built-in discrete action list.
    discretization_cfg = training_cfg.agent_config.discretization
    if discretization_cfg is not None:
        action_space = action_space.model_copy(
            update={"discretization": discretization_cfg}
        )
        action_space = action_space.resolve_discretization(
            robot_discrete_actions=robot_desc.actions.discrete
        )

    parameters = AgentParameters(
        laser_num_beams=robot_desc.laser.num_beams,
        laser_max_range=robot_desc.laser.range,
        min_linear_vel=cont.linear_range[0],
        max_linear_vel=cont.linear_range[1],
        min_angular_vel=cont.angular_range[0],
        max_angular_vel=cont.angular_range[1],
        min_translational_vel=cont.linear_range[0],
        max_translational_vel=cont.linear_range[1],
        robot_radius=robot_desc.robot_radius,
        safety_distance=general.safety_distance,
        goal_radius=general.goal_radius,
        max_steps=general.max_num_moves_per_eps,
    )

    training_cfg.agent_config = training_cfg.agent_config.model_copy(
        update={
            "robot": robot_desc.robot_model,
            "action_space": action_space,
            "parameters": parameters,
        }
    )


def create_test_agent(
    agent_name: str, config_path: Path, agents_dir: Path | None = None
) -> Path:
    """Create and save a minimal test agent.

    Returns the path to the created agent directory.
    """
    import yaml
    from rosnav_rl.utils.utils import load_yaml

    print(f"[create_test_agent] Loading TrainingCfg from {config_path}")
    from arena_training.arena_rosnav_rl.cfg.train import TrainingCfg  # noqa: PLC0415

    training_cfg = TrainingCfg.model_validate(load_yaml(config_path))
    training_cfg.agent_config.name = agent_name
    print(f"[create_test_agent] Using robot: {training_cfg.arena_cfg.robot.robot_description.robot_model}")

    # ── Build everything from robot description ────────────────────────────
    _populate_agent_spec(training_cfg)

    # ── Create RL_Agent (sets up spaces + model wrapper) ──────────────────
    print("[create_test_agent] Creating RL_Agent …")
    from rosnav_rl.rl_agent import RL_Agent  # noqa: PLC0415

    agent = RL_Agent(training_cfg.agent_config)
    print(f"[create_test_agent] Observation space: {agent.observation_space}")
    print(f"[create_test_agent] Action space:      {agent.action_space}")

    # ── Initialise model with random weights via mock env ─────────────────
    print("[create_test_agent] Initialising PPO model with random weights …")
    from stable_baselines3 import PPO  # noqa: PLC0415
    from rosnav_rl.utils.utils import make_mock_env  # noqa: PLC0415

    mock_env = make_mock_env(ns="", space_manager=agent.space_manager)

    # Build the PPO directly (as the trainer would, but without batch-size
    # checks and without an actual training run).  We use the same policy
    # kwargs as the AGENT_3 architecture so weights are compatible.
    policy_kwargs = agent.model._policy_description.get_kwargs()
    ppo = PPO(
        policy="MultiInputPolicy",
        env=mock_env,
        policy_kwargs=policy_kwargs,
        n_steps=64,    # minimal batch — only for model init, not training
        batch_size=64,
        device="cpu",
        verbose=0,
    )
    # Attach the PPO to the RL_Agent model so save() works normally
    agent.model._StableBaselinesModel__env = None  # reset env (clean state)
    agent.model._model = ppo

    # ── Save agent directory ───────────────────────────────────────────────
    resolved_dir = _resolve_agents_dir(agents_dir)
    agent_dir = resolved_dir / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Save training_config.yaml
    # Use mode='json' so all values serialise to JSON-compatible primitives
    # (Enum values → strings, etc.) — avoids Python-specific YAML tags that
    # yaml.FullLoader cannot round-trip.
    cfg_path = agent_dir / "training_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(training_cfg.model_dump(mode="json"), f, default_flow_style=False)
    print(f"[create_test_agent] Saved {cfg_path}")

    # Save best_model.zip (SB3 PPO format)
    model_path = agent_dir / "best_model"
    agent.model._model.save(str(model_path))
    # SB3 appends .zip automatically - make sure the file exists
    zip_path = agent_dir / "best_model.zip"
    if not zip_path.exists():
        print("[create_test_agent] WARNING: best_model.zip not found after save!")
    else:
        print(f"[create_test_agent] Saved {zip_path}")

    print(f"\n✓ Test agent '{agent_name}' created at {agent_dir}")
    print("\nTo start the action server with this agent run:")
    print(f"  source ~/arena5_ws/install/setup.bash")
    print(f"  export ROSNAV_AGENTS_DIR={resolved_dir}")
    print(f"  ros2 run rosnav_rl action_server.py --ros-args -p agent_name:={agent_name}")
    print("\nThen, in another terminal:")
    print("  ros2 service call /get_command rosnav_rl_msgs/srv/GetCommand {}")

    return agent_dir


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-name",
        default="test_agent",
        help="Name for the test agent (default: test_agent)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the training config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for agent artifacts "
            "(default: ROSNAV_AGENTS_DIR env var or arena_training/agents/)"
        ),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    create_test_agent(args.agent_name, args.config, agents_dir=args.agents_dir)


if __name__ == "__main__":
    main()
