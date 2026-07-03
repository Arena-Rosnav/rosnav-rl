"""Resolve where trained agents live on disk."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from rosnav_rl.cfg.agent import AgentConfig


def find_agents_dir() -> Path:
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
        # install/<pkg>/share/<pkg> -> 4 levels up is the workspace root
        ws_root = at_share.parents[3]
        candidates.append(ws_root / "src" / "Arena" / "arena_training" / "agents")
    except Exception:
        pass

    # 3. Walk up from this file looking for an "arena_training" dir that contains "agents"
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if parent.name == "arena_training" and (parent / "agents").is_dir():
            candidates.append(parent / "agents")
            break

    # 4. Workspace-relative paths via COLCON_PREFIX_PATH or AMENT_PREFIX_PATH
    for env_var in ("COLCON_PREFIX_PATH", "AMENT_PREFIX_PATH"):
        prefix_path = os.environ.get(env_var, "")
        if prefix_path:
            first_prefix = Path(prefix_path.split(":")[0])
            ws_root = first_prefix.parent  # install/ -> ws_root
            candidates.append(ws_root / "src" / "Arena" / "arena_training" / "agents")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return candidates[0] if candidates else Path("/agents")


def resolve_agent_dir(agent_name: str) -> Path:
    """Resolve a named agent's directory, or raise FileNotFoundError."""
    agents_dir = find_agents_dir()
    agent_dir = agents_dir / agent_name

    if agent_dir.is_dir():
        return agent_dir

    raise FileNotFoundError(
        f"Agent '{agent_name}' not found at: {agent_dir}\n"
        f"Available agents: {[d.name for d in agents_dir.iterdir() if d.is_dir()] if agents_dir.is_dir() else '(agents dir not found)'}\n"
        f"Set ROSNAV_AGENTS_DIR environment variable to override."
    )


def load_agent_spec(agent_dir: Path) -> "AgentConfig":
    """Load the ``AgentConfig`` embedded in a saved agent's ``training_config.yaml``.

    Only the ``agent_config`` subtree is validated — the rest of the training
    config (arena_cfg, resume, ...) is owned by the training entrypoint, not
    by deployment/inference consumers.
    """
    from rosnav_rl.cfg.agent import AgentConfig

    config_path = Path(agent_dir) / "training_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No training_config.yaml found in agent dir: {agent_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    try:
        agent_config_dict = raw["agent_config"]
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"'{config_path}' has no top-level 'agent_config' key"
        ) from e

    return AgentConfig.model_validate(agent_config_dict)


def resolve_observations_config_path(spec: "AgentConfig") -> Path:
    """Resolve the observations.yaml a saved agent's training run actually used.

    ``spec.observations_config`` is absolutized by arena_training's config
    loader at training time; ``None`` means the package's built-in default was
    used (a documented, legitimate case). If the field is set but the file no
    longer exists, fail loudly instead of silently swapping in the package
    default — that would silently deploy a different observation pipeline
    than the one the agent was trained against.
    """
    import importlib.resources

    if spec.observations_config is None:
        return Path(
            importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml"
        )

    path = Path(spec.observations_config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Agent '{spec.name}' was trained with observations_config="
            f"'{spec.observations_config}', but that file no longer exists. "
            f"Refusing to silently fall back to the package default observation "
            f"pipeline, since it would not match what the agent was trained on."
        )
    return path
