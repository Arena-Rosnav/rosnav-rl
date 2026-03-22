"""YAML serialisation helpers for rosnav_rl config models."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosnav_rl.cfg.agent import AgentConfig


def agent_config_to_commented_map(agent_cfg: "AgentConfig"):
    """Build a structured ``ruamel.yaml`` :class:`CommentedMap` for an :class:`AgentConfig`.

    Extracted here so both :meth:`AgentConfig.to_yaml` and
    :func:`arena_training …tools.general._build_training_commented_map` produce
    the agent section in exactly the same layout without duplicating logic.
    """
    from ruamel.yaml.comments import CommentedMap

    data = agent_cfg.model_dump(mode="json", exclude_none=True)
    root = CommentedMap()

    # identity
    for key in ("name", "robot"):
        if key in data:
            root[key] = data[key]
    root.yaml_set_comment_before_after_key("name", before="\nAgent identity")

    # pre-training intent
    for key in ("observations_config", "discretization"):
        if key in data:
            root[key] = data[key]

    # action space
    if "action_space" in data:
        root["action_space"] = data["action_space"]
        root.yaml_set_comment_before_after_key(
            "action_space", before="\nAction space (derived from robot description)"
        )

    # unified parameters (observation pipeline + environment/reward constants)
    if "parameters" in data:
        root["parameters"] = data["parameters"]
        root.yaml_set_comment_before_after_key(
            "parameters",
            before=(
                "\nObservation pipeline bounds and environment/reward constants.\n"
                "# Review and adjust before training - especially laser_num_beams,\n"
                "# laser_max_range, goal_radius, max_steps, and velocity bounds."
            ),
        )

    # framework
    root["framework"] = data["framework"]
    root.yaml_set_comment_before_after_key(
        "framework", before="\nRL framework configuration"
    )

    # reward
    if "reward" in data:
        root["reward"] = data["reward"]
        root.yaml_set_comment_before_after_key("reward", before="\nReward function")

    return root
