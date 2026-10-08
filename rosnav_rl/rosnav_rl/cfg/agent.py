"""Complete, self-contained agent specification.

:class:`AgentConfig` is **the** single source of truth for everything that
defines an RL agent — identity, robot type, action space, observations,
framework, reward, and training-environment parameters.

Save / load with classmethods::

    spec.to_yaml("my_agent.yaml")
    spec = AgentConfig.from_yaml("my_agent.yaml")

Or via dict round-trip::

    d = spec.to_dict()
    spec2 = AgentConfig.from_dict(d)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import BaseModel, Discriminator, Field, PrivateAttr, model_validator

from rosnav_rl.utils.name_generator import generate_agent_name

from ..model.dreamerv3.cfg import DreamerV3Cfg
from ..model.stable_baselines3.cfg import StableBaselinesCfg
from .action_spaces import ActionSpaceSpec, DiscretizationCfg
from .parameters import AgentParameters
from .reward import RewardCfg

if TYPE_CHECKING:
    from ruamel.yaml.comments import CommentedMap


class AgentConfig(BaseModel):
    """Complete specification of an RL agent.

    It is fully serialisable to / from
    YAML or plain dicts and captures *everything* needed to reconstruct
    the training pipeline.

    Attributes:
        name:  Agent identifier (auto-generated when ``None``).
        robot: Target robot model — derived from ``arena_robots`` at training time.
        observations_config: Path to the observations YAML (relative to the
            training config file, or absolute).  When ``None`` the built-in
            default observations are used.
        discretization: Optional discrete-action strategy.  When set, the
            trainer calls ``action_space.resolve_discretization()`` before
            training; the result is stored in ``action_space.discrete_actions``.
        action_space: Typed action space.  ``None`` until the trainer populates
            it from the robot description.  In saved agent configs it is always
            present.
        parameters: Unified observation + environment parameters. Review and override in the training config YAML before
            starting a run.  See :class:`~rosnav_rl.cfg.parameters.AgentParameters`.
        framework: RL framework configuration (SB3 or DreamerV3).
        reward: Reward function definition (optional, training only).
    """

    name: str | None = Field(None, description="Agent identifier. Auto-generated if omitted.")
    robot: str | None = Field(
        None, description="Robot model name (e.g. 'jackal', 'turtlebot3'). Derived from arena_robots at training time."
    )
    observations_config: str | None = Field(
        None, description="Path to observations YAML config (relative to training config file, or absolute)."
    )
    discretization: DiscretizationCfg | None = Field(
        None,
        description=(
            "Discrete-action strategy. Transferred onto action_space by the trainer at startup. "
            "Has no effect in saved agent configs (where action_space already carries resolved discrete_actions)."
        ),
    )

    action_space: ActionSpaceSpec | None = Field(
        None, description="Typed action space. Derived from robot description at training time."
    )
    parameters: AgentParameters = AgentParameters()

    framework: Annotated[
        StableBaselinesCfg | DreamerV3Cfg,
        Discriminator(discriminator="name"),
    ]
    reward: RewardCfg | None = None

    # True when the name was auto-generated (not user-supplied).
    # Used by model_copy to know whether to refresh the name when robot changes.
    _name_is_auto: bool = PrivateAttr(default=True)

    @model_validator(mode="after")
    def _auto_name(self) -> Self:
        if self.name is None:
            self.name = generate_agent_name(self.framework, robot=self.robot)
        else:
            self._name_is_auto = False
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Copy with automatic name refresh when ``robot`` is updated.

        If the caller supplies an explicit ``"name"`` key in *update* it is
        used as-is and the auto-flag is cleared.  Otherwise, whenever
        ``"robot"`` appears in *update* and the current name was auto-generated,
        the name is re-derived from the new robot + existing framework.
        """
        result = super().model_copy(update=update, deep=deep)
        if update and "name" in update:
            result._name_is_auto = False
        else:
            result._name_is_auto = self._name_is_auto
            if update and "robot" in update and self._name_is_auto:
                result.name = generate_agent_name(result.framework, robot=result.robot)
        return result

    # ------------------------------------------------------------------ #
    #  Serialisation
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def _to_commented_map(self) -> CommentedMap:
        """Delegates to :func:`rosnav_rl.utils.yaml_utils.agent_config_to_commented_map`."""
        from rosnav_rl.utils.yaml_utils import agent_config_to_commented_map

        return agent_config_to_commented_map(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save to a structured YAML file with section comments."""
        from ruamel.yaml import YAML

        ry = YAML()
        ry.default_flow_style = False
        ry.width = 120

        with open(path, "w") as f:
            ry.dump(self._to_commented_map(), f)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct from a plain dictionary."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return self.model_dump()
