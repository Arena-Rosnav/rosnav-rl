"""Typed action space specifications for different robot morphologies.

Each robot type is a self-contained Pydantic model that knows:
  - Its DOF structure and velocity/position ranges
  - How to create a gymnasium Space
  - How to decode model output into robot commands

Uses Pydantic discriminated unions so the YAML ``type`` field auto-selects
the correct class::

    # YAML
    action_space:
      type: differential_drive
      linear_range: [-0.5, 1.0]
      angular_range: [-1.0, 1.0]

    # Python
    spec = DifferentialDriveActionSpace(
        linear_range=(-0.5, 1.0),
        angular_range=(-1.0, 1.0),
    )
    gym_space = spec.get_gym_space()
    cmd = spec.decode(model_output)

Discretization
--------------
Discrete actions are configured via ``DiscretizationCfg`` on the action space
itself — not via a global ``is_discrete`` flag.  This keeps the intent
(continuous vs. discrete, and *how* to discretise) entirely within the action
space config::

    action_space:
      type: differential_drive
      discretization:
        strategy: robot_defined   # use the robot's built-in action list

    action_space:
      type: differential_drive
      discretization:
        strategy: navigational    # hand-crafted nav-optimised set (~11 actions)

    action_space:
      type: differential_drive
      discretization:
        strategy: uniform
        buckets_linear: 7
        buckets_angular: 9

    action_space:
      type: differential_drive
      discretization:
        strategy: exponential
        buckets_linear: 5
        buckets_angular: 7

At training time ``_populate_agent_spec`` calls
``action_space.resolve_discretization(robot_discrete_actions)`` which converts
the strategy config into a concrete ``discrete_actions`` list.  Until that call
``is_discrete`` remains ``False`` even if ``discretization`` is set.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, Discriminator, Field
from typing_extensions import Annotated


class DiscretizationStrategy(str, Enum):
    """Supported auto-discretization strategies for mobile action spaces.

    Attributes:
        ROBOT_DEFINED:  Use the discrete action list embedded in the robot's
                        ``model_params.yaml`` (hand-crafted by the robot maintainer).
                        Falls back to ``UNIFORM`` when the robot has no pre-defined
                        discrete actions.
        UNIFORM:        Evenly-spaced N×M grid over the velocity ranges.  Simple
                        and general; good baseline for any robot.
        NAVIGATIONAL:   A small (~12) hand-engineered set that covers the most
                        important motions for indoor navigation: stop, straight,
                        lean turns, tight spins, and a gentle reverse.  Trains
                        faster than large grids whilst maintaining coverage.
        EXPONENTIAL:    Logarithmically-spaced grid — denser near zero for precise
                        low-speed control, sparser at the extremes.  Well-suited to
                        cluttered environments that require fine-grained manoeuvring.
    """

    ROBOT_DEFINED = "robot_defined"
    UNIFORM = "uniform"
    NAVIGATIONAL = "navigational"
    EXPONENTIAL = "exponential"


class DiscretizationCfg(BaseModel):
    """Configuration for auto-discretization of a continuous action space.

    Place this under ``action_space.discretization`` in your training YAML.
    ``_populate_agent_spec`` will call ``resolve_discretization()`` at training
    startup to convert the strategy into a concrete ``discrete_actions`` list.

    Attributes:
        strategy:        Which algorithm to use (see ``DiscretizationStrategy``).
        buckets_linear:  Number of linear-velocity levels for UNIFORM / EXPONENTIAL.
        buckets_angular: Number of angular-velocity levels for UNIFORM / EXPONENTIAL.
    """

    strategy: DiscretizationStrategy = DiscretizationStrategy.ROBOT_DEFINED
    buckets_linear: int = Field(7, ge=2, description="Linear velocity buckets (UNIFORM / EXPONENTIAL)")
    buckets_angular: int = Field(9, ge=2, description="Angular velocity buckets (UNIFORM / EXPONENTIAL)")



class BaseActionSpace(BaseModel):
    """Base for all robot action space specifications.

    Subclass to define a new robot type.  Every subclass **must** set a
    ``type`` literal and implement :meth:`get_gym_space`, :meth:`decode`,
    and the :attr:`num_dof` property.
    """

    type: str

    # -- interface ----------------------------------------------------------

    def get_gym_space(self) -> spaces.Space:
        """Return the gymnasium action space for this robot type."""
        raise NotImplementedError

    def decode(self, action: np.ndarray) -> np.ndarray:
        """Decode model output into a robot command vector.

        Returns:
            np.ndarray of shape ``(3,)`` for mobile robots ``[vx, vy, wz]``,
            or ``(num_dof,)`` for manipulators / humanoids.
        """
        raise NotImplementedError

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def is_holonomic(self) -> bool:
        return self.type == "omnidirectional"

    @property
    def num_dof(self) -> int:
        raise NotImplementedError


# =====================================================================
#  Mobile robots
# =====================================================================


class DifferentialDriveActionSpace(BaseActionSpace):
    """2-DOF non-holonomic mobile robot: ``[linear_x, angular_z]``.

    Common platforms: TurtleBot, Burger, Jackal.
    Output command: ``[linear_x, 0.0, angular_z]``.

    Set ``discretization`` to enable auto-discretization at training time.
    The space stays continuous (``is_discrete=False``) until
    ``resolve_discretization()`` is called.
    """

    type: Literal["differential_drive"] = "differential_drive"
    linear_range: Tuple[float, float] = (-0.5, 1.0)
    angular_range: Tuple[float, float] = (-1.0, 1.0)
    discrete_actions: Optional[List[Dict[str, Any]]] = None
    discretization: Optional[DiscretizationCfg] = Field(
        None,
        description="Auto-discretization config. Set strategy here; resolved at training time.",
    )

    @property
    def is_discrete(self) -> bool:
        return self.discrete_actions is not None

    @property
    def num_dof(self) -> int:
        return 2

    def get_gym_space(self) -> spaces.Space:
        if self.is_discrete:
            return spaces.Discrete(len(self.discrete_actions))
        return spaces.Box(
            low=np.array([self.linear_range[0], self.angular_range[0]], dtype=np.float32),
            high=np.array([self.linear_range[1], self.angular_range[1]], dtype=np.float32),
        )

    def decode(self, action: np.ndarray) -> np.ndarray:
        if self.is_discrete:
            idx = int(action[0]) if hasattr(action, "__len__") else int(action)
            a = self.discrete_actions[idx]
            return np.array([a["linear"], 0.0, a["angular"]], dtype=np.float32)
        return np.array([action[0], 0.0, action[1]], dtype=np.float32)

    def resolve_discretization(
        self,
        robot_discrete_actions: Optional[List] = None,
    ) -> "DifferentialDriveActionSpace":
        """Resolve ``discretization`` config into a concrete ``discrete_actions`` list.

        Called by ``_populate_agent_spec`` at training startup.  Returns a new
        copy of this space with ``discrete_actions`` populated (and
        ``discretization`` cleared to avoid double-resolution).

        Args:
            robot_discrete_actions: The robot's built-in discrete action list
                (from ``model_params.yaml``).  Required for ``ROBOT_DEFINED``;
                ignored otherwise.

        Returns:
            A new ``DifferentialDriveActionSpace`` with ``is_discrete=True``.
        """
        if self.discretization is None:
            return self

        from rosnav_rl.utils.action_space.custom_discrete_action import (
            generate_discrete_action_dict,
            generate_exponential_actions,
            generate_navigational_actions,
        )

        strategy = self.discretization.strategy

        if strategy == DiscretizationStrategy.ROBOT_DEFINED:
            if robot_discrete_actions:
                actions = [
                    a.model_dump() if hasattr(a, "model_dump") else dict(a)
                    for a in robot_discrete_actions
                ]
            else:
                # No robot-defined actions — fall back to UNIFORM
                actions = generate_discrete_action_dict(
                    self.linear_range,
                    self.angular_range,
                    self.discretization.buckets_linear,
                    self.discretization.buckets_angular,
                )
        elif strategy == DiscretizationStrategy.NAVIGATIONAL:
            actions = generate_navigational_actions(self.linear_range, self.angular_range)
        elif strategy == DiscretizationStrategy.EXPONENTIAL:
            actions = generate_exponential_actions(
                self.linear_range,
                self.angular_range,
                self.discretization.buckets_linear,
                self.discretization.buckets_angular,
            )
        else:  # UNIFORM (default)
            actions = generate_discrete_action_dict(
                self.linear_range,
                self.angular_range,
                self.discretization.buckets_linear,
                self.discretization.buckets_angular,
            )

        return self.model_copy(update={"discrete_actions": actions, "discretization": None})

    def to_discrete(
        self, buckets_linear: int = 12, buckets_angular: int = 16
    ) -> DifferentialDriveActionSpace:
        """Return a discretised copy using a uniform grid (legacy helper).

        Prefer setting ``discretization=DiscretizationCfg(strategy=...)`` in
        your config and calling ``resolve_discretization()`` instead.
        """
        return self.model_copy(
            update={"discretization": DiscretizationCfg(
                strategy=DiscretizationStrategy.UNIFORM,
                buckets_linear=buckets_linear,
                buckets_angular=buckets_angular,
            )}
        ).resolve_discretization()

    # -- legacy bridge -----------------------------------------------------

    def _to_legacy_actions(self) -> Union[list, dict]:
        """Convert to the format expected by the old ``ActionSpaceManager``."""
        if self.discrete_actions:
            return self.discrete_actions
        return {
            "linear_range": list(self.linear_range),
            "angular_range": list(self.angular_range),
        }


class OmnidirectionalActionSpace(BaseActionSpace):
    """3-DOF holonomic mobile robot: ``[linear_x, linear_y, angular_z]``.

    Common platforms: Ridgeback, custom omni-wheel bases.
    Output command: ``[linear_x, linear_y, angular_z]``.

    Set ``discretization`` to enable auto-discretization at training time.
    """

    type: Literal["omnidirectional"] = "omnidirectional"
    linear_range_x: Tuple[float, float] = (-1.0, 1.0)
    linear_range_y: Tuple[float, float] = (-1.0, 1.0)
    angular_range: Tuple[float, float] = (-1.0, 1.0)
    discrete_actions: Optional[List[Dict[str, Any]]] = None
    discretization: Optional[DiscretizationCfg] = Field(
        None,
        description="Auto-discretization config. Set strategy here; resolved at training time.",
    )

    @property
    def is_discrete(self) -> bool:
        return self.discrete_actions is not None

    @property
    def num_dof(self) -> int:
        return 3

    def get_gym_space(self) -> spaces.Space:
        if self.is_discrete:
            return spaces.Discrete(len(self.discrete_actions))
        return spaces.Box(
            low=np.array(
                [self.linear_range_x[0], self.linear_range_y[0], self.angular_range[0]],
                dtype=np.float32,
            ),
            high=np.array(
                [self.linear_range_x[1], self.linear_range_y[1], self.angular_range[1]],
                dtype=np.float32,
            ),
        )

    def decode(self, action: np.ndarray) -> np.ndarray:
        if self.is_discrete:
            idx = int(action[0]) if hasattr(action, "__len__") else int(action)
            a = self.discrete_actions[idx]
            return np.array(
                [a["linear_x"], a["linear_y"], a["angular"]], dtype=np.float32
            )
        return np.asarray(action[:3], dtype=np.float32)

    def resolve_discretization(
        self,
        robot_discrete_actions: Optional[List] = None,
    ) -> "OmnidirectionalActionSpace":
        """Resolve ``discretization`` config into a concrete ``discrete_actions`` list.

        For omnidirectional robots only ROBOT_DEFINED and UNIFORM are meaningful;
        the other strategies fall back to UNIFORM.
        """
        if self.discretization is None:
            return self

        from rosnav_rl.utils.action_space.custom_discrete_action import (
            generate_discrete_action_dict,
        )

        strategy = self.discretization.strategy

        if strategy == DiscretizationStrategy.ROBOT_DEFINED and robot_discrete_actions:
            actions = [
                a.model_dump() if hasattr(a, "model_dump") else dict(a)
                for a in robot_discrete_actions
            ]
        else:
            # For omni, use uniform grid over x-range (y is treated symmetrically)
            actions = generate_discrete_action_dict(
                self.linear_range_x,
                self.angular_range,
                self.discretization.buckets_linear,
                self.discretization.buckets_angular,
            )

        return self.model_copy(update={"discrete_actions": actions, "discretization": None})

    def _to_legacy_actions(self) -> Union[list, dict]:
        if self.discrete_actions:
            return self.discrete_actions
        return {
            "linear_range": {
                "x": list(self.linear_range_x),
                "y": list(self.linear_range_y),
            },
            "angular_range": list(self.angular_range),
        }


# =====================================================================
#  Manipulators
# =====================================================================


class ManipulatorActionSpace(BaseActionSpace):
    """N-DOF robotic arm: ``[joint_1, ..., joint_n]``.

    Each joint has independent position / velocity limits.
    Output command: joint values directly.
    """

    type: Literal["manipulator"] = "manipulator"
    joint_limits: List[Tuple[float, float]]

    @property
    def num_dof(self) -> int:
        return len(self.joint_limits)

    def get_gym_space(self) -> spaces.Box:
        lows = np.array([lo for lo, _ in self.joint_limits], dtype=np.float32)
        highs = np.array([hi for _, hi in self.joint_limits], dtype=np.float32)
        return spaces.Box(low=lows, high=highs)

    def decode(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32)

    def _to_legacy_actions(self) -> dict:
        return {"joint_limits": [list(lim) for lim in self.joint_limits]}


# =====================================================================
#  Humanoids
# =====================================================================


class HumanoidActionSpace(BaseActionSpace):
    """Humanoid robot with locomotion + upper-body groups.

    Output command: ``[locomotion..., upper_body...]``.
    """

    type: Literal["humanoid"] = "humanoid"
    locomotion_dof: int = 6
    locomotion_range: Tuple[float, float] = (-1.0, 1.0)
    upper_body_joint_limits: List[Tuple[float, float]] = []

    @property
    def num_dof(self) -> int:
        return self.locomotion_dof + len(self.upper_body_joint_limits)

    def get_gym_space(self) -> spaces.Box:
        loco_lo = np.full(self.locomotion_dof, self.locomotion_range[0], dtype=np.float32)
        loco_hi = np.full(self.locomotion_dof, self.locomotion_range[1], dtype=np.float32)
        if self.upper_body_joint_limits:
            ub_lo = np.array([lo for lo, _ in self.upper_body_joint_limits], dtype=np.float32)
            ub_hi = np.array([hi for _, hi in self.upper_body_joint_limits], dtype=np.float32)
            return spaces.Box(
                low=np.concatenate([loco_lo, ub_lo]),
                high=np.concatenate([loco_hi, ub_hi]),
            )
        return spaces.Box(low=loco_lo, high=loco_hi)

    def decode(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32)

    def _to_legacy_actions(self) -> dict:
        return {
            "locomotion_dof": self.locomotion_dof,
            "locomotion_range": list(self.locomotion_range),
            "upper_body_joint_limits": [list(lim) for lim in self.upper_body_joint_limits],
        }


# =====================================================================
#  Discriminated union — auto-selects type from YAML ``type`` field
# =====================================================================

ActionSpaceSpec = Annotated[
    Union[
        DifferentialDriveActionSpace,
        OmnidirectionalActionSpace,
        ManipulatorActionSpace,
        HumanoidActionSpace,
    ],
    Discriminator("type"),
]
