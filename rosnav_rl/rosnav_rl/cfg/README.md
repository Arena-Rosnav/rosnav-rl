# Configuration Package (`cfg/`)

> Back to [README](../../README.md) · [Developer Guide](../../GUIDE.md) · [Tutorials](../../TUTORIALS.md)
>
> Pydantic v2-based type-safe configuration with discriminated unions, auto-validation, and full round-trip YAML serialization.

## Architecture Overview

```
cfg/
├── __init__.py       # Re-exports: AgentConfig, AgentParameters, typed action spaces, FrameworkCfg, RewardCfg, LoggingCfg
├── agent.py          # AgentConfig — single source of truth for the agent spec
├── parameters.py     # AgentParameters — unified observation + reward constants (replaces ObservationConfig + EnvironmentConfig)
├── action_spaces.py  # Typed action spaces: DifferentialDriveActionSpace, OmnidirectionalActionSpace, etc.
├── observation.py    # Backward-compat shim: ObservationConfig = AgentParameters
├── framework.py      # FrameworkCfg — abstract base for RL framework configs
├── reward.py         # RewardCfg, RewardFunctionDict
└── logging.py        # LoggingCfg, VERBOSE_TO_LEVEL, configure_rosnav_rl_logging
```

## Core Configuration Classes

### AgentConfig

The **single source of truth** for a complete RL agent. Contains everything the agent needs: robot identity, typed action space, observation spec, environment spec, framework config, and reward config.

```python
from rosnav_rl.cfg import AgentConfig
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace

spec = AgentConfig(
    name="my_ppo_agent",
    robot="jackal",
    action_space=DifferentialDriveActionSpace(
        linear_range=(-2.0, 2.0),
        angular_range=(-4.0, 4.0),
    ),
    framework=StableBaselinesCfg(
        algorithm=PPO_Cfg(
            architecture_name="AGENT_1",
            parameters=PPO_Algorithm_Cfg(learning_rate=3e-4),
        )
    ),
    reward=RewardCfg(
        reward_function_dict={
            "goal_reached":  {"reward": 15.0},
            "collision":     {"reward": -10.0},
        }
    ),
)
```

**Fields:**
| Field | Type | Description |
| --- | --- | --- |
| `name` | `str \| None` | Agent name (auto-generated from robot + framework if omitted) |
| `robot` | `str` | Robot identifier (e.g., `"jackal"`, `"turtlebot3_burger"`) |
| `action_space` | `ActionSpaceSpec` | Typed action space — discriminated union on `type` field |
| `observation` | `ObservationConfig` | Observation parameters (laser beams, velocity ranges, etc.) |
| `environment` | `EnvironmentConfig` | Environment parameters (robot radius, goal radius, max steps) |
| `framework` | `Annotated[Union[StableBaselinesCfg, DreamerV3Cfg], Discriminator("name")]` | RL framework config |
| `reward` | `RewardCfg \| None` | Reward function configuration (optional for inference) |
| `logging` | `LoggingCfg` | Per-namespace logging levels |

> **Arena training flow:** When using `arena_training`, you only need to specify `name`, `framework`, and `reward` in the YAML template. The `action_space`, `observation`, and `environment` fields are **derived automatically** from the robot's `model_params.yaml` and the training config.

### Typed Action Spaces

Each robot kinematic type has its own action space model with robot-specific fields. These replace the old `ActionSpaceCfg(is_discrete=...)`.

```python
from rosnav_rl.cfg.action_spaces import (
    DifferentialDriveActionSpace,
    OmnidirectionalActionSpace,
    HumanoidActionSpace,
    ManipulatorActionSpace,
)

# Non-holonomic robot (2-DOF: linear, angular)
diff_drive = DifferentialDriveActionSpace(
    linear_range=(-2.0, 2.0),
    angular_range=(-4.0, 4.0),
)

# Holonomic robot (3-DOF: linear_x, linear_y, angular)
omni = OmnidirectionalActionSpace(
    linear_range_x=(-1.0, 1.0),
    linear_range_y=(-1.0, 1.0),
    angular_range=(-2.0, 2.0),
)

# Discrete actions
discrete_drive = DifferentialDriveActionSpace(
    linear_range=(-2.0, 2.0),
    angular_range=(-4.0, 4.0),
    discrete_actions=[
        {"linear": 1.0, "angular": 0.0},
        {"linear": 0.0, "angular": 1.0},
        {"linear": 0.0, "angular": -1.0},
    ],
)
```

**Common properties** (all action space types):
| Property | Returns |
| --- | --- |
| `is_discrete` | `bool` — whether discrete actions are configured |
| `is_holonomic` | `bool` — whether the space includes lateral motion |
| `num_actions` | `int` — discrete action count or continuous DOF |
| `to_discrete()` | Convert continuous ranges to a default discrete grid |
| `get_gym_space()` | `gymnasium.spaces.Box` or `gymnasium.spaces.Discrete` |

**Discriminated union** — the `type` field selects the correct class from YAML:
```yaml
action_space:
  type: differential_drive    # ← discriminator value
  linear_range: [-2.0, 2.0]
  angular_range: [-4.0, 4.0]
```

### ObservationConfig

Parameters for observation normalization — laser config, velocity bounds, and semantic (pedestrian) data.

```python
### AgentParameters — unified observation + reward constants

`AgentParameters` is the single config model for all scalar constants. It
replaces the former `ObservationConfig` + `EnvironmentConfig` split.

```python
from rosnav_rl.cfg.parameters import AgentParameters

params = AgentParameters(
    laser_num_beams=720,
    laser_max_range=30.0,
    min_linear_vel=-2.0,
    max_linear_vel=2.0,
    min_angular_vel=-4.0,
    max_angular_vel=4.0,
    robot_radius=0.267,
    safety_distance=1.0,
    goal_radius=0.33,
    max_steps=350,
)

# Observation-space subset (17 keys, fed to every BaseObservationSpace constructor)
obs_kwargs = params.observation_kwargs()

# Reconstruct at inference time from a saved AgentConfig
params = AgentParameters.from_spec(agent_config)
```

> During training, `AgentParameters` is populated automatically from the robot
> description and arena config, then written to `agent.yaml`. **Review the
> `parameters:` block before training** — especially `laser_num_beams`,
> `laser_max_range`, velocity bounds, `robot_radius`, and `goal_radius`.

> `ObservationConfig` and `EnvironmentConfig` are kept as backward-compat aliases
> (`ObservationConfig = AgentParameters`) so existing code still works.

### FrameworkCfg

Abstract base class for all RL framework configurations. Only requires a `name` field.

```python
from rosnav_rl.cfg import FrameworkCfg

class MyFrameworkCfg(FrameworkCfg):
    name: str = "my_framework"
    # ... framework-specific fields
```

### RewardCfg

Configuration for the reward function.

```python
from rosnav_rl.cfg import RewardCfg

cfg = RewardCfg(
    reward_function_dict={
        "goal_reached":   {"reward": 15.0},
        "collision":      {"reward": -10.0, "bumper_zone": 0.05},
        "approach_goal":  {"pos_factor": 0.3, "neg_factor": 0.5, "_on_safe_dist_violation": True},
        "safe_distance":  {"reward": -0.15},
    },
    reward_unit_kwargs=None,   # Additional global kwargs for all units
    verbose=False,             # Enable reward breakdown logging
)
```

**Type aliases:**
```python
RewardUnitDict = Dict[str, Any]           # {"reward": 15.0, "_follow_subgoal": False}
RewardFunctionDict = Dict[str, RewardUnitDict]  # {"goal_reached": {...}, "collision": {...}}
```

### LoggingCfg

Per-namespace logging levels for rosnav_rl components.

```python
from rosnav_rl.cfg import LoggingCfg, configure_rosnav_rl_logging

cfg = LoggingCfg(
    default_level="INFO",
    overrides={
        "rosnav_rl.observations": "WARNING",  # Silence per-step obs noise
        "rosnav_rl.reward": "WARNING",         # Silence per-step reward breakdown
        "rosnav_rl.spaces": "WARNING",         # Silence space auto-load info
    },
)

configure_rosnav_rl_logging(cfg)
```

**Verbose mapping** (`VERBOSE_TO_LEVEL`):
| Verbose int | Log level |
| --- | --- |
| 0 | `WARNING` |
| 1 | `INFO` |
| 2 | `DEBUG` |

## SB3 Config Hierarchy

The Stable Baselines 3 integration uses a **three-tier Pydantic hierarchy** that cleanly separates universal, family-level, and algorithm-specific parameters. See the [Model Package README](../model/README.md) for full details.

```
SBAlgorithmParameters              (universal - every SB3 algorithm)
├── OnPolicyParameters             (PPO, A2C, TRPO, RecurrentPPO)
│   ├── PPO_Algorithm_Cfg
│   ├── A2C_Algorithm_Cfg
│   └── TRPO_Algorithm_Cfg
└── OffPolicyParameters            (SAC, TD3, DDPG, TQC, CrossQ)
    ├── SAC_Algorithm_Cfg
    ├── TD3_Algorithm_Cfg
    ├── TQC_Algorithm_Cfg
    └── CrossQ_Algorithm_Cfg
```

## Serialization

All configs support full `model_validate()` / `model_dump()` round-trip:

```python
# From dict/YAML
spec = AgentConfig.model_validate(yaml.safe_load(open("agent_config.yaml")))

# To dict (for saving)
data = spec.model_dump()
yaml.dump(data, open("agent_config.yaml", "w"))

# From YAML file (convenience method)
spec = AgentConfig.from_yaml("agent_config.yaml")
spec.to_yaml("agent_config_copy.yaml")
```

## Extending

### Adding a new RL framework

1. Create a `FrameworkCfg` subclass:
```python
class MyFrameworkCfg(FrameworkCfg):
    name: str = "my_framework"
    learning_rate: float = 1e-3
    batch_size: int = 64
```

2. Add to the discriminated union in `agent_spec.py`:
```python
framework: Annotated[
    Union[StableBaselinesCfg, DreamerV3Cfg, MyFrameworkCfg],
    Discriminator(discriminator="name"),
]
```

### Adding a new action space type

1. Add a new `BaseActionSpace` subclass in `action_spaces.py`:
```python
class QuadrupedActionSpace(BaseActionSpace):
    type: Literal["quadruped"] = "quadruped"
    gait_type: str = "trot"
    stride_range: tuple[float, float] = (0.0, 0.5)
    # ...
```

2. Add to the `ActionSpaceSpec` discriminated union.
