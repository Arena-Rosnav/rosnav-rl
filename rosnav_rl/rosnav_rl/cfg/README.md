# Configuration Package (`cfg/`)

> Back to [README](../../README.md) · [Developer Guide](../../GUIDE.md) · [Tutorials](../../TUTORIALS.md)
>
> Pydantic v2-based type-safe configuration with discriminated unions, auto-validation, and full round-trip YAML serialization.

## Architecture Overview

```
cfg/
├── __init__.py       # Re-exports: AgentCfg, FrameworkCfg, RewardCfg, ActionSpaceCfg, LoggingCfg
├── agent.py          # AgentCfg — top-level agent configuration (discriminated union)
├── framework.py      # FrameworkCfg — abstract base for RL framework configs
├── action_space.py   # ActionSpaceCfg, DiscreteFromBoxActionSpaceCfg
├── reward.py         # RewardCfg, RewardFunctionDict
└── logging.py        # LoggingCfg, VERBOSE_TO_LEVEL, configure_rosnav_rl_logging
```

## Core Configuration Classes

### AgentCfg

The top-level configuration for a complete RL agent. Uses Pydantic v2's `Discriminator` to automatically select the correct framework config based on the `name` field.

```python
from rosnav_rl.cfg import AgentCfg

cfg = AgentCfg(
    name="my_ppo_agent",        # Auto-generated if omitted (via check_name validator)
    robot="jackal",
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
    action_space=ActionSpaceCfg(is_discrete=False),
)
```

**Fields:**
| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Agent name (auto-generated from robot + framework if omitted) |
| `robot` | `str` | Robot identifier (e.g., `"jackal"`, `"turtlebot3_burger"`) |
| `framework` | `Annotated[Union[StableBaselinesCfg, DreamerV3Cfg], Discriminator("name")]` | RL framework config - type is selected based on `name` field |
| `reward` | `RewardCfg` | Reward function configuration |
| `action_space` | `ActionSpaceCfg` | Action space configuration |

**Discriminated Union:**
The `framework` field uses Pydantic's `Discriminator` on the `name` field:
- `name: "stable_baselines3"` → `StableBaselinesCfg`
- `name: "dreamer_v3"` → `DreamerV3Cfg`

This means YAML/JSON configs are automatically routed to the correct class:
```yaml
framework:
  name: stable_baselines3    # ← discriminator value
  algorithm:
    algorithm_name: PPO
    architecture_name: AGENT_1
    parameters:
      learning_rate: 0.0003
```

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

### ActionSpaceCfg

Configuration for the action space.

```python
from rosnav_rl.cfg import ActionSpaceCfg

# Continuous (default)
cfg = ActionSpaceCfg(is_discrete=False)

# Discrete with custom discretization
cfg = ActionSpaceCfg(
    is_discrete=True,
    custom_discretization=DiscreteFromBoxActionSpaceCfg(
        buckets_linear_vel=12,
        buckets_angular_vel=16,
    ),
)
```

`DiscreteFromBoxActionSpaceCfg` generates a discrete action dictionary by uniformly discretizing the continuous linear and angular velocity ranges into `buckets_linear_vel * buckets_angular_vel` actions.

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
cfg = AgentCfg.model_validate(yaml.safe_load(open("agent_config.yaml")))

# To dict (for saving)
data = cfg.model_dump()
yaml.dump(data, open("agent_config.yaml", "w"))

# From JSON
cfg = AgentCfg.model_validate_json(json_string)
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

2. Add to the discriminated union in `agent.py`:
```python
framework: Annotated[
    Union[StableBaselinesCfg, DreamerV3Cfg, MyFrameworkCfg],
    Discriminator(discriminator="name"),
]
```
