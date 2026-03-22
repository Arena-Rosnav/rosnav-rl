# RosNav-RL Developer Guide

> This guide is the comprehensive technical reference for RosNav-RL.
> It covers architecture, design patterns, code organization, and every core
> abstraction. For step-by-step tutorials see **[TUTORIALS.md](TUTORIALS.md)**.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture](#2-architecture)
3. [Setup & Installation](#3-setup--installation)
4. [Code Organization](#4-code-organization)
5. [Core Concepts](#5-core-concepts)
6. [SB3 Configuration Architecture](#6-sb3-configuration-architecture)
7. [Development Workflow](#7-development-workflow)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)

---

## 1. Executive Summary

**RosNav-RL** is a framework for building deep RL navigation agents in ROS 2.
It decouples algorithm, observations, reward, and deployment into independent,
composable modules so you can swap any part without touching the others.

### Technologies

| Technology | Role |
| --- | --- |
| ROS 2 Humble | Communication backbone |
| Python 3.10+ | Core language |
| PyTorch | Neural network backend |
| Stable-Baselines3 + sb3-contrib | 9 RL algorithms (PPO, A2C, TRPO, RecurrentPPO, SAC, TD3, DDPG, TQC, CrossQ) |
| DreamerV3 | World-model RL framework |
| Pydantic v2 | Type-safe configuration with discriminated unions |

### Key Capabilities

| Capability | How |
| --- | --- |
| Multi-algorithm | 9 SB3 algorithms + DreamerV3. Add a new one with a config file. |
| Composable spaces | Registry-based observation spaces. Parallel encoding via `ThreadPoolExecutor`. |
| Modular rewards | Declarative reward units in YAML. Parallel evaluation, safety categorization. |
| YAML observation pipeline | Collectors → Generators → dependency resolution via topological sort. |
| Schema-based validation | `RequiresProtocol` enforces typed dependency declarations across all subsystems. |
| Validate-once | First-call validation then zero-overhead steady-state in both spaces and rewards. |
| One-command deploy | `GetCommand` ROS 2 service wrapping any trained agent. |

---

## 2. Architecture

### Philosophy

The core principle is **modularity and separation of concerns**:

- 🧠 **RL Framework** — the brain (SB3, DreamerV3, or your own)
- 🔭 **Observations** — the eyes (ROS topic collectors + derived generators)
- 🚀 **Spaces** — the translator (encode observations, decode actions)
- 🎁 **Reward** — the motivation (composable reward units)
- ⚙️ **Config** — the control panel (Pydantic v2, discriminated unions)

### Design Patterns

| Pattern | Usage | Examples |
| --- | --- | --- |
| **Factory + Registry** | Class registration via decorators; lookup by name | `ModelFactory`, `SpaceFactory`, `RewardUnitFactory`, `AgentFactory`, `ObservationFactory` |
| **Protocol** | Shared contracts across subsystems | `RequiresProtocol` (schema-based deps), `ObservationCollector` (duck-typed interface) |
| **Mixin** | Cross-cutting concerns injected via MRO | `ErrorReportingMixin` (structured logging) |
| **Strategy** | Swappable execution strategies | `CollectorManager`, `GeneratorManager`, `SubscriptionManager`, `WaitingStrategy` |
| **Composite** | Tree structures evaluated as one | `RewardFunction` (aggregates `RewardUnit`s) |
| **Discriminated Union** | Type-safe polymorphic configs | `AgentConfig.framework` (Pydantic `Discriminator`), `ActionSpaceSpec` |

### Key Components

<p align="center">
  <img width="70%" src="img/rosnav_rl.png" />
</p>

| Component | Description |
| --- | --- |
| **RL Agent** | Top-level orchestrator. Owns model + spaces + reward function. |
| **Model** | `RL_Model` ABC. Implementations: `StableBaselinesModel`, `DreamerV3Model`. `ModelFactory` registry. |
| **Observations** | `ObservationManager` with `Collector` (ROS topics) and `Generator` (derived features). YAML-configured. |
| **Spaces** | `ObservationSpaceManager` (parallel encoding, validate-once) + `ActionSpaceManager` (discrete/continuous, holonomic). |
| **Reward** | `RewardFunction` composite. `RewardUnit` ABC with `RequiresProtocol` + `ErrorReportingMixin`. Parallel execution. |
| **Action Server** | ROS 2 `GetCommand` service. `ActionServer` ABC → `ArenaActionServer`. |
| **Cross-cutting** | `RequiresProtocol`, `ErrorReportingMixin`, `SchemaValidator`, `MissingObservationError` (smart suggestions). |

### Data Flow

<p align="center">
  <img width="60%" src="img/dataflow.png" />
</p>

```
1. Sensor Data Collection
   ObservationManager → Collectors grab ROS 2 topics → Generators derive features
   → structured ObservationDict

2. Observation Encoding
   ObservationSpaceManager → selects, normalizes, stacks observations
   → encoded tensor for the neural network

3. Action Selection
   RL_Model (policy network) → outputs raw action

4. Action Decoding
   ActionSpaceManager → decodes to action vector (shape depends on robot type)
   → float64[] action  +  string action_type

5. Execution
   Command sent to robot hardware / simulator
```

---

## 3. Setup & Installation

### Prerequisites

- ROS 2 Humble
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

```bash
# Clone
cd ~/colcon_ws/src
git clone https://github.com/Arena-Rosnav/rosnav-rl.git

# Python dependencies (uv creates a .venv and installs all deps)
cd rosnav-rl/rosnav_rl
uv sync

# Build ROS packages
cd ~/colcon_ws
colcon build --packages-select rosnav_rl rosnav_rl_msgs

# Source
source install/setup.bash
```

### Verify

```bash
# Run tests
cd ~/colcon_ws/src/rosnav-rl/rosnav_rl
python3 -m pytest tests/ -v

# Create a smoke-test agent (random weights)
python3 scripts/create_test_agent.py --agent-name test_agent
```

This writes `training_config.yaml` and `best_model.zip` (random weights) to
`Arena/arena_training/agents/test_agent/`.

---

## 4. Code Organization

```
rosnav_rl/
├── rl_agent.py              # RL_Agent — top-level orchestrator
├── __init__.py              # Public API re-exports
│
├── action_server/           # ROS 2 service server (GetCommand)
│   ├── base_server.py       # ActionServer ABC — ROS 2 service, ObservationCollector protocol
│   └── arena_server.py      # ArenaActionServer — agent loading, stores spec.parameters as AgentParameters
│
├── cfg/                     # Top-level Pydantic configuration
│   ├── agent.py             # AgentConfig — single source of truth, EnvironmentConfig
│   ├── action_spaces.py     # Typed action spaces: DifferentialDrive, Omnidirectional, DiscretizationCfg, etc.
│   ├── observation.py       # ObservationConfig — laser, velocity, semantic params
│   ├── framework.py         # FrameworkCfg ABC
│   ├── reward.py            # RewardCfg, RewardFunctionDict
│   └── logging.py           # LoggingCfg, VERBOSE_TO_LEVEL, configure_rosnav_rl_logging
│
├── model/                   # RL model implementations
│   ├── model.py             # RL_Model ABC (train, get_action, save, load, from_framework_cfg)
│   ├── model_factory.py     # ModelFactory — registry-based, no if/elif branching
│   ├── stable_baselines3/   # SB3 framework integration
│   │   ├── sb3_model.py     # StableBaselinesModel (structural _is_recurrent check)
│   │   ├── cfg/             # 3-tier config hierarchy
│   │   │   ├── base.py      # SBAlgorithmParameters → On/OffPolicyParameters
│   │   │   ├── ppo.py       # PPO_Algorithm_Cfg, PPO_Cfg
│   │   │   ├── a2c.py       # A2C_Algorithm_Cfg, A2C_Cfg
│   │   │   ├── trpo.py      # TRPO_Algorithm_Cfg, TRPO_Cfg
│   │   │   ├── sac.py       # SAC_Algorithm_Cfg, SAC_Cfg
│   │   │   ├── td3.py       # TD3_Algorithm_Cfg, TD3_Cfg
│   │   │   ├── tqc.py       # TQC_Algorithm_Cfg, TQC_Cfg
│   │   │   ├── crossq.py    # CrossQ_Algorithm_Cfg, CrossQ_Cfg
│   │   │   └── framework.py # StableBaselinesCfg (Union of all algorithm cfgs)
│   │   └── policy/          # Neural-network architecture descriptions
│   │       ├── agent_factory.py  # AgentFactory registry
│   │       ├── base_policy.py    # StableBaselinesPolicyDescription ABC
│   │       └── constants.py      # POLICY_TYPE & BASE_AGENT_ATTR (all 9 algos)
│   └── dreamerv3/           # DreamerV3 world-model integration
│
├── observations/            # Observation pipeline
│   ├── core/                # ObservationManager, ObservationPipeline
│   ├── data_sources/        # Collector[M,T] & Generator[T] + implementations
│   │   ├── base.py          # DataSource ABC, Collector, Generator
│   │   ├── collectors.py    # 9+ concrete collectors
│   │   └── generators.py    # 15+ generators
│   ├── strategies/          # Pluggable execution strategies
│   │   ├── collector.py     # CollectorManager
│   │   ├── generator.py     # GeneratorManager (dependency-ordered execution)
│   │   ├── subscription.py  # SubscriptionManager (message_filters sync)
│   │   └── waiting.py       # WaitingStrategy (first-message blocking)
│   ├── factory/             # YAML-driven construction
│   │   ├── factory.py       # ObservationFactory
│   │   └── resolver.py      # DependencyResolver (topological sort / Kahn's)
│   └── observations.yaml    # Default pipeline config
│
├── reward/                  # Reward system
│   ├── reward_function.py   # RewardFunction (composite, parallel ThreadPoolExecutor)
│   ├── constants.py         # REWARD_CONSTANTS, DONE_REASONS, per-unit DEFAULTS
│   ├── utils.py             # @check_params decorator
│   └── reward_units/
│       ├── base_reward_units.py   # RewardUnit ABC (RequiresProtocol + ErrorReportingMixin)
│       ├── reward_unit_factory.py # RewardUnitFactory (registry + @register)
│       └── reward_units.py        # 15+ concrete units
│
├── spaces/                  # Action & observation spaces
│   ├── space_manager/
│   │   └── base_space_manager.py  # BaseSpaceManager (encode + decode)
│   ├── observation_space/
│   │   ├── observation_space_manager.py  # ObservationSpaceManager (parallel encoding)
│   │   ├── observation_space_factory.py  # SpaceFactory (auto_name, aliases, categories)
│   │   ├── space_categories.py           # SpaceCategory enum (6 categories)
│   │   ├── normalization.py              # 4 normalizers (max_abs, min_max, standard, identity)
│   │   └── spaces/                       # Concrete observation spaces
│   │       ├── base_observation_space.py  # BaseObservationSpace ABC
│   │       ├── perception/               # Laser, vision
│   │       ├── navigation/               # Goal distance/angle, subgoal, plan
│   │       ├── dynamics/                 # Last action, velocity, kinematics
│   │       ├── environment/              # Pedestrians, obstacles
│   │       ├── localization/             # Robot pose
│   │       └── meta/                     # Step count, terminal flags
│   └── action_space/
│       └── action_space_manager.py       # ActionSpaceManager (discrete/continuous, holonomic)
│
├── states/                  # State containers
│   ├── simulation/
│   │   ├── container.py     # backward-compat shim: SimulationStateContainer = AgentParameters
│   │   └── states.py        # RobotState, TaskState, LaserState, VelocityState, …
│   └── distributor.py       # State distribution placeholder
│
└── utils/                   # Shared utilities
    ├── type_aliases/        # TypedDict, Protocol, SupportedRLFrameworks
    ├── validation/          # RequiresProtocol, SchemaValidator, MissingObservationError
    ├── logging/             # ErrorReportingMixin, ErrorCollector, ComponentType
    ├── curriculum/          # Curriculum learning utilities
    ├── rostopic/            # Namespace and topic helpers
    └── stable_baselines3/   # SB3-specific utilities
```

---

## 5. Core Concepts

### 5.1 The Agent

**`RL_Agent`** (`rl_agent.py`) is the top-level orchestrator. It wires together model, spaces, and reward from a single **`AgentConfig`**:

```python
class RL_Agent:
    def __init__(self, spec: AgentConfig):
        self._initialize_model(spec)          # → ModelFactory.create_model_instance()
        self._initialize_space_manager(spec)   # → BaseSpaceManager(...)
        self._initialize_reward_function(spec) # → RewardFunction(...) if reward config exists
```

`AgentConfig` is a Pydantic v2 model that serves as the **single source of truth** for the entire agent configuration — framework, typed action space, observation spec, environment spec, reward, and logging.

When training with Arena, the action space, observation, and environment specs are **derived automatically** from the robot's `model_params.yaml` (via `arena_robots`) and the training config. You never need to specify laser beam counts, velocity ranges, or action limits manually.

| Method / Property | Description |
| --- | --- |
| `initialize_model()` | Calls `model.setup_model()` to build the underlying algorithm |
| `load_model(path)` | Loads weights from disk (skips if already initialized) |
| `train(train_envs, eval_envs)` | Delegates to `model.train()` |
| `get_action(obs_dict)` | Returns `np.ndarray` from the policy network |
| `config` | Dict with spec, model config, space config |
| `model` | The `RL_Model` implementation (SB3 or DreamerV3) |
| `reward_function` | `RewardFunction` instance (or `None`) |
| `space_manager` | `BaseSpaceManager` (observation encoding + action decoding) |
| `observation_space` | `gymnasium.spaces.Dict` or single `Space` |
| `action_space` | `gymnasium.spaces.Discrete` or `Box` |

### 5.2 The Model Layer

**`RL_Model`** (`model/model.py`) is the abstract base every framework must implement:

| Abstract Method | Purpose |
| --- | --- |
| `setup_model()` | Initialize the algorithm |
| `train()` | Run a training iteration |
| `save()` / `load()` | Persist & restore weights |
| `get_action(obs)` | Inference |
| `from_framework_cfg()` | Classmethod — construct from a `FrameworkCfg` |

| Property | Purpose |
| --- | --- |
| `model` | Access the wrapped algorithm object |
| `algorithm_cfg` | Pydantic config |
| `observation_space_list` | List of `BaseObservationSpace` classes |
| `stack_size` | Temporal frame-stacking depth (default: 1) |
| `parameter_number` | Total trainable parameters |

**`ModelFactory`** maps framework identifiers → `RL_Model` subclasses. Registration via decorator:

```python
@ModelFactory.register(SupportedRLFrameworks.STABLE_BASELINES3)
class StableBaselinesModel(RL_Model): ...
```

Creation is fully delegated — no `if/elif` branching:

```python
model = ModelFactory.create_model_instance(framework_cfg=cfg, rl_agent=agent)
# Internally calls model_class.from_framework_cfg()
```

See [model/README.md](rosnav_rl/model/README.md) for full SB3 integration details.

### 5.3 Observations

The observation pipeline collects raw sensor data and derives features:

```
ROS Topics ──▶ Collectors ──▶ Generators ──▶ ObservationDict
```

- **`Collector[M, T]`** — subscribes to a ROS 2 topic, preprocesses via `_preprocess(msg) → T`
- **`Generator[T]`** — derives features from collector/generator outputs via `_generate(**deps) → T`
- **`DependencyResolver`** — topological sort (Kahn's algorithm) for generator execution order
- **`ObservationManager`** — central entry point, configured from YAML

```yaml
# observations.yaml
aliases:
  robot_pose: robot_pose_from_tf
datasources:
  front_laser:
    type: sensor_msgs/LaserScan    # ROS message type → resolves to LaserScanCollector
    params: { topic: "lidar" }
  dist_angle_to_goal:
    type: DistAngleToGoalGenerator # Generators always use class names
    params: {}
```

See [observations/README.md](rosnav_rl/observations/README.md) for the full DataSource hierarchy and strategy details.

### 5.4 Spaces

**Observation spaces** encode raw observations into neural-network-ready tensors.
**Action spaces** decode model outputs into robot commands.

**`BaseSpaceManager`** orchestrates both:

```python
encoded = space_manager.encode_observation(obs_dict)  # → np.ndarray or Dict
cmd     = space_manager.decode_action(raw_action)      # → [linear.x, linear.y, angular.z]
```

**`ObservationSpaceManager`** features:
- Parallel encoding via `ThreadPoolExecutor` (up to 4 workers)
- Validate-once on first call, then zero-overhead
- Auto-collapse: single space → direct return, multiple → `Dict`

**`SpaceFactory`** registration:
```python
@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class MyLaserSpace(BaseObservationSpace):
    requires = {"front_laser": LidarRanges}
    ...
```
Four registration modes: explicit name, auto-derived, with aliases, with category.

**`BaseObservationSpace`** provides:
- Schema-based `requires` (validated via `RequiresProtocol`)
- Normalization (`max_abs`, `min_max`, `standard`, `identity`)
- Decorators: `@apply_norvomalization`, `@check_dtype`
- Null fallback on encoding errors

**`ActionSpaceManager`** handles:
- Discrete (`spaces.Discrete`) vs continuous (`spaces.Box`)
- Holonomic (3-DOF) vs non-holonomic (2-DOF → 3-DOF with zero linear.y)

**Space categories** (`SpaceCategory` enum):

| Category | Examples |
| --- | --- |
| PERCEPTION | LaserScanSpace, ReducedLaserScanSpace, RGBDSpace |
| NAVIGATION | DistAngleToGoalSpace, RobustGoalSpace |
| DYNAMICS | LastActionSpace, MotionStateSpace, KinematicStateSpace |
| ENVIRONMENT | EnvironmentContextSpace, SpatialAwarenessSpace |
| LOCALIZATION | RobustOdometrySpace, PoseStabilizedSpace |
| META | IsFirstStepSpace, EpisodeStepSpace, MissionContextSpace |

See [spaces/README.md](rosnav_rl/spaces/README.md) for encoding/decoding pipeline diagrams.

### 5.5 Reward System

**`RewardFunction`** orchestrates a collection of `RewardUnit`s:

- **Parallel execution** — `ThreadPoolExecutor` (configurable workers + timeout)
- **Safety categorization** — units with `_on_safe_dist_violation=True` are only evaluated during safety violations
- **Validate-once** — `RequiresProtocol` validation on first `calculate_reward()` call, then disabled
- **Thread-safe** — `threading.Lock` for reward accumulation in parallel mode

```python
reward, info = reward_fn.calculate_reward(observations, simulation_state_container)
```

**`RewardUnit`** ABC (inherits `ErrorReportingMixin` + `RequiresProtocol`):
- `requires: ClassVar[Dict[str, Any]]` — typed dependency schema
- `add_reward(value)` — validates against NaN/Inf before accumulating
- `add_info(info)` — contributes to episode metadata
- `@check_params` on `__init__` — warns about large reward magnitudes (|reward| > 100)

Built-in units (15+):

| Category | Units |
| --- | --- |
| **Goal** | `goal_reached`, `approach_goal` |
| **Safety** | `safe_distance`, `collision`, `ped_safe_distance`, `obs_safe_distance`, `ped_type_safety_distance`, `ped_type_collision` |
| **Movement** | `no_movement`, `distance_travelled`, `reverse_drive`, `abrupt_velocity_change`, `root_velocity_difference`, `two_factor_velocity_difference` |

See [reward/README.md](rosnav_rl/reward/README.md) for the full reward unit reference.

### 5.6 Configuration

All configs are **Pydantic v2 `BaseModel`** instances supporting `model_validate()` / `model_dump()` round-trip.

**`AgentConfig`** — single source of truth for a complete RL agent:

```python
AgentConfig(
    name="my_agent",           # auto-generated if omitted
    robot="jackal",             # one has to specifyaction_space/observation/environment manually
    observations_config="path/to/observations.yaml",  # optional; relative to config file
    discretization=DiscretizationCfg(strategy="navigational"),  # optional; omit for continuous
    framework=StableBaselinesCfg(...),  # discriminated union on framework.name
    reward=RewardCfg(...),
)
```

`action_space`, `observation`, and `environment` are **`None` in the input YAML** and filled in by
the trainer from the robot description.

**Typed action spaces** — each robot kinematic type has its own Pydantic model
(`DifferentialDriveActionSpace`, `OmnidirectionalActionSpace`, etc.). The trainer always derives
the `type` and velocity ranges from the robot's `model_params.yaml`; you never need to specify
them in the training YAML.

**Discrete action configuration** — set `discretization` at the `agent_config` level to request
discrete actions.  The trainer calls `action_space.resolve_discretization()` at startup which
populates `action_space.discrete_actions` and switches the gym space to `gymnasium.spaces.Discrete`.

Available strategies:

| Strategy | Description | Extra fields |
|---|---|---|
| `robot_defined` | Use the robot’s built-in action list from `model_params.yaml` | — |
| `uniform` | Regular N×M grid over the velocity ranges | `buckets_linear` (default 7), `buckets_angular` (default 9) |
| `navigational` | ~12 hand-crafted forward-biased actions — fastest convergence | — |
| `exponential` | Log-spaced grid — fine control near zero, coarse at extremes | `buckets_linear` (default 5), `buckets_angular` (default 7) |

In a YAML training config:

```yaml
agent_config:
  name: my_agent
  observations_config: observations/observations.yaml  # relative to this config file
  discretization:
    strategy: navigational  # omit the discretization block entirely for continuous actions
  framework:
    name: stable_baselines3
    ...
```

The `framework` field is a discriminated union — Pydantic routes dicts to the correct class based on the `name` field:
- `"stable_baselines3"` → `StableBaselinesCfg`
- `"dreamer_v3"` → `DreamerV3Cfg`

See [cfg/README.md](rosnav_rl/cfg/README.md) for all config classes and serialization examples.

### 5.7 Agent Parameters (formerly SimulationStateContainer)

**`AgentParameters`** (`cfg/parameters.py`) is the single config model for all
scalar constants consumed by the observation pipeline, reward units, and
generators. There is no separate "simulation state container" — it was absorbed.

- **Observation-space fields** (laser, velocity, pedestrian, navigation, normalize)
  — fed via `observation_kwargs()` at construction time.
- **Reward/generator fields** (`robot_radius`, `safety_distance`, `goal_radius`,
  `max_steps`) — read per-step directly from this object.

At inference time use `spec.parameters` directly, or
`AgentParameters.from_spec(agent_config)` as a convenience wrapper.

### 5.8 Deployment

**`ActionServer`** (ABC) wraps any trained agent in a ROS 2 service:
- **Service**: `get_command` (`rosnav_rl_msgs/srv/GetCommand`)
  - Response: `string action_type` + `float64[] action`
  - `action_type` matches `BaseActionSpace.type` (e.g. `"differential_drive"`, `"manipulator"`)
  - `action` layout: `[vx, wz]` for diff-drive; `[vx, vy, wz]` for omni; `[j1..jN]` for manipulators
- **Protocol**: `ObservationCollector` — duck-typed, any object with `get_observations() → ObservationDict`
- **Error handling**: logs warnings, returns empty `action[]` on transient failures
- **Scene reset**: subscribes to `/scenario_reset` and calls `agent.model.reset()`

**`ArenaActionServer`** — Arena-specific implementation:
- Resolves agent directory via `ROSNAV_AGENTS_DIR`, `ament_index`, or path search
- Loads `training_config.yaml` → uses `spec.parameters` (`AgentParameters`) directly
- Creates `ObservationManager` from bundled or agent-specific `observations.yaml`

```bash
# Standalone
ros2 run rosnav_rl action_server.py --ros-args -p agent_name:=my_agent

# Arena integration (automatic)
arena launch local_planner:=rosnav_rl agent_name:=my_agent
```

See [action_server/README.md](rosnav_rl/action_server/README.md) for implementation details.

### 5.9 Cross-Cutting Protocols

These shared abstractions keep the system consistent:

| Protocol / Mixin | What it does | Used by |
| --- | --- | --- |
| **`RequiresProtocol`** | Schema-based dependency declarations via `requires: ClassVar[Dict[str, Any]]` | `BaseObservationSpace`, `Generator`, `RewardUnit` |
| **`ErrorReportingMixin`** | Structured error collection, categorized by `ComponentType` enum, with batch flush | `BaseObservationSpace`, `RewardUnit`, `Generator` |
| **`SchemaValidator`** | Runtime validation of `requires` dicts with smart typo suggestions | Used by validation framework |
| **`MissingObservationError`** | Rich error messages listing available alternatives | Raised during validation |

---

## 6. SB3 Configuration Architecture

The Stable Baselines 3 integration uses a **three-tier Pydantic hierarchy**:

```
SBAlgorithmParameters              (universal — every SB3 algorithm)
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

**Tier 1 — `SBAlgorithmParameters`**: `total_timesteps`, `learning_rate`, `batch_size`, `gamma`, `device`, `seed`, …

**Tier 2a — `OnPolicyParameters`**: `total_batch_size`, `n_epochs`, `gae_lambda`, `ent_coef`, `vf_coef`, `max_grad_norm`, `use_sde`, …

**Tier 2b — `OffPolicyParameters`**: `buffer_size`, `learning_starts`, `tau`, `train_freq`, `gradient_steps`, `optimize_memory_usage`, …

**Tier 3**: Each algorithm cfg only declares fields **unique** to it; everything else is inherited.

### Supported Algorithms

| Algorithm | Family | Library | Config |
| --- | --- | --- | --- |
| [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) | On-policy | stable_baselines3 | `PPO_Cfg` |
| [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) | On-policy | stable_baselines3 | `A2C_Cfg` |
| [TRPO](https://sb3-contrib.readthedocs.io/en/master/modules/trpo.html) | On-policy | sb3_contrib | `TRPO_Cfg` |
| [RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) | On-policy | sb3_contrib | `PPO_Cfg` (LSTM arch) |
| [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) | Off-policy | stable_baselines3 | `SAC_Cfg` |
| [TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html) | Off-policy | stable_baselines3 | `TD3_Cfg` |
| [DDPG](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html) | Off-policy | stable_baselines3 | `SBAlgorithmCfg` |
| [TQC](https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html) | Off-policy | sb3_contrib | `TQC_Cfg` |
| [CrossQ](https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html) | Off-policy | sb3_contrib | `CrossQ_Cfg` |

### Envelope Pattern

Each algorithm params class is wrapped in an `SBAlgorithmCfg` envelope holding:
- `architecture_name` — registry key for the neural-network description
- `checkpoint` — model file to load (default `"last_model"`)
- `parameters` — algorithm-specific hyper-parameters
- `normalization` — optional `VecNormalize` settings
- `callbacks` — training callback config
- `transfer_weights` — optional weight-transfer config

### Framework Envelope

`StableBaselinesCfg` is a discriminated union of all `*_Cfg` classes:

```python
cfg = StableBaselinesCfg.model_validate({
    "name": "stable_baselines3",
    "algorithm": {
        "architecture_name": "AGENT_1",
        "parameters": {
            "algorithm_name": "PPO",
            "total_timesteps": 5_000_000,
        }
    }
})
```

### Recurrent Models

`RecurrentPPO` (LSTM-based PPO) uses the same `StableBaselinesModel` class.
Recurrence is detected **structurally** at runtime:

```python
@property
def _is_recurrent(self) -> bool:
    return hasattr(self._model.policy, "lstm_actor")
```

No `isinstance` check needed — the code is agnostic to the concrete algorithm class.

### Policy Descriptions

Neural-network architectures are `StableBaselinesPolicyDescription` subclasses registered with `AgentFactory`:

```python
@AgentFactory.register("AGENT_1")
class AGENT_1(StableBaselinesPolicyDescription):
    algorithm_class = PPO
    observation_spaces = [
        spaces.perception.ReducedLaserScanSpace,
        spaces.navigation.DistAngleToSubgoalSpace,
        spaces.dynamics.LastActionSpace,
    ]
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU
```

The `POLICY_TYPE` dict in `constants.py` maps each SB3 algorithm class to its policy string (`"MultiInputPolicy"` or `"MultiInputLstmPolicy"`).

See [model/README.md](rosnav_rl/model/README.md) for the complete reference.

---

## 7. Development Workflow

### Training Workflow

1. **Create a Gym environment** — `gym.Env` subclass that interfaces with your simulation. Use `ObservationManager`, `BaseSpaceManager`, and `RewardFunction` in `step()` and `reset()`.

2. **Configure the agent** — `AgentConfig` with framework, typed action space, and reward settings. When training with Arena, the action space, observation, and environment specs are derived automatically from the robot description.

3. **Write a training script**:
   ```python
   agent = RL_Agent(spec)
   agent.initialize_model()
   agent.train(train_envs=envs, eval_envs=eval_envs)
   ```

### Deployment

```bash
# Standalone
ros2 run rosnav_rl action_server.py --ros-args -p agent_name:=my_agent

# Arena (automatic)
arena launch local_planner:=rosnav_rl agent_name:=my_agent
```

The server exposes `get_command` under the robot's namespace. Response contains `action_type` (e.g. `"differential_drive"`) and `action` (decoded command vector). On inference errors it returns empty `action[]` and logs a warning.

### Agent Directory Structure

```
arena_training/agents/<agent_name>/
├── training_config.yaml    # Full TrainingCfg (AgentConfig + ArenaCfg)
├── best_model.zip          # SB3 model checkpoint
└── observations.yaml       # (Optional) agent-specific observation config
```

The agent directory location is configurable via a 3-level fallback chain:
1. `agents_dir` field in `TrainingCfg` (highest priority)
2. `ROSNAV_AGENTS_DIR` environment variable
3. Default: `arena_training/agents/`

See [Tutorial 10](TUTORIALS.md#10-configuring-the-agent-directory) for details.

---

## 8. Hyperparameter Tuning

The `rosnav_rl.tuning` module provides Optuna-based hyperparameter optimisation
that integrates with the Pydantic config system.  Key components:

```
tuning/
├── __init__.py        # Public API exports
├── search_space.py    # FloatParam, IntParam, CategoricalParam models
├── sampler.py         # suggest_params() and apply_params() utilities
├── cfg.py             # TuningCfg — full study configuration model
└── callbacks.py       # TrialPruningCallback for SB3
```

### Design

Search spaces use **dot-notation config paths** so that any field in a
`TrainingCfg` can be tuned without code changes:

```yaml
search_space:
  agent_spec.framework.algorithm.parameters.learning_rate:
    type: float
    low: 1.0e-5
    high: 1.0e-3
    log: true
```

The flow is:
1. `suggest_params(trial, search_space)` → flat dict of sampled values
2. `apply_params(base_config_dict, params)` → deep-copied config with overrides
3. `TrainingCfg.model_validate(modified_dict)` → Pydantic validation
4. Normal training run → metric extraction → report to Optuna

The `TrialPruningCallback` plugs into SB3's callback system to report
intermediate metrics, enabling early stopping of unpromising trials.

See [Tutorial 9](TUTORIALS.md#9-hyperparameter-tuning) for the full walkthrough.

---

## Further Reading

| Document | What's inside |
| --- | --- |
| [TUTORIALS.md](TUTORIALS.md) | Step-by-step guides for all common tasks |
| [model/README.md](rosnav_rl/model/README.md) | RL_Model ABC, SB3 config hierarchy, ModelFactory, DreamerV3 |
| [observations/README.md](rosnav_rl/observations/README.md) | Collectors, Generators, YAML pipeline, DependencyResolver |
| [reward/README.md](rosnav_rl/reward/README.md) | RewardFunction, RewardUnit, safety categorization |
| [spaces/README.md](rosnav_rl/spaces/README.md) | SpaceFactory, encoding pipeline, ActionSpaceManager |
| [cfg/README.md](rosnav_rl/cfg/README.md) | AgentConfig, typed action spaces, discriminated unions, serialization |
| [action_server/README.md](rosnav_rl/action_server/README.md) | ROS 2 deployment, GetCommand service |
| Tuning module | `rosnav_rl/tuning/` — Optuna search spaces, samplers, pruning callback |
