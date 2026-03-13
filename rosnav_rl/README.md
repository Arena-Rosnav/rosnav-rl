# RosNav-RL

<p align="center">
  <img width="600" src="img/logo.png" alt="Rosnav-RL Logo"/>
</p>

> **The research toolkit for RL-based ROS 2 robot navigation.**
>
> RosNav-RL wraps any reinforcement learning framework - Stable-Baselines3,
> DreamerV3, or your own - with a unified, fully modular pipeline:
> sensor data collection and preprocessing, typed observation spaces,
> composable reward shaping, automatic dependency resolution between pipeline
> units, and Optuna-based hyperparameter search.
> Every layer is independently swappable so you can isolate and iterate on
> exactly the component you care about without rewriting the rest.

**The core idea:** define your robot's *sensors*, *observations*, *reward*, and
*algorithm* as composable YAML configs. Switch from PPO to DreamerV3, add a new
reward term, or plug in a custom observation generator - all without changing
a single line of training code.


| | |
| --- | --- |
| **ROS** | Humble (ROS 2 only) |
| **Python** | 3.10+ |
| **RL Backends** | Stable-Baselines3, sb3-contrib, DreamerV3 |
| **Config** | Pydantic v2 - type-safe, auto-validated, YAML round-trip |

---

## Key Features

- **Wrap any RL backend** - the common `RL_Model` interface means you swap SB3 ↔ DreamerV3 ↔ your own implementation without touching training code. Same observations, same reward, same config.
- **Declarative data pipeline** - define collectors by ROS message type (`sensor_msgs/LaserScan`); preprocessing and topic wiring happen automatically.
- **Dependency-resolved observation graph** - generators declare what they need; a topological sort determines execution order at startup so you never manage it manually.
- **Composable reward shaping** - stack reward units in YAML, evaluated in parallel with safety categorization and schema-validated inter-unit dependencies.
- **Built-in hyperparameter tuning** - Optuna integration with MedianPruner / HyperbandPruner, framework-specific pruning callbacks, and automatic best-param export.
- **One-command deployment** - `ros2 run rosnav_rl action_server.py` wraps any trained agent behind a `GetCommand` service.

---

## Quick Start

> **Arena users:** run `arena feature training install` - it handles everything below. Come back here for the API reference.

```bash
# 1. Clone into your colcon workspace
cd ~/colcon_ws/src
git clone --depth 1 https://github.com/Arena-Rosnav/rosnav-rl.git

# 2. Install Python dependencies
cd rosnav-rl/rosnav_rl && uv sync

# 3. Build ROS 2 packages
cd ~/colcon_ws
colcon build --packages-select rosnav_rl rosnav_rl_msgs
source install/setup.bash
```

### Deploy a pre-trained agent

The action server needs two things: a trained agent and an **observations config** that maps your robot's sensor topics to collector types.

```bash
# Create a test agent with random weights (no training run needed)
python3 scripts/create_test_agent.py --agent-name test_agent

# Start the action server
ros2 run rosnav_rl action_server.py --ros-args \
  -p agent_name:=test_agent \
  -p observations_config:=/path/to/observations.yaml
```

The observations config tells the server which ROS topics to subscribe to. A minimal example:

```yaml
# observations.yaml
datasources:
  front_laser:
    type: sensor_msgs/LaserScan    # ROS message type → resolves to LaserScanCollector
    params:
      topic: "scan"              # your lidar topic
      up_to_date_required: true
  goal_pose:
    type: geometry_msgs/PoseStamped
    params:
      topic: "goal_pose"
      up_to_date_required: false
  robot_pose_from_tf:
    type: RobotPoseTFGenerator     # Generators always use class names
    params: {}
```

A full annotated config with all available collectors and generators is at [`observations/observations.yaml`](rosnav_rl/observations/observations.yaml).

### Train an agent (minimal Python example)

```python
import rosnav_rl
from rosnav_rl.model.stable_baselines3.cfg import (
    StableBaselinesCfg, PPO_Cfg, PPO_Algorithm_Cfg,
)

agent_cfg = rosnav_rl.AgentCfg(
    robot="jackal",
    framework=StableBaselinesCfg(
        algorithm=PPO_Cfg(
            architecture_name="AGENT_1",
            parameters=PPO_Algorithm_Cfg(
                total_timesteps=5_000_000,
                learning_rate=3e-4,
            ),
        ),
    ),
    reward=rosnav_rl.RewardCfg(
        reward_function_dict={
            "goal_reached":  {"reward": 15.0},
            "collision":     {"reward": -10.0},
            "approach_goal": {"pos_factor": 0.3, "neg_factor": 0.5},
            "safe_distance": {"reward": -0.15},
        },
    ),
)

sim_state = rosnav_rl.SimulationStateContainer(...)
agent = rosnav_rl.RL_Agent(
    agent_cfg=agent_cfg,
    agent_state_container=sim_state.to_agent_state_container(),
)
agent.initialize_model()
agent.train(train_envs=train_envs, eval_envs=eval_envs)
```

For full training with a simulator see [`arena_training`](https://github.com/Arena-Rosnav/Arena-Training) and its [config reference](https://github.com/Arena-Rosnav/Arena-Training/tree/main/configs).

---

## Architecture at a Glance

```
Sensors ──▶ ObservationManager ──▶ ObservationSpaceManager ──▶ RL_Model ──▶ ActionSpaceManager ──▶ Robot
              (collect & generate)    (encode & normalize)     (policy)    (decode to Twist)

                                                                  ▲
                                                                  │
                                                           RewardFunction
                                                         (parallel, composable)
```

| Module | Role |
| --- | --- |
| **Observations** | ROS 2 topic → typed data via Collectors & Generators |
| **Spaces** | Encode observations, decode actions - registry-based, parallel |
| **Model** | Algorithm-agnostic training & inference (`RL_Model` ABC) |
| **Reward** | Composable reward units with safety categorization |
| **Config** | Pydantic v2 discriminated unions - YAML in, validated config out |
| **Action Server** | `GetCommand` ROS 2 service for real-time deployment |
| **States** | Typed dataclass containers for simulation & agent state |

---

## Documentation

| Document | Description |
| --- | --- |
| **[Developer Guide](GUIDE.md)** | Full architecture deep-dive, design patterns, data flow, code organization, and core concepts |
| **[Tutorials](TUTORIALS.md)** | Step-by-step guides: training agents, adding algorithms, creating observation spaces, building reward units, deploying |

### Sub-package docs

| Package | README |
| --- | --- |
| Model | [model/README.md](rosnav_rl/model/README.md) - RL_Model ABC, SB3 config hierarchy, ModelFactory |
| Observations | [observations/README.md](rosnav_rl/observations/README.md) - Collectors, Generators, YAML pipeline, DependencyResolver |
| Reward | [reward/README.md](rosnav_rl/reward/README.md) - RewardFunction, RewardUnit, safety categorization |
| Spaces | [spaces/README.md](rosnav_rl/spaces/README.md) - SpaceFactory, encoding pipeline, ActionSpaceManager |
| Config | [cfg/README.md](rosnav_rl/cfg/README.md) - AgentCfg, discriminated unions, serialization |
| Action Server | [action_server/README.md](rosnav_rl/action_server/README.md) - ROS 2 deployment, GetCommand service |

---

## Testing

```bash
cd rosnav_rl
python3 -m pytest tests/ -v
```

---

## Project Structure

```
rosnav_rl/
├── rl_agent.py           # RL_Agent - top-level orchestrator
├── cfg/                  # Pydantic v2 configuration (AgentCfg, RewardCfg, …)
├── model/                # RL_Model ABC + SB3 & DreamerV3 implementations
├── observations/         # Collector/Generator pipeline, YAML-driven
├── reward/               # Composable reward units + RewardFunction
├── spaces/               # Observation & action space management
├── states/               # SimulationStateContainer, AgentStateContainer
├── action_server/        # ROS 2 GetCommand service server
├── utils/                # RequiresProtocol, ErrorReportingMixin, validation
└── scripts/              # create_test_agent.py, test.py
```

---

## License

MIT - see [LICENSE.md](LICENSE.md).
