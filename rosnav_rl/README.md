# RosNav-RL

<p align="center">
  <img width="600" src="img/logo.png" alt="Rosnav-RL Logo"/>
</p>

> **Modular deep reinforcement learning for ROS 2 robot navigation.**
>
> Train, evaluate, and deploy RL navigation agents with swappable algorithms,
> composable observation spaces, and plug-and-play reward functions.

| | |
| --- | --- |
| **ROS** | Humble (ROS 2 only - no rospy/rospkg) |
| **Python** | 3.10+ |
| **RL Backends** | Stable-Baselines3, sb3-contrib, DreamerV3 |
| **Config** | Pydantic v2 (type-safe, auto-validated, YAML round-trip) |

---

## Key Features

- **Framework-agnostic** - common `RL_Model` interface lets you swap SB3 ↔ DreamerV3 without touching training code.
- **Composable observation spaces** - mix laser, goal, velocity, pedestrian, and custom spaces via a registry. Parallel encoding, auto-normalization.
- **Modular reward system** - stack reward units declaratively in YAML. Parallel evaluation, safety categorization, schema-based dependency validation.
- **Observation pipeline** - collectors → generators → agents. Dependency resolution via topological sort. Observation synchronization.
- **One-command deployment** - `ros2 run rosnav_rl action_server.py` wraps any trained agent in a `GetCommand` service.

---

## Quick Start

> **Arena users:** run `arena feature training install` — it handles everything below. Come back here for the API reference.

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

```bash
# Create a test agent with random weights (no training run needed)
python3 scripts/create_test_agent.py --agent-name test_agent

# Start the action server — exposes a GetCommand ROS 2 service
ros2 run rosnav_rl action_server.py --ros-args -p agent_name:=test_agent
```

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
