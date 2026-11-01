# RosNav-RL

<p align="center">
  <a href="https://docs.arena-rosnav.org/">
    <img width="600" src="rosnav_rl/img/logo.png" alt="Rosnav-RL Logo">
  </a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://docs.ros.org/en/humble/index.html"><img src="https://img.shields.io/badge/ROS-Humble-blue" alt="ROS Humble"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://stable-baselines3.readthedocs.io/en/master/"><img src="https://img.shields.io/badge/Stable--Baselines3-blue" alt="Stable-Baselines3"></a>
  <a href="https://github.com/NM512/dreamerv3-torch"><img src="https://img.shields.io/badge/DreamerV3-blue.svg" alt="DreamerV3"></a>
</p>

<p align="center">
  <strong>The research toolkit for RL-based robot navigation on ROS 2.</strong>
</p>

---

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

## 🚀 Quick Start

> **Arena-Rosnav users:** just run `arena feature training install` - skip to [Usage](#usage).

**Requirements:** ROS 2 Humble · Python 3.10+ · [uv](https://docs.astral.sh/uv/)

```bash
# 1. Clone
cd ~/colcon_ws/src
git clone --depth 1 https://github.com/Arena-Rosnav/rosnav-rl.git

# 2. Install Python deps (uv creates and manages the venv)
cd rosnav-rl/rosnav_rl
uv sync && source .venv/bin/activate

# 3. Build & source
cd ~/colcon_ws
colcon build --packages-select rosnav_rl rosnav_rl_msgs
source install/setup.bash
```

That's it. You're ready to deploy or train.

---

## 🖥️ Usage

### Deploy a trained agent

The action server needs two things: a trained agent and an **observations config** that tells it which ROS topics to subscribe to and how to interpret them.

```bash
# Don't have an agent yet? Create one with random weights in seconds:
python3 scripts/create_test_agent.py --agent-name test_agent
```

The observations config (`observations.yaml`) maps your robot's sensors to collector types. A minimal example for a laser + goal setup:

```yaml
# observations.yaml
datasources:
  front_laser:
    type: sensor_msgs/LaserScan
    params:
      topic: "scan"              # your lidar topic
      up_to_date_required: true
  goal_pose:
    type: geometry_msgs/PoseStamped
    params:
      topic: "goal_pose"
      up_to_date_required: false
  robot_pose_from_tf:
    type: RobotPoseTFGenerator
    params: {}
```

A full annotated example with all available collectors and generators lives at [`rosnav_rl/observations/observations.yaml`](rosnav_rl/rosnav_rl/observations/observations.yaml). Copy it, strip what you don't need, and point it at your topics.

```bash
# Start the action server
ros2 run rosnav_rl action_server.py --ros-args \
  -p agent_name:=test_agent \
  -p observations_config:=/path/to/observations.yaml

# Or via launch file
ros2 launch rosnav_rl action_server.launch.py \
  agent_name:=test_agent \
  observations_config:=/path/to/observations.yaml
```

The server reads sensor data from ROS 2 topics and returns a `geometry_msgs/Twist` via the `rosnav_rl_msgs/srv/GetCommand` service. On inference errors it logs a warning and returns zero velocity - it won't crash your robot.

```bash
# Poke it manually
ros2 service call /get_command rosnav_rl_msgs/srv/GetCommand {}
```

### Train an agent

The cleanest way to train is via [arena_training](https://github.com/Arena-Rosnav/Arena-Training), which wires up the gym environments and launch files. For a standalone training loop, use the Python API directly:

```python
import rosnav_rl
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import (
    StableBaselinesCfg, PPO_Cfg, PPO_Algorithm_Cfg,
)

spec = rosnav_rl.AgentConfig(
    robot="jackal",
    action_space=DifferentialDriveActionSpace(
        linear_range=(-2.0, 2.0),
        angular_range=(-4.0, 4.0),
    ),
    # Review and adjust these before every training run - especially
    # laser_num_beams / laser_max_range (must match your robot's LIDAR),
    # robot_radius / safety_distance, and goal_radius / max_steps.
    # When using arena_training these are auto-populated from the robot
    # description, but you should still verify them
    parameters=AgentParameters(
        laser_num_beams=720,
        laser_max_range=30.0,
        robot_radius=0.215,
        safety_distance=0.3,
        goal_radius=0.35,
        max_steps=500,
    ),
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

agent = rosnav_rl.RL_Agent(spec)
agent.initialize_model()
agent.train(train_envs=train_envs, eval_envs=eval_envs)
```

> **Before every training run:** open the generated `agent.yaml` and verify the
> `parameters:` block.  Key fields: `laser_num_beams`, `laser_max_range`,
> `robot_radius`, `safety_distance`, `goal_radius`, `max_steps`, and all
> velocity bounds.  A mismatch between these and your actual robot will silently
> degrade policy quality.  See the [AgentParameters reference](rosnav_rl/README.md#agentparameters--tuning-before-training)
> for a full field table.

To swap to DreamerV3, replace `StableBaselinesCfg(...)` with `DreamerV3Cfg(...)` - everything else stays the same. With Arena:

```bash
arena train sim:=gazebo train_config:=sb_training_config.yaml
```

Set the parallel-env count via `arena_cfg.general.n_envs` in the training
YAML.

### Swap the RL algorithm

It's a one-line YAML change - no Python rewrite required. See the [config reference](rosnav_rl/cfg/README.md) and [tutorials](rosnav_rl/TUTORIALS.md) for all options.

---

## 🧪 Testing

The test suite covers config loading, path resolution, model init, and the `GetCommand` service:

```bash
cd rosnav_rl
python3 -m pytest tests/ -v
```

No simulator needed - tests are fully offline. If you want a real smoke-test of the full inference path:

```bash
# Creates agents/test_agent/ with random-weight best_model.zip
python3 scripts/create_test_agent.py --agent-name test_agent
# Then spin up the action server and call it
ros2 run rosnav_rl action_server.py --ros-args -p agent_name:=test_agent
```

---

## 📦 Repository layout

| Path | Description |
|---|---|
| `rosnav_rl/` | Core Python package - models, observations, rewards, spaces, deployment |
| `rosnav_rl_msgs/` | ROS 2 message & service definitions (`GetCommand`, etc.) |

**Go deeper:**

| Document | What's in it |
|---|---|
| [rosnav_rl/README.md](rosnav_rl/README.md) | Architecture overview, data-flow diagram, module table |
| [rosnav_rl/GUIDE.md](rosnav_rl/GUIDE.md) | Full developer guide - design patterns, core concepts, code organization |
| [rosnav_rl/TUTORIALS.md](rosnav_rl/TUTORIALS.md) | Step-by-step: add an algorithm, build a reward unit, create an observation space |

---

## 📜 License

MIT - see [LICENSE.md](LICENSE.md).
