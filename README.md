# RosNav-RL: Deep Reinforcement Learning for Robot Navigation

<p align="center">
  <a href="https://docs.arena-rosnav.org/">
    <img width="600" src="rosnav_rl/img/logo.png" alt="Rosnav-RL Logo">
  </a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://docs.ros.org/en/humble/index.html">
    <img src="https://img.shields.io/badge/ROS-Humble-blue" alt="ROS Humble">
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://docs.arena-rosnav.org/">
    <img src="https://img.shields.io/badge/documentation-brightgreen" alt="Documentation">
  </a>
</p>
<p align="center">
    <a href="https://stable-baselines3.readthedocs.io/en/master/">
    <img src="https://img.shields.io/badge/Stable--Baselines3-blue" alt="Stable-Baselines3">
    </a>
    <a href="https://github.com/danijar/dreamerv3">
    <img src="https://img.shields.io/badge/DreamerV3-blue.svg" alt="DreamerV3">
    </a>
</p>

<p align="center">
  <strong>A highly-modular, flexible, and unified interface for developing deep reinforcement learning agents for autonomous robot navigation in ROS.</strong>
</p>

---

## 📖 About The Project

**Rosnav-RL** is a developer-centric framework for creating and training state-of-the-art navigation agents with deep reinforcement learning in ROS 2. It's built to accelerate your research by providing a flexible, modular, and powerful toolkit that gets out of your way.

> **ROS 2 only.** All ROS 1 / `rospy` / `rospkg` dependencies have been removed. The package targets ROS 2 Humble exclusively.

Forget being locked into a single RL library. Rosnav-RL's core strength is its plug-and-play architecture. Swap backends like **Stable-Baselines3** and **DreamerV3** with ease, and focus on what matters: designing, training, and deploying better navigation agents, faster.

### ✨ Key Features

*   **9 Algorithms Out of the Box**: PPO, A2C, TRPO, RecurrentPPO, SAC, TD3, DDPG, TQC, CrossQ. Add a new one with a single config file.
*   **Framework-Agnostic**: Common `RL_Model` interface — swap SB3 ↔ DreamerV3 without touching training code.
*   **Composable Observation Spaces**: Mix laser, goal, velocity, and custom spaces via a registry. Parallel encoding, auto-normalization.
*   **Modular Reward System**: Stack reward units declaratively in YAML. Parallel evaluation, safety categorization.
*   **YAML-Driven Observation Pipeline**: Collectors → Generators with automatic dependency resolution via topological sort.
*   **Type-Safe Configuration**: Pydantic v2 with discriminated unions, auto-validation, full YAML round-trip.
*   **One-Command Deployment**: `ros2 run rosnav_rl action_server.py` wraps any trained agent in a `GetCommand` service.

---

## 📦 Packages

This repository contains the following packages:

| Package | Description |
| :--- | :--- |
| **`rosnav_rl`** | The core package containing the reinforcement learning framework, agent definitions, and training pipelines. |
| **`rosnav_rl_msgs`** | Contains the ROS message and service definitions used by `rosnav_rl` for communication. |

| Document | Description |
| --- | --- |
| **[README](rosnav_rl/README.md)** | Package overview, quick start, architecture at a glance |
| **[Developer Guide](rosnav_rl/GUIDE.md)** | Full architecture deep-dive, design patterns, code organization, core concepts |
| **[Tutorials](rosnav_rl/TUTORIALS.md)** | Step-by-step guides: training, deploying, adding algorithms, spaces, rewards |

---

## 🚀 Getting Started

### Prerequisites

*   ROS 2 Humble installation
*   Python 3.10+
*   [uv](https://docs.astral.sh/uv/) (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 🛠️ Installation

1.  **Clone the repository into your colcon workspace's `src` directory:**

    ```bash
    cd /path/to/your/colcon_ws/src
    git clone https://github.com/Arena-Rosnav/rosnav-rl.git
    ```

2.  **Install dependencies using uv:**

    Navigate to the `rosnav_rl` package directory and run `uv sync`. uv will create a
    `.venv` virtual environment and install all required packages automatically.
    ```bash
    cd rosnav-rl/rosnav_rl
    uv sync
    # activate the venv (or prefix commands with `uv run`)
    source .venv/bin/activate
    ```

3.  **Build your workspace:**

    Navigate back to the root of your workspace and build the packages.
    ```bash
    cd /path/to/your/colcon_ws
    colcon build --packages-select rosnav_rl rosnav_rl_msgs
    ```

4.  **Source the workspace:**
    
    ```bash
    source install/setup.bash
    ```

---

## 🧪 Testing

A pytest suite covering path resolution, model loading, and the `get_command` service is included:

```bash
cd rosnav_rl   # package root
python3 -m pytest tests/ -v
```

To create a minimal test agent (random weights, no training required):

```bash
python3 scripts/create_test_agent.py --agent-name test_agent
```

This writes `training_config.yaml` and `best_model.zip` to `Arena/arena_training/agents/test_agent/`.

---

## 🖥️ Usage

**Standalone — start the action server with a trained agent:**

```bash
ros2 run rosnav_rl action_server.py --ros-args -p agent_name:=<your_agent>
```

Or via the provided launch file:

```bash
ros2 launch rosnav_rl action_server.launch.py agent_name:=<your_agent>
```

The server exposes a `get_command` service (`rosnav_rl_msgs/srv/GetCommand`) under the robot's namespace. It reads sensor data from its configured ROS 2 topics and returns a `geometry_msgs/Twist`. On inference errors it logs a warning and returns zero velocity instead of crashing.

```bash
# Call the service manually
ros2 service call /get_command rosnav_rl_msgs/srv/GetCommand {}
```

**Arena integration** — when using [Arena-Rosnav](https://github.com/Arena-Rosnav/arena-rosnav), the action server is started automatically:

```bash
arena launch local_planner:=rosnav_rl agent_name:=<your_agent>
```

Agent folders live in `Arena/arena_training/agents/<agent_name>/` and must contain `training_config.yaml` + `best_model.zip`.


---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
