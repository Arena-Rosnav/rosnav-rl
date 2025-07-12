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
  <a href="https://www.python.org/downloads/release/python-380/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
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

Forget being locked into a single RL library. Rosnav-RL's core strength is its plug-and-play architecture. Swap backends like **Stable-Baselines3** and **DreamerV3** with ease, and focus on what matters: designing, training, and deploying better navigation agents, faster.

### ✨ Key Features

*   **Experiment with Multiple RL Frameworks**: Don't get locked in. Our framework-agnostic design lets you leverage the best of different libraries like Stable-Baselines3 and DreamerV3 in the same project.
*   **Build Custom Agents in Minutes**: A deeply modular architecture with plug-and-play components for rewards, observations, and network layers means you can prototype new agent designs rapidly.
*   **Standardize Your Experiments**: Tame the complexity of multiple robot setups and tasks with a unified system for managing observation and action spaces.
*   **Configure with Confidence**: Pydantic-based configuration provides type-safety, auto-validation, and self-documenting schemas, eliminating frustrating runtime errors.
*   **Deploy Seamlessly in ROS 2**: Move from training to deployment effortlessly. A built-in ROS action server ensures your agent integrates perfectly into your robotics ecosystem.

---

## 📦 Packages

This repository contains the following packages:

| Package | Description |
| :--- | :--- |
| **`rosnav_rl`** | The core package containing the reinforcement learning framework, agent definitions, and training pipelines. |
| **`rosnav_rl_msgs`** | Contains the ROS message and service definitions used by `rosnav_rl` for communication. |

For a deep dive into the architecture and development workflow, please refer to the **[RosNav-RL Developer Guide](rosnav_rl/README.md)**.

---

## 🚀 Getting Started

### Prerequisites

*   ROS 2 Humble installation
*   Python 3.8+
*   Poetry

### 🛠️ Installation

1.  **Clone the repository into your colcon workspace's `src` directory:**

    ```bash
    cd /path/to/your/colcon_ws/src
    git clone https://github.com/Arena-Rosnav/rosnav-rl.git
    ```

2.  **Install dependencies using Poetry:**

    Navigate to the `rosnav_rl` package directory and run `poetry install`. This will install the necessary Python packages in a virtual environment.
    ```bash
    cd rosnav-rl/rosnav_rl
    poetry install
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

## 🖥️ Usage

To run the trained agent, you can use the provided launch file to start the ROS action server. This will make your agent available for navigation tasks within the ROS 2 ecosystem.

```bash
ros2 launch rosnav_rl rl_agent.launch.py
```

You can then send requests to the `/rosnav_rl/get_action` service to get actions from your agent.

---

## 🏗️ Project Architecture

The system is built around a modular RL architecture that separates concerns between the Reinforcement Learning Framework, Agent-specific Space Management, Reward Calculation, and Observation Handling. This design promotes flexibility and allows for easy modification of individual components without affecting the rest of the system.

<p align="center">
  <img width="70%" src="rosnav_rl/img/rosnav_rl.png" />
</p>

### Data Flow

The data flow is designed to be a sequential pipeline, from sensor data collection to robot command execution.

1.  **Input**: Raw data is collected from the environment from various sources.
2.  **Processing**: The `Observation Manager` handles the data, and the `Space Manager` transforms it into the agent's specific observation space.
3.  **Decision**: The agent's `Model` (including feature extractors and the policy network) processes the features and selects an action.
4.  **Output**: The `Action Space Manager` prepares the command for the robot.
5.  **Execution**: The command is sent to the robot for physical or simulated execution.

<p align="center">
  <img width="60%" src="rosnav_rl/img/dataflow.png" />
</p>

---

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
