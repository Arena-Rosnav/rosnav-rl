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
  <a href="http://wiki.ros.org/noetic">
    <img src="https://img.shields.io/badge/ROS-Noetic-blue" alt="ROS Noetic">
  </a>
  <a href="https://www.python.org/downloads/release/python-380/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://stable-baselines3.readthedocs.io/en/master/">
    <img src="https://img.shields.io/badge/Stable--Baselines3-blue" alt="Stable-Baselines3">
  </a>
  <a href="https://docs.arena-rosnav.org/">
    <img src="https://img.shields.io/badge/documentation-brightgreen" alt="Documentation">
  </a>
</p>

<p align="center">
  <strong>A highly-modular, flexible, and unified interface for developing deep reinforcement learning agents for autonomous robot navigation in ROS.</strong>
</p>

---

## 📖 About The Project

This repository contains the **Rosnav-RL** packages, a framework for constructing and training deep reinforcement learning agents for autonomous robot navigation. It provides a comprehensive set of tools for developing training pipelines and testing on various platforms.

The framework is designed to be modular and flexible, with a unified interface for defining agents, reward functions, and action/observation spaces. It supports extensive configuration to encourage experimentation and is intended to facilitate the use of multiple reinforcement learning libraries.

Originally developed for the [Arena-Rosnav](https://github.com/Arena-Rosnav/arena-rosnav) simulation environment, Rosnav-RL can be easily integrated into other simulation environments.

### ✨ Key Features

*   **Flexible Infrastructure**: A framework-agnostic design that supports multiple reinforcement learning backends for model development.
*   **Modular Design**: Clean separation between network architecture building blocks, allowing for straightforward customization and extension.
*   **Unified Encoding**: Standardized management of observation and action spaces across different navigation tasks and robot configurations.
*   **Robust Configuration**: Pydantic-based configuration management providing automatic validation, schema documentation, and serialization support.
*   **Extensible Components**: Easily add new observation types, neural network architectures, and reward functions for rapid agent development.
*   **Deployment-Ready**: Includes a ROS action server for seamless agent deployment and integration into larger systems.

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

*   ROS Noetic installation
*   Python 3.8+
*   Poetry

### 🛠️ Installation

You can add `rosnav-rl` to your catkin workspace.

1.  **Clone the repository into your `src` folder:**

    ```bash
    git clone <repository-url>
    ```

2.  **Install dependencies using Poetry:**

    Navigate to the `rosnav_rl` package directory and run:
    ```bash
    cd rosnav_rl
    poetry install
    ```

3.  **Build your workspace:**

    ```bash
    catkin_make
    # or
    catkin build
    ```

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
