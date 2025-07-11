# RosNav-RL Developer Guide

<p align="center">
  <img width="600" src="img/logo.png" alt="Rosnav-RL Logo"/>
</p>

## 📚 Table of Contents
- [Executive Summary](#1-executive-summary)
- [Project Architecture](#2-project-architecture)
- [Setup & Installation](#3-setup--installation)
- [Code Organization](#4-code-organization)
- [Core Concepts and Abstractions](#5-core-concepts-and-abstractions)
- [Development Workflow](#6-development-workflow)
- [Common Tasks & Tutorials](#7-common-tasks--tutorials)

---

## 🎯 1. Executive Summary

### Project Purpose
This is the official **Rosnav-RL** package. The Rosnav-RL framework provides tools to construct a
deep reinforcement learning agent for autonomous robot navigation, straight-forward development of a training pipeline and further testing across different platforms. 
It provides a highly-modular, flexible and unified interface for defining an agent with a neural network, reward function, action and observation space. 
The configuration layout provides a wide range of settings to encourage experimentation. The framework is
intended to facilitate multiple reinforcement learning libraries for the application on learning-based navigation systems for mobile robots in ROS1.

Rosnav-RL was initially developed for the [Arena-Rosnav](https://github.com/Arena-Rosnav/arena-rosnav) environment, which is a simulation platform for training and evaluating navigation systems. 
Though, the framework can be easily integrated into other simulation environments. 

<p align="center">
  <img width="50%" src="img/training_pipeline.png" />
</p>

### Main Technologies
- **ROS 1**: Robot Operating System
- **Python 3**: Core programming language
- **Stable-Baselines3**: RL framework
- **DreamerV3**: Model-based RL framework
- **PyTorch**: Deep Learning library

### ✨ Key Features
- **Flexible Infrastructure**: Framework-agnostic design supporting multiple reinforcement learning backends.
- **Modular Design**: Clean separation between network architecture building blocks for easy customization.
- **Unified Encoding**: Standardized observation and action space management.
- **Robust Configuration**: Pydantic-based configuration management with automatic validation.
- **Seamless Observation Handling**: Modular management for easy integration of new observation types.
- **Extensible Components**: Pre-tested neural network architectures and reward functions for quick development.
- **Deployment-Ready**: ROS action server for seamless agent deployment.

---

## 🏗️ 2. Project Architecture

### High-Level Overview
The system is built around a modular reinforcement learning architecture that separates concerns between:
- Reinforcement Learning Framework
- Agent-specific Space Management
- Reward Calculation
- Observation Handling
- Centralized Parameter Management

### Key Components
<p align="center">
  <img width="70%" src="img/rosnav_rl.png" />
</p>

```
RosNav-RL
├── RL Agent
│   ├── Model Architecture
│   ├── Space Manager (Action-/ Observation Space Manager)
│   └── Reward Function
├── Observations
│   ├── Observation Manager
│   └── Observation Units
└── States
    ├── Agent State Container
    └── Simulation State Container
```

### Data Flow
<p align="center">
  <img width="60%" src="img/dataflow.png" />
</p>

| Stage | Component | Function |
|-------|-----------|----------|
| 1. Input | Observations Dictionary | Raw data collection |
| 2. Processing | Observation Manager | Data handling |
| | (Observation) Space Manager | Space-specific transformation |
| 3. Decision | Model (Feature Extractors) | Feature engineering |
| | Model (Policy Network) | Action selection |
| 4. Output | (Action) Space Manager | Command preparation |
| 5. Execution | Robot Commands | Physical/ simulated execution |

---

## 🚀 3. Setup & Installation

### Prerequisites
- Working ROS Noetic installation
- Python 3.8+
- Poetry

### Environment Setup
**Single-package installation:**
```bash
# Clone repository
git clone https://github.com/your-org/rosnav-rl.git
cd rosnav-rl

# Install dependencies
poetry install
```
**Add to existing workspace:**
```bash
# Clone repository into your workspace's src folder
git clone https://github.com/your-org/rosnav-rl.git src/rosnav-rl

# Navigate to workspace
cd *path-to-workspace*
poetry add src/rosnav-rl
```

---

## 📂 4. Code Organization

### Directory Structure
```
agents/             # Agent storage
launch/             # ROS launch files
reward/             # Reward functions
rosnav_rl/
├── action_server/    # ROS action server implementation
├── cfg/             # Configuration management
├── model/           # RL model implementations
├── observations/    # Observation handling
├── reward/         # Reward system
├── spaces/         # Action and observation spaces
├── states/         # State management
└── utils/          # Utility functions
```

---

## 🤖 5. Core Concepts and Abstractions

#### Main Agent Interface
- **File**: [`rosnav_rl/rl_agent.py`](rosnav_rl/rl_agent.py)
- **Role**: Coordinates interactions between the Model, Space Manager, and Reward Function. Provides the main interface for Reinforcement Learning.

```python
class RL_Agent:
    def __init__(self, agent_cfg: AgentCfg, agent_state_container: AgentStateContainer):
        self._model = self.initialize_model()
        self._space_manager = self._initialize_space_manager()
        self._reward_function = self._initialize_reward_function()

    def train(self, *args, **kwargs):
        # Calls self._model.train() and trains the agent

    def get_action(self, observation: ObservationDict) -> np.ndarray:
        # Get action from model
```

#### Model Interface
- Implemented in [`rosnav_rl/model/model.py`](rosnav_rl/model/model.py)
- Encapsulates neural network and its framework-specific logic
- Responsible for training, saving and loading the model

```python

class RL_Model:
    @abstractmethod
    def setup_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_action(self, observation: "EncodedObservationDict", *args, **kwargs):
        pass
```

#### Agent Spaces and Observation Management
- Space Manager in [`rosnav_rl/spaces/space_manager/base_space_manager.py`](rosnav_rl/spaces/space_manager/base_space_manager.py)
- Action Space Manager in [`rosnav_rl/spaces/action_space/action_space_manager.py`](rosnav_rl/spaces/action_space/action_space_manager.py)
- Observation Space Manager in [`rosnav_rl/spaces/observation_space/observation_space_manager.py`](rosnav_rl/spaces/observation_space/observation_space_manager.py)
- Observation Manager in [`rosnav_rl/observations/observation_manager.py`](rosnav_rl/observations/observation_manager.py)


**Base Space Manager**
- Processes agent observations and actions
- Converts between different representations

```python
class BaseSpaceManager:
    def encode_observation(
        self, obs_dict: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        pass

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        pass
```

**Observation Space Manager**
- An agent observation space is made up of multiple observation space units - each responsible for a specific observation type and its processing

```python

class ObservationSpaceManager:
    _space_cls_list: ObservationSpaceList | Union[Type[BaseObservationSpace]]
    _observation_space: spaces.Dict

    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        # Retrieve observation and encode based on observation spaces
```

**Observation Spaces**
- Modular observation space component that can be combined to form an observation space
```python
class BaseObservationSpace:
    name: ClassVar[str]
    required_observation_units: ClassVar[
        List[Union[ObservationCollector, ObservationGenerator]]
    ] = []

    @property
    def space(self) -> spaces.Space:
        """
        Get the gym.Space object representing the observation space.
        """
        return self._space

    @abstractmethod
    def encode_observation(self, observation: ObservationDict) -> np.ndarray:
        pass
```

##### Observation Management

- Observation Manager in [`rosnav_rl/observations/observation_manager.py`](rosnav_rl/observations/observation_manager.py)
- Manages observation collection and generation

```python
class ObservationManager:
    _collectors: Dict[str, ObservationCollectorUnit]
    _generators: Dict[str, ObservationGeneratorUnit]

    def get_observations(
        self,
        *args,
        **extra_observations: ObservationDict,
    ) -> ObservationDict:
    # Collect and generate observations
```

**ObservationCollectorUnits**
- Implemented in [`rosnav_rl/observations/observation_collector_unit.py`](rosnav_rl/observations/collectors/base_collector.py)
- Collects observations from ROS topics and preprocesses them
- Modular structure through Observation Units
- Name, topic, and message type are defined in the class definition

```python
class ObservationCollectorUnit(
    BaseUnit, Generic[MessageType, ProcessedObservationType], ABC
):
    name: ClassVar[str]
    topic: ClassVar[str]
    msg_data_class: ClassVar[Type[MessageType]]
    data_class: ClassVar[Type[ProcessedObservationType]] = ProcessedObservationType

    def preprocess(self, msg: MessageType) -> ProcessedObservationType:
        # Preprocess message and return observation
```

**ObservationGeneratorUnits**
- Implemented in [`rosnav_rl/observations/generators/base_generator.py`](rosnav_rl/observations/generators/base_generator.py)
- Generates observations from collected data
- Name, required observation units, data type of the generated data and observation generation logic are defined in the class definition

```python
class ObservationGeneratorUnit(BaseUnit, Generic[GeneratedDataType], ABC):
    name: ClassVar[str]
    requires: ClassVar[List[BaseUnit]]
    data_class: Type[GeneratedDataType]

    def generate(
        self,
        obs_dict: dict,
        simulation_state_container: SimulationStateContainer,
        *args,
        **kwargs,
    ) -> GeneratedDataType:
        # Generate observation based on collected data
```

#### Reward System
- Reward Function in [`rosnav_rl/reward/reward_function.py`](rosnav_rl/reward/reward_function.py)
- Calculates rewards based on actions and states
- Modular structure through Reward Units

```python
class RewardFunction:
    self._reward_units: List[RewardUnit]
    
    def calculate_reward(self, obs_dict: ObservationDict, simulation_state_container: SimulationStateContainer):
        # Calculate reward based on current observation and state
```

**Reward Units**
- Modular reward components that be combined to form a reward function
```python
class RewardUnit(ABC):
    def add_reward(self, value: float):
    ...

    @abstractmethod
    def __call__(self, obs_dict: ObservationDict, state_container: SimulationStateContainer, *args, **kwargs):
        # Logic for calculating reward then calls add_reward()
```

#### State Management
- Agent States in [`rosnav_rl/states/agent/container.py`](rosnav_rl/states/agent/container.py)
- Simulation States in [`rosnav_rl/states/simulation/container.py`](rosnav_rl/states/simulation/container.py)
- Manages state of agent and simulation environment

#### Deployment
- Action Server in [`rosnav_rl/action_server/base_server.py`](rosnav_rl/action_server/base_server.py)
- Enables deployment of trained agents
- Provides ROS interface

## 6. Development Workflow

### Deployment
When using the Rosnav-RL Action Server, the agent seamlessly integrates into the ROS infrastructure. The action server handles communication between the agent and the ROS environment. Observations are collected and passed to the agent, which returns an action to be executed by the robot. 

To use it, ensure the `ObservationUnit`s have the correct topics and message types. The action can be requested via an external service call to `/rosnav_rl/get_action`. A launch file is provided to start the action server.

```python
import rospy
from rosnav_rl.srv import GetAction, GetActionRequest

rospy.wait_for_service(f"{self.ns}/rosnav_rl/get_action")
self._get_action_srv = rospy.ServiceProxy(
    f"{self.ns}/rosnav_rl/get_action", GetAction
)

action = self._get_action_srv(GetActionRequest()).action
```

### Training
1.  **Choose a simulator environment.**
2.  **Derive an Environment class** from `gym.Env` and implement the necessary methods using the provided interfaces:
    -   `BaseSpaceManager`: Preprocess observations for the model and decode actions for the environment.
    -   `ObservationManager`: Subscribe to topics and collect/generate observations.
    -   `RewardFunction`: Calculate rewards based on observations and state.
3.  **Instantiate the `SimulationStateContainer`** with the necessary parameters. The `AgentStateContainer` is derived from it via `.to_agent_state_container()` and holds space-specific parameters.
4.  **Initialize and train the agent** with your desired configuration and the new environment.

> **Note**: Ensure that the `ObservationUnits` of the `observations` module have the right topics and message types to collect the necessary data.

---

## 🎓 7. Common Tasks & Tutorials

Here are guides for common development tasks. For more details, check the documentation within each submodule.
- [Reward Functions](rosnav_rl/reward/reward_functions.md)
- [Observation Spaces](rosnav_rl/spaces/spaces.md)

> #### 🧩 Adding a New RL Framework
> 1.  Implement a new `RL_Model` inheriting from the base class. Implement methods for training, saving, loading, and action selection.
> 2.  Implement your new architectures. Define the required observation spaces for the model.
> 3.  Define new Pydantic configuration files for all settings of the new framework.
> 4.  For deployment, inherit a new action server for the new framework.

> #### 🏋️ Training a New Agent
> For a working integration into a training pipeline, check out the [Arena](https://github.com/Arena-Rosnav/arena-rosnav) repository and its [documentation](https://arena-rosnav.readthedocs.io/en/latest/).
> ```python
> import rosnav_rl
> 
> # Create configuration
> config = rosnav_rl.model.dreamerv3.cfg.DreamerV3Cfg(...) # or
> config = rosnav_rl.model.stable_baselines3.cfg.StableBaselinesCfg(...)
> 
> agent_cfg = rosnav_rl.AgentCfg(
>     name="my_agent",
>     robot="jackal",
>     framework=config,
> )
> 
> # Create state containers
> simulation_state_container = rosnav_rl.SimulationStateContainer(...)
> agent_state_container = simulation_state_container.to_agent_state_container()
> 
> # Create agent
> agent = rosnav_rl.RL_Agent(
>     agent_cfg=agent_cfg,
>     agent_state_container=agent_state_container
> )
> agent.initialize_model()
> 
> # Create environments
> train_envs = [create_env("train", i) for i in range(config.general.envs)]
> eval_envs = [create_env("eval", i) for i in range(config.general.envs)]
> 
> # Train agent
> agent.train(
>     train_envs=train_envs,
>     eval_envs=eval_envs
> )
> ```

> #### 🏗️ Adding New Model Architectures
> Implementation depends on the preferred framework:
> - [StableBaselines3 Guide](rosnav_rl/model/stable_baselines3/custommodel.md)
> - [DreamerV3 Guide](rosnav_rl/model/dreamerv3/package_description.md)

> #### 👀 Adding a New Observation Space
> Inherit from `BaseObservationSpace` and register it with the `SpaceFactory`. Set the `required_observation_units` attribute for automatic resolution. Access collected data from the `ObservationDict` and encode it for the model.
> ```python
> @SpaceFactory.register("laser")
> class LaserScanSpace(BaseObservationSpace):
>     name = "LASER"
>     required_observation_units = [LaserCollector]
> 
>     def __init__(self, laser_num_beams: int, laser_max_range: float, *args, **kwargs):
>         # ...
> 
>     def get_gym_space(self) -> spaces.Space:
>         # ...
> 
>     @BaseObservationSpace.apply_normalization
>     def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> LaserCollector.data_class:
>         return observation[LaserCollector.name]
> ```

> #### ➕ Adding a New Observation Unit
> Use `ObservationCollectorUnit` to collect and preprocess data from ROS topics. Use `ObservationGeneratorUnit` to generate new observations from collected data.
> ```python
> # Collector Example
> class LaserCollector(ObservationCollectorUnit[sensor_msgs.LaserScan, np.ndarray]):
>     name: ClassVar[str] = "laser_scan"
>     topic: ClassVar[str] = "scan"
>     # ...
>     def preprocess(self, msg: sensor_msgs.LaserScan) -> np.ndarray:
>         # ...
> 
> # Generator Example
> class MyObservationGeneratorUnit(ObservationGeneratorUnit[np.ndarray]):
>     name: ClassVar[str] = "my_observation_generator"
>     requires: ClassVar[List[BaseUnit]] = [LaserCollector]
>     # ...
>     def generate(self, obs_dict: ObservationDict, ...) -> np.ndarray:
>         # ...
> ```

> #### 🎁 Adding a New Reward Component
> Inherit from `RewardUnit` and register it with the `RewardUnitFactory`.
> ```python
> @RewardUnitFactory.register("my_reward")
> class MyRewardUnit(RewardUnit):
>     def __call__(self, state_container, *args, **kwargs):
>         # Calculate and return reward
>         self.add_reward(...)
>         self.add_info(...)
> ```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

For questions, please open an issue on the GitHub repository.

> *Note: This guide provides a high-level overview. For detailed implementation specifics, refer to the inline documentation within the code.*