# RosNav-RL Tutorials

> Step-by-step guides for common development tasks.
> For architecture and core concepts see the **[Developer Guide](GUIDE.md)**.

---

## Table of Contents

1. [Training a New Agent](#1-training-a-new-agent)
2. [Deploying a Trained Agent](#2-deploying-a-trained-agent)
3. [Adding a New SB3 Algorithm](#3-adding-a-new-sb3-algorithm)
4. [Adding a New RL Framework](#4-adding-a-new-rl-framework)
5. [Adding a New Observation Space](#5-adding-a-new-observation-space)
6. [Adding a New Observation Data Source](#6-adding-a-new-observation-data-source)
7. [Adding a New Reward Unit](#7-adding-a-new-reward-unit)
8. [Adding a New Model Architecture](#8-adding-a-new-model-architecture)

---

## 1. Training a New Agent

### Overview

Training requires three things: a **Gym environment**, an **agent config**, and
a **training script**. The framework handles model creation, space encoding,
and reward computation internally.

### Step 1 — Create a Gym Environment

Your environment's `step()` uses `ObservationManager` for sensor data and
`RewardFunction` for the reward signal:

```python
import gymnasium as gym
from rosnav_rl.observations import ObservationManager
from rosnav_rl.reward.reward_function import RewardFunction

class NavEnv(gym.Env):
    def __init__(self, obs_manager, reward_fn, space_manager):
        self.obs_manager = obs_manager
        self.reward_fn = reward_fn
        self.space_manager = space_manager

        self.observation_space = space_manager.observation_space
        self.action_space = space_manager.action_space

    def step(self, action):
        # 1. Execute action in simulation
        cmd = self.space_manager.decode_action(action)
        self._send_command(cmd)

        # 2. Collect observations
        obs_dict = self.obs_manager.get_observations()
        encoded_obs = self.space_manager.encode_observation(obs_dict)

        # 3. Calculate reward
        reward, info = self.reward_fn.calculate_reward(obs_dict, self.sim_state)

        return encoded_obs, reward, info.get("is_done", False), False, info

    def reset(self, **kwargs):
        self._reset_simulation()
        self.reward_fn.reset()
        obs_dict = self.obs_manager.get_observations()
        return self.space_manager.encode_observation(obs_dict), {}
```

### Step 2 — Configure the Agent

```python
import rosnav_rl
from rosnav_rl.model.stable_baselines3.cfg import (
    StableBaselinesCfg, PPO_Cfg, PPO_Algorithm_Cfg,
)

agent_cfg = rosnav_rl.AgentCfg(
    name="my_ppo_agent",        # auto-generated if omitted
    robot="jackal",
    framework=StableBaselinesCfg(
        algorithm=PPO_Cfg(
            architecture_name="AGENT_1",
            parameters=PPO_Algorithm_Cfg(
                total_timesteps=5_000_000,
                learning_rate=3e-4,
                clip_range=0.2,
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
    action_space=rosnav_rl.cfg.ActionSpaceCfg(is_discrete=False),
)
```

### Step 3 — Build and Train

```python
# Build state containers
sim_state = rosnav_rl.SimulationStateContainer(...)
agent_state = sim_state.to_agent_state_container()

# Create the agent
agent = rosnav_rl.RL_Agent(
    agent_cfg=agent_cfg,
    agent_state_container=agent_state,
)
agent.initialize_model()

# Create vectorized environments
train_envs = [make_env("train", i) for i in range(num_envs)]
eval_envs  = [make_env("eval", i) for i in range(num_eval_envs)]

# Train
agent.train(train_envs=train_envs, eval_envs=eval_envs)
```

### Step 4 — Verify

The trained agent is saved to `arena_training/agents/<agent_name>/` with:
- `training_config.yaml` — full configuration snapshot
- `best_model.zip` — model checkpoint

Load and test:
```python
agent.load_model(path="arena_training/agents/my_ppo_agent/best_model.zip")
action = agent.get_action(obs_dict)
```

> For a complete working example, see the
> **[Arena-Rosnav](https://github.com/Arena-Rosnav/arena-rosnav)** training
> scripts and [documentation](https://arena-rosnav.readthedocs.io/).

---

## 2. Deploying a Trained Agent

### Standalone

```bash
ros2 run rosnav_rl action_server.py --ros-args -p agent_name:=my_ppo_agent
```

Or via launch file:
```bash
ros2 launch rosnav_rl action_server.launch.py agent_name:=my_ppo_agent
```

### Arena Integration

When using Arena-Rosnav, the action server starts automatically:
```bash
arena launch local_planner:=rosnav_rl agent_name:=my_ppo_agent
```

### Calling the Service

The server exposes `get_command` (`rosnav_rl_msgs/srv/GetCommand`) returning
`geometry_msgs/Twist`. The agent reads its own ROS topics — the request is empty.

```bash
ros2 service call /get_command rosnav_rl_msgs/srv/GetCommand {}
```

From a ROS 2 node:
```python
import rclpy
from rosnav_rl_msgs.srv import GetCommand

rclpy.init()
node = rclpy.create_node("planner")
client = node.create_client(GetCommand, "get_command")
client.wait_for_service()

future = client.call_async(GetCommand.Request())
rclpy.spin_until_future_complete(node, future)
twist = future.result().twist  # geometry_msgs/Twist
# twist.linear.x, twist.linear.y, twist.angular.z

node.destroy_node()
rclpy.shutdown()
```

### Agent Directory Structure

```
arena_training/agents/my_ppo_agent/
├── training_config.yaml    # Full TrainingCfg (AgentCfg + ArenaCfg)
├── best_model.zip          # SB3 checkpoint
└── observations.yaml       # (Optional) agent-specific observation pipeline
```

If `observations.yaml` is not present, the default at
`rosnav_rl/observations/observations.yaml` is used.

### Custom Server

To deploy outside Arena, implement your own `ActionServer`:

```python
from rosnav_rl.action_server.base_server import ActionServer, ObservationCollector
from rosnav_rl.rl_agent import RL_Agent

class MyActionServer(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        agent = RL_Agent(agent_cfg=..., agent_state_container=...)
        agent.load_model(path="path/to/model.zip")
        return agent

    def _initialize_observation_collector(self) -> ObservationCollector:
        return MyObsCollector(node=self.node, namespace=self.namespace)

server = MyActionServer(agent_name="my_agent", namespace="/robot0")
server.start()  # Blocks — spins the ROS 2 node
```

> See [action_server/README.md](rosnav_rl/action_server/README.md) for full details.

---

## 3. Adding a New SB3 Algorithm

Adding a new Stable Baselines 3 (or sb3-contrib) algorithm requires **only configuration changes** — no modifications to `sb3_model.py` or `model_factory.py`.

### Step 1 — Create the Config

Create a file in `rosnav_rl/model/stable_baselines3/cfg/` (e.g., `myalgo.py`):

```python
from .base import OnPolicyParameters, SBAlgorithmCfg  # or OffPolicyParameters

class MyAlgo_Algorithm_Cfg(OnPolicyParameters):
    algorithm_name: str = "MyAlgo"
    my_custom_param: float = 0.42

class MyAlgo_Cfg(SBAlgorithmCfg):
    parameters: MyAlgo_Algorithm_Cfg
```

### Step 2 — Export

Add to `cfg/__init__.py`:
```python
from .myalgo import MyAlgo_Algorithm_Cfg, MyAlgo_Cfg
```

### Step 3 — Add to the Union

In `cfg/framework.py`, add to the algorithm union:
```python
algorithm: Union[..., MyAlgo_Cfg, SBAlgorithmCfg]
```

### Step 4 — Register the Policy Type

In `policy/constants.py`:
```python
from my_sb3_package import MyAlgo
POLICY_TYPE[MyAlgo] = "MultiInputPolicy"
```

### Step 5 — Add to Type Aliases

In `utils/type_aliases/models.py`, add `MyAlgo` to `_SupportedStableBaselinesModels`.

**That's it.** The `ModelFactory`, `StableBaselinesModel`, and the entire
training pipeline pick up the new algorithm automatically.

> See [model/README.md](rosnav_rl/model/README.md) for the full config hierarchy.

---

## 4. Adding a New RL Framework

To integrate a completely new RL library (e.g., CleanRL, RLlib):

### Step 1 — Implement `RL_Model`

```python
from rosnav_rl.model.model import RL_Model

class MyFrameworkModel(RL_Model):
    def setup_model(self, *args, **kwargs):
        self._model = ...  # Initialize your algorithm

    def train(self, *args, **kwargs):
        ...  # Training loop

    def save(self, path, *args, **kwargs):
        ...  # Persist weights

    def load(self, path, *args, **kwargs):
        ...  # Restore weights

    def get_action(self, observation, *args, **kwargs):
        ...  # Inference

    @classmethod
    def from_framework_cfg(cls, rl_agent, framework_cfg, *args, **kwargs):
        return cls(rl_agent=rl_agent, algorithm_cfg=framework_cfg.algorithm)

    @property
    def observation_space_list(self):
        return [...]  # BaseObservationSpace classes

    @property
    def observation_space_kwargs(self):
        return {}

    @property
    def parameter_number(self):
        return sum(p.numel() for p in self._model.parameters())
```

### Step 2 — Register with ModelFactory

```python
from rosnav_rl.model.model_factory import ModelFactory
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

ModelFactory.register_model(SupportedRLFrameworks.MY_FRAMEWORK, MyFrameworkModel)
```

### Step 3 — Create a FrameworkCfg

```python
from rosnav_rl.cfg.framework import FrameworkCfg

class MyFrameworkCfg(FrameworkCfg):
    name: str = "my_framework"
    learning_rate: float = 1e-3
    batch_size: int = 64
```

### Step 4 — Update AgentCfg

Add your config to the discriminated union in `cfg/agent.py`:
```python
framework: Annotated[
    Union[StableBaselinesCfg, DreamerV3Cfg, MyFrameworkCfg],
    Discriminator(discriminator="name"),
]
```

Now `AgentCfg(framework={"name": "my_framework", ...})` automatically creates
your config.

---

## 5. Adding a New Observation Space

Observation spaces encode specific raw observations into neural-network-ready
tensors. Each space is a `BaseObservationSpace` subclass registered with the
`SpaceFactory`.

### Step 1 — Create the Space

Create a file in the appropriate category directory under
`rosnav_rl/spaces/observation_space/spaces/` (e.g., `perception/my_space.py`):

```python
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import LidarRanges
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)

@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class FilteredLaserSpace(BaseObservationSpace):
    """Laser scan with median filtering and range clipping."""

    name = "FilteredLaserSpace"
    requires = {"front_laser": LidarRanges}  # Schema-based dependency declaration

    def __init__(
        self,
        laser_num_beams: int = 360,
        laser_max_range: float = 30.0,
        kernel_size: int = 5,
        *args, **kwargs,
    ):
        self._num_beams = laser_num_beams
        self._max_range = laser_max_range
        self._kernel = kernel_size
        super().__init__(*args, **kwargs)  # Pass normalize, normalizer to parent

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=self._max_range,
            shape=(self._num_beams,),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, front_laser: LidarRanges, *args, **kwargs) -> np.ndarray:
        # Typed kwargs match the 'requires' dict — no manual dict unpacking
        from scipy.ndimage import median_filter
        filtered = median_filter(front_laser, size=self._kernel)
        return np.clip(filtered, 0, self._max_range).astype(np.float32)
```

**Key points:**
- `requires` is a `ClassVar[Dict[str, Any]]` — declares typed data dependencies
- `encode_observation()` receives **typed keyword arguments** matching `requires`, not an `ObservationDict`
- `@apply_normalization` auto-normalizes if `normalize=True` was passed to the constructor
- Use `@BaseObservationSpace.check_dtype` to validate for NaN/Inf values
- `safe_encode_observation()` (called by the manager) wraps encoding with try/except and returns zeros on failure

### Step 2 — Register via Import

Add an import in the category's `__init__.py` to trigger registration:

```python
# spaces/perception/__init__.py
from . import my_space  # triggers @SpaceFactory.register
```

### Step 3 — Use It

Reference the space by its registered name in agent configurations:

```python
# By auto-name: "FilteredLaserSpace" (primary) or "filtered_laser" (auto-alias)
observation_spaces = ["FilteredLaserSpace", "DistAngleToGoalSpace", "LastActionSpace"]
```

Or in a policy description:
```python
@AgentFactory.register("MY_AGENT")
class MyAgent(StableBaselinesPolicyDescription):
    observation_spaces = [FilteredLaserSpace, DistAngleToSubgoalSpace, LastActionSpace]
    ...
```

### Available normalizers

| Name | Class | Behavior |
| --- | --- | --- |
| `max_abs` | `MaxAbsScaler` | Scales to [-1, 1] using space bounds |
| `min_max` | `MinMaxScaler` | Scales to [0, 1] |
| `standard` | `StandardScaler` | Zero mean, unit variance estimate from bounds |
| `identity` | `IdentityNormalizer` | No-op passthrough |

> See [spaces/README.md](rosnav_rl/spaces/README.md) for the full encoding/decoding pipeline.

---

## 6. Adding a New Observation Data Source

Data sources feed the observation pipeline. **Collectors** subscribe to ROS 2
topics; **Generators** derive features from other data sources.

### Adding a Collector

Subclass `Collector[RosMessageType, ProcessedType]` and implement `_preprocess()`:

```python
import numpy as np
import sensor_msgs.msg as sensor_msgs
from rosnav_rl.observations.data_sources.base import Collector

class DepthImageCollector(Collector[sensor_msgs.Image, np.ndarray]):
    """Collects depth images and converts to float32 arrays."""

    def _preprocess(self, msg: sensor_msgs.Image) -> np.ndarray:
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
```

### Adding a Generator

Subclass `Generator[OutputType]`, set `requires`, and implement `_generate()`:

```python
import numpy as np
from rosnav_rl.observations.data_sources.base import Generator
from rosnav_rl.observations.utils.types import LidarRanges

class MinObstacleDistGenerator(Generator[float]):
    """Derives the minimum obstacle distance from laser data."""

    requires = {"front_laser": LidarRanges}

    def _generate(self, front_laser: LidarRanges, **kwargs) -> float:
        return float(np.min(front_laser))
```

Dependencies declared in `requires` are validated and injected automatically.
The `DependencyResolver` uses topological sort (Kahn's algorithm) to determine
execution order. Circular dependencies raise a `ValueError`.

### Register in observations.yaml

Reference data sources by class name:

```yaml
datasources:
  depth_image:
    type: DepthImageCollector
    params:
      topic: "depth/image"
      up_to_date_required: true

  min_obstacle_dist:
    type: MinObstacleDistGenerator
    params: {}

# Optional: aliases decouple observation spaces from concrete sources
aliases:
  obstacle_distance: min_obstacle_dist
```

> See [observations/README.md](rosnav_rl/observations/README.md) for the full
> DataSource hierarchy, strategies, and YAML config reference.

---

## 7. Adding a New Reward Unit

Reward units are self-contained reward components. Each declares its data
dependencies via `requires` and contributes to the total reward via
`add_reward()`.

### Step 1 — Create the Unit

Add to `rosnav_rl/reward/reward_units/reward_units.py` (or a new file):

```python
import numpy as np

from rosnav_rl.observations.utils.types import LidarRanges, DistanceAngleMetrics
from rosnav_rl.reward.reward_units.base_reward_units import RewardUnit
from rosnav_rl.reward.reward_units.reward_unit_factory import RewardUnitFactory
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.reward.utils import check_params

@RewardUnitFactory.register("proximity_penalty")
class ProximityPenalty(RewardUnit):
    """Penalizes getting too close to obstacles."""

    # Schema-based typed dependencies — validated at runtime
    requires = {
        "front_laser": LidarRanges,
        "dist_angle_to_goal": DistanceAngleMetrics,
    }

    @check_params  # Warns if |reward| > 100
    def __init__(
        self,
        reward_function: RewardFunction,
        penalty: float = -0.5,
        threshold: float = 0.3,
        _on_safe_dist_violation: bool = False,
        *args, **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._penalty = penalty
        self._threshold = threshold

    def reset(self) -> None:
        """Reset episode-local state (called at episode start)."""
        pass

    def __call__(
        self,
        front_laser: LidarRanges,
        dist_angle_to_goal: DistanceAngleMetrics,
    ):
        """Called every step with the data declared in `requires`."""
        min_dist = np.min(front_laser)
        if min_dist < self._threshold:
            scaled = self._penalty * (1.0 - min_dist / self._threshold)
            self.add_reward(scaled)
            self.add_info({"min_obstacle_dist": float(min_dist)})
```

**Key points:**
- `requires` declares data dependencies as `{name: type}` — validated once on first call
- `add_reward(value)` validates against NaN, Inf, and non-numeric values
- `_on_safe_dist_violation` controls whether the unit runs during safety violations
- `@check_params` on `__init__` warns about large reward magnitudes

### Step 2 — Add Defaults (Optional)

In `rosnav_rl/reward/constants.py`:

```python
class DEFAULTS:
    class PROXIMITY_PENALTY:
        PENALTY: float = -0.5
        THRESHOLD: float = 0.3
        _ON_SAFE_DIST_VIOLATION: bool = False
```

### Step 3 — Use in Config

```yaml
reward:
  reward_function_dict:
    proximity_penalty:
      penalty: -1.0
      threshold: 0.5
      _on_safe_dist_violation: false
```

Or in Python:
```python
RewardCfg(
    reward_function_dict={
        "proximity_penalty": {
            "penalty": -1.0,
            "threshold": 0.5,
        },
    },
)
```

### Safety Categorization

```
_on_safe_dist_violation = True
  → Only evaluated when safety.violation == True
  → Prevents rewarding goal-approach while dangerously close to obstacles

_on_safe_dist_violation = False
  → Evaluated on EVERY step
```

> See [reward/README.md](rosnav_rl/reward/README.md) for the full built-in unit reference.

---

## 8. Adding a New Model Architecture

Model architectures define the neural network structure for an SB3 agent.

### Step 1 — Create a Policy Description

```python
import torch.nn as nn
from stable_baselines3 import PPO  # or SAC, TD3, etc.

from rosnav_rl.model.stable_baselines3.policy.agent_factory import AgentFactory
from rosnav_rl.model.stable_baselines3.policy.base_policy import (
    StableBaselinesPolicyDescription,
)
from rosnav_rl.spaces.observation_space import spaces

@AgentFactory.register("MY_CUSTOM_AGENT")
class MyCustomAgent(StableBaselinesPolicyDescription):
    algorithm_class = PPO

    # Observation spaces this architecture expects
    observation_spaces = [
        spaces.perception.ReducedLaserScanSpace,
        spaces.navigation.DistAngleToSubgoalSpace,
        spaces.dynamics.LastActionSpace,
    ]

    # Feature extractor (custom CNN/MLP that combines all observation spaces)
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=256)

    # Policy and value function network architecture
    net_arch = dict(pi=[128, 64], vf=[128, 64])
    activation_fn = nn.ReLU
```

### Step 2 — Use in Config

Reference the architecture by its registered name:

```python
PPO_Cfg(
    architecture_name="MY_CUSTOM_AGENT",
    parameters=PPO_Algorithm_Cfg(
        total_timesteps=5_000_000,
    ),
)
```

### DreamerV3 Architectures

For DreamerV3 model details, see
[dreamerv3/package_description.md](rosnav_rl/model/dreamerv3/package_description.md).

### Custom SB3 Policies

For advanced customization (custom policy classes, feature extractors), see
[stable_baselines3/custommodel.md](rosnav_rl/model/stable_baselines3/custommodel.md).

---

## Quick Reference

| I want to… | See |
| --- | --- |
| Train an agent | [Tutorial 1](#1-training-a-new-agent) |
| Deploy an agent | [Tutorial 2](#2-deploying-a-trained-agent) |
| Add a new SB3 algorithm | [Tutorial 3](#3-adding-a-new-sb3-algorithm) |
| Integrate a new RL library | [Tutorial 4](#4-adding-a-new-rl-framework) |
| Create a custom observation space | [Tutorial 5](#5-adding-a-new-observation-space) |
| Add a ROS topic collector | [Tutorial 6](#6-adding-a-new-observation-data-source) |
| Build a custom reward | [Tutorial 7](#7-adding-a-new-reward-unit) |
| Design a network architecture | [Tutorial 8](#8-adding-a-new-model-architecture) |
| Understand the architecture | [Developer Guide](GUIDE.md) |
| Configure an agent | [cfg/README.md](rosnav_rl/cfg/README.md) |
