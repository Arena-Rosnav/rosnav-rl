# Action Server

> Back to [README](../../README.md) · [Developer Guide](../../GUIDE.md) · [Tutorials](../../TUTORIALS.md)
>
> ROS 2 service server for deploying trained RL agents as real-time navigation controllers.

## Architecture Overview

```
action_server/
├── __init__.py
├── base_server.py    # ActionServer ABC — ROS2 service, ObservationCollector protocol
└── arena_server.py   # ArenaActionServer — Arena-specific agent loading and deployment
```

## Core Components

### ActionServer (ABC)

Abstract base class that wraps a trained `RL_Agent` in a ROS 2 service server. Provides the `GetCommand` service interface and handles scene resets.

**Service:** `get_command` (type `rosnav_rl_msgs/srv/GetCommand`)
- **Request**: Empty (agent polls its own ROS 2 topics)
- **Response**: `geometry_msgs/Twist` with `linear.x`, `linear.y`, `angular.z`

**Key features:**
- **`ObservationCollector` protocol**: Duck-typed interface — any object with `get_observations() -> ObservationDict` is accepted
- **Graceful error handling**: Logs warnings and returns zero velocity on transient failures (e.g., missing sensor data at startup)
- **Scene reset**: Subscribes to `/scenario_reset` (`std_msgs/Int16`) and calls `agent.model.reset()` on each reset

```python
class ActionServer(ABC):
    agent: RL_Agent = None
    observation_collector: ObservationCollector = None

    @abstractmethod
    def _initialize_agent(self) -> RL_Agent: ...
    @abstractmethod
    def _initialize_observation_collector(self) -> ObservationCollector: ...

    def start(self):
        """Initialize ROS services → agent → observation collector → spin."""
        self._initialize_ros()
        self.agent = self._initialize_agent()
        self.observation_collector = self._initialize_observation_collector()
        rclpy.spin(self.node)
```

### ArenaActionServer

The Arena-specific implementation that handles model loading from disk.

**Agent loading flow:**
1. Resolves agent directory via `_resolve_agent_dir(agent_name)`:
   - Checks `ROSNAV_AGENTS_DIR` environment variable
   - Searches via `ament_index` (package share directory)
   - Walks up from this file's path looking for `arena_training/agents/`
   - Checks `COLCON_PREFIX_PATH` / `AMENT_PREFIX_PATH`
2. Loads `training_config.yaml` as `TrainingCfg` (Pydantic)
3. Reconstructs `SimulationStateContainer` from the saved training config
4. Creates `RL_Agent` and loads `best_model.zip`

**Observation collector:**
- Uses `create_observation_manager_from_config()` with the agent-specific or default `observations.yaml`
- Passes the ROS 2 node for topic subscriptions

## Usage

### Starting the server standalone

```bash
ros2 run rosnav_rl action_server.py --ros-args -p agent_name:=my_trained_agent
```

### Via Arena launch (automatic)

When `local_planner:=rosnav_rl` and `train_mode:=false`, the server starts automatically:
```bash
arena launch local_planner:=rosnav_rl agent_name:=my_trained_agent
```

### Calling the service

```bash
# Empty request — agent polls its own ROS topics
ros2 service call /get_command rosnav_rl_msgs/srv/GetCommand {}
```

From a ROS 2 node:
```python
import rclpy
from rosnav_rl_msgs.srv import GetCommand

rclpy.init()
node = rclpy.create_node("planner")
client = node.create_client(GetCommand, "get_command")  # namespaced per robot
client.wait_for_service()

future = client.call_async(GetCommand.Request())
rclpy.spin_until_future_complete(node, future)
twist = future.result().twist  # geometry_msgs/Twist
# twist.linear.x, twist.linear.y, twist.angular.z

node.destroy_node()
rclpy.shutdown()
```

## Agent Directory Structure

```
arena_training/agents/<agent_name>/
├── training_config.yaml    # Full TrainingCfg (AgentCfg + ArenaCfg)
├── best_model.zip          # SB3 model checkpoint
└── observations.yaml       # (Optional) agent-specific observation pipeline config
```

If `observations.yaml` is not present in the agent directory, the server falls back to the default bundled config at `rosnav_rl/observations/observations.yaml`.

## Implementing a Custom Server

To create a server for a non-Arena environment:

```python
from rosnav_rl.action_server.base_server import ActionServer, ObservationCollector
from rosnav_rl.rl_agent import RL_Agent

class MyActionServer(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        # Load your agent however you need
        agent = RL_Agent(agent_cfg=..., agent_state_container=...)
        agent.load_model(path="path/to/model.zip")
        return agent

    def _initialize_observation_collector(self) -> ObservationCollector:
        # Return any object with get_observations() → ObservationDict
        return MyObservationCollector(node=self.node, namespace=self.namespace)

# Start the server
server = MyActionServer(agent_name="my_agent", namespace="/robot0")
server.start()  # Blocks — spins the ROS2 node
```
