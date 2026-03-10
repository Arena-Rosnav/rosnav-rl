# Observations Package

> Back to [README](../../README.md) · [Developer Guide](../../GUIDE.md) · [Tutorials](../../TUTORIALS.md)

The observations package provides a complete, YAML-driven pipeline for
collecting raw sensor data from ROS 2 topics and transforming it into
meaningful features for reinforcement learning agents.

## Architecture

```
ROS Topics ──▶ Collectors ──▶ [Sync] ──▶ ObservationManager
                                              │
                                         Generators (computed on-demand)
                                              │
                                         get_observations() ──▶ RL Agent
```

**Collectors** subscribe to ROS 2 topics, preprocess raw messages, and cache
the latest value.  **Generators** derive new features from collector outputs
(and from other generators), with execution order resolved automatically via
topological sort.  The entire pipeline is configured declaratively through
`observations.yaml`.

## Directory Structure

```
observations/
├── core/                    # Orchestration
│   ├── manager.py           # ObservationManager – central entry point
│   └── pipeline.py          # ObservationPipeline – collect → generate flow
├── data_sources/            # Data source base classes & implementations
│   ├── base.py              # DataSource, Collector[M, T], Generator[T]
│   ├── collectors.py        # Concrete collector classes
│   └── generators.py        # Concrete generator classes
├── strategies/              # Strategy-pattern components
│   ├── collector.py         # CollectorManager – collection logic
│   ├── generator.py         # GeneratorManager – execution & validation
│   ├── subscription.py      # SubscriptionManager – ROS topic setup & sync
│   └── waiting.py           # WaitingStrategy – polling for stale data
├── factory/                 # Creation & dependency resolution
│   ├── factory.py           # ObservationFactory – YAML → data-source instances
│   └── resolver.py          # DependencyResolver – topological ordering
├── utils/                   # Helpers
│   ├── constants.py         # Shared constants
│   ├── static.py            # Static utility functions
│   ├── types.py             # NumPy type aliases (Pose2D, LidarRanges, …)
│   ├── pose.py              # Pose conversion utilities
│   └── semantic.py          # Relative-position / velocity helpers
└── observations.yaml        # Default pipeline configuration
```

## Core Components

### ObservationManager

The central class that owns all collectors, generators, and strategies.
Create it from a YAML config file or dictionary:

```python
obs_manager = ObservationManager.from_config(
    config=config,
    node=ros_node,
    ns="jackal",
    simulation_state_container=sim_state,
)

# Every step:
obs = obs_manager.get_observations()
```

Key responsibilities:
- Owns all `Collector` and `Generator` instances
- Delegates collection to `CollectorManager`, generation to `GeneratorManager`
- Manages ROS subscriptions via `SubscriptionManager`
- Supports optional temporal synchronization across topics via `message_filters`

### DataSource Hierarchy

```
DataSource (ABC)
├── Collector[RosMessageType, ProcessedDataType]
│   ├── LaserScanCollector          sensor_msgs/LaserScan → LidarRanges
│   ├── OdometryCollector           nav_msgs/Odometry → Pose2D
│   ├── PoseStampedCollector        geometry_msgs/PoseStamped → Pose2D
│   ├── TwistCollector              geometry_msgs/Twist → RobotVelocity
│   ├── CollisionMonitorStateCollector
│   ├── PathCollector               nav_msgs/Path → NavigationPath
│   ├── ImageColorCollector         sensor_msgs/Image → ImageData
│   ├── PeopleCollector             people_msgs/People → PedestrianDetections
│   └── ArenaPedestrianCollector    arena_people_msgs → ArenaPedestrianDetections
└── Generator[OutputType]
    ├── RobotPoseTFGenerator                    TF tree → Pose2D
    ├── GoalLocationInRobotFrameGenerator       → RobotRelativePosition
    ├── SubgoalLocationInRobotFrameGenerator    → RobotRelativePosition
    ├── DistAngleToGoalGenerator                → DistanceAngleMetrics
    ├── DistAngleToSubgoalGenerator             → DistanceAngleMetrics
    ├── LaserSafeDistanceGenerator              → SafetyStatus
    ├── PedestrianRelativeLocationGenerator     → PedestrianRelativeLocations
    ├── PedestrianLocationGenerator             → PedestrianWorldLocations
    ├── PedestrianRelativeVel[X|Y]Generator     → PedestrianRelativeVelocities
    ├── PedestrianDistanceGenerator             → PedestrianTypeMinDistances
    ├── PedestrianTypeGenerator                 → PedestrianTypeArray
    ├── PedestrianSocialStateGenerator          → PedestrianSocialStates
    └── Arena* variants (ArenaPedestrian…)
```

### Dependency Resolution

Generators declare their inputs via a `requires` dict mapping data-source
names to expected types:

```python
class DistAngleToGoalGenerator(Generator[DistanceAngleMetrics]):
    requires = {
        "robot_pose": Pose2D,
        "goal_pose": GoalLocation,
    }
```

At initialisation the `DependencyResolver` builds a dependency graph and
produces a topological execution order.  Circular dependencies raise a
`ValueError`.

### Strategies

| Strategy              | Role |
| --------------------- | ---- |
| `CollectorManager`    | Iterates collectors, detects stale data, triggers waits |
| `GeneratorManager`    | Executes generators in dependency order with validation |
| `SubscriptionManager` | Sets up individual & synchronized (message_filters) ROS subscriptions |
| `WaitingStrategy`     | Polls collectors with simulation-time-aware timeouts |

## Configuration (`observations.yaml`)

The YAML file has three sections:

```yaml
# 1. Semantic aliases — decouple observation spaces from sensors
aliases:
  robot_pose: robot_pose_from_tf
  people_data: people_detections

# 2. Data sources — collectors and generators
datasources:
  front_laser:
    type: LaserScanCollector
    params:
      topic: "lidar"
      up_to_date_required: true

  robot_pose_from_tf:
    type: RobotPoseTFGenerator
    params: {}

  dist_angle_to_goal:
    type: DistAngleToGoalGenerator
    params: {}
```

Aliases let observation spaces reference logical names (e.g. `robot_pose`)
instead of concrete data-source names, making it trivial to swap sensors
without touching space definitions.

## Import API

```python
# Public API
from rosnav_rl.observations import ObservationManager, ObservationPipeline

# Data sources
from rosnav_rl.observations.data_sources.base import Collector, Generator, DataSource

# Strategies
from rosnav_rl.observations.strategies.collector import CollectorManager
from rosnav_rl.observations.strategies.generator import GeneratorManager
from rosnav_rl.observations.strategies.subscription import SubscriptionManager

# Factory & resolution
from rosnav_rl.observations.factory.factory import ObservationFactory
from rosnav_rl.observations.factory.resolver import DependencyResolver
```

## Adding a New Collector

1. Subclass `Collector[RosMessageType, ProcessedDataType]` in `data_sources/collectors.py`.
2. Implement `_preprocess(self, msg) -> ProcessedDataType`.
3. Reference it by class name in `observations.yaml`.

```python
class DepthImageCollector(Collector[sensor_msgs.Image, np.ndarray]):
    def _preprocess(self, msg: sensor_msgs.Image) -> np.ndarray:
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
```

## Adding a New Generator

1. Subclass `Generator[OutputType]` in `data_sources/generators.py`.
2. Set `requires` to declare dependencies.
3. Implement `_generate(self, **deps, simulation_state_container, **kwargs)`.

```python
class MinObstacleDistGenerator(Generator[float]):
    requires = {"front_laser": LidarRanges}

    def _generate(self, front_laser: LidarRanges, **kwargs) -> float:
        return float(np.min(front_laser))
```
