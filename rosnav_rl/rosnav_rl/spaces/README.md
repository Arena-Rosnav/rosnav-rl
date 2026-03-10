# Spaces Package

> Back to [README](../../README.md) · [Developer Guide](../../GUIDE.md) · [Tutorials](../../TUTORIALS.md)
>
> Action and observation space management with registry-based composition, parallel encoding, category-organized spaces, and configurable normalization.

## Architecture Overview

```
spaces/
├── space_manager/
│   └── base_space_manager.py          # BaseSpaceManager (orchestrator)
├── observation_space/
│   ├── observation_space_manager.py   # ObservationSpaceManager (parallel encoding)
│   ├── observation_space_factory.py   # SpaceFactory (registry + auto_name + aliases)
│   ├── space_categories.py            # SpaceCategory enum (6 categories)
│   ├── normalization.py               # Normalizer ABC + 4 implementations
│   └── spaces/                        # Concrete observation spaces
│       ├── base_observation_space.py   # BaseObservationSpace ABC
│       ├── perception/                 # Laser, vision
│       │   ├── laser/                  # LaserScanSpace, ReducedLaserScanSpace, ReliableLaserSpace, ...
│       │   └── vision/                 # RGBDSpace
│       ├── navigation/                 # DistAngleToGoalSpace, RobustGoalSpace, ...
│       ├── dynamics/                   # LastActionSpace, MotionStateSpace, KinematicStateSpace, ...
│       ├── environment/                # EnvironmentContextSpace, SpatialAwarenessSpace, ...
│       ├── localization/               # RobustOdometrySpace, PoseStabilizedSpace, ...
│       └── meta/                       # IsFirstStepSpace, EpisodeStepSpace, MissionContextSpace, ...
└── action_space/
    └── action_space_manager.py        # ActionSpaceManager (discrete/continuous, holonomic)
```

## Core Components

### BaseSpaceManager

The top-level orchestrator that sits between the environment and the RL model. It owns an `ActionSpaceManager` and an `ObservationSpaceManager`, providing unified `encode_observation()` and `decode_action()` methods.

```python
from rosnav_rl.spaces.space_manager.base_space_manager import BaseSpaceManager
from rosnav_rl.states import AgentStateContainer

manager = BaseSpaceManager(agent_state_container=agent_state_container)

# During each step:
encoded_obs = manager.encode_observation(obs_dict)  # → np.ndarray or dict
action_cmd  = manager.decode_action(raw_action)      # → np.ndarray [linear.x, linear.y, angular.z]
```

**Initialization flow:**
1. Merges settings from `AgentStateContainer.observation_space` into observation space constructor kwargs
2. Creates `ObservationSpaceManager` and calls `load_configuration(config)`
3. Creates `ActionSpaceManager` from `AgentStateContainer.action_space`

### ObservationSpaceManager

Manages a collection of `BaseObservationSpace` modules and produces encoded observations.

**Key features:**
- **Parallel encoding**: Uses `ThreadPoolExecutor` (up to 4 workers) when `parallel_encoding=True` and multiple spaces are loaded
- **Validate-once pattern**: `validate_observations=True` on first call, then permanently disabled for zero-overhead steady-state
- **Auto-collapse**: If only one space is loaded, returns the encoded value directly instead of a `Dict`
- **Performance caching**: Pre-caches `_space_required_keys` per space to avoid `dict.keys()` on every step

```python
from rosnav_rl.spaces.observation_space.observation_space_manager import ObservationSpaceManager

manager = ObservationSpaceManager(parallel_encoding=True)
manager.load_configuration({
    "ReliableLaserSpace": {"laser_num_beams": 360, "laser_max_range": 30.0},
    "DistAngleToGoalSpace": {"goal_max_dist": 10.0},
    "LastActionSpace": {},
})

# Properties
manager.observation_space       # → gymnasium.spaces.Dict or single Space
manager.space_list              # → list of BaseObservationSpace instances
manager.required_observations   # → {"ReliableLaserSpace": ["front_laser"], ...}
manager.get_available_spaces_by_category()  # → {"perception": [...], "navigation": [...], ...}
```

### SpaceFactory

Registry-based factory with flexible naming and categorization.

**Registration modes:**
```python
# 1. Explicit name
@SpaceFactory.register("laser_scan")
class LaserScanSpace(BaseObservationSpace): ...

# 2. Auto-derived names (PascalCase primary + snake_case alias)
@SpaceFactory.register(auto_name=True)
class MotionStateSpace(BaseObservationSpace): ...
# Registered as: "MotionStateSpace" (primary) + "motion_state" (alias)

# 3. Explicit name with aliases
@SpaceFactory.register("motion", aliases=["motion_state", "MotionStateSpace"])
class MotionSpace(BaseObservationSpace): ...

# 4. With category tag
@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class MyPerceptionSpace(BaseObservationSpace): ...
```

**API:**
| Method | Description |
| --- | --- |
| `register(name, category, auto_name, aliases)` | Class decorator for registration |
| `instantiate(name, **kwargs)` | Create space instance by name |
| `get_spaces_by_category()` | `Dict[str, list]` organized by category |
| `get_category(name)` | Category string for a space |
| `list_aliases(name)` | All registered names for a space's class |
| `find_space_by_class_name(class_name)` | Find registrations by Python class name |

### SpaceCategory

Type-safe enum for categorizing observation spaces:

| Category | Description | Example Spaces |
| --- | --- | --- |
| `PERCEPTION` | Sensor-based observations | `LaserScanSpace`, `ReducedLaserScanSpace`, `ReliableLaserSpace`, `RGBDSpace` |
| `NAVIGATION` | Goal and path-related | `DistAngleToGoalSpace`, `DistAngleToSubgoalSpace`, `RobustGoalSpace`, `MultiScaleGoalSpace` |
| `DYNAMICS` | Robot motion and action | `LastActionSpace`, `MotionStateSpace`, `KinematicStateSpace`, `TrajectoryStateSpace` |
| `ENVIRONMENT` | Environmental context | `EnvironmentContextSpace`, `SpatialAwarenessSpace`, `ObstacleProximitySpace` |
| `LOCALIZATION` | Robot pose | `RobustOdometrySpace`, `PoseStabilizedSpace`, `LocalizationCombinedSpace` |
| `META` | Episode metadata | `IsFirstStepSpace`, `IsTerminalStepSpace`, `EpisodeStepSpace`, `MissionContextSpace` |

### BaseObservationSpace

Abstract base for all observation spaces. Inherits from `ErrorReportingMixin` (structured logging) and `RequiresProtocol` (schema-based dependency validation).

**Class interface:**
```python
class MySpace(BaseObservationSpace):
    name: ClassVar[str] = "MY_SPACE"                    # Unique identifier
    requires: ClassVar[Dict[str, Any]] = {               # Typed dependency schema
        "front_laser": LidarRanges,
    }

    def get_gym_space(self) -> spaces.Space: ...         # Define shape/type
    def encode_observation(self, front_laser, ...) -> np.ndarray: ...  # Typed kwargs
    def reset(self) -> None: ...                          # Optional: episode reset
```

**Built-in features:**
- **Normalization**: Configurable via constructor (`normalize=True, normalizer="max_abs"`). Four normalizers:
  - `max_abs` (MaxAbsScaler): Scales to [-1, 1] using space bounds
  - `min_max` (MinMaxScaler): Scales to [0, 1]
  - `standard` (StandardScaler): Zero mean, unit variance estimate from bounds
  - `identity`: No-op passthrough
- **Decorators**:
  - `@BaseObservationSpace.apply_normalization` — auto-normalizes the return value
  - `@BaseObservationSpace.check_dtype` — validates array for NaN/Inf, replaces with zeros
- **Safe encoding**: `safe_encode_observation()` wraps `encode_observation()` with try/except, returning a properly shaped null array on failure
- **Null fallback**: `_create_null_observation()` generates zero-filled arrays matching the gym space for graceful degradation

### ActionSpaceManager

Manages discrete and continuous action spaces for holonomic and non-holonomic robots.

**Discrete actions:**
```python
manager = ActionSpaceManager(
    is_holonomic=False,
    is_discrete=True,
    actions=[
        {"linear": 0.2, "angular": 0.0},
        {"linear": 0.0, "angular": 0.5},
    ],
)
# action_space → spaces.Discrete(2)
# decode_action(0) → np.array([0.2, 0.0, 0.0])  # [linear.x, linear.y, angular.z]
```

**Continuous actions:**
```python
# Non-holonomic (2-DOF → 3-DOF with zero linear.y)
manager = ActionSpaceManager(
    is_holonomic=False,
    is_discrete=False,
    actions={"linear_range": [0.0, 1.0], "angular_range": [-1.0, 1.0]},
)
# action_space → spaces.Box(low=[0, -1], high=[1, 1])
# decode_action([0.5, 0.3]) → np.array([0.5, 0.0, 0.3])

# Holonomic (3-DOF)
manager = ActionSpaceManager(
    is_holonomic=True,
    is_discrete=False,
    actions={
        "linear_range": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        "angular_range": [-1.0, 1.0],
    },
)
# action_space → spaces.Box(low=[-1, -1, -1], high=[1, 1, 1])
# decode_action([0.5, -0.2, 0.3]) → np.array([0.5, -0.2, 0.3])
```

## Data Flow

### Encoding Pipeline (per step)

```
ObservationDict (from ObservationManager)
    │
    ▼
BaseSpaceManager.encode_observation(obs_dict)
    │
    ├─ ObservationSpaceManager.encode_observation(obs_dict)
    │   │
    │   ├─ [First call only] validate_observation_spaces() → then disable
    │   │
    │   ├─ For each loaded space:
    │   │   ├─ _extract_space_args(space_name, obs_dict) → {key: value, ...}
    │   │   └─ space.safe_encode_observation(**args) → np.ndarray
    │   │       ├─ encode_observation(front_laser=..., ...) → raw array
    │   │       ├─ @apply_normalization (if enabled)
    │   │       └─ On error: _create_null_observation() → zeros
    │   │
    │   └─ If parallel_encoding: ThreadPoolExecutor(max_workers=4)
    │
    └─ Return: np.ndarray (single space) or Dict[str, np.ndarray] (multiple spaces)
```

### Decoding Pipeline (per step)

```
Raw action from RL model (np.ndarray)
    │
    ▼
BaseSpaceManager.decode_action(action)
    │
    ├─ ActionSpaceManager.decode_action(action)
    │   ├─ If discrete: lookup action index → {"linear": x, "angular": z}
    │   ├─ If non-holonomic: [linear.x, angular.z] → [linear.x, 0.0, angular.z]
    │   └─ If holonomic: [linear.x, linear.y, angular.z] (passthrough)
    │
    └─ Return: np.ndarray [linear.x, linear.y, angular.z]
```

## Adding a New Observation Space

1. **Create the space** in the appropriate category directory (e.g., `spaces/perception/`):
```python
from rosnav_rl.observations.utils.types import LidarRanges
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import BaseObservationSpace

@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class MyCustomLaserSpace(BaseObservationSpace):
    name = "MyCustomLaserSpace"
    requires = {"front_laser": LidarRanges}

    def __init__(self, laser_num_beams: int = 360, laser_max_range: float = 30.0, *args, **kwargs):
        self._num_beams = laser_num_beams
        self._max_range = laser_max_range
        super().__init__(*args, **kwargs)  # Pass normalize, normalizer to parent

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=self._max_range, shape=(self._num_beams,), dtype=np.float32)

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, front_laser: LidarRanges, *args, **kwargs) -> np.ndarray:
        return np.clip(front_laser, 0, self._max_range).astype(np.float32)
```

2. **Import in category `__init__.py`** so auto-loading registers it:
```python
# spaces/perception/__init__.py
from .laser import basic_laser_spaces, laser_spaces
from . import my_custom_module  # triggers @SpaceFactory.register
```

3. **Use in agent config** by referencing the registered name:
```python
# "MyCustomLaserSpace" (primary) or "my_custom_laser" (auto-alias)
observation_spaces = ["MyCustomLaserSpace", "DistAngleToGoalSpace"]
```
