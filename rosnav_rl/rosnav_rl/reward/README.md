# Reward System

> Back to [README](../../README.md) · [Developer Guide](../../GUIDE.md) · [Tutorials](../../TUTORIALS.md)
>
> Modular, composable reward functions with parallel execution, safety categorization, and schema-based typed dependencies.

## Architecture Overview

```
reward/
├── reward_function.py           # RewardFunction — composite orchestrator
├── constants.py                 # REWARD_CONSTANTS, DONE_REASONS, DEFAULTS
├── utils.py                     # @check_params decorator, point cloud helpers
├── reward_functions.md          # Legacy reward documentation
└── reward_units/
    ├── base_reward_units.py     # RewardUnit ABC (RequiresProtocol + ErrorReportingMixin)
    ├── reward_unit_factory.py   # RewardUnitFactory — registry with @register decorator
    └── reward_units.py          # 15+ concrete reward unit implementations
```

## Core Components

### RewardFunction

The central orchestrator that manages a collection of `RewardUnit`s and aggregates their outputs into a single reward signal per step.

**Key features:**
- **Parallel execution**: Optional `ThreadPoolExecutor`-based parallel evaluation (up to `max_workers=8`, configurable timeout)
- **Safety categorization**: Units are automatically partitioned into two groups:
  - `_safe_dist_violation_units`: Only evaluated when a safety distance violation is detected
  - `_non_safe_dist_violation_units`: Evaluated on every step
- **Validate-once pattern**: `RequiresProtocol` validation runs on the first `calculate_reward()` call then permanently disables for zero-overhead steady-state
- **Thread-safe accumulation**: Uses `threading.Lock` for safe reward aggregation in parallel mode
- **Episode state tracking**: `RewardState` dataclass tracks `current_reward`, `info` dict, and `reward_overview` per step

```python
from rosnav_rl.reward.reward_function import RewardFunction

reward_fn = RewardFunction(
    function_dict={
        "goal_reached":  {"reward": 15.0},
        "collision":     {"reward": -10.0},
        "approach_goal": {"pos_factor": 0.3, "neg_factor": 0.5},
        "safe_distance": {"reward": -0.15},
    },
    parallel=True,          # Enable ThreadPoolExecutor
    max_workers=4,          # Worker threads
    timeout=0.1,            # Per-step timeout (seconds)
    verbose=1,              # Detailed logging
)

# On each step:
reward, info = reward_fn.calculate_reward(observations, simulation_state_container)
```

### RewardUnit (ABC)

The abstract base for all reward components. Inherits from both `ErrorReportingMixin` (structured logging) and `RequiresProtocol` (schema-based dependency validation).

**Key features:**
- **Schema-based `requires`**: A `ClassVar[Dict[str, Any]]` mapping data source names to their types. The framework validates at runtime that all dependencies are satisfied.
- **Safe reward API**: `add_reward(value)` validates against NaN, infinity, and non-numeric values before accumulating. `add_info(info)` contributes to episode metadata.
- **Parameter checks**: `check_parameters()` (invoked via `@check_params` decorator on `__init__`) warns about large reward magnitudes (|reward| > 100).
- **Safety flag**: `_on_safe_dist_violation` controls whether the unit is evaluated during safety violations.

```python
from rosnav_rl.reward.reward_units.base_reward_units import RewardUnit
from rosnav_rl.reward.reward_units.reward_unit_factory import RewardUnitFactory
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.reward.utils import check_params
from rosnav_rl.observations.utils.types import DistanceAngleMetrics
from rosnav_rl.states import SimulationStateContainer

@RewardUnitFactory.register("my_approach_reward")
class MyApproachReward(RewardUnit):
    """Example reward unit with schema-based typed dependencies."""

    requires = {
        "dist_angle_to_goal": DistanceAngleMetrics,
        "simulation_state_container": SimulationStateContainer,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        pos_factor: float = 0.3,
        neg_factor: float = 0.5,
        _on_safe_dist_violation: bool = True,
        *args, **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._pos_factor = pos_factor
        self._neg_factor = neg_factor
        self._last_distance = None

    def reset(self) -> None:
        """Reset episode-local state."""
        self._last_distance = None

    def __call__(
        self,
        dist_angle_to_goal: DistanceAngleMetrics,
        simulation_state_container: SimulationStateContainer,
    ):
        distance = dist_angle_to_goal[0]
        if self._last_distance is not None:
            delta = self._last_distance - distance
            factor = self._pos_factor if delta > 0 else self._neg_factor
            self.add_reward(delta * factor)
        self._last_distance = distance
```

### RewardUnitFactory

Registry-based factory with a `@register("name")` decorator. No if/elif branching — the factory looks up the class by name and instantiates it.

```python
# Registration (at import time)
@RewardUnitFactory.register("collision")
class RewardCollision(RewardUnit):
    ...

# Instantiation (at runtime, by RewardFunction)
unit = RewardUnitFactory.instantiate("collision", reward_function=self, reward=-10.0)
```

## Built-in Reward Units

| Registration Name | Class | Category | Default Reward | Safety-Sensitive |
| --- | --- | --- | --- | --- |
| `goal_reached` | `RewardGoalReached` | Goal | +15.0 | Yes |
| `approach_goal` | `RewardApproachGoal` | Goal | factors: 0.3/0.5 | Yes |
| `safe_distance` | `RewardSafeDistance` | Safety | -0.15 | — (always) |
| `collision` | `RewardCollision` | Safety | -10.0 | — (always) |
| `ped_safe_distance` | `RewardPedSafeDistance` | Safety | -0.14 | — |
| `obs_safe_distance` | `RewardObsSafeDistance` | Safety | -0.14 | — |
| `ped_type_safety_distance` | `RewardPedTypeSafetyDistance` | Safety | -0.25 | Yes |
| `ped_type_collision` | `RewardPedTypeCollision` | Safety | -10.0 | — |
| `no_movement` | `RewardNoMovement` | Movement | -0.01 | Yes |
| `distance_travelled` | `RewardDistanceTravelled` | Movement | factor: 0.005 | Yes |
| `reverse_drive` | `RewardReverseDrive` | Movement | 0.01 | Yes |
| `abrupt_velocity_change` | `RewardAbruptVelocityChange` | Movement | factors | Yes |
| `root_velocity_difference` | `RewardRootVelocityDifference` | Movement | K=500 | No |
| `two_factor_velocity_difference` | `RewardTwoFactorVelocityDifference` | Movement | α=0.01, β=0.025 | — |

## Configuration

Reward functions are configured via a dictionary in the `AgentCfg`:

```yaml
reward:
  reward_function_dict:
    goal_reached:
      reward: 15.0
      _follow_subgoal: false

    collision:
      reward: -15.0
      bumper_zone: 0.05

    approach_goal:
      pos_factor: 0.4
      neg_factor: 0.5
      _goal_update_threshold: 0.25
      _on_safe_dist_violation: true

    safe_distance:
      reward: -0.2

    factored_reverse_drive:
      factor: 0.05
      threshold: 0.0
      _on_safe_dist_violation: true

    two_factor_velocity_difference:
      alpha: 0.005
      beta: 0.0
      _on_safe_dist_violation: true
  verbose: false
```

Parameters prefixed with `_` are typically internal flags:
- `_on_safe_dist_violation`: Whether this unit is evaluated during safety violations
- `_follow_subgoal`: Whether to check subgoal instead of main goal
- `_goal_update_threshold`: Distance threshold for goal progress updates

## How It Works

### Reward Calculation Flow

1. **Initialization**: `RewardFunction.__init__()` iterates over `function_dict`, looks up each name in `RewardUnitFactory.registry`, and instantiates units with their parameters
2. **Safety categorization**: `_categorize_units_by_safety_sensitivity()` partitions units into two lists based on `_on_safe_dist_violation`
3. **Per-step execution**: `calculate_reward(obs, sim_state)` does:
   - Reset `RewardState` (zero reward, empty info)
   - Determine eligible units based on safety violation status
   - Execute units sequentially or in parallel
   - Return `(total_reward, info_dict)`
4. **Validation**: On the first call, `validate_reward_units()` checks that all `requires` keys are present in the observation dict. After validation passes, it's permanently disabled.

### Safety Categorization

```
RewardUnit._on_safe_dist_violation = True
  → Added to _safe_dist_violation_units
  → Only evaluated when safety.violation == True

RewardUnit._on_safe_dist_violation = False
  → Added to _non_safe_dist_violation_units
  → Evaluated on every step
```

This prevents rewarding goal-approaching behavior while the robot is dangerously close to obstacles.

## Adding a New Reward Unit

1. **Create the unit** in `reward_units.py` (or a new file):
   ```python
   @RewardUnitFactory.register("my_unit")
   class MyUnit(RewardUnit):
       requires = {"front_laser": LidarRanges}

       @check_params
       def __init__(self, reward_function, reward=-1.0, _on_safe_dist_violation=True, *args, **kwargs):
           super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
           self._reward = reward

       def __call__(self, front_laser: LidarRanges):
           if np.min(front_laser) < 0.3:
               self.add_reward(self._reward)
   ```

2. **Add to `__all__`** in `reward_units.py` (for auto-import).

3. **Add defaults** to `constants.py` `DEFAULTS` class (optional but recommended):
   ```python
   class DEFAULTS:
       class MY_UNIT:
           REWARD: float = -1.0
           _ON_SAFE_DIST_VIOLATION: bool = True
   ```

4. **Use in config**:
   ```yaml
   reward:
     reward_function_dict:
       my_unit:
         reward: -2.0
   ```
