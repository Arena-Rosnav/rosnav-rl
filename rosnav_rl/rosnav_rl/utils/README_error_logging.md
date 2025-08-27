# Unified Error Logging System for RosNav-RL

## Overview

The unified error logging system prevents log spam by collecting errors from all components (observation spaces, generators, collectors, reward units) and presenting them in organized summaries once per iteration. This provides better visibility into system-wide issues while maintaining clean, readable logs.

## 🎯 Key Benefits

- **Prevents Log Spam**: No more individual error messages for every component failure
- **Organized Reporting**: Errors grouped by severity, component type, and frequency
- **Thread-Safe**: Safe for parallel processing environments
- **Performance Optimized**: Minimal overhead with intelligent deduplication
- **Comprehensive Coverage**: Integrates across all RosNav-RL components

## 📁 File Structure

```
rosnav_rl/utils/
├── error_logging.py          # Core error collection and logging system
├── error_mixins.py           # Mixin classes for easy integration
└── README_error_logging.md   # This documentation
```

## 🔧 Core Components

### 1. Error Collector (`error_logging.py`)

The central error collection system with thread-safe operation:

```python
from rosnav_rl.utils.error_logging import (
    ErrorSeverity,
    ComponentType,
    get_error_collector,
    collect_error,
    flush_and_log_errors
)

# Collect an error
collect_error(
    component_type=ComponentType.OBSERVATION_SPACE,
    component_name="laser_scan_space",
    severity=ErrorSeverity.ERROR,
    message="Missing sensor data. Using null observation.",
    error_type="KeyError"
)

# Flush and log all collected errors (typically done once per iteration)
flush_and_log_errors()
```

### 2. Component Mixins (`error_mixins.py`)

Mixin classes that components can inherit from for easy error reporting:

```python
from rosnav_rl.utils.error_mixins import (
    ObservationSpaceErrorMixin,
    GeneratorErrorMixin,
    RewardUnitErrorMixin,
    CollectorErrorMixin,
    flush_errors_decorator
)

class MyObservationSpace(BaseObservationSpace, ObservationSpaceErrorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Error reporting is automatically set up
    
    def encode_observation(self, **kwargs):
        try:
            # Your encoding logic
            pass
        except ValueError as e:
            self._report_error(f"Encoding failed: {e}", error_type="ValueError")
            return self._create_null_observation()
```

### 3. Flush Decorator

Automatically flush errors at the end of major operations:

```python
@flush_errors_decorator
def process_step(self, observations):
    # All your processing logic
    # Errors are automatically flushed and logged at the end
    pass
```

## 🚀 Integration Guide

### Observation Spaces

Update your observation spaces to inherit from `ObservationSpaceErrorMixin`:

```python
from rosnav_rl.utils.error_mixins import ObservationSpaceErrorMixin

class LaserScanSpace(BaseObservationSpace, ObservationSpaceErrorMixin):
    def safe_encode_observation(self, *args, **kwargs):
        try:
            result = self.encode_observation(*args, **kwargs)
            if result is not None:
                return result
            self._report_warning("encode_observation() returned None")
        except KeyError as e:
            self._report_error(f"Missing data key {e}", error_type="KeyError")
        except Exception as e:
            self._report_error(f"Encoding failed: {e}", error_type=type(e).__name__)
        
        return self._create_null_observation()
```

### Generators

Update generators to use unified error reporting:

```python
from rosnav_rl.utils.error_mixins import GeneratorErrorMixin

class MyGenerator(Generator, GeneratorErrorMixin):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._set_error_reporting_info(
            component_type=self._component_type,
            component_name=name
        )
    
    def get_observation(self, obs_dict, **kwargs):
        try:
            return self._generate(**{k: obs_dict[k] for k in self.required_keys})
        except KeyError as e:
            self._report_error(f"Missing dependency: {e}", error_type="KeyError")
            return None
        except TypeError as e:
            self._report_error(f"Type mismatch: {e}", error_type="TypeError")
            return None
```

### Reward Units

Update reward units to use unified error reporting:

```python
from rosnav_rl.utils.error_mixins import RewardUnitErrorMixin

class MyRewardUnit(RewardUnit, RewardUnitErrorMixin):
    def __init__(self, reward_function, **kwargs):
        super().__init__(reward_function, **kwargs)
        self._set_error_reporting_info(
            component_type=self._component_type,
            component_name=self.__class__.__name__
        )
    
    def add_reward(self, value: float):
        if not isinstance(value, (int, float)):
            self._report_error(f"Invalid reward type: {type(value)}", error_type="TypeError")
            return
        
        if value != value:  # NaN check
            self._report_error("Reward cannot be NaN", error_type="ValueError")
            return
        
        # Add the reward normally
        self._reward_function.add_reward(value, called_by=self.__class__.__name__)
```

### System Managers

Use the flush decorator on main processing methods:

```python
from rosnav_rl.utils.error_mixins import flush_errors_decorator

class ObservationSpaceManager:
    @flush_errors_decorator
    def encode_observation(self, observations):
        # All observation encoding logic
        # Errors automatically flushed at the end
        pass

class RewardFunction:
    @flush_errors_decorator
    def get_reward(self, obs_dict, simulation_state, **kwargs):
        # All reward calculation logic
        # Errors automatically flushed at the end
        pass
```

## 📊 Error Report Format

The unified system generates organized error reports like this:

```
🚨 RosNav-RL System Status Report (15 issues)
================================================================================

❌ ERROR (12 issues)
------------------------------------------------------------
  📦 ObservationSpace (8 issues)
    • laser_scan_space: Missing observation data key 'sensor_data' (×3)
    • camera_space: Error during encoding: Invalid image format (×2)
    • pose_space: Unexpected error during encoding: Division by zero (×3)

  📦 Generator (4 issues)
    • goal_generator: Missing required dependency: 'robot_pose' (×2)
    • safety_generator: Type mismatch in '_generate' method: expected float, got str (×2)

⚠️ WARNING (3 issues)
------------------------------------------------------------
  📦 RewardUnit (3 issues)
    • approach_goal: Has large reward magnitude: 150.0 (×3)

💡 Next Steps:
  - Check observation data sources and configurations
  - Verify all required dependencies are available
  - Review component initialization parameters
  - Consider enabling debug mode for detailed analysis
```

## 🔄 Migration from Old System

### Before (Individual Error Messages)
```python
# Old approach - creates log spam
def safe_encode_observation(self, *args, **kwargs):
    try:
        return self.encode_observation(*args, **kwargs)
    except Exception as e:
        ros_warn(f"[{self.name}] Error: {e}. Using null observation.")
        return self._create_null_observation()
```

### After (Unified Error Collection)
```python
# New approach - errors collected and reported once per iteration
def safe_encode_observation(self, *args, **kwargs):
    try:
        return self.encode_observation(*args, **kwargs)
    except Exception as e:
        self._report_error(f"Error: {e}. Using null observation.", error_type=type(e).__name__)
        return self._create_null_observation()
```

## 🏃‍♂️ Running the Examples

### Basic Demo
```bash
cd /home/le/arena4_ws_exp/src/planners/rosnav_rl/rosnav_rl
python scripts/demo_unified_error_logging.py
```

### Integration Example
```bash
cd /home/le/arena4_ws_exp/src/planners/rosnav_rl/rosnav_rl
python scripts/integration_example.py
```

## 🎛️ Configuration

### Error Severity Levels
- `ErrorSeverity.INFO`: Informational messages
- `ErrorSeverity.WARNING`: Warning conditions
- `ErrorSeverity.ERROR`: Error conditions
- `ErrorSeverity.CRITICAL`: Critical system failures

### Component Types
- `ComponentType.OBSERVATION_SPACE`: Observation space components
- `ComponentType.GENERATOR`: Data generator components
- `ComponentType.COLLECTOR`: Data collector components
- `ComponentType.REWARD_UNIT`: Reward unit components
- `ComponentType.REWARD_FUNCTION`: Reward function components
- `ComponentType.OTHER`: Other system components

### Disabling Error Collection
```python
from rosnav_rl.utils.error_logging import get_error_collector

# Temporarily disable error collection
collector = get_error_collector()
collector.disable()

# Re-enable error collection
collector.enable()

# Clear all collected errors
collector.clear()
```

## 🧪 Testing

The system includes comprehensive examples demonstrating:
- Error collection across different component types
- Automatic error deduplication
- Organized error reporting
- Integration with existing RosNav-RL components

## 📈 Performance Impact

- **Minimal Overhead**: Error collection uses efficient data structures
- **Thread-Safe**: Concurrent access is handled safely
- **Memory Efficient**: Automatic deduplication prevents memory bloat
- **Smart Flushing**: Only logs when there are actual errors to report

## 🔧 Advanced Usage

### Custom Error Reporting
```python
from rosnav_rl.utils.error_logging import collect_error, ComponentType, ErrorSeverity

# Report a custom error
collect_error(
    component_type=ComponentType.OTHER,
    component_name="my_custom_component",
    severity=ErrorSeverity.WARNING,
    message="Custom warning message",
    error_type="CustomWarning"
)
```

### Manual Error Flushing
```python
from rosnav_rl.utils.error_logging import flush_and_log_errors

# Manually flush errors at any time
flush_and_log_errors()
```

## 🎯 Best Practices

1. **Use Appropriate Severity Levels**: Choose the right severity for each error type
2. **Provide Context**: Include relevant details in error messages
3. **Use Error Types**: Specify the exception type for better categorization
4. **Flush Regularly**: Use the decorator on main processing loops
5. **Test Error Paths**: Ensure your error reporting works as expected

## 🐛 Troubleshooting

### No Error Reports Appearing
- Check if error collection is enabled: `get_error_collector().enabled`
- Ensure you're calling `flush_and_log_errors()` or using the decorator
- Verify components are properly inheriting from error mixins

### Too Many Duplicate Errors
- The system automatically deduplicates identical errors
- If you're still seeing too many, consider using higher-level error categories

### Performance Issues
- Error collection has minimal overhead
- If concerned, you can disable collection in production: `get_error_collector().disable()`
