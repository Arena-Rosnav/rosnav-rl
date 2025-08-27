# Logger System Integration Guide

## Overview

The flexible logger system provides automatic backend detection and seamless integration with ROS2, ROS1, and Python environments. It works together with the unified error logging system to provide clean, organized logging output.

## 🔧 Quick Setup

### Basic Configuration (Auto-Detection)
```python
from rosnav_rl.utils.logger import configure_logger

# Automatically detects and configures the best available logger
logger = configure_logger("auto", name="my_component")
logger.info("System initialized")
```

### ROS2 Node Integration
```python
import rclpy
from rclpy.node import Node
from rosnav_rl.utils.logger import configure_logger

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        
        # Configure logger with ROS2 node
        self.logger = configure_logger(
            logger_type="ros2",
            node=self,  # Pass the ROS2 node
            name="my_system"
        )
        
        self.logger.info("ROS2 node initialized with integrated logging")
```

### Manual Backend Selection
```python
from rosnav_rl.utils.logger import configure_logger

# Specific backend selection
console_logger = configure_logger("console", colored=True)
python_logger = configure_logger("python", level="DEBUG")
ros1_logger = configure_logger("ros1", name="ros1_component")
silent_logger = configure_logger("silent")  # For testing
```

## 🏗️ Architecture Integration

### Component Base Classes

The logger system is designed to work seamlessly with your existing components:

```python
from rosnav_rl.utils.logger import get_logger
from rosnav_rl.utils.error_mixins import ObservationSpaceErrorMixin

class MyObservationSpace(BaseObservationSpace, ObservationSpaceErrorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Logger is automatically configured and available
        # Error reporting goes through unified system
        
    def encode_observation(self, **kwargs):
        try:
            # Your encoding logic
            return encoded_data
        except Exception as e:
            # This automatically uses the configured logger backend
            self._report_error(f"Encoding failed: {e}", error_type=type(e).__name__)
            return self._create_null_observation()
```

### System Managers

```python
from rosnav_rl.utils.error_mixins import flush_errors_decorator
from rosnav_rl.utils.logger import configure_logger

class ObservationSpaceManager:
    def __init__(self, ros_node=None):
        # Configure logger based on environment
        if ros_node:
            self.logger = configure_logger("ros2", node=ros_node)
        else:
            self.logger = configure_logger("auto")
    
    @flush_errors_decorator  # Automatically flushes errors with configured logger
    def encode_observation(self, observations):
        self.logger.debug("Starting observation encoding")
        # All your processing logic
        self.logger.info("Observation encoding completed")
```

## 🎯 Usage Patterns

### Environment-Aware Initialization

```python
class RosNavSystem:
    def __init__(self, ros_node=None, logger_type="auto", **logger_kwargs):
        # Flexible logger configuration
        if logger_type == "auto":
            if ros_node:
                self.logger = configure_logger("ros2", node=ros_node, **logger_kwargs)
            else:
                self.logger = configure_logger("auto", **logger_kwargs)
        else:
            self.logger = configure_logger(logger_type, node=ros_node, **logger_kwargs)
        
        self.logger.info(f"RosNav system initialized with {self.logger.__class__.__name__}")
```

### Training vs Production Logging

```python
class TrainingEnvironment:
    def __init__(self, debug_mode=False):
        if debug_mode:
            # Detailed console logging for development
            self.logger = configure_logger(
                "console", 
                colored=True, 
                name="training_debug"
            )
        else:
            # Quiet logging for production training
            self.logger = configure_logger(
                "python",
                level="WARNING",  # Only warnings and errors
                name="training_prod"
            )
```

### Multi-Component Systems

```python
class NavigationSystem:
    def __init__(self, ros_node=None):
        # Global system logger configuration
        self.system_logger = configure_logger(
            "ros2" if ros_node else "auto",
            node=ros_node,
            name="navigation_system"
        )
        
        # Individual components automatically use the configured logger
        self.observation_manager = ObservationSpaceManager()
        self.reward_function = RewardFunction()
        self.path_planner = PathPlanner()
        
        self.system_logger.info("Navigation system fully initialized")
```

## 🚀 Migration from Existing Code

### Before (Direct ROS Logging)
```python
import rospy

class OldComponent:
    def __init__(self):
        # Hard-coded ROS dependency
        pass
    
    def process(self):
        try:
            # processing logic
            pass
        except Exception as e:
            rospy.logerr(f"Processing failed: {e}")  # Direct ROS dependency
```

### After (Flexible Logging)
```python
from rosnav_rl.utils.logger import get_logger
from rosnav_rl.utils.error_mixins import flush_errors_decorator

class NewComponent:
    def __init__(self):
        # No hard dependencies - logger auto-configured
        self.logger = get_logger()
    
    @flush_errors_decorator  # Unified error handling
    def process(self):
        try:
            # processing logic  
            self.logger.info("Processing completed successfully")
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")  # Works with any backend
```

## 🔧 Advanced Configuration

### Custom Logger Backend

```python
from rosnav_rl.utils.logger import BaseLogger, set_logger

class CustomLogger(BaseLogger):
    def __init__(self, custom_backend):
        super().__init__("custom")
        self.backend = custom_backend
    
    def info(self, message: str, **kwargs):
        self.backend.log_info(message)
    
    # Implement other methods...

# Use custom logger
custom_logger = CustomLogger(my_backend)
set_logger(custom_logger)
```

### Environment-Specific Configuration

```python
import os
from rosnav_rl.utils.logger import configure_logger

# Environment-based configuration
env_type = os.getenv('ROSNAV_ENV', 'auto')

if env_type == 'ros2':
    logger = configure_logger('ros2', node=ros_node)
elif env_type == 'development':
    logger = configure_logger('console', colored=True)
elif env_type == 'production':
    logger = configure_logger('python', level='WARNING')
elif env_type == 'testing':
    logger = configure_logger('silent')
else:
    logger = configure_logger('auto')
```

## 🧪 Testing Integration

### Test-Friendly Configuration

```python
import unittest
from rosnav_rl.utils.logger import configure_logger

class TestMyComponent(unittest.TestCase):
    def setUp(self):
        # Use silent logger for tests
        self.logger = configure_logger('silent', name='test')
    
    def test_component_behavior(self):
        # Your tests run without log noise
        pass
```

### Debug Mode Testing

```python
class DebugTestCase(unittest.TestCase):
    def setUp(self):
        # Enable console logging for debugging
        debug = os.getenv('DEBUG_TESTS', 'false').lower() == 'true'
        logger_type = 'console' if debug else 'silent'
        self.logger = configure_logger(logger_type, colored=debug)
```

## 📊 Performance Considerations

### Logging Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: General system information  
- **WARNING**: Something unexpected but not critical
- **ERROR**: Serious problems that need attention
- **CRITICAL**: System failures

### Production Optimization
```python
# Production configuration - minimal logging overhead
production_logger = configure_logger(
    'python',
    level='ERROR',  # Only log errors and critical issues
    name='production_system'
)
```

### Development Configuration  
```python
# Development configuration - detailed logging
dev_logger = configure_logger(
    'console', 
    colored=True,  # Easy visual parsing
    name='development_system'
)
```

## 🛠️ Integration Checklist

- [ ] Configure logger in system initialization
- [ ] Update components to use unified error reporting mixins  
- [ ] Add flush decorators to main processing methods
- [ ] Test with different logger backends
- [ ] Configure appropriate logging levels for environment
- [ ] Remove hard-coded ROS logging dependencies
- [ ] Verify error collection and organized reporting
- [ ] Test ROS2 node integration if applicable

## 📚 Best Practices

1. **Configure Once**: Set up the logger at system initialization
2. **Use Mixins**: Leverage error reporting mixins for automatic integration
3. **Environment Awareness**: Use auto-detection or environment variables
4. **Appropriate Levels**: Choose the right log level for each message
5. **Flush Regularly**: Use decorators on main processing loops
6. **Test Coverage**: Test with different logger backends
7. **Production Ready**: Use appropriate log levels for production

## 🐛 Troubleshooting

### Logger Not Working
- Check if logger is properly configured: `get_logger().enabled`
- Verify environment has required dependencies
- Try explicit backend: `configure_logger('console')`

### ROS2 Integration Issues  
- Ensure node has `get_logger()` method
- Check node is properly initialized
- Verify ROS2 environment is active

### No Log Output
- Check logging level configuration
- Verify logger is not set to 'silent'
- Check if unified error reporting is flushing correctly

The flexible logger system provides seamless integration across all environments while maintaining clean, organized output through the unified error logging system.
