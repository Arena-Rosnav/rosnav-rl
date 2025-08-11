# Observations Package Structure

The observations package has been reorganized into a clean, modular architecture:

## 📁 **Directory Structure**

```
observations/
├── core/                    # Core orchestration components
│   ├── manager.py          # Main ObservationManager (was observation_manager.py)
│   └── pipeline.py         # Observation pipeline coordination
├── data_sources/           # Data source implementations  
│   ├── base.py            # Base interfaces (Collector, Generator, DataSource)
│   ├── collectors.py      # Collector implementations
│   └── generators.py      # Generator implementations
├── strategies/            # Strategy pattern implementations
│   ├── collection.py      # Collection strategy
│   ├── generator.py       # Generator execution strategy
│   ├── subscription.py    # ROS subscription management
│   └── waiting.py         # Waiting and polling strategy
├── factory/               # Factory and dependency resolution
│   ├── factory.py         # ObservationFactory for creating data sources
│   └── resolver.py        # Dependency resolution logic
├── utils/                 # Utilities and helpers
│   ├── constants.py       # Constants
│   ├── static.py          # Static utilities
│   ├── types.py           # Type annotations (was type_annotations.py)
│   ├── pose.py            # Pose utilities
│   └── semantic.py        # Semantic utilities
├── examples/              # Examples and documentation
│   └── usage.py           # Usage examples (was example_usage.py)
└── observations.yaml      # Configuration file
```

## 🎯 **Benefits of the New Structure**

1. **Clear Separation of Concerns**: Each module has a single, focused responsibility
2. **Logical Grouping**: Related files are organized together
3. **Easy Navigation**: Intuitive structure makes finding code easier
4. **Maintainability**: Modular design makes updates and testing simpler
5. **Backward Compatibility**: Main public API unchanged

## 📚 **Import Examples**

```python
# Main API (unchanged)
from observations import ObservationManager

# Core components
from observations.core import ObservationManager, ObservationPipeline

# Data sources
from observations.data_sources import Collector, Generator, DataSource

# Strategies
from observations.strategies import CollectionStrategy, GeneratorStrategy

# Factory and utilities
from observations.factory import ObservationFactory, DependencyResolver
```

## 🔧 **Migration Notes**

- **File Renames**: 
  - `observation_manager.py` → `core/manager.py`
  - `type_annotations.py` → `utils/types.py` 
  - `example_usage.py` → `examples/usage.py`
- **Import Updates**: All internal imports updated to new structure
- **Public API**: Unchanged - existing code continues to work
