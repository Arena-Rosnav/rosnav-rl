# Perception Package Structure

## Overview

The perception package has been reorganized into specialized sub-packages:

```
perception/
├── __init__.py                    # Main package imports  
├── laser/                        # Laser processing spaces
│   ├── __init__.py
│   ├── reliable_laser_space.py   # Production laser spaces
│   ├── legacy_laser_spaces.py    # Legacy laser spaces
│   └── production_examples.py    # Usage examples
└── vision/                       # Vision processing spaces  
    ├── __init__.py
    └── legacy_vision_spaces.py   # Legacy vision spaces
```

## Production Spaces

### ReliableLaserSpace
- **Purpose**: Single-sensor laser processing with proven reliability features
- **Features**: Range validation, NaN handling, median filtering, beam reduction
- **Use Cases**: Standard robot navigation with single LIDAR

### MultiLaserFusionSpace ⭐ NEW
- **Purpose**: Multi-sensor laser fusion for high-reliability applications
- **Features**: 
  - Confidence-based sensor weighting
  - Angular overlap handling  
  - Robust outlier detection
  - Graceful sensor failure handling
  - Multiple fusion strategies (weighted_average, confidence_max, min_distance)
- **Use Cases**: 
  - Dual/quad LIDAR setups
  - Warehouse automation
  - Safety-critical navigation
  - Redundant sensor systems

## Legacy Spaces

### Laser Spaces
- `LaserScanSpace`: Original laser scan processing
- `ReducedLaserScanSpace`: Beam reduction for performance

### Vision Spaces  
- `RGBDSpace`: RGB-D camera processing

## Production Usage

```python
from rosnav_rl.spaces.observation_space.spaces_new.perception import MultiLaserFusionSpace

# Dual laser setup
config = {
    "laser_configs": [
        {"topic": "/scan_front", "weight": 1.0, "angle_offset": 0.0},
        {"topic": "/scan_rear", "weight": 0.9, "angle_offset": 3.14159}
    ],
    "fusion_method": "weighted_average",
    "confidence_threshold": 0.7
}

fusion_space = MultiLaserFusionSpace(**config)
```

## Key Benefits

1. **Reliability**: Proven algorithms, robust error handling
2. **Performance**: Optimized for real-time applications  
3. **Safety**: Conservative fusion for collision avoidance
4. **Flexibility**: Multiple fusion strategies for different use cases
5. **Monitoring**: Built-in sensor health tracking and validation
