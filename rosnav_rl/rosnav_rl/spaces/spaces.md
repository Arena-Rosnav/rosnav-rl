# Observation Spaces in Reinforcement Learning

Observation spaces define how reinforcement learning agents perceive their environment through environmental information. This guide explains the core components and implementation details of our observation space system.

## Architecture Overview

### ObservationManager
The ObservationManager coordinates data collection from various sources:
- Manages observation collector units (ObservationUnits)
- Aggregates data into a central observation dictionary
- Handles ROS topic subscriptions and message processing
- Provides validation and waiting mechanisms for observations

### ObservationSpaceManager
The ObservationSpaceManager handles the encoding and combination of observation spaces:
- Manages multiple observation space instances
- Creates combined observation spaces
- Provides encoding methods for observations
- Maintains configuration and initialization parameters

### BaseObservationSpace
The foundation class for all observation spaces provides:
- Abstract interface for defining observation spaces
- Built-in normalization capabilities
- Type checking and validation
- Required implementation of:
  - `get_gym_space()`
  - `encode_observation()`

## Implementation Guide

### Creating a New Observation Space

```python

@SpaceFactory.register("laser")
class CustomObservationSpace(BaseObservationSpace):
    name = "CUSTOM_SPACE"
    required_observation_units = []

    def get_gym_space(self) -> spaces.Space:
    return spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(10,),
        dtype=np.float32
    )

    @BaseObservationSpace.check_dtype
    @BaseObservationSpace.apply_normalization
    def encode_observation(self, observation: dict, **kwargs) -> np.ndarray:
        # Transform observation data
        processed_data = np.array([...], dtype=np.float32)
        return processed_data
```

Example implementation of a laser scan observation space:
```python
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory

@SpaceFactory.register("laser")
class LaserScanSpace(BaseObservationSpace):
    """
    Represents the observation space for laser scan data.

    Args:
        laser_num_beams (int): The number of laser beams.
        laser_max_range (float): The maximum range of the laser.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _num_beams (int): The number of laser beams.
        _max_range (float): The maximum range of the laser.
    """

    name = "LASER"
    required_observation_units = [LaserCollector]

    def __init__(
        self, laser_num_beams: int, laser_max_range: float, *args, **kwargs
    ) -> None:
        self._num_beams = laser_num_beams
        self._max_range = laser_max_range
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for laser scan data.

        Returns:
            spaces.Space: The Gym observation space.
        """
        return spaces.Box(
            low=0,
            high=self._max_range,
            shape=(1, self._num_beams),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> LaserCollector.data_class:
        """
        Encodes the laser scan observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            ndarray: The encoded laser scan observation.
        """
        return observation[LaserCollector.name][np.newaxis, :]
```
### Usage Example

```python
from observation_space_manager import ObservationSpaceManager

# Initialize observation space manager with required spaces and parameters
obs_space_manager = ObservationSpaceManager(
    space_list=[LaserScanSpace],
    space_kwargs={"laser_num_beams": 360, "laser_max_range": 20, "normalize": True}
)

# Encode observations
encoded_obs = obs_space_manager.encode_observation(observation_dict)

# LaserCollector.name in observation_dict == True -> Laser data needs to be available
```

## Implementation Steps

1. **Data Collection**
   - Ensure simulation provides required data
   - Create ObservationUnit for data collection
   - Add unit to ObservationManager

2. **Space Definition**
   - Create new class inheriting from BaseObservationSpace
   - Define gym space in get_gym_space()
   - Implement encode_observation()

3. **Integration**
   - Add space to ObservationSpaceManager
   - Configure normalization if needed
   - Update model configuration

## Best Practices

- Keep observation spaces modular and focused
- Implement proper data validation
- Use appropriate normalization methods
- Handle edge cases gracefully
- Document space requirements and assumptions
- Test with various input conditions