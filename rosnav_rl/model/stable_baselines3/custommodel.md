# Custom Stable-Baselines3 Policies in Rosnav-RL

The `rosnav-rl` framework provides a flexible interface for implementing custom reinforcement learning policies using Stable-Baselines3. This guide explains how to create your own policy implementations using the `StableBaselinesPolicyDescription` base class.

## Core Concepts

The `StableBaselinesPolicyDescription` class serves as an abstract base class for defining custom policies. It provides a standardized interface that integrates seamlessly with Stable-Baselines3 algorithms while adding specific functionality for robotic navigation tasks.

## Implementing a Custom Policy

To create a custom policy, extend the `StableBaselinesPolicyDescription` class and implement all required abstract properties:

```python
@AgentFactory.register("CustomAgent")
class CustomAgent(StableBaselinesPolicyDescription):
    algorithm_class = PPO  # or any other SB3 algorithm
    observation_spaces = [
        spaces.StackedLaserMapSpace,
        spaces.DistAngleToSubgoalSpace,
        spaces.LastActionSpace,
    ]
    observation_space_kwargs = {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "normalize": True,
    }
    features_extractor_class = CustomFeatureExtractor
    features_extractor_kwargs = {
        "features_dim": 256,
    }
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU

```


## Required Properties

### Algorithm Selection
Specify which Stable-Baselines3 algorithm to use:

```python
  algorithm_class: Type[BaseAlgorithm] = PPO
```


### Observation Spaces
Define which observation spaces your agent requires:

```python
  observation_spaces = [
    spaces.StackedLaserMapSpace,
    spaces.DistAngleToSubgoalSpace,
  # Add more observation spaces as needed
  ]
```


### Network Architecture
Configure the policy and value function networks:
```python
  net_arch = dict(
    pi=[64, 64], # Policy network architecture
    vf=[64, 64] # Value function network architecture
  )
```

### Feature Extraction
Specify how raw observations should be processed:
```python
  features_extractor_class = CustomFeatureExtractor
  features_extractor_kwargs = {
    "features_dim": 256,
    # Additional extractor parameters
  }
```


## Optional Properties

### Observation Space Configuration
Customize observation space parameters:

```python
  observation_space_kwargs = {
    "roi_in_m": 20,
    "feature_map_size": 80,
    "normalize": True,
  }
```


### Stack Size
Define how many observations should be stacked:

```python
  @property
  def stack_size(self) -> int:
  return 4 # Default is 1
```

## Advanced Example

Here's a more sophisticated example using RecurrentPPO:

```python
@AgentFactory.register("RecurrentNavigationAgent")
class RecurrentNavigationAgent(StableBaselinesPolicyDescription):
  algorithm_class = RecurrentPPO
  observation_spaces = [
    spaces.StackedLaserMapSpace,
    spaces.PedestrianVelXSpace,
    spaces.PedestrianVelYSpace,
    spaces.DistAngleToSubgoalSpace,
  ]
  observation_space_kwargs = {
    "roi_in_m": 20,
    "feature_map_size": 80,
    "laser_stack_size": 10,
    "normalize": True,
    "goal_max_dist": 10,
  }
  features_extractor_class = CustomRecurrentExtractor
  features_extractor_kwargs = {
    "features_dim": 512,
    "lstm_hidden_size": 128,
    "n_lstm_layers": 2,
  }
  net_arch = dict(pi=[256, 128], vf=[256, 64])
  activation_fn = nn.GELU
```


## Integration with Arena-Rosnav

Once implemented, your custom policy can be used in the Arena-Rosnav framework by:

1. Registering it with the `@AgentFactory.register` decorator
2. Specifying it in your training configuration
3. The framework will automatically handle the integration with Stable-Baselines3

Your policy will then be ready for training and evaluation in various navigation scenarios within the Arena-Rosnav environment.
