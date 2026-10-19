# DreamerV3 Framework for RosNav

A PyTorch-based implementation of DreamerV3 for robot navigation tasks integrated with Rosnav-RL.

## Table of Contents

- [Introduction](#introduction)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Custom Environments](#custom-environments)
- [Observation Spaces](#observation-spaces)
- [Training](#training)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Introduction

This module implements DreamerV3, a state-of-the-art model-based reinforcement learning agent. It leverages world models to learn environment dynamics and make decisions in partially observable environments. This implementation is specifically adapted for robotic navigation tasks in ROS environments.

Key features:
- World model training with latent dynamics prediction
- Actor-critic architecture for policy optimization
- Modular observation space handling
- Parallel environment execution
- Checkpoint management
- ROS integration

## Configuration

The DreamerV3 agent uses a comprehensive configuration system divided into multiple sections:

```python
from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg

config = DreamerV3Cfg(
    general=GeneralConfig(
        logdir="./logdir",
        seed=0,
        steps=1_000_000,
        eval_every=10_000,
        eval_episode_num=5,
        log_every=1_000,
        prefill=5_000,
        envs=4,  # Number of parallel environments
    ),
    world_model=WorldModelConfig(
        deter_size=4096,
        stoch_size=32,
        stoch_discrete=32,
        initial_std=0.0,
        hidden_size=1024,
        layers=2,
    ),
    actor=ActorConfig(
        layers=4,
        units=1024,
        dist="auto",  # "auto", "onehot", "normal", "binary"
        min_std=0.1,
    ),
    critic=CriticConfig(
        layers=4,
        units=1024,
    ),
    training=TrainingConfig(
        batch_size=16,
        batch_length=64,
        train_ratio=512,
        learning_rate=1e-4,
        grad_clip=1000,
        discount=0.997,
        horizon=15,  # Imagination horizon
        actor_entropy=1e-4,
        slow_target=True,
        slow_target_fraction=1,
        slow_target_update=100,
    ),
)
```

### Adjusting Neural Network Architecture

You can modify the neural network architecture by changing the configuration parameters:

1. **World Model**:
   - `deter_size`: Size of deterministic state
   - `stoch_size`: Size of stochastic state
   - `stoch_discrete`: Number of categorical variables in stochastic state
   - `hidden_size`: Hidden layer size for RSSM
   - `layers`: Number of layers in encoder and decoder networks

2. **Actor Network**:
   - `layers`: Number of layers
   - `units`: Units per layer
   - `dist`: Distribution type for action sampling

3. **Critic Network**:
   - `layers`: Number of layers
   - `units`: Units per layer

## Architecture

The DreamerV3 architecture consists of:

### World Model Components
- **RSSM (Recurrent State-Space Model)**: Learns environment dynamics
- **Encoder**: Encodes observations into latent representations
- **Decoder**: Reconstructs observations from latent states
- **Reward Predictor**: Predicts rewards from latent states

### Agent Components
- **Actor**: Policy network for action selection
- **Critic**: Value network for state evaluation
- **Exploration**: Mechanisms for efficient exploration

## Custom Environments

DreamerV3 requires environments following the Gym interface. For ROS integration, several wrappers are provided in the `envs/wrappers.py` file:

```python
from rosnav_rl.model.dreamerv3.envs.wrappers import (
    WoTruncatedFlag,
    TimeLimit,
    SelectAction,
    UUID,
    ResetWoInfo,
    ChannelFirsttoLast
)
```

### Essential Wrappers

1. **WoTruncatedFlag**: Removes truncated flag for compatibility
2. **TimeLimit**: Sets maximum episode length
3. **SelectAction**: Selects action from action dictionary
4. **UUID**: Adds unique identifier to each environment
5. **ResetWoInfo**: Handles environment reset without requiring info
6. **ChannelFirsttoLast**: Converts observations format as needed

### Environment Setup Example

```python
from functools import partial
from rosnav_rl.model.dreamerv3 import (
    WoTruncatedFlag,
    TimeLimit,
    SelectAction,
    UUID,
    ResetWoInfo,
    ChannelFirsttoLast,
    Parallel
)

# Define environment creation function
def make_env(mode, idx):
    # Create your ROS environment here
    env = YourROSEnvironment()
    
    # Apply wrappers
    env = WoTruncatedFlag(env)
    env = TimeLimit(env, duration=max_steps)
    env = SelectAction(env, key="action")
    env = UUID(env)
    env = ResetWoInfo(env)
    env = ChannelFirsttoLast(env)
    return env

# Create parallel environments
train_env_fncs = [lambda: make_env("train", i) for i in range(n_envs)]
train_envs = [Parallel(init_fnc, "process") for init_fnc in train_env_fncs]
```

## Observation Spaces

DreamerV3 supports various observation types through the RosNav-RL observation space system.

### Configuring Observation Spaces

In your agent configuration, specify which observation spaces to use:

```python
from rosnav_rl.spaces import observation_space as spaces

class DreamerV3Model(RL_Model):
    # Define which observation spaces the model uses
    @property
    def observation_space_list(self) -> List["BaseObservationSpace"]:
        return [
            spaces.StackedLaserMapSpace,
            spaces.PedestrianVelXSpace,
            spaces.PedestrianVelYSpace,
            spaces.DistAngleToSubgoalSpace,
            spaces.LastActionSpace,
            spaces.IsFirstStepSpace,
            spaces.IsTerminalStepSpace,
        ]
    
    # Define parameters for observation spaces
    @property
    def observation_space_kwargs(self) -> Dict[str, Any]:
        return {
            "roi_in_m": 40,
            "feature_map_size": 80,
            "laser_stack_size": 10,
            "normalize": True,
            "goal_max_dist": 10,
        }
```

## Training

### Training Loop

The DreamerV3 training loop alternates between:

1. **Evaluation**: Assessing agent performance
2. **Training**: 
   - Environment interaction
   - World model updates
   - Actor-critic updates
   - Exploration policy updates

### Customizing Training

To customize training behavior, modify the training configuration:

```python
training_config = TrainingConfig(
    # Environment interaction
    batch_size=16,          # Batch size for training
    batch_length=64,        # Sequence length for recurrent processing
    train_ratio=512,        # Ratio of model updates to environment steps
    
    # Optimization parameters
    learning_rate=1e-4,     # Learning rate for all components
    grad_clip=1000,         # Gradient clipping value
    
    # Actor-critic parameters
    discount=0.997,         # Discount factor for returns
    horizon=15,             # Planning horizon for imagination
    actor_entropy=1e-4,     # Entropy regularization coefficient
    
    # Target network settings
    slow_target=True,       # Use slow-moving target networks
    slow_target_fraction=1, # Target network update rate
    slow_target_update=100, # Target network update frequency
)
```

## Deployment

Once trained, you can deploy the model using the RosNav-RL action server:

```python
from rosnav_rl import RL_Agent
from rosnav_rl.action_server import ROSNavRLServer

# Load agent
agent = RL_Agent(agent_cfg, agent_state_container)
agent.initialize_model()

# Load model weights
agent.model.load("latest")

# Start server
server = ROSNavRLServer(agent)
server.start()
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size and batch length
   - Decrease model size (hidden units, layers)
   - Reduce number of parallel environments

2. **Training Instability**
   - Lower learning rate
   - Increase gradient clipping threshold
   - Adjust actor entropy

3. **ROS Integration Problems**
   - Ensure observation collectors match ROS topics
   - Check message types and formats
   - Verify action space matches robot controller

### Logs Analysis

Logs are stored in the specified `logdir` with the following structure:
```
logdir/
  ├── train_eps/       # Training episode data
  ├── eval_eps/        # Evaluation episode data
  ├── latest.pt       # Latest model checkpoint
  ├── training_config.yaml       # Training hyperparameters
  └── metrics.jsonl   # Training metrics
```

### Credits

https://github.com/NM512/dreamerv3-torch