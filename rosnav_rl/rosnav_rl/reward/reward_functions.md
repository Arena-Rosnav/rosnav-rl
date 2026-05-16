# Reward Functions in Arena-Rosnav

The reward system in Arena-Rosnav consists of modular components that enable flexible and customizable reward calculations during training. This guide explains the core components and available reward units.

## Core Components

**RewardFunction Class**
- Manages multiple reward units
- Tracks current step rewards
- Handles environment information
- Provides reward accumulation methods

**RewardUnit Class**
- Calculates specific aspects of rewards
- Handles observations and returns values
- Can be combined for complex reward functions

## Available Reward Units

Reward units 

**Goal-Related**
- RewardGoalReached: Rewards reaching target
- RewardApproachGoal: Progressive reward for goal approach

**Safety-Related**
- RewardSafeDistance: Penalizes safety violations
- RewardPedSafeDistance: For pedestrian safety
- RewardObsSafeDistance: For obstacle safety
- RewardCollision: Handles collision events
- RewardPedTypeSafetyDistance: Type-specific pedestrian safety
- RewardPedTypeCollision: Type-specific collision handling

**Movement-Related**
- RewardNoMovement: Penalizes lack of movement
- RewardDistanceTravelled: Based on distance covered
- RewardReverseDrive: Handles reverse driving
- RewardAbruptVelocityChange: For sudden velocity changes
- RewardRootVelocityDifference: Velocity difference between actions
- RewardAngularVelocityConstraint: Angular velocity control
- RewardLinearVelBoost: Linear velocity incentives

## Creating Custom Reward Units

```python
@RewardUnitFactory.register("my_reward_unit")
class MyRewardUnit(RewardUnit):
   required_observation_units = [RequiredObservation1, RequiredObservation2]

   def __init__(self, reward_function, *args, **kwargs):
      super().__init__(reward_function, True, *args, **kwargs)
      # Initialize parameters
    
   def __call__(self, obs_dict, simulation_state_container, *args, **kwargs):
      # Implement reward logic
```

## Configuration

Create reward functions by combining reward units in your reward configurations (either in python or YAML format) as dictionary:

```yaml
reward:
   reward_function_dict: 
      goal_reached:
         reward: 15

      factored_safe_distance:
         factor: -0.2

      collision:
         reward: -15

      approach_goal:
         pos_factor: 0.4
         neg_factor: 0.5
         _goal_update_threshold: 0.25
         _follow_subgoal: true
         _on_safe_dist_violation: true

      factored_reverse_drive:
         factor: 0.05
         threshold: 0.0
         _on_safe_dist_violation: true

      two_factor_velocity_difference:
         alpha: 0.005
         beta: 0.0
         _on_safe_dist_violation: true

      ped_type_collision:
         type_reward_pairs:
            0: -2.5
            1: -5

      ped_type_factored_safety_distance:
         type_factor_pairs:
            0: -0.1
            1: -0.2
         safety_distance: 1.2

      max_steps_exceeded:
         reward: -15

      angular_vel_constraint:
         penalty_factor: -0.05
         threshold: 0.5

      linear_vel_boost:
         reward_factor: 0.01
         threshold: 0.0
   reward_unit_kwargs: null
   verbose: false # print reward values
```

or in python:

```python
from rosnav_rl.reward.reward_function import RewardCfg

RewardCfg(
   reward_function_dict={
      "goal_reached": {"reward": 15},
      "factored_safe_distance": {"factor": -0.2},
      "collision": {"reward": -15},
      "approach_goal": {
         "pos_factor": 0.4,
         "neg_factor": 0.5,
         "_goal_update_threshold": 0.25,
         "_follow_subgoal": True,
         "_on_safe_dist_violation": True,
      },
      "factored_reverse_drive": {
         "factor": 0.05,
         "threshold": 0.0,
         "_on_safe_dist_violation": True,
      },
      "two_factor_velocity_difference": {
         "alpha": 0.005,
         "beta": 0.0,
         "_on_safe_dist_violation": True,
      },
      "ped_type_collision": {"type_reward_pairs": {0: -2.5, 1: -5}},
      "ped_type_factored_safety_distance": {
         "type_factor_pairs": {0: -0.1, 1: -0.2},
         "safety_distance": 1.2,
      },
      "max_steps_exceeded": {"reward": -15},
      "angular_vel_constraint": {
         "penalty_factor": -0.05,
         "threshold": 0.5,
      },
      "linear_vel_boost": {
         "reward_factor": 0.01,
         "threshold": 0.0,
      },
   },
   reward_unit_kwargs=None,
   verbose=False,
)
```


## Best Practices

- Keep reward values proportional to importance
- Use negative rewards for undesired behaviors
- Combine complementary reward units
- Test configurations thoroughly
- Monitor reward statistics during training
- Consider safety-critical behaviors first