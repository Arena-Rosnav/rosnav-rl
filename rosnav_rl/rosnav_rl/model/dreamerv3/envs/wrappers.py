import datetime
import gymnasium as gym
import numpy as np
import uuid


class TimeLimit(gym.Wrapper):
    """A wrapper for OpenAI Gym environments that imposes a time limit on episodes.

    This wrapper terminates episodes after a specified number of steps, regardless of
    the environment's natural termination conditions. When the time limit is reached,
    the done flag is set to True and the episode ends.

    Args:
        env (gym.Env): The environment to apply the wrapper to.
        duration (int): Maximum number of steps allowed in an episode.

    Attributes:
        _duration (int): Maximum number of steps allowed in an episode.
        _step (int): Current step count in the episode. None if environment needs reset.

    Examples:
        env = TimeLimit(gym.make('CartPole-v1'), duration=500)
    """

    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        # assert self._step is not None, "Must reset environment."
        if self._step is None:
            raise AssertionError("Must reset environment.")
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    """A wrapper for OpenAI Gym environments that normalizes the action space to [-1, 1].

    This wrapper scales the action space of the environment to be within the range [-1, 1].
    It handles both finite and infinite action bounds, preserving the original bounds where
    they are infinite.

    Args:
        env (gym.Env): The environment to normalize actions for.

    Attributes:
        _mask (np.ndarray): Boolean mask indicating which actions have finite bounds.
        _low (np.ndarray): Lower bounds of the original action space.
        _high (np.ndarray): Upper bounds of the original action space.
        action_space (gym.spaces.Box): Normalized action space with bounds mostly in [-1, 1].

    Example:
        >>> env = NormalizeActions(gym.make('Pendulum-v1'))
        >>> normalized_action = env.action_space.sample()  # Will be in [-1, 1]
    """

    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    """A wrapper that converts a discrete action space to a one-hot encoded continuous action space.

    This wrapper transforms the environment's discrete action space into a continuous space
    where actions are represented as one-hot vectors. The wrapper ensures that input actions
    are valid one-hot encodings before passing them to the underlying environment.

    Args:
        env (gym.Env): The environment to apply the wrapper to. Must have a discrete action space.

    Properties:
        action_space (gym.spaces.Box): The new action space representing one-hot vectors.
            Has shape (n,) where n is the number of discrete actions in the original space.

    Methods:
        step(action): Takes a one-hot encoded action vector and executes it in the environment.
        reset(): Resets the environment to initial state.
        _sample_action(): Helper method to sample a random valid one-hot action.

    Raises:
        AssertionError: If the input environment doesn't have a discrete action space.
        ValueError: If the provided action is not a valid one-hot vector.
    """

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    """A wrapper that adds reward to observation space.

    This wrapper adds reward value from the environment step into the
    observation dictionary with the key 'obs_reward'. If 'obs_reward'
    is not already present in the observation space, it creates a new
    Box space for it.

    Args:
        env (gym.Env): The environment to apply the wrapper to.

    Note:
        The reward is added as a single float value in a numpy array
        of shape (1,). For the initial reset, the reward is set to 0.0.
    """

    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    """A wrapper for gym environments that selects a specific component of the action.

    This wrapper is useful when dealing with multi-component action spaces where only
    a specific part of the action vector needs to be passed to the environment.

    Args:
        env (gym.Env): The environment to apply the wrapper to.
        key (str or int): The key or index to select the relevant action component.

    Methods:
        step(action): Executes the environment step with the selected action component.

    Example:
        env = SelectAction(env, key='motor_commands')
        # If action = {'motor_commands': [1,2,3], 'other_commands': [4,5,6]}
        # Only [1,2,3] will be passed to the underlying environment
    """

    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    """A wrapper that assigns a unique identifier to each environment instance.

    This wrapper generates a unique ID for the environment by combining a timestamp
    with a UUID (Universally Unique Identifier). The ID is updated each time the
    environment is reset.

    Args:
        env: The environment to wrap.

    Attributes:
        id (str): A unique identifier string in the format 'YYYYMMDDTHHMMSS-uuid'.
    """

    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()


class ResetWoInfo(gym.Wrapper):
    """A wrapper that resets the environment without returning info.

    This wrapper is useful when the info returned by the environment is not needed
    and can be discarded. It removes the info dictionary from the output of the
    reset method.

    Args:
        env (gym.Env): The environment to apply the wrapper to.

    Methods:
        reset(): Resets the environment without returning info.

    Example:
        env = ResetWoInfo(env)
        obs = env.reset()  # Returns only the observation
    """

    def reset(self):
        return self.env.reset()[0]


class ChannelFirsttoLast(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        updated_obs = {
            k: np.moveaxis(v, 0, -1)
            for k, v in obs.items()
            if isinstance(v, np.ndarray) and v.ndim == 3
        }
        obs.update(updated_obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs: dict = self.env.reset()
        updated_obs = {
            k: np.moveaxis(v, 0, -1)
            for k, v in obs.items()
            if isinstance(v, np.ndarray) and v.ndim == 3
        }
        obs.update(updated_obs)
        return obs


class WoTruncatedFlag(gym.Wrapper):
    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        return obs, reward, done, info
