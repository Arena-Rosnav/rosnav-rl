from typing import List

from stable_baselines3.common.vec_env import VecEnv, VecNormalize
import rosnav_rl.utils.stable_baselines3.vec_frame_stack as vec_frame_stack


def apply_vec_framestack(env: VecEnv, stack_size: int) -> VecEnv:
    return vec_frame_stack.VecFrameStack(
        env, n_stack=stack_size, channels_order="first"
    )


def apply_vec_normalize(
    env: VecEnv,
    path: str = None,
    is_training: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    gamma: float = 0.99,
    epsilon: float = 1e-8,
    norm_obs_key: List[str] = None,
) -> VecEnv:
    """
    Apply vector normalization to the environment.

    Parameters:
    env (VecEnv): The environment to normalize.
    is_training (bool): Whether the environment is in training mode.
    path (str, optional): Path to load the normalization statistics from. Defaults to None.
    norm_obs (bool, optional): Whether to normalize observations. Defaults to True.
    norm_reward (bool, optional): Whether to normalize rewards. Defaults to True.
    clip_obs (float, optional): Clip value for observations. Defaults to 10.0.
    clip_reward (float, optional): Clip value for rewards. Defaults to 10.0.
    gamma (float, optional): Discount factor for rewards. Defaults to 0.99.
    epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-8.
    norm_obs_key (List[str], optional): Keys for normalizing specific observations. Defaults to None.

    Returns:
    VecEnv: The normalized environment.
    """
    if path:
        return VecNormalize.load(load_path=path, venv=env)
    return VecNormalize(
        env,
        training=is_training,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
        epsilon=epsilon,
        norm_obs_key=norm_obs_key,
    )


def get_vec_normalize(env: VecEnv) -> VecNormalize:
    if isinstance(env, VecNormalize):
        return env

    # Unwrap the environment until we find VecNormalize
    if hasattr(env, "venv"):
        return get_vec_normalize(env.venv)

    return None


def get_vec_framestack(env: VecEnv) -> vec_frame_stack.VecFrameStack:
    if isinstance(env, vec_frame_stack.VecFrameStack):
        return env

    # Unwrap the environment until we find VecFrameStack
    if hasattr(env, "venv"):
        return get_vec_framestack(env.venv)

    return None
