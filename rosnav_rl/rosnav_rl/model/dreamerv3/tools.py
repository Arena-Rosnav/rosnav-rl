import collections
import datetime
import io
import json
import os
import pathlib
import random
import re
import time
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import distributions as torchd
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

to_np = lambda x: x.detach().cpu().float().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model: torch.nn.Module):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    """
    A context manager for timing CUDA operations.

    This class utilizes CUDA events to record timing information for operations executed within its context.
    It prints the elapsed time in seconds upon exiting the context.

    Example:
        with TimeRecording('Forward pass:') as tr:
            output = model(input)
        # This will print: Forward pass: <time in seconds>

    Args:
        comment (str): A string to be printed before the timing information when the context is exited.

    Notes:
        - This class is designed for use with CUDA operations, as it uses CUDA events for timing.
        - The timing is done in milliseconds internally and converted to seconds for output.
        - Requires PyTorch and CUDA to be available.
    """

    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    """
    A logging utility for tracking and visualizing training metrics.

    This class provides functionality to log scalar values, images, and videos during
    training, and writes them to disk using TensorBoard's SummaryWriter. It also
    supports computation of frames per second (fps) and offline logging.

    Attributes:
        _logdir (Path): Directory where logs are saved.
        _writer (SummaryWriter): TensorBoard SummaryWriter instance.
        _last_step (int, optional): Last recorded step for fps calculation.
        _last_time (float, optional): Last recorded time for fps calculation.
        _scalars (dict): Dictionary to store scalar values before writing.
        _images (dict): Dictionary to store images before writing.
        _videos (dict): Dictionary to store videos before writing.
        step (int): Current step counter.

    Methods:
        scalar(name, value): Log a scalar value.
        image(name, value): Log an image.
        video(name, value): Log a video.
        write(fps=False, step=False): Write all logged data to disk.
        _compute_fps(step): Calculate frames per second between steps.
        offline_scalar(name, value, step): Log a scalar value without buffering.
        offline_video(name, value, step): Log a video without buffering.
    """

    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        # Pretty-print: group metrics by category, one group per line
        _GROUPS = [
            ("Episode ", lambda k: k in ("dataset_size", "train_return", "train_length",
                                         "train_episodes", "eval_return", "eval_length",
                                         "eval_episodes", "fps", "update_count")),
            ("World   ", lambda k: k in ("model_loss", "model_grad_norm", "kl", "kl_free",
                                         "dyn_loss", "rep_loss", "dyn_scale", "rep_scale",
                                         "prior_ent", "post_ent")),
            ("Heads   ", lambda k: k.endswith("_loss") and k not in ("model_loss",
                                         "actor_loss", "value_loss", "dyn_loss", "rep_loss")),
            ("Actor   ", lambda k: k.startswith("actor") or k.startswith("imag")
                                   or k.startswith("normed") or k.startswith("EMA")
                                   or k in ("value_mean", "value_std", "value_min", "value_max",
                                            "target_mean", "target_std", "target_min", "target_max")),
            ("Critic  ", lambda k: k.startswith("value_loss") or k == "value_grad_norm"),
        ]
        used = set()
        lines = []
        for label, matcher in _GROUPS:
            items = [(k, v) for k, v in scalars if matcher(k) and k not in used]
            if items:
                used.update(k for k, _ in items)
                lines.append(f"  {label}| " + "  ".join(f"{k} {v:.3g}" for k, v in items))
        # Any remaining metrics go in a catch-all line
        rest = [(k, v) for k, v in scalars if k not in used]
        if rest:
            lines.append("  Other   | " + "  ".join(f"{k} {v:.3g}" for k, v in rest))
        print(f"[{step}]")
        print("\n".join(lines))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


_State = Tuple[int, int, np.ndarray, np.ndarray, List[np.ndarray], Any, List[float]]


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
    no_image_key=False,
) -> _State:
    """Simulates the interaction between an agent and environments, collecting transitions and logging metrics.

    This function handles both training and evaluation scenarios, managing environment resets, agent actions,
    and data collection into a cache. It also handles logging of various metrics and saving of episodes.

    Args:
        agent (callable): A function that takes observations, done flags, and agent state, returning actions and new state.
        envs (list): List of environments to simulate.
        cache (OrderedDict): Cache to store transitions and episodes.
        directory (str): Directory path to save episode data.
        logger (Logger): Logger object to record metrics and videos.
        is_eval (bool, optional): Whether this is an evaluation run. Defaults to False.
        limit (int, optional): Maximum number of episodes to keep in cache. Defaults to None.
        steps (int, optional): Number of steps to simulate. Defaults to 0.
        episodes (int, optional): Number of episodes to simulate. Defaults to 0.
        state (tuple, optional): Previous simulation state (step, episode, done, length, obs, agent_state, reward).
            Defaults to None.
        no_image_key (bool, optional): Whether to ignore image logging. Defaults to False.
    Returns:
        tuple: A 7-element tuple containing:
            - remaining_steps (int): Steps left to simulate
            - remaining_episodes (int): Episodes left to simulate
            - done (ndarray): Boolean array indicating if each env is done
            - length (ndarray): Current episode lengths
            - obs (list): Current observations
            - agent_state: Current agent state
            - reward (list): Current rewards

    Note:
        Either steps or episodes should be non-zero to determine simulation length.
        The cache is cleared to keep only the last item when in evaluation mode.
    """
    def reset_envs():
        results = [env.reset() for env in envs]
        results = [r() for r in results]
    
    first_iter = True
    # Resolve pause callable once — Parallel.__getattr__ re-raises remote
    # AttributeError as a plain Exception, so getattr(..., None) is not enough.
    try:
        _pause = envs[0].pause
    except Exception:
        _pause = None

    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        if first_iter:
            first_iter = False
            reset_envs()
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        if _pause is not None:
            _pause(True)
        action, agent_state = agent(obs, done, agent_state)
        if _pause is not None:
            _pause(False)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # Snapshot ids NOW so they don't change if a reset is triggered later
            done_ids = [envs[i].id for i in indices]
            # Save all done episodes BEFORE erasing any of them from cache.
            # Calling erase_over_episodes inside the loop can delete a later
            # done-env's episode before it is saved, causing a KeyError.
            for i, eid in zip(indices, done_ids):
                save_episodes(directory, {eid: cache[eid]})

            for i, eid in zip(indices, done_ids):
                length = len(cache[eid]["reward"]) - 1
                score = float(np.array(cache[eid]["reward"]).sum())
                if not no_image_key:
                    video = cache[eid]["image"]
                # record logs given from environments
                for key in list(cache[eid].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[eid][key]).sum())
                        )
                        # log items won't be used later
                        cache[eid].pop(key)

                if not is_eval:
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    if not no_image_key:
                        logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True

            if not is_eval:
                # Erase once after all done episodes are saved
                step_in_dataset = erase_over_episodes(cache, limit)
                logger.scalar(f"dataset_size", step_in_dataset)
                logger.scalar(f"train_episodes", len(cache))
                logger.write(step=logger.step)
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


def add_to_cache(cache, id, transition):
    """Add a transition to a cache dictionary using an ID as key.

    This function adds transition data to a cache dictionary, organizing it by ID. If the ID
    exists, it appends the transition values to existing lists. If not, it creates a new
    entry with the transition values as single-element lists.

    Args:
        cache (dict): The cache dictionary to store transitions
        id (str): The identifier for the transition
        transition (dict): A dictionary containing the transition data with various keys and values

    Note:
        If an ID exists but is missing some keys present in the new transition,
        those keys are initialized with zeros before adding the new value.
    """
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    """
    Erase episodes from cache to maintain dataset size limit.

    This function processes episodes in reverse chronological order and removes episodes
    that would exceed the specified dataset size limit.

    Args:
        cache (dict): Dictionary containing episodes with their corresponding data.
                     Each episode must have a 'reward' key with its sequence length.
        dataset_size (int): Maximum number of steps to keep in dataset.
                           If 0 or None, keeps all episodes.

    Returns:
        int: Total number of steps remaining in dataset after erasing episodes.

    Example:
        >>> cache = {
            0: {'reward': [1, 2, 3]},
            1: {'reward': [4, 5]}
        }
        >>> erase_over_episodes(cache, 3)
        2
    """
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    """Convert numpy array to specified precision.

    This function converts a numpy array to a specified numerical precision while preserving
    the base data type (float, int, uint8, or bool).

    Args:
        value: Input value to be converted to numpy array with specified precision
        precision (int, optional): Bit precision for float and integer types.
            Supported values are 16, 32, or 64. Defaults to 32.

    Returns:
        np.ndarray: Array with the specified precision maintaining original data type

    Raises:
        NotImplementedError: If input data type is not float, signed integer, uint8, or bool

    Example:
        >>> convert([1.5, 2.5], precision=16)  # Returns float16 array
        >>> convert([1, 2], precision=64)      # Returns int64 array
    """
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    """Save episodes data to compressed NPZ files in the specified directory.

    This function saves episode data in NPZ format, where each episode file is named with
    its corresponding filename and length. The episodes are saved in a compressed format
    using numpy's savez_compressed.

    Args:
        directory (str or pathlib.Path): The target directory where episodes will be saved.
            Will be created if it doesn't exist.
        episodes (dict): A dictionary where keys are filenames and values are episode data
            dictionaries containing at least a "reward" key to determine episode length.

    Returns:
        bool: True if episodes were successfully saved.

    Example:
        >>> episodes = {
        ...     'episode1': {'reward': [1, 2, 3], 'action': [0, 1, 0]},
        ...     'episode2': {'reward': [1, 2], 'action': [1, 0]}
        ... }
        >>> save_episodes('./data', episodes)
        True
    """
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(
    generator: Generator[Dict[str, NDArray], None, None], batch_size: int
) -> Generator[Dict[str, NDArray], None, None]:
    """Convert a generator of dictionaries into a generator of batched dictionaries.

    This function takes a generator that yields dictionaries containing numpy arrays and
    converts it into a generator that yields batched versions of those dictionaries.

    Args:
        generator (Generator[dict[str, NDArray], None, None]): A generator that yields
            dictionaries where each value is a numpy array.
        batch_size (int): The desired size of each batch.

    Returns:
        Generator[dict[str, NDArray], None, None]: A generator that yields dictionaries
            where each value is a batched numpy array with shape (batch_size, *original_shape).

    Example:
        >>> def sample_generator():
        ...     while True:
        ...         yield {'x': np.array([1, 2, 3])}
        >>> batched_gen = from_generator(sample_generator(), batch_size=2)
        >>> next(batched_gen)
        {'x': array([[1, 2, 3],
                     [1, 2, 3]])}
    """
    while True:
        batch = [next(generator) for _ in range(batch_size)]
        data = {
            key: np.stack([sample[key] for sample in batch], axis=0)
            for key in batch[0].keys()
        }
        yield data


def sample_episodes(episodes, length, seed=0):
    """
    Sample fixed-length sequences from a collection of episodes.

    This generator function samples sequences of specified length from multiple episodes,
    potentially concatenating data from different episodes to reach the desired length.

    Args:
        episodes (dict): Dictionary of episodes, where each episode is a dict containing
                        trajectory data with numpy arrays as values.
        length (int): Desired length of the sampled sequences.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Yields:
        dict: A dictionary containing sampled sequences with keys matching the input episodes,
              excluding keys starting with 'log_'. The sequences have the following properties:
              - Each sequence has exactly 'length' timesteps
              - When concatenating episodes, 'is_first' flag is set to True at the start
                of each new episode segment (if 'is_first' exists in the data)
              - All sequences are guaranteed to include at least one transition
              (minimum 2 timesteps from the source episode)

    Note:
        The sampling probability for each episode is proportional to its length.
        Episodes with less than 2 timesteps are skipped.
    """
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    """Load episode data from a directory containing .npz files.

    This function loads episode data stored in .npz files from a specified directory. Each episode
    is stored as a dictionary containing numpy arrays for different aspects of the episode.

    Args:
        directory (str or pathlib.Path): Path to the directory containing episode .npz files
        limit (int, optional): Maximum total number of timesteps to load across all episodes.
            If None, loads all episodes. Defaults to None.
        reverse (bool, optional): If True, loads episodes in reverse chronological order.
            If False, loads in chronological order. Defaults to True.

    Returns:
        collections.OrderedDict: An ordered dictionary mapping episode filenames (without extension
            when reverse=True, full path when reverse=False) to episode data dictionaries. Each
            episode dictionary contains numpy arrays keyed by the original .npz file's keys.

    Raises:
        Any exceptions during file loading will be caught and printed, skipping problematic files.

    Notes:
        - Episodes are counted towards the limit based on their reward length minus 1
        - The function assumes each .npz file contains episode data with a "reward" key
        - Invalid or corrupted .npz files will be skipped with an error message
    """
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    """A wrapper class for sampling-based distribution operations.

    This class provides methods to compute statistical properties of a distribution
    through Monte Carlo sampling when analytical solutions are not available or
    difficult to compute.

    Args:
        dist: The underlying distribution object to sample from.
        samples (int, optional): Number of samples to use for estimating statistics. Defaults to 100.

    Properties:
        name: Returns the name of the distribution class ("SampleDist").

    Methods:
        mean(): Estimates the mean of the distribution through sampling.
        mode(): Estimates the mode by finding the sample with highest probability.
        entropy(): Estimates the entropy through sampling.

    Note:
        This class delegates unknown attributes to the underlying distribution
        object through __getattr__.
    """

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """A OneHotCategorical distribution with straight-through gradients.

    This class extends PyTorch's OneHotCategorical distribution to support straight-through
    gradients and optional uniform mixture for better exploration.

    Args:
        logits (torch.Tensor, optional): The log-probabilities of the distribution.
            Either logits or probs must be provided.
        probs (torch.Tensor, optional): The probabilities of the distribution.
            Either logits or probs must be provided.
        unimix_ratio (float, optional): Ratio for mixing the distribution with a uniform
            distribution. Must be between 0 and 1. Defaults to 0.0.

    Methods:
        mode(): Returns the mode of the distribution with straight-through gradients.
        sample(sample_shape=(), seed=None): Samples from the distribution with
            straight-through gradients.

    Note:
        The straight-through gradient estimator allows gradients to flow through
        discrete samples by treating the forward pass as discrete but the backward
        pass as continuous.
    """

    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    """Discretized Distribution Class for handling discretized continuous values.

    This class implements a discretized distribution using a fixed number of buckets
    between specified low and high values. It provides functionality for computing
    mean, mode, and log probabilities of the distribution.

    Args:
        logits (torch.Tensor): The unnormalized log probabilities of the distribution.
        low (float, optional): Lower bound of the discretization range. Defaults to -20.0.
        high (float, optional): Upper bound of the discretization range. Defaults to 20.0.
        transfwd (callable, optional): Forward transform function. Defaults to symlog.
        transbwd (callable, optional): Backward transform function. Defaults to symexp.
        device (str, optional): Device to store tensors on. Defaults to "cuda".

    Attributes:
        logits (torch.Tensor): The unnormalized log probabilities.
        probs (torch.Tensor): Softmax probabilities computed from logits.
        buckets (torch.Tensor): Linear space of 255 points between low and high.
        width (float): Width of each bucket.
        transfwd (callable): Forward transform function.
        transbwd (callable): Backward transform function.

    Methods:
        mean(): Computes the mean of the distribution.
        mode(): Computes the mode of the distribution.
        log_prob(x): Computes the log probability of input x.
        log_prob_target(target): Computes the log probability for a given target distribution.

    Note:
        The distribution uses 255 discrete buckets and implements interpolation
        for computing log probabilities of continuous values.
    """

    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    """MSE (Mean Squared Error) Distribution class for calculating log probabilities.

    This class implements a distribution that uses mean squared error (MSE) to calculate
    log probabilities between predicted and actual values.

    Args:
        mode (torch.Tensor): The predicted/expected values tensor.
        agg (str, optional): Aggregation method for the loss calculation.
            Can be either "sum" or "mean". Defaults to "sum".

    Attributes:
        _mode (torch.Tensor): Stores the predicted/expected values.
        _agg (str): Stores the aggregation method.

    Methods:
        mode(): Returns the mode of the distribution.
        mean(): Returns the mean of the distribution (same as mode).
        log_prob(value): Calculates negative MSE loss between mode and value.

    Raises:
        NotImplementedError: If aggregation method is not "sum" or "mean".
        AssertionError: If shapes of mode and value tensors don't match.
    """

    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    """A class to handle symmetric logarithmic distributions.

    This class implements symmetric logarithmic (symlog) distribution calculations,
    particularly useful for handling data that spans several orders of magnitude.

    Args:
        mode (torch.Tensor): The mode of the distribution in symlog space.
        dist (str, optional): Distance metric to use. Can be 'mse' or 'abs'. Defaults to 'mse'.
        agg (str, optional): Aggregation method for loss calculation. Can be 'sum' or 'mean'. Defaults to 'sum'.
        tol (float, optional): Tolerance value for numerical stability. Defaults to 1e-8.

    Attributes:
        _mode (torch.Tensor): Stored mode of the distribution.
        _dist (str): Selected distance metric.
        _agg (str): Selected aggregation method.
        _tol (float): Tolerance value for computations.

    Methods:
        mode(): Returns the exponential of the mode.
        mean(): Returns the exponential of the mode (identical to mode()).
        log_prob(value): Computes the negative loss based on the distance between mode and value.

    Note:
        The class assumes the existence of symlog and symexp functions for
        symmetric logarithmic transformations.
    """

    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    """A wrapper class for continuous probability distributions with optional value clamping.

    This class wraps a base distribution and provides additional functionality for handling
    continuous probability distributions, including optional absolute maximum value clamping.

    Args:
        dist (torch.distributions.Distribution): The base probability distribution.
        absmax (float, optional): The absolute maximum value for clamping. If None, no clamping is applied.

    Attributes:
        mean: The mean of the underlying distribution.
        absmax: The absolute maximum value for clamping.

    Methods:
        entropy(): Returns the entropy of the distribution.
        mode(): Returns the mode of the distribution with optional clamping.
        sample(sample_shape=()): Samples from the distribution with optional clamping.
        log_prob(x): Computes the log probability of x under the distribution.

    Note:
        When absmax is specified, both mode() and sample() methods will clamp their outputs
        to the range [-absmax, absmax] while preserving the direction of the vectors.
    """

    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    """A wrapper class for Bernoulli distribution.

    This class wraps a Bernoulli distribution object and provides additional functionality
    for sampling, computing log probabilities, entropy, and mode calculations.

    Attributes:
        _dist: The underlying Bernoulli distribution object
        mean: The mean of the distribution

    Methods:
        entropy(): Computes the entropy of the distribution
        mode(): Computes the mode of the distribution with gradient preservation
        sample(sample_shape): Generates samples from the distribution
        log_prob(x): Computes the log probability of the input x

    Args:
        dist: A Bernoulli distribution object to wrap

    Returns:
        A Bernoulli distribution wrapper object
    """

    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    """Unnormalized Huber distribution based on Normal distribution.

    This class implements an unnormalized probability distribution using a Huber-like loss function.
    It inherits from torch.distributions.normal.Normal but modifies the log probability calculation
    to use a smoothed L1/L2 loss.

    Args:
        loc (Tensor): The mean of the distribution (often referred to as μ).
        scale (Tensor): The standard deviation of the distribution (often referred to as σ).
        threshold (float, optional): The threshold parameter for the Huber-like loss. Defaults to 1.
        **kwargs: Additional arguments passed to the parent Normal distribution.

    Notes:
        The log probability is computed using a modified Huber-like loss:
        -[sqrt((x - μ)² + threshold²) - threshold]

        Unlike a standard probability distribution, this does not integrate to 1
        and is therefore unnormalized.
    """

    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    """Truncated Normal distribution with clipping and scaling.

    A subclass of PyTorch's Normal distribution that implements truncation through clipping
    and optional scaling of samples.

    Args:
        loc (Tensor): The mean of the normal distribution
        scale (Tensor): The standard deviation of the normal distribution
        low (float): Lower bound for clipping
        high (float): Upper bound for clipping
        clip (float, optional): Small offset from bounds for numerical stability. Defaults to 1e-6.
        mult (float, optional): Multiplicative factor for scaling samples. Defaults to 1.

    Example:
        >>> dist = SafeTruncatedNormal(torch.zeros(1), torch.ones(1), -2, 2)
        >>> sample = dist.sample()  # Returns a value between -2 and 2

    Note:
        The implementation uses a reparameterization trick to maintain gradients
        while enforcing the bounds through soft clipping.
    """

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    """A bijector implementing the hyperbolic tangent function and its inverse.

    This bijector applies the tanh function as the forward transformation and its
    inverse (arctanh) as the inverse transformation. It also computes the log of
    the absolute determinant of the Jacobian matrix.

    The tanh function maps real numbers to the interval (-1, 1), making it useful
    for constraining variables to this range.

    Methods:
        _forward(x): Applies tanh transformation.
        _inverse(y): Applies inverse tanh (arctanh) transformation.
        _forward_log_det_jacobian(x): Computes the log determinant of the Jacobian
            matrix for the forward transformation.

    Note:
        The inverse transformation clamps values to (-0.99999997, 0.99999997) when
        they are within [-1, 1] to prevent numerical instabilities near the boundaries.
    """

    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    """
    Recursively compute lambda returns for a given sequence using reverse scanning.

    This function implements a reverse scan operation to compute cumulative returns,
    typically used in reinforcement learning for calculating lambda returns.

    Args:
        fn (callable): A function that takes the previous output and current inputs
                      to compute the next output.
        inputs (tuple of torch.Tensor): Input tensors to process. All tensors should
                                      have the same first dimension (sequence length).
        start (torch.Tensor): Initial value to start the scanning process.

    Returns:
        tuple of torch.Tensor: Processed outputs after scanning, reshaped and reversed.
                             Returns a tuple containing the computed lambda returns.

    Note:
        - The function processes the sequence in reverse order.
        - Output tensors are reshaped to have a singleton dimension before being returned.
        - This implementation is specifically designed for lambda return calculations
          in reinforcement learning contexts.
    """
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    """Calculate lambda returns for temporal difference learning.

    This function computes a mixture of n-step returns using the lambda parameter for
    temporal difference (TD) learning, also known as TD(λ) learning.

    Args:
        reward (torch.Tensor): Tensor of rewards for each timestep.
        value (torch.Tensor): Tensor of value estimates for each timestep.
        pcont (Union[float, torch.Tensor]): Continuation probability (discount factor).
            Can be a constant float or a tensor of the same shape as reward.
        bootstrap (Optional[torch.Tensor]): Bootstrap value for computing returns beyond
            the last timestep. If None, defaults to zero tensor.
        lambda_ (float): Lambda parameter that determines the mixing between different
            n-step returns. λ=1 gives Monte Carlo returns, λ=0 gives one-step returns.
        axis (int): The time axis along which to compute returns.

    Returns:
        torch.Tensor: Computed lambda returns with the same shape as the input tensors.

    Raises:
        AssertionError: If reward and value tensors don't have the same number of dimensions.

    Note:
        The implementation uses an efficient scan operation to compute returns in reverse,
        combining one-step returns with bootstrapped future returns according to lambda.
    """
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        """Initialize an Optimizer wrapper.

        This class wraps various PyTorch optimizers with additional functionality
        for gradient clipping and weight decay.

        Args:
            name (str): The name of the optimizer instance
            parameters (iterable): Iterable of parameters to optimize
            lr (float): Learning rate
            eps (float, optional): Term added to denominator for numerical stability. Defaults to 1e-4
            clip (float, optional): Gradient clipping threshold. If None, no clipping is performed. Defaults to None
            wd (float, optional): Weight decay coefficient. Must be between 0 and 1. Defaults to None
            wd_pattern (str, optional): Regex pattern for selecting variables for weight decay. Defaults to r".*"
            opt (str, optional): Optimizer type ('adam', 'nadam', 'adamax', 'sgd', 'momentum'). Defaults to "adam"
            use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False

        Raises:
            AssertionError: If weight decay is not between 0 and 1, or if clip is less than 1
        """
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        """Update optimizer parameters based on computed loss.

        Args:
            loss (torch.Tensor): Zero-dimensional tensor containing the computed loss.
            params (iterable): Iterable of parameters to optimize.
            retain_graph (bool, optional): If True, retains computation graph. Defaults to True.

        Returns:
            dict: Dictionary containing metrics:
                - '{name}_loss': Detached loss value as numpy array
                - '{name}_grad_norm': Gradient norm after clipping as numpy array

        Raises:
            AssertionError: If loss tensor is not zero-dimensional.
        """
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = to_np(norm)
        return metrics

    def _apply_weight_decay(self, varibs):
        """Apply weight decay to a list of variables.

        This method implements weight decay by scaling the variables by (1 - weight_decay_rate).
        Currently only supports applying weight decay to all variables (non-trivial patterns
        are not implemented).

        Args:
            varibs (list): List of torch.nn.Parameter variables to apply weight decay to.

        Raises:
            NotImplementedError: If weight decay pattern is non-trivial (not ".*").
        """
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    """Parse and convert arguments to their appropriate type based on a default value.

    This function returns a lambda function that can parse both string and non-string inputs
    according to the type of the provided default value.

    Args:
        default: The default value that determines the target type for conversion.
                 Can be None, bool, int, float, list, tuple or any other type.

    Returns:
        callable: A lambda function that takes an input x and:
                 - If x is a string: parses it according to default's type
                 - If x is not a string: converts it to match default's container type
                                        (if default is list/tuple) or returns x as is

    Examples:
        >>> parser = args_type(True)
        >>> parser("True")  # Returns True
        >>> parser = args_type(5)
        >>> parser("5")  # Returns 5
        >>> parser = args_type([1])
        >>> parser("1,2,3")  # Returns (1, 2, 3)
    """

    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    """
    A function that performs a static scan operation over input tensors.

    This function applies a given function sequentially over inputs, maintaining and accumulating state.
    Similar to tf.scan but implemented for PyTorch tensors, handling both dictionary and non-dictionary states.

    Args:
        fn (callable): A function that takes the previous state and current inputs and returns the next state
        inputs (tuple): A tuple of input tensors to scan over
        start: Initial state value that can be either a dictionary of tensors or a list of tensors/dictionaries

    Returns:
        list: A list containing either:
            - A single dictionary of concatenated tensor outputs if the state is a dictionary
            - Multiple tensors/dictionaries if the state is a list of multiple elements

    Note:
        - The function handles both dictionary and non-dictionary state types
        - Output tensors are stacked along dimension 0
        - Each element of the output preserves the structure of the state (dict or tensor)
    """
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    """A class that tracks occurrences of events based on step intervals.

    This class helps determine how many times an event should occur given a step counter
    and a specified interval. It's useful for periodic checks or actions in iterative processes.

    Args:
        every (int): The interval between events. If 0 or None, always returns 0.

    Attributes:
        _every (int): Stores the interval between events
        _last (int): Stores the last step where the event occurred

    Methods:
        __call__(step): Returns number of times event should occur since last check

    Example:
        >>> counter = Every(10)
        >>> counter(5)  # First call returns 1
        1
        >>> counter(25)  # Returns 2 (two intervals of 10 have passed)
        2
        >>> counter(30)  # Returns 0 (less than one interval has passed)
        0
    """

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    """A callable class that checks if a step count has reached a limit.

    This class provides a simple way to implement step-based termination conditions,
    particularly useful in training loops or iterative processes.

    Args:
        until (int, optional): The maximum number of steps. If None, always returns True.

    Returns:
        bool: True if the current step is less than the until value, or if until is None.
              False if the current step has reached or exceeded the until value.

    Examples:
        >>> checker = Until(100)
        >>> checker(50)
        True
        >>> checker(150)
        False
        >>> unlimited = Until(None)
        >>> unlimited(1000)
        True
    """

    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    """Initialize weights for neural network layers using specific initialization schemes.

    This function applies custom weight initialization to Linear, Conv2d, ConvTranspose2d
    and LayerNorm layers using truncated normal distribution or constant values.

    Args:
        m (torch.nn.Module): Neural network module whose weights need to be initialized.
                            Supported types are nn.Linear, nn.Conv2d, nn.ConvTranspose2d,
                            and nn.LayerNorm.

    Details:
        - For Linear, Conv2d and ConvTranspose2d layers:
            * Calculates scale based on input and output features
            * Initializes weights using truncated normal distribution
            * Sets bias to 0 if present
        - For LayerNorm:
            * Sets weight to 1
            * Sets bias to 0 if present

    Note:
        The standard deviation for truncated normal is calculated using a specific
        scaling factor (0.87962566103423978) derived from the model architecture.
    """
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    """Initialize network weights using uniform distribution.

    This function returns a weight initialization function that can be applied to neural network modules.
    It performs uniform initialization for Linear, Conv2d, ConvTranspose2d and LayerNorm layers with
    calculated limits based on the given scale and layer dimensions.

    Args:
        given_scale (float): Base scale factor for weight initialization

    Returns:
        function: Weight initialization function that can be applied to network modules

    The initialization details:
    - For Linear, Conv2d and ConvTranspose2d layers:
        - Weights are initialized from uniform distribution U(-limit, limit)
        - Limit is calculated as sqrt(3 * scale), where scale = given_scale / ((in_features + out_features)/2)
        - Biases are initialized to 0 if present
    - For LayerNorm:
        - Weights are initialized to 1
        - Biases are initialized to 0 if present
    """

    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    """Recursively collects optimizer state dictionaries from a given object and its attributes.

    This function traverses through an object's attributes and nested modules to find all
    PyTorch optimizers and collects their state dictionaries. It handles cyclic references
    and nested torch.nn.Module instances.

    Args:
        obj: The object to traverse through. Can be any Python object with attributes.
        path (str, optional): Current attribute path for nested objects. Defaults to "".
        optimizers_state_dicts (dict, optional): Dictionary to store optimizer state dicts.
            Defaults to None and will be initialized as empty dict.
        visited (set, optional): Set of visited object IDs to prevent cyclic traversal.
            Defaults to None and will be initialized as empty set.

    Returns:
        dict: A dictionary mapping attribute paths to optimizer state dictionaries.
            Keys are string paths (e.g., "model.optimizer") and values are the
            corresponding optimizer's state_dict().

    Example:
        >>> model = MyModel()
        >>> optimizer_states = recursively_collect_optim_state_dict(model)
        >>> # Returns {'model.optimizer': optimizer_state_dict}
    """
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    """
    Recursively loads optimizer state dictionaries into nested objects.

    This function traverses a nested object structure using dot-notation paths and loads
    optimizer state dictionaries at the specified locations.

    Args:
        obj: The root object containing nested optimizers.
        optimizers_state_dicts (dict): Dictionary mapping dot-notation paths to optimizer
            state dictionaries. The paths specify the attribute traversal sequence to reach
            the target optimizer.

    Example:
        If optimizers_state_dicts = {'model.encoder.optim': state_dict},
        this will load state_dict into obj.model.encoder.optim

    Note:
        The target attributes must exist and be valid optimizers that support load_state_dict().
    """
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)
