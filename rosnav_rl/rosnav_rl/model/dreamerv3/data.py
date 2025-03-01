from typing import Dict, Generator, OrderedDict
import rosnav_rl.model.dreamerv3.tools as tools
import rosnav_rl.model.dreamerv3.cfg as cfg

import numpy as np
from pathlib import Path


def count_steps(folder: Path) -> int:
    """
    Counts the total number of steps in a given folder containing .npz files.

    The function assumes each .npz file name ends with a number indicating steps,
    separated by a hyphen (e.g., 'file-100.npz' has 99 steps).

    Args:
        folder: A pathlib.Path object pointing to directory containing .npz files

    Returns:
        int: Total number of steps across all .npz files in the folder,
             where each file contributes (number in filename - 1) steps

    Example:
        For a folder containing:
            episode-100.npz  (contributes 99 steps)
            episode-50.npz   (contributes 49 steps)
        The function would return 148
    """
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(
    episodes: OrderedDict[str, np.ndarray], config: cfg.DreamerV3Cfg
) -> Generator[Dict[str, np.ndarray], None, None]:
    """Creates a dataset from a collection of episodes.

    This function converts episodic data into a dataset format suitable for training.
    It samples sequences of specified length from episodes and batches them together.

    Args:
        episodes (OrderedDict[str, np.ndarray]): Collection of episodes where each episode
            is a dictionary mapping observation/action names to numpy arrays.
        config (DreamerV3Cfg): Configuration object containing training parameters like
            batch_size and batch_length.

    Returns:
        Generator[Dict[str, np.ndarray], None, None]: A generator yielding batches of
            sequences, where each batch is a dictionary mapping names to numpy arrays
            of shape [batch_size, batch_length, ...].
    """
    generator = tools.sample_episodes(episodes, config.training.batch_length)
    dataset = tools.from_generator(generator, config.training.batch_size)
    return dataset
