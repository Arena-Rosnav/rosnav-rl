import functools
import logging
import pathlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Generator, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import wandb
from torch import distributions as torchd

import rosnav_rl.model.dreamerv3.data as data_tools
import rosnav_rl.model.dreamerv3.dreamer as dreamer
import rosnav_rl.model.dreamerv3.tools as tools
from rosnav_rl.model.dreamerv3.parallel import Parallel

if TYPE_CHECKING:
    from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg


_log = logging.getLogger(__name__)

to_np = lambda x: x.detach().cpu().numpy()


def set_runtime_configuration(config: "DreamerV3Cfg"):
    """
    Configure runtime settings for the Dreamer V3 agent.

    This function initializes the random seed and sets deterministic execution mode if requested.

    Args:
        config (DreamerV3Cfg): Configuration object containing general settings like seed and
                              deterministic_run flag.
    """
    tools.set_seed_everywhere(config.general.seed)

    if config.general.deterministic_run:
        tools.enable_deterministic_run()


def prepare_config(config: "DreamerV3Cfg"):
    """
    Prepare the DreamerV3 configuration by adjusting directories and time parameters.

    This function sets up the training and evaluation directories if they are not specified,
    and adjusts various time-related parameters by dividing them by the action repeat value
    from the environment configuration.

    Parameters:
        config (DreamerV3Cfg): The configuration object for DreamerV3 containing all settings.

    Returns:
        pathlib.Path: The expanded log directory path.
    """
    logdir = pathlib.Path(config.general.logdir).expanduser()

    config.general.traindir = config.general.traindir or logdir / "train_eps"
    config.general.evaldir = config.general.evaldir or logdir / "eval_eps"
    if config.environment.action_repeat > 1:
        config.training.steps //= config.environment.action_repeat
        config.training.eval_every //= config.environment.action_repeat
        config.general.log_every //= config.environment.action_repeat
        # config.environment.time_limit //= config.environment.action_repeat TODO: Adjust time limit of environment

    return logdir


def prepare_directories(config: "DreamerV3Cfg", logdir: pathlib.Path):
    """
    Prepare directories for the DreamerV3 model training and evaluation.

    This function ensures that all necessary directories exist for storing training logs,
    model checkpoints, and evaluation results. It creates the main log directory and
    specific directories for training and evaluation as defined in the configuration.

    Args:
        config (DreamerV3Cfg): Configuration object containing directory paths.
        logdir (pathlib.Path): Path to the main log directory.

    Returns:
        None

    Note:
        This function creates directories with the parents parameter set to True,
        which means it will create any necessary parent directories.
    """
    _log.info("Logdir: %s", logdir)

    logdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    config.general.traindir.mkdir(parents=True, exist_ok=True)
    config.general.evaldir.mkdir(parents=True, exist_ok=True)


def prepare_logger(config: "DreamerV3Cfg", logdir: pathlib.Path) -> tools.Logger:
    """
    Prepare and initialize a logger for the DreamerV3 model.

    Args:
        config (DreamerV3Cfg): Configuration object for DreamerV3 that contains general and environment settings.
        logdir (pathlib.Path): Directory path where logs will be stored.

    Returns:
        tools.Logger: Initialized logger object with the appropriate step count considering the action repeat factor.

    Note:
        This function calculates the current step count from the training directory specified in the config
        and adjusts it based on the environment's action repeat parameter.
    """
    step = data_tools.count_steps(config.general.traindir)
    return tools.Logger(logdir, config.environment.action_repeat * step)


def load_episodes(config: "DreamerV3Cfg") -> Tuple[OrderedDict, OrderedDict]:
    """
    Load training and evaluation episodes for the DreamerV3 model.

    This function loads episodes from the training and evaluation directories specified in the configuration.
    For training episodes, it respects the dataset size limit from the configuration.
    For evaluation episodes, it limits to 1 episode.

    Args:
        config (DreamerV3Cfg): Configuration object containing paths and settings for loading episodes.

    Returns:
        tuple: A tuple containing:
            - train_eps: Training episodes loaded from the training directory.
            - eval_eps: Evaluation episodes loaded from the evaluation directory.
    """
    if config.general.offline_traindir:
        directory = config.general.offline_traindir.format(**vars(config))
    else:
        directory = config.general.traindir

    train_eps = tools.load_episodes(directory, limit=config.training.dataset_size)

    if config.general.offline_evaldir:
        directory = config.general.offline_evaldir.format(**vars(config))
    else:
        directory = config.general.evaldir

    eval_eps = tools.load_episodes(directory, limit=1)

    return train_eps, eval_eps


def prefill_dataset(
    config: "DreamerV3Cfg",
    train_envs: List[Parallel],
    train_eps: OrderedDict,
    logger: tools.Logger,
    action_space: gym.spaces.Space,
    observation_space: gym.spaces.Dict,
):
    """Prefills the dataset with random actions if no offline dataset is provided.

    This function is used to ensure there is enough data for training before the main training loop starts,
    which is particularly important for DreamerV3 which learns models from experience.

    Args:
        config (DreamerV3Cfg): Configuration object containing all parameters for DreamerV3.
        train_envs (List[Parallel]): List of parallelized training environments.
        train_eps (OrderedDict): Dictionary to track training episodes.
        logger (tools.Logger): Logger object for tracking metrics.
        action_space (gym.spaces.Space): The action space of the environment.
        observation_space (gym.spaces.Dict): The observation space of the environment.

    Returns:
        state: The final state after prefilling the dataset with random actions.

    Notes:
        - If an offline training directory is specified in config, no prefilling occurs.
        - For discrete action spaces, a one-hot distribution is used for random actions.
        - For continuous action spaces, a uniform distribution between low and high bounds is used.
        - The function simulates the random agent in the environment until enough steps are collected.
    """
    _log.info(
        "Action Space: %s  |  Observation Space keys: %s",
        action_space,
        list(observation_space.keys()),
    )

    _num_actions = (
        action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    )
    _image_available = observation_space.get("image", None) is not None

    # Prefill dataset if no offline dataset is provided
    if not config.general.offline_traindir:
        prefill = max(
            0,
            config.training.prefill_steps
            - data_tools.count_steps(config.general.traindir),
        )
        _log.info("Prefilling dataset (%d steps).", prefill)
        if hasattr(action_space, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(_num_actions).repeat(len(train_envs), 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(action_space.low).repeat(len(train_envs), 1),
                    torch.tensor(action_space.high).repeat(len(train_envs), 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.general.traindir,
            logger,
            limit=config.training.dataset_size,
            steps=prefill,
            no_image_key=not _image_available,
        )
        logger.step += prefill * config.environment.action_repeat
        _log.debug("Logger step after prefill: %d", logger.step)
        return state


def make_datasets(
    config: "DreamerV3Cfg", train_eps: OrderedDict, eval_eps: OrderedDict
) -> Tuple[
    Generator[Dict[str, np.ndarray], None, None],
    Generator[Dict[str, np.ndarray], None, None],
]:
    """
    Create training and evaluation datasets from episode data for DreamerV3.

    This function processes training and evaluation episode data into datasets that can be used
    for training and evaluating the DreamerV3 model.

    Args:
        config (DreamerV3Cfg): Configuration object for DreamerV3 model containing dataset parameters.
        train_eps (OrderedDict): Ordered dictionary containing training episodes data.
        eval_eps (OrderedDict): Ordered dictionary containing evaluation episodes data.

    Returns:
        tuple: A tuple containing:
            - Training dataset generator that yields batches of data.
            - Evaluation dataset generator that yields batches of data.
            Both generators produce dictionaries where keys are feature names and values are numpy arrays.
    """
    return data_tools.make_dataset(train_eps, config), data_tools.make_dataset(
        eval_eps, config
    )


def create_agent(
    config: "DreamerV3Cfg",
    action_space: gym.spaces.Space,
    observation_space: gym.spaces.Dict,
    logger: tools.Logger,
    train_dataset: Generator[Dict[str, np.ndarray], None, None],
) -> dreamer.Dreamer:
    """
    Create and initialize a DreamerV3 agent.

    This function constructs a DreamerV3 agent using the provided configuration and spaces,
    then moves it to the specified device and disables gradient calculation.

    Args:
        config (DreamerV3Cfg): Configuration object for the DreamerV3 agent.
        action_space (gym.spaces.Space): The action space of the environment.
        observation_space (gym.spaces.Dict): The observation space of the environment.
        logger (tools.Logger): Logger for tracking agent metrics.
        train_dataset (Generator[Dict[str, np.ndarray], None, None]): Generator yielding training batches.

    Returns:
        dreamer.Dreamer: The initialized DreamerV3 agent with gradients disabled.
    """
    agent = dreamer.Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        train_dataset,
    ).to(config.general.device)
    agent.requires_grad_(requires_grad=False)
    return agent


def load_checkpoint(agent: dreamer.Dreamer, logdir: pathlib.Path):
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt", weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False


def train(
    config: "DreamerV3Cfg",
    agent: dreamer.Dreamer,
    train_envs: List[Parallel],
    eval_envs: List[Parallel],
    train_eps: OrderedDict,
    eval_eps: OrderedDict,
    logger: tools.Logger,
    eval_dataset: Generator[Dict[str, np.ndarray], None, None],
    logdir: pathlib.Path,
    is_image_available: bool,
    state: tools._State = None,
    log_wandb: bool = False,
    after_eval_fn=None,
):
    """
    Train and evaluate a Dreamer agent using the specified configuration.

    This function trains a DreamerV3 agent in a loop, alternating between training and
    evaluation phases. It also periodically saves the agent's state and optimizer states.

    Args:
        config (DreamerV3Cfg): Configuration object with all parameters for training.
        agent (dreamer.Dreamer): The initialized Dreamer agent to train.
        train_envs (List[Parallel]): List of training environments.
        eval_envs (List[Parallel]): List of evaluation environments.
        train_eps (OrderedDict): Training episode storage.
        eval_eps (OrderedDict): Evaluation episode storage.
        logger (tools.Logger): Logger for metrics and videos.
        eval_dataset (Generator[Dict[str, np.ndarray], None, None]): Dataset for evaluation samples.
        logdir (pathlib.Path): Directory path to save model checkpoints.
        is_image_available (bool): Flag indicating if image observations are available.
        state (tools._State, optional): Training state object for resuming training. Defaults to None.
        log_wandb (bool, optional): Whether to log metrics to Weights & Biases. Defaults to False.

    Returns:
        None: The function doesn't return a value but saves the model during training.
    """
    try:
        # Continue training until we reach or exceed the target steps
        while agent._step < config.training.steps + config.training.eval_every:
            logger.write()

            # Run evaluation phase if configured (skip until model has been trained at least once)
            if config.training.eval_episode_num > 0 and agent._update_count > 0:
                _log.info(
                    "\n" + "=" * 60 + "\n"
                    "  EVALUATION  |  step=%d / %d  |  updates=%d\n" + "=" * 60,
                    agent._step, config.training.steps, agent._update_count,
                )
                _run_evaluation(
                    agent=agent,
                    eval_envs=eval_envs,
                    eval_eps=eval_eps,
                    eval_dataset=eval_dataset,
                    config=config,
                    logger=logger,
                    is_image_available=is_image_available,
                )
                if after_eval_fn is not None:
                    after_eval_fn(
                        logger._scalars.get("eval_return", float("-inf"))
                    )

            # Run training phase
            _log.info(
                "\n" + "-" * 60 + "\n"
                "  TRAINING    |  step=%d / %d  |  eval_every=%d\n" + "-" * 60,
                agent._step, config.training.steps, config.training.eval_every,
            )
            state = _run_training(
                agent=agent,
                train_envs=train_envs,
                train_eps=train_eps,
                config=config,
                logger=logger,
                state=state,
                is_image_available=is_image_available,
            )

            # Log metrics to Weights & Biases if configured
            if log_wandb:
                _log_to_wandb(agent.metrics)

            # Save model checkpoint
            _save_checkpoint(agent=agent, logdir=logdir)
    finally:
        # Clean up environment resources
        _close_environments(train_envs + eval_envs)


def _run_evaluation(
    agent: dreamer.Dreamer,
    eval_envs: List[Parallel],
    eval_eps: OrderedDict,
    eval_dataset: Generator[Dict[str, np.ndarray], None, None],
    config: "DreamerV3Cfg",
    logger: tools.Logger,
    is_image_available: bool,
):
    """Run the evaluation phase of training."""
    _log.info("Running evaluation (%d episode(s)).", config.training.eval_episode_num)
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(
        eval_policy,
        eval_envs,
        eval_eps,
        config.general.evaldir,
        logger,
        is_eval=True,
        episodes=config.training.eval_episode_num,
        no_image_key=not is_image_available,
    )

    # Log prediction videos if available and configured
    if is_image_available and config.general.video_pred_log:
        video_pred = agent._wm.video_pred(next(eval_dataset))
        logger.video("eval_openl", to_np(video_pred))


def _run_training(
    agent: dreamer.Dreamer,
    train_envs: List[Parallel],
    train_eps: OrderedDict,
    config: "DreamerV3Cfg",
    logger: tools.Logger,
    state: tools._State,
    is_image_available: bool,
):
    """Run the training phase and return the updated state."""
    return tools.simulate(
        agent,
        train_envs,
        train_eps,
        config.general.traindir,
        logger,
        limit=config.training.dataset_size,
        steps=config.training.eval_every,
        state=state,
        no_image_key=not is_image_available,
    )


def _save_checkpoint(agent: dreamer.Dreamer, logdir: pathlib.Path):
    """Save the agent and optimizer states to a checkpoint file."""
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


def _close_environments(envs: List[Parallel]):
    """Safely close all environment instances."""
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def _log_to_wandb(metrics: Dict):
    """Log metrics to Weights & Biases (no-op if wandb is not active)."""
    if wandb.run is None:
        return
    for key, value in metrics.items():
        wandb.log({key: wandb_format_value(value)})


def wandb_format_value(value):
    """Format values for Weights & Biases logging."""
    if isinstance(value, float):
        return value
    elif isinstance(value, list):
        if len(value) == 0:
            return 0.0
        if isinstance(value[-1], float):
            return value[-1]
        elif isinstance(value[-1], np.ndarray):
            return wandb_format_value(value[-1])
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        if value.ndim == 1:
            return value[-1]
        else:
            return value.mean()
    return float(value)
