import functools
import pathlib
from collections import OrderedDict
from functools import partial
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple

import gym.spaces
import gymnasium as gym
import numpy as np
import rospy
import torch
from torch import distributions as torchd

import rosnav_rl.model.dreamerv3.data as data_tools
import rosnav_rl.model.dreamerv3.envs.wrappers as wrappers
import rosnav_rl.model.dreamerv3.exploration as expl
import rosnav_rl.model.dreamerv3.models as models
import rosnav_rl.model.dreamerv3.tools as tools
import rosnav_rl.model.dreamerv3.dreamer as dreamer
from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg
from rosnav_rl.model.dreamerv3.parallel import Damy, Parallel

if TYPE_CHECKING:
    from rosnav_rl.reward.reward_function import RewardFunction
    from rosnav_rl.rl_agent import RL_Agent
    from rosnav_rl.spaces import BaseSpaceManager
    from rosnav_rl.states import SimulationStateContainer

to_np = lambda x: x.detach().cpu().numpy()


def setup_arena_env(config, id) -> gym.Env:
    import rl_utils.cfg.train as arena_cfg
    import rl_utils.envs.flatland_gymnasium_env as arena_flatland
    import rl_utils.tools.config as arena_config
    import rl_utils.tools.states as tools

    import rosnav_rl.rl_agent as rosnav_rl_agent
    import rosnav_rl.states.simulation as simulation_states

    _config: arena_cfg.TrainingCfg = arena_config.load_training_config(
        "training_config.yaml"
    )

    sim_states: simulation_states.SimulationStateContainer = tools.get_arena_states(
        goal_radius=_config.framework_cfg.general.goal_radius,
        max_steps=_config.framework_cfg.general.max_num_moves_per_eps,
        is_discrete=_config.agent_cfg.action_space.is_discrete,
        safety_distance=_config.framework_cfg.general.safety_distance,
        robot_cfg=_config.framework_cfg.robot,
        task_modules_cfg=_config.framework_cfg.task,
    )
    rl_agent = rosnav_rl_agent.RL_Agent(
        agent_cfg=config,
        agent_state_container=sim_states.to_agent_state_container(),
    )
    env = arena_flatland.FlatlandEnv(
        ns=f"sim_{id+1}/sim_{id+1}_jackal",
        rl_agent=rl_agent,
        simulation_state_container=sim_states,
        max_steps_per_episode=_config.framework_cfg.general.max_num_moves_per_eps,
        init_ros_node=False,
    )
    env = wrappers.WoTruncatedFlag(env)
    env = wrappers.TimeLimit(
        env, duration=_config.framework_cfg.general.max_num_moves_per_eps
    )
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.ResetWoInfo(env)
    env = wrappers.ChannelFirsttoLast(env)
    return env


def make_arena_env(
    config: DreamerV3Cfg,
    space_manager: "BaseSpaceManager",
    reward_function: "RewardFunction",
    sim_states: "SimulationStateContainer",
    id: int = 0,
):
    """
    Creates and configures a Flatland environment for robot navigation.

    This function initializes a FlatlandEnv with appropriate wrappers for use
    with DreamerV3 reinforcement learning architecture.

    Args:
        config (DreamerV3Cfg): Configuration object containing environment parameters.
        space_manager (BaseSpaceManager, optional): Manager for action and observation spaces.
        reward_function (RewardFunction, optional): Custom reward function for the environment.
        sim_states (SimulationStateContainer, optional): Container for simulation states.
        id (int, optional): Identifier for the environment instance, used to create unique namespaces.
            Defaults to 0.

    Returns:
        gym.Env: The configured Flatland environment with all necessary wrappers applied.

    Note:
        The environment is wrapped with several layers:
        - WoTruncatedFlag: Handles the truncated flag
        - TimeLimit: Enforces maximum episode duration
        - SelectAction: Simplifies action handling
        - UUID: Adds unique identifier
        - ResetWoInfo: Modifies reset behavior
        - ChannelFirsttoLast: Adjusts observation channel order
    """
    import rl_utils.envs.flatland_gymnasium_env as arena_flatland

    env = arena_flatland.FlatlandEnv(
        ns=f"sim_{id+1}/sim_{id+1}_jackal",
        space_manager=space_manager,
        reward_function=reward_function,
        simulation_state_container=sim_states,
        max_steps_per_episode=config.environment.time_limit,
        start_ros_node=True,
    )
    env = wrappers.WoTruncatedFlag(env)
    env = wrappers.TimeLimit(env, duration=config.environment.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.ResetWoInfo(env)
    env = wrappers.ChannelFirsttoLast(env)
    return env


def set_runtime_configuration(config: DreamerV3Cfg):
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


def prepare_config(config: DreamerV3Cfg):
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
    config.training.steps //= config.environment.action_repeat
    config.training.eval_every //= config.environment.action_repeat
    config.general.log_every //= config.environment.action_repeat
    config.environment.time_limit //= config.environment.action_repeat

    return logdir


def prepare_directories(config: DreamerV3Cfg, logdir: pathlib.Path):
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
    print("Logdir", logdir)

    logdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    config.general.traindir.mkdir(parents=True, exist_ok=True)
    config.general.evaldir.mkdir(parents=True, exist_ok=True)


def prepare_logger(config: DreamerV3Cfg, logdir: pathlib.Path) -> tools.Logger:
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


def load_episodes(config: DreamerV3Cfg) -> Tuple[OrderedDict, OrderedDict]:
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


def make_envs(
    config: DreamerV3Cfg, rl_agent: "RL_Agent", sim_states: "SimulationStateContainer"
):
    """
    Creates training and evaluation environments for the DreamerV3 agent.

    Args:
        config (DreamerV3Cfg): Configuration object containing DreamerV3 parameters.
        rl_agent (RL_Agent): Reinforcement learning agent with space_manager and reward_function attributes.
        sim_states (SimulationStateContainer): Container for simulation states.

    Returns:
        tuple: Two lists containing:
            - train_envs: List of training environments.
            - eval_envs: List of evaluation environments.

    Note:
        Depending on the configuration, environments can be created in parallel or sequentially.
        When not in parallel mode, environments are wrapped with a Damy wrapper.
    """
    env_args = dict(
        config=config,
        space_manager=rl_agent.space_manager,
        reward_function=rl_agent.reward_function,
        sim_states=sim_states,
    )

    def create_env_fnc(_id):
        def make_env():
            return make_arena_env(
                id=_id,
                **env_args,
            )

        return make_env

    if config.general.parallel:
        train_envs = [
            Parallel(create_env_fnc(i), "daemon")
            for i in range(config.environment.envs)
        ]
        eval_envs = train_envs
    else:
        train_envs = [
            make_arena_env(id=i, **env_args) for i in range(config.environment.envs)
        ]
        eval_envs = train_envs

        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    return train_envs, eval_envs


def prefill_dataset(
    config: DreamerV3Cfg,
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
    print("Action Space", action_space)
    print("Observation Space", observation_space)

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
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(action_space, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(_num_actions).repeat(config.environment.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(action_space.low).repeat(config.environment.envs, 1),
                    torch.tensor(action_space.high).repeat(config.environment.envs, 1),
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
        print(f"Logger: ({logger.step} steps).")
        return state


def make_datasets(
    config: DreamerV3Cfg, train_eps: OrderedDict, eval_eps: OrderedDict
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
    config: DreamerV3Cfg,
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
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False


def train(
    config: DreamerV3Cfg,
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

    Returns:
        None: The function doesn't return a value but saves the model during training.
    """
    # make sure eval will be executed once after config.steps
    while agent._step < config.training.steps + config.training.eval_every:
        logger.write()
        if config.training.eval_episode_num > 0:
            print("Start evaluation.")
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
            if is_image_available and config.general.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.general.traindir,
            logger,
            limit=config.training.dataset_size,
            steps=config.training.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


def main(config: DreamerV3Cfg):
    """
    Main function to set up and run the Dreamer agent training and evaluation.

    Args:
        config: Configuration object containing all necessary parameters for training and evaluation.

    Side Effects:
        - Initializes ROS node
        - Creates training and evaluation environments
        - Loads or pre-fills datasets
        - Trains and evaluates the Dreamer agent
        - Logs metrics and saves checkpoints
    """
    tools.set_seed_everywhere(config.general.seed)

    if config.general.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.general.logdir).expanduser()

    config.general.traindir = config.general.traindir or logdir / "train_eps"
    config.general.evaldir = config.general.evaldir or logdir / "eval_eps"
    config.training.steps //= config.environment.action_repeat
    config.training.eval_every //= config.environment.action_repeat
    config.general.log_every //= config.environment.action_repeat
    config.environment.time_limit //= config.environment.action_repeat
    print("Logdir", logdir)

    logdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    config.general.traindir.mkdir(parents=True, exist_ok=True)
    config.general.evaldir.mkdir(parents=True, exist_ok=True)
    step = data_tools.count_steps(config.general.traindir)

    # step in logger is environmental step
    logger = tools.Logger(logdir, config.environment.action_repeat * step)

    print("Create envs.")

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

    rospy.init_node("dreamerv3", anonymous=True)

    make = lambda id: make_arena_env(config, id)
    # make = lambda id: make_dmc_env(config, id)

    # train_envs = [make(i) for i in range(config.environment.envs)]
    # eval_envs = train_envs  # [make("eval", i) for i in range(config.envs)]

    if config.general.parallel:
        train_envs = [
            Parallel(partial(make, i), "process")
            for i in range(config.environment.envs)  # bind from functools
        ]
        eval_envs = train_envs
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    acts = train_envs[0].action_space
    print("Action Space", acts)

    _num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    _image_available = train_envs[0].observation_space.get("image", None) is not None

    state = None

    # Prefill dataset if no offline dataset is provided
    if not config.general.offline_traindir:
        prefill = max(
            0,
            config.training.prefill_steps
            - data_tools.count_steps(config.general.traindir),
        )
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(_num_actions).repeat(config.environment.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.environment.envs, 1),
                    torch.tensor(acts.high).repeat(config.environment.envs, 1),
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
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = data_tools.make_dataset(train_eps, config)
    eval_dataset = data_tools.make_dataset(eval_eps, config)

    agent = dreamer.Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.general.device)
    agent.requires_grad_(requires_grad=False)

    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.training.steps + config.training.eval_every:
        logger.write()
        if config.training.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.general.evaldir,
                logger,
                is_eval=True,
                episodes=config.training.eval_episode_num,
                no_image_key=not _image_available,
            )
            if _image_available and config.general.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.general.traindir,
            logger,
            limit=config.training.dataset_size,
            steps=config.training.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
