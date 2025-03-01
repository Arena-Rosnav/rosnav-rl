import os
import pathlib
import sys
from typing import Generator

sys.path.append(str(pathlib.Path(__file__).parent))

import rosnav_rl.model.dreamerv3.exploration as expl
import gymnasium
import models
import numpy as np
import rosnav_rl.model.dreamerv3.tools as tools
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfg import DreamerV3Cfg

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(
        self,
        obs_space: gymnasium.Space,
        act_space: gymnasium.Space,
        config: DreamerV3Cfg,
        logger: tools.Logger,
        dataset: Generator,
    ):
        """Initialize the Dreamer agent.

        Args:
            obs_space: Observation space of the environment
            act_space: Action space of the environment
            config: Configuration object containing all hyperparameters
            logger: Logger object for tracking metrics and training progress
            dataset: Dataset object for storing and sampling experiences

        The agent consists of three main components:
        1. World Model: Learns to model the environment dynamics
        2. Task Behavior: Learns to optimize task-specific rewards
        3. Exploration Behavior: Implements different exploration strategies

        The agent uses various tracking tools to manage:
        - Logging frequency
        - Training frequency
        - Pretraining
        - Environment resets
        - Exploration duration

        Attributes:
            _wm (WorldModel): World model for environment dynamics prediction
            _task_behavior (ImagBehavior): Policy for task optimization
            _expl_behavior: Exploration strategy (random, greedy, or plan2explore)
        """

        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.general.log_every)
        batch_steps = config.training.batch_size * config.training.batch_length
        self._should_train = tools.Every(batch_steps / config.training.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.environment.reset_every)
        self._should_expl = tools.Until(
            int(config.model.exploration.until / config.environment.action_repeat)
        )
        self._metrics = {}

        # this is update step
        self._step = logger.step // config.environment.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm, act_space)

        self._obs_space = obs_space
        self._act_space = act_space

        if (
            config.general.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        def reward(f, s, a):
            """
            Calculate reward based on world model predictions.

            Args:
                f: Feature/latent representation from the world model
                s: State representation (unused)
                a: Action representation (unused)

            Returns:
                float: Mean reward prediction from the world model's reward head
            """
            return self._wm.heads["reward"](f).mean()

        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(
                config, self._wm, reward, self._act_space
            ),
        )[config.model.exploration.behavior]().to(self._config.general.device)

    def __call__(self, obs, reset, state=None, training=True):
        """Executes the main training and inference loop of the agent.

        This method handles both training and inference modes, managing the training steps,
        logging of metrics, and policy execution.

        Args:
            obs: The observation from the environment.
            reset: Boolean tensor indicating which episodes have reset.
            state: Optional; The previous state of the policy. Defaults to None.
            training (bool): Whether to run in training mode. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - policy_output: The action and other outputs from the policy
                - state: The updated state of the policy

        Side Effects:
            - Updates internal step counter
            - Performs training steps if in training mode
            - Logs metrics and videos if configured
            - Updates logger step count
        """
        step = self._step
        if training:
            steps = (
                self._config.training.pretrain_steps
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.general.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.environment.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        """
        Executes the policy to determine the next action based on current observation and state.

        Args:
            obs (dict): Current observation from the environment
            state (tuple, optional): Previous latent state and action. Defaults to None.
            training (bool): Whether the policy is being used for training or evaluation.

        Returns:
            tuple: Contains:
                - dict: Policy output with:
                    - action (torch.Tensor): Selected action
                    - logprob (torch.Tensor): Log probability of the selected action
                - tuple: Updated state containing (latent, action)

        Details:
            - Preprocesses observation and encodes it
            - Updates latent state using world model dynamics
            - Selects action based on:
                - Evaluation mode: Uses mode of task behavior actor
                - Training mode:
                    - Exploration: Samples from exploration behavior actor
                    - Otherwise: Samples from task behavior actor
            - Handles special case for onehot_gumble action distribution
        """
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.model.behavior.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.model.actor.dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1),
                (
                    self._act_space.n
                    if hasattr(self._act_space, "n")
                    else self._act_space.shape[0]
                ),
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        """
        Trains the Dreamer agent using provided data.

        This method performs the main training loop for the Dreamer agent, including:
        1. Training the world model
        2. Training the task behavior (policy)
        3. Training the exploration behavior (if not greedy)

        Args:
            data: Training data containing observations, actions, and rewards

        Returns:
            None. Updates internal metrics dictionary with training statistics.

        Side Effects:
            - Updates the world model
            - Updates the task behavior policy
            - Updates the exploration behavior (if not greedy)
            - Accumulates metrics in self._metrics

        Note:
            The method tracks various training metrics for each component and stores them
            in the internal metrics dictionary, appending new values to existing metric lists.
        """
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post

        def reward(f, s, a):
            """
            Calculate the reward based on feature, state, and action.

            Args:
                f: Feature tensor
                s: State tensor
                a: Action tensor

            Returns:
                torch.Tensor: The predicted reward from the world model's reward head,
                using the mode of the distribution.
            """
            return self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()

        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.model.exploration.behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
