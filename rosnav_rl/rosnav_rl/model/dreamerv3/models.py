import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..dreamerv3 import networks, tools

to_np = lambda x: x.detach().cpu().numpy()

if TYPE_CHECKING:
    from ..dreamerv3 import cfg
    

class RewardEMA:
    """Reward Exponential Moving Average (EMA) normalization class.
    
    This class computes the quantiles of reward values and uses them to normalize rewards
    through an exponential moving average approach. It helps stabilize learning by adaptively
    scaling rewards based on their distribution.
    
    Attributes:
        device (torch.device): The device where the tensors are stored.
        alpha (float): The smoothing factor for the exponential moving average (default: 1e-2).
        range (torch.Tensor): A tensor containing the quantile values [0.05, 0.95] used for normalization.
    """

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        """
        Compute the quantile of the input tensor and update the exponential moving average (EMA) values in-place.

        Args:
            x (torch.Tensor): Input tensor.
            ema_vals (torch.Tensor): Tensor containing the EMA values to be updated.

        Returns:
            tuple: A tuple containing the offset and scale values, both detached from the computation graph.
        """
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    """World model that combines neural network components for prediction and planning in reinforcement learning.
    
    This class implements a world model that integrates encoders, decoders, dynamics models, and prediction heads
    to learn an internal representation of the environment. It serves as the foundation for model-based 
    reinforcement learning algorithms in the DreamerV3 framework.
    
        encoder (MultiEncoder): Encodes observations into latent embeddings.
        embed_size (int): Size of the encoder's output embedding.
        dynamics (RSSM): Recurrent state-space model for predicting state transitions.
        heads (nn.ModuleDict): Dictionary of prediction heads including:
            - decoder: Reconstructs observations from latent states
            - reward: Predicts expected rewards
            - cont: Predicts continuation probability (1-termination)
        _step (int): Current training step.
        _use_amp (bool): Whether to use automatic mixed precision.
        _config (DreamerV3Cfg): Configuration object with model hyperparameters.
    
    Methods:
        _train(data): Trains the model on a batch of data, computing losses and updating parameters.
        preprocess(obs): Preprocesses observations for model input.
        video_pred(data): Generates video prediction based on input data for visualization.
    """
    
    def __init__(self, obs_space, act_space, step, config: "cfg.DreamerV3Cfg"):
        """Initialize the WorldModel class.

        This class represents a world model that combines various neural network components
        for prediction and planning in reinforcement learning.

        Args:
            obs_space (gym.spaces.Dict): Observation space containing the shapes of observations.
            act_space (gym.spaces.Box): Action space defining the range and dimension of actions.
            step (int): Current training step.
            config (Config): Configuration object containing model hyperparameters including:
                - precision: Numerical precision (16 or 32 bit)
                - encoder: Parameters for the MultiEncoder
                - decoder: Parameters for the MultiDecoder
                - dyn_stoch: Size of stochastic state
                - dyn_deter: Size of deterministic state
                - dyn_hidden: Size of hidden layers in dynamics
                - dyn_rec_depth: Depth of recurrent network
                - dyn_discrete: Whether to use discrete or continuous state space
                - various other architecture and training parameters

        Attributes:
            encoder (MultiEncoder): Encodes observations into embeddings.
            dynamics (RSSM): Recurrent state-space model for dynamics prediction.
            heads (nn.ModuleDict): Dictionary containing decoder, reward and continuation heads.
            _model_opt (Optimizer): Optimizer for the model parameters.
            _scales (dict): Loss scaling factors for different model components.
        """

        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.general.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(
            shapes, device=config.general.device, **config.model.encoder.model_dump()
        )
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.model.dyn_stoch,
            config.model.dyn_deter,
            config.model.dyn_hidden,
            config.model.dyn_rec_depth,
            config.model.dyn_discrete,
            config.model.act,
            config.model.norm,
            config.model.dyn_mean_act,
            config.model.dyn_std_act,
            config.model.dyn_min_std,
            config.model.unimix_ratio,
            config.model.initial,
            (act_space.n if hasattr(act_space, "n") else act_space.shape[0]),
            self.embed_size,
            config.general.device,
        )
        self.heads = nn.ModuleDict()
        if config.model.dyn_discrete:
            feat_size = (
                config.model.dyn_stoch * config.model.dyn_discrete
                + config.model.dyn_deter
            )
        else:
            feat_size = config.model.dyn_stoch + config.model.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size,
            shapes,
            device=config.general.device,
            **config.model.decoder.model_dump(),
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.model.reward_head.dist == "symlog_disc" else (),
            config.model.reward_head.layers,
            config.model.units,
            config.model.act,
            config.model.norm,
            dist=config.model.reward_head.dist,
            outscale=config.model.reward_head.outscale,
            device=config.general.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.model.cont_head.layers,
            config.model.units,
            config.model.act,
            config.model.norm,
            dist="binary",
            outscale=config.model.cont_head.outscale,
            device=config.general.device,
            name="Cont",
        )
        for name in config.model.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.training.model_lr,
            config.training.opt_eps,
            config.training.grad_clip,
            config.model.weight_decay,
            opt=config.training.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.model.reward_head.loss_scale,
            cont=config.model.cont_head.loss_scale,
        )

    def _train(self, data):
        """
        Train the model on a batch of data.

        This method processes a batch of data through the model's components (encoder, dynamics, heads),
        calculates various losses including KL divergence, and performs optimization.

        Args:
            data (dict): A dictionary containing:
                - action: Actions tensor of shape (batch_size, batch_length, act_dim)
                - image: Image tensor of shape (batch_size, batch_length, h, w, ch)
                - reward: Reward tensor of shape (batch_size, batch_length)
                - discount: Discount tensor of shape (batch_size, batch_length)
                - is_first: Binary tensor indicating start of episodes
                - Additional keys corresponding to prediction heads

        Returns:
            tuple:
                - post (dict): Posterior state distributions with detached tensors
                - context (dict): Dictionary containing:
                    - embed: Encoded observations
                    - feat: Features from posterior states
                    - kl: KL divergence values
                    - postent: Posterior distribution entropy
                - metrics (dict): Training metrics including:
                    - Various loss values for each prediction head
                    - KL divergence statistics
                    - Entropy values for prior and posterior distributions
                    - Scale values for KL, dynamics, and representation losses
        """

        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.model.kl_free
                dyn_scale = self._config.model.dyn_scale
                rep_scale = self._config.model.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.model.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        """
        Preprocesses the observation dictionary by converting values to tensors and normalizing data.

        Args:
            obs (dict): Dictionary containing observation data with keys like 'image', 'discount',
                       'is_first', 'is_terminal', etc.

        Returns:
            dict: Processed observation dictionary with:
                - All values converted to torch tensors on specified device
                - Image values normalized to [0,1] range
                - Discount factor applied if present
                - Continuation signal ('cont') computed from terminal states
                - Discount values expanded with additional dimension if present

        Raises:
            AssertionError: If 'is_first' or 'is_terminal' keys are missing from observation dict

        Note:
            - 'is_first' is required for hidden state initialization during training
            - 'is_terminal' is required for continuation head training
        """

        obs = {
            k: torch.tensor(v, device=self._config.general.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.model.behavior.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        """
        Generate video prediction based on input data.

        Args:
            data (dict): A dictionary containing the input data with the following keys:
                - "image" (torch.Tensor): The input images.
                - "action" (torch.Tensor): The actions taken.
                - "is_first" (torch.Tensor): A tensor indicating the first step in a sequence.

        Returns:
            torch.Tensor: A tensor containing the concatenated truth, model prediction, and error images.
        """
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    """ImagBehavior is a neural network module for imagination-based behavior generation in the DreamerV3 model.

    This class implements a behavior model that uses world model predictions to generate actions and 
    estimate values through imagination-based planning. The model consists of actor and value networks 
    that are trained using imagined trajectories from the world model.

    Attributes:
        actor (MLP): Policy network that maps latent states to action distributions
        value (MLP): Value network that estimates state values
        _slow_value (MLP): Target value network for stable learning (if enabled)
        _actor_opt (Optimizer): Optimizer for the actor network
        _value_opt (Optimizer): Optimizer for the value network
        ema_vals (torch.Tensor): Buffer for exponential moving average values if reward_EMA is enabled
        reward_ema (RewardEMA): Reward normalization using exponential moving average

    Methods:
        _train(start, objective): Train actor and value networks using imagined trajectories
        _imagine(start, policy, horizon): Simulate trajectories using the world model
        _compute_target(imag_feat, imag_state, reward): Compute target values using lambda returns
        _compute_actor_loss(imag_feat, imag_action, target, weights, base): Calculate actor loss
        _update_slow_target(): Update parameters of the slow target value network
    """    
        
    def __init__(self, config: "cfg.DreamerV3Cfg", world_model, act_space):
        """
        Initialize the ImagBehavior class for imagination-based behavior generation.

        This class implements a behavior model that uses world model predictions to generate actions
        and estimate values in an imagination-based planning setting.

        Args:
            config: Configuration object containing model parameters including:
                - precision: Model precision (16 or 32 bit)
                - dyn_discrete: Boolean for discrete dynamics
                - dyn_stoch: Stochastic dimension size
                - dyn_deter: Deterministic dimension size
                - num_actions: Number of action dimensions
                - actor: Dict containing actor network parameters
                - critic: Dict containing critic network parameters
                - weight_decay: Weight decay for optimization
                - opt: Optimizer type
                - device: Computing device
                - reward_EMA: Boolean for using reward exponential moving average
            world_model: World model instance for environment dynamics prediction

        Attributes:
            actor (MLP): Actor network for policy generation
            value (MLP): Value network for state value estimation
            _slow_value: Copy of value network for slow target updates (if enabled)
            _actor_opt (Optimizer): Optimizer for actor network
            _value_opt (Optimizer): Optimizer for value network
            ema_vals (torch.Tensor): Buffer for EMA values if reward_EMA is enabled
            reward_ema (RewardEMA): Reward EMA calculator if enabled
        """

        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.general.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.model.dyn_discrete:
            feat_size = (
                config.model.dyn_stoch * config.model.dyn_discrete
                + config.model.dyn_deter
            )
        else:
            feat_size = config.model.dyn_stoch + config.model.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            ((act_space.n if hasattr(act_space, "n") else act_space.shape[0]),),
            config.model.actor.layers,
            config.model.units,
            config.model.act,
            config.model.norm,
            config.model.actor.dist,
            config.model.actor.std,
            config.model.actor.min_std,
            config.model.actor.max_std,
            absmax=1.0,
            temp=config.model.actor.temp,
            unimix_ratio=config.model.actor.unimix_ratio,
            outscale=config.model.actor.outscale,
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.model.critic.dist == "symlog_disc" else (),
            config.model.critic.layers,
            config.model.units,
            config.model.act,
            config.model.norm,
            config.model.critic.dist,
            outscale=config.model.critic.outscale,
            device=config.general.device,
            name="Value",
        )
        if config.model.critic.slow_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(
            wd=config.model.weight_decay,
            opt=config.training.opt,
            use_amp=self._use_amp,
        )
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.model.actor.lr,
            config.model.actor.eps,
            config.model.actor.grad_clip,
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.model.critic.lr,
            config.model.critic.eps,
            config.model.critic.grad_clip,
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.environment.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.general.device)
            )
            self.reward_ema = RewardEMA(device=self._config.general.device)

    def _train(
        self,
        start,
        objective,
    ):
        """Train the actor and value networks using imagined trajectories.

        Args:
            start: Initial state from which to start imagination
            objective: Function that computes rewards from imagined features, states, and actions

        Returns:
            tuple: Contains:
                - imag_feat (torch.Tensor): Imagined features
                - imag_state (torch.Tensor): Imagined states
                - imag_action (torch.Tensor): Imagined actions
                - weights (torch.Tensor): Importance weights for loss computation
                - metrics (dict): Training metrics including:
                    - Value statistics
                    - Target statistics
                    - Imagined reward statistics
                    - Imagined action statistics
                    - Actor entropy
                    - Optimization metrics

        The method performs the following steps:
        1. Updates slow target network
        2. Imagines trajectories using current actor
        3. Computes actor loss using imagined trajectories
        4. Computes value loss using imagined trajectories
        5. Updates both actor and value networks using their respective optimizers
        """
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.model.behavior.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= (
                    self._config.model.actor.entropy * actor_ent[:-1, ..., None]
                )
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.model.critic.slow_target:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.model.actor.dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        """
        Simulates the imagination process over a given horizon using the provided policy.

        Args:
            start (dict): A dictionary containing the initial states.
            policy (callable): A policy function that takes in features and returns an action distribution.
            horizon (int): The number of steps to simulate.

        Returns:
            tuple: A tuple containing:
                - feats (torch.Tensor): The features at each step of the imagination.
                - states (dict): A dictionary of states at each step.
                - actions (torch.Tensor): The actions taken at each step.
        """
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        """
        Compute the target values for training.

        This function calculates the target values used for training the model. It
        computes the discount factor based on whether the world model has a continuous
        head or not. It then calculates the value and target using lambda return.

        Args:
            imag_feat (torch.Tensor): Imagined features from the model.
            imag_state (torch.Tensor): Imagined states from the model.
            reward (torch.Tensor): Rewards obtained from the environment.

        Returns:
            tuple: A tuple containing:
                - target (torch.Tensor): The computed target values.
                - weights (torch.Tensor): The weights for each step.
                - value (torch.Tensor): The value predictions for each step.
        """
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = (
                self._config.model.behavior.discount
                * self._world_model.heads["cont"](inp).mean
            )
        else:
            discount = self._config.model.behavior.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.model.behavior.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        """
        Compute the actor loss for the DreamerV3 model.

        Args:
            imag_feat (torch.Tensor): Imagined features from the model.
            imag_action (torch.Tensor): Imagined actions taken by the model.
            target (list of torch.Tensor): Target values for the actor.
            weights (torch.Tensor): Weights for the loss calculation.
            base (torch.Tensor): Baseline values for advantage calculation.

        Returns:
            tuple: A tuple containing:
                - actor_loss (torch.Tensor): The computed actor loss.
                - metrics (dict): A dictionary of metrics for monitoring training.
        """
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.environment.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.model.behavior.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.model.behavior.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.model.behavior.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.model.behavior.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.model.behavior.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        """
        Update the parameters of the slow target network.

        This method updates the parameters of the slow target network based on the
        parameters of the current value network. The update is performed using a
        weighted average of the current parameters and the slow target parameters,
        controlled by the 'slow_target_fraction' configuration parameter.

        The update is performed only if the 'slow_target' configuration parameter
        is set to True and the number of updates is a multiple of the
        'slow_target_update' configuration parameter.

        The method increments the update counter after performing the update.

        Configuration parameters:
            - critic["slow_target"]: Boolean indicating whether to use slow target updates.
            - critic["slow_target_update"]: Integer specifying the frequency of slow target updates.
            - critic["slow_target_fraction"]: Float specifying the fraction of the current parameters
            to use in the weighted average update.

        Attributes:
            - self._config: Configuration object containing the parameters for the update.
            - self._updates: Counter for the number of updates performed.
            - self.value: Current value network.
            - self._slow_value: Slow target value network.
        """
        if self._config.model.critic.slow_target:
            if self._updates % self._config.model.critic.slow_target_update == 0:
                mix = self._config.model.critic.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
