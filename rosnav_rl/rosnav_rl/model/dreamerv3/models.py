import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..dreamerv3 import networks, tools
from ..dreamerv3.social import dims as social_dims
from ..dreamerv3.social.dali import DALI as SocialDALI, dali_loss as social_dali_loss
from ..dreamerv3.social.gat import GAT as SocialGAT
from ..dreamerv3.social.height import HeightGAT
from ..dreamerv3.social.context import (
    SocialContextEncoder,
    context_infonce_loss as social_context_infonce_loss,
    context_kl_loss as social_context_kl_loss,
    context_pred_floor as social_context_pred_floor,
    context_pred_loss as social_context_pred_loss,
)

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
        # cSRSSM: crowd-behavior context b conditions the RSSM transition. context_size==0
        # (context disabled or social disabled) keeps img_step's cat byte-identical to baseline.
        _context_size = (
            config.model.social.context.b_dim
            if config.model.social.enabled and config.model.social.context.enabled
            else 0
        )
        # TSSM (cell_type "tssm") is a subclass with a restructured observe()/img_step();
        # it takes two extra width knobs. GRU/TransformerCell go through plain RSSM.
        _dyn_cls = (
            networks.TSSM if config.model.social.cell_type == "tssm" else networks.RSSM
        )
        _dyn_kwargs = (
            dict(
                tssm_num_layers=config.model.social.tssm_num_layers,
                tssm_ff_mult=config.model.social.tssm_ff_mult,
            )
            if config.model.social.cell_type == "tssm"
            else {}
        )
        self.dynamics = _dyn_cls(
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
            cell_type=config.model.social.cell_type,
            transformer_ctx_len=config.model.social.transformer_ctx_len,
            transformer_num_heads=config.model.social.transformer_num_heads,
            context_size=_context_size,
            **_dyn_kwargs,
        )
        self.heads = nn.ModuleDict()
        # The decoder reconstructs observations from the base latent (h, z); the reward and cont
        # heads consume the augmented feature s_hat_t = (h, z, c, d). With social disabled both
        # widths are equal, so the baseline network stays byte-identical (M0 firebreak).
        base_feat_size = social_dims.base_feat_size(config)
        aug_feat_size = social_dims.augmented_feat_size(config)
        self.heads["decoder"] = networks.MultiDecoder(
            base_feat_size,
            shapes,
            device=config.general.device,
            **config.model.decoder.model_dump(),
        )
        self.heads["reward"] = networks.MLP(
            aug_feat_size,
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
            aug_feat_size,
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
        # Pedestrian reconstruction head (M2): decode peds from base_feat so
        # GAT/DALI can run on decoded peds during imagination (decode-then-GAT invariant).
        # Key matches the obs-space class name so data["PedestrianNodeSetSpace"] is found.
        # Only instantiated when social is enabled; zero overhead for the baseline.
        if config.model.social.enabled:
            _social = config.model.social
            _ped_out_dim = _social.max_peds * (_social.node_feat_dim + 1)
            self.heads["PedestrianNodeSetSpace"] = networks.MLP(
                base_feat_size,
                (_ped_out_dim,),
                config.model.decoder.mlp_layers,
                config.model.units,
                config.model.act,
                config.model.norm,
                dist="normal",
                outscale=config.model.decoder.outscale,
                device=config.general.device,
                name="PedsDecoder",
            )
        # Social GAT (C1): instantiated here so it is part of model_opt gradient graph.
        # variant=="height" swaps in the HEIGHT-style per-edge-type attention (Part 2); same
        # interface/out_dim as GAT, so nothing downstream needs to know which one is active.
        self._social_gat: SocialGAT | None = None
        if config.model.social.enabled:
            _gat_cls = HeightGAT if config.model.social.gat.variant == "height" else SocialGAT
            self._social_gat = _gat_cls(
                cfg=config.model.social.gat,
                node_feat_dim=config.model.social.node_feat_dim,
                deter_size=config.model.dyn_deter,
                device=config.general.device,
            )
        # DALI (C2, M4): GRU dynamics context encoder + L_dyn aux loss.
        # Only instantiated when both social and dali are enabled.
        self._social_dali: SocialDALI | None = None
        if config.model.social.enabled and config.model.social.dali.enabled:
            self._social_dali = SocialDALI(
                cfg=config.model.social.dali,
                node_feat_dim=config.model.social.node_feat_dim,
                device=config.general.device,
            )
        # cSRSSM (Part 1): crowd-behavior context encoder q(b|window). Conditions the RSSM
        # transition (self.dynamics, context_size set above), not the feature.
        self._social_context: SocialContextEncoder | None = None
        if config.model.social.enabled and config.model.social.context.enabled:
            self._social_context = SocialContextEncoder(
                cfg=config.model.social.context,
                node_feat_dim=config.model.social.node_feat_dim,
                device=config.general.device,
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
            warmup_steps=config.training.warmup_steps,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.model.reward_head.loss_scale,
            cont=config.model.cont_head.loss_scale,
        )
        if config.model.social.enabled:
            self._scales["PedestrianNodeSetSpace"] = config.model.social.peds_recon_scale
        # Extended set of grad heads: static config + social head when enabled.
        # Not stored in config because the assertion "name in self.heads" would fail
        # for "PedestrianNodeSetSpace" when social.enabled=False.
        self._grad_heads: set = set(config.model.grad_heads)
        if config.model.social.enabled:
            self._grad_heads.add("PedestrianNodeSetSpace")

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

        # SE(2) global frame augmentation: random rotation + translation applied to
        # both RobotPoseSpace and PedestrianNodeSetSpace, forcing decoder equivariance.
        _se2_cfg = self._config.model.social
        if _se2_cfg.use_se2_frame_canon and _se2_cfg.se2_augment_prob > 0:
            from ..dreamerv3.se2_utils import augment_se2 as _augment_se2
            data = _augment_se2(
                data,
                max_peds=_se2_cfg.max_peds,
                node_feat_dim=_se2_cfg.node_feat_dim,
                p=_se2_cfg.se2_augment_prob,
            )

        # Invariant 5: compute per-step relative poses from anchor frame for SE(2)
        # frame canonicalization.  This is pure pose arithmetic — no grad needed.
        _pose_rel_bt = None
        if (
            _se2_cfg.use_se2_frame_canon
            and self._social_gat is not None
            and "RobotPoseSpace" in data
        ):
            with torch.no_grad():
                from ..dreamerv3.se2_utils import compute_se2_relative_poses

                _pose_rel_bt = compute_se2_relative_poses(
                    data["RobotPoseSpace"], data["is_first"]
                )

        with tools.RequiresGrad(self):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
                embed = self.encoder(data)

                # cSRSSM (Part 1): infer crowd-behavior context b once per sequence, from a
                # window of the real (ground-truth) ped node-set, before the transition scan.
                # Held fixed across T (b is slow/global per sequence) -- see social/context.py.
                context_b = None
                context_kl = embed.new_zeros(())
                context_infonce = embed.new_zeros(())
                context_pred = embed.new_zeros(())
                # Diagnostics only (never added to the loss): unscaled prediction loss, its
                # zero-velocity persistence floor, and the gap between them. A vanishing gap is
                # the context-collapse early warning (§2.1 of the proposal).
                context_pred_raw = embed.new_zeros(())
                context_pred_floor = embed.new_zeros(())
                if self._social_context is not None and "PedestrianNodeSetSpace" in data:
                    _ctx_cfg = _se2_cfg.context
                    N_c, F_c = _se2_cfg.max_peds, _se2_cfg.node_feat_dim
                    _peds_bt = data["PedestrianNodeSetSpace"]  # (B, T, N*(F+1))
                    B_c, T_c = _peds_bt.shape[:2]
                    K_c = min(_ctx_cfg.window, T_c)
                    _peds_win = _peds_bt[:, :K_c].view(B_c, K_c, N_c, F_c + 1)
                    _win_feat, _win_valid = _peds_win[..., :F_c], _peds_win[..., -1]

                    b_mean, b_std = self._social_context(_win_feat, _win_valid)
                    context_b = self._social_context.sample(b_mean, b_std, sample=True)
                    context_kl = _ctx_cfg.kl_scale * social_context_kl_loss(b_mean, b_std)

                    # Prediction-driven identifiability (primary, VariBAD-style): from
                    # (b, pooled crowd summary at step t) predict the pooled summary at t+1,
                    # for steps AFTER the inference window only -- b must extrapolate the
                    # regime, not memorize the window. See context_pred_loss docstring.
                    if _ctx_cfg.pred_scale > 0 and T_c > K_c:
                        _fut = _peds_bt[:, K_c - 1 :].view(B_c, -1, N_c, F_c + 1)
                        _fut_feat, _fut_valid = _fut[..., :F_c], _fut[..., -1]
                        M_c = _fut_feat.shape[1]
                        _pooled_fut = self._social_context.pool_step(
                            _fut_feat.reshape(B_c * M_c, N_c, F_c),
                            _fut_valid.reshape(B_c * M_c, N_c),
                        ).view(B_c, M_c, F_c)
                        _step_valid = _fut_valid.sum(-1) > 0  # (B, M)
                        context_pred_raw = social_context_pred_loss(
                            self._social_context, context_b, _pooled_fut, _step_valid
                        )
                        context_pred = _ctx_cfg.pred_scale * context_pred_raw
                        with torch.no_grad():
                            context_pred_floor = social_context_pred_floor(
                                _pooled_fut, _step_valid
                            )

                    # InfoNCE identifiability (secondary/ablation-only, see context.py):
                    # encode the two halves of the same window independently; same-sequence
                    # halves are positives. Off by default (infonce_scale: 0).
                    _half = K_c // 2
                    if _half >= 1 and _ctx_cfg.infonce_scale > 0:
                        m1, s1 = self._social_context(
                            _win_feat[:, :_half], _win_valid[:, :_half]
                        )
                        m2, s2 = self._social_context(
                            _win_feat[:, _half : 2 * _half], _win_valid[:, _half : 2 * _half]
                        )
                        b1 = self._social_context.sample(m1, s1, sample=True)
                        b2 = self._social_context.sample(m2, s2, sample=True)
                        _b_cat = torch.cat([b1, b2], dim=0)
                        _ep_id = torch.arange(B_c, device=_b_cat.device).repeat(2)
                        context_infonce = _ctx_cfg.infonce_scale * social_context_infonce_loss(
                            _b_cat, _ep_id
                        )

                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"], context=context_b
                )
                if context_b is not None:
                    # Broadcast the per-sequence b across T so it flattens correctly when
                    # `post` is later used as `start` for imagination (see ImagBehavior._imagine).
                    post["context_b"] = context_b.unsqueeze(1).expand(-1, embed.shape[1], -1)
                kl_free = self._config.model.kl_free
                dyn_scale = self._config.model.dyn_scale
                rep_scale = self._config.model.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                # Compute both feat variants once; share across head iterations.
                # Decoder and ped-reconstruction heads use base_feat (M0 firebreak).
                # Reward, cont, and any future social-aware heads use aug_feat.
                _base_feat = self.dynamics.get_feat(post)
                # Invariant 5: pass pose_rel so decoded peds are in current frame for GAT/DALI.
                _aug_feat  = (
                    self._get_augmented_feat(post, pose_rel=_pose_rel_bt)
                    if self._social_gat is not None
                    else _base_feat
                )
                _BASE_FEAT_HEADS = {"decoder", "PedestrianNodeSetSpace"}
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._grad_heads
                    feat = _base_feat if name in _BASE_FEAT_HEADS else _aug_feat
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred

                # Transform PedestrianNodeSetSpace targets to anchor frame so the
                # decoder is supervised on anchor-frame peds (Invariant 5).
                _loss_targets = data
                if _pose_rel_bt is not None and "PedestrianNodeSetSpace" in data:
                    from ..dreamerv3.se2_utils import apply_se2_to_peds_flat
                    _peds_bt = data["PedestrianNodeSetSpace"]   # (B, T, N*(F+1))
                    _B_t, _T_t = _peds_bt.shape[:2]
                    _peds_anch = apply_se2_to_peds_flat(
                        _pose_rel_bt.reshape(_B_t * _T_t, 3),
                        _peds_bt.reshape(_B_t * _T_t, -1),
                        _se2_cfg.max_peds,
                        _se2_cfg.node_feat_dim,
                    ).reshape(_B_t, _T_t, -1)
                    _loss_targets = dict(data)
                    _loss_targets["PedestrianNodeSetSpace"] = _peds_anch

                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(_loss_targets[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                # DALI auxiliary forward-prediction loss (M4.2, observe-time only).
                # Trains the DALI GRU and ped decoder on next-step ped prediction.
                # L_dyn is never computed inside imagination rollouts.
                _dali_aux = torch.zeros((), device=self._config.general.device)
                if self._social_dali is not None:
                    _soc = self._config.model.social
                    K = _soc.dali.k_steps
                    # Decoded ped sequence from the ped reconstruction head.
                    _peds_seq = preds["PedestrianNodeSetSpace"].mode()  # (B, T, N*(F+1))
                    B_s, T_s, _ = _peds_seq.shape
                    N_s = _soc.max_peds
                    F_s = _soc.node_feat_dim
                    if T_s > K:
                        _peds_seq = _peds_seq.view(B_s, T_s, N_s, F_s + 1)
                        _ped_feats = _peds_seq[..., :F_s]              # (B, T, N, F)
                        _validity  = _peds_seq[..., -1]                # (B, T, N)
                        # Build sliding K-step windows via unfold over T dimension.
                        _pf = _ped_feats.permute(0, 2, 3, 1)           # (B, N, F, T)
                        _wins = _pf.unfold(-1, K, 1)                   # (B, N, F, T-K+1, K)
                        _wins = _wins.permute(0, 3, 1, 4, 2)           # (B, T-K+1, N, K, F)
                        _val_wins = _validity[:, K - 1:]               # (B, T-K+1, N)
                        # Consecutive window pairs: traj_t → predict traj_tp1
                        _n_win = T_s - K                               # number of valid pairs
                        _traj_t   = _wins[:, :_n_win].reshape(-1, N_s, K, F_s)
                        _traj_tp1 = _wins[:, 1:_n_win + 1].reshape(-1, N_s, K, F_s)
                        _v_t      = _val_wins[:, :_n_win].reshape(-1, N_s)
                        _dali_aux = _soc.dali.lambda_dyn * social_dali_loss(
                            self._social_dali, _traj_t, _traj_tp1, _v_t
                        )
            metrics = self._model_opt(
                torch.mean(model_loss) + _dali_aux + context_kl + context_infonce + context_pred,
                self.parameters(),
            )

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["dali_aux_loss"] = to_np(_dali_aux)
        metrics["context_kl_loss"] = to_np(context_kl)
        metrics["context_infonce_loss"] = to_np(context_infonce)
        metrics["context_pred_loss"] = to_np(context_pred)
        metrics["context_pred_raw"] = to_np(context_pred_raw)
        metrics["context_pred_floor"] = to_np(context_pred_floor)
        metrics["context_pred_gap"] = to_np(context_pred_floor - context_pred_raw)
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
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

    def step_social_context(
        self, ctx_window: "dict | None", obs: dict, is_first: torch.Tensor
    ) -> "tuple[dict | None, torch.Tensor | None]":
        """Maintain the rolling K-step ped window for deploy-time context inference (Phase 1).

        Called once per real env step, before ``dynamics.obs_step``, when
        ``social.context.enabled``. Deploy uses ``b = mean`` (single, fully-realtime rollout);
        Phase-2 K-mode counterfactual planning would sample here instead.

        Args:
            ctx_window: Previous ``{"feat": (B,K,N,F), "valid": (B,K,N)}`` window, or None at
                rollout start.
            obs: Preprocessed observation dict containing "PedestrianNodeSetSpace".
            is_first: ``(B,)`` bool/float, True where a new episode just started -- those rows'
                windows are reseeded with the current frame (mirrors ``RSSM.obs_step``'s own
                is_first handling) rather than rolling across the episode boundary.

        Returns:
            (new_ctx_window, b): ``new_ctx_window`` to persist in the caller's policy state,
            ``b`` (mean of q(b|window), shape (B, b_dim)) to pass into ``obs_step``.
        """
        if self._social_context is None:
            return None, None
        _social = self._config.model.social
        N_c, F_c = _social.max_peds, _social.node_feat_dim
        K_c = _social.context.window
        peds = obs["PedestrianNodeSetSpace"].view(-1, N_c, F_c + 1)
        feat, valid = peds[..., :F_c], peds[..., -1]  # (B, N, F), (B, N)
        reset_feat = feat.unsqueeze(1).expand(-1, K_c, -1, -1).contiguous()
        reset_valid = valid.unsqueeze(1).expand(-1, K_c, -1).contiguous()

        if ctx_window is None or torch.sum(is_first) == len(is_first):
            window_feat, window_valid = reset_feat, reset_valid
        else:
            window_feat = torch.cat([ctx_window["feat"][:, 1:], feat.unsqueeze(1)], dim=1)
            window_valid = torch.cat([ctx_window["valid"][:, 1:], valid.unsqueeze(1)], dim=1)
            if torch.sum(is_first) > 0:
                m = is_first.float()[:, None, None]
                window_feat = window_feat * (1.0 - m.unsqueeze(-1)) + reset_feat * m.unsqueeze(-1)
                window_valid = window_valid * (1.0 - m) + reset_valid * m

        with torch.no_grad():
            b_mean, _ = self._social_context(window_feat, window_valid)

        return {"feat": window_feat, "valid": window_valid}, b_mean

    def _get_augmented_feat(
        self,
        state: dict,
        imag_step: int = 0,
        ped_buf: "torch.Tensor | None" = None,
        pose_rel: "torch.Tensor | None" = None,
        _return_decoded_peds: bool = False,
    ):
        """Compute ŝ_t = cat(base_feat, c_t [, d_t]) for reward/cont/actor heads.

        When ``social.enabled=False`` returns base_feat unchanged.

        At imagination step > ``imag_backprop_steps``, the ped-decoder call is
        wrapped in ``torch.no_grad()`` to cap error amplification.

        Args:
            state:               RSSM posterior/prior dict (keys: stoch, deter, …).
            imag_step:           Imagination horizon index (0 = first step or observe-time).
            ped_buf:             ``(B, N, K, F)`` rolling trajectory buffer threaded through
                                 ``_imagine``.  When provided, DALI uses this real K-step window
                                 instead of the single-frame-repeat fallback.
            pose_rel:            ``(B, 3)`` SE(2) pose of the current step relative to the anchor
                                 frame [x, y, theta].  When provided and ``use_se2_frame_canon``
                                 is True, decoded ped positions are transformed from anchor frame
                                 into the current robot frame before GAT/DALI.
            _return_decoded_peds: When True, return ``(aug_feat, decoded_peds)`` so the caller
                                 can reuse decoded_peds for ``_shift_ped_buf`` without double-decoding.

        Returns:
            aug_feat tensor, or ``(aug_feat, decoded_peds)`` when ``_return_decoded_peds=True``.
        """
        import torch as _torch
        base_feat = self.dynamics.get_feat(state)    # (B, feat) or (B, T, feat)

        if self._social_gat is None:
            if _return_decoded_peds:
                return base_feat, None
            return base_feat

        _social = self._config.model.social
        cap = _social.dali.imag_backprop_steps       # 5 by default

        # int() guards against 0-dim tensors from static_scan's torch.arange()[i].
        detach_peds = int(imag_step) > cap

        # GAT/DALI expect (B, ...) 2D leading dims.  Observe-time state has (B, T, ...)
        # shape — fold the T axis into the batch axis for all GAT/DALI calls.
        _leading = base_feat.shape[:-1]              # (B,) or (B, T)
        _needs_fold = base_feat.ndim == 3
        if _needs_fold:
            _B_orig, _T = base_feat.shape[:2]
            _BT = _B_orig * _T
            state_2d = {k: v.reshape(_BT, *v.shape[2:]) for k, v in state.items()}
            base_feat_2d = base_feat.reshape(_BT, base_feat.shape[-1])
            pose_rel_2d = pose_rel.reshape(_BT, 3) if pose_rel is not None else None
        else:
            state_2d = state
            base_feat_2d = base_feat
            pose_rel_2d = pose_rel

        with (_torch.no_grad() if detach_peds else _torch.enable_grad()):
            peds_dist = self.heads["PedestrianNodeSetSpace"](base_feat_2d)
            decoded_peds = peds_dist.mode()           # (BT, N*(F+1))

        # Invariant 5: SE(2) frame canonicalization — transform decoded peds from
        # anchor frame into current robot frame before GAT/DALI.
        if pose_rel_2d is not None and _social.use_se2_frame_canon:
            from ..dreamerv3.se2_utils import se2_inverse, apply_se2_to_peds_flat
            T_inv = se2_inverse(pose_rel_2d)
            decoded_peds = apply_se2_to_peds_flat(
                T_inv, decoded_peds, _social.max_peds, _social.node_feat_dim
            )

        c_t = self._social_gat(
            decoded_peds,
            state_2d["deter"],
            max_peds=_social.max_peds,
            node_feat_dim=_social.node_feat_dim,
        )
        parts = [base_feat_2d, c_t]

        if self._social_dali is not None:
            N = _social.max_peds
            F = _social.node_feat_dim
            BT = decoded_peds.shape[0]
            peds_2d = decoded_peds.view(BT, N, F + 1)
            validity = peds_2d[..., -1]              # (BT, N) decoded validity scores

            if ped_buf is not None:
                # Rolling-buffer path: real K-step decoded-ped trajectory.
                # Buffer was shifted before this call — contains frames [t-K, t-1].
                with (_torch.no_grad() if detach_peds else _torch.enable_grad()):
                    d_t = self._social_dali(ped_buf, validity)
            else:
                # Single-frame repeat fallback: observe-time or DALI-disabled path.
                K = _social.dali.k_steps
                ped_feats = peds_2d[..., :F]
                traj = ped_feats.unsqueeze(2).expand(-1, -1, K, -1)   # (BT, N, K, F)
                with (_torch.no_grad() if detach_peds else _torch.enable_grad()):
                    d_t = self._social_dali(traj, validity)
            parts.append(d_t)

        aug_feat = _torch.cat(parts, dim=-1)         # (BT, aug_dim)

        # Unfold back to (B, T, aug_dim) for observe-time callers.
        if _needs_fold:
            aug_feat = aug_feat.reshape(_leading + (aug_feat.shape[-1],))
            decoded_peds = decoded_peds.reshape(_leading + (decoded_peds.shape[-1],))

        if _return_decoded_peds:
            return aug_feat, decoded_peds
        return aug_feat

    def _shift_ped_buf(
        self,
        decoded_peds_flat: "torch.Tensor",
        prev_buf: "torch.Tensor",
        detach: bool = False,
    ) -> "torch.Tensor":
        """Shift K-step ped trajectory buffer: drop oldest frame, append current decoded peds.

        Args:
            decoded_peds_flat: ``(B, N*(F+1))`` tensor from ``heads["PedestrianNodeSetSpace"].mode()``.
            prev_buf:          ``(B, N, K, F)`` current K-step buffer.
            detach:            Detach new frame from grad graph (used beyond imag_backprop_steps cap).

        Returns:
            ``(B, N, K, F)`` new buffer with oldest frame dropped and current peds appended.
        """
        import torch as _torch
        _social = self._config.model.social
        N, F = _social.max_peds, _social.node_feat_dim
        B = decoded_peds_flat.shape[0]
        new_frame = decoded_peds_flat.view(B, N, F + 1)[..., :F]   # (B, N, F)
        if detach:
            new_frame = new_frame.detach()
        return _torch.cat([prev_buf[:, :, 1:, :], new_frame.unsqueeze(2)], dim=2)

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

        # cSRSSM: context_size > 0 requires context on every observe/imagine call — infer
        # b (mean, deploy-style) from the same 6-example slice used below.
        context_b = None
        if self._social_context is not None and "PedestrianNodeSetSpace" in data:
            _social = self._config.model.social
            _ctx_cfg = _social.context
            N_c, F_c = _social.max_peds, _social.node_feat_dim
            _peds_bt = data["PedestrianNodeSetSpace"][:6]
            K_c = min(_ctx_cfg.window, _peds_bt.shape[1])
            _peds_win = _peds_bt[:, :K_c].view(6, K_c, N_c, F_c + 1)
            with torch.no_grad():
                b_mean, _ = self._social_context(_peds_win[..., :F_c], _peds_win[..., -1])
            context_b = b_mean

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5], context=context_b
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init, context=context_b)
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
        # Actor and value both consume the augmented feature s_hat_t = (h, z, c, d); with social
        # disabled this equals the base feature, keeping the baseline byte-identical (M0 firebreak).
        feat_size = social_dims.augmented_feat_size(config)
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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
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
        """Simulates imagination trajectories over the given horizon.

        When DALI is enabled, threads a K-step decoded-ped trajectory buffer through
        the scan state so DALI receives a real rolling window at each step instead of
        a single-frame repeat.  The buffer is seeded from the start-state decoded peds
        (K identical copies) and shifts by one frame on every imagination step.

        Args:
            start:   RSSM state dict ``(B, T, …)`` — flattened to ``(B*T, …)`` inside.
            policy:  Actor callable taking detached aug_feat, returning action distribution.
            horizon: Number of imagination steps.

        Returns:
            (feats, states, actions) — each ``(horizon, B*T, …)``.
        """
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        # cSRSSM: b was attached to `post`/`start` broadcast over T so it flattens alongside
        # the state; pop it out here and close over it (constant across the horizon), since
        # img_step's returned state dict never carries it forward (unlike ped_buf/pose_accum).
        _context_b = start.pop("context_b", None)
        # TSSM: the per-layer K/V cache must be *threaded*, not stacked — static_scan
        # pre-allocates a (horizon, ...) output buffer per state key, which for the
        # (B*T, ctx_len, layers*2*deter) cache would multiply imagination memory by the
        # horizon. The scan is strictly sequential, so carry the cache in a closure box:
        # merged into the state before img_step, popped from its output after.
        _tssm_keys = ("tssm_cache", "tssm_cnt")
        _tssm_box = (
            {k: start.pop(k) for k in _tssm_keys} if "tssm_cache" in start else None
        )

        wm = self._world_model
        _social = wm._config.model.social
        _use_dali_buf = wm._social_dali is not None
        _use_se2 = _social.use_se2_frame_canon and wm._social_gat is not None
        _cap = _social.dali.imag_backprop_steps if _use_dali_buf else 5

        if _use_dali_buf:
            # Seed the K-step buffer: decode peds from the (flattened) start state,
            # then repeat the single frame K times to fill the initial window.
            _N, _F, _K = _social.max_peds, _social.node_feat_dim, _social.dali.k_steps
            _BF = start["deter"].shape[0]
            with torch.no_grad():
                _s_base = dynamics.get_feat(start)                        # (BF, base_feat)
                _s_peds = wm.heads["PedestrianNodeSetSpace"](_s_base).mode()  # (BF, N*(F+1))
            _s_feats = _s_peds.view(_BF, _N, _F + 1)[..., :_F]           # (BF, N, F)
            init_buf = _s_feats.unsqueeze(2).expand(-1, -1, _K, -1).contiguous()  # (BF, N, K, F)

        if _use_se2:
            from ..dreamerv3.se2_utils import integrate_se2 as _integrate_se2
            _BF = start["deter"].shape[0]
            _dev = start["deter"].device
            init_pose = torch.zeros(_BF, 3, device=_dev)  # identity: anchor = start frame
            if _social.action_holonomic:
                _action_scale = torch.tensor(
                    [_social.action_scale_linear, _social.action_scale_linear_y, _social.action_scale_angular],
                    device=_dev,
                )
            else:
                _action_scale = torch.tensor(
                    [_social.action_scale_linear, _social.action_scale_angular],
                    device=_dev,
                )
            _dt = _social.kinematics_dt
            _holonomic = _social.action_holonomic

        def step(prev, t):
            if _use_dali_buf and _use_se2:
                state, _, _, ped_buf, pose_accum = prev
            elif _use_dali_buf:
                state, _, _, ped_buf = prev
                pose_accum = None
            elif _use_se2:
                state, _, _, pose_accum = prev
                ped_buf = None
            else:
                state, _, _ = prev
                ped_buf = None
                pose_accum = None

            # Compute augmented feat.  When DALI buffer is active, also return decoded_peds
            # so _shift_ped_buf can reuse them without a second decoder forward pass.
            # Invariant 5: pass pose_accum (= P_k) so decoded peds are rotated to current frame.
            if _use_dali_buf:
                feat, decoded_peds = wm._get_augmented_feat(
                    state, imag_step=t, ped_buf=ped_buf, pose_rel=pose_accum,
                    _return_decoded_peds=True
                )
                detach = int(t) > _cap
                new_buf = wm._shift_ped_buf(decoded_peds, ped_buf, detach=detach)
            else:
                feat = wm._get_augmented_feat(state, imag_step=t, pose_rel=pose_accum)

            inp = feat.detach()
            action = policy(inp).sample()
            if _tssm_box is not None:
                # Re-attach the threaded K/V cache (stripped from succ below so
                # static_scan never stacks it over the horizon).
                state = {**state, **_tssm_box}
            if self._config.model.behavior.use_imagination_checkpointing:
                succ = torch.utils.checkpoint.checkpoint(
                    lambda s, a: dynamics.img_step(s, a, context=_context_b),
                    state,
                    action,
                    use_reentrant=False,
                )
            else:
                succ = dynamics.img_step(state, action, context=_context_b)
            if _tssm_box is not None:
                for _k in _tssm_keys:
                    _tssm_box[_k] = succ.pop(_k)

            # Integrate SE(2) pose for next step: P_{t+1} = compose(P_t, delta(action_t)).
            if _use_se2:
                new_pose = _integrate_se2(pose_accum, action, _action_scale, dt=_dt, holonomic=_holonomic)

            if _use_dali_buf and _use_se2:
                return succ, feat, action, new_buf, new_pose
            elif _use_dali_buf:
                return succ, feat, action, new_buf
            elif _use_se2:
                return succ, feat, action, new_pose
            else:
                return succ, feat, action

        if _use_dali_buf and _use_se2:
            succ, feats, actions, _ped_bufs, _poses = tools.static_scan(
                step, [torch.arange(horizon)], (start, None, None, init_buf, init_pose)
            )
        elif _use_dali_buf:
            succ, feats, actions, _ped_bufs = tools.static_scan(
                step, [torch.arange(horizon)], (start, None, None, init_buf)
            )
        elif _use_se2:
            succ, feats, actions, _poses = tools.static_scan(
                step, [torch.arange(horizon)], (start, None, None, init_pose)
            )
        else:
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
            # cont head expects aug_feat (same size as reward/actor); use imag_feat directly.
            discount = (
                self._config.model.behavior.discount
                * self._world_model.heads["cont"](imag_feat).mean
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
        else:
            adv = target - base

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
