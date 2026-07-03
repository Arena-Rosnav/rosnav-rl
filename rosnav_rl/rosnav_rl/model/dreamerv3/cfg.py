from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal

from rosnav_rl.cfg.framework import FrameworkCfg
from rosnav_rl.utils.type_aliases.rl_frameworks import SupportedRLFrameworks


class GeneralCfg(BaseModel):
    """Configuration for general training parameters.

    This class defines general configuration parameters used across the training process.

    Attributes:
        logdir (Optional[str]): Directory path for storing logs.
        traindir (Optional[str]): Directory path for training data.
        evaldir (Optional[str]): Directory path for evaluation data.
        offline_traindir (str): Directory path for offline training data. Defaults to empty string.
        offline_evaldir (str): Directory path for offline evaluation data. Defaults to empty string.
        seed (int): Random seed for reproducibility. Defaults to 0.
        deterministic_run (bool): Whether to enforce deterministic behavior. Defaults to False.
        steps (float): Total number of training steps. Defaults to 1e6.
        parallel (bool): Whether to use parallel processing. Defaults to False.
        log_every (float): Frequency of logging in steps. Defaults to 1e4.
        device (str): Computing device to use ('cpu' or 'cuda'). Defaults to "cpu".
        compile (bool): Whether to compile the model. Defaults to True.
        precision (int): Numerical precision in bits. Defaults to 32. Use 16 to enable bfloat16 AMP (recommended over float16 to avoid overflow in RSSM logits).
        debug (bool): Whether to enable debug mode. Defaults to False.
        video_pred_log (bool): Whether to log video predictions. Defaults to True.
    """

    logdir: Optional[Union[str, Path]] = None  # Set by trainer; override for custom path
    traindir: Optional[Union[str, Path]] = None
    evaldir: Optional[Union[str, Path]] = None
    offline_traindir: Union[str, Path] = ""
    offline_evaldir: Union[str, Path] = ""
    seed: int = 0
    deterministic_run: bool = False
    parallel: bool = True
    log_every: float = 1e4
    device: str = "cpu"
    compile: bool = True
    precision: int = 32
    debug: bool = False
    video_pred_log: bool = False


class EnvironmentCfg(BaseModel):
    """Configuration class for environment settings.

    This class defines the basic environment configuration parameters used in training.

    Attributes:
        envs (int): Number of parallel environments to run. Default is 1.
        reset_every (int): Steps interval for environment reset. 0 means no reset. Defaults to 0.
        action_repeat (int): Number of times to repeat each action per step taken in the environment. Default is 1.
        time_limit (int): Maximum number of steps per episode. Default is 1000.
        grayscale (bool): Whether to convert observations to grayscale. Default is False.
        prefill_steps (int): Number of random steps to take before training begins. Default is 2500.
        reward_EMA (bool): Whether to use Exponential Moving Average for rewards. Default is True.
    """

    reset_every: int = 0
    action_repeat: int = 1
    grayscale: bool = False
    reward_EMA: bool = True


class EncoderCfg(BaseModel):
    """Configuration class for the Encoder module.

    This class defines the configuration parameters for the encoder network architecture,
    which can process both MLP and CNN inputs.

    Attributes:
        mlp_keys (str): Observation Keys from EncodedObsDict considered for MLP inputs. Default: "$^" (empty pattern)
        cnn_keys (str): Observation Keys from EncodedObsDict considered for CNN inputs. Default: "image"
        act (str): Activation function to use. Default: "SiLU"
        norm (bool): Whether to use normalization. Default: True
        cnn_depth (int): Number of channels in CNN layers. Default: 32
        kernel_size (int): Size of CNN kernels. Default: 4
        minres (int): Minimum resolution for CNN processing. Default: 4
        mlp_layers (int): Number of layers in MLP. Default: 5
        mlp_units (int): Number of units per MLP layer. Default: 1024
        symlog_inputs (bool): Whether to apply symmetric log transformation to inputs. Default: True
    """

    mlp_keys: str = "DIST_ANGLE_TO_SUBGOAL|PedestrianNodeSetSpace"
    cnn_keys: str = (
        "PEDESTRIAN_SOCIAL_STATE|PEDESTRIAN_TYPE|PEDESTRIAN_VEL_X|PEDESTRIAN_VEL_Y|STACKED_LASER_MAP"
    )
    act: str = "SiLU"
    norm: bool = True
    cnn_depth: int = 1
    kernel_size: int = 4
    minres: int = 5
    mlp_layers: int = 5
    mlp_units: int = 128
    symlog_inputs: bool = True
    is_channels_first: bool = True


class DecoderCfg(BaseModel):
    """Configuration class for the Decoder network architecture.

    This class defines the configuration parameters for a decoder network that can process
    both CNN-based (image) and MLP-based (vector) inputs.

    Attributes:
        mlp_keys (str): Observation Keys from EncodedObsDict considered for MLP inputs. Default: "$^" (empty pattern)
        cnn_keys (str): Observation Keys from EncodedObsDict considered for CNN inputs. Default: "image"
        act (str): Activation function, default "SiLU".
        norm (bool): Whether to use normalization, default True.
        cnn_depth (int): Base depth for CNN layers, default 32.
        kernel_size (int): Kernel size for CNN layers, default 4.
        minres (int): Minimum resolution for CNN processing, default 4.
        mlp_layers (int): Number of layers in MLP, default 5.
        mlp_units (int): Number of units per MLP layer, default 1024.
        cnn_sigmoid (bool): Whether to use sigmoid activation in CNN output, default False.
        image_dist (str): Distribution type for image outputs, default "mse".
        vector_dist (str): Distribution type for vector outputs, default "symlog_mse".
        outscale (float): Output scaling factor, default 1.0.
    """

    mlp_keys: str = "DIST_ANGLE_TO_SUBGOAL"
    cnn_keys: str = (
        "PEDESTRIAN_SOCIAL_STATE|PEDESTRIAN_TYPE|PEDESTRIAN_VEL_X|PEDESTRIAN_VEL_Y|STACKED_LASER_MAP"
    )
    act: str = "SiLU"
    norm: bool = True
    cnn_depth: int = 1
    kernel_size: int = 4
    minres: int = 5
    mlp_layers: int = 5
    mlp_units: int = 128
    cnn_sigmoid: bool = False
    image_dist: str = "mse"
    vector_dist: str = "symlog_mse"
    outscale: float = 1.0
    is_channels_first: bool = True


class ActorCfg(BaseModel):
    """Configuration for the actor network.

    Attributes:
        layers (int): Number of layers in the actor network.
        dist (str): Distribution type for action sampling.
        entropy (float): Entropy regularization coefficient.
        unimix_ratio (float): Ratio for mixing uniform distribution.
        std (str): Type of standard deviation ('learned' or fixed).
        min_std (float): Minimum standard deviation.
        max_std (float): Maximum standard deviation.
        temp (float): Temperature parameter for action sampling.
        lr (float): Learning rate.
        eps (float): Epsilon for optimizer stability.
        grad_clip (float): Gradient clipping threshold.
        outscale (float): Output scaling factor.
    """

    layers: int = 1
    dist: str = "normal"
    entropy: float = 3e-4
    unimix_ratio: float = 0.01
    std: str = "learned"
    min_std: float = 0.1
    max_std: float = 1.0
    temp: float = 0.1
    lr: float = 3e-5
    eps: float = 1e-5
    grad_clip: float = 100.0
    outscale: float = 1.0


class CriticCfg(BaseModel):
    """Configuration for the critic network.

    Attributes:
        layers (int): Number of layers in the critic network.
        dist (str): Distribution type for value estimation.
        slow_target (bool): Whether to use slow-moving target network.
        slow_target_update (int): Update frequency for target network.
        slow_target_fraction (float): Update fraction for target network.
        lr (float): Learning rate.
        eps (float): Epsilon for optimizer stability.
        grad_clip (float): Gradient clipping threshold.
        outscale (float): Output scaling factor.
    """

    layers: int = 1
    dist: str = "symlog_disc"
    slow_target: bool = True
    slow_target_update: int = 1
    slow_target_fraction: float = 0.02
    lr: float = 3e-5
    eps: float = 1e-5
    grad_clip: float = 100.0
    outscale: float = 0.0


class RewardHeadCfg(BaseModel):
    """Configuration for the reward prediction head.

    Attributes:
        layers (int): Number of layers in the reward network.
        dist (str): Distribution type for reward prediction.
        loss_scale (float): Scaling factor for reward loss.
        outscale (float): Output scaling factor.
    """

    layers: int = 1
    dist: str = "symlog_disc"
    loss_scale: float = 1.0
    outscale: float = 0.0


class ContHeadCfg(BaseModel):
    """Configuration for the continuation head.

    Attributes:
        layers (int): Number of layers in the continuation network.
        loss_scale (float): Scaling factor for continuation loss.
        outscale (float): Output scaling factor.
    """

    layers: int = 1
    loss_scale: float = 1.0
    outscale: float = 1.0


class ExplorationCfg(BaseModel):
    """Configuration for exploration and disagreement models.

    Attributes:
        behavior (str): Type of exploration behavior.
        until (int): Steps to explore.
        extr_scale (float): Extrinsic exploration reward scale.
        intr_scale (float): Intrinsic exploration reward scale.
        disag_target (str): Disagreement model target.
        disag_log (bool): Enable disagreement logging.
        disag_models (int): Number of disagreement models.
        disag_offset (int): Disagreement model offset.
        disag_layers (int): Layers in disagreement models.
        disag_units (int): Units per disagreement layer.
        disag_action_cond (bool): Action-conditioned disagreement.
    """

    behavior: str = "greedy"
    until: int = 0
    extr_scale: float = 0.0
    intr_scale: float = 1.0
    disag_target: str = "stoch"
    disag_log: bool = True
    disag_models: int = 10
    disag_offset: int = 1
    disag_layers: int = 4
    disag_units: int = 400
    disag_action_cond: bool = False


class BehaviorCfg(BaseModel):
    """Configuration for agent behavior settings.

    Attributes:
        discount (float): Future reward discount factor.
        discount_lambda (float): Lambda discount parameter.
        imag_horizon (int): Imagination horizon length.
        imag_gradient (str): Imagination gradient type.
        imag_gradient_mix (float): Gradient mixing ratio.
        eval_state_mean (bool): Use mean state in evaluation.
        use_imagination_checkpointing (bool): Wrap img_step in gradient checkpointing to save VRAM. Default: False
    """

    discount: float = 0.997
    discount_lambda: float = 0.95
    imag_horizon: int = 15
    imag_gradient: str = "dynamics"
    imag_gradient_mix: float = 0.0
    eval_state_mean: bool = False
    use_imagination_checkpointing: bool = False


class SocialGATCfg(BaseModel):
    """Configuration for the Social-RSSM graph attention network (C1).

    Attributes:
        out_dim (int): Dimension of the social context vector c_t. Default: 64
        heads (int): Number of heterogeneous attention heads. Default: 4
        hidden (int): Hidden dimension of the attention projections. Default: 128
        layers (int): Number of GAT layers. Default: 2
        radius_m (float): Distance threshold (meters) for proximity edges. Default: 4.0
        converge_angle_deg (float): Relative-heading threshold (degrees) below which a
            velocity-converging edge is added. Default: 90.0
        variant (str): Graph encoder, "gat" (shared-Q soft-blend, default) or "height"
            (HEIGHT-style separate per-edge-type multi-head attention).
        obstacle_edges (bool): Enable obstacle-agent (OA) edges. HEIGHT only; requires
            ObstacleNodeSetSpace observations. Default: False (obstacles stay in the laser CNN).
        max_obstacles (int): Fixed number of obstacle nodes O (padded/masked). Default: 8
        obstacle_radius_m (float): Distance threshold (meters) for obstacle-agent edges. Default: 3.0
    """

    out_dim: int = 64
    heads: int = 4
    hidden: int = 128
    layers: int = 2
    radius_m: float = 4.0
    converge_angle_deg: float = 90.0
    variant: Literal["gat", "height"] = "gat"
    obstacle_edges: bool = False
    max_obstacles: int = 8
    obstacle_radius_m: float = 3.0


class SocialDALICfg(BaseModel):
    """Configuration for the DALI dynamics-context encoder (C2).

    Attributes:
        enabled (bool): Whether DALI (d_t and L_dyn) is active. Default: False
        out_dim (int): Dimension of the dynamics context vector d_t. Default: 64
        hidden (int): Hidden dimension of the trajectory GRU. Default: 64
        k_steps (int): Trajectory window length. FIXED at 40 (= 2 s at 20 Hz) so the GRU
            sees a full SFM/HSFM avoidance cycle. Default: 40
        lambda_dyn (float): Weight of the auxiliary forward-prediction loss in the ELBO. Default: 0.1
        imag_backprop_steps (int): Number of imagination steps over which L_dyn and the
            d_t->actor gradient through imagined peds are back-propagated. Beyond this, d_t is
            computed with detached decoded peds to bound error amplification. Default: 5
    """

    enabled: bool = False
    out_dim: int = 64
    hidden: int = 64
    k_steps: int = 40
    lambda_dyn: float = 0.1
    imag_backprop_steps: int = 5


class SocialContextCfg(BaseModel):
    """Configuration for the Contextual Social RSSM (cSRSSM) crowd-behavior context b.

    Casts social navigation as a contextual MDP: a slow, probabilistic crowd-behavior code
    ``b`` (aggressiveness, yielding, preferred speed, personal-space, local policy) is inferred
    from a trajectory window and conditions the RSSM *transition* (unlike GAT/DALI, which only
    enter the feature). This enables zero-shot generalization across crowd behaviors and
    counterfactual social rollouts. Inferred once per sequence and held fixed across the
    imagination horizon (realtime: no graph-in-transition loop).

    Attributes:
        enabled (bool): Master switch. When False, context_size is 0 and the RSSM transition is
            byte-identical to baseline. Default: False
        b_dim (int): Dimension of the context vector b fed into the transition. Default: 16
        hidden (int): Hidden width of the context encoder (reuses the DALI trajectory GRU).
            Default: 64
        window (int): Trajectory-window length k the encoder summarizes. Default: 8
        kl_scale (float): Weight of KL(q(b)||N(0,1)) in the ELBO (information bottleneck). Default: 1.0
        pred_scale (float): Weight of the prediction-driven identifiability term (primary
            anti-collapse mechanism, VariBAD-style): from (b, pooled crowd summary at t)
            predict the pooled summary at t+1 for steps beyond the inference window. Requires
            batch_length > window to have any effect. Default: 0.1
        infonce_scale (float): Weight of the InfoNCE identifiability term (secondary /
            ablation-only; known false-negative issue under domain randomization -- see
            context.py). Default: 0.0
    """

    enabled: bool = False
    b_dim: int = 16
    hidden: int = 64
    window: int = 8
    kl_scale: float = 1.0
    pred_scale: float = 0.1
    infonce_scale: float = 0.0


class SocialCurriculumCfg(BaseModel):
    """Configuration for the staged, gated social-training curriculum (M6.1).

    Stages advance only when the gate metric is met; a stage that hits its step cap without
    passing halts training with a diagnostic rather than silently proceeding.

    Step caps are additive: S1 activates at warmup_steps, S2 at warmup_steps+gat_steps, etc.

    Attributes:
        enabled (bool): Whether the gated curriculum callback is active. Default: False
        warmup_steps (int): S0 world-model warmup step cap. Default: 500_000
        gat_steps (int): S1 additional steps for GAT-online stage. Default: 1_000_000
        dali_steps (int): S2 additional steps for DALI-online stage. Default: 1_000_000
        density_steps (int): S3 additional steps for pedestrian-density ramp. Default: 2_500_000
        recon_mse_gate (float): Max ped-reconstruction MSE (m^2) to pass S0. Default: 0.10
        probe_acc_gate (float): Min cross-simulator probe accuracy to pass S0. Default: 0.70
        success_gate (float): Min success rate to pass S1 (0.0 = accept any). Default: 0.0
        ct_ablation_drop_gate (float): Min success-rate drop when zeroing c_t (proves C1 useful).
            Default: 0.03 (3 pp).
        collision_gate (float): Ratio of S2 collision rate to S1; must be ≤ this to pass S2.
            Default: 1.0 (collision must not increase).
    """

    enabled: bool = False
    warmup_steps: int = 500_000
    gat_steps: int = 1_000_000
    dali_steps: int = 1_000_000
    density_steps: int = 2_500_000
    recon_mse_gate: float = 0.10
    probe_acc_gate: float = 0.70
    success_gate: float = 0.0
    ct_ablation_drop_gate: float = 0.03
    collision_gate: float = 1.0


class SocialCfg(BaseModel):
    """Configuration for the Social-Dreamer augmentation (C1 + C2).

    When ``enabled`` is False the augmented state reduces to the baseline DreamerV3 feature
    ``concat(stoch, deter)`` and no social parameters are added to any network. This is the
    firebreak that keeps the baseline run byte-identical and prevents C1/C2 dimensional drift
    (all heads read their input width from ``social.dims.augmented_feat_size``).

    Attributes:
        enabled (bool): Master switch for the social augmentation. Default: False
        max_peds (int): Fixed number of pedestrian nodes N (padded/masked). Default: 8
        node_feat_dim (int): Node feature width F (Dx, Dy, vx, vy, social_state). Default: 5
        peds_recon_scale (float): Loss scale for the pedestrian reconstruction head. Default: 1.0
        cell_type (str): RSSM deterministic backbone. "gru" (default), "transformer"
            (M5.1: causal sliding-window attention cell inside the sequential scan), or
            "tssm" (M5.2: STORM-style TSSM — obs-only posterior + one parallel causal
            attention pass over the whole sequence in observe(); imagination stays
            sequential with a per-layer KV cache).
        gat (SocialGATCfg): Graph attention network configuration.
        dali (SocialDALICfg): Dynamics-context encoder configuration.
        curriculum (SocialCurriculumCfg): Staged training-curriculum configuration.
        use_se2_frame_canon (bool): Enable SE(2) frame canonicalization. When True, decoded
            pedestrian positions are transformed from the anchor frame to the current robot frame
            before GAT/DALI, eliminating accumulating bias over imagination horizons. Default: False
        kinematics_dt (float): Control period in seconds (10 Hz → 0.1). Used by integrate_se2
            for imagination pose accumulation. Default: 0.1
        se2_augment_prob (float): Probability of applying random SE(2) augmentation per batch
            element during training. Default: 0.5
        action_scale_linear (float): Physical linear velocity range (m/s). Populated by
            arena_trainer from robot_description. Default: 1.0
        action_scale_angular (float): Physical angular velocity range (rad/s). Populated by
            arena_trainer from robot_description. Default: 1.0
    """

    enabled: bool = False
    max_peds: int = 8
    node_feat_dim: int = 5
    peds_recon_scale: float = 1.0
    cell_type: Literal["gru", "transformer", "tssm"] = "gru"
    transformer_ctx_len: int = 64      # sliding-window length (TransformerCell and TSSM)
    transformer_num_heads: int = 4     # attention heads (TransformerCell and TSSM)
    tssm_num_layers: int = 2           # TSSM only: causal attention blocks in the h-pathway
    tssm_ff_mult: int = 2              # TSSM only: block MLP width multiplier (ff = mult*deter)
    gat: SocialGATCfg = SocialGATCfg()
    dali: SocialDALICfg = SocialDALICfg()
    context: SocialContextCfg = SocialContextCfg()
    curriculum: SocialCurriculumCfg = SocialCurriculumCfg()
    use_se2_frame_canon: bool = False
    kinematics_dt: float = 0.1
    se2_augment_prob: float = 0.5
    action_scale_linear: float = 1.0
    action_scale_linear_y: float = 1.0
    action_scale_angular: float = 1.0
    action_holonomic: bool = False


class ModelCfg(BaseModel):
    """Configuration class for DreamerV3 model parameters.

    This class defines the configuration for the world model and its components in DreamerV3.

    Attributes:
        dyn_hidden (int): Number of hidden units in dynamics network. Default: 1024
        dyn_deter (int): Size of deterministic state. Default: 1024
        dyn_stoch (int): Size of stochastic state. Default: 32
        dyn_discrete (int): Number of categorical classes for discrete state. Default: 32
        dyn_rec_depth (int): Number of recurrent layers in dynamics network. Default: 1
        dyn_mean_act (str): Activation function for mean in dynamics network. Default: "none"
        dyn_std_act (str): Activation function for standard deviation in dynamics. Default: "sigmoid2"
        dyn_min_std (float): Minimum standard deviation value. Default: 0.1
        grad_heads (List[str]): Components that receive gradients. Default: ["decoder", "reward", "cont"]
        units (int): Number of units in dense layers. Default: 1024
        act (str): Default activation function. Default: "SiLU"
        norm (bool): Whether to use layer normalization. Default: True
        dyn_scale (float): Scale factor for dynamics loss. Default: 0.5
        rep_scale (float): Scale factor for representation loss. Default: 0.1
        kl_free (float): Free bits in KL divergence. Default: 1.0
        weight_decay (float): L2 regularization strength. Default: 0.0
        unimix_ratio (float): Uniform mixture ratio for discrete distributions. Default: 0.01
        initial (str): Initialization method for model parameters. Default: "learned"
        encoder (EncoderCfg): Configuration for encoder network
        decoder (DecoderCfg): Configuration for decoder network
        actor (ActorCfg): Configuration for actor network
        critic (CriticCfg): Configuration for critic network
        reward_head (RewardHeadCfg): Configuration for reward prediction head
        cont_head (ContHeadCfg): Configuration for continuation prediction head
        behavior (BehaviorCfg): Configuration for behavior module
        exploration (ExplorationCfg): Configuration for exploration strategy
    """

    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    kl_free: float = 1.0
    weight_decay: float = 0.0
    unimix_ratio: float = 0.01
    initial: str = "learned"
    dyn_hidden: int = 128
    dyn_deter: int = 128
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_rec_depth: int = 1
    dyn_mean_act: str = "none"
    dyn_std_act: str = "sigmoid2"
    dyn_min_std: float = 0.1
    grad_heads: List[str] = ["decoder", "reward", "cont"]
    units: int = 128
    act: str = "SiLU"
    norm: bool = True
    encoder: EncoderCfg = EncoderCfg()
    decoder: DecoderCfg = DecoderCfg()
    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()
    reward_head: RewardHeadCfg = RewardHeadCfg()
    cont_head: ContHeadCfg = ContHeadCfg()
    behavior: BehaviorCfg = BehaviorCfg()
    exploration: ExplorationCfg = ExplorationCfg()
    social: SocialCfg = SocialCfg()


class TrainingCfg(BaseModel):
    """Training configuration for the DreamerV3 agent.

    This class defines the configuration parameters used during the training process
    of the DreamerV3 agent, including hyperparameters for optimization, training schedule,
    and evaluation settings.

    Attributes:
        steps (float): Total number of training steps (1e8)
        batch_size (int): Number of sequences per training batch (16)
        batch_length (int): Length of sequences in the training batch (64)
        train_ratio (int): Training steps per environment step (16)
        pretrain_steps (int): Number of steps for pretraining (100)
        prefill_steps (int): Number of steps to prefill the replay buffer (5000)
        model_lr (float): Learning rate for model optimization (1e-4)
        opt_eps (float): Epsilon for optimizer numerical stability (1e-8)
        grad_clip (int): Gradient clipping threshold (1000)
        dataset_size (int): Maximum size of the replay buffer (1000000)
        opt (str): Optimizer type, currently supporting 'adam'
        eval_every (float): Number of steps between evaluations (1e5)
        eval_episode_num (int): Number of episodes for evaluation (20)
        warmup_steps (int): Linear LR warmup steps for model optimizer (0 = disabled). Default: 1000
    """

    steps: float = 1e8
    batch_size: int = 16
    batch_length: int = 64
    train_ratio: int = 16
    pretrain_steps: int = 100
    prefill_steps: int = 200
    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: int = 1000
    dataset_size: int = 200
    opt: str = "adam"
    eval_every: float = 200
    eval_episode_num: int = 5
    warmup_steps: int = 1000


class DreamerV3Cfg(FrameworkCfg):
    """Configuration class for DreamerV3 agent.

    This class serves as the main configuration container for the DreamerV3 agent,
    organizing all configuration parameters into logical groups.

    Attributes:
        general (GeneralCfg): General configuration parameters like logging, device settings.
        environment (EnvironmentCfg): Environment-specific configuration parameters.
        model (ModelCfg): Model architecture and hyperparameters configuration.
        training (TrainingCfg): Training-related configuration parameters.

    Note:
        This class inherits from BaseModel and allows for arbitrary types in its configuration.
    """

    name: Literal[SupportedRLFrameworks.DREAMER_V3] = SupportedRLFrameworks.DREAMER_V3
    general: GeneralCfg = GeneralCfg()
    environment: EnvironmentCfg = EnvironmentCfg()
    model: ModelCfg = ModelCfg()
    training: TrainingCfg = TrainingCfg()

    observation_space_list: List[str] = Field(
        default_factory=lambda: [
            "LaserCartesianMapSpace",
            "PedestrianVelXSpace",
            "PedestrianVelYSpace",
            "PedestrianTypeSpace",
            "PedestrianSocialStateSpace",
            "PedestrianNodeSetSpace",
            "PedestrianMaskSpace",
            "RobotPoseSpace",
            "DistAngleToGoalSpace",
            "LastActionSpace",
            "IsFirstStepSpace",
            "IsTerminalStepSpace",
        ],
        description=(
            "Names of registered SpaceFactory observation spaces used by this model."
        ),
    )
    observation_space_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "reduced_num_beams": 72,  # 720 beams / 10 = 3.75 deg angular resolution
            "normalize": True,
            "goal_max_dist": 10,
            # Kwargs for feature-map spaces (PedestrianVel/Type/SocialState).
            # These are collected but NOT encoded (not in mlp_keys/cnn_keys);
            # they just need to init without errors.
            "roi_in_m": 40,
            "feature_map_size": 80,
            "laser_stack_size": 10,
        },
        description="Shared kwargs passed to each observation space on construction.",
    )

    class Config:
        arbitrary_types_allowed = True
