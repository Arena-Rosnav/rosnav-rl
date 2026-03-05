from pathlib import Path
from typing import ClassVar, List, Optional, Union

from pydantic import BaseModel
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
        precision (int): Numerical precision in bits. Defaults to 32.
        debug (bool): Whether to enable debug mode. Defaults to False.
        video_pred_log (bool): Whether to log video predictions. Defaults to True.
    """

    logdir: Optional[Union[str, Path]] = (
        "/home/le/arena4_ws/src/planners/rosnav_rl/rosnav_rl" + "/agents"
    )  # TODO: get the path from the config rp.get_path("rosnav_rl")
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
    """

    discount: float = 0.997
    discount_lambda: float = 0.95
    imag_horizon: int = 15
    imag_gradient: str = "dynamics"
    imag_gradient_mix: float = 0.0
    eval_state_mean: bool = False


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

    class Config:
        arbitrary_types_allowed = True
