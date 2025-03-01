import math
from typing import Dict
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import rosnav_rl.model.dreamerv3.tools as tools


def add_batch_dim(shape: tuple, is_channels_first: bool = False):
    if len(shape) == 3:
        return shape

    if is_channels_first and shape[0] == 1:
        return shape

    if not is_channels_first and shape[-1] == 1:
        return shape

    if is_channels_first:
        return (1, *shape)
    return (*shape, 1)


def add_batch_dim_to_cnn_shapes(
    cnn_shapes: Dict[str, tuple], is_channels_first: bool = False
):
    return {k: add_batch_dim(v, is_channels_first) for k, v in cnn_shapes.items()}


def translate_to_channels_last(shapes):
    translated_shapes = {}
    for key, shape in shapes.items():
        if len(shape) < 4:
            if len(shape) == 3:
                translated_shapes[key] = (shape[1], shape[2], shape[0])
            elif len(shape) == 2:
                translated_shapes[key] = (shape[1], shape[0])
            else:
                translated_shapes[key] = shape
        else:
            translated_shapes[key] = shape

    return translated_shapes


class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        """Initialize the RSSM (Recurrent State-Space Model) class.

        This class implements the Recurrent State-Space Model used in DreamerV3 for world modeling.

        Args:
            stoch (int, optional): Size of stochastic state. Defaults to 30.
            deter (int, optional): Size of deterministic state. Defaults to 200.
            hidden (int, optional): Size of hidden layers. Defaults to 200.
            rec_depth (int, optional): Depth of recurrent network. Defaults to 1.
            discrete (bool, optional): Whether to use discrete or continuous state space. Defaults to False.
            act (str, optional): Activation function to use. Defaults to "SiLU".
            norm (bool, optional): Whether to use layer normalization. Defaults to True.
            mean_act (str, optional): Activation function for mean. Defaults to "none".
            std_act (str, optional): Activation function for standard deviation. Defaults to "softplus".
            min_std (float, optional): Minimum standard deviation. Defaults to 0.1.
            unimix_ratio (float, optional): Uniform mixture ratio for discrete models. Defaults to 0.01.
            initial (str, optional): Initial state distribution type. Defaults to "learned".
            num_actions (int, optional): Number of possible actions. Defaults to None.
            embed (int, optional): Size of embedding. Defaults to None.
            device (str, optional): Device to use for computation. Defaults to None.

        Returns:
            None

        Note:
            The model architecture includes:
            - Input layers for processing state and action
            - GRU cell for recurrent processing
            - Output layers for image and observation predictions
            - Separate stat layers for discrete/continuous state spaces
        """
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.zeros([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        """
        Processes a sequence of observations, actions, and state indicators to generate posterior and prior states.

        Args:
            embed (torch.Tensor): Embedded observations with shape (batch, time, channels).
            action (torch.Tensor): Action sequence with shape (batch, time, action_dim).
            is_first (torch.Tensor): Binary tensor indicating first steps with shape (batch, time).
            state (Optional[tuple]): Initial state tuple containing posterior and prior states. Defaults to None.

        Returns:
            tuple: A pair of dictionaries (post, prior) containing:
                - post: Posterior state distributions with shape (batch, time, stoch, discrete_num)
                - prior: Prior state distributions with shape (batch, time, stoch, discrete_num)

        Notes:
            - Transforms input tensors from (batch, time, ch) to (time, batch, ch) for processing
            - Uses static_scan to sequentially process the time dimension
            - Returns tensors transformed back to original batch-first format
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        """
        Imagine future states given an action sequence and initial state.

        Args:
            action (torch.Tensor): Action sequence tensor to imagine forward.
                Expected shape: [batch_size, sequence_length, action_dims]
            state (dict): Initial state dictionary containing RSSM state tensors.
                Must contain keys for model state representation.

        Returns:
            dict: Prior state dictionary containing imagined sequence of states.
                Keys match input state dict with imagined values.
                Tensors have shape [batch_size, sequence_length, ...]

        Notes:
            - Swaps batch and time dimensions for internal processing
            - Uses static_scan to iterate img_step function over action sequence
            - Swaps dimensions back before returning results
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        Performs an observation step in the world model, updating the state based on previous state, action, and current embedding.

        Args:
            prev_state (dict or None): Previous state containing stochastic and deterministic components.
                                      If None, initializes new state.
            prev_action (torch.Tensor): Previous action tensor of shape (batch_size, action_dim).
            embed (torch.Tensor): Current observation embedding.
            is_first (torch.Tensor): Boolean tensor indicating first steps in batch sequences.
            sample (bool, optional): Whether to sample from the distribution or take the mode. Defaults to True.

        Returns:
            tuple:
                - post (dict): Posterior state containing:
                    - stoch (torch.Tensor): Stochastic state component
                    - deter (torch.Tensor): Deterministic state component
                    - stats: Additional distribution statistics
                - prior (dict): Prior state prediction before observation
        """
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """
        Performs one step of the image transition model, computing the prior state distribution.

        This method takes the previous latent state and action to predict the next latent state
        using a recurrent neural network architecture.

        Args:
            prev_state (dict): Previous latent state containing:
                - stoch (torch.Tensor): Stochastic state component
                - deter (torch.Tensor): Deterministic state component
            prev_action (torch.Tensor): Previous action taken
            sample (bool, optional): Whether to sample from the distribution or take the mode.
                                   Defaults to True.

        Returns:
            dict: Prior state prediction containing:
                - stoch (torch.Tensor): Predicted stochastic state component
                - deter (torch.Tensor): Predicted deterministic state component
                - Additional distribution statistics from sufficient statistics layer
        """
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
        is_channels_first=False,
        device="cuda",
    ):
        """Initialize MultiEncoder.

        A hybrid encoder that combines CNN and MLP architectures for processing mixed input types.

        Args:
            shapes (dict): Dictionary of input shapes for each input type
            mlp_keys (str): Regex pattern to match keys for MLP inputs
            cnn_keys (str): Regex pattern to match keys for CNN inputs
            act (str): Activation function to use
            norm (str): Normalization method to use
            cnn_depth (int): Number of channels in CNN layers
            kernel_size (int): Kernel size for CNN layers
            minres (int): Minimum resolution for CNN
            mlp_layers (int): Number of layers in MLP
            mlp_units (int): Number of units per MLP layer
            symlog_inputs (bool): Whether to apply symmetric log to inputs
            is_channels_first (bool, optional): Whether input tensors are channels-first. Defaults to False.
            device (str, optional): Device to place model on. Defaults to "cuda".

        Attributes:
            cnn_shapes (dict): Dictionary of shapes for CNN inputs
            mlp_shapes (dict): Dictionary of shapes for MLP inputs
            outdim (int): Total output dimension of the encoder
            _cnn (ConvEncoder): CNN component of the encoder
            _mlp (MLP): MLP component of the encoder

        Notes:
            - Excludes certain keys like "is_first", "is_last", "is_terminal", "reward"
            - Automatically handles channel order conversion if needed
            - Combines both CNN and MLP features into a single output representation
        """
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")

        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in [2, 3] and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }

        self.cnn_shapes = add_batch_dim_to_cnn_shapes(
            self.cnn_shapes, is_channels_first
        )
        # translate to channels last if is_channels_first is True, shapes is dictionary of shapes
        if is_channels_first:
            self.cnn_shapes = translate_to_channels_last(self.cnn_shapes)
            self.mlp_shapes = translate_to_channels_last(self.mlp_shapes)

        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
                device=device,
            )
            self.outdim += mlp_units

    def forward(self, obs):
        """Forward pass through the encoder network.

        Takes a dictionary of observations and processes them through CNN and MLP networks depending on
        the presence of cnn_shapes and mlp_shapes configurations. Concatenates the outputs from both networks.

        Args:
            obs (dict): Dictionary containing observation tensors. Keys should match those specified in
                       cnn_shapes and mlp_shapes during initialization.

        Returns:
            torch.Tensor: Concatenated output features from CNN and MLP networks.
        """
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
        is_channels_first=False,
        device="cuda",
    ):
        """Initialize the MultiDecoder.

        A decoder network that can handle both CNN and MLP outputs, supporting both image and vector distributions.

        Args:
            feat_size (int): Size of input feature vector
            shapes (dict): Dictionary mapping output names to their shapes
            mlp_keys (str): Regex pattern for identifying MLP output keys
            cnn_keys (str): Regex pattern for identifying CNN output keys
            act (str): Activation function to use
            norm (str): Normalization method to use
            cnn_depth (int): Depth of CNN decoder
            kernel_size (int): Kernel size for CNN layers
            minres (int): Minimum resolution for CNN decoder
            mlp_layers (int): Number of layers in MLP decoder
            mlp_units (int): Number of units per MLP layer
            cnn_sigmoid (bool): Whether to use sigmoid activation in CNN output
            image_dist (str): Type of image distribution to use
            vector_dist (str): Type of vector distribution to use
            outscale (float): Output scale factor
            is_channels_first (bool, optional): Whether input tensors are channels-first. Defaults to False.
            device (str, optional): Device to run the model on. Defaults to "cuda".

        Returns:
            None: Initializes the MultiDecoder object with CNN and/or MLP decoders based on input shapes.

        Note:
            - Handles both CNN and MLP outputs separately
            - Automatically detects shape types and routes them to appropriate decoders
            - Supports channel-first to channel-last conversion
            - Excludes certain keys ('is_first', 'is_last', 'is_terminal') from processing
        """
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in [2, 3] and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }

        self.cnn_shapes = add_batch_dim_to_cnn_shapes(
            self.cnn_shapes, is_channels_first
        )
        # translate to channels last if is_channels_first is True, shapes is dictionary of shapes
        if is_channels_first:
            self.cnn_shapes = translate_to_channels_last(self.cnn_shapes)
            self.mlp_shapes = translate_to_channels_last(self.mlp_shapes)

        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
                device=device,
            )
        self._image_dist = image_dist

    def forward(self, features):
        """Forward pass through the decoder network.

        This method processes input features through CNN and MLP networks to generate
        distributions for reconstructing observations.

        Args:
            features (torch.Tensor): Input features to be decoded.

        Returns:
            dict: Dictionary containing distributions for each output shape, where:
                - If CNN shapes exist, contains image distributions for each CNN output
                - If MLP shapes exist, contains distributions from MLP outputs
                Keys are shape names and values are corresponding distributions.
        """
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        """Initialize the ConvEncoder.

        A convolutional encoder that progressively downsamples the input image using
        strided convolutions, with optional layer normalization and activation functions.

        Args:
            input_shape (tuple): Shape of input images (height, width, channels).
            depth (int, optional): Initial number of output channels. Will double at each stage. Defaults to 32.
            act (str, optional): Activation function from torch.nn to use. Defaults to "SiLU".
            norm (bool, optional): Whether to use layer normalization. Defaults to True.
            kernel_size (int, optional): Size of convolutional kernels. Defaults to 4.
            minres (int, optional): Minimum resolution to downsample to. Determines number of stages. Defaults to 4.

        Attributes:
            outdim (int): Dimension of flattened output.
            layers (nn.Sequential): Sequential container of encoder layers.
        """
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        """
        Forward pass of the neural network.

        Takes an observation tensor and processes it through the network's layers.
        The input tensor is expected to have shape (batch, time, height, width, channels).

        Args:
            obs (torch.Tensor): Input observation tensor with shape (batch, time, height, width, channels)

        Returns:
            torch.Tensor: Processed tensor with shape (batch, time, feature_dim) where feature_dim
                         depends on the network's output dimensions

        Process:
            1. Normalizes input by subtracting 0.5
            2. Reshapes from (batch, time, h, w, ch) to (batch * time, h, w, ch)
            3. Permutes to channel-first format (batch * time, ch, h, w)
            4. Passes through network layers
            5. Flattens the spatial dimensions
            6. Reshapes back to (batch, time, feature_dim)
        """
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        """Initialize the Convolutional Decoder network.

        This decoder network upsamples a feature vector into an image through a series of transposed
        convolution layers. The network progressively increases spatial dimensions while reducing channels.

        Args:
            feat_size (int): Size of the input feature vector
            shape (tuple, optional): Expected output shape (channels, height, width). Defaults to (3, 64, 64).
            depth (int, optional): Base depth for computing channel dimensions. Defaults to 32.
            act (torch.nn.Module, optional): Activation function to use. Defaults to nn.ELU.
            norm (bool, optional): Whether to use layer normalization. Defaults to True.
            kernel_size (int, optional): Size of convolutional kernels. Defaults to 4.
            minres (int, optional): Minimum resolution for the initial spatial dimensions. Defaults to 4.
            outscale (float, optional): Output layer weight initialization scale. Defaults to 1.0.
            cnn_sigmoid (bool, optional): Whether to apply sigmoid to the output. Defaults to False.

        Attributes:
            _shape (tuple): Output shape of the decoder
            _cnn_sigmoid (bool): Flag for sigmoid activation
            _minres (int): Minimum resolution
            _embed_size (int): Size of the embedding
            layers (nn.Sequential): Sequential container of decoder layers
        """
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth

            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        """
        Calculate padding values for 'SAME' padding in convolution/deconvolution operations.

        This function computes the padding needed to maintain the same spatial dimensions
        when performing convolution or deconvolution operations.

        Args:
            k (int): Kernel size
            s (int): Stride
            d (int): Dilation rate

        Returns:
            tuple: A pair of integers (pad, outpad) where:
                - pad: The amount of padding to add on both sides
                - outpad: The adjustment needed for output padding in deconvolution
        """
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        """Forward pass of the decoder network.

        This method transforms the input features through a series of deconvolutional layers
        to reconstruct the original image space.

        Args:
            features (torch.Tensor): Input tensor of shape (batch, time, feature_dim)
            dtype (torch.dtype, optional): Data type for computations. Defaults to None.

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (batch, time, height, width, channels).
                         Values are either sigmoid-activated or shifted by 0.5 based on self._cnn_sigmoid.

        Shape:
            - Input: (batch, time, feature_dim)
            - Output: (batch, time, height, width, channels)

        Note:
            The method performs the following transformations:
            1. Linear projection of features
            2. Reshaping to spatial dimensions
            3. Channel-first conversion for CNN operations
            4. Deconvolution through network layers
            5. Reshaping and permuting to final output format
        """
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        """Initialize GRUCell layer.

        A Gated Recurrent Unit (GRU) cell that transforms input features through a gated mechanism.

        Args:
            inp_size (int): Size of the input features
            size (int): Size of the hidden state
            norm (bool, optional): Whether to apply layer normalization. Defaults to True.
            act (callable, optional): Activation function. Defaults to torch.tanh.
            update_bias (float, optional): Initial bias for the update gate. Defaults to -1.

        Notes:
            The GRU cell implements the following transformations:
            - Creates a linear layer that maps concatenated input and hidden state to gates
            - Optionally applies layer normalization

        """
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        """Forward pass for a custom GRU-like cell.

        Args:
            inputs (torch.Tensor): Input tensor to the cell
            state (list): Previous state wrapped in a list (Keras convention)

        Returns:
            tuple: Contains:
                - torch.Tensor: New state/output of the cell
                - list: New state wrapped in a list (Keras convention)

        Details:
            Implements a modified GRU cell with:
            - Combined input+state processing
            - Reset gate with sigmoid activation
            - Candidate activation with custom activation function
            - Update gate with sigmoid activation and bias
            - Output mixing via interpolation between candidate and previous state
        """
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    """A 2D convolution layer with automatic 'same' padding.

    This layer extends PyTorch's Conv2d to automatically calculate and apply padding
    to preserve the input spatial dimensions, similar to TensorFlow's 'SAME' padding behavior.

    The padding is calculated dynamically based on:
        - Input dimensions
        - Kernel size
        - Stride
        - Dilation

    The padding is applied equally to both sides of each spatial dimension when possible,
    with any odd amount of padding being added to the right/bottom.

    Inherits from:
        torch.nn.Conv2d

    Methods:
        calc_same_pad: Calculates the required padding for a single dimension
        forward: Applies padding and performs the convolution operation
    """

    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        """Performs forward pass through the convolutional layer with 'SAME' padding.

        This method ensures that the output has the same spatial dimensions as the input by
        automatically calculating and applying the necessary padding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor after convolution with same padding applied

        Note:
            The padding is calculated and applied to maintain spatial dimensions according
            to the 'SAME' padding scheme from TensorFlow, ensuring output dimensions match
            input dimensions when accounting for stride and dilation.
        """
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    """Layer Normalization for image channel dimension.

    This module applies Layer Normalization over the channel dimension of an image tensor.
    It expects input in BCHW format (Batch, Channel, Height, Width) and normalizes across
    the channel dimension.

    Args:
        ch (int): Number of channels in the input image tensor
        eps (float, optional): A small constant for numerical stability. Default: 1e-03

    Input Shape:
        - Input: (batch_size, channels, height, width)
        - Output: (batch_size, channels, height, width)

    Note:
        The implementation permutes the tensor to move channels to the last dimension,
        applies layer normalization, and then permutes back to BCHW format.
    """

    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        """
        Applies layer normalization to the input tensor with a specific permutation pattern.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Normalized tensor with same shape as input (batch_size, channels, height, width)

        Note:
            The function performs the following steps:
            1. Permutes input to (batch_size, height, width, channels)
            2. Applies layer normalization
            3. Permutes back to original format (batch_size, channels, height, width)
        """
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
