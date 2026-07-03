import math
import re
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as torchd
from torch import nn

from ..dreamerv3 import tools


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
    """Recurrent State-Space Model (RSSM) for world modeling in DreamerV3.

    This module implements the core world model component used in DreamerV3, representing
    the environment dynamics through a combination of deterministic and stochastic state
    components. The RSSM can use either discrete or continuous state representations.

    The model consists of:
        1. A deterministic recurrent component (GRU-based)
        2. A stochastic component that can be discrete (OneHotDist) or continuous (Normal)
        3. Transition model that predicts next states from current state and action
        4. Posterior encoder that incorporates observation embeddings

    Key methods:
        - initial: Initialize state tensors for a new sequence
        - observe: Process a sequence of observations to generate posterior and prior states
        - imagine_with_action: Roll out state predictions using only actions (no observations)
        - obs_step: Single step inference of posterior state given observation and prior
        - img_step: Single step inference of prior state given previous state and action
        - kl_loss: Compute KL divergence between posterior and prior distributions

    The model supports:
        - Multiple discrete/continuous distribution parameterizations
        - Learned or zero-initialized state
        - Variable recurrent depth
        - Layer normalization
    """
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
        cell_type: str = "gru",
        transformer_ctx_len: int = 64,
        transformer_num_heads: int = 4,
        context_size: int = 0,
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
        # cSRSSM: crowd-behavior context b concatenated into the transition input. 0 = disabled
        # (baseline byte-identical). Held fixed across the imagination horizon (see img_step).
        self._context_size = context_size

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_dim += self._context_size
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell_type = cell_type
        if cell_type == "transformer":
            # Causal sliding-window attention cell wrapped as a recurrent cell (M5.1),
            # KV-cached (M5.1.1) so past-output K/V projections aren't recomputed
            # every step -- see TransformerCell docstring for the O(ctx_len)->O(1)
            # rationale. For the sequence-parallel observe() variant see TSSM (M5.2).
            # State shape note: state["deter"] stays (B, deter) for get_feat compat.
            # The K/V cache (B, ctx_len, 2*deter) lives in state["deter_seq"].
            self._cell = TransformerCell(
                inp_size=self._hidden,
                size=self._deter,
                num_heads=transformer_num_heads,
                ctx_len=transformer_ctx_len,
                norm=norm,
            )
        elif cell_type == "tssm":
            # TSSM (M5.2) builds its own attention stack in the subclass; only valid when
            # constructed via the TSSM class (RSSM with cell_type="tssm" alone would crash
            # in img_step). No cell, no compile: the TSSM step path builds data-dependent
            # attention masks that would break/recompile under torch.compile.
            self._cell = None
        else:
            self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        if self._cell is not None:
            self._cell.apply(tools.weight_init)
            # Compile the GRU/Transformer cell forward and the full img_step kernel for
            # throughput. obs_step is intentionally left eager: it has data-dependent Python
            # branches on is_first (networks.py:349,355) that would cause repeated
            # recompilation under compile. img_step is called by obs_step internally, so this
            # compile still covers the hot path.
            self._cell.forward = torch.compile(self._cell.forward, mode="default")
            self.img_step = torch.compile(self.img_step, mode="default")

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
        if self._cell_type == "transformer":
            # Transformer carries a K/V cache (2*deter: K and V halves); zeros is the
            # right initial (see TransformerCell docstring for the cache layout).
            state["deter_seq"] = torch.zeros(
                batch_size, self._cell.ctx_len, 2 * self._deter, device=self._device
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None, context=None):
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
        # context (cSRSSM crowd-behavior code b) is inferred once per sequence and held fixed
        # across the scan, so it's closed over rather than passed as a per-step scanned input.
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first, context=context
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state, context=None):
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
        # context held fixed across the imagination horizon (closed over, not scanned).
        prior = tools.static_scan(
            lambda prev_state, prev_act: self.img_step(prev_state, prev_act, context=context),
            [action],
            state,
        )
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

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, context=None):
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

        prior = self.img_step(prev_state, prev_action, context=context)
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
        if "deter_seq" in prior:
            post["deter_seq"] = prior["deter_seq"]
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True, context=None):
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
        if context is not None and self._context_size > 0:
            # cSRSSM: crowd-behavior context b held fixed across horizon, conditions transition
            x = torch.cat([prev_stoch, prev_action, context], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        _new_deter_seq = None
        if self._cell_type == "transformer":
            # Hoist window outside loop so each iteration feeds the previous output forward.
            deter_seq = prev_state["deter_seq"]           # (B, L, 2*deter) -- K/V cache
        else:
            deter = prev_state["deter"]                   # (B, deter)
        for _ in range(self._rec_depth):
            if self._cell_type == "transformer":
                # TransformerCell state: (B, ctx_len, 2*deter) K/V cache.
                # deter_seq carries the cache; deter carries the output.
                x, new_seq_list = self._cell(x, [deter_seq])
                deter = x                                  # (B, deter) — current output
                deter_seq = new_seq_list[0]               # updated window for next iter
                _new_deter_seq = deter_seq                # (B, L, deter)
            else:
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
        if _new_deter_seq is not None:
            prior["deter_seq"] = _new_deter_seq
        return prior

    def get_stoch(self, deter):
        """
        Generates stochastic part of latent state from deterministic part.

        This method processes the deterministic latent state through the image output layers and 
        statistical layers to produce parameters for a distribution. It then returns the mode
        of this distribution as the stochastic part of the latent state.

        Args:
            deter: Tensor representing the deterministic part of the latent state.

        Returns:
            Tensor representing the mode of the stochastic part of the latent state.
        """
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        """Calculate sufficient statistics for a given layer.
        
        This method computes the distribution parameters for either discrete or continuous latent variables.
        
        Args:
            name (str): The name of the layer, either "ims" for image statistics or "obs" for observation statistics.
            x (torch.Tensor): The input tensor to transform.
        
        Returns:
            dict: For discrete distributions, returns a dictionary with 'logit' key containing the logits.
                  For continuous distributions, returns a dictionary with 'mean' and 'std' keys.
                  - For discrete case, logits have shape [..., stoch, discrete]
                  - For continuous case:
                      - mean is transformed according to self._mean_act
                      - std is transformed according to self._std_act and has minimum value added
        
        Raises:
            NotImplementedError: If the name is neither "ims" nor "obs".
            
        Note:
            - mean is transformed according to self._mean_act which can be either:
                - "none": No transformation
                - "tanh5": Tanh transformation scaled by 5.0
            - std is transformed according to self._std_act which can be:
                - "softplus": Apply softplus function
                - "abs": Take absolute value and add 1
                - "sigmoid": Apply sigmoid function
                - "sigmoid2": Apply sigmoid and multiply by 2
            - std has minimum value added from self._min_std
        """
        
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
                "softplus": lambda: torch.nn.functional.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """
        Calculate Kullback-Leibler divergence loss between posterior and prior distributions.
        
        This method computes two components of KL divergence:
        1. Representation loss: KL divergence from posterior to prior (with prior gradients detached)
        2. Dynamics loss: KL divergence from posterior (with gradients detached) to prior
        
        Both losses are clipped to a minimum value of 'free' and then combined with scaling factors.
        
        Args:
            post (dict): Posterior distribution parameters
            prior (dict): Prior distribution parameters
            free (float): Minimum value for KL divergence (free bits)
            dyn_scale (float): Scaling factor for dynamics loss
            rep_scale (float): Scaling factor for representation loss
            
        Returns:
            tuple:
                - loss (Tensor): Combined KL divergence loss
                - value (Tensor): Raw representation loss before clipping
                - dyn_loss (Tensor): Clipped dynamics loss
                - rep_loss (Tensor): Clipped representation loss
        """
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
        rep_loss = torch.clamp(rep_loss, min=free)
        dyn_loss = torch.clamp(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class _TSSMBlock(nn.Module):
    """One pre-LN causal attention block of the TSSM deterministic pathway.

    Structure (pre-LN transformer block):
        x = x + o_proj(attention(ln1(x)))
        x = x + mlp(ln2(x))

    Position information is a T5-style learned relative bias per (head, offset) added to the
    attention logits, offset in [0, ctx_len] (0 = attending to self). Relative rather than
    absolute so deployment-length rollouts never run off a positional-embedding table.

    Two entry points that must stay mathematically identical (see
    ``test_tssm_parallel_observe_matches_sequential``):
      - ``forward_parallel``: whole sequence in one masked attention pass (training).
      - ``forward_step``: one token against a K/V cache of past tokens (imagination/deploy).

    The returned k/v are projections of the block *input* (post-ln1) for that position — these
    are what the sequential path caches, and what the parallel path computes for all positions
    at once.
    """

    def __init__(self, size: int, num_heads: int, ctx_len: int, ff_mult: int) -> None:
        super().__init__()
        assert size % num_heads == 0, f"size {size} must be divisible by num_heads {num_heads}"
        self._size = size
        self._num_heads = num_heads
        self._head_dim = size // num_heads

        self.ln1 = nn.LayerNorm(size, eps=1e-3)
        self.q_proj = nn.Linear(size, size, bias=False)
        self.k_proj = nn.Linear(size, size, bias=False)
        self.v_proj = nn.Linear(size, size, bias=False)
        self.o_proj = nn.Linear(size, size, bias=False)
        self.ln2 = nn.LayerNorm(size, eps=1e-3)
        self.mlp = nn.Sequential(
            nn.Linear(size, ff_mult * size, bias=False),
            nn.SiLU(),
            nn.Linear(ff_mult * size, size, bias=False),
        )
        # T5-style relative attention bias per head over window offsets [0..ctx_len].
        # Zero-init: no positional preference until learned. (nn.Parameter, so untouched
        # by tools.weight_init, which only visits Linear/LayerNorm modules.)
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, ctx_len + 1))

    def _attn_bias(
        self,
        allowed: "torch.Tensor",  # bool, broadcastable to (B, 1, Q, K)
        offsets: "torch.Tensor",  # long, (Q, K) or (K,)
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        rel = self.rel_bias[:, offsets]  # (H, Q, K) or (H, K)
        if rel.dim() == 2:
            rel = rel.unsqueeze(1)  # (H, 1, K)
        rel = rel.unsqueeze(0).to(dtype)  # (1, H, Q, K)
        neg = torch.finfo(dtype).min
        return torch.where(allowed, rel, torch.full_like(rel, neg))

    def _split_heads(self, x: "torch.Tensor", batch: int, length: int) -> "torch.Tensor":
        return x.view(batch, length, self._num_heads, self._head_dim).transpose(1, 2)

    def forward_parallel(self, x, allowed, offsets):
        # x: (B, T, size); allowed: (B, 1, T, T) bool; offsets: (T, T) long
        batch, length, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = F.scaled_dot_product_attention(
            self._split_heads(q, batch, length),
            self._split_heads(k, batch, length),
            self._split_heads(v, batch, length),
            attn_mask=self._attn_bias(allowed, offsets, q.dtype),
        )
        attn = attn.transpose(1, 2).reshape(batch, length, self._size)
        x = x + self.o_proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x, k, v

    def forward_step(self, x, k_cache, v_cache, allowed, offsets):
        # x: (B, size); caches: (B, S, size); allowed: (B, 1, 1, S+1); offsets: (S+1,)
        batch, _ = x.shape
        h = self.ln1(x)
        q, k_self, v_self = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        k = torch.cat([k_cache, k_self.unsqueeze(1)], dim=1)  # (B, S+1, size)
        v = torch.cat([v_cache, v_self.unsqueeze(1)], dim=1)
        length = k.shape[1]
        attn = F.scaled_dot_product_attention(
            self._split_heads(q.unsqueeze(1), batch, 1),
            self._split_heads(k, batch, length),
            self._split_heads(v, batch, length),
            attn_mask=self._attn_bias(allowed, offsets, q.dtype),
        )
        attn = attn.transpose(1, 2).reshape(batch, self._size)
        x = x + self.o_proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x, k_self, v_self


class TSSM(RSSM):
    """Transformer State-Space Model (M5.2): sequence-parallel world-model training.

    STORM/TransDreamer-style restructuring of the RSSM. Two coupled changes:

    1. **Posterior decoupled from the deterministic path**: q(z_t | o_t) conditions on the
       observation embedding only (RSSM: q(z_t | h_t, o_t)). This is a genuine change to the
       probabilistic model, not an optimization — the posterior loses h-conditioning, which is
       the price of parallelism (STORM demonstrates it works well in practice).
    2. **Deterministic path as causal attention over (z, a) tokens**: h_t attends over the
       token sequence u_s = f(z_{s-1}, a_{s-1}[, b]) for s in the last ctx_len steps of the
       same episode segment.

    Because of (1), during world-model training all posteriors can be sampled from the logged
    observations at once, then all h_t computed in ONE parallel masked-attention pass, then all
    priors p(z_t | h_t) in parallel — ``observe()`` has no sequential scan. This is the actual
    transformer training-speed advantage the recurrent-cell backbones (GRUCell/TransformerCell)
    structurally cannot have.

    Imagination (``imagine_with_action``) remains sequential regardless of backbone: each step
    consumes the model's own just-sampled z. It runs step-by-step with a per-layer K/V cache
    (same idea as TransformerCell's cache, M5.1.1).

    State dict extends RSSM's with:
      - ``tssm_cache``: (B, ctx_len, num_layers*2*deter) per-layer K/V ring buffer,
        right-filled (newest at index -1), layer l at slice [l*2D, (l+1)*2D) as (k, v).
      - ``tssm_cnt``:   (B, 1) count of valid cache slots (0 after reset).

    Memory note: ``observe()``'s returned post carries a per-position cache
    (B, T, ctx_len, num_layers*2*deter) so that every position can serve as an imagination
    start (Dreamer flattens post over (B, T)). This scales linearly in num_layers and ctx_len —
    the price of a non-Markovian deterministic state. Keep num_layers/ctx_len modest.
    """

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
        cell_type: str = "tssm",
        transformer_ctx_len: int = 64,
        transformer_num_heads: int = 4,
        context_size: int = 0,
        tssm_num_layers: int = 2,
        tssm_ff_mult: int = 2,
    ):
        super().__init__(
            stoch,
            deter,
            hidden,
            rec_depth,
            discrete,
            act,
            norm,
            mean_act,
            std_act,
            min_std,
            unimix_ratio,
            initial,
            num_actions,
            embed,
            device,
            cell_type="tssm",
            transformer_ctx_len=transformer_ctx_len,
            transformer_num_heads=transformer_num_heads,
            context_size=context_size,
        )
        self._ctx_len = transformer_ctx_len
        self._num_layers = tssm_num_layers
        act_cls = getattr(torch.nn, act)

        # Posterior is observation-only (the decoupling that makes parallel observe()
        # possible): rebuild _obs_out_layers with embed-only input. The parent built it
        # with deter+embed input; this replaces it before any forward pass.
        obs_out_layers = [nn.Linear(self._embed, self._hidden, bias=False)]
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act_cls())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        # Token embedding: _img_in_layers output (hidden) -> attention width (deter).
        self._token_in = nn.Linear(self._hidden, self._deter, bias=False)
        self._token_in.apply(tools.weight_init)
        self._blocks = nn.ModuleList(
            _TSSMBlock(self._deter, transformer_num_heads, transformer_ctx_len, tssm_ff_mult)
            for _ in range(tssm_num_layers)
        )
        self._blocks.apply(tools.weight_init)
        self._ln_f = nn.LayerNorm(self._deter, eps=1e-03)

    def initial(self, batch_size):
        state = super().initial(batch_size)
        state["tssm_cache"] = torch.zeros(
            batch_size,
            self._ctx_len,
            self._num_layers * 2 * self._deter,
            device=self._device,
        )
        state["tssm_cnt"] = torch.zeros(batch_size, 1, device=self._device)
        return state

    def _token_input(self, prev_stoch, prev_action, context):
        """Shared token construction for img_step (per step) and observe (per sequence)."""
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        if context is not None and self._context_size > 0:
            x = torch.cat([prev_stoch, prev_action, context], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
        return self._token_in(self._img_in_layers(x))

    def observe(self, embed, action, is_first, state=None, context=None, sample=True):
        """Sequence-parallel world-model training pass — no static_scan.

        Args/returns match RSSM.observe (batch-first in and out); ``sample=False`` uses
        distribution modes everywhere (used by the parallel==sequential equivalence test).
        """
        batch, length = embed.shape[:2]
        state_was_none = state is None
        if state_was_none:
            state = self.initial(batch)
        else:
            # Warm-cache continuation is a sequential (obs_step) concern; the training
            # path always starts fresh (models.py calls observe without state).
            assert torch.all(state["tssm_cnt"] == 0), (
                "TSSM.observe requires an empty attention cache"
            )
        is_first = is_first.to(torch.float32)
        if state_was_none:
            # RSSM.obs_step resets to initial state + zero action whenever prev_state is
            # None, regardless of is_first[:, 0] — replicate that exactly.
            is_first = is_first.clone()
            is_first[:, 0] = 1.0

        # --- posterior q(z_t | o_t): fully parallel over T ---
        x = self._obs_out_layers(embed)  # (B, T, hidden)
        post_stats = self._suff_stats_layer("obs", x)
        post_dist = self.get_dist(post_stats)
        post_stoch = post_dist.sample() if sample else post_dist.mode()

        # --- tokens u_t = f(z_{t-1}, a_{t-1}[, b]) with is_first resets ---
        prev_stoch = torch.cat([state["stoch"].unsqueeze(1), post_stoch[:, :-1]], dim=1)
        init = self.initial(batch)
        f_stoch = is_first.reshape(batch, length, *([1] * (prev_stoch.dim() - 2)))
        prev_stoch = prev_stoch * (1.0 - f_stoch) + init["stoch"].unsqueeze(1) * f_stoch
        prev_action = action * (1.0 - is_first.unsqueeze(-1))
        ctx_seq = (
            context.unsqueeze(1).expand(batch, length, -1)
            if context is not None and self._context_size > 0
            else None
        )
        tokens = self._token_input(prev_stoch, prev_action, ctx_seq)  # (B, T, deter)

        # --- causal segment-window attention, one parallel pass ---
        # allowed[b, q, s]: s is in q's past (incl. self), within the sliding window, and in
        # the same episode segment (attention never crosses an is_first boundary — the
        # parallel equivalent of the sequential cache reset).
        seg = torch.cumsum(is_first, dim=1)  # (B, T)
        idx = torch.arange(length, device=embed.device)
        off = idx[:, None] - idx[None, :]  # (T, T): query index - key index
        allowed = (
            (off >= 0)
            & (off <= self._ctx_len)
            & (seg[:, :, None] == seg[:, None, :])
        ).unsqueeze(1)  # (B, 1, T, T)
        offsets = off.clamp(0, self._ctx_len)  # (T, T)

        h = tokens
        layer_kv = []
        for blk in self._blocks:
            h, k, v = blk.forward_parallel(h, allowed, offsets)
            layer_kv.append(torch.cat([k, v], dim=-1))  # (B, T, 2*deter)
        deter = self._ln_f(h)  # (B, T, deter)

        # --- prior p(z_t | h_t): parallel ---
        x = self._img_out_layers(deter)
        prior_stats = self._suff_stats_layer("ims", x)
        prior_dist = self.get_dist(prior_stats)
        prior_stoch = prior_dist.sample() if sample else prior_dist.mode()

        # --- per-position K/V caches so every position can start an imagination rollout ---
        # Window for position t = tokens [t-S+1 .. t] (right-filled), matching the sequential
        # cache state *after* step t (img_step appends its own token before returning).
        token_kv = torch.cat(layer_kv, dim=-1)  # (B, T, L*2*deter)
        pad = torch.zeros(
            batch, self._ctx_len, token_kv.shape[-1],
            dtype=token_kv.dtype, device=token_kv.device,
        )  # full S zeros even when T < S (zeros_like on a slice would truncate)
        padded = torch.cat([pad, token_kv], dim=1)  # (B, S+T, L*2*D)
        # unfold -> (B, T+1, L*2*D, S); windows starting at 1 give [t-S+1 .. t].
        windows = padded.unfold(1, self._ctx_len, 1)[:, 1 : length + 1]
        cache = windows.permute(0, 1, 3, 2)  # (B, T, S, L*2*D)
        # Valid-slot count = tokens since segment start (inclusive), capped at S.
        seg_start = torch.cummax(is_first * idx.to(is_first.dtype), dim=1).values  # (B, T)
        cnt = (idx.to(is_first.dtype) - seg_start + 1.0).clamp(max=self._ctx_len)
        cnt = cnt.unsqueeze(-1)  # (B, T, 1)
        # Zero out cross-segment slots (tokens from before the segment's is_first reset).
        # They are already excluded by the cnt mask at attention time, but the sequential
        # path zeroes them (is_first blend resets the cache), so enforce the same
        # invariant here: invalid slots hold zeros, byte-identical to obs_step's cache.
        slot_pos = idx[:, None] - (self._ctx_len - 1) + torch.arange(
            self._ctx_len, device=embed.device
        )[None, :]  # (T, S): token position held by slot j for query position t
        slot_valid = (
            slot_pos[None].to(is_first.dtype) >= seg_start[:, :, None]
        )  # (B, T, S)
        cache = cache * slot_valid.unsqueeze(-1).to(cache.dtype)

        post = {
            "stoch": post_stoch,
            "deter": deter,
            **post_stats,
            "tssm_cache": cache,
            "tssm_cnt": cnt,
        }
        prior = {"stoch": prior_stoch, "deter": deter, **prior_stats}
        return post, prior

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, context=None):
        """Single-step posterior update (deployment path). Posterior is obs-only."""
        # is_first initialization/blending: identical to RSSM.obs_step (init cache/cnt are
        # zeros, so the elementwise blend implements the cache reset).
        if prev_state == None or torch.sum(is_first) == len(is_first):  # noqa: E711
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action = prev_action * (1.0 - is_first)
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action, context=context)
        x = self._obs_out_layers(embed)  # obs-only: the STORM decoupling
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {
            "stoch": stoch,
            "deter": prior["deter"],
            **stats,
            "tssm_cache": prior["tssm_cache"],
            "tssm_cnt": prior["tssm_cnt"],
        }
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True, context=None):
        """One imagination step: attend the new token over the per-layer K/V cache."""
        batch = prev_action.shape[0]
        token = self._token_input(prev_state["stoch"], prev_action, context)  # (B, deter)
        cache = prev_state["tssm_cache"]  # (B, S, L*2*D)
        cnt = prev_state["tssm_cnt"]  # (B, 1)

        # Slot j holds the token at offset S-j from the current step; slot valid iff among
        # the last cnt written. Self (appended column) is always valid, offset 0.
        slots = torch.arange(self._ctx_len, device=token.device)
        valid = slots[None, :] >= (self._ctx_len - cnt)  # (B, S)
        allowed = torch.cat(
            [valid, torch.ones(batch, 1, dtype=torch.bool, device=token.device)], dim=1
        )[:, None, None]  # (B, 1, 1, S+1)
        offsets = torch.cat(
            [
                torch.arange(self._ctx_len, 0, -1, device=token.device),
                torch.zeros(1, dtype=torch.long, device=token.device),
            ]
        )  # (S+1,)

        width = 2 * self._deter
        h = token
        new_kv = []
        for layer_i, blk in enumerate(self._blocks):
            k_cache = cache[..., layer_i * width : layer_i * width + self._deter]
            v_cache = cache[..., layer_i * width + self._deter : (layer_i + 1) * width]
            h, k_new, v_new = blk.forward_step(h, k_cache, v_cache, allowed, offsets)
            new_kv.append(torch.cat([k_new, v_new], dim=-1))  # (B, 2*deter)
        deter = self._ln_f(h)

        new_cache = torch.cat(
            [cache[:, 1:], torch.cat(new_kv, dim=-1).unsqueeze(1)], dim=1
        )
        new_cnt = (cnt + 1.0).clamp(max=self._ctx_len)

        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        return {
            "stoch": stoch,
            "deter": deter,
            **stats,
            "tssm_cache": new_cache,
            "tssm_cnt": new_cnt,
        }

    def imagine_with_action(self, action, state, context=None):
        """Sequential rollout with a chained (not stored) cache.

        Manual loop instead of static_scan: static_scan stacks every state entry per step,
        which would multiply imagination memory by the horizon for the (B, S, L*2*D) cache.
        The cache is threaded through the loop and only (stoch, deter, stats) are collected.
        """
        assert isinstance(state, dict), state
        outputs = []
        current = state
        for t in range(action.shape[1]):
            current = self.img_step(current, action[:, t], context=context)
            outputs.append(
                {k: v for k, v in current.items() if k not in ("tssm_cache", "tssm_cnt")}
            )
        return {k: torch.stack([o[k] for o in outputs], dim=1) for k in outputs[0]}


class MultiEncoder(nn.Module):
    """A flexible multi-modal encoder combining CNN and MLP networks for processing different observation types.
    
    This encoder handles both image-like (CNN-compatible) and vector (MLP-compatible) inputs, routing them
    through appropriate neural network architectures and concatenating their features. It automatically 
    determines which inputs should go to CNN vs MLP based on input shapes and regex patterns.
        shapes (dict): Dictionary mapping observation keys to their shapes.

    Attributes:
        cnn_shapes (dict): Shapes of observations to be processed by CNN.
        mlp_shapes (dict): Shapes of observations to be processed by MLP.
        outdim (int): Total output dimension of the combined encoder.
        _cnn (ConvEncoder, optional): CNN encoder network if CNN inputs are present.
        _mlp (MLP, optional): MLP encoder network if MLP inputs are present.
    
    Notes:
        - The encoder automatically filters out excluded keys like 'is_first', 'is_terminal', etc.
        - The output dimension is the sum of CNN and MLP output dimensions.
        - For CNN inputs, all observations are stacked along the channel dimension.
    """
    
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
    """"MultiDecoder is a versatile neural network module for decoding latent features into multiple output formats.

    This decoder can handle both CNN-based outputs (e.g., images) and MLP-based outputs (e.g., vectors),
    supporting different distribution types for each. It automatically routes outputs to the appropriate
    decoder network based on shape and naming patterns.

    Attributes:
        cnn_shapes (dict): Dictionary of shapes for CNN outputs
        mlp_shapes (dict): Dictionary of shapes for MLP outputs
        _cnn (ConvDecoder, optional): CNN-based decoder network
        _mlp (MLP, optional): MLP-based decoder network
        _image_dist (str): Type of distribution to use for image outputs

    Example:
        ```python
        decoder = MultiDecoder(
            feat_size=128,
            shapes={"image": (64, 64, 3), "vector": (10,)},
            mlp_keys=".*vector.*",
            cnn_keys=".*image.*",
            act="relu",
            norm="none",
            cnn_depth=3,
            kernel_size=4,
            minres=4,
            mlp_layers=2,
            mlp_units=100,
            cnn_sigmoid=True,
            image_dist="normal",
            vector_dist="normal",
            outscale=1.0
        
        features = torch.randn(32, 128)  # Batch size 32, feature size 128
        distributions = decoder(features)
        ```
    """
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


class TransformerCell(nn.Module):
    """Causal sliding-window attention cell for TransDreamerV3 (M5.1).

    Replaces the GRU in RSSM with a fixed-window causal attention mechanism.
    State shape: ``(B, ctx_len, 2*size)`` — a ring buffer of cached K/V projections of
    past outputs (K and V concatenated on the last dim), not raw context vectors.

    At each step:
      1. Project input → query ``(B, size)``, plus its own K/V ("current token queries
         all history, including itself" — see below)
      2. Attend the query over ``[cached K/V of past outputs; self K/V]``
      3. Output ``(B, size)`` via linear + residual + layernorm
      4. Project the *output* to new K/V, shift window: drop oldest cache entry, append

    Interface matches ``GRUCell``:
      ``forward(inputs, [ctx_state]) -> (output, [new_ctx_state])``

    Shape note: ``state["deter"]`` in the RSSM state dict always holds the current-step
    output ``(B, deter_dim)``.  The context window ``(B, ctx_len, 2*deter_dim)`` is stored
    separately in ``state["deter_seq"]`` and is invisible to ``get_feat``.

    KV-cache rationale: the naive formulation (attend over a raw ``(B, ctx_len, size)``
    history and re-run the joint Q/K/V projection on the whole window every step) redoes
    the K/V projection of ``ctx_len`` unchanged past outputs on every single step —
    ``O(ctx_len)`` projection work per step. Since those past outputs never change once
    written, their K/V projections are cached here instead of recomputed, cutting the
    per-step K/V cost to ``O(1)`` (one new output's worth). This is a pure performance
    optimization: the attention computation itself, and therefore the output, is
    unchanged (see ``test_kv_cache_matches_naive_recompute``).

    Note (scope): this still does not give the usual transformer training-speed
    advantage of parallelizing over a whole known sequence in one matmul — it is called
    once per timestep inside the same sequential ``static_scan`` loop GRU uses, for both
    ``observe`` and ``imagine_with_action``. Full sequence-parallel training would require
    decoupling the deterministic pathway from the interleaved posterior/prior computation
    in ``RSSM.observe`` — a larger, unimplemented restructuring (tracked as M5.2), and is
    fundamentally inapplicable to ``imagine_with_action`` regardless of backbone, since
    each imagined step depends on the model's own just-sampled stochastic latent.
    """

    def __init__(
        self,
        inp_size: int,
        size: int,
        num_heads: int = 4,
        ctx_len: int = 64,
        norm: bool = True,
    ) -> None:
        super().__init__()
        assert size % num_heads == 0, f"size {size} must be divisible by num_heads {num_heads}"
        self._size = size
        self._ctx_len = ctx_len
        self._num_heads = num_heads
        self._head_dim = size // num_heads

        # Project input to the attention embedding space.
        self.inp_proj = nn.Linear(inp_size, size, bias=False)
        # Manual Q/K/V projections (replaces nn.MultiheadAttention) so K/V of past
        # outputs can be cached instead of recomputed from raw history every step.
        self.q_proj = nn.Linear(size, size, bias=False)
        self.k_proj = nn.Linear(size, size, bias=False)
        self.v_proj = nn.Linear(size, size, bias=False)
        self.out_proj = nn.Linear(size, size, bias=False)
        self.norm = nn.LayerNorm(size, eps=1e-3) if norm else nn.Identity()

    @property
    def state_size(self) -> int:
        return self._size

    @property
    def ctx_len(self) -> int:
        return self._ctx_len

    def _split_heads(self, x: "torch.Tensor", batch: int, length: int) -> "torch.Tensor":
        # (B, L, size) -> (B, H, L, head_dim)
        return x.view(batch, length, self._num_heads, self._head_dim).transpose(1, 2)

    def forward(
        self,
        inputs: "torch.Tensor",    # (B, inp_size)
        state: list,               # [(B, ctx_len, 2*size)] -- [..., :size]=K cache, [..., size:]=V cache
    ):
        batch = inputs.shape[0]
        kv_cache = state[0]                              # (B, L, 2*size)
        k_hist, v_hist = kv_cache[..., : self._size], kv_cache[..., self._size :]

        q_in = self.inp_proj(inputs)                     # (B, size)
        q = self.q_proj(q_in)                             # (B, size)
        # The current token is also a key/value candidate for its own query, matching
        # the original "kv = cat([ctx, q])" self-attention behavior.
        k_self = self.k_proj(q_in)                        # (B, size)
        v_self = self.v_proj(q_in)                        # (B, size)

        k = torch.cat([k_hist, k_self.unsqueeze(1)], dim=1)  # (B, L+1, size)
        v = torch.cat([v_hist, v_self.unsqueeze(1)], dim=1)  # (B, L+1, size)
        length = k.shape[1]

        qh = self._split_heads(q.unsqueeze(1), batch, 1)          # (B, H, 1, D)
        kh = self._split_heads(k, batch, length)                   # (B, H, L+1, D)
        vh = self._split_heads(v, batch, length)                   # (B, H, L+1, D)

        attn_out = F.scaled_dot_product_attention(qh, kh, vh)      # (B, H, 1, D)
        out_raw = attn_out.transpose(1, 2).reshape(batch, self._size)  # (B, size)
        out = self.norm(self.out_proj(out_raw) + out_raw)  # residual + norm

        # New cache entries are projections of the *output* (matches the original
        # design, where future steps attend to past outputs, not raw inputs).
        k_out = self.k_proj(out)                          # (B, size)
        v_out = self.v_proj(out)                          # (B, size)
        new_k_cache = torch.cat([k_hist[:, 1:], k_out.unsqueeze(1)], dim=1)  # (B, L, size)
        new_v_cache = torch.cat([v_hist[:, 1:], v_out.unsqueeze(1)], dim=1)  # (B, L, size)
        new_kv_cache = torch.cat([new_k_cache, new_v_cache], dim=-1)          # (B, L, 2*size)

        return out, [new_kv_cache]


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
