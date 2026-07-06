"""Tests for bswap_divergence (Item 3: counterfactual b-swap imagination divergence).

Run with ROS environment sourced (or in the arena container):
    cd src/Arena/arena_training/deps/rosnav_rl/rosnav_rl
    python -m pytest tests/test_counterfactual.py -v
"""

import types

import pytest
import torch

from rosnav_rl.model.dreamerv3 import networks
from rosnav_rl.model.dreamerv3.social.counterfactual import bswap_divergence

CELL_TYPES = ("gru", "transformer", "tssm")


def _make_wm(cell_type, b_dim=3, ped_dim=6):
    common = dict(
        stoch=4, deter=16, hidden=12, discrete=4, num_actions=2, embed=10,
        device="cpu", transformer_ctx_len=4, transformer_num_heads=2,
        context_size=b_dim,
    )
    if cell_type == "tssm":
        dyn = networks.TSSM(**common, tssm_num_layers=1)
    else:
        dyn = networks.RSSM(**common, cell_type=cell_type)
    feat_dim = dyn.get_feat(dyn.initial(1)).shape[-1]
    head = networks.MLP(
        feat_dim, (ped_dim,), layers=1, units=8, dist="normal",
        device="cpu", name="test_ped_head",
    )
    return types.SimpleNamespace(dynamics=dyn, heads={"PedestrianNodeSetSpace": head})


def _start_state(wm, batch, warmup_len, b_dim):
    """Posterior state from a real observe() pass -- never dynamics.initial().

    For the transformer/TSSM backbones this is what carries a populated K/V cache;
    imagining from ``initial()`` would diverge the two rollouts for reasons unrelated
    to ``b`` (an empty cache), which is exactly the trap Item 3's spec calls out.
    The dynamics was built with context_size=b_dim > 0, so observe() needs a (throwaway)
    warmup context too -- its value doesn't matter, only that some real posterior/cache
    state comes out the other end.
    """
    dyn = wm.dynamics
    embed = torch.randn(batch, warmup_len, dyn._embed)
    action = torch.randn(batch, warmup_len, dyn._num_actions)
    is_first = torch.zeros(batch, warmup_len)
    is_first[:, 0] = 1.0
    warmup_b = torch.randn(batch, b_dim)
    post, _ = dyn.observe(embed, action, is_first, context=warmup_b)
    return {k: v[:, -1] for k, v in post.items()}


@pytest.mark.parametrize("cell_type", CELL_TYPES)
class TestBswapDivergence:
    def test_shape_is_horizon_and_nonnegative(self, cell_type):
        torch.manual_seed(0)
        b_dim, batch, horizon = 3, 5, 4
        wm = _make_wm(cell_type, b_dim=b_dim)
        start = _start_state(wm, batch, warmup_len=3, b_dim=b_dim)
        actions = torch.randn(batch, horizon, wm.dynamics._num_actions)
        b_a = torch.randn(batch, b_dim)
        b_b = torch.randn(batch, b_dim)

        div = bswap_divergence(wm, start, actions, b_a, b_b)

        assert div.shape == (horizon,)
        assert torch.all(div >= 0.0)

    def test_identical_b_gives_exactly_zero_divergence(self, cell_type):
        # img_step samples stochastically and imagine_with_action doesn't expose `sample`;
        # bswap_divergence controls for this by snapshotting/restoring the RNG state around
        # the paired rollouts (see its docstring), so identical b must give identical draws.
        torch.manual_seed(0)
        b_dim, batch, horizon = 3, 4, 5
        wm = _make_wm(cell_type, b_dim=b_dim)
        start = _start_state(wm, batch, warmup_len=3, b_dim=b_dim)
        actions = torch.randn(batch, horizon, wm.dynamics._num_actions)
        b = torch.randn(batch, b_dim)

        div = bswap_divergence(wm, start, actions, b, b)

        assert torch.allclose(div, torch.zeros_like(div), atol=1e-6)

    def test_different_b_gives_nonzero_divergence(self, cell_type):
        # Sanity check the metric actually responds to swapping b (context_size > 0 wires
        # b into the transition, per networks.py img_step/_token_input).
        torch.manual_seed(0)
        b_dim, batch, horizon = 3, 4, 5
        wm = _make_wm(cell_type, b_dim=b_dim)
        start = _start_state(wm, batch, warmup_len=3, b_dim=b_dim)
        actions = torch.randn(batch, horizon, wm.dynamics._num_actions)
        b_a = torch.randn(batch, b_dim)
        b_b = b_a + 10.0  # far apart -> should not coincidentally land at zero divergence

        div = bswap_divergence(wm, start, actions, b_a, b_b)

        assert torch.any(div > 1e-6)
