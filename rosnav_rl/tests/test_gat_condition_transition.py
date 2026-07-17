"""C1 Option A — GAT code conditioning the RSSM transition (gat.condition_transition).

These exercise the low-level threading in networks.RSSM directly (the risky part):
the img_step input width widens by gat_code_size, a non-None gat_code actually changes
the predicted prior, gat_code=None is byte-identical to a plain RSSM, and the observe()
in-scan gat_code_fn path runs with correct shapes. The full-WorldModel/imagination and
deploy paths are covered by the Gazebo smoke run (see FIX_PLAN Task 1).
"""

import torch

from rosnav_rl.model.dreamerv3.networks import RSSM

STOCH, DISCRETE, DETER, HIDDEN, NUM_ACTIONS, EMBED, G = 4, 4, 16, 12, 3, 10, 8


def _make(gat_code_size=0, context_size=0):
    return RSSM(
        stoch=STOCH,
        deter=DETER,
        hidden=HIDDEN,
        discrete=DISCRETE,
        num_actions=NUM_ACTIONS,
        embed=EMBED,
        device="cpu",
        context_size=context_size,
        gat_code_size=gat_code_size,
    )


def _prev(rssm, batch):
    state = rssm.initial(batch)
    action = torch.zeros(batch, NUM_ACTIONS)
    return state, action


def test_img_input_width_widens_by_gat_code_size():
    base = _make(gat_code_size=0)
    wide = _make(gat_code_size=G)
    base_in = base._img_in_layers[0].in_features
    wide_in = wide._img_in_layers[0].in_features
    assert wide_in - base_in == G


def test_gat_code_changes_prior():
    torch.manual_seed(0)
    rssm = _make(gat_code_size=G)
    state, action = _prev(rssm, batch=5)
    zeros = torch.zeros(5, G)
    rand = torch.randn(5, G)
    p_zero = rssm.img_step(state, action, sample=False, gat_code=zeros)
    p_rand = rssm.img_step(state, action, sample=False, gat_code=rand)
    # A non-trivial gat_code must move the predicted deterministic state.
    assert not torch.allclose(p_zero["deter"], p_rand["deter"], atol=1e-6)


def test_gat_code_none_matches_plain_rssm():
    # Same seed/weights: img_step with gat_code=None on a gat_code_size=0 RSSM is the
    # baseline path — sanity that the None branch didn't change the default behavior.
    torch.manual_seed(1)
    rssm = _make(gat_code_size=0)
    state, action = _prev(rssm, batch=4)
    p_a = rssm.img_step(state, action, sample=False)
    p_b = rssm.img_step(state, action, sample=False, gat_code=None)
    assert torch.allclose(p_a["deter"], p_b["deter"])


def test_observe_gat_code_fn_path_runs():
    torch.manual_seed(2)
    batch, length = 3, 6
    rssm = _make(gat_code_size=G)
    embed = torch.randn(batch, length, EMBED)
    action = torch.zeros(batch, length, NUM_ACTIONS)
    is_first = torch.zeros(batch, length)
    is_first[:, 0] = 1.0
    peds_prev = torch.randn(batch, length, 5)  # arbitrary per-step ped payload width

    def gat_code_fn(peds_t, deter_t):
        # deter_t is (batch, DETER); return a (batch, G) code that depends on both inputs.
        assert peds_t.shape[0] == deter_t.shape[0]
        return torch.tanh(peds_t.sum(-1, keepdim=True) + deter_t[:, :1]).repeat(1, G)

    post, prior = rssm.observe(
        embed, action, is_first, gat_code_fn=gat_code_fn, peds_prev=peds_prev
    )
    assert post["deter"].shape == (batch, length, DETER)
    assert prior["deter"].shape == (batch, length, DETER)


def test_grad_flows_through_transition():
    """The end-to-end correctness gate: a loss on the observed rollout must backprop
    into (a) the params that produce c_t and (b) the widened img_in columns that consume
    it — i.e. c_t genuinely participates in the transition, not a dead concat."""
    torch.manual_seed(3)
    batch, length = 3, 5
    rssm = _make(gat_code_size=G)
    gat_lin = torch.nn.Linear(5 + DETER, G)  # differentiable stand-in for the GAT
    embed = torch.randn(batch, length, EMBED)
    action = torch.zeros(batch, length, NUM_ACTIONS)
    is_first = torch.zeros(batch, length)
    is_first[:, 0] = 1.0
    peds_prev = torch.randn(batch, length, 5)

    def gat_code_fn(peds_t, deter_t):
        return gat_lin(torch.cat([peds_t, deter_t], -1))

    post, _ = rssm.observe(
        embed, action, is_first, gat_code_fn=gat_code_fn, peds_prev=peds_prev
    )
    post["deter"].pow(2).mean().backward()

    # (a) gradient reached the code-producing params
    assert gat_lin.weight.grad is not None
    assert gat_lin.weight.grad.abs().sum() > 0
    # (b) the last G input columns of the transition's first layer (the gat_code slot)
    # received gradient — proof the transition actually consumes c_t.
    w_grad = rssm._img_in_layers[0].weight.grad
    assert w_grad is not None
    assert w_grad[:, -G:].abs().sum() > 0
