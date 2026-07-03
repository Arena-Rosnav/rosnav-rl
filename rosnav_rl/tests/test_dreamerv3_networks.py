"""Tests for the RSSM recurrent cell implementations (GRUCell, TransformerCell).

Run with ROS environment sourced:
    cd ~/arena_ws && source arena
    cd src/Arena/arena_training/deps/rosnav_rl/rosnav_rl
    python -m pytest tests/test_dreamerv3_networks.py -v
"""

import pytest
import torch

from rosnav_rl.model.dreamerv3.networks import TSSM, GRUCell, TransformerCell


# =====================================================================
#  TransformerCell
# =====================================================================

class TestTransformerCell:
    def _make_cell(self, size=64, num_heads=4, ctx_len=8, inp_size=32):
        return TransformerCell(
            inp_size=inp_size, size=size, num_heads=num_heads, ctx_len=ctx_len
        )

    def test_output_and_state_shapes(self):
        batch, size, ctx_len, inp_size = 5, 64, 8, 32
        cell = self._make_cell(size=size, ctx_len=ctx_len, inp_size=inp_size)
        ctx = torch.zeros(batch, ctx_len, 2 * size)
        inputs = torch.randn(batch, inp_size)

        out, new_state = cell(inputs, [ctx])

        assert out.shape == (batch, size)
        assert len(new_state) == 1
        assert new_state[0].shape == (batch, ctx_len, 2 * size)

    def test_context_window_slides_fifo(self):
        # Feed three distinct steps and confirm the K/V cache is a proper sliding FIFO
        # window: after each step, the newest slot holds k_proj(out)/v_proj(out) for
        # this step's output, and the oldest slot is dropped.
        batch, size, ctx_len, inp_size = 2, 16, 3, 8
        cell = self._make_cell(size=size, ctx_len=ctx_len, inp_size=inp_size)
        ctx = torch.zeros(batch, ctx_len, 2 * size)

        outputs = []
        for _ in range(ctx_len + 1):
            inputs = torch.randn(batch, inp_size)
            out, (ctx,) = cell(inputs, [ctx])
            outputs.append(out)

        # After ctx_len+1 steps, the cache holds k_proj/v_proj of the last ctx_len
        # outputs, oldest first.
        k_cache, v_cache = ctx[..., :size], ctx[..., size:]
        for i in range(ctx_len):
            expected_out = outputs[-(ctx_len - i)]
            with torch.no_grad():
                expected_k = cell.k_proj(expected_out)
                expected_v = cell.v_proj(expected_out)
            assert torch.allclose(k_cache[:, i], expected_k, atol=1e-6)
            assert torch.allclose(v_cache[:, i], expected_v, atol=1e-6)

    def test_kv_cache_matches_naive_recompute(self):
        # The KV-cache is a pure performance optimization: caching k_proj(out)/v_proj(out)
        # of past outputs must be mathematically identical to recomputing those
        # projections from raw output history on every step. This test runs both
        # formulations side by side (sharing weights) across a window rollover and
        # asserts the outputs match exactly.
        batch, size, ctx_len, inp_size, num_heads = 3, 32, 5, 12, 4
        cell = self._make_cell(
            size=size, num_heads=num_heads, ctx_len=ctx_len, inp_size=inp_size
        )
        cell.eval()

        cached_state = torch.zeros(batch, ctx_len, 2 * size)
        raw_history = torch.zeros(batch, ctx_len, size)  # naive path: raw past outputs

        torch.manual_seed(0)
        with torch.no_grad():
            for _ in range(ctx_len + 3):
                inputs = torch.randn(batch, inp_size)

                cached_out, (cached_state,) = cell(inputs, [cached_state])

                naive_out = self._naive_forward(cell, inputs, raw_history, num_heads)
                raw_history = torch.cat(
                    [raw_history[:, 1:], naive_out.unsqueeze(1)], dim=1
                )

                assert torch.allclose(cached_out, naive_out, atol=1e-5), (
                    "KV-cached output diverged from naive full-recompute reference"
                )

    @staticmethod
    def _naive_forward(cell, inputs, raw_history, num_heads):
        """Reference: recompute K/V for the whole raw-output history every step."""
        batch, ctx_len, size = raw_history.shape
        head_dim = size // num_heads

        q_in = cell.inp_proj(inputs)
        q = cell.q_proj(q_in)
        k_self = cell.k_proj(q_in)
        v_self = cell.v_proj(q_in)

        k_hist = cell.k_proj(raw_history)  # (B, L, size) -- recomputed from scratch
        v_hist = cell.v_proj(raw_history)

        k = torch.cat([k_hist, k_self.unsqueeze(1)], dim=1)
        v = torch.cat([v_hist, v_self.unsqueeze(1)], dim=1)
        length = k.shape[1]

        def split_heads(x, length):
            return x.view(batch, length, num_heads, head_dim).transpose(1, 2)

        qh = split_heads(q.unsqueeze(1), 1)
        kh = split_heads(k, length)
        vh = split_heads(v, length)

        attn_out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
        out_raw = attn_out.transpose(1, 2).reshape(batch, size)
        return cell.norm(cell.out_proj(out_raw) + out_raw)

    def test_gradient_flows_to_input_projection_and_attention(self):
        batch, size, ctx_len, inp_size = 3, 32, 4, 16
        cell = self._make_cell(size=size, ctx_len=ctx_len, inp_size=inp_size)
        ctx = torch.zeros(batch, ctx_len, 2 * size)
        inputs = torch.randn(batch, inp_size, requires_grad=True)

        out, _ = cell(inputs, [ctx])
        out.sum().backward()

        assert inputs.grad is not None
        assert torch.any(inputs.grad != 0)
        assert cell.inp_proj.weight.grad is not None
        assert torch.any(cell.inp_proj.weight.grad != 0)
        for name, proj in (
            ("q_proj", cell.q_proj),
            ("k_proj", cell.k_proj),
            ("v_proj", cell.v_proj),
            ("out_proj", cell.out_proj),
        ):
            assert proj.weight.grad is not None, f"no grad for {name}"
            assert torch.any(proj.weight.grad != 0), f"zero grad for {name}"

    def test_size_must_be_divisible_by_num_heads(self):
        with pytest.raises(AssertionError):
            TransformerCell(inp_size=16, size=30, num_heads=4, ctx_len=8)

    def test_state_size_and_ctx_len_properties(self):
        cell = self._make_cell(size=64, ctx_len=8, inp_size=32)
        assert cell.state_size == 64
        assert cell.ctx_len == 8

    def test_interface_matches_gru_cell(self):
        # Both cells must be drop-in compatible: forward(inputs, [state]) -> (out, [new_state]),
        # since RSSM selects between them purely via a cell_type string (networks.py:162-175).
        batch, size, inp_size = 4, 32, 16
        gru = GRUCell(inp_size, size)
        transformer = self._make_cell(size=size, ctx_len=8, inp_size=inp_size)

        inputs = torch.randn(batch, inp_size)
        gru_out, gru_state = gru(inputs, [torch.zeros(batch, size)])
        t_out, t_state = transformer(inputs, [torch.zeros(batch, 8, 2 * size)])

        assert gru_out.shape == (batch, size)
        assert t_out.shape == (batch, size)
        assert isinstance(gru_state, list) and len(gru_state) == 1
        assert isinstance(t_state, list) and len(t_state) == 1


# =====================================================================
#  TSSM (M5.2): sequence-parallel observe()
# =====================================================================

class TestTSSM:
    def _make(self, ctx_len=4, num_layers=2, num_actions=3, embed=10):
        return TSSM(
            stoch=4,
            deter=16,
            hidden=12,
            discrete=4,
            num_actions=num_actions,
            embed=embed,
            device="cpu",
            transformer_ctx_len=ctx_len,
            transformer_num_heads=2,
            tssm_num_layers=num_layers,
        )

    def test_parallel_observe_matches_sequential(self):
        # THE correctness proof for M5.2: one parallel masked-attention pass over the
        # whole sequence must produce exactly what the step-by-step obs_step chain
        # produces — including across a mid-sequence episode reset (is_first) and a
        # sliding-window truncation (T > ctx_len). sample=False makes both paths
        # deterministic (posterior depends only on the per-step embed).
        torch.manual_seed(0)
        batch, length, ctx_len, num_actions, embed_dim = 2, 9, 4, 3, 10
        tssm = self._make(ctx_len=ctx_len, num_actions=num_actions, embed=embed_dim)
        tssm.eval()

        embed = torch.randn(batch, length, embed_dim)
        action = torch.randn(batch, length, num_actions)
        is_first = torch.zeros(batch, length)
        is_first[:, 0] = 1.0
        is_first[1, 5] = 1.0  # mid-sequence reset in one batch row only

        with torch.no_grad():
            post_p, prior_p = tssm.observe(embed, action, is_first, sample=False)

            state = None
            for t in range(length):
                post_s, prior_s = tssm.obs_step(
                    state, action[:, t], embed[:, t], is_first[:, t], sample=False
                )
                for key in ("stoch", "deter", "logit"):
                    assert torch.allclose(
                        post_p[key][:, t], post_s[key], atol=1e-5
                    ), f"post[{key}] diverged at t={t}"
                # Prior stoch is *sampled* inside obs_step (matches parent RSSM
                # semantics) and never trained on — compare the deterministic prior
                # quantities only (logit is what the KL sees).
                for key in ("deter", "logit"):
                    assert torch.allclose(
                        prior_p[key][:, t], prior_s[key], atol=1e-5
                    ), f"prior[{key}] diverged at t={t}"
                assert torch.allclose(
                    post_p["tssm_cnt"][:, t], post_s["tssm_cnt"], atol=1e-6
                ), f"cnt diverged at t={t}"
                assert torch.allclose(
                    post_p["tssm_cache"][:, t], post_s["tssm_cache"], atol=1e-5
                ), f"cache diverged at t={t}"
                state = post_s

    def test_prior_stats_match_sequential(self):
        # Split from the test above for a clean signal: the prior logits (what the KL
        # trains on) must match between the parallel and sequential formulations.
        torch.manual_seed(1)
        batch, length, num_actions, embed_dim = 2, 6, 3, 10
        tssm = self._make(ctx_len=8, num_actions=num_actions, embed=embed_dim)
        tssm.eval()

        embed = torch.randn(batch, length, embed_dim)
        action = torch.randn(batch, length, num_actions)
        is_first = torch.zeros(batch, length)
        is_first[:, 0] = 1.0

        with torch.no_grad():
            _, prior_p = tssm.observe(embed, action, is_first, sample=False)
            state = None
            for t in range(length):
                post_s, prior_s = tssm.obs_step(
                    state, action[:, t], embed[:, t], is_first[:, t], sample=False
                )
                assert torch.allclose(prior_p["logit"][:, t], prior_s["logit"], atol=1e-5)
                assert torch.allclose(prior_p["deter"][:, t], prior_s["deter"], atol=1e-5)
                state = post_s

    def test_imagine_with_action_shapes_and_no_cache(self):
        # The manual imagination loop must thread (not stack) the cache: outputs carry
        # only stoch/deter/stats, at (B, H, ...) shapes.
        torch.manual_seed(0)
        batch, horizon, num_actions = 3, 5, 3
        tssm = self._make(num_actions=num_actions)
        state = tssm.initial(batch)
        actions = torch.randn(batch, horizon, num_actions)

        prior = tssm.imagine_with_action(actions, state)

        assert "tssm_cache" not in prior and "tssm_cnt" not in prior
        assert prior["deter"].shape == (batch, horizon, 16)
        assert prior["stoch"].shape == (batch, horizon, 4, 4)
        assert prior["logit"].shape == (batch, horizon, 4, 4)

    def test_img_step_cache_chaining(self):
        # cnt increments per step and saturates at ctx_len; cache shape stays fixed.
        torch.manual_seed(0)
        batch, ctx_len, num_actions = 2, 4, 3
        tssm = self._make(ctx_len=ctx_len, num_actions=num_actions)
        state = tssm.initial(batch)
        for step in range(ctx_len + 3):
            state = tssm.img_step(state, torch.randn(batch, num_actions))
            expected_cnt = min(step + 1, ctx_len)
            assert torch.all(state["tssm_cnt"] == expected_cnt)
            assert state["tssm_cache"].shape == (batch, ctx_len, 2 * 2 * 16)

    def test_observe_gradients_reach_attention_stack(self):
        torch.manual_seed(0)
        batch, length, num_actions, embed_dim = 2, 5, 3, 10
        tssm = self._make(num_actions=num_actions, embed=embed_dim)
        embed = torch.randn(batch, length, embed_dim, requires_grad=True)
        action = torch.randn(batch, length, num_actions)
        is_first = torch.zeros(batch, length)
        is_first[:, 0] = 1.0

        post, prior = tssm.observe(embed, action, is_first)
        (post["deter"].sum() + prior["logit"].sum()).backward()

        assert embed.grad is not None and torch.any(embed.grad != 0)
        assert tssm._token_in.weight.grad is not None
        for i, blk in enumerate(tssm._blocks):
            for name, p in blk.named_parameters():
                assert p.grad is not None, f"no grad for blocks[{i}].{name}"

    def test_initial_state_contains_empty_cache(self):
        tssm = self._make(ctx_len=4, num_layers=2)
        state = tssm.initial(3)
        assert state["tssm_cache"].shape == (3, 4, 2 * 2 * 16)
        assert torch.all(state["tssm_cache"] == 0)
        assert torch.all(state["tssm_cnt"] == 0)
        assert state["deter"].shape == (3, 16)
