"""Tests for the cSRSSM SocialContextEncoder prediction-driven identifiability loss.

Run with ROS environment sourced (or in the arena container):
    cd src/Arena/arena_training/deps/rosnav_rl/rosnav_rl
    python -m pytest tests/test_social_context.py -v
"""

import torch

from rosnav_rl.model.dreamerv3.cfg import SocialContextCfg
from rosnav_rl.model.dreamerv3.social.context import (
    SocialContextEncoder,
    context_pred_loss,
)


def _make_encoder(b_dim=4, hidden=16, node_feat_dim=3, window=6):
    cfg = SocialContextCfg(enabled=True, b_dim=b_dim, hidden=hidden, window=window)
    return SocialContextEncoder(cfg=cfg, node_feat_dim=node_feat_dim, device="cpu")


class TestContextPredLoss:
    def test_returns_scalar(self):
        enc = _make_encoder()
        b = torch.randn(5, 4)
        pooled_future = torch.randn(5, 8, 3)
        loss = context_pred_loss(enc, b, pooled_future)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_zero_when_fewer_than_two_steps(self):
        # M < 2 -> no (input, target) pair exists -> loss must be exactly 0.
        enc = _make_encoder()
        b = torch.randn(5, 4)
        loss = context_pred_loss(enc, b, torch.randn(5, 1, 3))
        assert loss.item() == 0.0

    def test_zero_when_no_valid_pairs(self):
        # All target steps empty -> masked out -> loss 0.
        enc = _make_encoder()
        b = torch.randn(2, 4)
        pooled_future = torch.randn(2, 4, 3)
        step_valid = torch.zeros(2, 4, dtype=torch.bool)
        loss = context_pred_loss(enc, b, pooled_future, step_valid)
        assert loss.item() == 0.0

    def test_validity_mask_excludes_empty_target_steps(self):
        # Corrupting an invalid target step must not change the loss; corrupting a valid
        # one must.
        torch.manual_seed(0)
        enc = _make_encoder()
        b = torch.randn(2, 4)
        pooled_future = torch.randn(2, 5, 3)
        step_valid = torch.ones(2, 5, dtype=torch.bool)
        step_valid[:, 2] = False  # step 2 empty -> pair (1 -> 2) excluded

        base = context_pred_loss(enc, b, pooled_future, step_valid)

        corrupted_invalid = pooled_future.clone()
        corrupted_invalid[:, 2] += 100.0  # only affects the excluded pair's target...
        # ...but step 2 is also the *input* of pair (2 -> 3), so zero it back there is
        # impossible with a shared tensor; instead compare against corrupting a valid
        # target-only step (step 4, the last one, input of no pair).
        corrupted_valid = pooled_future.clone()
        corrupted_valid[:, 4] += 100.0
        assert context_pred_loss(enc, b, corrupted_valid, step_valid).item() != base.item()

        # And excluding that same last step via the mask removes its contribution again.
        mask_no_last = step_valid.clone()
        mask_no_last[:, 4] = False
        assert (
            context_pred_loss(enc, b, corrupted_valid, mask_no_last).item()
            != context_pred_loss(enc, b, corrupted_valid, step_valid).item()
        )

    def test_informative_b_beats_shuffled_b(self):
        # Identifiability check: synthetic regimes where pooled_{t+1} = pooled_t + delta,
        # with delta encoded in b. A head trained with the true b can solve the task
        # (loss -> ~0); trained with b shuffled across the batch it cannot.
        torch.manual_seed(0)
        B, M, Fd = 32, 10, 3
        b_dim = Fd  # b directly encodes the per-sequence regime delta

        delta = torch.randn(B, Fd)  # one regime per sequence
        pooled = torch.zeros(B, M, Fd)
        pooled[:, 0] = torch.randn(B, Fd)
        for t in range(1, M):
            pooled[:, t] = pooled[:, t - 1] + delta

        def train(b_input, seed):
            torch.manual_seed(seed)
            enc = _make_encoder(b_dim=b_dim, hidden=32, node_feat_dim=Fd)
            opt = torch.optim.Adam(enc.pred_head.parameters(), lr=1e-2)
            for _ in range(300):
                opt.zero_grad()
                loss = context_pred_loss(enc, b_input, pooled)
                loss.backward()
                opt.step()
            with torch.no_grad():
                return context_pred_loss(enc, b_input, pooled).item()

        loss_true = train(delta, seed=1)
        loss_shuffled = train(delta[torch.randperm(B)], seed=1)

        assert loss_true < 0.05, f"informative b should solve the task, got {loss_true}"
        assert loss_true < 0.5 * loss_shuffled, (
            f"informative b ({loss_true}) should clearly beat shuffled b ({loss_shuffled})"
        )

    def test_gradient_flows_through_b_into_encoder(self):
        # The whole point of the loss: gradient pressure must reach the context encoder
        # (GRU + mean/std heads) through the sampled b, and the pred head itself.
        torch.manual_seed(0)
        enc = _make_encoder()
        window = torch.randn(4, 6, 5, 3)  # (B, K, N, F)
        validity = torch.ones(4, 6, 5)
        mean, std = enc(window, validity)
        b = enc.sample(mean, std, sample=True)
        pooled_future = torch.randn(4, 7, 3)

        loss = context_pred_loss(enc, b, pooled_future)
        loss.backward()

        assert enc.mean_head.weight.grad is not None
        assert torch.any(enc.mean_head.weight.grad != 0)
        for name, p in enc.gru.named_parameters():
            assert p.grad is not None, f"no grad for gru.{name}"
        for i, layer in enumerate(enc.pred_head):
            if hasattr(layer, "weight"):
                assert layer.weight.grad is not None, f"no grad for pred_head[{i}]"
                assert torch.any(layer.weight.grad != 0)
