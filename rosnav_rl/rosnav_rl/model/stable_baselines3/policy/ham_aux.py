"""PPO with an auxiliary loss that trains the crowd predictor on the pedestrian motion in the rollout buffer."""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from ..crowd_labels import build_labels

CROWD_KEY = "CrowdSetSpace"
POSE_KEY = "CrowdRobotPoseSpace"


class PPO_HamAux(PPO):
    def __init__(self, *args, aux_weight: float = 1.0, **kwargs):
        self.aux_weight = aux_weight
        super().__init__(*args, **kwargs)

    def _aux_data(self):
        """Labels for every buffer step, flattened env-major like the rollout buffer samples."""
        rb = self.rollout_buffer
        crowd = rb.observations[CROWD_KEY]
        pose = rb.observations[POSE_KEY]
        target, valid, robot_vel = build_labels(crowd, pose, rb.episode_starts)
        return (
            rb.swap_and_flatten(crowd[..., :5]),
            rb.swap_and_flatten(target),
            rb.swap_and_flatten(valid),
            rb.swap_and_flatten(robot_vel),
        )

    def _aux_loss(self, predictor, crowd, target, valid, robot_vel):
        state, _, _ = predictor(crowd, robot_vel)
        err = ((state - target) ** 2).sum(dim=-1)
        w = valid.float()
        return (err * w).sum() / w.sum().clamp(min=1.0)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        predictor = self.policy.features_extractor.predictor
        aux = self._aux_data()
        crowd_all, target_all, valid_all, vel_all = (th.as_tensor(x, device=self.device) for x in aux)

        entropy_losses, pg_losses, value_losses, clip_fractions, aux_losses = [], [], [], [], []
        continue_training = True
        rb = self.rollout_buffer
        total = rb.buffer_size * rb.n_envs

        if not rb.generator_ready:
            for key, obs in rb.observations.items():
                rb.observations[key] = rb.swap_and_flatten(obs)
            for name in ["actions", "values", "log_probs", "advantages", "returns"]:
                rb.__dict__[name] = rb.swap_and_flatten(rb.__dict__[name])
            rb.generator_ready = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            indices = np.random.permutation(total)
            for start in range(0, total, self.batch_size):
                batch = indices[start : start + self.batch_size]
                rollout_data = rb._get_samples(batch)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                loss_1 = advantages * ratio
                loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(loss_1, loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                b = th.as_tensor(batch, device=self.device)
                aux_loss = self._aux_loss(predictor, crowd_all[b], target_all[b], valid_all[b], vel_all[b])
                aux_losses.append(aux_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss = loss + self.aux_weight * aux_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(rb.values.flatten(), rb.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
