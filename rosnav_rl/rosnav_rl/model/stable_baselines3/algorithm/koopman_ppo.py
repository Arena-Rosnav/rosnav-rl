import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from ..policy.feature_extractors.koopman_head import KoopmanLinearHead, MLPControlHead


class KoopmanPPO(PPO):
    """PPO variant integrating a linear-latent forward auxiliary loss."""

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, int | float | list[int]] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
        aux_loss_weight: float = 0.1,
    ) -> None:
        self.aux_loss_weight: float = aux_loss_weight
        self.pred_head: nn.Module | None = None
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _init_head(self, latent_dim: int, action_dim: int) -> nn.Module:
        return KoopmanLinearHead(latent_dim, action_dim)

    def _setup_model(self) -> None:
        super()._setup_model()
        latent_dim: int = self.policy.features_extractor.features_dim
        action_dim: int = int(self.action_space.shape[0])

        self.pred_head = self._init_head(latent_dim, action_dim).to(self.device)
        self.policy.optimizer.add_param_group({"params": self.pred_head.parameters()})

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range: float = self.clip_range(self._current_progress_remaining)
        clip_range_vf: float | None = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        aux_losses: list[float] = []
        assert self.pred_head is not None

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions: torch.Tensor = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values: torch.Tensor
                log_prob: torch.Tensor
                entropy: torch.Tensor | None
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages: torch.Tensor = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio: torch.Tensor = torch.exp(log_prob - rollout_data.old_log_prob)
                surr1: torch.Tensor = advantages * ratio
                surr2: torch.Tensor = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

                if clip_range_vf is None:
                    values_pred: torch.Tensor = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss: torch.Tensor = F.mse_loss(rollout_data.returns, values_pred)

                entropy_loss: torch.Tensor = -torch.mean(-log_prob if entropy is None else entropy)

                # Koopman Auxiliary Loss with stopgrad and episode termination mask
                z_t: torch.Tensor = self.policy.features_extractor(rollout_data.observations)
                z_hat_next: torch.Tensor = self.pred_head(z_t, rollout_data.actions)

                with torch.no_grad():
                    z_next: torch.Tensor = self.policy.features_extractor(rollout_data.next_observations)

                mask: torch.Tensor = (1.0 - rollout_data.dones.float()).unsqueeze(-1)
                squared_err: torch.Tensor = F.mse_loss(z_hat_next, z_next, reduction="none")
                aux_loss: torch.Tensor = (squared_err * mask).sum() / (mask.sum() + 1e-8)

                loss: torch.Tensor = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.aux_loss_weight * aux_loss
                )

                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                aux_losses.append(aux_loss.item())

        self._n_updates += self.n_epochs
        self.logger.record("train/aux_loss", float(torch.tensor(aux_losses).mean()))


class MLPPPO(KoopmanPPO):
    """PPO variant using a non-linear MLP next-latent predictor for ablation."""

    def _init_head(self, latent_dim: int, action_dim: int) -> nn.Module:
        return MLPControlHead(latent_dim, action_dim)
