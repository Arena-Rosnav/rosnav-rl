"""PPO with a rollout-buffer auxiliary objective for one-second pedestrian prediction.

The public policy observation remains AGENT_5. Raw tracked crowd states and robot
poses are passed only in the transition ``info`` dictionary and captured during
rollout collection; they are not part of the policy input or reward calculation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from ..crowd_labels import build_labels, map_targets_to_feature_crowd

# Keys used only inside transition info, never added to the Gym observation.
CROWD_INFO_KEY = "_ham_aux_crowd_set"
POSE_INFO_KEY = "_ham_aux_robot_pose"
MAX_CROWD = 24
CROWD_FEATURES = 6  # x, y, vx, vy, valid mask, stable track ID


class _InfoCaptureCallbackProxy:
    """Transparent proxy that captures transition infos inside SB3's rollout loop."""

    def __init__(self, callback, capture_step):
        self._callback = callback
        self._capture_step = capture_step
        self._locals: dict[str, Any] = {}

    def update_locals(self, locals_: dict[str, Any]) -> None:
        self._locals = locals_
        self._callback.update_locals(locals_)

    def on_step(self) -> bool:
        self._capture_step(self._locals)
        return self._callback.on_step()

    def __getattr__(self, name: str):
        return getattr(self._callback, name)


class PPO_HamAux(PPO):
    """PPO whose policy loss is augmented with a supervised crowd-dynamics loss."""

    def __init__(self, *args, aux_weight: float = 1.0, **kwargs):
        self.aux_weight = float(aux_weight)
        self._ham_crowd_steps: list[np.ndarray] = []
        self._ham_pose_steps: list[np.ndarray] = []
        self._ham_rollout_crowd: np.ndarray | None = None
        self._ham_rollout_pose: np.ndarray | None = None
        # The current raw state is kept across rollout boundaries and aligned
        # with the observation that PPO is about to store in its rollout buffer.
        self._ham_current_crowd: list[np.ndarray] | None = None
        self._ham_current_pose: list[np.ndarray] | None = None
        self._ham_active_env = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validated_crowd(info: dict[str, Any]) -> np.ndarray:
        if CROWD_INFO_KEY not in info:
            raise RuntimeError(
                f"Missing {CROWD_INFO_KEY!r} in env info. Ensure ArenaBaseEnv._observe "
                "adds the raw crowd side-channel; do not add CrowdSetSpace to policy observations."
            )
        crowd = np.asarray(info[CROWD_INFO_KEY], dtype=np.float32)
        if crowd.ndim != 2 or crowd.shape[1] < CROWD_FEATURES:
            raise RuntimeError(f"Expected raw crowd shape (K,6), got {crowd.shape}")
        out = np.zeros((MAX_CROWD, CROWD_FEATURES), dtype=np.float32)
        k = min(MAX_CROWD, crowd.shape[0])
        out[:k] = crowd[:k, :CROWD_FEATURES]
        out[:, 4] = (out[:, 4] > 0.5).astype(np.float32)
        return out

    @staticmethod
    def _validated_pose(info: dict[str, Any]) -> np.ndarray:
        if POSE_INFO_KEY not in info:
            raise RuntimeError(
                f"Missing {POSE_INFO_KEY!r} in env info. Ensure ArenaBaseEnv._observe "
                "adds robot pose for the auxiliary labels."
            )
        pose = np.asarray(info[POSE_INFO_KEY], dtype=np.float32).reshape(-1)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise RuntimeError(f"Expected finite robot pose [x,y,yaw], got {pose}")
        return pose

    @classmethod
    def _validated_state(cls, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return cls._validated_crowd(info), cls._validated_pose(info)

    @staticmethod
    def _vecenv_reset_infos(env) -> list[dict[str, Any]]:
        """Find reset infos carrying HamAux data through nested VecEnv wrappers.

        SB3 VecEnvWrapper instances maintain their own reset_infos, which may
        remain [{}] instead of forwarding the underlying VecEnv's reset infos.
        Traverse the wrapper chain and return the first valid side-channel.
        """
        current = env
        seen: set[int] = set()
        inspected: list[str] = []

        while current is not None and id(current) not in seen:
            seen.add(id(current))
            reset_infos = getattr(current, "reset_infos", None)

            if reset_infos is not None:
                try:
                    infos = list(reset_infos)
                except TypeError:
                    infos = []

                if infos and all(
                    isinstance(info, dict)
                    and CROWD_INFO_KEY in info
                    and POSE_INFO_KEY in info
                    for info in infos
                ):
                    return infos

                if infos:
                    inspected.append(
                        f"{type(current).__name__}: "
                        f"{[sorted(info.keys()) if isinstance(info, dict) else type(info).__name__ for info in infos]}"
                    )

            current = getattr(current, "venv", None)

        raise RuntimeError(
            "HamAux could not find valid reset_infos containing both crowd and pose "
            f"keys through the VecEnv wrapper chain. Inspected: {inspected}"
        )

    def _capture_rollout_step(self, locals_: dict[str, Any]) -> None:
        infos = locals_.get("infos")
        dones = locals_.get("dones")
        if infos is None or dones is None:
            raise RuntimeError("SB3 rollout callback did not expose infos/dones; cannot align HamAux labels.")
        if self._ham_current_crowd is None or self._ham_current_pose is None:
            raise RuntimeError("HamAux current state was not initialized from VecEnv.reset_infos.")
        if len(infos) != len(self._ham_current_crowd):
            raise RuntimeError("HamAux VecEnv info count does not match the current state count.")

        # `RolloutBuffer` stores the observation from *before* env.step(). Append
        # the matching current state, not `infos` (which describes the next state).
        self._ham_crowd_steps.append(np.stack(self._ham_current_crowd, axis=0))
        self._ham_pose_steps.append(np.stack(self._ham_current_pose, axis=0))

        dones_array = np.asarray(dones, dtype=bool).reshape(-1)
        reset_infos = self._vecenv_reset_infos(self._ham_active_env) if dones_array.any() else None
        next_crowd: list[np.ndarray] = []
        next_pose: list[np.ndarray] = []
        for i, info in enumerate(infos):
            # VecEnv auto-resets done environments before returning from step().
            # Their next observation corresponds to reset_infos, not terminal info.
            state_info = reset_infos[i] if dones_array[i] else info
            crowd, pose = self._validated_state(state_info)
            next_crowd.append(crowd)
            next_pose.append(pose)
        self._ham_current_crowd, self._ham_current_pose = next_crowd, next_pose

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        self._ham_crowd_steps = []
        self._ham_pose_steps = []
        self._ham_rollout_crowd = None
        self._ham_rollout_pose = None
        self._ham_active_env = env
        if self._ham_current_crowd is None or self._ham_current_pose is None:
            reset_infos = self._vecenv_reset_infos(env)
            states = [self._validated_state(info) for info in reset_infos]
            self._ham_current_crowd = [state[0] for state in states]
            self._ham_current_pose = [state[1] for state in states]
        capture_callback = _InfoCaptureCallbackProxy(callback, self._capture_rollout_step)
        completed = super().collect_rollouts(env, capture_callback, rollout_buffer, n_rollout_steps)
        if completed:
            expected = rollout_buffer.buffer_size
            if len(self._ham_crowd_steps) != expected or len(self._ham_pose_steps) != expected:
                raise RuntimeError(
                    "HamAux side-channel length does not match PPO rollout buffer: "
                    f"crowd={len(self._ham_crowd_steps)}, pose={len(self._ham_pose_steps)}, expected={expected}."
                )
            # T,N,K,6 and T,N,3 align with the pre-action observation at every rollout index.
            self._ham_rollout_crowd = np.stack(self._ham_crowd_steps, axis=0)
            self._ham_rollout_pose = np.stack(self._ham_pose_steps, axis=0)
        return completed

    def _aux_data(self):
        """Build targets aligned to AGENT_5 feature-map inputs and recorded last actions."""
        rb = self.rollout_buffer
        if self._ham_rollout_crowd is None or self._ham_rollout_pose is None:
            raise RuntimeError("HamAux labels are absent; collect a complete rollout before calling train().")
        steps, envs = rb.episode_starts.shape
        if self._ham_rollout_crowd.shape[:2] != (steps, envs):
            raise RuntimeError(
                f"HamAux data {self._ham_rollout_crowd.shape[:2]} does not match rollout buffer "
                f"{rb.episode_starts.shape}."
            )

        # Raw tracks provide stable IDs and targets, but MUST NOT be used as predictor
        # inputs: inference sees only AGENT_5's maps. Reconstruct the training input
        # from the stored maps using the exact same extractor code as inference.
        extractor = self.policy.features_extractor
        map_keys = ("PedestrianTypeSpace", "PedestrianVelXSpace", "PedestrianVelYSpace")
        map_obs = {}
        map_height = map_width = None
        for key in map_keys:
            arr = rb.observations.get(key)
            if arr is None:
                raise RuntimeError(f"AGENT_5 observation {key!r} is missing from the rollout buffer.")
            arr = np.asarray(arr)
            if arr.ndim != 4 or arr.shape[:2] != (steps, envs):
                raise RuntimeError(f"Expected rollout map {key!r} with shape (T,N,H,W), got {arr.shape}.")
            map_height, map_width = arr.shape[-2:]
            map_obs[key] = th.as_tensor(arr.reshape(steps * envs, map_height, map_width), device=self.device)

        map_crowd = extractor.crowd_from_feature_maps(map_obs).detach().cpu().numpy()
        map_crowd = map_crowd.reshape(steps, envs, map_crowd.shape[-2], 5)
        raw_target, raw_valid, _ = build_labels(
            self._ham_rollout_crowd, self._ham_rollout_pose, rb.episode_starts
        )
        target, valid = map_targets_to_feature_crowd(
            map_crowd, self._ham_rollout_crowd, raw_target, raw_valid,
            roi_in_m=extractor._map_roi_m, map_height=map_height, map_width=map_width,
        )

        debug_count = getattr(self, "_ham_aux_debug_count", 0)
        if debug_count < 5:
            n_raw = int((self._ham_rollout_crowd[..., 4] > 0.5).sum())
            n_raw_valid = int(raw_valid.sum())
            n_feature = int((map_crowd[..., 4] > 0.5).sum())
            n_mapped = int(valid.sum())
            print(
                f"[HamAux diagnostic] raw_live={n_raw}, "
                f"raw_valid={n_raw_valid}, feature_live={n_feature}, "
                f"mapped_valid={n_mapped}, target_shape={target.shape}",
                flush=True,
            )
            self._ham_aux_debug_count = debug_count + 1

        # Condition on the same physical [vx, vy, yaw_rate] command used at inference.
        # Do not leak future robot motion into training.
        action_arr = np.asarray(rb.observations.get("LastActionSpace"))
        if action_arr.ndim != 3 or action_arr.shape[:2] != (steps, envs):
            raise RuntimeError(f"Expected LastActionSpace rollout shape (T,N,A), got {action_arr.shape}.")
        action_tensor = th.as_tensor(action_arr.reshape(steps * envs, -1), device=self.device)
        robot_command = extractor._robot_velocity(action_tensor).detach().cpu().numpy()
        if robot_command.shape != (steps * envs, 3):
            raise RuntimeError(f"Expected physical robot commands with shape (T*N,3), got {robot_command.shape}.")
        robot_command = robot_command.reshape(steps, envs, 3)

        return (
            rb.swap_and_flatten(map_crowd),
            rb.swap_and_flatten(target),
            rb.swap_and_flatten(valid),
            rb.swap_and_flatten(robot_command),
        )

    @staticmethod
    def _aux_loss(predictor, crowd, target, valid, robot_command):
        state, _, _ = predictor(crowd, robot_command)
        error = ((state - target) ** 2).sum(dim=-1)
        weights = valid.to(dtype=error.dtype)
        return (error * weights).sum() / weights.sum().clamp(min=1.0)

    def _excluded_save_params(self) -> list[str]:
        """Exclude transient rollout state and active VecEnv from model archives."""
        excluded = super()._excluded_save_params()
        excluded.extend([
            "_ham_active_env",
            "_ham_crowd_steps",
            "_ham_pose_steps",
            "_ham_rollout_crowd",
            "_ham_rollout_pose",
            "_ham_current_crowd",
            "_ham_current_pose",
        ])
        return excluded

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
        approx_kl_divs, last_loss = [], None
        continue_training = True
        rb = self.rollout_buffer
        total = rb.buffer_size * rb.n_envs

        if not rb.generator_ready:
            for key, obs in rb.observations.items():
                rb.observations[key] = rb.swap_and_flatten(obs)
            for name in ["actions", "values", "log_probs", "advantages", "returns"]:
                rb.__dict__[name] = rb.swap_and_flatten(rb.__dict__[name])
            rb.generator_ready = True

        for _epoch in range(self.n_epochs):
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
                policy_loss = -th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                pg_losses.append(policy_loss.detach().item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.detach().item())

                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.detach().item())

                batch_indices = th.as_tensor(batch, device=self.device)
                aux_loss = self._aux_loss(
                    predictor,
                    crowd_all[batch_indices],
                    target_all[batch_indices],
                    valid_all[batch_indices],
                    vel_all[batch_indices],
                )
                aux_losses.append(aux_loss.detach().item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.aux_weight * aux_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                last_loss = loss.detach().item()

            self._n_updates += 1
            if not continue_training:
                break

        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/aux_loss", float(np.mean(aux_losses)))
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/loss", float(last_loss if last_loss is not None else 0.0))
        self.logger.record("train/explained_variance", explained_variance(rb.values.flatten(), rb.returns.flatten()))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
