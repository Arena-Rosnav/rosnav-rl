from typing import ClassVar

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RecurrentDictRolloutBufferSamples
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.vec_env import VecNormalize


class PackedObservations(dict[str, th.Tensor]):
    """The real rows of a padded recurrent minibatch, with the mask that places them in it."""

    def __init__(self, rows: dict[str, th.Tensor], mask: th.Tensor):
        super().__init__(rows)
        self.mask = mask


class PackedRecurrentDictRolloutBuffer(RecurrentDictRolloutBuffer):
    """Recurrent dict rollout buffer whose minibatch observations leave out the sequence padding."""

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: VecNormalize | None = None,
    ) -> RecurrentDictRolloutBufferSamples:
        samples = super()._get_samples(batch_inds, env_change, env)
        mask = samples.mask > 1e-8
        rows = {key: obs[mask] for key, obs in samples.observations.items()}
        return samples._replace(observations=PackedObservations(rows, mask))


def _unpack(features: th.Tensor, mask: th.Tensor) -> th.Tensor:
    padded = features.new_zeros((mask.shape[0], *features.shape[1:]))
    padded[mask] = features
    return padded


class PackedRecurrentActorCriticPolicy(RecurrentMultiInputActorCriticPolicy):
    """Recurrent multi-input actor-critic policy that extracts features of packed minibatches on their real rows."""

    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor | None = None
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        features = super().extract_features(obs, features_extractor)
        if not isinstance(obs, PackedObservations):
            return features
        if isinstance(features, tuple):
            return _unpack(features[0], obs.mask), _unpack(features[1], obs.mask)
        return _unpack(features, obs.mask)


class PackedRecurrentPPO(RecurrentPPO):
    """RecurrentPPO whose updates run the features extractor on the real minibatch rows, never on padding."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        **RecurrentPPO.policy_aliases,
        "MultiInputLstmPolicy": PackedRecurrentActorCriticPolicy,
    }

    def _setup_model(self) -> None:
        super()._setup_model()
        buffer = self.rollout_buffer
        if isinstance(buffer, RecurrentDictRolloutBuffer) and isinstance(self.policy, PackedRecurrentActorCriticPolicy):
            self.rollout_buffer = PackedRecurrentDictRolloutBuffer(
                buffer.buffer_size,
                buffer.observation_space,
                buffer.action_space,
                buffer.hidden_state_shape,
                buffer.device,
                gamma=buffer.gamma,
                gae_lambda=buffer.gae_lambda,
                n_envs=buffer.n_envs,
            )
