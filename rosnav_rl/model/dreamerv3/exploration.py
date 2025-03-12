from typing import TYPE_CHECKING

import torch
from torch import distributions as torchd
from torch import nn

from ..dreamerv3 import models, networks, tools

if TYPE_CHECKING:
    from ..dreamerv3 import cfg

class Random(nn.Module):
    def __init__(self, config: "cfg.DreamerV3Cfg", act_space):
        super(Random, self).__init__()
        self._config = config
        self._act_space = act_space

    def actor(self, feat):
        if self._config.model.actor.dist == "onehot":
            return tools.OneHotDist(
                torch.zeros(
                    (
                        self._act_space.n
                        if hasattr(self._act_space, "n")
                        else self._act_space.shape[0]
                    ),
                    device=self._config.general.device,
                ).repeat(self._config.environment.envs, 1)
            )
        else:
            return torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(
                        self._act_space.low, device=self._config.general.device
                    ).repeat(self._config.envs, 1),
                    torch.tensor(
                        self._act_space.high, device=self._config.general.device
                    ).repeat(self._config.envs, 1),
                ),
                1,
            )

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(nn.Module):
    def __init__(self, config: "cfg.DreamerV3Cfg", world_model, reward, act_space):
        super(Plan2Explore, self).__init__()
        self._config = config
        self._use_amp = True if config.general.precision == 16 else False
        self._reward = reward
        self._behavior = models.ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        if config.model.dyn_discrete:
            feat_size = (
                config.model.dyn_stoch * config.model.dyn_discrete
                + config.model.dyn_deter
            )
            stoch = config.model.dyn_stoch * config.model.dyn_discrete
        else:
            feat_size = config.model.dyn_stoch + config.model.dyn_deter
            stoch = config.model.dyn_stoch
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.model.dyn_deter,
            "feat": config.model.dyn_stoch + config.model.dyn_deter,
        }[self._config.model.exploration.disag_target]
        kw = dict(
            inp_dim=feat_size
            + (
                act_space.n
                if hasattr(act_space, "n")
                else (
                    act_space.shape[0]
                    if config.model.exploration.disag_action_cond
                    else 0
                )
            ),  # pytorch version
            shape=size,
            layers=config.model.exploration.disag_layers,
            units=config.model.exploration.disag_units,
            act=config.model.act,
        )
        self._networks = nn.ModuleList(
            [networks.MLP(**kw) for _ in range(config.model.exploration.disag_models)]
        )
        kw = dict(
            wd=config.model.weight_decay, opt=config.training.opt, use_amp=self._use_amp
        )
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.training.model_lr,
            config.training.opt_eps,
            config.training.grad_clip,
            **kw
        )

    def train(self, start, context, data):
        with tools.RequiresGrad(self._networks):
            metrics = {}
            stoch = start["stoch"]
            if self._config.model.dyn_discrete:
                stoch = torch.reshape(
                    stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                )
            target = {
                "embed": context["embed"],
                "stoch": stoch,
                "deter": start["deter"],
                "feat": context["feat"],
            }[self._config.model.exploration.disag_target]
            inputs = context["feat"]
            if self._config.model.exploration.disag_action_cond:
                inputs = torch.concat(
                    [
                        inputs,
                        torch.tensor(
                            data["action"], device=self._config.general.device
                        ),
                    ],
                    -1,
                )
            metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self._behavior._train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        inputs = feat
        if self._config.model.exploration.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)
        preds = torch.cat(
            [head(inputs, torch.float32).mode()[None] for head in self._networks], 0
        )
        disag = torch.mean(torch.std(preds, 0), -1)[..., None]
        if self._config.model.exploration.disag_log:
            disag = torch.log(disag)
        reward = self._config.model.exploration.intr_scale * disag
        if self._config.model.exploration.extr_scale:
            reward += self._config.model.exploration.extr_scale * self._reward(
                feat, state, action
            )
        return reward

    def _train_ensemble(self, inputs, targets):
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            if self._config.model.exploration.disag_offset:
                targets = targets[:, self._config.model.exploration.disag_offset :]
                inputs = inputs[:, : -self._config.model.exploration.disag_offset]
            targets = targets.detach()
            inputs = inputs.detach()
            preds = [head(inputs) for head in self._networks]
            likes = torch.cat(
                [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
            )
            loss = -torch.mean(likes)
        metrics = self._expl_opt(loss, self._networks.parameters())
        return metrics
