from __future__ import annotations

from typing import TYPE_CHECKING

from .policy.base_policy import StableBaselinesPolicyDescription
from .sb3_model import StableBaselinesModel

if TYPE_CHECKING:
    import rosnav_rl.model.stable_baselines3.policy.agent_factory as agent_factory_module


def import_models() -> agent_factory_module.AgentFactory:
    import rosnav_rl.model.stable_baselines3.policy.agent_factory as agent_factory_module
    import rosnav_rl.model.stable_baselines3.policy.sb3_policy.paper

    return agent_factory_module.AgentFactory
