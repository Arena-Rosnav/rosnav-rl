from .policy.base_policy import StableBaselinesPolicy
from .sb3_model import StableBaselinesModel


def import_models() -> "AgentFactory":
    import rosnav_rl.model.stable_baselines3.policy.agent_factory as agent_factory_module
    import rosnav_rl.model.stable_baselines3.policy.sb3_policy.paper

    return agent_factory_module.AgentFactory
