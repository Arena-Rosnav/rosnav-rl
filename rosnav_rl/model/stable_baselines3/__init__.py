from .policy.base_policy import StableBaselinesPolicy
from .sb3_agent import StableBaselinesAgent


def import_models() -> "AgentFactory":
    import rosnav_rl.model.stable_baselines3.policy.agent_factory as agent_factory_module
    import rosnav_rl.model.stable_baselines3.policy.sb3_policy.paper
    import rosnav_rl.model.stable_baselines3.policy.sb3_policy.custom_sb3_policy

    return agent_factory_module.AgentFactory
