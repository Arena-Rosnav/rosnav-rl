from rl_utils.utils.observation_collector import ObservationDict
from rosnav_rl.reward.reward_function import RewardFunction

"""
- reward function init
- refactor models, add normalization, add stacking

"""


class RL_Agent:
    model = None
    reward_function = None
    space_manager = None

    @property
    def observation_space(self):
        pass

    @property
    def action_space(self):
        pass

    def get_reward(self, observation: ObservationDict) -> float:
        pass

    def get_action(self, observation: ObservationDict) -> int:
        pass
