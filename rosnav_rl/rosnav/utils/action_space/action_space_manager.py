import numpy as np

from gymnasium import spaces


class ActionSpaceManager:
    def __init__(
        self,
        holonomic: bool,
        action_space_discrete: bool,
        actions: dict,
        stacked: bool,
        *args,
        **kwargs
    ) -> None:
        self._holonomic = holonomic
        self._discrete = action_space_discrete
        self._actions = actions
        self._stacked = stacked

        self._space = self.get_action_space()

    @property
    def actions(self):
        return self._actions

    @property
    def action_space(self):
        return self._space

    @property
    def shape(self):
        return self._space.shape

    def get_action_space(self):
        if self._discrete:
            # self._discrete_actions is a list, each element is a dict with the keys ["name", 'linear','angular']
            return spaces.Discrete(len(self._actions))

        linear_range = self._actions["linear_range"]
        angular_range = self._actions["angular_range"]

        if not self._holonomic:
            return spaces.Box(
                low=np.array([linear_range[0], angular_range[0]]),
                high=np.array([linear_range[1], angular_range[1]]),
                dtype=np.float32,
            )

        linear_range_x, linear_range_y = (
            linear_range["x"],
            linear_range["y"],
        )

        return spaces.Box(
            low=np.array(
                [
                    linear_range_x[0],
                    linear_range_y[0],
                    angular_range[0],
                ]
            ),
            high=np.array(
                [
                    linear_range_x[1],
                    linear_range_y[1],
                    angular_range[1],
                ]
            ),
            dtype=np.float32,
        )

    def decode_action(self, action):
        if self._stacked:
            action = action[0] if action.ndim == 2 else action

        if self._discrete:
            return self._extend_action_array(self._translate_disc_action(action))

        return self._extend_action_array(action)

    def _extend_action_array(self, action: np.ndarray) -> np.ndarray:
        if self._holonomic:
            assert (
                self._holonomic and len(action) == 3
            ), "Robot is holonomic but action with only two freedoms of movement provided"

            return action
        else:
            assert (
                not self._holonomic and len(action) == 2
            ), "Robot is non-holonomic but action with more than two freedoms of movement provided"
            return np.array([action[0], 0, action[1]])

    def _translate_disc_action(self, action: int):
        return np.array(
            [self._actions[action]["linear"], self._actions[action]["angular"]]
        )
