import numpy as np

from gymnasium import spaces


class ActionSpaceManager:
    """Action Space Manager for Reinforcement Learning Environments in ROS Navigation.

    This class manages the action space for both holonomic and non-holonomic robots.
    It handles discrete and continuous action spaces, and provides methods for
    working with actions in the reinforcement learning environment.

    Attributes:
        _holonomic (bool): Flag indicating if the robot is holonomic (can move in any direction).
        _discrete (bool): Flag indicating if the action space is discrete.
        _actions (list): Available actions or action ranges.
        _space (object): The action space object from gym.spaces.

    Usage:
        For a holonomic robot with continuous actions:
            manager = ActionSpaceManager(
                is_holonomic=True,
                is_discrete=False,
                actions={
                    "linear_range": {
                        "x": [-1.0, 1.0],
                        "y": [-1.0, 1.0]
                    },
                    "angular_range": [-1.0, 1.0]

        For a non-holonomic robot with discrete actions:
            manager = ActionSpaceManager(
                is_holonomic=False,
                is_discrete=True,
                actions=[
                    {"linear": 0.2, "angular": 0.0},
                    {"linear": 0.0, "angular": 0.5},
                    # More discrete actions...
    """

    def __init__(
        self,
        is_holonomic: bool,
        is_discrete: bool,
        actions: list,
        # stacked: bool,
        *args,
        **kwargs,
    ) -> None:
        self._holonomic = is_holonomic
        self._discrete = is_discrete
        self._actions = actions
        # self._stacked = stacked

        self._space = self.get_action_space()

    @property
    def actions(self):
        """
        Get the available actions.

        Returns:
            dict: Dictionary containing the available actions.
        """
        return self._actions

    @property
    def action_space(self):
        """
        Get the action space.

        Returns:
            object: The action space.
        """
        return self._space

    @property
    def shape(self):
        """
        Get the shape of the action space.

        Returns:
            tuple: The shape of the action space.
        """
        return self._space.shape

    def get_action_space(self):
        """
        Get the action space based on the configuration.

        Returns:
            object: The action space object.
        """
        if self._discrete:
            return spaces.Discrete(len(self._actions))

        linear_range = self._actions["linear_range"]
        angular_range = self._actions["angular_range"]

        if not self._holonomic:
            return spaces.Box(
                low=np.array([linear_range[0], angular_range[0]], dtype=np.float32),
                high=np.array([linear_range[1], angular_range[1]], dtype=np.float32),
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

    def decode_action(self, action) -> np.ndarray:
        """
        Decode the action.

        Args:
            action: The action to decode.

        Returns:
            np.ndarray: The decoded action.
        """
        if self._discrete:
            # Ensure action is a scalar int for discrete indexing
            if hasattr(action, '__len__'):
                action = int(action[0])
            else:
                action = int(action)
            return self._extend_action_array(self._translate_disc_action(action))

        return self._extend_action_array(action)

    def _extend_action_array(self, action: np.ndarray) -> np.ndarray:
        """
        Extend the action array.

        Args:
            action (np.ndarray): The action array.

        Returns:
            np.ndarray: The extended action array.
        """
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
        """
        Translate the discrete action.

        Args:
            action (int): The discrete action.

        Returns:
            np.ndarray: The translated action.
        """
        return np.array(
            [self._actions[action]["linear"], self._actions[action]["angular"]]
        )

    @property
    def config(self) -> dict:
        """
        Get the configuration.

        Returns:
            dict: The configuration.
        """
        return {
            "holonomic": self._holonomic,
            "action_space_discrete": self._discrete,
            "actions": self._actions,
            # "stacked": self._stacked,
            "space": self.action_space,
        }
