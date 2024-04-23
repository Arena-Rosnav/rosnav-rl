from gymnasium import spaces


class SwitchActionSpace:

    def __init__(self, num_planners: int, *args, **kwargs) -> None:
        self._num_planners = num_planners
        self._space = spaces.Discrete(self._num_planners)

    @property
    def action_space(self):
        """
        Get the action space.

        Returns:
            object: The action space.
        """
        return self._space

    def decode_action(self, action):
        """
        Decode the action.

        Args:
            action (int): The action to decode.

        Returns:
            int: The decoded action.
        """
        return action
