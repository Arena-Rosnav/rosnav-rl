from typing import Callable, Type, Union

from rosnav_rl.utils.sb3agent_format_check import check_format
from stable_baselines3.common.policies import BasePolicy

from .base_policy import StableBaselinesPolicy


class AgentFactory:
    """The factory class for creating agents"""

    registry = {}
    """ Internal registry for available agents """

    @classmethod
    def register(cls, name: str) -> Callable:
        """Class method to register agent class to the internal registry.

        Args:
            name (str): The name of the agent.

        Returns:
            The agent class itself.
        """

        def inner_wrapper(wrapped_class) -> Callable:
            assert name not in cls.registry, f"Agent '{name}' already exists!"
            assert issubclass(wrapped_class, StableBaselinesPolicy) or issubclass(
                wrapped_class, BasePolicy
            ), f"Wrapped class {wrapped_class.__name__} is neither of type 'StableBaselinesPolicy' nor 'BasePolicy!'"

            if issubclass(wrapped_class, StableBaselinesPolicy):
                check_format(wrapped_class)

            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    # end register()

    @classmethod
    def instantiate(
        cls, name: str, **kwargs
    ) -> Union[Type[StableBaselinesPolicy], Type[BasePolicy]]:
        """Factory command to create the agent.
        This method gets the appropriate agent class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        Args:
            name (str): The name of the agent to create.agent_class

        Returns:
            An instance of the agent that is created.
        """
        assert name in cls.registry, f"Agent '{name}' is not registered!"
        agent_class = cls.registry[name]

        if issubclass(agent_class, StableBaselinesPolicy):
            return agent_class(**kwargs)
        else:
            return agent_class
