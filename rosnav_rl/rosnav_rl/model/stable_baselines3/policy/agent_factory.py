from collections.abc import Callable
from typing import Any, TypeVar

from stable_baselines3.common.policies import BasePolicy

from .base_policy import StableBaselinesPolicyDescription

_AgentClassT = TypeVar(
    "_AgentClassT",
    bound=type[StableBaselinesPolicyDescription] | type[BasePolicy],
)


class AgentFactory:
    """The factory class for creating agents"""

    registry: dict[str, type[StableBaselinesPolicyDescription] | type[BasePolicy]] = {}
    """ Internal registry for available agents """

    @classmethod
    def register(cls, name: str) -> Callable[[_AgentClassT], _AgentClassT]:
        """Class method to register agent class to the internal registry.

        Args:
            name (str): The name of the agent.

        Returns:
            The agent class itself.
        """

        def inner_wrapper(wrapped_class: _AgentClassT) -> _AgentClassT:
            assert name not in cls.registry, f"Agent '{name}' already exists!"
            assert issubclass(wrapped_class, StableBaselinesPolicyDescription) or issubclass(
                wrapped_class, BasePolicy
            ), (
                f"Wrapped class {wrapped_class.__name__} is neither of type 'StableBaselinesPolicyDescription' nor 'BasePolicy!'"
            )
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    # end register()

    @classmethod
    def instantiate(cls, name: str, **kwargs: Any) -> StableBaselinesPolicyDescription | type[BasePolicy]:
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

        if issubclass(agent_class, StableBaselinesPolicyDescription):
            return agent_class(**kwargs)
        else:
            return agent_class


def _auto_load(cls: type[AgentFactory]) -> None:
    """Automatically import and register all agent classes in the sb3_policy directory."""
    import importlib
    import inspect
    import pkgutil

    from . import sb3_policy

    # Find all modules in the sb3_policy package
    package = sb3_policy
    prefix = package.__name__ + "."
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        if ispkg:
            continue

        module = importlib.import_module(modname)

        # Register all subclasses of StableBaselinesPolicyDescription or BasePolicy
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if issubclass(obj, StableBaselinesPolicyDescription) or issubclass(obj, BasePolicy):
                # Use class name as registry key if not already registered
                if name not in cls.registry:
                    cls.registry[name] = obj


_auto_load(AgentFactory)
