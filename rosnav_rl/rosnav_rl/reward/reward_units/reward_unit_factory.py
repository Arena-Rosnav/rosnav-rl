from typing import Callable, Dict, Type, TypeVar

from .base_reward_units import RewardUnit

# Define a type variable for the RewardUnit class
T = TypeVar("T", bound=RewardUnit)


class RewardUnitFactory:
    """
    Factory class for creating reward units.

    This class manages the registration and instantiation of reward units.
    It provides a way to register reward unit classes by name and instantiate
    them later using that name.

    Attributes:
        registry (Dict[str, Type[RewardUnit]]): Dictionary mapping unit names to their classes.
    """

    registry: Dict[str, Type[RewardUnit]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Register a reward unit class with a given name.

        Args:
            name: The name to register the reward unit under.

        Returns:
            A decorator function that registers the class.

        Example:
            @RewardUnitFactory.register("goal_reached")
            class RewardGoalReached(RewardUnit):
                ...
        """

        def inner_wrapper(wrapped_class: Type[T]) -> Type[T]:
            if name in cls.registry:
                raise ValueError(f"RewardUnit '{name}' already exists!")

            if not issubclass(wrapped_class, RewardUnit):
                raise TypeError(
                    f"Class {wrapped_class.__name__} must inherit from RewardUnit"
                )

            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def instantiate(cls, name: str) -> Type[RewardUnit]:
        """
        Get the reward unit class registered under the given name.

        Args:
            name: The name of the reward unit to instantiate.

        Returns:
            The reward unit class.

        Raises:
            KeyError: If no reward unit is registered under the given name.
        """
        if name not in cls.registry:
            raise KeyError(f"RewardUnit '{name}' is not registered!")

        return cls.registry[name]
