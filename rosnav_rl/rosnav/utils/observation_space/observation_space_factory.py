from typing import Callable

from .spaces.base_observation_space import BaseObservationSpace


class SpaceFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            assert (
                name not in cls.registry
            ), f"ObservationSpace '{name}' already exists!"
            assert issubclass(
                wrapped_class, BaseObservationSpace
            ), f"Wrapped class {wrapped_class.__name__} is not a subclass of 'BaseObservationSpace'!"

            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def instantiate(cls, name: str, **kwargs) -> BaseObservationSpace:
        assert name in cls.registry, f"ObservationSpace '{name}' is not registered!"
        space_class = cls.registry[name]

        return space_class(**kwargs)
