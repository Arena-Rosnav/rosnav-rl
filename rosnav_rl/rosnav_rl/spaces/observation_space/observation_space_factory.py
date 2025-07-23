from typing import Callable, Dict, Any, Union

from .spaces.base_observation_space import BaseObservationSpace
from .space_categories import SpaceCategory


class SpaceFactory:
    registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: Union[str, SpaceCategory] = SpaceCategory.UNCATEGORIZED,
    ) -> Callable:
        """Register a space with optional category metadata.

        Args:
            name: Unique name for the space
            category: Category for grouping (perception, navigation, etc.)
                     Can be a string or SpaceCategory enum
        """
        # Convert string to SpaceCategory if needed
        if isinstance(category, str):
            category = SpaceCategory.from_string(category)

        def inner_wrapper(wrapped_class) -> Callable:
            assert (
                name not in cls.registry
            ), f"ObservationSpace '{name}' already exists!"
            assert issubclass(
                wrapped_class, BaseObservationSpace
            ), f"Wrapped class {wrapped_class.__name__} is not a subclass of 'BaseObservationSpace'!"

            cls.registry[name] = {
                "class": wrapped_class,
                "category": category.value,  # Store as string for JSON serialization
                "module": wrapped_class.__module__,
            }
            return wrapped_class

        return inner_wrapper

    @classmethod
    def instantiate(cls, name: str, **kwargs) -> BaseObservationSpace:
        """Instantiate a registered space by name."""
        assert name in cls.registry, f"ObservationSpace '{name}' is not registered!"
        space_info = cls.registry[name]
        space_class = space_info["class"]

        return space_class(**kwargs)

    @classmethod
    @classmethod
    def get_spaces_by_category(cls) -> Dict[str, list]:
        """Get all registered spaces organized by category."""
        categories = {}
        for space_name, space_info in cls.registry.items():
            category = space_info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(space_name)
        return categories

    @classmethod
    def get_category(cls, name: str) -> str:
        """Get the category of a registered space."""
        assert name in cls.registry, f"ObservationSpace '{name}' is not registered!"
        return cls.registry[name]["category"]

    @classmethod
    def get_category_enum(cls, name: str) -> SpaceCategory:
        """Get the category of a registered space as SpaceCategory enum."""
        category_str = cls.get_category(name)
        return SpaceCategory.from_string(category_str)
