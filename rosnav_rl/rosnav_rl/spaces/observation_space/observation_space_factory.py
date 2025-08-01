from typing import TYPE_CHECKING, Any, Callable, Dict, Union

from .space_categories import SpaceCategory

if TYPE_CHECKING:
    from .spaces.base_observation_space import BaseObservationSpace


class SpaceFactory:
    registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str = None,
        category: Union[str, SpaceCategory] = SpaceCategory.UNCATEGORIZED,
        auto_name: bool = False,
        aliases: list = None,
    ) -> Callable:
        """Register a space with flexible naming options.

        Args:
            name: Explicit name for the space. If None and auto_name=True, derives from class name.
            category: Category for grouping (perception, navigation, etc.)
                     Can be a string or SpaceCategory enum.
            auto_name: If True and name is None, automatically derive names from class name.
                - Registers PascalCase class name (with 'Space' suffix) as the primary name.
                - Also registers snake_case version (without 'Space' suffix) as an alias.
            aliases: List of alternative names to register this space under.

        Examples:
            @SpaceFactory.register("motion_state")  # Register with explicit name
            @SpaceFactory.register(auto_name=True)  # Registers both 'MotionStateSpace' and 'motion_state' as alias
            @SpaceFactory.register("motion", aliases=["motion_state", "MotionStateSpace"])
        """
        # Convert string to SpaceCategory if needed
        if isinstance(category, str):
            category = SpaceCategory.from_string(category)

        def inner_wrapper(wrapped_class) -> Callable:
            # Determine the primary registration name and aliases
            primary_name = name
            auto_aliases = []
            if primary_name is None and auto_name:
                # Auto-derive from class name
                class_name = wrapped_class.__name__
                # Always use PascalCase class name (with 'Space' suffix) as primary name
                pascal_name = class_name
                # Convert PascalCase to snake_case (without 'Space' suffix)
                import re

                if pascal_name.endswith("Space"):
                    snake_base = pascal_name[:-5]
                else:
                    snake_base = pascal_name
                snake_name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", snake_base)
                snake_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_name).lower()
                primary_name = pascal_name
                auto_aliases.append(snake_name)

            if primary_name is None:
                raise ValueError(
                    f"Must provide either 'name' or set 'auto_name=True' for {wrapped_class.__name__}"
                )

            import rosnav_rl.spaces.observation_space.spaces.base_observation_space as base_space

            assert issubclass(
                wrapped_class, base_space.BaseObservationSpace
            ), f"Wrapped class {wrapped_class.__name__} is not a subclass of 'BaseObservationSpace'!"

            # Register primary name
            assert (
                primary_name not in cls.registry
            ), f"ObservationSpace '{primary_name}' already exists!"

            space_info = {
                "class": wrapped_class,
                "category": category.value,
                "module": wrapped_class.__module__,
                "class_name": wrapped_class.__name__,  # Store for debugging/introspection
            }

            cls.registry[primary_name] = space_info

            # Register auto-generated alias (snake_case) if using auto_name
            for alias in auto_aliases:
                if alias not in cls.registry:
                    cls.registry[alias] = space_info

            # Register aliases if provided
            if aliases:
                for alias in aliases:
                    assert (
                        alias not in cls.registry
                    ), f"ObservationSpace alias '{alias}' already exists!"
                    cls.registry[alias] = space_info

            return wrapped_class

        return inner_wrapper

    @classmethod
    def instantiate(cls, name: str, **kwargs) -> "BaseObservationSpace":
        """Instantiate a registered space by name."""
        assert name in cls.registry, f"ObservationSpace '{name}' is not registered!"
        space_info = cls.registry[name]
        space_class = space_info["class"]

        return space_class(**kwargs)

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

    @classmethod
    def get_class_name(cls, name: str) -> str:
        """Get the actual class name of a registered space."""
        assert name in cls.registry, f"ObservationSpace '{name}' is not registered!"
        return cls.registry[name]["class_name"]

    @classmethod
    def list_aliases(cls, primary_name: str) -> list:
        """List all aliases for a given space (including the primary name)."""
        if primary_name not in cls.registry:
            return []

        target_class = cls.registry[primary_name]["class"]
        aliases = []

        for reg_name, space_info in cls.registry.items():
            if space_info["class"] == target_class:
                aliases.append(reg_name)

        return aliases

    @classmethod
    def find_space_by_class_name(cls, class_name: str) -> list:
        """Find all registration names for a given class name."""
        matches = []
        for reg_name, space_info in cls.registry.items():
            if space_info["class_name"] == class_name:
                matches.append(reg_name)
        return matches
