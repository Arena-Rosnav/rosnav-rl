"""Model Factory for creating RL models dynamically based on configuration."""

import importlib
from typing import TYPE_CHECKING, Callable, Dict, Tuple, Type

from rosnav_rl.cfg.framework import FrameworkCfg
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

if TYPE_CHECKING:
    from rosnav_rl.rl_agent import RL_Agent
    from rosnav_rl.model.model import RL_Model


class ModelFactory:
    """Factory class for creating RL models dynamically based on framework configuration.

    This factory pattern eliminates the need for explicit isinstance checks and makes
    adding new frameworks more straightforward. New frameworks can be registered
    using the @ModelFactory.register decorator.

    Example:
        @ModelFactory.register(SupportedRLFrameworks.NEW_FRAMEWORK)
        class NewFrameworkModel(RL_Model):
            pass
    """

    _model_registry: Dict[SupportedRLFrameworks, Type["RL_Model"]] = {}
    # Backend modules are imported lazily on first lookup (via get_model_class)
    # so constructing an SB3 agent doesn't also pull in DreamerV3's torch/jax
    # dependency chain, and vice versa.
    _lazy_import_paths: Dict[SupportedRLFrameworks, Tuple[str, str]] = {
        SupportedRLFrameworks.STABLE_BASELINES3: (
            "rosnav_rl.model.stable_baselines3",
            "StableBaselinesModel",
        ),
        SupportedRLFrameworks.DREAMER_V3: (
            "rosnav_rl.model.dreamerv3.dreamerv3_model",
            "DreamerV3Model",
        ),
    }

    @classmethod
    def register(cls, framework: SupportedRLFrameworks) -> Callable:
        """Decorator to register a model class for a specific framework.

        Args:
            framework: The RL framework identifier

        Returns:
            Decorator function that registers the model class

        Example:
            @ModelFactory.register(SupportedRLFrameworks.STABLE_BASELINES3)
            class StableBaselinesModel(RL_Model):
                pass
        """

        def decorator(model_class: Type["RL_Model"]) -> Type["RL_Model"]:
            cls._model_registry[framework] = model_class
            return model_class

        return decorator

    @classmethod
    def register_model(
        cls, framework: SupportedRLFrameworks, model_class: Type["RL_Model"]
    ) -> None:
        """Register a model class for a specific framework (programmatic registration).

        Args:
            framework: The RL framework identifier
            model_class: The model class to associate with the framework
        """
        cls._model_registry[framework] = model_class

    @classmethod
    def get_model_class(cls, framework: SupportedRLFrameworks) -> Type["RL_Model"]:
        """Resolve a framework identifier to its model class, importing the
        backend module on first lookup only (see ``_lazy_import_paths``).

        Args:
            framework: The RL framework identifier

        Returns:
            The model class registered (or lazily importable) for that framework

        Raises:
            ValueError: If the framework is not supported
        """
        if framework not in cls._model_registry:
            if framework not in cls._lazy_import_paths:
                raise ValueError(
                    f"Unsupported RL framework: {framework}. "
                    f"Supported frameworks: {cls.get_supported_frameworks()}"
                )
            module_path, class_name = cls._lazy_import_paths[framework]
            module = importlib.import_module(module_path)
            cls._model_registry[framework] = getattr(module, class_name)

        return cls._model_registry[framework]

    @classmethod
    def create_model_instance(
        cls, framework_cfg: FrameworkCfg, rl_agent: "RL_Agent", **kwargs
    ) -> "RL_Model":
        """Create an RL model instance based on the framework configuration.

        Each registered model class is responsible for accepting a uniform
        ``(rl_agent, algorithm_cfg)`` signature.  The factory simply looks
        up the right class and delegates — no framework-specific branching.

        Args:
            framework_cfg: Configuration object containing framework settings
            rl_agent: The RL agent instance
            **kwargs: Forwarded to the model class's constructor (e.g.
                ``inference_only=True`` for :class:`DreamerV3Model`). Model
                classes that don't accept a given kwarg will raise a
                ``TypeError`` — callers must only pass kwargs known to be
                supported by the resolved framework's model class.

        Returns:
            An instance of the appropriate RL model

        Raises:
            ValueError: If the framework is not supported
        """
        model_class = cls.get_model_class(framework_cfg.name)
        return model_class.from_framework_cfg(
            rl_agent=rl_agent,
            framework_cfg=framework_cfg,
            **kwargs,
        )

    @classmethod
    def get_supported_frameworks(cls) -> list[SupportedRLFrameworks]:
        """Get a list of all supported frameworks.

        Returns:
            List of supported framework identifiers
        """
        return list(
            {*cls._model_registry.keys(), *cls._lazy_import_paths.keys()}
        )
