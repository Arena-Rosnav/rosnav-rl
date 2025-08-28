"""Model Factory for creating RL models dynamically based on configuration."""

from typing import TYPE_CHECKING, Callable, Dict, Type

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
    def create_model_instance(
        cls, framework_cfg: FrameworkCfg, rl_agent: "RL_Agent"
    ) -> "RL_Model":
        """Create an RL model instance based on the framework configuration.

        Args:
            framework_cfg: Configuration object containing framework settings
            rl_agent: The RL agent instance

        Returns:
            An instance of the appropriate RL model

        Raises:
            ValueError: If the framework is not supported
        """
        framework_name = framework_cfg.name

        if framework_name not in cls._model_registry:
            raise ValueError(
                f"Unsupported RL framework: {framework_name}. "
                f"Supported frameworks: {list(cls._model_registry.keys())}"
            )

        model_class = cls._model_registry[framework_name]

        # Create model with appropriate parameters based on framework type
        if framework_name == SupportedRLFrameworks.STABLE_BASELINES3:
            return model_class(
                rl_agent=rl_agent,
                algorithm_cfg=framework_cfg.algorithm,
            )
        elif framework_name == SupportedRLFrameworks.DREAMER_V3:
            return model_class(rl_agent=rl_agent, algorithm_cfg=framework_cfg)
        else:
            # Fallback for future frameworks - assume they follow the DreamerV3 pattern
            return model_class(rl_agent=rl_agent, algorithm_cfg=framework_cfg)

    @classmethod
    def get_supported_frameworks(cls) -> list[SupportedRLFrameworks]:
        """Get a list of all supported frameworks.

        Returns:
            List of supported framework identifiers
        """
        return list(cls._model_registry.keys())


# Register the available models
def _register_models():
    """Register all available model implementations."""
    from rosnav_rl.model.stable_baselines3 import StableBaselinesModel
    from rosnav_rl.model.dreamerv3.dreamerv3_model import DreamerV3Model

    ModelFactory.register_model(
        SupportedRLFrameworks.STABLE_BASELINES3, StableBaselinesModel
    )
    ModelFactory.register_model(SupportedRLFrameworks.DREAMER_V3, DreamerV3Model)


# Auto-register models when module is imported
_register_models()
