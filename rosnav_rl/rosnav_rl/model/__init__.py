from .model import RL_Model


def __getattr__(name: str):
    # Backend submodules pull in heavy, framework-specific deps (torch/sb3
    # for stable_baselines3, torch/jax for dreamerv3) — deferred so
    # `import rosnav_rl.model` doesn't pay for both regardless of which
    # backend is actually used (see ModelFactory.get_model_class).
    import importlib

    if name == "StableBaselinesModel":
        return importlib.import_module(".stable_baselines3", __name__).StableBaselinesModel
    if name == "stable_baselines3":
        return importlib.import_module(".stable_baselines3", __name__)
    if name == "DreamerV3Model":
        return importlib.import_module(".dreamerv3", __name__).DreamerV3Model
    if name == "dreamerv3":
        return importlib.import_module(".dreamerv3", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
