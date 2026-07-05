_LAZY_MODULES = {
    "Dreamer": ".dreamer",
    "DreamerV3Model": ".dreamerv3_model",
    "UUID": ".envs.wrappers",
    "ChannelFirsttoLast": ".envs.wrappers",
    "RenameObsForDreamer": ".envs.wrappers",
    "ResetWoInfo": ".envs.wrappers",
    "SelectAction": ".envs.wrappers",
    "TimeLimit": ".envs.wrappers",
    "WoTruncatedFlag": ".envs.wrappers",
    "Damy": ".parallel",
    "Parallel": ".parallel",
}


def __getattr__(name: str):
    # Dreamer/envs.wrappers/parallel pull in torch + the full DreamerV3
    # network stack. Deferred so importing a lightweight sibling (e.g.
    # `rosnav_rl.model.dreamerv3.cfg`, a plain pydantic module reached via
    # `cfg/agent.py`'s AgentConfig) doesn't pay that cost too.
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
