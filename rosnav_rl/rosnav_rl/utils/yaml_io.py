"""YAML 1.2 reading and writing through ruamel, the one YAML library of rosnav_rl and arena_training."""

from __future__ import annotations

import io
import os
from typing import Any

from ruamel.yaml import YAML, YAMLError

__all__ = ["YAMLError", "dump_yaml", "dumps_yaml", "load_yaml", "loads_yaml"]


def _reader() -> YAML:
    return YAML(typ="safe", pure=True)


def _writer() -> YAML:
    yaml = YAML()
    yaml.default_flow_style = False
    return yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The mapping a YAML file holds, as plain Python data."""
    with open(path, encoding="utf-8") as file:
        return _reader().load(file)


def loads_yaml(text: str) -> dict[str, Any]:
    """The mapping a YAML string holds, as plain Python data."""
    return _reader().load(text)


def dump_yaml(data: object, path: str | os.PathLike[str]) -> None:
    """Write *data* to a YAML file in block style, keys in insertion order."""
    with open(path, "w", encoding="utf-8") as file:
        _writer().dump(data, file)


def dumps_yaml(data: object) -> str:
    """*data* as a YAML string in block style, keys in insertion order."""
    stream = io.StringIO()
    _writer().dump(data, stream)
    return stream.getvalue()
