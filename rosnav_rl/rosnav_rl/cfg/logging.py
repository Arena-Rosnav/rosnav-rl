"""rosnav_rl logging configuration and helpers.

Defines ``LoggingCfg`` — a pydantic model for per-namespace log levels — and
``configure_rosnav_rl_logging()``, which applies those levels to the Python
``logging`` hierarchy and the global ``ErrorCollector``.

This module is intentionally scoped to the ``rosnav_rl`` package:
- It controls ``rosnav_rl.*`` stdlib logger namespaces.
- It sets the ``ErrorCollector.min_severity`` threshold.
- It knows nothing about trainers or framework-specific loggers
  (those are arena_training concerns, see ``arena_rosnav_rl.tools.log_utils``).

Exported from ``rosnav_rl.cfg``:
    from rosnav_rl.cfg import LoggingCfg, configure_rosnav_rl_logging, VERBOSE_TO_LEVEL

Example YAML fragment (inside ``arena_cfg:``):

    logging:
      default_level: INFO
      overrides:
        rosnav_rl.observations: WARNING   # silence per-step obs noise
        rosnav_rl.reward:       WARNING   # silence per-step reward breakdown
        rosnav_rl.spaces:       WARNING   # silence space auto-load info
        rosnav_rl.model.dreamerv3.helper: INFO   # stage-transition banners
"""

from __future__ import annotations

import logging as _logging
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field
from rosnav_rl.utils.logging.error_logging import (
    ErrorSeverity,
    get_error_collector,
)
# ── Type alias ──────────────────────────────────────────────────────────────

_LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Mapping: verbose int (from general_cfg.verbose) → stdlib log level
VERBOSE_TO_LEVEL: Dict[int, int] = {
    0: _logging.WARNING,
    1: _logging.INFO,
    2: _logging.DEBUG,
}

# Mapping: stdlib log level → ErrorCollector min_severity threshold
_STDLIB_TO_ERROR_SEVERITY: Dict[int, ErrorSeverity] = {
    _logging.DEBUG: ErrorSeverity.DEBUG,
    _logging.INFO: ErrorSeverity.INFO,
    _logging.WARNING: ErrorSeverity.WARNING,
    _logging.ERROR: ErrorSeverity.ERROR,
    _logging.CRITICAL: ErrorSeverity.CRITICAL,
}


# ── Config model ────────────────────────────────────────────────────────────

class LoggingCfg(BaseModel):
    """Per-namespace logging levels for rosnav_rl components and the ROS 2 system.

    ``default_level`` is applied to the entire ``rosnav_rl`` root logger first.
    ``overrides`` entries are applied afterwards and take precedence.

    ``ros_level`` controls the ROS 2 / rcutils root log level (the level you
    would set with ``--log-level`` on the CLI).  ``ros_overrides`` allows
    per-node fine-tuning, e.g. silencing a single noisy node while keeping the
    rest at INFO.
    """

    default_level: _LOG_LEVEL = Field(
        default="INFO",
        description="Log level applied to the entire 'rosnav_rl' namespace.",
    )
    overrides: Dict[str, _LOG_LEVEL] = Field(
        default_factory=lambda: {
            "rosnav_rl.observations": "WARNING",
            "rosnav_rl.reward": "WARNING",
            "rosnav_rl.spaces": "WARNING",
        },
        description=(
            "Per-namespace overrides applied after default_level. "
            "Keys are Python logger names, e.g. 'rosnav_rl.reward'."
        ),
    )
    ros_level: Optional[_LOG_LEVEL] = Field(
        default=None,
        description=(
            "ROS 2 root logger level (rcutils/rclpy). "
            "None = do not touch the ROS 2 log level. "
            "Equivalent to passing '--log-level <level>' on the CLI."
        ),
    )
    ros_overrides: Dict[str, _LOG_LEVEL] = Field(
        default_factory=dict,
        description=(
            "Per-node ROS 2 logger level overrides applied after ros_level. "
            "Keys are ROS 2 logger names (typically node names, e.g. "
            "'Arena_Trainer', 'task_generator_node')."
        ),
    )


# ── Helper function ─────────────────────────────────────────────────────────

def configure_rosnav_rl_logging(
    logging_cfg: Optional[LoggingCfg],
    verbose: int = 0,
) -> None:
    """Apply log levels to all ``rosnav_rl.*`` stdlib namespaces and the
    global ``ErrorCollector`` min-severity threshold.

    Priority (highest first):
    1. Per-namespace ``overrides`` in *logging_cfg*
    2. ``logging_cfg.default_level`` → ``rosnav_rl`` root logger
    3. Fallback: map *verbose* int (0→WARNING, 1→INFO, 2→DEBUG)

    The same effective root level is also forwarded to the ``ErrorCollector``
    so that per-step INFO/WARNING noise can be suppressed without losing the
    collect-then-dump paradigm.

    Additionally serializes the config into env vars
    ``ROSNAV_RL_LOG_LEVEL`` / ``ROSNAV_RL_LOG_OVERRIDES`` so that child
    processes (dreamerv3.Parallel workers, etc.) can bootstrap their own
    loggers without a separate call to this function.

    Args:
        logging_cfg: ``LoggingCfg`` from ``arena_cfg.logging``, or *None* to
                     fall back to the *verbose* integer mapping.
        verbose:     Integer verbose level, used only when *logging_cfg* is
                     *None* (0=WARNING, 1=INFO, 2=DEBUG).
    """
    import json as _json
    import os as _os

    fallback_level = VERBOSE_TO_LEVEL.get(
        int(verbose), _logging.INFO if verbose else _logging.WARNING
    )

    if logging_cfg is not None:
        root_level = getattr(_logging, logging_cfg.default_level, _logging.INFO)
        _logging.getLogger("rosnav_rl").setLevel(root_level)
        for ns, lvl_str in logging_cfg.overrides.items():
            _logging.getLogger(ns).setLevel(
                getattr(_logging, lvl_str, root_level)
            )
        # ── Propagate to child processes via env vars ──────────────────────
        _os.environ["ROSNAV_RL_LOG_LEVEL"] = logging_cfg.default_level
        _os.environ["ROSNAV_RL_LOG_OVERRIDES"] = _json.dumps(dict(logging_cfg.overrides))
    else:
        level_str = {
            _logging.WARNING: "WARNING",
            _logging.INFO: "INFO",
            _logging.DEBUG: "DEBUG",
        }.get(fallback_level, "WARNING")
        _logging.getLogger("rosnav_rl").setLevel(fallback_level)
        _os.environ["ROSNAV_RL_LOG_LEVEL"] = level_str
        _os.environ.pop("ROSNAV_RL_LOG_OVERRIDES", None)

    # Sync ErrorCollector threshold to the effective root level so the
    # collect-then-dump report also respects the verbosity setting.
    effective_root = (
        getattr(_logging, logging_cfg.default_level, _logging.INFO)
        if logging_cfg is not None
        else fallback_level
    )
    error_severity = _STDLIB_TO_ERROR_SEVERITY.get(effective_root, ErrorSeverity.INFO)
    get_error_collector().set_min_severity(error_severity)
