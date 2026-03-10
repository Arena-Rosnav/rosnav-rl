"""
Generic logger abstraction for RosNav-RL components.

This module provides a flexible logging system that can work with different
logging backends including ROS2, standard Python logging, and custom loggers.
The system automatically detects the available logging infrastructure and
provides a consistent interface across all components.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol
from enum import Enum
import sys


class LogLevel(Enum):
    """Standard log levels across all logger implementations."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerProtocol(Protocol):
    """Protocol defining the interface for logger implementations."""

    def debug(self, message: str, **kwargs) -> None: ...
    def info(self, message: str, **kwargs) -> None: ...
    def warning(self, message: str, **kwargs) -> None: ...
    def error(self, message: str, **kwargs) -> None: ...
    def critical(self, message: str, **kwargs) -> None: ...


class BaseLogger(ABC):
    """Abstract base class for all logger implementations."""

    def __init__(self, name: str = "rosnav_rl", **kwargs):
        self.name = name
        self.enabled = True

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        pass

    @abstractmethod
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        pass

    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Log a message at the specified level."""
        if not self.enabled:
            return

        level_method = {
            LogLevel.DEBUG: self.debug,
            LogLevel.INFO: self.info,
            LogLevel.WARNING: self.warning,
            LogLevel.ERROR: self.error,
            LogLevel.CRITICAL: self.critical,
        }.get(level)

        if level_method:
            level_method(message, **kwargs)

    def enable(self) -> None:
        """Enable logging."""
        self.enabled = True

    def disable(self) -> None:
        """Disable logging."""
        self.enabled = False


class ROS2Logger(BaseLogger):
    """ROS2 node-based logger implementation."""

    def __init__(self, node: Any, name: str = "rosnav_rl", **kwargs):
        """Initialize ROS2 logger with a ROS2 node.

        Args:
            node: ROS2 node instance with get_logger() method
            name: Logger name (defaults to node name if available)
        """
        super().__init__(name, **kwargs)
        self.node = node

        # Try to get the ROS2 logger from the node
        if hasattr(node, "get_logger"):
            self.ros_logger = node.get_logger()
            # Use node name if available
            if hasattr(node, "get_name"):
                self.name = node.get_name()
        else:
            raise ValueError(
                f"Provided node does not have get_logger() method: {type(node)}"
            )

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message using ROS2 logger."""
        if self.enabled and hasattr(self.ros_logger, "debug"):
            self.ros_logger.debug(message)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message using ROS2 logger."""
        if self.enabled:
            self.ros_logger.info(message)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message using ROS2 logger."""
        if self.enabled:
            self.ros_logger.warning(message)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message using ROS2 logger."""
        if self.enabled:
            self.ros_logger.error(message)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message using ROS2 logger."""
        if self.enabled:
            self.ros_logger.fatal(message)  # ROS2 uses 'fatal' for critical


class ROS1Logger(BaseLogger):
    """ROS1 rospy-based logger implementation."""

    def __init__(self, name: str = "rosnav_rl", **kwargs):
        """Initialize ROS1 logger using rospy."""
        super().__init__(name, **kwargs)

        try:
            import rospy

            self.rospy = rospy
        except ImportError:
            raise ImportError("rospy not available for ROS1Logger")

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message using rospy."""
        if self.enabled:
            self.rospy.logdebug(f"[{self.name}] {message}")

    def info(self, message: str, **kwargs) -> None:
        """Log an info message using rospy."""
        if self.enabled:
            self.rospy.loginfo(f"[{self.name}] {message}")

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message using rospy."""
        if self.enabled:
            self.rospy.logwarn(f"[{self.name}] {message}")

    def error(self, message: str, **kwargs) -> None:
        """Log an error message using rospy."""
        if self.enabled:
            self.rospy.logerr(f"[{self.name}] {message}")

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message using rospy."""
        if self.enabled:
            self.rospy.logfatal(f"[{self.name}] {message}")


class PythonLogger(BaseLogger):
    """Thin stdlib-logging adapter.

    Delegates every call to ``logging.getLogger(name)``, which means it
    participates fully in the Python ``logging`` hierarchy: level filtering,
    propagation to the root handler, and external ``setLevel()`` calls all
    work as expected.

    **No handler is installed here.** The stdlib root / application handler
    configuration is responsible for output.  This prevents duplicate log
    lines and allows the ``rosnav_rl.*`` namespace levels set by
    ``configure_rosnav_rl_logging()`` to take effect naturally.
    """

    def __init__(self, name: str = "rosnav_rl", level: str = "INFO", **kwargs):
        """Initialize Python logger adapter.

        Args:
            name:  Logger name — should be a dotted namespace like
                   ``'rosnav_rl'`` or ``'rosnav_rl.errors'`` so it sits
                   correctly in the stdlib hierarchy.
            level: Initial log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
                   Prefer controlling this from outside via
                   ``logging.getLogger(name).setLevel(...)``.
        """
        super().__init__(name, **kwargs)
        import logging as _stdlib_logging

        self.logger = _stdlib_logging.getLogger(name)
        numeric_level = getattr(_stdlib_logging, level.upper(), _stdlib_logging.INFO)
        self.logger.setLevel(numeric_level)
        # No handler installed — let the application's logging configuration handle output.

    def debug(self, message: str, **kwargs) -> None:
        if self.enabled:
            self.logger.debug(message)

    def info(self, message: str, **kwargs) -> None:
        if self.enabled:
            self.logger.info(message)

    def warning(self, message: str, **kwargs) -> None:
        if self.enabled:
            self.logger.warning(message)

    def error(self, message: str, **kwargs) -> None:
        if self.enabled:
            self.logger.error(message)

    def critical(self, message: str, **kwargs) -> None:
        if self.enabled:
            self.logger.critical(message)


class ConsoleLogger(BaseLogger):
    """Simple console-based logger implementation."""

    def __init__(self, name: str = "rosnav_rl", colored: bool = True, **kwargs):
        """Initialize console logger.

        Args:
            name: Logger name
            colored: Whether to use colored output
        """
        super().__init__(name, **kwargs)
        self.colored = colored

        # ANSI color codes
        self.colors = (
            {
                LogLevel.DEBUG: "\033[36m",  # Cyan
                LogLevel.INFO: "\033[32m",  # Green
                LogLevel.WARNING: "\033[33m",  # Yellow
                LogLevel.ERROR: "\033[31m",  # Red
                LogLevel.CRITICAL: "\033[35m",  # Magenta
                "RESET": "\033[0m",  # Reset
            }
            if colored
            else {}
        )

    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format message with colors and level."""
        if self.colored:
            color = self.colors.get(level, "")
            reset = self.colors.get("RESET", "")
            return f"{color}[{level.value}] [{self.name}] {message}{reset}"
        else:
            return f"[{level.value}] [{self.name}] {message}"

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message to console."""
        if self.enabled:
            print(self._format_message(LogLevel.DEBUG, message))

    def info(self, message: str, **kwargs) -> None:
        """Log an info message to console."""
        if self.enabled:
            print(self._format_message(LogLevel.INFO, message))

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message to console."""
        if self.enabled:
            print(self._format_message(LogLevel.WARNING, message))

    def error(self, message: str, **kwargs) -> None:
        """Log an error message to console."""
        if self.enabled:
            print(self._format_message(LogLevel.ERROR, message), file=sys.stderr)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message to console."""
        if self.enabled:
            print(self._format_message(LogLevel.CRITICAL, message), file=sys.stderr)


class SilentLogger(BaseLogger):
    """Silent logger that doesn't output anything (useful for testing)."""

    def debug(self, message: str, **kwargs) -> None:
        pass

    def info(self, message: str, **kwargs) -> None:
        pass

    def warning(self, message: str, **kwargs) -> None:
        pass

    def error(self, message: str, **kwargs) -> None:
        pass

    def critical(self, message: str, **kwargs) -> None:
        pass


class LoggerFactory:
    """Factory for creating appropriate logger instances."""

    @staticmethod
    def create_logger(
        logger_type: str = "auto",
        node: Optional[Any] = None,
        name: str = "rosnav_rl",
        **kwargs,
    ) -> BaseLogger:
        """Create a logger instance based on the specified type.

        Args:
            logger_type: Type of logger ("auto", "ros2", "ros1", "python", "console", "silent")
            node: ROS2 node instance (required for ros2 logger)
            name: Logger name
            **kwargs: Additional arguments for the logger

        Returns:
            Logger instance

        Raises:
            ValueError: If logger type is invalid or required dependencies are missing
        """
        if logger_type == "auto":
            return LoggerFactory._create_auto_logger(node, name, **kwargs)
        elif logger_type == "ros2":
            if node is None:
                raise ValueError("ROS2 logger requires a node parameter")
            return ROS2Logger(node, name, **kwargs)
        elif logger_type == "ros1":
            return ROS1Logger(name, **kwargs)
        elif logger_type == "python":
            return PythonLogger(name, **kwargs)
        elif logger_type == "console":
            return ConsoleLogger(name, **kwargs)
        elif logger_type == "silent":
            return SilentLogger(name, **kwargs)
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")

    @staticmethod
    def _create_auto_logger(node: Optional[Any], name: str, **kwargs) -> BaseLogger:
        """Automatically detect and create the best available logger."""
        # Try ROS2 first if node is provided
        if node is not None:
            try:
                return ROS2Logger(node, name, **kwargs)
            except (ImportError, ValueError):
                pass

        # Fall back to Python logging
        try:
            return PythonLogger(name, **kwargs)
        except ImportError:
            pass

        # Final fallback to console logger
        return ConsoleLogger(name, **kwargs)


# Global logger instance
_global_logger: Optional[BaseLogger] = None


def get_logger() -> BaseLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = LoggerFactory.create_logger()
    return _global_logger


def set_logger(logger: BaseLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger


def configure_logger(
    logger_type: str = "auto",
    node: Optional[Any] = None,
    name: str = "rosnav_rl",
    **kwargs,
) -> BaseLogger:
    """Configure and set the global logger."""
    logger = LoggerFactory.create_logger(logger_type, node, name, **kwargs)
    set_logger(logger)
    return logger


__all__ = [
    "LogLevel",
    "LoggerProtocol",
    "BaseLogger",
    "ROS2Logger",
    "ROS1Logger",
    "PythonLogger",
    "ConsoleLogger",
    "SilentLogger",
    "LoggerFactory",
    "get_logger",
    "set_logger",
    "configure_logger",
]
