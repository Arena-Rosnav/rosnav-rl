"""
Unified error logging system for RosNav-RL components.

This module provides a centralized error collection and logging system that
prevents log spam by collecting errors from all components (observation spaces,
generators, reward units) and logging them in an organized way once per iteration.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict

from .logger import get_logger, LogLevel


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentType(Enum):
    """Types of components that can report errors."""

    OBSERVATION_SPACE = "ObservationSpace"
    GENERATOR = "Generator"
    COLLECTOR = "Collector"
    REWARD_UNIT = "RewardUnit"
    REWARD_FUNCTION = "RewardFunction"
    OTHER = "Other"


@dataclass
class ErrorMessage:
    """Container for error messages."""

    component_type: ComponentType
    component_name: str
    severity: ErrorSeverity
    message: str
    error_type: Optional[str] = None
    count: int = 1

    def __hash__(self):
        """Make ErrorMessage hashable for deduplication."""
        return hash((self.component_type, self.component_name, self.message))

    def __eq__(self, other):
        """Check equality for deduplication."""
        if not isinstance(other, ErrorMessage):
            return False
        return (
            self.component_type == other.component_type
            and self.component_name == other.component_name
            and self.message == other.message
        )


class ErrorCollector:
    """Thread-safe error collector for unified logging."""

    def __init__(self):
        self._errors: Dict[ErrorMessage, int] = {}
        self._lock = threading.Lock()
        self._enabled = True

    def add_error(
        self,
        component_type: ComponentType,
        component_name: str,
        severity: ErrorSeverity,
        message: str,
        error_type: Optional[str] = None,
    ) -> None:
        """Add an error to the collector.

        Args:
            component_type: Type of component reporting the error
            component_name: Name of the specific component
            severity: Severity level of the error
            message: Error message
            error_type: Type of exception (if applicable)
        """
        if not self._enabled:
            return

        error_msg = ErrorMessage(
            component_type=component_type,
            component_name=component_name,
            severity=severity,
            message=message,
            error_type=error_type,
        )

        with self._lock:
            if error_msg in self._errors:
                self._errors[error_msg] += 1
            else:
                self._errors[error_msg] = 1

    def flush_and_log(self) -> None:
        """Flush all collected errors and log them in an organized way."""
        if not self._enabled:
            return

        with self._lock:
            if not self._errors:
                return

            errors_copy = dict(self._errors)
            self._errors.clear()

        self._log_organized_errors(errors_copy)

    def _log_organized_errors(self, errors: Dict[ErrorMessage, int]) -> None:
        """Log errors in an organized format."""
        if not errors:
            return

        # Group errors by severity and component type
        by_severity = defaultdict(lambda: defaultdict(list))
        total_errors = 0

        for error_msg, count in errors.items():
            by_severity[error_msg.severity][error_msg.component_type].append(
                (error_msg, count)
            )
            total_errors += count

        # Format the log message
        lines = [
            f"🚨 RosNav-RL System Status Report ({total_errors} issues)",
            "=" * 80,
        ]

        # Log by severity (most severe first)
        severity_order = [
            ErrorSeverity.CRITICAL,
            ErrorSeverity.ERROR,
            ErrorSeverity.WARNING,
            ErrorSeverity.INFO,
        ]

        for severity in severity_order:
            if severity not in by_severity:
                continue

            severity_count = sum(
                count
                for component_errors in by_severity[severity].values()
                for _, count in component_errors
            )

            # Severity header
            emoji = self._get_severity_emoji(severity)
            lines.extend(
                [
                    f"{emoji} {severity.value} ({severity_count} issues)",
                    "-" * 60,
                ]
            )

            # Group by component type
            for component_type, component_errors in by_severity[severity].items():
                if not component_errors:
                    continue

                component_count = sum(count for _, count in component_errors)
                lines.append(f"  📦 {component_type.value} ({component_count} issues)")

                # Sort by count (most frequent first)
                component_errors.sort(key=lambda x: x[1], reverse=True)

                for error_msg, count in component_errors:
                    count_str = f" (×{count})" if count > 1 else ""
                    lines.append(
                        f"    • {error_msg.component_name}: {error_msg.message}{count_str}"
                    )

                lines.append("")

        # Add summary and suggestions
        lines.extend(
            [
                "💡 Next Steps:",
                "  - Check observation data sources and configurations",
                "  - Verify all required dependencies are available",
                "  - Review component initialization parameters",
                "  - Consider enabling debug mode for detailed analysis",
                "",
            ]
        )

        # Log the complete report
        report = "\n".join(lines)
        self._output_log(
            report, severity=max(by_severity.keys(), key=lambda x: x.value)
        )

    def _get_severity_emoji(self, severity: ErrorSeverity) -> str:
        """Get emoji for severity level."""
        return {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🔥",
        }.get(severity, "❓")

    def _output_log(self, message: str, severity: ErrorSeverity) -> None:
        """Output the log message using the flexible logger system."""
        logger = get_logger()

        # Map our error severity to logger levels
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.ERROR]:
            logger.error(message)
        elif severity == ErrorSeverity.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

    def enable(self) -> None:
        """Enable error collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable error collection."""
        self._enabled = False

    def clear(self) -> None:
        """Clear all collected errors."""
        with self._lock:
            self._errors.clear()

    @property
    def error_count(self) -> int:
        """Get total number of unique errors."""
        with self._lock:
            return len(self._errors)

    @property
    def total_error_occurrences(self) -> int:
        """Get total number of error occurrences (including repeats)."""
        with self._lock:
            return sum(self._errors.values())


# Global error collector instance
_global_error_collector = ErrorCollector()


def get_error_collector() -> ErrorCollector:
    """Get the global error collector instance."""
    return _global_error_collector


def collect_error(
    component_type: ComponentType,
    component_name: str,
    severity: ErrorSeverity,
    message: str,
    error_type: Optional[str] = None,
) -> None:
    """Convenience function to collect an error.

    Args:
        component_type: Type of component reporting the error
        component_name: Name of the specific component
        severity: Severity level of the error
        message: Error message
        error_type: Type of exception (if applicable)
    """
    _global_error_collector.add_error(
        component_type=component_type,
        component_name=component_name,
        severity=severity,
        message=message,
        error_type=error_type,
    )


def flush_and_log_errors() -> None:
    """Flush and log all collected errors."""
    _global_error_collector.flush_and_log()


__all__ = [
    "ErrorSeverity",
    "ComponentType",
    "ErrorCollector",
    "get_error_collector",
    "collect_error",
    "flush_and_log_errors",
]
