"""
Mixin classes for components to use unified error logging.

This module provides mixins that components can inherit from to easily
integrate with the unified error logging system.
"""

from typing import Optional
from contextlib import contextmanager

from .error_logging import (
    ComponentType,
    ErrorSeverity,
    collect_error,
    flush_and_log_errors,
)


class ErrorReportingMixin:
    """Mixin class to add error reporting capabilities to components."""

    def __init__(
        self,
        component_type: Optional[ComponentType] = None,
        component_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # These should be set by the inheriting class
        self._component_type: Optional[ComponentType] = component_type
        self._component_name: Optional[str] = component_name

    def _report_error(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        component_name: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        """Report an error to the unified logging system.

        Args:
            message: Error message to report
            severity: Severity level of the error
            error_type: Type of exception (if applicable)
        """
        if not self._component_type or not self._component_name:
            # Fallback to direct logging if not properly initialized
            print(f"ERROR: {self.__class__.__name__}: {message}")
            return

        collect_error(
            component_type=self._component_type,
            component_name=component_name or self._component_name,
            severity=severity,
            message=message,
            error_type=error_type,
        )

    def _report_warning(self, message: str) -> None:
        """Report a warning message."""
        self._report_error(message, severity=ErrorSeverity.WARNING)

    def _report_info(self, message: str) -> None:
        """Report an info message."""
        self._report_error(message, severity=ErrorSeverity.INFO)

    def _report_critical(self, message: str, error_type: Optional[str] = None) -> None:
        """Report a critical error."""
        self._report_error(
            message, severity=ErrorSeverity.CRITICAL, error_type=error_type
        )

    @contextmanager
    def _error_context(self, operation: str):
        """Context manager for catching and reporting errors during operations.

        Args:
            operation: Description of the operation being performed

        Usage:
            with self._error_context("initializing data source"):
                # code that might fail
                self._initialize_data()
        """
        try:
            yield
        except Exception as e:
            error_type = type(e).__name__
            message = f"Failed {operation}: {str(e)}"
            self._report_error(message, error_type=error_type)
            # Re-raise the exception so calling code can handle it appropriately
            raise


def flush_errors_decorator(func):
    """Decorator to flush errors after function execution.

    Use this decorator on methods that should trigger error flushing,
    typically at the end of major operations or iterations.

    Usage:
        @flush_errors_decorator
        def step(self):
            # your step logic
            pass
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            flush_and_log_errors()

    return wrapper


__all__ = [
    "ErrorReportingMixin",
    "flush_errors_decorator",
]
