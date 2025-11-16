"""Logging mixin for consistent logging across PromptCraft components.

This module provides a reusable mixin class that standardizes logging
setup and configuration across all project components.
"""

import logging
from typing import Any


class LoggerMixin:
    """Mixin to provide consistent logging setup across components.

    This mixin provides a standardized way to set up logging for any class
    in the PromptCraft project. It ensures consistent logger naming and
    configuration while allowing for component-specific customization.

    Example:
        >>> class MyComponent(LoggerMixin):
        ...     def __init__(self):
        ...         super().__init__(logger_name="my_component")
        ...         self.logger.info("Component initialized")
    """

    def __init__(
        self,
        logger_name: str | None = None,
        log_level: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize logger for the component.

        Args:
            logger_name: Custom logger name. If None, uses class name.
            log_level: Custom log level. If None, uses existing level.
            *args: Additional positional arguments for parent classes.
            **kwargs: Additional keyword arguments for parent classes.
        """
        super().__init__(*args, **kwargs)

        # Create logger with hierarchical naming
        if logger_name is None:
            logger_name = f"promptcraft.{self.__class__.__module__}.{self.__class__.__name__}"
        elif not logger_name.startswith("promptcraft."):
            logger_name = f"promptcraft.{logger_name}"

        self.logger = logging.getLogger(logger_name)

        # Set custom log level if provided
        if log_level is not None:
            self.logger.setLevel(log_level)

        # Ensure logger has at least INFO level if not set
        if not self.logger.level:
            self.logger.setLevel(logging.INFO)

    def log_method_entry(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log method entry with parameters (for debugging).

        Args:
            method_name: Name of the method being entered.
            *args: Method arguments.
            **kwargs: Method keyword arguments.
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "Entering %s with args=%s, kwargs=%s",
                method_name,
                args,
                {k: v for k, v in kwargs.items() if not k.startswith("_")},
            )

    def log_method_exit(self, method_name: str, result: Any = None) -> None:
        """Log method exit with result (for debugging).

        Args:
            method_name: Name of the method being exited.
            result: Method return value.
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            if result is not None:
                self.logger.debug("Exiting %s with result=%s", method_name, result)
            else:
                self.logger.debug("Exiting %s", method_name)

    def log_error_with_context(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        method_name: str | None = None,
    ) -> None:
        """Log error with additional context information.

        Args:
            error: The exception that occurred.
            context: Additional context information.
            method_name: Name of the method where error occurred.
        """
        context = context or {}
        location = f" in {method_name}" if method_name else ""

        self.logger.error(
            "Error%s: %s - Context: %s",
            location,
            error,
            context,
            exc_info=True,
        )

    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log performance metrics in a structured format.

        Args:
            metric_name: Name of the performance metric.
            value: Metric value.
            unit: Unit of measurement.
            context: Additional context information.
        """
        context = context or {}
        self.logger.info(
            "PERF_METRIC: %s=%.2f%s - Context: %s",
            metric_name,
            value,
            unit,
            context,
        )

    def log_state_change(
        self,
        from_state: str,
        to_state: str,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log state transitions in a structured format.

        Args:
            from_state: Previous state.
            to_state: New state.
            reason: Reason for state change.
            context: Additional context information.
        """
        context = context or {}
        reason_text = f" (reason: {reason})" if reason else ""

        self.logger.info(
            "STATE_CHANGE: %s -> %s%s - Context: %s",
            from_state,
            to_state,
            reason_text,
            context,
        )

    def log_business_event(
        self,
        event_name: str,
        event_data: dict[str, Any] | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Log business events in a structured format.

        Args:
            event_name: Name of the business event.
            event_data: Event-specific data.
            level: Log level for the event.
        """
        event_data = event_data or {}
        self.logger.log(
            level,
            "BUSINESS_EVENT: %s - Data: %s",
            event_name,
            event_data,
        )


class StructuredLoggerMixin(LoggerMixin):
    """Enhanced logger mixin with structured logging capabilities.

    This extends LoggerMixin with additional structured logging features
    for better observability and monitoring integration.
    """

    def __init__(
        self,
        component_id: str | None = None,
        correlation_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize structured logger.

        Args:
            component_id: Unique identifier for this component instance.
            correlation_id: Correlation ID for request tracing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.component_id = component_id or f"{self.__class__.__name__}_instance"
        self.correlation_id = correlation_id

    def _get_structured_context(
        self,
        additional_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get base structured context for logging.

        Args:
            additional_context: Additional context to merge.

        Returns:
            Combined context dictionary.
        """
        context = {
            "component_id": self.component_id,
            "component_class": self.__class__.__name__,
        }

        if self.correlation_id:
            context["correlation_id"] = self.correlation_id

        if additional_context:
            context.update(additional_context)

        return context

    def log_structured(
        self,
        level: int,
        message: str,
        event_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log with structured context.

        Args:
            level: Log level.
            message: Log message.
            event_type: Type of event (e.g., 'api_call', 'state_change').
            **kwargs: Additional structured data.
        """
        context = self._get_structured_context(kwargs)

        if event_type:
            context["event_type"] = event_type

        self.logger.log(level, "%s - Context: %s", message, context)

    def log_api_call(
        self,
        endpoint: str,
        method: str = "GET",
        status_code: int | None = None,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log API call with structured data.

        Args:
            endpoint: API endpoint called.
            method: HTTP method.
            status_code: Response status code.
            duration_ms: Call duration in milliseconds.
            **kwargs: Additional context.
        """
        context = {
            "endpoint": endpoint,
            "method": method,
            **kwargs,
        }

        if status_code is not None:
            context["status_code"] = status_code

        if duration_ms is not None:
            context["duration_ms"] = duration_ms

        self.log_structured(
            logging.INFO,
            f"API call: {method} {endpoint}",
            event_type="api_call",
            **context,
        )


def get_component_logger(
    component_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Get a standardized logger for a component.

    Args:
        component_name: Name of the component.
        level: Log level to set.

    Returns:
        Configured logger instance.
    """
    logger_name = f"promptcraft.{component_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger
