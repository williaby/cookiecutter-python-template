"""
Observability utilities for agent system.

This module provides structured logging and OpenTelemetry integration for
comprehensive monitoring and debugging of the agent system. It includes
tracing, metrics, and structured logging for production observability.
"""

import functools
import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from src.utils.time_utils import utc_timestamp

# Optional OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None  # type: ignore[assignment]


class StructuredLogger:
    """Structured logger with JSON formatting and correlation IDs."""

    def __init__(self, name: str, correlation_id: str | None = None) -> None:
        """Initialize structured logger.

        Args:
            name: Logger name
            correlation_id: Optional correlation ID for request tracking
        """
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id
        self._local = threading.local()

        # Configure structured logging format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _get_context(self) -> dict[str, Any]:
        """Get current logging context."""
        context = {
            "timestamp": utc_timestamp(),
            "logger": self.logger.name,
            "thread_id": threading.current_thread().ident,
        }

        if self.correlation_id:
            context["correlation_id"] = self.correlation_id

        # Add span context if available
        if OPENTELEMETRY_AVAILABLE and trace:
            span = trace.get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                context.update(
                    {
                        "trace_id": format(span_context.trace_id, "032x"),
                        "span_id": format(span_context.span_id, "016x"),
                    },
                )

        return context

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with structured context."""
        context = self._get_context()
        context.update(kwargs)
        self.logger.info(message, extra={"structured_data": context})

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with structured context."""
        context = self._get_context()
        context.update(kwargs)
        self.logger.error(message, extra={"structured_data": context})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with structured context."""
        context = self._get_context()
        context.update(kwargs)
        self.logger.warning(message, extra={"structured_data": context})

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with structured context."""
        context = self._get_context()
        context.update(kwargs)
        self.logger.debug(message, extra={"structured_data": context})


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": utc_timestamp(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add structured data if available
        if hasattr(record, "structured_data"):
            log_entry.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class OpenTelemetryInstrumentor:
    """OpenTelemetry instrumentation for agent system."""

    def __init__(self, service_name: str = "promptcraft-agents") -> None:
        """Initialize OpenTelemetry instrumentation.

        Args:
            service_name: Name of the service for tracing
        """
        self.service_name = service_name
        self.tracer: Any = None
        self.initialized = False

        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Create resource with service information
        resource = Resource.create({"service.name": self.service_name, "service.version": "1.0.0"})

        # Setup tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Setup Jaeger exporter (optional)
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
        except Exception as e:  # nosec B110 - Jaeger is optional dependency
            # Jaeger not available, continue without it
            # This is expected when Jaeger is not configured and is not an error
            logging.getLogger(__name__).debug("Jaeger exporter not available: %s", e)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        # Setup logging instrumentation
        LoggingInstrumentor().instrument(set_logging_format=True)

        self.initialized = True

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any:
        """Start a new span.

        Args:
            name: Span name
            attributes: Optional span attributes

        Returns:
            Span context manager
        """
        if not self.initialized or not self.tracer:
            return NoOpSpan()

        span = self.tracer.start_span(name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        return span

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes: Any) -> Any:
        """Context manager for tracing operations.

        Args:
            operation_name: Name of the operation
            **attributes: Additional span attributes
        """
        if not self.initialized or not self.tracer:
            yield
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            # Set attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class NoOpSpan:
    """No-op span for when OpenTelemetry is not available."""

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass


class ObservabilityMixin:
    """Mixin class for adding observability capabilities to other classes."""

    def __init__(self) -> None:
        """Initialize observability capabilities."""
        if not hasattr(super(), "__init__"):
            super().__init__()

        # Initialize observability components
        self._logger = create_structured_logger(self.__class__.__name__)
        self._instrumentor = get_instrumentor()
        self._metrics = get_metrics_collector()

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with structured context."""
        self._logger.info(message, **kwargs)

    def log_error(self, message: str, **kwargs: Any) -> None:
        """Log error message with structured context."""
        self._logger.error(message, **kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with structured context."""
        self._logger.warning(message, **kwargs)

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with structured context."""
        self._logger.debug(message, **kwargs)

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes: Any) -> Any:
        """Context manager for tracing operations."""
        with self._instrumentor.trace_operation(operation_name, **attributes) as span:
            yield span

    def increment_metric(self, metric_name: str, value: int = 1) -> None:
        """Increment a metric counter."""
        self._metrics.increment_counter(metric_name, value)

    def record_duration(self, metric_name: str, duration: float) -> None:
        """Record a duration metric."""
        self._metrics.record_duration(metric_name, duration)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        return self._metrics.get_metrics()


class AgentMetrics:
    """Agent system metrics collection."""

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self.metrics = {
            "agent_executions_total": 0,
            "agent_executions_success": 0,
            "agent_executions_failed": 0,
            "agent_execution_duration_seconds": [],
            "agent_timeouts_total": 0,
            "agent_registrations_total": 0,
            "agent_cache_hits": 0,
            "agent_cache_misses": 0,
        }
        self._lock = threading.Lock()

    def increment_counter(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            if metric_name in self.metrics:
                current_value = self.metrics[metric_name]
                if isinstance(current_value, int):
                    self.metrics[metric_name] = current_value + value

    def record_duration(self, metric_name: str, duration: float) -> None:
        """Record a duration metric."""
        with self._lock:
            if metric_name in self.metrics:
                metric_value = self.metrics[metric_name]
                if isinstance(metric_value, list):
                    metric_value.append(duration)

                    # Keep only last N measurements to prevent memory bloat
                    max_measurements = 1000
                    if len(metric_value) > max_measurements:
                        self.metrics[metric_name] = metric_value[-max_measurements:]

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            snapshot = self.metrics.copy()

            # Calculate duration statistics
            durations_value = snapshot.get("agent_execution_duration_seconds", [])
            if isinstance(durations_value, list) and durations_value:
                durations = durations_value  # type: list[float]
                snapshot["agent_execution_duration_stats"] = {
                    "count": len(durations),
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "p95": self._percentile(durations, 95),
                    "p99": self._percentile(durations, 99),
                }

            return snapshot

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


# Global instances
_observability_instrumentor = None
_metrics_collector = None


def get_instrumentor() -> "OpenTelemetryInstrumentor":
    """Get global OpenTelemetry instrumentor."""
    global _observability_instrumentor  # noqa: PLW0603
    if _observability_instrumentor is None:
        _observability_instrumentor = OpenTelemetryInstrumentor()
    return _observability_instrumentor


def get_metrics_collector() -> "AgentMetrics":
    """Get global metrics collector."""
    global _metrics_collector  # noqa: PLW0603
    if _metrics_collector is None:
        _metrics_collector = AgentMetrics()
    return _metrics_collector


def trace_agent_operation(operation_name: str) -> Callable:
    """Decorator for tracing agent operations.

    Args:
        operation_name: Name of the operation to trace
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            instrumentor = get_instrumentor()
            metrics = get_metrics_collector()

            # Extract agent information from arguments
            agent_id = None
            if args and hasattr(args[0], "agent_id"):
                agent_id = args[0].agent_id

            attributes = {"operation": operation_name, "function": func.__name__}

            if agent_id:
                attributes["agent_id"] = agent_id

            with instrumentor.trace_operation(operation_name, **attributes) as span:
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    # Record success metrics
                    metrics.increment_counter("agent_executions_success")

                    # Add result attributes to span
                    if span and hasattr(result, "confidence"):
                        span.set_attribute("result_confidence", result.confidence)

                    return result

                except Exception as e:
                    # Record failure metrics
                    metrics.increment_counter("agent_executions_failed")

                    # Add error attributes to span
                    if span:
                        span.set_attribute("error_type", type(e).__name__)
                        span.set_attribute("error_message", str(e))

                    raise

                finally:
                    # Record duration
                    duration = time.time() - start_time
                    metrics.record_duration("agent_execution_duration_seconds", duration)
                    metrics.increment_counter("agent_executions_total")

                    if span:
                        span.set_attribute("duration_seconds", duration)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            instrumentor = get_instrumentor()
            metrics = get_metrics_collector()

            # Extract agent information from arguments
            agent_id = None
            if args and hasattr(args[0], "agent_id"):
                agent_id = args[0].agent_id

            attributes = {"operation": operation_name, "function": func.__name__}

            if agent_id:
                attributes["agent_id"] = agent_id

            with instrumentor.trace_operation(operation_name, **attributes) as span:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # Record success metrics
                    metrics.increment_counter("agent_executions_success")

                    return result

                except Exception as e:
                    # Record failure metrics
                    metrics.increment_counter("agent_executions_failed")

                    # Add error attributes to span
                    if span:
                        span.set_attribute("error_type", type(e).__name__)
                        span.set_attribute("error_message", str(e))

                    raise

                finally:
                    # Record duration
                    duration = time.time() - start_time
                    metrics.record_duration("agent_execution_duration_seconds", duration)
                    metrics.increment_counter("agent_executions_total")

                    if span:
                        span.set_attribute("duration_seconds", duration)

        # Return appropriate wrapper based on function type
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return sync_wrapper

    return decorator


def create_structured_logger(name: str, correlation_id: str | None = None) -> StructuredLogger:
    """Create a structured logger instance.

    Args:
        name: Logger name
        correlation_id: Optional correlation ID

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, correlation_id)


def log_agent_event(event_type: str, agent_id: str, message: str, level: str = "info", **additional_data: Any) -> None:
    """Log an agent system event with structured data.

    Args:
        event_type: Type of event (registration, execution, error, etc.)
        agent_id: Agent identifier
        message: Log message
        level: Log level (info, warning, error, debug)
        **additional_data: Additional structured data
    """
    logger = create_structured_logger("agent_system")

    event_data = {"event_type": event_type, "agent_id": agent_id, **additional_data}

    log_method = getattr(logger, level, logger.info)
    log_method(message, **event_data)


# Export public API
__all__ = [
    "AgentMetrics",
    "ObservabilityMixin",
    "OpenTelemetryInstrumentor",
    "StructuredLogger",
    "create_structured_logger",
    "get_instrumentor",
    "get_metrics_collector",
    "log_agent_event",
    "trace_agent_operation",
]
