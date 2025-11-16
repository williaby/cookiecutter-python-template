"""Circuit Breaker Implementation.

This module provides a robust circuit breaker implementation for handling
failures in external service calls with configurable thresholds, exponential
backoff, and thread-safe operation.

Following the approved Phase 1 Issue NEW-11 implementation plan.
"""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from src.utils.observability import get_metrics_collector, trace_agent_operation
from src.utils.secure_random import secure_jitter
from src.utils.time_utils import utc_now

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states following classic pattern."""

    CLOSED = "closed"  # Normal operation - allows all requests
    OPEN = "open"  # Failing state - blocks all requests
    HALF_OPEN = "half_open"  # Testing recovery - allows limited requests


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    def __init__(
        self,
        message: str,
        state: CircuitBreakerState,
        failure_count: int = 0,
        last_failure_time: datetime | None = None,
    ) -> None:
        super().__init__(message)
        self.state = state
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open and blocks request."""

    def __init__(
        self,
        failure_count: int = 0,
        last_failure_time: datetime | None = None,
        recovery_time: datetime | None = None,
    ) -> None:
        message = f"Circuit breaker is open. Failures: {failure_count}, Recovery at: {recovery_time}"
        super().__init__(
            message=message,
            state=CircuitBreakerState.OPEN,
            failure_count=failure_count,
            last_failure_time=last_failure_time,
        )
        self.recovery_time = recovery_time


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # State transition thresholds
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    recovery_timeout: int = 60  # Seconds before testing recovery

    # Retry and backoff configuration
    max_retries: int = 3  # Maximum retry attempts
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Maximum delay between retries
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add jitter to prevent thundering herd

    # Health check configuration
    health_check_interval: int = 30  # Seconds between health checks
    health_check_timeout: float = 5.0  # Timeout for health check calls

    # Monitoring configuration
    enable_metrics: bool = True  # Enable metrics collection
    enable_tracing: bool = True  # Enable distributed tracing

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.backoff_multiplier <= 1:
            raise ValueError("backoff_multiplier must be greater than 1")


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    # State information
    current_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    state_since: datetime = field(default_factory=utc_now)

    # Counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0

    # Current failure tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Timing information
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    next_retry_time: datetime | None = None

    # Health check information
    last_health_check: datetime | None = None
    health_check_count: int = 0
    health_check_failures: int = 0

    def get_success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

    def get_failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100.0


class CircuitBreaker:
    """Thread-safe circuit breaker with exponential backoff and health checks.

    Implements the classic circuit breaker pattern with three states:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Failing fast, all requests rejected
    - HALF_OPEN: Testing recovery, limited requests allowed

    Features:
    - Thread-safe operation for concurrent requests
    - Exponential backoff with jitter
    - Configurable failure and recovery thresholds
    - Health check integration
    - Comprehensive metrics and monitoring
    - Graceful degradation support
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        health_check_func: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            config: Configuration object (uses defaults if None)
            health_check_func: Optional async function for health checks
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.health_check_func = health_check_func

        # Thread-safe state management
        self._lock = threading.RLock()
        self._metrics = CircuitBreakerMetrics()

        # Background tasks
        self._health_check_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Metrics collection
        self._metrics_collector = None
        if self.config.enable_metrics:
            try:
                self._metrics_collector = get_metrics_collector()
            except Exception as e:
                logger.warning("Failed to initialize metrics collector: %s", e)

        logger.info(
            "Circuit breaker '%s' initialized with config: %s",
            self.name,
            self.config,
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._metrics.current_state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics (returns a copy for thread safety)."""
        with self._lock:
            # Create a deep copy to avoid race conditions
            return CircuitBreakerMetrics(
                current_state=self._metrics.current_state,
                state_since=self._metrics.state_since,
                total_requests=self._metrics.total_requests,
                successful_requests=self._metrics.successful_requests,
                failed_requests=self._metrics.failed_requests,
                rejected_requests=self._metrics.rejected_requests,
                consecutive_failures=self._metrics.consecutive_failures,
                consecutive_successes=self._metrics.consecutive_successes,
                last_failure_time=self._metrics.last_failure_time,
                last_success_time=self._metrics.last_success_time,
                next_retry_time=self._metrics.next_retry_time,
                last_health_check=self._metrics.last_health_check,
                health_check_count=self._metrics.health_check_count,
                health_check_failures=self._metrics.health_check_failures,
            )

    def _transition_to_state(self, new_state: CircuitBreakerState) -> None:
        """Transition to a new state with proper logging and metrics.

        Args:
            new_state: Target state to transition to
        """
        old_state = self._metrics.current_state
        if old_state == new_state:
            return

        self._metrics.current_state = new_state
        self._metrics.state_since = utc_now()

        logger.info(
            "Circuit breaker '%s' state transition: %s -> %s",
            self.name,
            old_state.value,
            new_state.value,
        )

        # Reset counters on state transitions
        if new_state == CircuitBreakerState.CLOSED:
            self._metrics.consecutive_failures = 0
            self._metrics.next_retry_time = None
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self._metrics.consecutive_successes = 0

        # Record metrics
        if self._metrics_collector and hasattr(self._metrics_collector, "record_counter"):
            try:
                self._metrics_collector.record_counter(
                    f"circuit_breaker.{self.name}.state_transitions",
                    1,
                    {"from_state": old_state.value, "to_state": new_state.value},
                )
            except Exception as e:
                logger.warning("Failed to record state transition metric: %s", e)

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state.

        Returns:
            True if request should be allowed, False otherwise
        """
        current_time = utc_now()

        if self._metrics.current_state == CircuitBreakerState.CLOSED:
            return True

        if self._metrics.current_state == CircuitBreakerState.OPEN:
            # Check if enough time has passed for recovery attempt
            if self._metrics.next_retry_time is None or current_time >= self._metrics.next_retry_time:
                self._transition_to_state(CircuitBreakerState.HALF_OPEN)
                return True
            return False

        # Allow limited requests to test recovery in HALF_OPEN state
        return self._metrics.current_state == CircuitBreakerState.HALF_OPEN

    def _record_success(self) -> None:
        """Record a successful operation."""
        self._metrics.total_requests += 1
        self._metrics.successful_requests += 1
        self._metrics.consecutive_failures = 0
        self._metrics.consecutive_successes += 1
        self._metrics.last_success_time = utc_now()

        # Transition logic
        if self._metrics.current_state == CircuitBreakerState.HALF_OPEN:
            if self._metrics.consecutive_successes >= self.config.success_threshold:
                self._transition_to_state(CircuitBreakerState.CLOSED)

        # Record metrics
        if self._metrics_collector and hasattr(self._metrics_collector, "record_counter"):
            try:
                self._metrics_collector.record_counter(f"circuit_breaker.{self.name}.successes", 1)
            except Exception as e:
                logger.warning("Failed to record success metric: %s", e)

    def _record_failure(self) -> None:
        """Record a failed operation."""
        self._metrics.total_requests += 1
        self._metrics.failed_requests += 1
        self._metrics.consecutive_failures += 1
        self._metrics.consecutive_successes = 0
        self._metrics.last_failure_time = utc_now()

        # Calculate next retry time with exponential backoff
        delay = min(
            self.config.base_delay * (self.config.backoff_multiplier ** (self._metrics.consecutive_failures - 1)),
            self.config.max_delay,
        )

        if self.config.jitter:
            delay = secure_jitter(delay, 0.1)  # 10% jitter

        self._metrics.next_retry_time = utc_now() + timedelta(seconds=delay)

        # Transition logic
        if (
            self._metrics.current_state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]
            and self._metrics.consecutive_failures >= self.config.failure_threshold
        ):
            self._transition_to_state(CircuitBreakerState.OPEN)

        # Record metrics
        if self._metrics_collector and hasattr(self._metrics_collector, "record_counter"):
            try:
                self._metrics_collector.record_counter(f"circuit_breaker.{self.name}.failures", 1)
            except Exception as e:
                logger.warning("Failed to record failure metric: %s", e)

    def _record_rejection(self) -> None:
        """Record a rejected request."""
        self._metrics.rejected_requests += 1

        # Record metrics
        if self._metrics_collector and hasattr(self._metrics_collector, "record_counter"):
            try:
                self._metrics_collector.record_counter(f"circuit_breaker.{self.name}.rejections", 1)
            except Exception as e:
                logger.warning("Failed to record rejection metric: %s", e)

    @contextmanager
    def _request_context(self) -> Any:
        """Context manager for request tracking and state management."""
        with self._lock:
            if not self._should_allow_request():
                self._record_rejection()
                raise CircuitBreakerOpenError(
                    failure_count=self._metrics.consecutive_failures,
                    last_failure_time=self._metrics.last_failure_time,
                    recovery_time=self._metrics.next_retry_time,
                )

        try:
            yield
            with self._lock:
                self._record_success()
        except Exception:
            with self._lock:
                self._record_failure()
            raise

    async def call_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Exception: Original exception from the function
        """
        if self.config.enable_tracing:
            # Apply tracing decorator to the function
            traced_func = trace_agent_operation(f"circuit_breaker.{self.name}.call")(func)
            return await self._execute_with_retries(traced_func, *args, **kwargs)
        return await self._execute_with_retries(func, *args, **kwargs)

    async def _execute_with_retries(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry logic and circuit breaker protection."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                with self._request_context():
                    return await func(*args, **kwargs)

            except CircuitBreakerOpenError:
                # Don't retry if circuit breaker is open
                raise

            except Exception as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    # Calculate retry delay
                    delay = min(
                        self.config.base_delay * (self.config.backoff_multiplier**attempt),
                        self.config.max_delay,
                    )

                    if self.config.jitter:
                        delay = secure_jitter(delay, 0.1)

                    logger.warning(
                        "Circuit breaker '%s' retry %d/%d after %.2fs: %s",
                        self.name,
                        attempt + 1,
                        self.config.max_retries,
                        delay,
                        str(e),
                    )

                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Circuit breaker '%s' all retries exhausted: %s",
                        self.name,
                        str(e),
                    )

        # All retries exhausted
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No exception captured during retries")

    def call_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a synchronous function with circuit breaker protection.

        Args:
            func: Synchronous function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Exception: Original exception from the function
        """
        with self._request_context():
            return func(*args, **kwargs)

    def decorator(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply circuit breaker protection to a function.

        Args:
            func: Function to protect

        Returns:
            Protected function
        """
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await self.call_async(func, *args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call_sync(func, *args, **kwargs)

        return sync_wrapper

    async def health_check(self) -> bool:
        """Perform health check if configured.

        Returns:
            True if healthy, False otherwise
        """
        if not self.health_check_func:
            return True

        try:
            with self._lock:
                self._metrics.health_check_count += 1
                self._metrics.last_health_check = utc_now()

            # Execute health check with timeout
            result = await asyncio.wait_for(
                self.health_check_func(),
                timeout=self.config.health_check_timeout,
            )

            if not result:
                with self._lock:
                    self._metrics.health_check_failures += 1

            return result

        except Exception as e:
            logger.warning("Health check failed for circuit breaker '%s': %s", self.name, e)
            with self._lock:
                self._metrics.health_check_failures += 1
            return False

    async def start_health_monitoring(self) -> None:
        """Start background health check monitoring."""
        if not self.health_check_func or self._health_check_task:
            return

        async def health_check_loop() -> None:
            """Background health check loop."""
            while not self._shutdown_event.is_set():
                try:
                    healthy = await self.health_check()

                    # If unhealthy and we're closed, consider opening
                    if not healthy and self.state == CircuitBreakerState.CLOSED:
                        with self._lock:
                            self._record_failure()

                    await asyncio.sleep(self.config.health_check_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Health check loop error: %s", e)
                    await asyncio.sleep(self.config.health_check_interval)

        self._health_check_task = asyncio.create_task(health_check_loop())
        logger.info("Started health monitoring for circuit breaker '%s'", self.name)

    async def stop_health_monitoring(self) -> None:
        """Stop background health check monitoring."""
        if self._health_check_task:
            self._shutdown_event.set()
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task
            self._health_check_task = None
            logger.info("Stopped health monitoring for circuit breaker '%s'", self.name)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_state = self._metrics.current_state
            self._metrics = CircuitBreakerMetrics()
            logger.info(
                "Circuit breaker '%s' reset from state '%s'",
                self.name,
                old_state.value,
            )

    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            self._transition_to_state(CircuitBreakerState.OPEN)
            # Set next retry time to prevent immediate recovery when forced open
            self._metrics.next_retry_time = utc_now() + timedelta(seconds=self.config.recovery_timeout)
            logger.warning("Circuit breaker '%s' forced to OPEN state", self.name)

    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        with self._lock:
            self._transition_to_state(CircuitBreakerState.CLOSED)
            logger.info("Circuit breaker '%s' forced to CLOSED state", self.name)

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for monitoring.

        Returns:
            Dictionary containing health and operational status
        """
        metrics = self.metrics

        return {
            "name": self.name,
            "healthy": metrics.current_state != CircuitBreakerState.OPEN,
            "state": metrics.current_state.value,
            "state_since": metrics.state_since.isoformat(),
            "metrics": {
                "total_requests": metrics.total_requests,
                "success_rate": metrics.get_success_rate(),
                "failure_rate": metrics.get_failure_rate(),
                "consecutive_failures": metrics.consecutive_failures,
                "consecutive_successes": metrics.consecutive_successes,
                "rejected_requests": metrics.rejected_requests,
            },
            "timing": {
                "last_failure": metrics.last_failure_time.isoformat() if metrics.last_failure_time else None,
                "last_success": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                "next_retry": metrics.next_retry_time.isoformat() if metrics.next_retry_time else None,
            },
            "health_checks": {
                "last_check": metrics.last_health_check.isoformat() if metrics.last_health_check else None,
                "total_checks": metrics.health_check_count,
                "failed_checks": metrics.health_check_failures,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "max_retries": self.config.max_retries,
            },
        }


def create_circuit_breaker_config_from_settings(settings: Any) -> CircuitBreakerConfig:
    """Create circuit breaker configuration from application settings.

    Args:
        settings: ApplicationSettings instance

    Returns:
        CircuitBreakerConfig with values from settings
    """
    return CircuitBreakerConfig(
        failure_threshold=settings.circuit_breaker_failure_threshold,
        success_threshold=settings.circuit_breaker_success_threshold,
        recovery_timeout=settings.circuit_breaker_recovery_timeout,
        max_retries=settings.circuit_breaker_max_retries,
        base_delay=settings.circuit_breaker_base_delay,
        max_delay=settings.circuit_breaker_max_delay,
        backoff_multiplier=settings.circuit_breaker_backoff_multiplier,
        jitter=settings.circuit_breaker_jitter_enabled,
        health_check_interval=settings.circuit_breaker_health_check_interval,
        health_check_timeout=settings.circuit_breaker_health_check_timeout,
        enable_metrics=settings.performance_monitoring_enabled,
        enable_tracing=settings.performance_monitoring_enabled,
    )


def create_openrouter_circuit_breaker(settings: Any) -> CircuitBreaker:
    """Create a circuit breaker specifically configured for OpenRouter integration.

    Args:
        settings: ApplicationSettings instance

    Returns:
        CircuitBreaker configured for OpenRouter API calls
    """
    config = create_circuit_breaker_config_from_settings(settings)

    async def openrouter_health_check() -> bool:
        """Health check function for OpenRouter API."""
        try:
            # Import here to avoid circular imports
            from src.mcp_integration.mcp_client import MCPConnectionState  # noqa: PLC0415
            from src.mcp_integration.openrouter_client import OpenRouterClient  # noqa: PLC0415

            # Create a minimal client for health check
            client = OpenRouterClient(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                timeout=5.0,  # Short timeout for health check
            )

            # Perform health check
            health_status = await client.health_check()
            # Use connection_state to determine health (based on MCPHealthStatus definition)
            return health_status.connection_state == MCPConnectionState.CONNECTED

        except Exception as e:
            logger.warning("OpenRouter health check failed: %s", e)
            return False

    return CircuitBreaker(
        name="openrouter",
        config=config,
        health_check_func=openrouter_health_check if settings.health_check_enabled else None,
    )


# Global circuit breaker instances
_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_breaker_lock = threading.Lock()


def get_circuit_breaker(name: str, settings: Any = None) -> CircuitBreaker | None:
    """Get or create a circuit breaker instance by name.

    Args:
        name: Circuit breaker identifier
        settings: ApplicationSettings instance (required for first-time creation)

    Returns:
        CircuitBreaker instance or None if not found and no settings provided
    """
    with _circuit_breaker_lock:
        if name in _circuit_breakers:
            return _circuit_breakers[name]

        if not settings:
            return None

        # Create circuit breaker based on name
        if name == "openrouter":
            circuit_breaker = create_openrouter_circuit_breaker(settings)
        else:
            # Default circuit breaker with settings configuration
            config = create_circuit_breaker_config_from_settings(settings)
            circuit_breaker = CircuitBreaker(name=name, config=config)

        _circuit_breakers[name] = circuit_breaker
        return circuit_breaker


def register_circuit_breaker(name: str, circuit_breaker: CircuitBreaker) -> None:
    """Register a custom circuit breaker instance.

    Args:
        name: Circuit breaker identifier
        circuit_breaker: CircuitBreaker instance to register
    """
    with _circuit_breaker_lock:
        _circuit_breakers[name] = circuit_breaker
        logger.info("Registered circuit breaker '%s'", name)


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all registered circuit breaker instances.

    Returns:
        Dictionary of all circuit breakers by name
    """
    with _circuit_breaker_lock:
        return _circuit_breakers.copy()


async def start_all_health_monitoring() -> None:
    """Start health monitoring for all registered circuit breakers."""
    circuit_breakers = get_all_circuit_breakers()

    for name, circuit_breaker in circuit_breakers.items():
        try:
            await circuit_breaker.start_health_monitoring()
            logger.info("Started health monitoring for circuit breaker '%s'", name)
        except Exception as e:
            logger.error("Failed to start health monitoring for '%s': %s", name, e)


async def stop_all_health_monitoring() -> None:
    """Stop health monitoring for all registered circuit breakers."""
    circuit_breakers = get_all_circuit_breakers()

    for name, circuit_breaker in circuit_breakers.items():
        try:
            await circuit_breaker.stop_health_monitoring()
            logger.info("Stopped health monitoring for circuit breaker '%s'", name)
        except Exception as e:
            logger.error("Failed to stop health monitoring for '%s': %s", name, e)


def reset_all_circuit_breakers() -> None:
    """Reset all registered circuit breakers to initial state."""
    circuit_breakers = get_all_circuit_breakers()

    for name, circuit_breaker in circuit_breakers.items():
        try:
            circuit_breaker.reset()
            logger.info("Reset circuit breaker '%s'", name)
        except Exception as e:
            logger.error("Failed to reset circuit breaker '%s': %s", name, e)


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerMetrics",
    "CircuitBreakerOpenError",
    "CircuitBreakerState",
    "create_circuit_breaker_config_from_settings",
    "create_openrouter_circuit_breaker",
    "get_all_circuit_breakers",
    "get_circuit_breaker",
    "register_circuit_breaker",
    "reset_all_circuit_breakers",
    "start_all_health_monitoring",
    "stop_all_health_monitoring",
]
