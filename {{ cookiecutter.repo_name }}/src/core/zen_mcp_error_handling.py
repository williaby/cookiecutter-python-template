"""Zen MCP error handling and resilience system for C.R.E.A.T.E. framework.

This module provides comprehensive error handling for Zen MCP integration,
using the modular resilience strategies for enhanced maintainability and reusability.
It implements circuit breaker patterns, retry strategies, and composite resilience
handling to ensure robust operation in distributed systems.

The module provides:
- Circuit breaker pattern implementation for fault tolerance
- Retry strategy with exponential backoff and secure jitter
- Composite resilience handler for multi-strategy coordination
- Mock client for testing error handling scenarios
- High-level integration layer with fallback mechanisms

Architecture:
    The resilience system follows the Strategy pattern with composable strategies.
    Each strategy can be used independently or combined through the composite handler.
    The system integrates with the secure random utilities for cryptographically
    secure jitter and backoff calculations.

Key Components:
    - CircuitBreakerStrategy: Implements circuit breaker fault tolerance
    - RetryStrategy: Provides retry logic with exponential backoff
    - CompositeResilienceHandler: Coordinates multiple resilience strategies
    - ZenMCPIntegration: High-level integration with fallback support
    - Factory functions: Pre-configured integration instances

Dependencies:
    - asyncio: For asynchronous operation support
    - time: For timing and delay calculations
    - src.utils.resilience: For base resilience strategy interfaces
    - src.utils.logging_mixin: For structured logging support
    - src.utils.secure_random: For cryptographically secure randomness

Called by:
    - src.agents.create_agent: For resilient agent execution
    - src.core.query_counselor: For robust query processing
    - src.main.py: For FastAPI error handling middleware
    - Agent implementations: For external service calls

Complexity: O(1) for circuit breaker operations, O(n) for retry attempts where n is max_retries
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from src.utils.logging_mixin import LoggerMixin
from src.utils.resilience import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    CompositeResilienceHandler,
    ResilienceError,
    ResilienceStrategy,
    RetryConfig,
    RetryExhaustedError,
)
from src.utils.secure_random import SecureRandom, secure_random


class ErrorCategory:
    """Error category constants for structured error handling."""

    NETWORK = "network"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class ErrorSeverity:
    """Error severity levels for structured error handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy:
    """Recovery strategy constants for error handling."""

    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    FAIL_FAST = "fail_fast"


class ErrorContext:
    """Error context information for structured error handling."""

    def __init__(
        self,
        error_type: str,
        severity: str = ErrorSeverity.MEDIUM,
        category: str = ErrorCategory.UNKNOWN,
        recovery_strategy: str = RecoveryStrategy.RETRY,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize error context.

        Args:
            error_type: Type of error
            severity: Error severity level
            category: Error category
            recovery_strategy: Recovery strategy to use
            metadata: Additional metadata
        """
        self.error_type = error_type
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            "error_type": self.error_type,
            "severity": self.severity,
            "category": self.category,
            "recovery_strategy": self.recovery_strategy,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class ZenMCPError(ResilienceError):
    """Base exception for Zen MCP errors."""


class ZenMCPConnectionError(ZenMCPError):
    """Exception raised for connection errors."""


class ZenMCPTimeoutError(ZenMCPError):
    """Exception raised for timeout errors."""


class ZenMCPRateLimitError(ZenMCPError):
    """Exception raised for rate limit errors."""


class CircuitBreakerStrategy(ResilienceStrategy[Any], LoggerMixin):
    """Circuit breaker resilience strategy for Zen MCP integration."""

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        logger_name: str = "zen_mcp.circuit_breaker",
    ) -> None:
        """Initialize circuit breaker strategy.

        Args:
            config: Circuit breaker configuration.
            logger_name: Custom logger name.
        """
        super().__init__(logger_name=logger_name)
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

    async def execute(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with circuit breaker protection.

        This method implements the core circuit breaker logic, managing state transitions
        and failure tracking to provide fault tolerance for external service calls.

        Args:
            func: Async function to execute.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open.
            ZenMCPError: If function execution fails.

        Time Complexity: O(1) - Constant time state checks and transitions
        Space Complexity: O(1) - Fixed memory for state tracking

        Called by:
            - CompositeResilienceHandler.execute_with_protection()
            - Direct usage in agent implementations
            - Test scenarios for circuit breaker validation

        Calls:
            - _should_attempt_reset(): State transition logic
            - _transition_to_half_open(): State management
            - _on_success(): Success handling
            - _on_failure(): Failure handling
        """
        self.log_method_entry("execute", func.__name__, *args, **kwargs)

        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.logger.warning("Circuit breaker is OPEN, blocking call")
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            self.log_method_exit("execute", "success")
            return result
        except Exception as e:
            self._on_failure()
            self.log_error_with_context(e, {"function": func.__name__}, "execute")
            raise ZenMCPError(f"Function call failed: {e}") from e

    def should_continue(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should continue after failure.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if circuit breaker allows continuation.
        """
        # Circuit breaker strategy is stateful - parameters not directly used
        # but kept for interface compliance
        _ = exception, attempt
        return self.state != CircuitBreakerState.OPEN

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status of circuit breaker.

        Returns:
            Health status information.
        """
        return {
            "healthy": self.state == CircuitBreakerState.CLOSED,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "time_to_reset": max(
                0,
                self.config.recovery_timeout - (time.time() - self.last_failure_time),
            ),
        }

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset.

        Returns:
            True if should attempt reset, False otherwise.
        """
        if self.state != CircuitBreakerState.OPEN:
            return False
        return (time.time() - self.last_failure_time) > self.config.recovery_timeout

    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state."""
        self.log_state_change(
            self.state.value,
            CircuitBreakerState.HALF_OPEN.value,
            "recovery_attempt",
        )
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0

    def _on_success(self) -> None:
        """Handle successful function execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.log_state_change(
                    self.state.value,
                    CircuitBreakerState.CLOSED.value,
                    "recovery_complete",
                )
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle function execution failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.log_state_change(
                self.state.value,
                CircuitBreakerState.OPEN.value,
                "recovery_failed",
            )
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.log_state_change(
                self.state.value,
                CircuitBreakerState.OPEN.value,
                f"threshold_exceeded_{self.failure_count}",
            )
            self.state = CircuitBreakerState.OPEN


class RetryStrategy(ResilienceStrategy[Any], LoggerMixin):
    """Retry resilience strategy with exponential backoff and secure jitter."""

    def __init__(
        self,
        config: RetryConfig | None = None,
        secure_rng: SecureRandom | None = None,
        logger_name: str = "zen_mcp.retry",
    ) -> None:
        """Initialize retry strategy.

        Args:
            config: Retry configuration.
            secure_rng: Secure random number generator.
            logger_name: Custom logger name.
        """
        super().__init__(logger_name=logger_name)
        self.config = config or RetryConfig()
        self.secure_rng = secure_rng or secure_random

    async def execute(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry policy.

        This method implements exponential backoff retry logic with secure jitter
        to provide resilience against transient failures in distributed systems.

        Args:
            func: Async function to execute.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            RetryExhaustedError: If all retries fail.

        Time Complexity: O(n) where n is max_retries (worst case)
        Space Complexity: O(1) - Fixed memory regardless of retry count

        Called by:
            - CompositeResilienceHandler.execute_with_protection()
            - Direct usage in agent implementations
            - High-level integration layers

        Calls:
            - _calculate_delay(): Exponential backoff calculation
            - asyncio.sleep(): Delay between retry attempts
            - secure_rng.exponential_backoff_jitter(): Secure delay calculation
        """
        self.log_method_entry("execute", func.__name__, *args, **kwargs)
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info("Retry succeeded on attempt %d", attempt + 1)
                self.log_method_exit("execute", "success")
                return result

            except Exception as e:
                last_exception = e

                # Check if this exception type is retryable
                if not any(isinstance(e, exc_type) for exc_type in self.config.retryable_exceptions):
                    self.logger.warning(
                        "Non-retryable exception: %s",
                        type(e).__name__,
                    )
                    raise

                if attempt == self.config.max_retries:
                    break

                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    "Retry attempt %d/%d failed, retrying in %.2f seconds: %s",
                    attempt + 1,
                    self.config.max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

        if last_exception is not None:
            self.log_error_with_context(
                last_exception,
                {"attempts": self.config.max_retries + 1},
                "execute",
            )
        raise RetryExhaustedError(
            f"All {self.config.max_retries + 1} retry attempts failed: {last_exception}",
        ) from last_exception

    def should_continue(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should continue after failure.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if should retry, False otherwise.
        """
        if attempt >= self.config.max_retries:
            return False

        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status of retry strategy.

        Returns:
            Health status information.
        """
        return {
            "healthy": True,
            "max_retries": self.config.max_retries,
            "base_delay": self.config.base_delay,
            "max_delay": self.config.max_delay,
            "jitter_enabled": self.config.jitter,
        }

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with secure jitter.

        Args:
            attempt: Retry attempt number (0-based).

        Returns:
            Delay in seconds.
        """
        # Use secure exponential backoff with jitter
        return self.secure_rng.exponential_backoff_jitter(
            self.config.base_delay,
            attempt,
            self.config.max_delay,
        )


class MockZenMCPClient(LoggerMixin):
    """Mock Zen MCP client for testing error handling with secure randomness."""

    def __init__(
        self,
        failure_rate: float = 0.0,
        secure_rng: SecureRandom | None = None,
        logger_name: str = "zen_mcp.mock_client",
    ) -> None:
        """Initialize mock client.

        Args:
            failure_rate: Probability of simulated failure (0.0-1.0).
            secure_rng: Secure random number generator.
            logger_name: Custom logger name.
        """
        super().__init__(logger_name=logger_name)
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be between 0.0 and 1.0")

        self.failure_rate = failure_rate
        self.call_count = 0
        self.secure_rng = secure_rng or secure_random

    async def process_prompt(self, prompt: str) -> dict[str, Any]:
        """Process prompt with optional failure simulation.

        Args:
            prompt: Input prompt to process.

        Returns:
            Mock response dictionary.

        Raises:
            ZenMCPConnectionError: If failure is simulated.
        """
        self.call_count += 1

        if self.secure_rng.random() < self.failure_rate:
            error_msg = f"Simulated failure on call {self.call_count}"
            self.logger.warning("Simulating failure: %s", error_msg)
            raise ZenMCPConnectionError(error_msg)

        response = {
            "enhanced_prompt": f"Enhanced: {prompt}",
            "metadata": {"call_count": self.call_count},
        }

        self.logger.debug("Mock processing successful: call %d", self.call_count)
        return response


class ZenMCPIntegration(LoggerMixin):
    """High-level integration layer for Zen MCP with modular resilience."""

    def __init__(
        self,
        client: Any | None = None,
        resilience_handler: CompositeResilienceHandler | None = None,
        logger_name: str = "zen_mcp.integration",
    ) -> None:
        """Initialize Zen MCP integration.

        Args:
            client: Zen MCP client instance.
            resilience_handler: Composite resilience handler.
            logger_name: Custom logger name.
        """
        super().__init__(logger_name=logger_name)
        self.client = client or MockZenMCPClient()

        # Create default resilience handler if none provided
        if resilience_handler is None:
            circuit_breaker = CircuitBreakerStrategy()
            retry_strategy = RetryStrategy()
            self.resilience_handler = CompositeResilienceHandler(
                [circuit_breaker, retry_strategy],
                self.logger,
            )
        else:
            self.resilience_handler = resilience_handler

    async def enhance_prompt(self, prompt: str) -> dict[str, Any]:
        """Enhance prompt using Zen MCP with comprehensive error handling.

        Args:
            prompt: Input prompt to enhance.

        Returns:
            Enhanced prompt response.

        Raises:
            ZenMCPError: If enhancement fails completely.
        """
        self.log_method_entry("enhance_prompt", prompt)

        async def primary_func() -> dict[str, Any]:
            return await self.client.process_prompt(prompt)

        async def fallback_func() -> dict[str, Any]:
            self.logger.info("Using fallback prompt enhancement for: %s", prompt[:50])
            return {
                "enhanced_prompt": f"[FALLBACK] {prompt}",
                "metadata": {"fallback": True},
            }

        try:
            result: dict[str, Any] = await self.resilience_handler.execute_with_protection(
                primary_func,
                fallback_func,
            )
            self.log_method_exit("enhance_prompt", "success")
            return result

        except Exception as e:
            self.log_error_with_context(e, {"prompt_length": len(prompt)}, "enhance_prompt")
            raise ZenMCPError(f"Failed to enhance prompt: {e}") from e

    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.

        Returns:
            Current circuit breaker state.
        """
        for strategy in self.resilience_handler.strategies:
            if isinstance(strategy, CircuitBreakerStrategy):
                return strategy.state
        return CircuitBreakerState.CLOSED

    def get_failure_count(self) -> int:
        """Get current failure count from circuit breaker.

        Returns:
            Number of failures.
        """
        for strategy in self.resilience_handler.strategies:
            if isinstance(strategy, CircuitBreakerStrategy):
                return strategy.failure_count
        return 0

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status.

        Returns:
            Health status from all resilience strategies.
        """
        return self.resilience_handler.get_health_status()


# Factory functions for common configurations
def create_default_zen_mcp_integration(
    client: Any | None = None,
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    retry_config: RetryConfig | None = None,
) -> ZenMCPIntegration:
    """Create Zen MCP integration with default resilience configuration.

    Args:
        client: Optional Zen MCP client.
        circuit_breaker_config: Optional circuit breaker configuration.
        retry_config: Optional retry configuration.

    Returns:
        Configured ZenMCPIntegration instance.
    """
    circuit_breaker = CircuitBreakerStrategy(circuit_breaker_config)
    retry_strategy = RetryStrategy(retry_config)

    resilience_handler = CompositeResilienceHandler([circuit_breaker, retry_strategy])

    return ZenMCPIntegration(client, resilience_handler)


def create_high_availability_zen_mcp_integration(
    client: Any | None = None,
) -> ZenMCPIntegration:
    """Create Zen MCP integration optimized for high availability.

    Args:
        client: Optional Zen MCP client.

    Returns:
        High-availability configured ZenMCPIntegration instance.
    """
    # More aggressive circuit breaker for faster failure detection
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30,
        success_threshold=2,
    )

    # More retries with longer delays for resilience
    retry_config = RetryConfig(
        max_retries=5,
        base_delay=2.0,
        max_delay=120.0,
        exponential_base=1.5,
    )

    return create_default_zen_mcp_integration(client, circuit_breaker_config, retry_config)


def create_fast_fail_zen_mcp_integration(
    client: Any | None = None,
) -> ZenMCPIntegration:
    """Create Zen MCP integration optimized for fast failure detection.

    Args:
        client: Optional Zen MCP client.

    Returns:
        Fast-fail configured ZenMCPIntegration instance.
    """
    # Quick failure detection
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=15,
        success_threshold=1,
    )

    # Minimal retries for fast response
    retry_config = RetryConfig(
        max_retries=1,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=2.0,
    )

    return create_default_zen_mcp_integration(client, circuit_breaker_config, retry_config)


# Aliases for backward compatibility with tests
ZenMCPCircuitBreaker = CircuitBreakerStrategy
ZenMCPRetryPolicy = RetryStrategy


class ZenMCPErrorHandler:
    """High-level error handler for Zen MCP operations."""

    def __init__(
        self,
        circuit_breaker: CircuitBreakerStrategy | None = None,
        retry_policy: RetryStrategy | None = None,
    ) -> None:
        """Initialize error handler.

        Args:
            circuit_breaker: Circuit breaker strategy
            retry_policy: Retry policy strategy
        """
        self.circuit_breaker = circuit_breaker or CircuitBreakerStrategy()
        self.retry_policy = retry_policy or RetryStrategy()
        self.logger = logging.getLogger(__name__)

    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """Handle an error with the appropriate strategy.

        Args:
            error: The error that occurred
            context: Error context information

        Returns:
            True if error was handled successfully, False otherwise
        """
        try:
            if context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self.circuit_breaker.should_continue(error, 1)
            if context.recovery_strategy == RecoveryStrategy.RETRY:
                return self.retry_policy.should_continue(error, 1)
            # Default handling
            self.logger.warning("Unhandled error: %s", error)
            return False
        except Exception as e:
            self.logger.error("Error handling failed: %s", e)
            return False

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of error handler components."""
        return {
            "circuit_breaker": self.circuit_breaker.get_health_status(),
            "retry_policy": self.retry_policy.get_health_status(),
        }


# Factory functions for test compatibility

def create_error_handler(config: dict[str, Any] | None = None) -> ZenMCPIntegration:
    """
    Create a configured error handler instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ZenMCPIntegration: Configured error handler
    """
    if config is None:
        config = {}
        
    return ZenMCPIntegration(
        circuit_breaker_config=config.get("circuit_breaker", {}),
        retry_config=config.get("retry", {})
    )


def log_error_with_context(error: Exception, context: dict[str, Any]) -> None:
    """
    Log error with additional context information.
    
    Args:
        error: Exception to log
        context: Additional context for the error
    """
    import logging
    
    logger = logging.getLogger("zen_mcp_error_handling")
    context_str = " ".join(f"{k}={v}" for k, v in context.items())
    logger.error("Zen MCP Error: %s | Context: %s", str(error), context_str)
