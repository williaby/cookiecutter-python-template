"""Common resilience patterns and interfaces for project-wide use.

This module provides reusable resilience patterns that can be used across
the entire PromptCraft project, including circuit breakers, retry policies,
and other error handling strategies.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ResilienceStrategy(Generic[T], ABC):
    """Abstract base class for all resilience strategies."""

    @abstractmethod
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with resilience strategy applied.

        Args:
            func: Async function to execute.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            Exception: Strategy-specific exceptions.
        """

    @abstractmethod
    def should_continue(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should continue after failure.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number (0-based).

        Returns:
            True if operation should continue, False otherwise.
        """

    @abstractmethod
    def get_health_status(self) -> dict[str, Any]:
        """Get current health status of the strategy.

        Returns:
            Dictionary containing health metrics.
        """


class ResilienceError(Exception):
    """Base exception for resilience-related errors."""


class CircuitBreakerOpenError(ResilienceError):
    """Raised when circuit breaker is in OPEN state."""


class RetryExhaustedError(ResilienceError):
    """Raised when all retry attempts are exhausted."""


class CircuitBreakerState(Enum):
    """States for circuit breaker pattern."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests due to failures
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker strategy."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""

    max_retries: int = 3  # Maximum retry attempts
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add jitter to delays
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,),
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base must be greater than 1")


class CompositeResilienceHandler(Generic[T]):
    """Compose multiple resilience strategies for flexible error handling."""

    def __init__(
        self,
        strategies: list[ResilienceStrategy[T]],
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize composite handler.

        Args:
            strategies: List of resilience strategies to apply.
            logger: Optional logger instance.
        """
        self.strategies = strategies
        self.logger = logger or logging.getLogger(__name__)

    async def execute_with_protection(
        self,
        func: Callable[..., Awaitable[T]],
        fallback_func: Callable[..., Awaitable[T]] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with all resilience strategies applied.

        Args:
            func: Primary function to execute.
            fallback_func: Optional fallback function.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            ResilienceError: If all strategies fail and no fallback.
        """
        try:
            # Apply strategies in sequence
            protected_func = func
            for strategy in self.strategies:
                protected_func = self._wrap_with_strategy(protected_func, strategy)

            return await protected_func(*args, **kwargs)

        except Exception as e:
            self.logger.error("All resilience strategies failed: %s", e)

            if fallback_func:
                try:
                    self.logger.info("Attempting fallback function")
                    return await fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error("Fallback function failed: %s", fallback_error)
                    raise ResilienceError(
                        f"Both primary and fallback functions failed: {e}",
                    ) from fallback_error

            raise ResilienceError(f"Function execution failed: {e}") from e

    def _wrap_with_strategy(
        self,
        func: Callable[..., Awaitable[T]],
        strategy: ResilienceStrategy[T],
    ) -> Callable[..., Awaitable[T]]:
        """Wrap function with a resilience strategy.

        Args:
            func: Function to wrap.
            strategy: Strategy to apply.

        Returns:
            Wrapped function.
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await strategy.execute(func, *args, **kwargs)

        return wrapper

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all strategies.

        Returns:
            Combined health status from all strategies.
        """
        status: dict[str, Any] = {"strategies": {}, "overall_healthy": True}
        strategies_dict: dict[str, Any] = {}

        for i, strategy in enumerate(self.strategies):
            strategy_status = strategy.get_health_status()
            strategy_name = strategy.__class__.__name__
            strategies_dict[f"{strategy_name}_{i}"] = strategy_status

            # Check if this strategy indicates unhealthy state
            if not strategy_status.get("healthy", True):
                status["overall_healthy"] = False

        status["strategies"] = strategies_dict
        return status


def resilience_decorator(
    strategies: list[ResilienceStrategy[T]],
    fallback_func: Callable[..., Awaitable[T]] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator to apply resilience strategies to a function.

    Args:
        strategies: List of resilience strategies.
        fallback_func: Optional fallback function.

    Returns:
        Decorated function with resilience applied.
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        handler = CompositeResilienceHandler(strategies)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await handler.execute_with_protection(
                func,
                fallback_func,
                *args,
                **kwargs,
            )

        return wrapper

    return decorator
