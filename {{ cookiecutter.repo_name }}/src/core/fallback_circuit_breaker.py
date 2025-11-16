"""
Advanced Circuit Breaker Implementation for Conservative Fallback Chain

This module provides a specialized circuit breaker implementation designed
specifically for the conservative fallback chain system. It includes advanced
features like graduated state transitions, adaptive thresholds, and integration
with the fallback system's error classification.

Key Features:
- Graduated state transitions (CLOSED -> HALF_OPEN -> OPEN)
- Adaptive failure thresholds based on error types
- Integration with error classification system
- Health monitoring and metrics collection
- Automatic recovery with configurable strategies
- Cascade failure prevention

Architecture:
    The circuit breaker follows the classic pattern with enhancements:
    - CLOSED: Normal operation, monitoring failures
    - HALF_OPEN: Testing recovery, limited requests allowed
    - OPEN: Blocking requests, waiting for recovery timeout

Dependencies:
    - asyncio: For asynchronous operation support
    - time: For timing and state management
    - src.core.conservative_fallback_chain: For error context integration
    - src.utils.resilience: For base circuit breaker interfaces
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.logging_mixin import LoggerMixin

logger = logging.getLogger(__name__)


class AdvancedCircuitBreakerState(Enum):
    """Enhanced circuit breaker states with graduated transitions"""

    CLOSED = "closed"  # Normal operation
    HALF_OPEN = "half_open"  # Testing recovery
    OPEN = "open"  # Blocking requests
    FORCED_OPEN = "forced_open"  # Manually opened (admin)
    DEGRADED = "degraded"  # Partial functionality


class FailurePattern(Enum):
    """Types of failure patterns for adaptive behavior"""

    INTERMITTENT = "intermittent"  # Sporadic failures
    SUSTAINED = "sustained"  # Continuous failures
    CASCADING = "cascading"  # Failures spreading across system
    TIMEOUT_HEAVY = "timeout_heavy"  # Predominantly timeout failures
    OVERLOAD = "overload"  # System overload pattern


@dataclass
class CircuitBreakerMetrics:
    """Comprehensive metrics for circuit breaker monitoring"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    state_transitions: int = 0
    time_in_states: dict[AdvancedCircuitBreakerState, float] = field(default_factory=dict)
    failure_rate: float = 0.0
    avg_response_time: float = 0.0
    last_state_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "timeouts": self.timeouts,
            "state_transitions": self.state_transitions,
            "failure_rate": self.failure_rate,
            "avg_response_time": self.avg_response_time,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "time_in_states": {state.value: duration for state, duration in self.time_in_states.items()},
            "last_state_change": self.last_state_change,
        }


@dataclass
class AdaptiveThreshold:
    """Adaptive threshold configuration based on failure patterns"""

    base_failure_threshold: int = 5
    base_success_threshold: int = 3
    base_timeout: float = 60.0

    # Pattern-specific modifiers
    pattern_modifiers: dict[FailurePattern, dict[str, float]] = field(
        default_factory=lambda: {
            FailurePattern.INTERMITTENT: {"failure_mult": 1.5, "timeout_mult": 0.8},
            FailurePattern.SUSTAINED: {"failure_mult": 0.7, "timeout_mult": 1.2},
            FailurePattern.CASCADING: {"failure_mult": 0.5, "timeout_mult": 1.5},
            FailurePattern.TIMEOUT_HEAVY: {"failure_mult": 0.8, "timeout_mult": 2.0},
            FailurePattern.OVERLOAD: {"failure_mult": 0.6, "timeout_mult": 1.8},
        },
    )

    def get_adjusted_thresholds(self, pattern: FailurePattern) -> dict[str, int | float]:
        """Get pattern-adjusted thresholds"""
        modifiers = self.pattern_modifiers.get(pattern, {"failure_mult": 1.0, "timeout_mult": 1.0})

        return {
            "failure_threshold": int(self.base_failure_threshold * modifiers["failure_mult"]),
            "success_threshold": self.base_success_threshold,
            "timeout": self.base_timeout * modifiers["timeout_mult"],
        }


class FailureAnalyzer:
    """Analyzes failure patterns for adaptive circuit breaker behavior"""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.failure_history = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)

    def record_request(self, success: bool, response_time: float, error_type: str | None = None) -> None:
        """Record a request result for pattern analysis"""
        self.failure_history.append(
            {
                "success": success,
                "timestamp": time.time(),
                "response_time": response_time,
                "error_type": error_type,
            },
        )
        self.response_times.append(response_time)

    def detect_failure_pattern(self) -> FailurePattern:
        """Detect the current failure pattern"""
        if len(self.failure_history) < 10:
            return FailurePattern.INTERMITTENT

        recent_failures = [r for r in list(self.failure_history)[-20:] if not r["success"]]

        if not recent_failures:
            return FailurePattern.INTERMITTENT

        # Analyze failure characteristics
        total_recent = len(list(self.failure_history)[-20:])
        failure_rate = len(recent_failures) / total_recent

        # Check for timeout dominance
        timeout_failures = sum(1 for f in recent_failures if f.get("error_type") == "timeout")
        timeout_ratio = timeout_failures / len(recent_failures) if recent_failures else 0

        # Check for sustained failures (consecutive)
        consecutive_failures = 0
        for record in reversed(list(self.failure_history)[-10:]):
            if not record["success"]:
                consecutive_failures += 1
            else:
                break

        # Detect patterns
        if timeout_ratio > 0.7:
            return FailurePattern.TIMEOUT_HEAVY
        if consecutive_failures >= 8:
            return FailurePattern.SUSTAINED
        if failure_rate > 0.8:
            return FailurePattern.CASCADING
        if self._detect_overload_pattern():
            return FailurePattern.OVERLOAD
        return FailurePattern.INTERMITTENT

    def _detect_overload_pattern(self) -> bool:
        """Detect system overload pattern"""
        if len(self.response_times) < 10:
            return False

        recent_times = list(self.response_times)[-10:]
        avg_time = sum(recent_times) / len(recent_times)

        # Consider overload if average response time is significantly elevated
        return avg_time > 2.0  # 2+ seconds average response time

    def get_pattern_confidence(self) -> float:
        """Get confidence in current pattern detection (0.0 to 1.0)"""
        if len(self.failure_history) < 5:
            return 0.2
        if len(self.failure_history) < 15:
            return 0.6
        return 0.9


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies"""

    @abstractmethod
    async def attempt_recovery(self, circuit_breaker: "FallbackCircuitBreaker") -> bool:
        """Attempt recovery and return True if successful"""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get name of recovery strategy"""


class GradualRecoveryStrategy(RecoveryStrategy):
    """Gradual recovery strategy with limited test requests"""

    def __init__(self, test_request_limit: int = 3) -> None:
        self.test_request_limit = test_request_limit
        self.test_requests_made = 0

    async def attempt_recovery(self, circuit_breaker: "FallbackCircuitBreaker") -> bool:
        """Attempt gradual recovery"""
        if self.test_requests_made >= self.test_request_limit:
            return False

        self.test_requests_made += 1
        # Recovery logic would be implemented here
        # For now, simulate recovery attempt
        await asyncio.sleep(0.1)
        return True

    def get_strategy_name(self) -> str:
        return "gradual_recovery"

    def reset(self) -> None:
        """Reset test request counter"""
        self.test_requests_made = 0


class AggressiveRecoveryStrategy(RecoveryStrategy):
    """Aggressive recovery strategy for critical systems"""

    async def attempt_recovery(self, circuit_breaker: "FallbackCircuitBreaker") -> bool:
        """Attempt aggressive recovery"""
        # More aggressive recovery logic
        await asyncio.sleep(0.05)
        return True

    def get_strategy_name(self) -> str:
        return "aggressive_recovery"


class FallbackCircuitBreaker(LoggerMixin):
    """
    Advanced circuit breaker implementation for conservative fallback chain

    Features:
    - Adaptive thresholds based on failure patterns
    - Multiple recovery strategies
    - Comprehensive metrics and monitoring
    - Integration with error classification
    - Cascade failure prevention
    """

    def __init__(
        self,
        adaptive_threshold: AdaptiveThreshold | None = None,
        recovery_strategy: RecoveryStrategy | None = None,
        logger_name: str = "fallback_circuit_breaker",
    ) -> None:
        super().__init__(logger_name=logger_name)

        self.adaptive_threshold = adaptive_threshold or AdaptiveThreshold()
        self.recovery_strategy = recovery_strategy or GradualRecoveryStrategy()

        # State management
        self.state = AdvancedCircuitBreakerState.CLOSED
        self.state_start_time = time.time()

        # Failure tracking
        self.failure_analyzer = FailureAnalyzer()
        self.current_pattern = FailurePattern.INTERMITTENT
        self.pattern_confidence = 0.0

        # Counters
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.requests_in_half_open = 0

        # Timing
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.last_state_change = time.time()

        # Metrics
        self.metrics = CircuitBreakerMetrics()

        # Configuration
        self.max_requests_in_half_open = 5
        self.forced_open_timeout = 300.0  # 5 minutes for forced open

        # Health monitoring
        self.health_check_interval = 30.0  # 30 seconds
        self.last_health_check = time.time()

    async def execute(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        request_start = time.time()

        # Check if request should be allowed
        if not await self._should_allow_request():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN - request blocked")

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Record success
            response_time = time.time() - request_start
            await self._on_success(response_time)

            return result

        except Exception as e:
            # Record failure
            response_time = time.time() - request_start
            error_type = self._classify_error(e)
            await self._on_failure(response_time, error_type)

            raise

    async def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on current state"""
        await self._update_state()

        if self.state == AdvancedCircuitBreakerState.CLOSED:
            return True
        if self.state == AdvancedCircuitBreakerState.HALF_OPEN:
            return self.requests_in_half_open < self.max_requests_in_half_open
        if self.state == AdvancedCircuitBreakerState.DEGRADED:
            # Allow some requests in degraded mode
            return time.time() % 2 < 1  # Allow 50% of requests
        return False  # OPEN or FORCED_OPEN

    async def _on_success(self, response_time: float) -> None:
        """Handle successful request"""
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.last_success_time = time.time()

        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self._update_metrics(response_time)

        # Record for pattern analysis
        self.failure_analyzer.record_request(True, response_time)

        # Handle state transitions
        if self.state == AdvancedCircuitBreakerState.HALF_OPEN:
            self.requests_in_half_open += 1

            # Check if we should transition to CLOSED
            current_thresholds = self._get_current_thresholds()
            if self.consecutive_successes >= current_thresholds["success_threshold"]:
                await self._transition_to_state(AdvancedCircuitBreakerState.CLOSED, "recovery_successful")

        elif self.state == AdvancedCircuitBreakerState.DEGRADED:
            # Check if we should transition to CLOSED
            if self.consecutive_successes >= 5:  # Recover from degraded after 5 successes
                await self._transition_to_state(AdvancedCircuitBreakerState.CLOSED, "degraded_recovery")

        self.log_method_exit("_on_success", f"consecutive_successes={self.consecutive_successes}")

    async def _on_failure(self, response_time: float, error_type: str) -> None:
        """Handle failed request"""
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        if error_type == "timeout":
            self.metrics.timeouts += 1
        self._update_metrics(response_time)

        # Record for pattern analysis
        self.failure_analyzer.record_request(False, response_time, error_type)

        # Update failure pattern
        self.current_pattern = self.failure_analyzer.detect_failure_pattern()
        self.pattern_confidence = self.failure_analyzer.get_pattern_confidence()

        # Handle state transitions
        current_thresholds = self._get_current_thresholds()

        if self.state == AdvancedCircuitBreakerState.CLOSED:
            if self.consecutive_failures >= current_thresholds["failure_threshold"]:
                # Decide between OPEN and DEGRADED based on pattern
                if self.current_pattern in [FailurePattern.CASCADING, FailurePattern.SUSTAINED]:
                    await self._transition_to_state(
                        AdvancedCircuitBreakerState.OPEN,
                        f"failure_threshold_exceeded_{self.current_pattern.value}",
                    )
                else:
                    await self._transition_to_state(
                        AdvancedCircuitBreakerState.DEGRADED,
                        f"degraded_mode_{self.current_pattern.value}",
                    )

        elif self.state == AdvancedCircuitBreakerState.HALF_OPEN:
            # Any failure in half-open transitions back to OPEN
            await self._transition_to_state(AdvancedCircuitBreakerState.OPEN, "half_open_failure")

        elif self.state == AdvancedCircuitBreakerState.DEGRADED:
            # Too many failures in degraded mode -> OPEN
            if self.consecutive_failures >= 3:
                await self._transition_to_state(AdvancedCircuitBreakerState.OPEN, "degraded_failure_threshold")

        self.log_error_with_context(
            Exception(f"Circuit breaker failure: {error_type}"),
            {"consecutive_failures": self.consecutive_failures, "pattern": self.current_pattern.value},
            "_on_failure",
        )

    async def _update_state(self) -> None:
        """Update circuit breaker state based on timeouts and conditions"""
        current_time = time.time()
        time_in_state = current_time - self.state_start_time

        # Update time in state metrics
        if self.state not in self.metrics.time_in_states:
            self.metrics.time_in_states[self.state] = 0
        self.metrics.time_in_states[self.state] += current_time - self.last_health_check

        if self.state == AdvancedCircuitBreakerState.OPEN:
            current_thresholds = self._get_current_thresholds()
            if time_in_state >= current_thresholds["timeout"]:
                await self._transition_to_state(AdvancedCircuitBreakerState.HALF_OPEN, "timeout_recovery_attempt")

        elif self.state == AdvancedCircuitBreakerState.FORCED_OPEN:
            if time_in_state >= self.forced_open_timeout:
                await self._transition_to_state(AdvancedCircuitBreakerState.HALF_OPEN, "forced_open_timeout")

        self.last_health_check = current_time

    async def _transition_to_state(self, new_state: AdvancedCircuitBreakerState, reason: str) -> None:
        """Transition to a new state with proper logging and metrics"""
        old_state = self.state
        self.state = new_state
        self.state_start_time = time.time()
        self.last_state_change = time.time()
        self.metrics.state_transitions += 1
        self.metrics.last_state_change = time.time()

        # Reset counters based on new state
        if new_state == AdvancedCircuitBreakerState.HALF_OPEN:
            self.requests_in_half_open = 0
            if isinstance(self.recovery_strategy, GradualRecoveryStrategy):
                self.recovery_strategy.reset()

        # Log state transition
        self.log_state_change(old_state.value, new_state.value, reason)

        # Attempt recovery if transitioning to HALF_OPEN
        if new_state == AdvancedCircuitBreakerState.HALF_OPEN:
            try:
                recovery_successful = await self.recovery_strategy.attempt_recovery(self)
                if not recovery_successful:
                    self.logger.warning("Recovery strategy failed, remaining in HALF_OPEN")
            except Exception as e:
                self.logger.error(f"Recovery strategy error: {e}")

    def _get_current_thresholds(self) -> dict[str, int | float]:
        """Get current thresholds based on detected failure pattern"""
        return self.adaptive_threshold.get_adjusted_thresholds(self.current_pattern)

    def _classify_error(self, exception: Exception) -> str:
        """Classify error type for pattern analysis"""
        error_str = str(exception).lower()
        error_type = type(exception).__name__.lower()

        if "timeout" in error_str or "timeout" in error_type:
            return "timeout"
        if "connection" in error_str or "network" in error_str:
            return "network"
        if "memory" in error_str or "oom" in error_str:
            return "memory"
        if "overload" in error_str or "throttle" in error_str:
            return "overload"
        return "unknown"

    def _update_metrics(self, response_time: float) -> None:
        """Update circuit breaker metrics"""
        # Update failure rate
        if self.metrics.total_requests > 0:
            self.metrics.failure_rate = self.metrics.failed_requests / self.metrics.total_requests

        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if self.metrics.avg_response_time == 0:
            self.metrics.avg_response_time = response_time
        else:
            self.metrics.avg_response_time = alpha * response_time + (1 - alpha) * self.metrics.avg_response_time

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status"""
        current_thresholds = self._get_current_thresholds()

        return {
            "healthy": self.state in [AdvancedCircuitBreakerState.CLOSED, AdvancedCircuitBreakerState.DEGRADED],
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "failure_pattern": self.current_pattern.value,
            "pattern_confidence": self.pattern_confidence,
            "current_thresholds": current_thresholds,
            "time_in_current_state": time.time() - self.state_start_time,
            "requests_in_half_open": self.requests_in_half_open,
            "recovery_strategy": self.recovery_strategy.get_strategy_name(),
            "metrics": self.metrics.to_dict(),
        }

    def force_open(self, reason: str = "manual_intervention") -> None:
        """Manually force circuit breaker open"""
        self.logger.warning(f"Circuit breaker manually forced OPEN: {reason}")
        # For synchronous force operations, update state directly
        old_state = self.state
        self.state = AdvancedCircuitBreakerState.FORCED_OPEN
        self.state_start_time = time.time()
        self.last_state_change = time.time()
        self.logger.info(f"STATE_CHANGE: {old_state.value} -> forced_open (reason: {reason})")

    def force_close(self, reason: str = "manual_intervention") -> None:
        """Manually force circuit breaker closed"""
        self.logger.info(f"Circuit breaker manually forced CLOSED: {reason}")
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        # For synchronous force operations, update state directly
        old_state = self.state
        self.state = AdvancedCircuitBreakerState.CLOSED
        self.state_start_time = time.time()
        self.last_state_change = time.time()
        self.logger.info(f"STATE_CHANGE: {old_state.value} -> closed (reason: {reason})")

    def reset_metrics(self) -> None:
        """Reset all metrics (for testing/admin)"""
        self.metrics = CircuitBreakerMetrics()
        self.failure_analyzer = FailureAnalyzer()
        self.logger.info("Circuit breaker metrics reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""


# Factory functions for common configurations
def create_conservative_circuit_breaker() -> FallbackCircuitBreaker:
    """Create circuit breaker optimized for conservative fallback"""
    adaptive_threshold = AdaptiveThreshold(
        base_failure_threshold=3,  # Conservative - fail fast
        base_success_threshold=5,  # Conservative - require more successes
        base_timeout=30.0,  # Shorter timeout for faster recovery
    )

    recovery_strategy = GradualRecoveryStrategy(test_request_limit=3)

    return FallbackCircuitBreaker(
        adaptive_threshold=adaptive_threshold,
        recovery_strategy=recovery_strategy,
    )


def create_high_availability_circuit_breaker() -> FallbackCircuitBreaker:
    """Create circuit breaker optimized for high availability"""
    adaptive_threshold = AdaptiveThreshold(
        base_failure_threshold=5,  # More tolerant
        base_success_threshold=3,  # Faster recovery
        base_timeout=60.0,  # Longer timeout for stability
    )

    recovery_strategy = AggressiveRecoveryStrategy()

    return FallbackCircuitBreaker(
        adaptive_threshold=adaptive_threshold,
        recovery_strategy=recovery_strategy,
    )


def create_performance_circuit_breaker() -> FallbackCircuitBreaker:
    """Create circuit breaker optimized for performance"""
    adaptive_threshold = AdaptiveThreshold(
        base_failure_threshold=2,  # Very sensitive
        base_success_threshold=2,  # Quick recovery
        base_timeout=15.0,  # Fast timeout
    )

    recovery_strategy = AggressiveRecoveryStrategy()

    return FallbackCircuitBreaker(
        adaptive_threshold=adaptive_threshold,
        recovery_strategy=recovery_strategy,
    )
