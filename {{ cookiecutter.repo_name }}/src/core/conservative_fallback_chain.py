"""
Conservative Fallback Mechanism Chain for Dynamic Function Loading

This module implements a robust, production-ready fallback mechanism chain that
ensures users never lose functionality when the dynamic function loading system
encounters issues or edge cases. It follows Opus 4.1's recommendation to favor
over-inclusion over functionality loss.

Architecture:
    The fallback system implements a five-tier progressive degradation strategy:

    Level 1: High-Confidence Detection (≥70% confidence)
    Level 2: Medium-Confidence Detection (30-69% confidence)
    Level 3: Low-Confidence/Ambiguous Tasks (<30% confidence)
    Level 4: Detection Failure (complete detection failure)
    Level 5: System Emergency (function loading system unavailable)

Key Components:
    - FallbackController: Central orchestration of fallback logic
    - ErrorClassifier: Categorizes and prioritizes different failure types
    - RecoveryManager: Automated recovery and retry logic
    - CircuitBreaker: Prevents cascade failures
    - LearningCollector: Feeds failure data back to improve detection
    - PerformanceMonitor: Comprehensive observability for fallback events

Dependencies:
    - asyncio: For asynchronous operation support
    - time: For timing and performance tracking
    - src.core.task_detection: For integration with detection system
    - src.utils.resilience: For circuit breaker and retry strategies
    - src.utils.logging_mixin: For structured logging
"""

import asyncio
import logging
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any

from src.core.task_detection import DetectionResult, TaskDetectionSystem
from src.core.task_detection_config import DetectionMode, TaskDetectionConfig
from src.utils.logging_mixin import LoggerMixin

logger = logging.getLogger(__name__)


class FallbackLevel(Enum):
    """Fallback level constants for progressive degradation"""
    HIGH_CONFIDENCE = "high_confidence"      # ≥70% confidence
    MEDIUM_CONFIDENCE = "medium_confidence"  # 30-69% confidence
    LOW_CONFIDENCE = "low_confidence"        # <30% confidence
    DETECTION_FAILURE = "detection_failure"  # Complete detection failure
    SYSTEM_EMERGENCY = "system_emergency"    # Function loading unavailable


class ErrorType(Enum):
    """Error types for classification and handling"""
    TIMEOUT = "timeout"
    NETWORK_FAILURE = "network_failure"
    MEMORY_PRESSURE = "memory_pressure"
    VERSION_MISMATCH = "version_mismatch"
    DETECTION_FAILURE = "detection_failure"
    SYSTEM_OVERLOAD = "system_overload"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels for prioritization"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FallbackMetrics:
    """Metrics for fallback system performance tracking"""
    activation_count: int = 0
    total_recovery_time: float = 0.0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    level_activations: dict[FallbackLevel, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: dict[ErrorType, int] = field(default_factory=lambda: defaultdict(int))
    performance_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            "activation_count": self.activation_count,
            "total_recovery_time": self.total_recovery_time,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "level_activations": {level.value: count for level, count in self.level_activations.items()},
            "error_counts": {error.value: count for error, count in self.error_counts.items()},
            "performance_metrics": self.performance_metrics,
            "success_rate": self.successful_recoveries / max(1, self.activation_count),
            "avg_recovery_time": self.total_recovery_time / max(1, self.successful_recoveries),
        }


@dataclass
class ErrorContext:
    """Context information for error classification and handling"""
    error_type: ErrorType
    severity: ErrorSeverity
    timestamp: float
    query: str
    context: dict[str, Any]
    stack_trace: str | None = None
    recovery_attempt: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackDecision:
    """Decision result from fallback controller"""
    level: FallbackLevel
    categories_to_load: dict[str, bool]
    confidence_threshold: float
    expected_function_count: int
    rationale: str
    performance_impact: str
    recovery_strategy: str


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies"""

    def __init__(self) -> None:
        self.error_patterns = {
            # Timeout-related patterns
            ErrorType.TIMEOUT: [
                "timeout", "timed out", "deadline", "expired",
                "asyncio.TimeoutError", "concurrent.futures.TimeoutError",
            ],

            # Network failure patterns
            ErrorType.NETWORK_FAILURE: [
                "connection", "network", "dns", "socket", "http",
                "ConnectionError", "NetworkError", "aiohttp",
            ],

            # Memory pressure patterns
            ErrorType.MEMORY_PRESSURE: [
                "memory", "oom", "allocation", "heap", "garbage",
                "MemoryError", "OutOfMemoryError",
            ],

            # Version mismatch patterns
            ErrorType.VERSION_MISMATCH: [
                "version", "compatibility", "deprecated", "schema",
                "VersionError", "CompatibilityError",
            ],

            # Detection failure patterns
            ErrorType.DETECTION_FAILURE: [
                "detection", "classification", "analysis", "parsing",
                "DetectionError", "ClassificationError",
            ],

            # System overload patterns
            ErrorType.SYSTEM_OVERLOAD: [
                "overload", "throttle", "rate limit", "capacity",
                "OverloadError", "ThrottleError",
            ],
        }

        self.severity_indicators = {
            ErrorSeverity.CRITICAL: [
                "critical", "fatal", "emergency", "panic", "abort",
            ],
            ErrorSeverity.HIGH: [
                "error", "exception", "failed", "failure", "broken",
            ],
            ErrorSeverity.MEDIUM: [
                "warning", "warn", "degraded", "slow", "retry",
            ],
            ErrorSeverity.LOW: [
                "info", "notice", "debug", "trace", "minor",
            ],
        }

    def _get_severity_for_error_type(self, error_type: ErrorType) -> ErrorSeverity | None:
        """Get the appropriate severity level for a specific error type"""
        error_type_severity_map = {
            # Critical system-level errors
            ErrorType.SYSTEM_OVERLOAD: ErrorSeverity.CRITICAL,
            
            # High severity errors - resource exhaustion and critical failures
            ErrorType.MEMORY_PRESSURE: ErrorSeverity.HIGH,
            ErrorType.CONFIGURATION_ERROR: ErrorSeverity.HIGH,
            
            # Medium severity errors - operational issues
            ErrorType.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorType.NETWORK_FAILURE: ErrorSeverity.MEDIUM,
            ErrorType.VERSION_MISMATCH: ErrorSeverity.MEDIUM,
            
            # Low severity errors - functional issues
            ErrorType.DETECTION_FAILURE: ErrorSeverity.LOW,
            
            # Unknown errors default to None (use string matching)
            ErrorType.UNKNOWN: None,
        }
        
        return error_type_severity_map.get(error_type)

    def classify_error(self, exception: Exception, context: dict[str, Any]) -> ErrorContext:
        """Classify an error and create context for handling"""
        error_str = str(exception).lower()
        error_type_str = type(exception).__name__.lower()
        stack_trace = traceback.format_exc()

        # Classify error type
        error_type = ErrorType.UNKNOWN
        for etype, patterns in self.error_patterns.items():
            if any(pattern in error_str or pattern in error_type_str for pattern in patterns):
                error_type = etype
                break

        # Classify severity based on error type first, then fallback to string matching
        severity = self._get_severity_for_error_type(error_type)
        
        # If no type-specific severity, use string matching
        if severity is None:
            severity = ErrorSeverity.MEDIUM  # Default
            for sev, indicators in self.severity_indicators.items():
                if any(indicator in error_str for indicator in indicators):
                    severity = sev
                    break

        # Extract query from context
        query = context.get("query", "")

        return ErrorContext(
            error_type=error_type,
            severity=severity,
            timestamp=time.time(),
            query=query,
            context=context,
            stack_trace=stack_trace,
            metadata={
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
            },
        )

    def should_trigger_circuit_breaker(self, error_context: ErrorContext) -> bool:
        """Determine if error should trigger circuit breaker"""
        # Critical errors always trigger circuit breaker
        if error_context.severity == ErrorSeverity.CRITICAL:
            return True

        # Network failures and timeouts trigger circuit breaker
        if error_context.error_type in [ErrorType.NETWORK_FAILURE, ErrorType.TIMEOUT]:
            return True

        # System overload triggers circuit breaker
        return error_context.error_type == ErrorType.SYSTEM_OVERLOAD

    def get_recommended_recovery_strategy(self, error_context: ErrorContext) -> str:
        """Get recommended recovery strategy for error"""
        if error_context.error_type == ErrorType.TIMEOUT:
            return "retry_with_backoff"
        if error_context.error_type == ErrorType.NETWORK_FAILURE:
            return "circuit_breaker_with_fallback"
        if error_context.error_type == ErrorType.MEMORY_PRESSURE:
            return "reduce_load_and_retry"
        if error_context.error_type == ErrorType.DETECTION_FAILURE:
            return "safe_default_loading"
        if error_context.error_type == ErrorType.SYSTEM_OVERLOAD:
            return "throttle_and_fallback"
        return "conservative_fallback"


class PerformanceMonitor:
    """Monitors performance and provides health checks for fallback system"""

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self.response_times = deque(maxlen=max_history)
        self.memory_usage = deque(maxlen=max_history)
        self.error_rates = deque(maxlen=max_history)
        self.last_health_check = time.time()

        # Performance thresholds
        self.max_response_time = 5.0  # 5 second hard timeout
        self.max_memory_mb = 100      # 100MB memory limit
        self.max_error_rate = 0.1     # 10% error rate threshold

    def record_operation(self, duration: float, memory_usage: float, had_error: bool) -> None:
        """Record operation metrics"""
        self.response_times.append(duration)
        self.memory_usage.append(memory_usage)
        self.error_rates.append(1.0 if had_error else 0.0)

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status"""
        current_time = time.time()

        # Calculate metrics
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        max_response_time = max(self.response_times) if self.response_times else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        error_rate = sum(self.error_rates) / len(self.error_rates) if self.error_rates else 0

        # Health checks
        response_time_healthy = max_response_time < self.max_response_time
        memory_healthy = avg_memory < self.max_memory_mb
        error_rate_healthy = error_rate < self.max_error_rate

        overall_healthy = response_time_healthy and memory_healthy and error_rate_healthy

        return {
            "healthy": overall_healthy,
            "last_check": current_time,
            "metrics": {
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "avg_memory_mb": avg_memory,
                "error_rate": error_rate,
                "total_operations": len(self.response_times),
            },
            "thresholds": {
                "max_response_time": self.max_response_time,
                "max_memory_mb": self.max_memory_mb,
                "max_error_rate": self.max_error_rate,
            },
            "status": {
                "response_time_healthy": response_time_healthy,
                "memory_healthy": memory_healthy,
                "error_rate_healthy": error_rate_healthy,
            },
        }

    def should_trigger_emergency_mode(self) -> bool:
        """Determine if system should enter emergency mode"""
        if not self.response_times:
            return False

        # Check if recent operations are consistently slow
        recent_times = list(self.response_times)[-10:]  # Last 10 operations
        if len(recent_times) >= 5 and all(t > self.max_response_time * 0.8 for t in recent_times):
            return True

        # Check if error rate is too high
        recent_errors = list(self.error_rates)[-20:]  # Last 20 operations
        if len(recent_errors) >= 10:
            recent_error_rate = sum(recent_errors) / len(recent_errors)
            if recent_error_rate > self.max_error_rate * 2:
                return True

        return False


class RecoveryManager:
    """Manages automated recovery and retry logic"""

    def __init__(self, config: TaskDetectionConfig | None = None) -> None:
        self.config = config or TaskDetectionConfig()
        self.retry_attempts = defaultdict(int)
        self.recovery_history = deque(maxlen=100)
        self.max_retry_attempts = 3
        self.base_retry_delay = 1.0
        self.max_retry_delay = 30.0

    async def attempt_recovery(self, error_context: ErrorContext,
                             detection_system: TaskDetectionSystem) -> DetectionResult | None:
        """Attempt to recover from detection failure"""
        recovery_start = time.time()
        attempt_key = f"{error_context.error_type.value}_{error_context.query[:50]}"

        # Check if we've exceeded retry attempts
        if self.retry_attempts[attempt_key] >= self.max_retry_attempts:
            logger.warning(f"Max retry attempts exceeded for {attempt_key}")
            return None

        self.retry_attempts[attempt_key] += 1
        current_attempt = self.retry_attempts[attempt_key]

        # Calculate retry delay with exponential backoff
        delay = min(self.base_retry_delay * (2 ** (current_attempt - 1)), self.max_retry_delay)

        logger.info(f"Attempting recovery (attempt {current_attempt}/{self.max_retry_attempts}) "
                   f"after {delay}s delay for error: {error_context.error_type.value}")

        # Wait before retry
        await asyncio.sleep(delay)

        try:
            # Adjust context for recovery attempt
            recovery_context = error_context.context.copy()
            recovery_context["recovery_attempt"] = current_attempt
            recovery_context["original_error"] = error_context.error_type.value

            # Apply recovery strategy based on error type
            if error_context.error_type == ErrorType.MEMORY_PRESSURE:
                # Reduce processing complexity for memory issues
                recovery_context["force_lightweight"] = True

            elif error_context.error_type == ErrorType.TIMEOUT:
                # Extend timeout for timeout issues
                recovery_context["extended_timeout"] = True

            # Attempt detection with recovery context
            result = await detection_system.detect_categories(error_context.query, recovery_context)

            # Record successful recovery
            recovery_time = time.time() - recovery_start
            self.recovery_history.append({
                "success": True,
                "error_type": error_context.error_type.value,
                "attempt": current_attempt,
                "recovery_time": recovery_time,
                "timestamp": time.time(),
            })

            # Reset retry counter on success
            self.retry_attempts[attempt_key] = 0

            logger.info(f"Recovery successful after {recovery_time:.2f}s")
            return result

        except Exception as e:
            # Record failed recovery
            recovery_time = time.time() - recovery_start
            self.recovery_history.append({
                "success": False,
                "error_type": error_context.error_type.value,
                "attempt": current_attempt,
                "recovery_time": recovery_time,
                "error_message": str(e),
                "timestamp": time.time(),
            })

            logger.warning(f"Recovery attempt {current_attempt} failed after {recovery_time:.2f}s: {e}")
            return None

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics"""
        if not self.recovery_history:
            return {"total_attempts": 0, "success_rate": 0.0}

        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_history if r["success"])

        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts,
            "avg_recovery_time": sum(r["recovery_time"] for r in self.recovery_history) / total_attempts,
            "recent_attempts": list(self.recovery_history)[-10:],  # Last 10 attempts
        }


class LearningCollector:
    """Collects failure data for improving detection accuracy"""

    def __init__(self, max_samples: int = 1000) -> None:
        self.max_samples = max_samples
        self.failure_patterns = deque(maxlen=max_samples)
        self.success_patterns = deque(maxlen=max_samples)
        self.query_patterns = defaultdict(list)

    def record_failure(self, query: str, context: dict[str, Any],
                      error_context: ErrorContext, fallback_level: FallbackLevel) -> None:
        """Record a detection failure for learning"""
        pattern = {
            "query": query,
            "context_keys": list(context.keys()),
            "error_type": error_context.error_type.value,
            "severity": error_context.severity.value,
            "fallback_level": fallback_level.value,
            "timestamp": time.time(),
            "query_length": len(query),
            "query_complexity": self._estimate_query_complexity(query),
        }

        self.failure_patterns.append(pattern)
        self.query_patterns[self._get_query_category(query)].append(pattern)

    def record_success(self, query: str, context: dict[str, Any],
                      result: DetectionResult) -> None:
        """Record a successful detection for learning"""
        pattern = {
            "query": query,
            "context_keys": list(context.keys()),
            "detected_categories": [k for k, v in result.categories.items() if v],
            "confidence_scores": result.confidence_scores,
            "detection_time": result.detection_time_ms,
            "timestamp": time.time(),
            "query_length": len(query),
            "query_complexity": self._estimate_query_complexity(query),
        }

        self.success_patterns.append(pattern)
        self.query_patterns[self._get_query_category(query)].append(pattern)

    def get_learning_insights(self) -> dict[str, Any]:
        """Generate insights from collected data"""
        if not self.failure_patterns and not self.success_patterns:
            return {"insights": [], "recommendations": []}

        insights = []
        recommendations = []

        # Analyze failure patterns
        if self.failure_patterns:
            error_types = defaultdict(int)
            fallback_levels = defaultdict(int)

            for pattern in self.failure_patterns:
                error_types[pattern["error_type"]] += 1
                fallback_levels[pattern["fallback_level"]] += 1

            most_common_error = max(error_types, key=error_types.get)
            most_common_fallback = max(fallback_levels, key=fallback_levels.get)

            insights.append(f"Most common error type: {most_common_error} ({error_types[most_common_error]} occurrences)")
            insights.append(f"Most common fallback level: {most_common_fallback} ({fallback_levels[most_common_fallback]} occurrences)")

            if error_types.get("timeout", 0) > len(self.failure_patterns) * 0.3:
                recommendations.append("Consider increasing detection timeout threshold")

            if fallback_levels.get("detection_failure", 0) > len(self.failure_patterns) * 0.5:
                recommendations.append("Consider improving base detection algorithms")

        # Analyze success patterns for optimization opportunities
        if self.success_patterns:
            avg_detection_time = sum(p["detection_time"] for p in self.success_patterns) / len(self.success_patterns)
            insights.append(f"Average successful detection time: {avg_detection_time:.2f}ms")

            if avg_detection_time > 100:
                recommendations.append("Consider optimizing detection performance")

        return {
            "insights": insights,
            "recommendations": recommendations,
            "total_failures": len(self.failure_patterns),
            "total_successes": len(self.success_patterns),
            "failure_rate": len(self.failure_patterns) / (len(self.failure_patterns) + len(self.success_patterns)) if (self.failure_patterns or self.success_patterns) else 0,
        }

    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity (0.0 to 1.0)"""
        word_count = len(query.split())
        base_complexity = min(1.0, word_count / 20.0)

        complexity_indicators = ["and", "or", "multiple", "complex", "analyze"]
        indicator_bonus = sum(0.1 for indicator in complexity_indicators if indicator in query.lower())

        return min(1.0, base_complexity + indicator_bonus)

    def _get_query_category(self, query: str) -> str:
        """Categorize query for pattern analysis"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["git", "commit", "branch"]):
            return "git"
        if any(word in query_lower for word in ["test", "testing", "spec"]):
            return "test"
        if any(word in query_lower for word in ["debug", "error", "bug"]):
            return "debug"
        if any(word in query_lower for word in ["security", "auth", "permission"]):
            return "security"
        if any(word in query_lower for word in ["analyze", "review", "understand"]):
            return "analysis"
        return "general"


class ConservativeFallbackChain(LoggerMixin):
    """
    Main fallback controller implementing the five-tier progressive degradation strategy
    """

    def __init__(self,
                 detection_system: TaskDetectionSystem,
                 config: TaskDetectionConfig | None = None) -> None:
        super().__init__(logger_name="conservative_fallback")

        self.detection_system = detection_system
        self.config = config or TaskDetectionConfig()

        # Initialize components
        self.error_classifier = ErrorClassifier()
        self.performance_monitor = PerformanceMonitor()
        self.recovery_manager = RecoveryManager(config)
        self.learning_collector = LearningCollector()

        # Metrics tracking
        self.metrics = FallbackMetrics()

        # Circuit breaker for system protection
        self.circuit_breaker_open = False
        self.circuit_breaker_last_failure = 0.0
        self.circuit_breaker_failure_count = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60.0  # 60 seconds

        # Emergency mode tracking
        self.emergency_mode = False
        self.emergency_mode_start = 0.0
        self.emergency_mode_duration = 300.0  # 5 minutes

        # Function category definitions for fallback levels
        self.tier_definitions = {
            FallbackLevel.HIGH_CONFIDENCE: {
                "categories": set(),  # Determined dynamically based on detection
                "expected_count": 35,
                "description": "High-confidence detected categories only",
            },
            FallbackLevel.MEDIUM_CONFIDENCE: {
                "categories": {"core", "git", "analysis"},  # Base + top detection + buffer
                "expected_count": 50,
                "description": "Detected categories plus safety buffer",
            },
            FallbackLevel.LOW_CONFIDENCE: {
                "categories": {"core", "git", "analysis", "debug"},  # Safe conservative default
                "expected_count": 40,
                "description": "Core development functionality",
            },
            FallbackLevel.DETECTION_FAILURE: {
                "categories": {"core", "git", "analysis", "debug", "test", "quality", "security"},  # Almost everything
                "expected_count": 85,
                "description": "Comprehensive coverage minus external tools",
            },
            FallbackLevel.SYSTEM_EMERGENCY: {
                "categories": {"core", "git", "analysis", "debug", "test", "quality", "security", "external", "infrastructure"},  # Everything
                "expected_count": 98,
                "description": "Complete function availability",
            },
        }

    async def get_function_categories(self, query: str,
                                    context: dict[str, Any] | None = None) -> tuple[dict[str, bool], FallbackDecision]:
        """
        Main entry point for conservative fallback chain function loading

        Returns tuple of (categories_to_load, fallback_decision)
        """
        operation_start = time.time()
        had_error = False

        if context is None:
            context = {}

        try:
            # Check if we're in emergency mode
            if self._is_emergency_mode_active():
                return self._emergency_mode_loading(query, context)

            # Check circuit breaker status
            if self._is_circuit_breaker_open():
                return self._circuit_breaker_fallback(query, context)

            # Attempt normal detection with timeout
            try:
                result = await asyncio.wait_for(
                    self.detection_system.detect_categories(query, context),
                    timeout=5.0,  # 5 second hard timeout
                )

                # Successful detection - determine fallback level based on confidence
                decision = self._make_fallback_decision(result, query, context)
                categories = self._apply_fallback_decision(decision, result)

                # Record success
                self.learning_collector.record_success(query, context, result)

                return categories, decision

            except TimeoutError:
                # Timeout - trigger fallback
                error_context = ErrorContext(
                    error_type=ErrorType.TIMEOUT,
                    severity=ErrorSeverity.HIGH,
                    timestamp=time.time(),
                    query=query,
                    context=context,
                )
                return await self._handle_detection_failure(error_context)

        except Exception as e:
            had_error = True

            # Classify the error
            error_context = self.error_classifier.classify_error(e, context)

            # Update circuit breaker
            self._update_circuit_breaker(error_context)

            # Record failure for learning
            self.learning_collector.record_failure(query, context, error_context, FallbackLevel.DETECTION_FAILURE)

            # Handle the failure
            return await self._handle_detection_failure(error_context)

        finally:
            # Record operation metrics
            operation_time = time.time() - operation_start
            memory_usage = 0.0  # Would use actual memory profiling in production
            self.performance_monitor.record_operation(operation_time, memory_usage, had_error)

            # Update metrics
            self.metrics.activation_count += 1
            if had_error:
                self.metrics.failed_recoveries += 1
            else:
                self.metrics.successful_recoveries += 1

    def _make_fallback_decision(self, result: DetectionResult,
                              query: str, context: dict[str, Any]) -> FallbackDecision:
        """Make fallback decision based on detection confidence"""

        # Calculate overall confidence as max of category scores
        max_confidence = max(result.confidence_scores.values()) if result.confidence_scores else 0.0

        # Determine fallback level
        if max_confidence >= 0.7:
            level = FallbackLevel.HIGH_CONFIDENCE
            rationale = f"High confidence detection (max: {max_confidence:.2f})"

        elif max_confidence >= 0.3:
            level = FallbackLevel.MEDIUM_CONFIDENCE
            rationale = f"Medium confidence detection (max: {max_confidence:.2f}), adding safety buffer"

        else:
            level = FallbackLevel.LOW_CONFIDENCE
            rationale = f"Low confidence detection (max: {max_confidence:.2f}), using safe default"

        # Get tier definition
        tier_def = self.tier_definitions[level]

        # For high confidence, use detected categories
        if level == FallbackLevel.HIGH_CONFIDENCE:
            detected_categories = {k for k, v in result.categories.items() if v}
            categories_to_load = detected_categories
        else:
            categories_to_load = tier_def["categories"].copy()

            # For medium confidence, add detected categories to base set
            if level == FallbackLevel.MEDIUM_CONFIDENCE:
                detected_categories = {k for k, v in result.categories.items() if v}
                categories_to_load.update(detected_categories)

        # Apply conservative bias
        if self.config.mode == DetectionMode.CONSERVATIVE:
            # Add extra categories for conservative mode
            if level == FallbackLevel.HIGH_CONFIDENCE:
                categories_to_load.add("analysis")  # Always include analysis for safety
            elif level == FallbackLevel.MEDIUM_CONFIDENCE:
                categories_to_load.add("quality")  # Add quality tools

        # Convert to boolean dict
        all_categories = {"core", "git", "analysis", "debug", "test", "quality", "security", "external", "infrastructure"}
        categories_dict = {cat: cat in categories_to_load for cat in all_categories}

        return FallbackDecision(
            level=level,
            categories_to_load=categories_dict,
            confidence_threshold=max_confidence,
            expected_function_count=len(categories_to_load) * 10,  # Rough estimate
            rationale=rationale,
            performance_impact="minimal" if level == FallbackLevel.HIGH_CONFIDENCE else "moderate",
            recovery_strategy="none_needed",
        )

    def _apply_fallback_decision(self, decision: FallbackDecision,
                               result: DetectionResult) -> dict[str, bool]:
        """Apply the fallback decision and return categories to load"""

        # Update metrics
        self.metrics.level_activations[decision.level] += 1

        # Log decision
        self.logger.info(
            f"Fallback decision: {decision.level.value} - {decision.rationale} "
            f"(loading {sum(decision.categories_to_load.values())} categories)",
        )

        return decision.categories_to_load

    async def _handle_detection_failure(self, error_context: ErrorContext) -> tuple[dict[str, bool], FallbackDecision]:
        """Handle detection failure with progressive fallback"""

        self.logger.warning(f"Detection failure: {error_context.error_type.value} - {error_context.metadata.get('exception_message', 'Unknown error')}")

        # Update error metrics
        self.metrics.error_counts[error_context.error_type] += 1

        # Attempt recovery first
        recovery_result = await self.recovery_manager.attempt_recovery(error_context, self.detection_system)

        if recovery_result:
            # Recovery successful - use recovered result
            decision = self._make_fallback_decision(recovery_result, error_context.query, error_context.context)
            categories = self._apply_fallback_decision(decision, recovery_result)

            self.metrics.successful_recoveries += 1
            return categories, decision

        # Recovery failed - use progressive fallback
        if error_context.severity == ErrorSeverity.CRITICAL or self.performance_monitor.should_trigger_emergency_mode():
            # Critical error or performance issues - emergency mode
            return self._emergency_mode_loading(error_context.query, error_context.context)

        if error_context.error_type in [ErrorType.TIMEOUT, ErrorType.NETWORK_FAILURE]:
            # Network/timeout issues - detection failure level
            level = FallbackLevel.DETECTION_FAILURE
            rationale = f"Detection failure due to {error_context.error_type.value}, using comprehensive fallback"

        else:
            # Other issues - low confidence level
            level = FallbackLevel.LOW_CONFIDENCE
            rationale = f"Detection unavailable due to {error_context.error_type.value}, using safe default"

        # Get categories for fallback level
        tier_def = self.tier_definitions[level]
        categories_dict = {cat: cat in tier_def["categories"] for cat in
                          ("core", "git", "analysis", "debug", "test", "quality", "security", "external", "infrastructure")}

        decision = FallbackDecision(
            level=level,
            categories_to_load=categories_dict,
            confidence_threshold=0.0,
            expected_function_count=tier_def["expected_count"],
            rationale=rationale,
            performance_impact="high" if level == FallbackLevel.DETECTION_FAILURE else "moderate",
            recovery_strategy=self.error_classifier.get_recommended_recovery_strategy(error_context),
        )

        # Record failure
        self.learning_collector.record_failure(error_context.query, error_context.context, error_context, level)

        return categories_dict, decision

    def _emergency_mode_loading(self, query: str, context: dict[str, Any]) -> tuple[dict[str, bool], FallbackDecision]:
        """Emergency mode - load everything immediately"""

        if not self.emergency_mode:
            self.emergency_mode = True
            self.emergency_mode_start = time.time()
            self.logger.critical("ENTERING EMERGENCY MODE - Loading all functions")

        # Load absolutely everything
        tier_def = self.tier_definitions[FallbackLevel.SYSTEM_EMERGENCY]
        categories_dict = dict.fromkeys(tier_def["categories"], True)

        decision = FallbackDecision(
            level=FallbackLevel.SYSTEM_EMERGENCY,
            categories_to_load=categories_dict,
            confidence_threshold=0.0,
            expected_function_count=tier_def["expected_count"],
            rationale="EMERGENCY MODE: System unavailable, loading all functions immediately",
            performance_impact="maximum",
            recovery_strategy="emergency_full_load",
        )

        self.metrics.level_activations[FallbackLevel.SYSTEM_EMERGENCY] += 1

        return categories_dict, decision

    def _circuit_breaker_fallback(self, query: str, context: dict[str, Any]) -> tuple[dict[str, bool], FallbackDecision]:
        """Circuit breaker is open - use detection failure level"""

        self.logger.warning("Circuit breaker is OPEN - using detection failure fallback")

        tier_def = self.tier_definitions[FallbackLevel.DETECTION_FAILURE]
        categories_dict = {cat: cat in tier_def["categories"] for cat in
                          ("core", "git", "analysis", "debug", "test", "quality", "security", "external", "infrastructure")}

        decision = FallbackDecision(
            level=FallbackLevel.DETECTION_FAILURE,
            categories_to_load=categories_dict,
            confidence_threshold=0.0,
            expected_function_count=tier_def["expected_count"],
            rationale="Circuit breaker OPEN - using comprehensive fallback",
            performance_impact="high",
            recovery_strategy="circuit_breaker_protection",
        )

        return categories_dict, decision

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_breaker_open:
            return False

        # Check if timeout has elapsed
        if time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout:
            self.circuit_breaker_open = False
            self.circuit_breaker_failure_count = 0
            self.logger.info("Circuit breaker reset - normal operation resumed")
            return False

        return True

    def _update_circuit_breaker(self, error_context: ErrorContext) -> None:
        """Update circuit breaker state based on error"""
        if self.error_classifier.should_trigger_circuit_breaker(error_context):
            self.circuit_breaker_failure_count += 1
            self.circuit_breaker_last_failure = time.time()

            if self.circuit_breaker_failure_count >= self.circuit_breaker_threshold:
                self.circuit_breaker_open = True
                self.logger.critical(f"Circuit breaker OPENED after {self.circuit_breaker_failure_count} failures")

    def _is_emergency_mode_active(self) -> bool:
        """Check if emergency mode is active"""
        if not self.emergency_mode:
            return False

        # Check if emergency mode should end
        if time.time() - self.emergency_mode_start > self.emergency_mode_duration:
            self.emergency_mode = False
            self.logger.info("Emergency mode ended - resuming normal operation")
            return False

        return True

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status of fallback system"""
        performance_status = self.performance_monitor.get_health_status()
        recovery_stats = self.recovery_manager.get_recovery_stats()
        learning_insights = self.learning_collector.get_learning_insights()

        return {
            "system_status": {
                "healthy": performance_status["healthy"] and not self.circuit_breaker_open and not self.emergency_mode,
                "circuit_breaker_open": self.circuit_breaker_open,
                "emergency_mode": self.emergency_mode,
                "last_health_check": time.time(),
            },
            "performance": performance_status,
            "recovery": recovery_stats,
            "learning": learning_insights,
            "metrics": self.metrics.to_dict(),
            "circuit_breaker": {
                "open": self.circuit_breaker_open,
                "failure_count": self.circuit_breaker_failure_count,
                "last_failure": self.circuit_breaker_last_failure,
                "threshold": self.circuit_breaker_threshold,
                "timeout": self.circuit_breaker_timeout,
            },
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker (for admin/testing)"""
        self.circuit_breaker_open = False
        self.circuit_breaker_failure_count = 0
        self.logger.info("Circuit breaker manually reset")

    def exit_emergency_mode(self) -> None:
        """Manually exit emergency mode (for admin/testing)"""
        self.emergency_mode = False
        self.logger.info("Emergency mode manually disabled")


# Factory function for easy integration
def create_conservative_fallback_chain(detection_system: TaskDetectionSystem,
                                     config: TaskDetectionConfig | None = None) -> ConservativeFallbackChain:
    """Create a conservative fallback chain with proper configuration"""

    if config is None:
        config = TaskDetectionConfig()
        # Apply conservative mode for safety
        config.apply_mode_preset(DetectionMode.CONSERVATIVE)

    return ConservativeFallbackChain(detection_system, config)


# Decorator for automatic fallback protection
def with_conservative_fallback(fallback_chain: ConservativeFallbackChain):
    """Decorator to add conservative fallback protection to functions"""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Extract query and context from args/kwargs if possible
                query = kwargs.get("query", args[0] if args else "")
                context = kwargs.get("context", args[1] if len(args) > 1 else {})

                # Use fallback chain for recovery
                categories, decision = await fallback_chain.get_function_categories(query, context)

                # Return fallback result with metadata
                return {
                    "fallback_used": True,
                    "fallback_decision": decision,
                    "categories": categories,
                    "original_error": str(e),
                }

        return wrapper

    return decorator
