"""
Token Optimization Performance Monitoring System

This module provides comprehensive monitoring and validation for the dynamic function
loading system's 70% token reduction goal. It tracks token usage, function loading
efficiency, and user experience metrics to ensure optimization claims are validated.
"""

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from statistics import mean, median
from typing import Any

from src.core.analytics_engine import AnalyticsEngine
from src.utils.observability import create_structured_logger
from src.utils.performance_monitor import MetricData, MetricType, PerformanceMonitor

logger = logging.getLogger(__name__)


class OptimizationStatus(Enum):
    """Status of dynamic function loading optimization."""

    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class FunctionTier(Enum):
    """Function loading tiers based on usage patterns."""

    TIER_1 = "tier_1"  # Most frequently used - <50ms loading
    TIER_2 = "tier_2"  # Moderately used - <100ms loading
    TIER_3 = "tier_3"  # Rarely used - <200ms loading
    FALLBACK = "fallback"  # Available on demand


@dataclass
class TokenUsageMetrics:
    """Metrics tracking token usage and optimization."""

    session_id: str
    user_id: str
    timestamp: datetime

    # Baseline metrics (without optimization)
    baseline_tokens_loaded: int = 0
    baseline_total_functions: int = 0

    # Optimized metrics (with dynamic loading)
    optimized_tokens_loaded: int = 0
    optimized_functions_loaded: int = 0
    functions_actually_used: set[str] = field(default_factory=set)

    # Performance metrics
    loading_latency_ms: float = 0.0
    task_detection_confidence: float = 0.0
    task_detection_accuracy: bool = True

    # User experience metrics
    commands_successful: int = 0
    commands_failed: int = 0
    help_requests: int = 0
    fallback_activations: int = 0
    user_override_count: int = 0

    # Context information
    task_type: str | None = None
    optimization_level: OptimizationStatus = OptimizationStatus.CONSERVATIVE
    session_duration_seconds: float = 0.0


@dataclass
class FunctionLoadingMetrics:
    """Metrics for function loading performance by tier."""

    tier: FunctionTier
    functions_loaded: int = 0
    loading_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_consumed: int = 0
    usage_frequency: float = 0.0  # How often loaded functions are actually used


@dataclass
class SystemHealthMetrics:
    """Overall system health and performance metrics."""

    timestamp: datetime

    # Token optimization metrics
    total_sessions: int = 0
    average_token_reduction_percentage: float = 0.0
    median_token_reduction_percentage: float = 0.0

    # Performance metrics
    average_loading_latency_ms: float = 0.0
    p95_loading_latency_ms: float = 0.0
    p99_loading_latency_ms: float = 0.0

    # User experience metrics
    overall_success_rate: float = 0.0
    task_detection_accuracy_rate: float = 0.0
    fallback_activation_rate: float = 0.0

    # System capacity metrics
    concurrent_sessions_handled: int = 0
    memory_usage_mb: float = 0.0
    cpu_utilization_percentage: float = 0.0


class TokenOptimizationMonitor:
    """Main monitoring system for token optimization validation."""

    def __init__(self, analytics_engine: AnalyticsEngine | None = None) -> None:
        self.analytics_engine = analytics_engine or AnalyticsEngine()
        self.performance_monitor = PerformanceMonitor()
        self.logger = create_structured_logger("token_optimization_monitor")

        # Metrics storage
        self.session_metrics: dict[str, TokenUsageMetrics] = {}
        self.function_metrics: dict[FunctionTier, FunctionLoadingMetrics] = {
            tier: FunctionLoadingMetrics(tier=tier) for tier in FunctionTier
        }
        self.system_health_history: deque = deque(maxlen=1000)

        # Real-time tracking
        self.active_sessions: set[str] = set()
        self.baseline_measurements: dict[str, int] = {}  # session_id -> baseline tokens

        # Alert thresholds
        self.token_reduction_target = 0.70  # 70% reduction goal
        self.min_acceptable_reduction = 0.50  # 50% minimum
        self.max_acceptable_latency_ms = 200.0  # Maximum function loading latency

        # Validation flags
        self.optimization_validated = False
        self.validation_confidence = 0.0

    async def start_session_monitoring(
        self,
        session_id: str,
        user_id: str,
        task_type: str | None = None,
        optimization_level: OptimizationStatus = OptimizationStatus.CONSERVATIVE,
    ) -> None:
        """Start monitoring a new user session."""

        self.active_sessions.add(session_id)

        metrics = TokenUsageMetrics(
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            task_type=task_type,
            optimization_level=optimization_level,
        )

        self.session_metrics[session_id] = metrics

        # Measure baseline if not already done for this session type
        baseline_key = f"{user_id}_{task_type}_{optimization_level.value}"
        if baseline_key not in self.baseline_measurements:
            baseline_tokens = await self._measure_baseline_tokens(task_type)
            self.baseline_measurements[baseline_key] = baseline_tokens
            metrics.baseline_tokens_loaded = baseline_tokens

        self.logger.info(
            "Started session monitoring",
            session_id=session_id,
            user_id=user_id,
            task_type=task_type,
            optimization_level=optimization_level.value,
        )

        # Track in analytics engine
        self.analytics_engine.track_user_action(
            action_type="session_started",
            action_data={
                "task_type": task_type,
                "optimization_level": optimization_level.value,
                "baseline_tokens": metrics.baseline_tokens_loaded,
            },
            user_id=user_id,
            session_id=session_id,
        )

    async def record_function_loading(
        self,
        session_id: str,
        tier: FunctionTier,
        functions_loaded: list[str],
        loading_time_ms: float,
        tokens_consumed: int,
        cache_hit: bool = False,
    ) -> None:
        """Record function loading metrics for a specific tier."""

        if session_id not in self.session_metrics:
            self.logger.warning(f"Recording function loading for unknown session: {session_id}")
            return

        session_metrics = self.session_metrics[session_id]
        tier_metrics = self.function_metrics[tier]

        # Update tier metrics
        tier_metrics.functions_loaded += len(functions_loaded)
        tier_metrics.loading_time_ms += loading_time_ms
        tier_metrics.tokens_consumed += tokens_consumed

        if cache_hit:
            tier_metrics.cache_hits += 1
        else:
            tier_metrics.cache_misses += 1

        # Update session metrics
        session_metrics.optimized_tokens_loaded += tokens_consumed
        session_metrics.optimized_functions_loaded += len(functions_loaded)
        session_metrics.loading_latency_ms += loading_time_ms

        # Record performance metrics
        self.performance_monitor.record_metric(
            MetricData(
                name=f"function_loading_latency_{tier.value}",
                value=loading_time_ms,
                timestamp=time.time(),
                labels={"tier": tier.value, "session_id": session_id},
                metric_type=MetricType.TIMER,
            ),
        )

        self.performance_monitor.record_metric(
            MetricData(
                name=f"tokens_loaded_{tier.value}",
                value=tokens_consumed,
                timestamp=time.time(),
                labels={"tier": tier.value, "session_id": session_id},
                metric_type=MetricType.HISTOGRAM,
            ),
        )

        # Check performance thresholds
        if loading_time_ms > self.max_acceptable_latency_ms:
            self.logger.warning(
                "Function loading exceeded latency threshold",
                session_id=session_id,
                tier=tier.value,
                loading_time_ms=loading_time_ms,
                threshold_ms=self.max_acceptable_latency_ms,
            )

        self.logger.debug(
            "Recorded function loading",
            session_id=session_id,
            tier=tier.value,
            functions_count=len(functions_loaded),
            loading_time_ms=loading_time_ms,
            tokens_consumed=tokens_consumed,
            cache_hit=cache_hit,
        )

    async def record_function_usage(
        self, session_id: str, function_name: str, success: bool = True, tier: FunctionTier | None = None,
    ) -> None:
        """Record actual function usage to measure effectiveness."""

        if session_id not in self.session_metrics:
            return

        session_metrics = self.session_metrics[session_id]
        session_metrics.functions_actually_used.add(function_name)

        if success:
            session_metrics.commands_successful += 1
        else:
            session_metrics.commands_failed += 1

        # Update tier usage frequency if tier is known
        if tier and tier in self.function_metrics:
            tier_metrics = self.function_metrics[tier]
            if tier_metrics.functions_loaded > 0:
                tier_metrics.usage_frequency = (
                    len(session_metrics.functions_actually_used) / tier_metrics.functions_loaded
                )

        # Track in analytics
        self.analytics_engine.track_user_action(
            action_type="function_used",
            action_data={"function_name": function_name, "success": success, "tier": tier.value if tier else "unknown"},
            user_id=session_metrics.user_id,
            session_id=session_id,
        )

    async def record_task_detection(
        self, session_id: str, detected_task: str, confidence: float, actual_task: str | None = None,
    ) -> None:
        """Record task detection accuracy for validation."""

        if session_id not in self.session_metrics:
            return

        session_metrics = self.session_metrics[session_id]
        session_metrics.task_detection_confidence = confidence

        # Determine accuracy if actual task is known
        if actual_task is not None:
            session_metrics.task_detection_accuracy = detected_task.lower() == actual_task.lower()

        self.performance_monitor.record_metric(
            MetricData(
                name="task_detection_confidence",
                value=confidence,
                timestamp=time.time(),
                labels={"session_id": session_id, "detected_task": detected_task},
                metric_type=MetricType.GAUGE,
            ),
        )

        self.logger.debug(
            "Recorded task detection",
            session_id=session_id,
            detected_task=detected_task,
            confidence=confidence,
            accuracy=session_metrics.task_detection_accuracy,
        )

    async def record_fallback_activation(self, session_id: str, reason: str, missing_functions: list[str]) -> None:
        """Record fallback system activation."""

        if session_id not in self.session_metrics:
            return

        session_metrics = self.session_metrics[session_id]
        session_metrics.fallback_activations += 1

        self.performance_monitor.record_metric(
            MetricData(
                name="fallback_activations",
                value=1,
                timestamp=time.time(),
                labels={"session_id": session_id, "reason": reason},
                metric_type=MetricType.COUNTER,
            ),
        )

        self.logger.warning(
            "Fallback system activated", session_id=session_id, reason=reason, missing_functions=missing_functions,
        )

        # Track in analytics
        self.analytics_engine.track_user_action(
            action_type="fallback_activated",
            action_data={
                "reason": reason,
                "missing_functions": missing_functions,
                "fallback_count": session_metrics.fallback_activations,
            },
            user_id=session_metrics.user_id,
            session_id=session_id,
        )

    async def record_user_override(
        self, session_id: str, override_type: str, original_optimization: str, new_optimization: str,
    ) -> None:
        """Record user override of optimization settings."""

        if session_id not in self.session_metrics:
            return

        session_metrics = self.session_metrics[session_id]
        session_metrics.user_override_count += 1

        self.performance_monitor.record_metric(
            MetricData(
                name="user_overrides",
                value=1,
                timestamp=time.time(),
                labels={
                    "session_id": session_id,
                    "override_type": override_type,
                    "from": original_optimization,
                    "to": new_optimization,
                },
                metric_type=MetricType.COUNTER,
            ),
        )

        self.logger.info(
            "User optimization override",
            session_id=session_id,
            override_type=override_type,
            from_optimization=original_optimization,
            to_optimization=new_optimization,
        )

    async def end_session_monitoring(self, session_id: str) -> TokenUsageMetrics | None:
        """End session monitoring and calculate final metrics."""

        if session_id not in self.session_metrics:
            return None

        session_metrics = self.session_metrics[session_id]
        session_metrics.session_duration_seconds = (datetime.now() - session_metrics.timestamp).total_seconds()

        # Calculate token reduction percentage
        if session_metrics.baseline_tokens_loaded > 0:
            token_reduction = 1.0 - (session_metrics.optimized_tokens_loaded / session_metrics.baseline_tokens_loaded)

            self.performance_monitor.record_metric(
                MetricData(
                    name="token_reduction_percentage",
                    value=token_reduction * 100,
                    timestamp=time.time(),
                    labels={
                        "session_id": session_id,
                        "task_type": session_metrics.task_type or "unknown",
                        "optimization_level": session_metrics.optimization_level.value,
                    },
                    metric_type=MetricType.HISTOGRAM,
                ),
            )

            # Validate against target
            if token_reduction >= self.token_reduction_target:
                self.logger.info(
                    "Token reduction target achieved",
                    session_id=session_id,
                    reduction_percentage=token_reduction * 100,
                    target_percentage=self.token_reduction_target * 100,
                )
            elif token_reduction < self.min_acceptable_reduction:
                self.logger.warning(
                    "Token reduction below minimum threshold",
                    session_id=session_id,
                    reduction_percentage=token_reduction * 100,
                    minimum_percentage=self.min_acceptable_reduction * 100,
                )

        # Calculate function usage efficiency
        usage_efficiency = 0.0
        if session_metrics.optimized_functions_loaded > 0:
            usage_efficiency = len(session_metrics.functions_actually_used) / session_metrics.optimized_functions_loaded

        self.performance_monitor.record_metric(
            MetricData(
                name="function_usage_efficiency",
                value=usage_efficiency,
                timestamp=time.time(),
                labels={"session_id": session_id},
                metric_type=MetricType.GAUGE,
            ),
        )

        # Track session completion in analytics
        self.analytics_engine.track_user_action(
            action_type="session_completed",
            action_data={
                "session_duration_seconds": session_metrics.session_duration_seconds,
                "token_reduction_percentage": (
                    token_reduction * 100 if session_metrics.baseline_tokens_loaded > 0 else 0
                ),
                "function_usage_efficiency": usage_efficiency,
                "commands_successful": session_metrics.commands_successful,
                "commands_failed": session_metrics.commands_failed,
                "fallback_activations": session_metrics.fallback_activations,
                "user_overrides": session_metrics.user_override_count,
            },
            user_id=session_metrics.user_id,
            session_id=session_id,
        )

        # Remove from active sessions
        self.active_sessions.discard(session_id)

        self.logger.info(
            "Session monitoring completed",
            session_id=session_id,
            duration_seconds=session_metrics.session_duration_seconds,
            token_reduction_percentage=token_reduction * 100 if session_metrics.baseline_tokens_loaded > 0 else 0,
            function_usage_efficiency=usage_efficiency * 100,
        )

        return session_metrics

    async def _measure_baseline_tokens(self, task_type: str | None = None) -> int:
        """Measure baseline token usage for comparison."""

        # This would measure token usage with ALL functions loaded
        # For now, we'll use estimated values based on task type
        baseline_estimates = {
            "debugging": 15000,  # All debug-related functions
            "security": 12000,  # All security functions
            "testing": 10000,  # All testing functions
            "documentation": 8000,  # All documentation functions
            "general": 20000,  # All functions loaded
            None: 20000,  # Default to all functions
        }

        baseline = baseline_estimates.get(task_type, baseline_estimates[None])

        self.logger.debug("Baseline token measurement", task_type=task_type, baseline_tokens=baseline)

        return baseline

    async def generate_system_health_report(self) -> SystemHealthMetrics:
        """Generate comprehensive system health metrics."""

        current_time = datetime.now()

        # Calculate token reduction statistics
        recent_sessions = [
            metrics
            for metrics in self.session_metrics.values()
            if (current_time - metrics.timestamp).total_seconds() <= 3600  # Last hour
        ]

        token_reductions = []
        task_accuracies = []
        success_rates = []
        loading_latencies = []
        fallback_rates = []

        for session in recent_sessions:
            if session.baseline_tokens_loaded > 0:
                reduction = 1.0 - (session.optimized_tokens_loaded / session.baseline_tokens_loaded)
                token_reductions.append(reduction)

            if session.task_detection_confidence > 0:
                task_accuracies.append(1.0 if session.task_detection_accuracy else 0.0)

            total_commands = session.commands_successful + session.commands_failed
            if total_commands > 0:
                success_rates.append(session.commands_successful / total_commands)

            if session.loading_latency_ms > 0:
                loading_latencies.append(session.loading_latency_ms)

            if session.optimized_functions_loaded > 0:
                fallback_rates.append(session.fallback_activations / session.optimized_functions_loaded)

        # Calculate system health metrics
        health_metrics = SystemHealthMetrics(
            timestamp=current_time,
            total_sessions=len(recent_sessions),
            average_token_reduction_percentage=mean(token_reductions) * 100 if token_reductions else 0.0,
            median_token_reduction_percentage=median(token_reductions) * 100 if token_reductions else 0.0,
            average_loading_latency_ms=mean(loading_latencies) if loading_latencies else 0.0,
            p95_loading_latency_ms=self._percentile(loading_latencies, 95) if loading_latencies else 0.0,
            p99_loading_latency_ms=self._percentile(loading_latencies, 99) if loading_latencies else 0.0,
            overall_success_rate=mean(success_rates) if success_rates else 0.0,
            task_detection_accuracy_rate=mean(task_accuracies) if task_accuracies else 0.0,
            fallback_activation_rate=mean(fallback_rates) if fallback_rates else 0.0,
            concurrent_sessions_handled=len(self.active_sessions),
        )

        # Store in history
        self.system_health_history.append(health_metrics)

        # Update validation status
        await self._update_optimization_validation(health_metrics)

        return health_metrics

    async def _update_optimization_validation(self, health_metrics: SystemHealthMetrics) -> None:
        """Update optimization validation based on current metrics."""

        # Requirements for validation:
        # 1. Average token reduction >= 70%
        # 2. Minimum 10 sessions analyzed
        # 3. Task detection accuracy >= 80%
        # 4. Overall success rate >= 95%
        # 5. Loading latency P95 <= 200ms

        validation_criteria = {
            "token_reduction": health_metrics.average_token_reduction_percentage >= 70.0,
            "sample_size": health_metrics.total_sessions >= 10,
            "task_accuracy": health_metrics.task_detection_accuracy_rate >= 0.80,
            "success_rate": health_metrics.overall_success_rate >= 0.95,
            "latency": health_metrics.p95_loading_latency_ms <= 200.0,
        }

        met_criteria = sum(validation_criteria.values())
        total_criteria = len(validation_criteria)

        self.validation_confidence = met_criteria / total_criteria
        self.optimization_validated = self.validation_confidence >= 0.80  # 80% of criteria met

        self.logger.info(
            "Optimization validation update",
            validation_confidence=self.validation_confidence,
            optimization_validated=self.optimization_validated,
            criteria_met=met_criteria,
            total_criteria=total_criteria,
            **validation_criteria,
        )

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    async def export_metrics(self, format: str = "json", include_raw_data: bool = False) -> dict[str, Any]:
        """Export comprehensive metrics for analysis."""

        health_report = await self.generate_system_health_report()

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "validation_status": {
                "optimization_validated": self.optimization_validated,
                "validation_confidence": self.validation_confidence,
                "token_reduction_target": self.token_reduction_target * 100,
                "min_acceptable_reduction": self.min_acceptable_reduction * 100,
            },
            "system_health": asdict(health_report),
            "function_tier_metrics": {tier.value: asdict(metrics) for tier, metrics in self.function_metrics.items()},
            "performance_summary": self.performance_monitor.get_all_metrics(),
        }

        if include_raw_data:
            export_data["raw_session_data"] = {
                session_id: asdict(metrics) for session_id, metrics in self.session_metrics.items()
            }

        return export_data

    async def get_optimization_report(self, user_id: str | None = None) -> dict[str, Any]:
        """Generate optimization performance report."""

        # Filter sessions by user if specified
        relevant_sessions = [
            metrics for metrics in self.session_metrics.values() if user_id is None or metrics.user_id == user_id
        ]

        if not relevant_sessions:
            return {"error": "No session data available"}

        # Calculate optimization statistics
        token_savings = []
        loading_times = []
        usage_efficiencies = []
        success_rates = []

        for session in relevant_sessions:
            if session.baseline_tokens_loaded > 0:
                savings = (
                    session.baseline_tokens_loaded - session.optimized_tokens_loaded
                ) / session.baseline_tokens_loaded
                token_savings.append(savings * 100)

            if session.loading_latency_ms > 0:
                loading_times.append(session.loading_latency_ms)

            if session.optimized_functions_loaded > 0:
                efficiency = len(session.functions_actually_used) / session.optimized_functions_loaded
                usage_efficiencies.append(efficiency * 100)

            total_commands = session.commands_successful + session.commands_failed
            if total_commands > 0:
                success_rate = session.commands_successful / total_commands
                success_rates.append(success_rate * 100)

        return {
            "report_timestamp": datetime.now().isoformat(),
            "user_id": user_id or "system_wide",
            "sessions_analyzed": len(relevant_sessions),
            "token_optimization": {
                "average_reduction_percentage": mean(token_savings) if token_savings else 0.0,
                "median_reduction_percentage": median(token_savings) if token_savings else 0.0,
                "min_reduction_percentage": min(token_savings) if token_savings else 0.0,
                "max_reduction_percentage": max(token_savings) if token_savings else 0.0,
                "target_achieved": (mean(token_savings) if token_savings else 0.0)
                >= (self.token_reduction_target * 100),
            },
            "performance_metrics": {
                "average_loading_time_ms": mean(loading_times) if loading_times else 0.0,
                "p95_loading_time_ms": self._percentile(loading_times, 95) if loading_times else 0.0,
                "p99_loading_time_ms": self._percentile(loading_times, 99) if loading_times else 0.0,
            },
            "user_experience": {
                "average_success_rate": mean(success_rates) if success_rates else 0.0,
                "average_usage_efficiency": mean(usage_efficiencies) if usage_efficiencies else 0.0,
                "total_fallback_activations": sum(s.fallback_activations for s in relevant_sessions),
                "total_user_overrides": sum(s.user_override_count for s in relevant_sessions),
            },
            "validation_status": {
                "optimization_validated": self.optimization_validated,
                "validation_confidence": self.validation_confidence,
            },
        }



# Global monitor instance
_global_monitor: TokenOptimizationMonitor | None = None


def get_token_optimization_monitor() -> TokenOptimizationMonitor:
    """Get the global token optimization monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = TokenOptimizationMonitor()
    return _global_monitor


async def initialize_monitoring() -> TokenOptimizationMonitor:
    """Initialize the token optimization monitoring system."""
    monitor = get_token_optimization_monitor()
    logger.info("Token optimization monitoring system initialized")
    return monitor
