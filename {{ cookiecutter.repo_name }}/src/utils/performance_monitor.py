"""Performance monitoring system for C.R.E.A.T.E. framework.

This module provides comprehensive performance monitoring, metrics collection,
and SLA compliance tracking for the C.R.E.A.T.E. framework.
"""

import asyncio
import contextlib
import logging
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, median
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricData:
    """Container for metric data."""

    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring."""

    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    min_processing_time: float = float("inf")
    max_processing_time: float = 0.0
    p95_processing_time: float = 0.0
    p99_processing_time: float = 0.0
    current_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    active_connections: int = 0


class PerformanceMonitor:
    """Core performance monitoring system."""

    def __init__(self, max_samples: int = 1000) -> None:
        """Initialize performance monitor.

        Args:
            max_samples: Maximum number of samples to keep in memory.
        """
        self.max_samples = max_samples
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.counters: dict[str, int] = defaultdict(int)
        self.gauges: dict[str, float] = defaultdict(float)
        self.timers: dict[str, list[float]] = defaultdict(list)
        self.logger = logger

    def record_metric(self, metric: MetricData) -> None:
        """Record a metric.

        Args:
            metric: Metric data to record.
        """
        timestamp = metric.timestamp or time.time()

        if metric.metric_type == MetricType.COUNTER:
            self.counters[metric.name] += int(metric.value)
        elif metric.metric_type == MetricType.GAUGE:
            self.gauges[metric.name] = metric.value
        elif metric.metric_type == MetricType.HISTOGRAM:
            self.metrics[metric.name].append(metric.value)
        elif metric.metric_type == MetricType.TIMER:
            self.timers[metric.name].append(metric.value)

        # Keep general metrics history
        self.metrics[f"{metric.name}_history"].append(
            {"value": metric.value, "timestamp": timestamp, "labels": metric.labels},
        )

        self.logger.debug("Recorded metric: %s = %f at %f", metric.name, metric.value, timestamp)

    def get_counter(self, name: str) -> int:
        """Get counter value.

        Args:
            name: Counter name.

        Returns:
            Counter value.
        """
        return self.counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get gauge value.

        Args:
            name: Gauge name.

        Returns:
            Gauge value.
        """
        return self.gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics.

        Args:
            name: Histogram name.

        Returns:
            Dictionary with histogram statistics.
        """
        values = list(self.metrics[name])
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "median": median(values),
            "p95": self._percentile(sorted_values, 95),
            "p99": self._percentile(sorted_values, 99),
        }

    def get_timer_stats(self, name: str) -> dict[str, float]:
        """Get timer statistics.

        Args:
            name: Timer name.

        Returns:
            Dictionary with timer statistics.
        """
        values = self.timers.get(name, [])
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "median": median(values),
            "p95": self._percentile(sorted_values, 95),
            "p99": self._percentile(sorted_values, 99),
        }

    def _percentile(self, sorted_values: list[float], percentile: float) -> float:
        """Calculate percentile from sorted values.

        Args:
            sorted_values: Sorted list of values.
            percentile: Percentile to calculate (0-100).

        Returns:
            Percentile value.
        """
        if not sorted_values:
            return 0.0

        index = int((percentile / 100) * (len(sorted_values) - 1))
        return sorted_values[index]

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all current metrics.

        Returns:
            Dictionary containing all metrics.
        """
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: self.get_histogram_stats(name) for name in self.metrics if not name.endswith("_history")
            },
            "timers": {name: self.get_timer_stats(name) for name in self.timers},
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.metrics.clear()
        self.timers.clear()
        self.logger.info("All metrics have been reset")


class SLAMonitor:
    """SLA compliance monitoring system."""

    def __init__(self, sla_targets: dict[str, float] | None = None) -> None:
        """Initialize SLA monitor.

        Args:
            sla_targets: Dictionary of SLA targets.
        """
        self.sla_targets = sla_targets or {
            "response_time_p95": 2.0,  # 95th percentile response time < 2s
            "response_time_p99": 5.0,  # 99th percentile response time < 5s
            "success_rate": 0.99,  # 99% success rate
            "availability": 0.999,  # 99.9% availability
        }
        self.violations: list[dict[str, Any]] = []
        self.logger = logger

    def check_sla_compliance(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Check SLA compliance against current metrics.

        Args:
            metrics: Current system metrics.

        Returns:
            Dictionary with SLA compliance status.
        """
        compliance_status = {}

        # Check response time SLAs
        timer_stats = metrics.get("timers", {})
        for timer_name, stats in timer_stats.items():
            if "response_time" in timer_name:
                p95_time = stats.get("p95", 0)
                p99_time = stats.get("p99", 0)

                compliance_status[f"{timer_name}_p95"] = {
                    "target": self.sla_targets["response_time_p95"],
                    "actual": p95_time,
                    "compliant": p95_time <= self.sla_targets["response_time_p95"],
                }

                compliance_status[f"{timer_name}_p99"] = {
                    "target": self.sla_targets["response_time_p99"],
                    "actual": p99_time,
                    "compliant": p99_time <= self.sla_targets["response_time_p99"],
                }

        # Check success rate SLA
        counters = metrics.get("counters", {})
        total_requests = counters.get("total_requests", 0)
        successful_requests = counters.get("successful_requests", 0)

        if total_requests > 0:
            success_rate = successful_requests / total_requests
            compliance_status["success_rate"] = {
                "target": self.sla_targets["success_rate"],
                "actual": success_rate,
                "compliant": success_rate >= self.sla_targets["success_rate"],
            }

        # Record violations
        for metric_name, status in compliance_status.items():
            if not status["compliant"]:
                violation = {
                    "metric": metric_name,
                    "target": status["target"],
                    "actual": status["actual"],
                    "timestamp": time.time(),
                }
                self.violations.append(violation)
                self.logger.warning(
                    "SLA violation detected: %s (target: %f, actual: %f)",
                    metric_name,
                    status["target"],
                    status["actual"],
                )

        return compliance_status

    def get_violations(self, since: float | None = None) -> list[dict[str, Any]]:
        """Get SLA violations.

        Args:
            since: Optional timestamp to filter violations.

        Returns:
            List of SLA violations.
        """
        if since is None:
            return self.violations

        return [v for v in self.violations if v["timestamp"] >= since]

    def clear_violations(self) -> None:
        """Clear all recorded violations."""
        self.violations.clear()
        self.logger.info("All SLA violations have been cleared")


class PerformanceTracker:
    """Context manager for tracking performance of operations."""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str, labels: dict[str, str] | None = None) -> None:
        """Initialize performance tracker.

        Args:
            monitor: Performance monitor instance.
            operation_name: Name of the operation being tracked.
            labels: Optional labels for the metric.
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceTracker":
        """Enter context manager."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self.start_time is not None:
            duration = time.time() - self.start_time

            # Record timing metric
            metric = MetricData(
                name=f"{self.operation_name}_duration",
                value=duration,
                timestamp=time.time(),
                labels=self.labels,
                metric_type=MetricType.TIMER,
            )
            self.monitor.record_metric(metric)

            # Record success/failure counters
            if exc_type is None:
                success_metric = MetricData(
                    name=f"{self.operation_name}_success",
                    value=1,
                    timestamp=time.time(),
                    labels=self.labels,
                    metric_type=MetricType.COUNTER,
                )
                self.monitor.record_metric(success_metric)
            else:
                error_metric = MetricData(
                    name=f"{self.operation_name}_error",
                    value=1,
                    timestamp=time.time(),
                    labels=self.labels,
                    metric_type=MetricType.COUNTER,
                )
                self.monitor.record_metric(error_metric)


class SystemResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self, monitor: PerformanceMonitor) -> None:
        """Initialize system resource monitor.

        Args:
            monitor: Performance monitor instance.
        """
        self.monitor = monitor
        self.logger = logger
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None

    async def start_monitoring(self, interval: float = 5.0) -> None:
        """Start system resource monitoring.

        Args:
            interval: Monitoring interval in seconds.
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        self.logger.info("Started system resource monitoring")

    async def stop_monitoring(self) -> None:
        """Stop system resource monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        self.logger.info("Stopped system resource monitoring")

    async def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop.

        Args:
            interval: Monitoring interval in seconds.
        """
        while self._monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop: %s", e)
                await asyncio.sleep(interval)

    async def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        timestamp = time.time()

        # Memory usage (simplified - in real implementation would use psutil)
        memory_usage = sys.getsizeof(self.monitor.metrics) / 1024 / 1024  # MB

        memory_metric = MetricData(
            name="memory_usage_mb",
            value=memory_usage,
            timestamp=timestamp,
            metric_type=MetricType.GAUGE,
        )
        self.monitor.record_metric(memory_metric)

        # CPU usage placeholder (would use psutil in real implementation)
        cpu_metric = MetricData(
            name="cpu_usage_percent",
            value=0.0,
            timestamp=timestamp,
            metric_type=MetricType.GAUGE,  # Placeholder
        )
        self.monitor.record_metric(cpu_metric)


# Global performance monitor instance
global_monitor = PerformanceMonitor()
global_sla_monitor = SLAMonitor()
global_resource_monitor = SystemResourceMonitor(global_monitor)


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance.

    Returns:
        Global performance monitor.
    """
    return global_monitor


def get_sla_monitor() -> SLAMonitor:
    """Get the global SLA monitor instance.

    Returns:
        Global SLA monitor.
    """
    return global_sla_monitor


def get_resource_monitor() -> SystemResourceMonitor:
    """Get the global resource monitor instance.

    Returns:
        Global resource monitor.
    """
    return global_resource_monitor


def track_performance(operation_name: str, labels: dict[str, str] | None = None) -> PerformanceTracker:
    """Create a performance tracker for an operation.

    Args:
        operation_name: Name of the operation.
        labels: Optional labels for the metric.

    Returns:
        Performance tracker context manager.
    """
    return PerformanceTracker(global_monitor, operation_name, labels)
