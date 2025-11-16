"""Unit tests for performance monitoring system.

This module contains comprehensive tests for the performance monitoring
components including metrics collection, SLA monitoring, and resource tracking.
"""

import asyncio
import time

import pytest

from src.utils.performance_monitor import (
    MetricData,
    MetricType,
    PerformanceMonitor,
    PerformanceTracker,
    SLAMonitor,
    SystemResourceMonitor,
    get_performance_monitor,
    get_sla_monitor,
    track_performance,
)


class TestMetricData:
    """Test cases for MetricData class."""

    def test_metric_data_creation(self):
        """Test MetricData creation with required fields."""
        metric = MetricData(name="test_metric", value=42.0, timestamp=time.time())

        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.timestamp > 0
        assert metric.labels == {}
        assert metric.metric_type == MetricType.GAUGE

    def test_metric_data_with_labels(self):
        """Test MetricData creation with labels."""
        labels = {"service": "test", "version": "1.0"}
        metric = MetricData(
            name="test_metric",
            value=42.0,
            timestamp=time.time(),
            labels=labels,
            metric_type=MetricType.COUNTER,
        )

        assert metric.labels == labels
        assert metric.metric_type == MetricType.COUNTER


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor(max_samples=100)

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.max_samples == 100
        assert len(self.monitor.metrics) == 0
        assert len(self.monitor.counters) == 0
        assert len(self.monitor.gauges) == 0
        assert len(self.monitor.timers) == 0

    def test_record_counter_metric(self):
        """Test recording counter metrics."""
        metric = MetricData(name="requests_total", value=1, timestamp=time.time(), metric_type=MetricType.COUNTER)

        self.monitor.record_metric(metric)
        self.monitor.record_metric(metric)

        assert self.monitor.get_counter("requests_total") == 2

    def test_record_gauge_metric(self):
        """Test recording gauge metrics."""
        metric = MetricData(name="memory_usage", value=512.5, timestamp=time.time(), metric_type=MetricType.GAUGE)

        self.monitor.record_metric(metric)
        assert self.monitor.get_gauge("memory_usage") == 512.5

        # Update gauge value
        metric.value = 1024.0
        self.monitor.record_metric(metric)
        assert self.monitor.get_gauge("memory_usage") == 1024.0

    def test_record_histogram_metric(self):
        """Test recording histogram metrics."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for value in values:
            metric = MetricData(
                name="response_time",
                value=value,
                timestamp=time.time(),
                metric_type=MetricType.HISTOGRAM,
            )
            self.monitor.record_metric(metric)

        stats = self.monitor.get_histogram_stats("response_time")
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0

    def test_record_timer_metric(self):
        """Test recording timer metrics."""
        times = [0.1, 0.2, 0.15, 0.3, 0.25]

        for time_val in times:
            metric = MetricData(
                name="process_time",
                value=time_val,
                timestamp=time.time(),
                metric_type=MetricType.TIMER,
            )
            self.monitor.record_metric(metric)

        stats = self.monitor.get_timer_stats("process_time")
        assert stats["count"] == 5
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3
        assert stats["mean"] == 0.2

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        values = list(range(1, 101))  # 1 to 100

        for value in values:
            metric = MetricData(
                name="test_values",
                value=value,
                timestamp=time.time(),
                metric_type=MetricType.HISTOGRAM,
            )
            self.monitor.record_metric(metric)

        stats = self.monitor.get_histogram_stats("test_values")
        assert stats["p95"] == 95
        assert stats["p99"] == 99

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        # Add different types of metrics
        counter_metric = MetricData(name="counter_test", value=5, timestamp=time.time(), metric_type=MetricType.COUNTER)

        gauge_metric = MetricData(name="gauge_test", value=42.0, timestamp=time.time(), metric_type=MetricType.GAUGE)

        self.monitor.record_metric(counter_metric)
        self.monitor.record_metric(gauge_metric)

        all_metrics = self.monitor.get_all_metrics()

        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "timers" in all_metrics

        assert all_metrics["counters"]["counter_test"] == 5
        assert all_metrics["gauges"]["gauge_test"] == 42.0

    def test_reset_metrics(self):
        """Test resetting all metrics."""
        metric = MetricData(name="test_metric", value=42.0, timestamp=time.time(), metric_type=MetricType.COUNTER)

        self.monitor.record_metric(metric)
        assert self.monitor.get_counter("test_metric") == 42.0

        self.monitor.reset_metrics()
        assert self.monitor.get_counter("test_metric") == 0


class TestSLAMonitor:
    """Test cases for SLAMonitor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sla_monitor = SLAMonitor()

    def test_sla_monitor_initialization(self):
        """Test SLA monitor initialization."""
        assert "response_time_p95" in self.sla_monitor.sla_targets
        assert "response_time_p99" in self.sla_monitor.sla_targets
        assert "success_rate" in self.sla_monitor.sla_targets
        assert "availability" in self.sla_monitor.sla_targets

    def test_sla_monitor_custom_targets(self):
        """Test SLA monitor with custom targets."""
        custom_targets = {
            "response_time_p95": 1.0,
            "success_rate": 0.95,
        }

        sla_monitor = SLAMonitor(custom_targets)
        assert sla_monitor.sla_targets["response_time_p95"] == 1.0
        assert sla_monitor.sla_targets["success_rate"] == 0.95

    def test_check_sla_compliance_success(self):
        """Test SLA compliance checking with compliant metrics."""
        metrics = {
            "counters": {
                "total_requests": 100,
                "successful_requests": 99,
            },
            "timers": {
                "response_time": {
                    "p95": 1.5,
                    "p99": 2.5,
                },
            },
        }

        compliance_status = self.sla_monitor.check_sla_compliance(metrics)

        assert compliance_status["response_time_p95"]["compliant"] is True
        assert compliance_status["response_time_p99"]["compliant"] is True
        assert compliance_status["success_rate"]["compliant"] is True

    def test_check_sla_compliance_violations(self):
        """Test SLA compliance checking with violations."""
        metrics = {
            "counters": {
                "total_requests": 100,
                "successful_requests": 90,  # 90% success rate (below 99% target)
            },
            "timers": {
                "response_time": {
                    "p95": 3.0,  # Above 2.0s target
                    "p99": 6.0,  # Above 5.0s target
                },
            },
        }

        compliance_status = self.sla_monitor.check_sla_compliance(metrics)

        assert compliance_status["response_time_p95"]["compliant"] is False
        assert compliance_status["response_time_p99"]["compliant"] is False
        assert compliance_status["success_rate"]["compliant"] is False

        # Check that violations were recorded
        violations = self.sla_monitor.get_violations()
        assert len(violations) == 3

    def test_get_violations_with_time_filter(self):
        """Test getting violations with time filter."""
        # Create test metrics that violate SLA
        metrics = {
            "counters": {
                "total_requests": 100,
                "successful_requests": 90,
            },
            "timers": {
                "response_time": {
                    "p95": 3.0,
                    "p99": 6.0,
                },
            },
        }

        # Record violations
        self.sla_monitor.check_sla_compliance(metrics)

        # Get violations since current time (should be empty)
        recent_violations = self.sla_monitor.get_violations(since=time.time())
        assert len(recent_violations) == 0

        # Get all violations
        all_violations = self.sla_monitor.get_violations()
        assert len(all_violations) == 3

    def test_clear_violations(self):
        """Test clearing violations."""
        # Create violations
        metrics = {
            "counters": {
                "total_requests": 100,
                "successful_requests": 90,
            },
            "timers": {
                "response_time": {
                    "p95": 3.0,
                    "p99": 6.0,
                },
            },
        }

        self.sla_monitor.check_sla_compliance(metrics)
        assert len(self.sla_monitor.get_violations()) == 3

        self.sla_monitor.clear_violations()
        assert len(self.sla_monitor.get_violations()) == 0


class TestPerformanceTracker:
    """Test cases for PerformanceTracker class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor()

    def test_performance_tracker_success(self):
        """Test performance tracker with successful operation."""
        with PerformanceTracker(self.monitor, "test_operation"):
            time.sleep(0.01)  # Simulate work

        # Check that metrics were recorded
        timer_stats = self.monitor.get_timer_stats("test_operation_duration")
        assert timer_stats["count"] == 1
        assert timer_stats["min"] > 0

        success_count = self.monitor.get_counter("test_operation_success")
        assert success_count == 1

    def test_performance_tracker_failure(self):
        """Test performance tracker with failed operation."""
        try:
            with PerformanceTracker(self.monitor, "test_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Check that error metric was recorded
        error_count = self.monitor.get_counter("test_operation_error")
        assert error_count == 1

        # Duration should still be recorded
        timer_stats = self.monitor.get_timer_stats("test_operation_duration")
        assert timer_stats["count"] == 1

    def test_performance_tracker_with_labels(self):
        """Test performance tracker with labels."""
        labels = {"service": "test", "version": "1.0"}

        with PerformanceTracker(self.monitor, "test_operation", labels):
            time.sleep(0.01)

        # Check that metrics exist (labels are stored in history)
        timer_stats = self.monitor.get_timer_stats("test_operation_duration")
        assert timer_stats["count"] == 1


class TestSystemResourceMonitor:
    """Test cases for SystemResourceMonitor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor()
        self.resource_monitor = SystemResourceMonitor(self.monitor)

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring."""
        # Start monitoring
        await self.resource_monitor.start_monitoring(interval=0.1)

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop monitoring
        await self.resource_monitor.stop_monitoring()

        # Check that metrics were collected
        memory_usage = self.monitor.get_gauge("memory_usage_mb")
        assert memory_usage >= 0

    @pytest.mark.asyncio
    async def test_monitoring_already_started(self):
        """Test starting monitoring when already started."""
        await self.resource_monitor.start_monitoring(interval=0.1)

        # Starting again should not cause issues
        await self.resource_monitor.start_monitoring(interval=0.1)

        await self.resource_monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stopping_not_started_monitoring(self):
        """Test stopping monitoring when not started."""
        # Should not cause issues
        await self.resource_monitor.stop_monitoring()


class TestGlobalFunctions:
    """Test cases for global functions."""

    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor = get_performance_monitor()
        assert isinstance(monitor, PerformanceMonitor)

        # Should return same instance
        monitor2 = get_performance_monitor()
        assert monitor is monitor2

    def test_get_sla_monitor(self):
        """Test getting global SLA monitor."""
        sla_monitor = get_sla_monitor()
        assert isinstance(sla_monitor, SLAMonitor)

    def test_track_performance(self):
        """Test track_performance function."""
        tracker = track_performance("test_operation")
        assert isinstance(tracker, PerformanceTracker)
        assert tracker.operation_name == "test_operation"

    def test_track_performance_with_labels(self):
        """Test track_performance function with labels."""
        labels = {"service": "test"}
        tracker = track_performance("test_operation", labels)
        assert tracker.labels == labels


class TestMetricType:
    """Test cases for MetricType enum."""

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"
