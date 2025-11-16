"""Unit tests for monitoring module functions directly."""

import time
from unittest.mock import Mock, patch

import pytest

from src.api.routers.monitoring import (
    HealthResponse,
    MetricsResponse,
    SLAComplianceResponse,
    SLAViolationResponse,
    clear_sla_violations,
    get_health,
    get_metrics,
    get_sla_compliance,
    get_sla_violations,
    get_system_resources,
    reset_metrics,
    start_resource_monitoring,
    stop_resource_monitoring,
)
from src.utils.performance_monitor import (
    PerformanceMonitor,
    SLAMonitor,
    SystemResourceMonitor,
)


@pytest.fixture
def mock_performance_monitor():
    """Create mock performance monitor."""
    mock = Mock(spec=PerformanceMonitor)
    mock.get_all_metrics.return_value = {
        "counters": {"api_requests": 100, "errors": 5},
        "gauges": {"memory_usage_mb": 512.5, "cpu_usage_percent": 45.2},
        "histograms": {"response_time": {"p50": 0.1, "p95": 0.5, "p99": 1.0}},
        "timers": {"query_duration": {"mean": 0.15, "max": 2.0, "min": 0.01}},
    }
    mock.reset_metrics.return_value = None
    return mock


@pytest.fixture
def mock_sla_monitor():
    """Create mock SLA monitor."""
    mock = Mock(spec=SLAMonitor)
    mock.check_sla_compliance.return_value = {
        "response_time": {"compliant": True, "target": 2.0, "actual": 0.5},
        "error_rate": {"compliant": True, "target": 0.05, "actual": 0.02},
        "memory_usage": {"compliant": False, "target": 500, "actual": 512.5},
    }
    mock.get_violations.return_value = [
        {
            "metric": "memory_usage",
            "timestamp": time.time() - 300,
            "target": 500,
            "actual": 512.5,
            "severity": "warning",
        },
    ]
    mock.clear_violations.return_value = None
    return mock


@pytest.fixture
def mock_resource_monitor():
    """Create mock resource monitor."""
    mock = Mock(spec=SystemResourceMonitor)
    mock.start_monitoring.return_value = None
    mock.stop_monitoring.return_value = None
    return mock


class TestMonitoringFunctions:
    """Test monitoring functions directly."""

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @pytest.mark.asyncio
    async def test_get_metrics(self, mock_get_monitor, mock_performance_monitor):
        """Test get_metrics function."""
        mock_get_monitor.return_value = mock_performance_monitor

        result = await get_metrics()

        assert isinstance(result, MetricsResponse)
        assert result.counters["api_requests"] == 100
        assert result.gauges["memory_usage_mb"] == 512.5
        assert result.histograms["response_time"]["p95"] == 0.5
        assert result.timers["query_duration"]["mean"] == 0.15
        assert result.timestamp > 0

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @patch("src.api.routers.monitoring.get_sla_monitor")
    @pytest.mark.asyncio
    async def test_get_sla_compliance(self, mock_get_sla, mock_get_perf, mock_performance_monitor, mock_sla_monitor):
        """Test get_sla_compliance function."""
        mock_get_perf.return_value = mock_performance_monitor
        mock_get_sla.return_value = mock_sla_monitor

        result = await get_sla_compliance()

        assert isinstance(result, SLAComplianceResponse)
        assert result.overall_compliant is False
        assert result.compliance_status["response_time"]["compliant"] is True
        assert result.compliance_status["memory_usage"]["compliant"] is False
        assert result.timestamp > 0

    @patch("src.api.routers.monitoring.get_sla_monitor")
    @pytest.mark.asyncio
    async def test_get_sla_violations(self, mock_get_sla, mock_sla_monitor):
        """Test get_sla_violations function."""
        mock_get_sla.return_value = mock_sla_monitor

        result = await get_sla_violations()

        assert isinstance(result, SLAViolationResponse)
        assert result.total_violations == 1
        assert len(result.violations) == 1
        assert result.violations[0]["metric"] == "memory_usage"
        assert result.timestamp > 0

    @patch("src.api.routers.monitoring.get_sla_monitor")
    @pytest.mark.asyncio
    async def test_get_sla_violations_with_since(self, mock_get_sla, mock_sla_monitor):
        """Test get_sla_violations with since parameter."""
        mock_get_sla.return_value = mock_sla_monitor
        since_time = time.time() - 600

        result = await get_sla_violations(since=since_time)

        mock_sla_monitor.get_violations.assert_called_once_with(since_time)
        assert isinstance(result, SLAViolationResponse)

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @patch("src.api.routers.monitoring.get_sla_monitor")
    @pytest.mark.asyncio
    async def test_get_health(self, mock_get_sla, mock_get_perf, mock_performance_monitor, mock_sla_monitor):
        """Test get_health function."""
        mock_get_perf.return_value = mock_performance_monitor
        mock_get_sla.return_value = mock_sla_monitor

        result = await get_health()

        assert isinstance(result, HealthResponse)
        assert result.status == "degraded"
        assert result.sla_compliant is False
        assert result.metrics_count == 6  # 2 counters + 2 gauges + 1 histogram + 1 timer
        assert result.timestamp > 0

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @pytest.mark.asyncio
    async def test_reset_metrics(self, mock_get_monitor, mock_performance_monitor):
        """Test reset_metrics function."""
        mock_get_monitor.return_value = mock_performance_monitor

        result = await reset_metrics()

        assert "message" in result
        assert result["message"] == "All metrics have been reset"
        assert "timestamp" in result
        mock_performance_monitor.reset_metrics.assert_called_once()

    @patch("src.api.routers.monitoring.get_sla_monitor")
    @pytest.mark.asyncio
    async def test_clear_sla_violations(self, mock_get_sla, mock_sla_monitor):
        """Test clear_sla_violations function."""
        mock_get_sla.return_value = mock_sla_monitor

        result = await clear_sla_violations()

        assert "message" in result
        assert result["message"] == "All SLA violations have been cleared"
        assert "timestamp" in result
        mock_sla_monitor.clear_violations.assert_called_once()

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @pytest.mark.asyncio
    async def test_get_system_resources(self, mock_get_monitor, mock_performance_monitor):
        """Test get_system_resources function."""
        mock_get_monitor.return_value = mock_performance_monitor

        result = await get_system_resources()

        assert result["memory_usage_mb"] == 512.5
        assert result["cpu_usage_percent"] == 45.2
        assert "timestamp" in result

    @patch("src.api.routers.monitoring.get_resource_monitor")
    @pytest.mark.asyncio
    async def test_start_resource_monitoring(self, mock_get_monitor, mock_resource_monitor):
        """Test start_resource_monitoring function."""
        mock_get_monitor.return_value = mock_resource_monitor

        result = await start_resource_monitoring()

        assert "message" in result
        assert result["message"] == "Resource monitoring started"
        assert "timestamp" in result

    @patch("src.api.routers.monitoring.get_resource_monitor")
    @pytest.mark.asyncio
    async def test_stop_resource_monitoring(self, mock_get_monitor, mock_resource_monitor):
        """Test stop_resource_monitoring function."""
        mock_get_monitor.return_value = mock_resource_monitor

        result = await stop_resource_monitoring()

        assert "message" in result
        assert result["message"] == "Resource monitoring stopped"
        assert "timestamp" in result


class TestEdgeCases:
    """Test edge cases for monitoring functions."""

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @pytest.mark.asyncio
    async def test_empty_metrics(self, mock_get_monitor):
        """Test with empty metrics."""
        mock_monitor = Mock(spec=PerformanceMonitor)
        mock_monitor.get_all_metrics.return_value = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {},
        }
        mock_get_monitor.return_value = mock_monitor

        result = await get_metrics()

        assert result.counters == {}
        assert result.gauges == {}
        assert result.histograms == {}
        assert result.timers == {}

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @patch("src.api.routers.monitoring.get_sla_monitor")
    @pytest.mark.asyncio
    async def test_all_sla_compliant(self, mock_get_sla, mock_get_perf, mock_performance_monitor):
        """Test when all SLAs are compliant."""
        mock_get_perf.return_value = mock_performance_monitor

        sla_monitor = Mock(spec=SLAMonitor)
        sla_monitor.check_sla_compliance.return_value = {
            "response_time": {"compliant": True},
            "error_rate": {"compliant": True},
            "memory_usage": {"compliant": True},
        }
        mock_get_sla.return_value = sla_monitor

        result = await get_sla_compliance()

        assert result.overall_compliant is True

    @patch("src.api.routers.monitoring.get_sla_monitor")
    @pytest.mark.asyncio
    async def test_no_violations(self, mock_get_sla):
        """Test when there are no SLA violations."""
        sla_monitor = Mock(spec=SLAMonitor)
        sla_monitor.get_violations.return_value = []
        mock_get_sla.return_value = sla_monitor

        result = await get_sla_violations()

        assert result.total_violations == 0
        assert result.violations == []

    @patch("src.api.routers.monitoring.get_performance_monitor")
    @pytest.mark.asyncio
    async def test_metrics_count_calculation(self, mock_get_monitor):
        """Test metrics count calculation in health endpoint."""
        mock_monitor = Mock(spec=PerformanceMonitor)
        mock_monitor.get_all_metrics.return_value = {
            "counters": {"a": 1, "b": 2, "c": 3},
            "gauges": {"d": 4, "e": 5},
            "histograms": {"f": {}},
            "timers": {},
        }
        mock_get_monitor.return_value = mock_monitor

        with patch("src.api.routers.monitoring.get_sla_monitor") as mock_get_sla:
            sla_monitor = Mock(spec=SLAMonitor)
            sla_monitor.check_sla_compliance.return_value = {}
            mock_get_sla.return_value = sla_monitor

            result = await get_health()

            assert result.metrics_count == 6  # 3 counters + 2 gauges + 1 histogram


class TestMonitoringModels:
    """Test monitoring response models."""

    def test_metrics_response_model(self):
        """Test MetricsResponse model."""
        response = MetricsResponse(
            counters={"test": 1},
            gauges={"test": 1.0},
            histograms={"test": {"p50": 0.5}},
            timers={"test": {"mean": 0.1}},
            timestamp=time.time(),
        )
        assert response.counters["test"] == 1
        assert response.gauges["test"] == 1.0

    def test_sla_compliance_response_model(self):
        """Test SLAComplianceResponse model."""
        response = SLAComplianceResponse(
            compliance_status={"test": {"compliant": True}},
            overall_compliant=True,
            timestamp=time.time(),
        )
        assert response.overall_compliant is True

    def test_sla_violation_response_model(self):
        """Test SLAViolationResponse model."""
        response = SLAViolationResponse(
            violations=[{"metric": "test"}],
            total_violations=1,
            timestamp=time.time(),
        )
        assert response.total_violations == 1

    def test_health_response_model(self):
        """Test HealthResponse model."""
        response = HealthResponse(
            status="healthy",
            uptime=100.0,
            metrics_count=10,
            sla_compliant=True,
            timestamp=time.time(),
        )
        assert response.status == "healthy"
        assert response.sla_compliant is True
