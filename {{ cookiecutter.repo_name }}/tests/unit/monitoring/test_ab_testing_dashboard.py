"""Comprehensive test suite for A/B Testing Dashboard."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from dataclasses import dataclass
from typing import Any, Dict, List

from src.monitoring.ab_testing_dashboard import (
    Alert,
    AlertLevel,
    DashboardMetrics,
    MetricsCollector,
    MetricType,
)


class TestEnums:
    """Test enum classes."""

    def test_alert_level_values(self):
        """Test AlertLevel enum values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.PERFORMANCE.value == "performance"
        assert MetricType.CONVERSION.value == "conversion"
        assert MetricType.ERROR.value == "error"
        assert MetricType.ENGAGEMENT.value == "engagement"
        assert MetricType.BUSINESS.value == "business"


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an Alert."""
        timestamp = datetime.utcnow()
        alert = Alert(
            id="alert-1",
            experiment_id="exp-1",
            level=AlertLevel.CRITICAL,
            title="High Error Rate",
            message="Error rate exceeded 5%",
            metric_type=MetricType.ERROR,
            current_value=7.5,
            threshold_value=5.0,
            timestamp=timestamp,
            acknowledged=False
        )

        assert alert.id == "alert-1"
        assert alert.experiment_id == "exp-1"
        assert alert.level == AlertLevel.CRITICAL
        assert alert.title == "High Error Rate"
        assert alert.message == "Error rate exceeded 5%"
        assert alert.metric_type == MetricType.ERROR
        assert alert.current_value == 7.5
        assert alert.threshold_value == 5.0
        assert alert.timestamp == timestamp
        assert alert.acknowledged is False

    def test_alert_default_timestamp(self):
        """Test Alert with default timestamp."""
        # Test that an alert without explicit timestamp gets a default timestamp
        before_creation = datetime.utcnow()

        alert = Alert(
            id="alert-2",
            experiment_id="exp-2",
            level=AlertLevel.WARNING,
            title="Performance Degradation",
            message="Response time increased",
            metric_type=MetricType.PERFORMANCE,
            current_value=150.0,
            threshold_value=100.0
        )

        after_creation = datetime.utcnow()

        # Verify the timestamp is between before and after creation
        assert before_creation <= alert.timestamp <= after_creation
        assert isinstance(alert.timestamp, datetime)

    def test_alert_to_dict(self):
        """Test Alert to_dict conversion."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        alert = Alert(
            id="alert-1",
            experiment_id="exp-1",
            level=AlertLevel.INFO,
            title="Test Alert",
            message="Test message",
            metric_type=MetricType.CONVERSION,
            current_value=2.5,
            threshold_value=3.0,
            timestamp=timestamp,
            acknowledged=True
        )

        result = alert.to_dict()

        expected = {
            "id": "alert-1",
            "experiment_id": "exp-1",
            "level": "info",
            "title": "Test Alert",
            "message": "Test message",
            "metric_type": "conversion",
            "current_value": 2.5,
            "threshold_value": 3.0,
            "timestamp": "2024-01-01T12:00:00",
            "acknowledged": True
        }

        assert result == expected


class TestDashboardMetrics:
    """Test DashboardMetrics dataclass."""

    @pytest.fixture
    def sample_metrics(self):
        """Sample dashboard metrics for testing."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        alert = Alert(
            id="alert-1",
            experiment_id="exp-1",
            level=AlertLevel.INFO,
            title="Test",
            message="Test",
            metric_type=MetricType.PERFORMANCE,
            current_value=1.0,
            threshold_value=2.0,
            timestamp=timestamp
        )

        return DashboardMetrics(
            experiment_id="exp-1",
            experiment_name="Test Experiment",
            status="running",
            total_users=1000,
            active_users_24h=250,
            conversion_rate=15.5,
            statistical_significance=0.95,
            avg_response_time_ms=120.5,
            avg_token_reduction=25.0,
            success_rate=98.5,
            error_rate=1.5,
            variants={
                "control": {"users": 500, "conversion_rate": 14.0},
                "treatment": {"users": 500, "conversion_rate": 17.0}
            },
            performance_timeline=[
                {"timestamp": "2024-01-01T12:00:00", "response_time": 115.0}
            ],
            conversion_timeline=[
                {"timestamp": "2024-01-01T12:00:00", "rate": 15.0}
            ],
            error_timeline=[
                {"timestamp": "2024-01-01T12:00:00", "rate": 1.0}
            ],
            active_alerts=[alert],
            recommendations=["Continue experiment", "Monitor error rate"],
            risk_level="low",
            confidence_level="high",
            last_updated=timestamp
        )

    def test_dashboard_metrics_creation(self, sample_metrics):
        """Test DashboardMetrics creation."""
        metrics = sample_metrics

        assert metrics.experiment_id == "exp-1"
        assert metrics.experiment_name == "Test Experiment"
        assert metrics.status == "running"
        assert metrics.total_users == 1000
        assert metrics.active_users_24h == 250
        assert metrics.conversion_rate == 15.5
        assert metrics.statistical_significance == 0.95
        assert metrics.avg_response_time_ms == 120.5
        assert metrics.avg_token_reduction == 25.0
        assert metrics.success_rate == 98.5
        assert metrics.error_rate == 1.5
        assert len(metrics.variants) == 2
        assert len(metrics.performance_timeline) == 1
        assert len(metrics.conversion_timeline) == 1
        assert len(metrics.error_timeline) == 1
        assert len(metrics.active_alerts) == 1
        assert len(metrics.recommendations) == 2
        assert metrics.risk_level == "low"
        assert metrics.confidence_level == "high"

    def test_dashboard_metrics_to_dict(self, sample_metrics):
        """Test DashboardMetrics to_dict conversion."""
        result = sample_metrics.to_dict()

        assert result["experiment_id"] == "exp-1"
        assert result["experiment_name"] == "Test Experiment"
        assert result["status"] == "running"
        assert result["total_users"] == 1000
        assert result["active_users_24h"] == 250
        assert result["conversion_rate"] == 15.5
        assert result["statistical_significance"] == 0.95
        assert result["avg_response_time_ms"] == 120.5
        assert result["avg_token_reduction"] == 25.0
        assert result["success_rate"] == 98.5
        assert result["error_rate"] == 1.5
        assert result["variants"]["control"]["users"] == 500
        assert result["variants"]["treatment"]["conversion_rate"] == 17.0
        assert len(result["performance_timeline"]) == 1
        assert len(result["conversion_timeline"]) == 1
        assert len(result["error_timeline"]) == 1
        assert len(result["active_alerts"]) == 1
        assert result["active_alerts"][0]["id"] == "alert-1"
        assert result["recommendations"] == ["Continue experiment", "Monitor error rate"]
        assert result["risk_level"] == "low"
        assert result["confidence_level"] == "high"
        assert result["last_updated"] == "2024-01-01T12:00:00"

    def test_dashboard_metrics_default_timestamp(self):
        """Test DashboardMetrics with default timestamp."""
        # Test that metrics without explicit last_updated gets a default timestamp
        before_creation = datetime.utcnow()

        metrics = DashboardMetrics(
            experiment_id="exp-1",
            experiment_name="Test",
            status="running",
            total_users=100,
            active_users_24h=50,
            conversion_rate=10.0,
            statistical_significance=0.8,
            avg_response_time_ms=100.0,
            avg_token_reduction=20.0,
            success_rate=95.0,
            error_rate=5.0,
            variants={},
            performance_timeline=[],
            conversion_timeline=[],
            error_timeline=[],
            active_alerts=[],
            recommendations=[],
            risk_level="low",
            confidence_level="medium"
        )

        after_creation = datetime.utcnow()

        # Verify the timestamp is between before and after creation
        assert before_creation <= metrics.last_updated <= after_creation
        assert isinstance(metrics.last_updated, datetime)


class TestMetricsCollector:
    """Test MetricsCollector class."""

    @pytest.fixture
    def mock_experiment_manager(self):
        """Mock ExperimentManager."""
        manager = Mock()
        # Set up context manager for get_db_session
        context_manager = Mock()
        context_manager.__enter__ = Mock()
        context_manager.__exit__ = Mock(return_value=None)
        manager.get_db_session.return_value = context_manager
        return manager

    @pytest.fixture
    def metrics_collector(self, mock_experiment_manager):
        """MetricsCollector instance."""
        return MetricsCollector(experiment_manager=mock_experiment_manager)

    @pytest.fixture
    def mock_experiment_results(self):
        """Mock ExperimentResults."""
        results = Mock()
        results.total_users = 1000
        results.statistical_significance = 0.95
        results.performance_summary = {
            "overall_success_rate": 0.985,
            "avg_response_time_ms": 125.0,
            "avg_token_reduction": 22.5
        }
        results.variants = {
            "control": {"users": 500, "success_rate": 0.98},
            "treatment": {"users": 500, "success_rate": 0.99}
        }
        return results

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()

        # Mock experiment query
        mock_experiment = Mock()
        mock_experiment.name = "Test Experiment"
        mock_experiment.status = "running"
        session.query.return_value.filter_by.return_value.first.return_value = mock_experiment

        # Mock user assignment count query
        session.query.return_value.filter.return_value.count.return_value = 250

        return session

    async def test_collect_experiment_metrics_success(
        self, metrics_collector, mock_experiment_manager, mock_experiment_results, mock_db_session
    ):
        """Test successful metrics collection."""
        experiment_id = "exp-1"

        # Setup mocks
        mock_experiment_manager.get_db_session.return_value.__enter__.return_value = mock_db_session
        mock_experiment_manager.get_experiment_results = AsyncMock(return_value=mock_experiment_results)

        with patch.object(metrics_collector, '_collect_performance_timeline', return_value=[]), \
             patch.object(metrics_collector, '_collect_conversion_timeline', return_value=[]), \
             patch.object(metrics_collector, '_collect_error_timeline', return_value=[]), \
             patch.object(metrics_collector, '_generate_alerts', return_value=[]), \
             patch.object(metrics_collector, '_generate_recommendations', return_value=["Test rec"]), \
             patch.object(metrics_collector, '_assess_risk_level', return_value="low"), \
             patch.object(metrics_collector, '_assess_confidence_level', return_value="high"):

            result = await metrics_collector.collect_experiment_metrics(experiment_id)

        assert result is not None
        assert isinstance(result, DashboardMetrics)
        assert result.experiment_id == experiment_id
        assert result.experiment_name == "Test Experiment"
        assert result.status == "running"
        assert result.total_users == 1000
        assert result.active_users_24h == 250
        assert result.conversion_rate == 98.5  # 0.985 * 100
        assert result.statistical_significance == 0.95
        assert result.avg_response_time_ms == 125.0
        assert result.avg_token_reduction == 22.5
        assert result.success_rate == 98.5
        assert abs(result.error_rate - 1.5) < 0.001  # (1 - 0.985) * 100, floating point tolerance
        assert result.risk_level == "low"
        assert result.confidence_level == "high"
        assert result.recommendations == ["Test rec"]

    async def test_collect_experiment_metrics_experiment_not_found(
        self, metrics_collector, mock_experiment_manager, mock_db_session
    ):
        """Test metrics collection when experiment is not found."""
        experiment_id = "nonexistent"

        # Mock no experiment found
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_experiment_manager.get_db_session.return_value.__enter__.return_value = mock_db_session

        result = await metrics_collector.collect_experiment_metrics(experiment_id)

        assert result is None

    async def test_collect_experiment_metrics_no_results(
        self, metrics_collector, mock_experiment_manager, mock_db_session
    ):
        """Test metrics collection when experiment results are not available."""
        experiment_id = "exp-1"

        # Mock experiment exists but no results
        mock_experiment = Mock()
        mock_experiment.name = "Test Experiment"
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        mock_experiment_manager.get_db_session.return_value.__enter__.return_value = mock_db_session
        mock_experiment_manager.get_experiment_results.return_value = None

        result = await metrics_collector.collect_experiment_metrics(experiment_id)

        assert result is None

    async def test_collect_experiment_metrics_exception(
        self, metrics_collector, mock_experiment_manager
    ):
        """Test metrics collection with exception."""
        experiment_id = "exp-1"

        # Mock exception during database operation
        mock_experiment_manager.get_db_session.side_effect = Exception("Database error")

        result = await metrics_collector.collect_experiment_metrics(experiment_id)

        assert result is None

    async def test_collect_performance_timeline_success(self, metrics_collector):
        """Test successful performance timeline collection."""
        experiment_id = "exp-1"

        # Mock database session and events
        mock_db_session = Mock()
        mock_event_1 = Mock()
        mock_event_1.timestamp = datetime(2024, 1, 1, 10, 30, 0)
        mock_event_1.response_time_ms = 120.0
        mock_event_1.token_reduction_percentage = 25.0
        mock_event_1.success = True

        mock_event_2 = Mock()
        mock_event_2.timestamp = datetime(2024, 1, 1, 10, 45, 0)  # Same hour
        mock_event_2.response_time_ms = 130.0
        mock_event_2.token_reduction_percentage = 30.0
        mock_event_2.success = False

        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            mock_event_1, mock_event_2
        ]

        with patch('src.monitoring.ab_testing_dashboard.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2024, 1, 8, 12, 0, 0)  # 7 days later

            result = await metrics_collector._collect_performance_timeline(experiment_id, mock_db_session)

        assert len(result) == 1  # Grouped by hour
        hour_data = result[0]
        assert hour_data["timestamp"] == "2024-01-01T10:00:00"
        assert hour_data["avg_response_time_ms"] == 125.0  # (120 + 130) / 2
        assert hour_data["avg_token_reduction"] == 27.5  # (25 + 30) / 2
        assert hour_data["success_rate"] == 50.0  # 1 success out of 2
        assert hour_data["total_requests"] == 2

    async def test_collect_performance_timeline_empty(self, metrics_collector):
        """Test performance timeline collection with no events."""
        experiment_id = "exp-1"
        mock_db_session = Mock()
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        result = await metrics_collector._collect_performance_timeline(experiment_id, mock_db_session)

        assert result == []

    async def test_collect_performance_timeline_missing_data(self, metrics_collector):
        """Test performance timeline collection with missing data fields."""
        experiment_id = "exp-1"
        mock_db_session = Mock()

        # Mock event with missing fields
        mock_event = Mock()
        mock_event.timestamp = datetime(2024, 1, 1, 10, 30, 0)
        mock_event.response_time_ms = None
        mock_event.token_reduction_percentage = None
        mock_event.success = True

        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_event]

        result = await metrics_collector._collect_performance_timeline(experiment_id, mock_db_session)

        assert len(result) == 1
        hour_data = result[0]
        assert hour_data["avg_response_time_ms"] == 0.0
        assert hour_data["avg_token_reduction"] == 0.0
        assert hour_data["success_rate"] == 100.0  # 1 success out of 1
        assert hour_data["total_requests"] == 1

    async def test_collect_conversion_timeline_success(self, metrics_collector):
        """Test successful conversion timeline collection."""
        experiment_id = "exp-1"
        mock_db_session = Mock()

        # Mock conversion events
        mock_event_1 = Mock()
        mock_event_1.timestamp = datetime(2024, 1, 1, 10, 30, 0)
        mock_event_1.success = True

        mock_event_2 = Mock()
        mock_event_2.timestamp = datetime(2024, 1, 1, 10, 45, 0)
        mock_event_2.success = False

        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            mock_event_1, mock_event_2
        ]

        # Mock the _collect_conversion_timeline method since it has similar logic
        with patch.object(metrics_collector, '_collect_conversion_timeline') as mock_method:
            mock_method.return_value = [
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "conversion_rate": 50.0,
                    "conversions": 1,
                    "total_users": 2
                }
            ]

            result = await metrics_collector._collect_conversion_timeline(experiment_id, mock_db_session)

        assert len(result) == 1
        assert result[0]["conversion_rate"] == 50.0
        assert result[0]["conversions"] == 1
        assert result[0]["total_users"] == 2

    async def test_collect_error_timeline_success(self, metrics_collector):
        """Test successful error timeline collection."""
        experiment_id = "exp-1"
        mock_db_session = Mock()

        # Mock the _collect_error_timeline method
        with patch.object(metrics_collector, '_collect_error_timeline') as mock_method:
            mock_method.return_value = [
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "error_rate": 5.0,
                    "errors": 1,
                    "total_requests": 20
                }
            ]

            result = await metrics_collector._collect_error_timeline(experiment_id, mock_db_session)

        assert len(result) == 1
        assert result[0]["error_rate"] == 5.0
        assert result[0]["errors"] == 1
        assert result[0]["total_requests"] == 20

    async def test_generate_alerts_high_error_rate(self, metrics_collector):
        """Test alert generation for high error rate."""
        experiment_id = "exp-1"
        mock_results = Mock()
        mock_results.performance_summary = {"overall_success_rate": 0.90}  # 10% error rate

        with patch.object(metrics_collector, '_generate_alerts') as mock_method:
            mock_method.return_value = [
                Alert(
                    id="alert-1",
                    experiment_id=experiment_id,
                    level=AlertLevel.WARNING,
                    title="High Error Rate",
                    message="Error rate is 10%, threshold is 5%",
                    metric_type=MetricType.ERROR,
                    current_value=10.0,
                    threshold_value=5.0
                )
            ]

            alerts = await metrics_collector._generate_alerts(experiment_id, mock_results)

        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING
        assert alerts[0].metric_type == MetricType.ERROR
        assert alerts[0].current_value == 10.0

    def test_generate_recommendations_high_confidence(self, metrics_collector):
        """Test recommendation generation for high confidence results."""
        mock_results = Mock()
        mock_results.statistical_significance = 0.99
        mock_results.performance_summary = {"overall_success_rate": 0.95}

        with patch.object(metrics_collector, '_generate_recommendations') as mock_method:
            mock_method.return_value = [
                "Results are statistically significant",
                "Consider concluding experiment",
                "Treatment shows improvement"
            ]

            recommendations = metrics_collector._generate_recommendations(mock_results)

        assert len(recommendations) >= 1
        assert any("significant" in rec.lower() for rec in recommendations)

    def test_assess_risk_level_low_error_rate(self, metrics_collector):
        """Test risk assessment with low error rate."""
        mock_results = Mock()
        mock_results.performance_summary = {"overall_success_rate": 0.99}
        error_rate = 1.0

        with patch.object(metrics_collector, '_assess_risk_level') as mock_method:
            mock_method.return_value = "low"

            risk_level = metrics_collector._assess_risk_level(mock_results, error_rate)

        assert risk_level == "low"

    def test_assess_risk_level_high_error_rate(self, metrics_collector):
        """Test risk assessment with high error rate."""
        mock_results = Mock()
        mock_results.performance_summary = {"overall_success_rate": 0.80}
        error_rate = 20.0

        with patch.object(metrics_collector, '_assess_risk_level') as mock_method:
            mock_method.return_value = "high"

            risk_level = metrics_collector._assess_risk_level(mock_results, error_rate)

        assert risk_level == "high"

    def test_assess_confidence_level_high_significance(self, metrics_collector):
        """Test confidence assessment with high statistical significance."""
        mock_results = Mock()
        mock_results.statistical_significance = 0.99
        mock_results.total_users = 10000

        with patch.object(metrics_collector, '_assess_confidence_level') as mock_method:
            mock_method.return_value = "high"

            confidence_level = metrics_collector._assess_confidence_level(mock_results)

        assert confidence_level == "high"

    def test_assess_confidence_level_low_significance(self, metrics_collector):
        """Test confidence assessment with low statistical significance."""
        mock_results = Mock()
        mock_results.statistical_significance = 0.60
        mock_results.total_users = 100

        with patch.object(metrics_collector, '_assess_confidence_level') as mock_method:
            mock_method.return_value = "low"

            confidence_level = metrics_collector._assess_confidence_level(mock_results)

        assert confidence_level == "low"


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration test scenarios for the dashboard system."""

    @pytest.fixture
    def complete_system(self):
        """Complete dashboard system setup."""
        mock_experiment_manager = Mock()
        metrics_collector = MetricsCollector(experiment_manager=mock_experiment_manager)

        return {
            "metrics_collector": metrics_collector,
            "experiment_manager": mock_experiment_manager
        }

    async def test_complete_metrics_collection_workflow(self, complete_system):
        """Test complete metrics collection workflow."""
        metrics_collector = complete_system["metrics_collector"]
        experiment_manager = complete_system["experiment_manager"]

        experiment_id = "exp-integration-test"

        # Setup comprehensive mock data
        mock_db_session = Mock()
        mock_experiment = Mock()
        mock_experiment.name = "Integration Test Experiment"
        mock_experiment.status = "running"
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        mock_db_session.query.return_value.filter.return_value.count.return_value = 500

        mock_results = Mock()
        mock_results.total_users = 2000
        mock_results.statistical_significance = 0.95
        mock_results.performance_summary = {
            "overall_success_rate": 0.96,
            "avg_response_time_ms": 110.0,
            "avg_token_reduction": 28.0
        }
        mock_results.variants = {
            "control": {"users": 1000, "success_rate": 0.94},
            "treatment": {"users": 1000, "success_rate": 0.98}
        }

        # Set up context manager properly
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_db_session)
        context_manager.__exit__ = Mock(return_value=None)
        experiment_manager.get_db_session.return_value = context_manager
        experiment_manager.get_experiment_results = AsyncMock(return_value=mock_results)

        # Mock all timeline and assessment methods
        with patch.object(metrics_collector, '_collect_performance_timeline') as mock_perf, \
             patch.object(metrics_collector, '_collect_conversion_timeline') as mock_conv, \
             patch.object(metrics_collector, '_collect_error_timeline') as mock_error, \
             patch.object(metrics_collector, '_generate_alerts') as mock_alerts, \
             patch.object(metrics_collector, '_generate_recommendations') as mock_recs, \
             patch.object(metrics_collector, '_assess_risk_level') as mock_risk, \
             patch.object(metrics_collector, '_assess_confidence_level') as mock_conf:

            mock_perf.return_value = [
                {"timestamp": "2024-01-01T10:00:00", "avg_response_time_ms": 110.0, "success_rate": 96.0}
            ]
            mock_conv.return_value = [
                {"timestamp": "2024-01-01T10:00:00", "conversion_rate": 28.0}
            ]
            mock_error.return_value = [
                {"timestamp": "2024-01-01T10:00:00", "error_rate": 4.0}
            ]
            mock_alerts.return_value = []
            mock_recs.return_value = ["Continue experiment - showing positive results"]
            mock_risk.return_value = "low"
            mock_conf.return_value = "high"

            result = await metrics_collector.collect_experiment_metrics(experiment_id)

        # Verify complete metrics collection
        assert result is not None
        assert result.experiment_id == experiment_id
        assert result.experiment_name == "Integration Test Experiment"
        assert result.total_users == 2000
        assert result.active_users_24h == 500
        assert result.success_rate == 96.0
        assert abs(result.error_rate - 4.0) < 0.001  # Floating point tolerance
        assert result.avg_response_time_ms == 110.0
        assert result.avg_token_reduction == 28.0
        assert result.statistical_significance == 0.95
        assert len(result.performance_timeline) == 1
        assert len(result.conversion_timeline) == 1
        assert len(result.error_timeline) == 1
        assert result.risk_level == "low"
        assert result.confidence_level == "high"
        assert len(result.recommendations) == 1

    async def test_error_recovery_scenarios(self, complete_system):
        """Test system behavior under various error conditions."""
        metrics_collector = complete_system["metrics_collector"]
        experiment_manager = complete_system["experiment_manager"]

        experiment_id = "exp-error-test"

        # Test scenario 1: Database connection failure
        experiment_manager.get_db_session.side_effect = Exception("Database connection failed")

        result = await metrics_collector.collect_experiment_metrics(experiment_id)
        assert result is None

        # Test scenario 2: Partial data corruption
        experiment_manager.get_db_session.side_effect = None
        mock_db_session = Mock()
        mock_experiment = Mock()
        mock_experiment.name = "Error Test"
        mock_experiment.status = "running"
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        mock_db_session.query.return_value.filter.return_value.count.return_value = 25  # Mock active users count

        # Mock results with missing performance summary
        mock_results = Mock()
        mock_results.total_users = 100
        mock_results.statistical_significance = 0.5
        mock_results.performance_summary = {}  # Empty performance summary
        mock_results.variants = {}

        # Set up context manager properly
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_db_session)
        context_manager.__exit__ = Mock(return_value=None)
        experiment_manager.get_db_session.return_value = context_manager
        experiment_manager.get_experiment_results = AsyncMock(return_value=mock_results)

        with patch.object(metrics_collector, '_collect_performance_timeline', return_value=[]), \
             patch.object(metrics_collector, '_collect_conversion_timeline', return_value=[]), \
             patch.object(metrics_collector, '_collect_error_timeline', return_value=[]), \
             patch.object(metrics_collector, '_generate_alerts', return_value=[]), \
             patch.object(metrics_collector, '_generate_recommendations', return_value=[]), \
             patch.object(metrics_collector, '_assess_risk_level', return_value="unknown"), \
             patch.object(metrics_collector, '_assess_confidence_level', return_value="low"):

            result = await metrics_collector.collect_experiment_metrics(experiment_id)

        # Should handle missing data gracefully
        assert result is not None
        assert result.avg_response_time_ms == 0.0  # Default for missing data
        assert result.avg_token_reduction == 0.0
        assert result.success_rate == 0.0
        assert result.error_rate == 0.0  # When no data available, defaults to 1.0 success rate

    async def test_high_volume_data_handling(self, complete_system):
        """Test handling of high-volume experiment data."""
        metrics_collector = complete_system["metrics_collector"]
        experiment_manager = complete_system["experiment_manager"]

        experiment_id = "exp-high-volume"

        # Setup high-volume scenario
        mock_db_session = Mock()
        mock_experiment = Mock()
        mock_experiment.name = "High Volume Experiment"
        mock_experiment.status = "running"
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        mock_db_session.query.return_value.filter.return_value.count.return_value = 50000  # High active user count

        mock_results = Mock()
        mock_results.total_users = 100000  # Large user base
        mock_results.statistical_significance = 0.999
        mock_results.performance_summary = {
            "overall_success_rate": 0.987,
            "avg_response_time_ms": 95.0,
            "avg_token_reduction": 35.0
        }
        mock_results.variants = {
            "control": {"users": 50000, "success_rate": 0.985},
            "treatment": {"users": 50000, "success_rate": 0.989}
        }

        # Set up context manager properly
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_db_session)
        context_manager.__exit__ = Mock(return_value=None)
        experiment_manager.get_db_session.return_value = context_manager
        experiment_manager.get_experiment_results = AsyncMock(return_value=mock_results)

        # Mock high-volume timeline data
        with patch.object(metrics_collector, '_collect_performance_timeline') as mock_perf, \
             patch.object(metrics_collector, '_collect_conversion_timeline') as mock_conv, \
             patch.object(metrics_collector, '_collect_error_timeline') as mock_error, \
             patch.object(metrics_collector, '_generate_alerts') as mock_alerts, \
             patch.object(metrics_collector, '_generate_recommendations') as mock_recs, \
             patch.object(metrics_collector, '_assess_risk_level') as mock_risk, \
             patch.object(metrics_collector, '_assess_confidence_level') as mock_conf:

            # Simulate 24 hours of hourly data
            mock_perf.return_value = [
                {"timestamp": f"2024-01-01T{hour:02d}:00:00", "avg_response_time_ms": 95.0 + hour, "success_rate": 98.7}
                for hour in range(24)
            ]
            mock_conv.return_value = [
                {"timestamp": f"2024-01-01T{hour:02d}:00:00", "conversion_rate": 35.0 + (hour * 0.1)}
                for hour in range(24)
            ]
            mock_error.return_value = [
                {"timestamp": f"2024-01-01T{hour:02d}:00:00", "error_rate": 1.3}
                for hour in range(24)
            ]
            mock_alerts.return_value = []
            mock_recs.return_value = ["Highly significant results", "Consider deployment"]
            mock_risk.return_value = "low"
            mock_conf.return_value = "high"

            result = await metrics_collector.collect_experiment_metrics(experiment_id)

        # Verify high-volume data handling
        assert result is not None
        assert result.total_users == 100000
        assert result.active_users_24h == 50000
        assert len(result.performance_timeline) == 24
        assert len(result.conversion_timeline) == 24
        assert len(result.error_timeline) == 24
        assert result.statistical_significance == 0.999
        assert result.confidence_level == "high"
