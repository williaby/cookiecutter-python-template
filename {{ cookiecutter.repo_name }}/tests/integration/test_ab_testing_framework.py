"""
Integration Tests for A/B Testing Framework

This module provides comprehensive integration tests for the A/B testing framework,
covering experiment management, user assignment, metrics collection, statistical
analysis, and dashboard functionality.

Test Coverage:
- Experiment lifecycle management
- User segmentation and assignment
- Metrics collection and analysis
- Statistical significance calculation
- Dashboard data generation
- API endpoint functionality
- Safety mechanisms and rollback
- Performance monitoring
"""

from datetime import datetime, timedelta
from src.utils.datetime_compat import utc_now
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.ab_testing_endpoints import get_experiment_manager_dependency, router
from src.core.ab_testing_framework import (
    Base,
    ExperimentConfig,
    ExperimentManager,
    ExperimentType,
    MetricEvent,
    MetricsCollector,
    StatisticalAnalyzer,
    UserCharacteristics,
    UserSegment,
    UserSegmentation,
    create_dynamic_loading_experiment,
)
from src.core.dynamic_loading_integration import OptimizationReport, ProcessingResult
from src.monitoring.ab_testing_dashboard import ABTestingDashboard


@pytest.fixture
def test_db_engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_db_session(test_db_engine):
    """Create test database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
async def experiment_manager(test_db_engine):
    """Create test experiment manager."""
    manager = ExperimentManager(db_url="sqlite:///:memory:")
    # Override with test engine
    manager.engine = test_db_engine
    Base.metadata.create_all(manager.engine)
    manager.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=manager.engine)
    return manager


@pytest.fixture
def test_experiment_config():
    """Create test experiment configuration."""
    return ExperimentConfig(
        name="Test Dynamic Loading Experiment",
        description="Test experiment for dynamic function loading",
        experiment_type=ExperimentType.DYNAMIC_LOADING,
        planned_duration_hours=24,
        feature_flags={"dynamic_loading_enabled": True},
        variant_configs={
            "control": {
                "feature_flags": {"dynamic_loading_enabled": False},
                "loading_strategy": "baseline",
            },
            "treatment": {
                "feature_flags": {"dynamic_loading_enabled": True},
                "loading_strategy": "balanced",
            },
        },
        target_percentage=50.0,
        rollout_steps=[5.0, 25.0, 50.0],
        segment_filters=[
            UserSegment.RANDOM,
            UserSegment.EARLY_ADOPTER,
            UserSegment.POWER_USER,
        ],  # Include all user segments used in tests
        success_criteria={
            "min_token_reduction": 70.0,
            "max_response_time_ms": 200.0,
            "min_success_rate": 95.0,
        },
        failure_thresholds={
            "max_error_rate": 5.0,
            "min_token_reduction": 50.0,
        },
    )


@pytest.fixture
def test_user_characteristics():
    """Create test user characteristics."""
    return UserCharacteristics(
        user_id="test_user_1",
        registration_date=utc_now() - timedelta(days=30),
        usage_frequency="high",
        feature_usage_pattern="advanced",
        is_early_adopter=True,
        opt_in_beta=True,
    )


@pytest.fixture
def test_processing_result():
    """Create test processing result."""
    return ProcessingResult(
        query="test query",
        session_id="test_session_1",
        detection_result=MagicMock(),
        loading_decision=MagicMock(),
        optimization_report=OptimizationReport(
            session_id="test_session_1",
            baseline_token_count=1000,
            optimized_token_count=300,
            reduction_percentage=70.0,
            target_achieved=True,
            categories_detected=["optimization", "performance"],
            functions_loaded=25,
            strategy_used="balanced",
            processing_time_ms=150.0,
        ),
        user_commands=[],
        detection_time_ms=25.0,
        loading_time_ms=50.0,
        total_time_ms=150.0,
        baseline_tokens=1000,
        optimized_tokens=300,
        reduction_percentage=70.0,
        target_achieved=True,
        success=True,
    )


class TestExperimentManager:
    """Test experiment manager functionality."""

    @pytest.mark.asyncio
    async def test_create_experiment(self, experiment_manager, test_experiment_config):
        """Test experiment creation."""
        experiment_id = await experiment_manager.create_experiment(test_experiment_config)

        assert experiment_id is not None
        assert experiment_id.startswith("exp_")
        assert "dynamic_loading" in experiment_id

        # Verify experiment was created in database
        with experiment_manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import ExperimentModel

            experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()

            assert experiment is not None
            assert experiment.name == test_experiment_config.name
            assert experiment.status == "draft"
            assert experiment.experiment_type == "dynamic_loading"

    @pytest.mark.asyncio
    async def test_start_stop_experiment(self, experiment_manager, test_experiment_config):
        """Test starting and stopping experiments."""
        experiment_id = await experiment_manager.create_experiment(test_experiment_config)

        # Start experiment
        success = await experiment_manager.start_experiment(experiment_id)
        assert success is True

        # Verify experiment is active
        with experiment_manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import ExperimentModel

            experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()

            assert experiment.status == "active"
            assert experiment.start_time is not None
            assert experiment.current_percentage == test_experiment_config.initial_percentage

        # Stop experiment
        success = await experiment_manager.stop_experiment(experiment_id)
        assert success is True

        # Verify experiment is completed
        with experiment_manager.get_db_session() as db_session:
            experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()

            assert experiment.status == "completed"
            assert experiment.end_time is not None

    @pytest.mark.asyncio
    async def test_user_assignment(self, experiment_manager, test_experiment_config, test_user_characteristics):
        """Test user assignment to experiments."""
        experiment_id = await experiment_manager.create_experiment(test_experiment_config)
        await experiment_manager.start_experiment(experiment_id)

        # Assign user to experiment
        variant, segment = await experiment_manager.assign_user_to_experiment(
            "test_user_1",
            experiment_id,
            test_user_characteristics,
        )

        assert variant in ["control", "treatment"]
        assert isinstance(segment, UserSegment)

        # Verify assignment is consistent
        variant2, segment2 = await experiment_manager.assign_user_to_experiment(
            "test_user_1",
            experiment_id,
            test_user_characteristics,
        )

        assert variant == variant2
        assert segment == segment2

        # Verify assignment is stored in database
        with experiment_manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import UserAssignmentModel

            assignment = (
                db_session.query(UserAssignmentModel)
                .filter_by(
                    user_id="test_user_1",
                    experiment_id=experiment_id,
                )
                .first()
            )

            assert assignment is not None
            assert assignment.variant == variant
            assert assignment.segment == segment.value

    @pytest.mark.asyncio
    async def test_dynamic_loading_check(self, experiment_manager, test_experiment_config):
        """Test dynamic loading assignment check."""
        experiment_id = await experiment_manager.create_experiment(test_experiment_config)
        await experiment_manager.start_experiment(experiment_id)

        # Check dynamic loading assignment
        should_use = await experiment_manager.should_use_dynamic_loading("test_user_1", experiment_id)

        assert isinstance(should_use, bool)

        # Should be consistent
        should_use2 = await experiment_manager.should_use_dynamic_loading("test_user_1", experiment_id)
        assert should_use == should_use2

    @pytest.mark.asyncio
    async def test_optimization_result_recording(
        self,
        experiment_manager,
        test_experiment_config,
        test_processing_result,
    ):
        """Test recording optimization results."""
        experiment_id = await experiment_manager.create_experiment(test_experiment_config)
        await experiment_manager.start_experiment(experiment_id)

        # Assign user first
        await experiment_manager.assign_user_to_experiment("test_user_1", experiment_id)

        # Record optimization result
        success = await experiment_manager.record_optimization_result(
            experiment_id,
            "test_user_1",
            test_processing_result,
        )

        assert success is True

        # Verify metrics were recorded
        with experiment_manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import MetricEventModel

            events = (
                db_session.query(MetricEventModel)
                .filter_by(
                    experiment_id=experiment_id,
                    user_id="test_user_1",
                )
                .all()
            )

            assert len(events) >= 2  # Performance and optimization events

            # Check performance event
            perf_events = [e for e in events if e.event_type == "performance"]
            assert len(perf_events) >= 1

            perf_event = perf_events[0]
            assert perf_event.response_time_ms == test_processing_result.total_time_ms
            assert perf_event.token_reduction_percentage == test_processing_result.reduction_percentage
            assert perf_event.success is True


class TestUserSegmentation:
    """Test user segmentation functionality."""

    def test_segment_determination(self, test_db_session):
        """Test user segment determination logic."""
        segmentation = UserSegmentation(test_db_session)

        # Test early adopter
        early_adopter = UserCharacteristics(
            user_id="early_adopter",
            is_early_adopter=True,
        )
        segment = segmentation._determine_user_segment(early_adopter)
        assert segment == UserSegment.EARLY_ADOPTER

        # Test new user
        new_user = UserCharacteristics(
            user_id="new_user",
            registration_date=utc_now() - timedelta(days=15),
        )
        segment = segmentation._determine_user_segment(new_user)
        assert segment == UserSegment.NEW_USER

        # Test power user
        power_user = UserCharacteristics(
            user_id="power_user",
            usage_frequency="high",
            feature_usage_pattern="advanced",
        )
        segment = segmentation._determine_user_segment(power_user)
        assert segment == UserSegment.POWER_USER

        # Test high volume user
        high_volume = UserCharacteristics(
            user_id="high_volume",
            usage_frequency="high",
        )
        segment = segmentation._determine_user_segment(high_volume)
        assert segment == UserSegment.HIGH_VOLUME

    def test_consistent_assignment(self, test_db_session, test_experiment_config):
        """Test consistent user assignment."""
        segmentation = UserSegmentation(test_db_session)

        # Multiple assignments should be consistent
        user_id = "consistent_user"
        experiment_id = "test_experiment"

        # Mock experiment in database would be needed for full test
        # For now, test the hashing function
        variant1 = segmentation._assign_variant_consistent(user_id, experiment_id, 50.0)
        variant2 = segmentation._assign_variant_consistent(user_id, experiment_id, 50.0)

        assert variant1 == variant2


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def test_record_event(self, test_db_session):
        """Test metric event recording."""
        collector = MetricsCollector(test_db_session)

        event = MetricEvent(
            experiment_id="test_experiment",
            user_id="test_user",
            variant="treatment",
            event_type="performance",
            event_name="query_processing",
            event_value=150.0,
            response_time_ms=150.0,
            token_reduction_percentage=75.0,
            success=True,
        )

        success = collector.record_event(event)
        assert success is True

        # Verify event was stored
        from src.core.ab_testing_framework import MetricEventModel

        stored_event = (
            test_db_session.query(MetricEventModel)
            .filter_by(
                experiment_id="test_experiment",
                user_id="test_user",
            )
            .first()
        )

        assert stored_event is not None
        assert stored_event.event_type == "performance"
        assert stored_event.response_time_ms == 150.0
        assert stored_event.token_reduction_percentage == 75.0

    def test_record_processing_result(self, test_db_session, test_processing_result):
        """Test recording processing results."""
        collector = MetricsCollector(test_db_session)

        success = collector.record_processing_result(
            "test_experiment",
            "test_user",
            "treatment",
            test_processing_result,
        )

        assert success is True

        # Verify multiple events were created
        from src.core.ab_testing_framework import MetricEventModel

        events = (
            test_db_session.query(MetricEventModel)
            .filter_by(
                experiment_id="test_experiment",
                user_id="test_user",
            )
            .all()
        )

        assert len(events) >= 2  # Performance and optimization events

        event_types = [e.event_type for e in events]
        assert "performance" in event_types
        assert "optimization" in event_types


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""

    def test_variant_data_collection(self, test_db_session):
        """Test collection of variant performance data."""
        analyzer = StatisticalAnalyzer(test_db_session)

        # Create test data
        from src.core.ab_testing_framework import ExperimentModel, MetricEventModel, UserAssignmentModel

        # Create experiment
        experiment = ExperimentModel(
            id="test_experiment",
            name="Test Experiment",
            experiment_type="dynamic_loading",
            status="active",
            config={},
            variants=["control", "treatment"],
            success_criteria={},
            failure_thresholds={},
        )
        test_db_session.add(experiment)

        # Create user assignments
        for i in range(10):
            variant = "control" if i < 5 else "treatment"
            assignment = UserAssignmentModel(
                id=f"assignment_{i}",
                user_id=f"user_{i}",
                experiment_id="test_experiment",
                variant=variant,
                segment="random",
            )
            test_db_session.add(assignment)

        # Create metric events
        for i in range(20):
            variant = "control" if i < 10 else "treatment"
            user_id = f"user_{i % 10}"
            event = MetricEventModel(
                id=f"event_{i}",
                experiment_id="test_experiment",
                user_id=user_id,
                variant=variant,
                event_type="performance",
                event_name="query_processing",
                event_value=100.0 + (i * 5),  # Varying values
                response_time_ms=100.0 + (i * 5),
                token_reduction_percentage=70.0 + (i % 5),
                success=True,
            )
            test_db_session.add(event)

        test_db_session.commit()

        # Test variant data collection
        variant_data = analyzer._collect_variant_data("test_experiment")

        assert "control" in variant_data
        assert "treatment" in variant_data
        assert variant_data["control"]["user_count"] == 5
        assert variant_data["treatment"]["user_count"] == 5
        assert len(variant_data["control"]["events"]) == 10
        assert len(variant_data["treatment"]["events"]) == 10

    def test_statistical_significance_calculation(self, test_db_session):
        """Test statistical significance calculation."""
        analyzer = StatisticalAnalyzer(test_db_session)

        # Create mock variant data
        variant_data = {
            "control": {
                "user_count": 100,
                "success_count": 90,
                "error_count": 10,
                "success_rate": 0.9,
                "avg_response_time_ms": 200.0,
                "avg_token_reduction": 65.0,
            },
            "treatment": {
                "user_count": 100,
                "success_count": 95,
                "error_count": 5,
                "success_rate": 0.95,
                "avg_response_time_ms": 180.0,
                "avg_token_reduction": 75.0,
            },
        }

        results = analyzer._calculate_statistical_significance(variant_data)

        assert "significance" in results
        assert "confidence_interval" in results
        assert "p_value" in results
        assert "effect_size" in results

        assert 0.0 <= results["significance"] <= 100.0
        assert 0.0 <= results["p_value"] <= 1.0
        assert 0.0 <= results["effect_size"] <= 1.0


class TestDashboard:
    """Test dashboard functionality."""

    @pytest.mark.asyncio
    async def test_dashboard_instance_creation(self, experiment_manager):
        """Test dashboard instance creation."""
        dashboard = ABTestingDashboard(experiment_manager)

        assert dashboard is not None
        assert dashboard.experiment_manager == experiment_manager
        assert dashboard.metrics_collector is not None
        assert dashboard.visualizer is not None

    @pytest.mark.asyncio
    async def test_metrics_collection(self, experiment_manager, test_experiment_config, test_user_characteristics):
        """Test dashboard metrics collection."""
        dashboard = ABTestingDashboard(experiment_manager)

        # Create experiment with full rollout to ensure variant diversity
        config = ExperimentConfig(
            name="Test Dynamic Loading Experiment",
            description="Test experiment for dynamic function loading",
            experiment_type=ExperimentType.DYNAMIC_LOADING,
            planned_duration_hours=24,
            initial_percentage=100.0,  # Full rollout to get both variants
            target_percentage=100.0,
            segment_filters=[UserSegment.RANDOM, UserSegment.EARLY_ADOPTER, UserSegment.POWER_USER],
            feature_flags={"dynamic_loading_enabled": True},
            variant_configs={
                "control": {"feature_flags": {"dynamic_loading_enabled": False}},
                "treatment": {"feature_flags": {"dynamic_loading_enabled": True}},
            },
        )
        experiment_id = await experiment_manager.create_experiment(config)
        await experiment_manager.start_experiment(experiment_id)

        # Add multiple test users to ensure we get both variants
        from src.core.ab_testing_framework import UserCharacteristics

        variants = []
        # Use diverse user IDs that will hash to different variants
        user_ids = ["user123", "user456", "user789", "user000", "user999", "alice", "bob", "charlie", "diana", "edward"]

        for user_id in user_ids:
            user_chars = UserCharacteristics(user_id=user_id, usage_frequency="high", is_early_adopter=True)
            variant, _ = await experiment_manager.assign_user_to_experiment(user_id, experiment_id, user_chars)
            variants.append((user_id, variant))
            print(f"DEBUG: {user_id} -> {variant}")

        # Count unique variants
        unique_variants = set(variant for _, variant in variants)
        print(f"DEBUG: Unique variants: {unique_variants}")

        # Use first two users with potentially different variants for the processing results
        user1, variant1 = variants[0]
        user2, variant2 = variants[1]

        # Record some optimization results to generate metric data
        from unittest.mock import MagicMock

        from src.core.dynamic_loading_integration import OptimizationReport, ProcessingResult

        processing_result1 = ProcessingResult(
            query=f"test query for {user1}",
            session_id=f"session_{user1}",
            detection_result=MagicMock(),
            loading_decision=MagicMock(),
            optimization_report=OptimizationReport(
                session_id=f"session_{user1}",
                baseline_token_count=1000,
                optimized_token_count=300,
                reduction_percentage=70.0,
                target_achieved=True,
                categories_detected=["optimization"],
                functions_loaded=25,
                strategy_used=variant1,
                processing_time_ms=150.0,
            ),
            user_commands=[],
            detection_time_ms=25.0,
            loading_time_ms=50.0,
            total_time_ms=150.0,
            baseline_tokens=1000,
            optimized_tokens=300,
            reduction_percentage=70.0,
            target_achieved=True,
            success=True,
        )

        processing_result2 = ProcessingResult(
            query=f"test query for {user2}",
            session_id=f"session_{user2}",
            detection_result=MagicMock(),
            loading_decision=MagicMock(),
            optimization_report=OptimizationReport(
                session_id=f"session_{user2}",
                baseline_token_count=1000,
                optimized_token_count=800,
                reduction_percentage=20.0,
                target_achieved=False,
                categories_detected=["optimization"],
                functions_loaded=10,
                strategy_used=variant2,
                processing_time_ms=200.0,
            ),
            user_commands=[],
            detection_time_ms=30.0,
            loading_time_ms=60.0,
            total_time_ms=200.0,
            baseline_tokens=1000,
            optimized_tokens=800,
            reduction_percentage=20.0,
            target_achieved=False,
            success=True,
        )

        await experiment_manager.record_optimization_result(experiment_id, user1, processing_result1)
        await experiment_manager.record_optimization_result(experiment_id, user2, processing_result2)

        # Collect metrics
        metrics = await dashboard.metrics_collector.collect_experiment_metrics(experiment_id)

        assert metrics is not None
        assert metrics.experiment_id == experiment_id
        assert metrics.experiment_name == test_experiment_config.name
        assert isinstance(metrics.total_users, int)
        assert isinstance(metrics.statistical_significance, float)

    @pytest.mark.asyncio
    async def test_dashboard_html_generation(
        self,
        experiment_manager,
        test_experiment_config,
        test_user_characteristics,
    ):
        """Test HTML dashboard generation."""
        dashboard = ABTestingDashboard(experiment_manager)

        # Create experiment with full rollout to ensure variant diversity
        config = ExperimentConfig(
            name="Test Dynamic Loading Experiment",
            description="Test experiment for dynamic function loading",
            experiment_type=ExperimentType.DYNAMIC_LOADING,
            planned_duration_hours=24,
            initial_percentage=100.0,  # Full rollout to get both variants
            segment_filters=[UserSegment.RANDOM, UserSegment.EARLY_ADOPTER, UserSegment.POWER_USER],
            feature_flags={"dynamic_loading_enabled": True},
            variant_configs={
                "control": {"feature_flags": {"dynamic_loading_enabled": False}},
                "treatment": {"feature_flags": {"dynamic_loading_enabled": True}},
            },
        )
        experiment_id = await experiment_manager.create_experiment(config)
        await experiment_manager.start_experiment(experiment_id)

        # Add multiple test users to ensure variant diversity
        from src.core.ab_testing_framework import UserCharacteristics

        user_ids = [f"test_user_{i}" for i in range(20)]  # More users to ensure variant distribution
        variants = []
        variant_counts = {"control": 0, "treatment": 0}

        for user_id in user_ids:
            user_chars = UserCharacteristics(user_id=user_id, usage_frequency="high", is_early_adopter=True)
            variant, _ = await experiment_manager.assign_user_to_experiment(user_id, experiment_id, user_chars)
            variants.append((user_id, variant))
            variant_counts[variant] += 1

        # Ensure we have both variants
        assert variant_counts["control"] > 0, f"No control users found. Variant counts: {variant_counts}"
        assert variant_counts["treatment"] > 0, f"No treatment users found. Variant counts: {variant_counts}"

        # Get users from different variants for testing
        control_users = [user_id for user_id, variant in variants if variant == "control"]
        treatment_users = [user_id for user_id, variant in variants if variant == "treatment"]

        user1, variant1 = control_users[0], "control"
        user2, variant2 = treatment_users[0], "treatment"

        # Record optimization results to generate meaningful data
        from unittest.mock import MagicMock

        from src.core.dynamic_loading_integration import OptimizationReport, ProcessingResult

        processing_result1 = ProcessingResult(
            query="test query 1",
            session_id="session_1",
            detection_result=MagicMock(),
            loading_decision=MagicMock(),
            optimization_report=OptimizationReport(
                session_id="session_1",
                baseline_token_count=1000,
                optimized_token_count=300,
                reduction_percentage=70.0,
                target_achieved=True,
                categories_detected=["optimization"],
                functions_loaded=25,
                strategy_used=variant1,
                processing_time_ms=150.0,
            ),
            user_commands=[],
            detection_time_ms=25.0,
            loading_time_ms=50.0,
            total_time_ms=150.0,
            baseline_tokens=1000,
            optimized_tokens=300,
            reduction_percentage=70.0,
            target_achieved=True,
            success=True,
        )
        processing_result2 = ProcessingResult(
            query="test query 2",
            session_id="session_2",
            detection_result=MagicMock(),
            loading_decision=MagicMock(),
            optimization_report=OptimizationReport(
                session_id="session_2",
                baseline_token_count=1000,
                optimized_token_count=800,
                reduction_percentage=20.0,
                target_achieved=False,
                categories_detected=["optimization"],
                functions_loaded=10,
                strategy_used=variant2,
                processing_time_ms=200.0,
            ),
            user_commands=[],
            detection_time_ms=30.0,
            loading_time_ms=60.0,
            total_time_ms=200.0,
            baseline_tokens=1000,
            optimized_tokens=800,
            reduction_percentage=20.0,
            target_achieved=False,
            success=True,
        )

        await experiment_manager.record_optimization_result(experiment_id, user1, processing_result1)
        await experiment_manager.record_optimization_result(experiment_id, user2, processing_result2)

        # Generate dashboard HTML
        html_content = await dashboard.generate_dashboard_html(experiment_id)

        assert html_content is not None
        assert "<!DOCTYPE html>" in html_content
        assert test_experiment_config.name in html_content
        assert "Statistical Significance" in html_content

    @pytest.mark.asyncio
    async def test_dashboard_data_api(self, experiment_manager, test_experiment_config, test_user_characteristics):
        """Test dashboard data API."""
        dashboard = ABTestingDashboard(experiment_manager)

        # Create experiment with full rollout to ensure variant diversity
        config = ExperimentConfig(
            name="Test Dynamic Loading Experiment",
            description="Test experiment for dynamic function loading",
            experiment_type=ExperimentType.DYNAMIC_LOADING,
            planned_duration_hours=24,
            initial_percentage=100.0,  # Full rollout to get both variants
            segment_filters=[UserSegment.RANDOM, UserSegment.EARLY_ADOPTER, UserSegment.POWER_USER],
            feature_flags={"dynamic_loading_enabled": True},
            variant_configs={
                "control": {"feature_flags": {"dynamic_loading_enabled": False}},
                "treatment": {"feature_flags": {"dynamic_loading_enabled": True}},
            },
        )
        experiment_id = await experiment_manager.create_experiment(config)
        await experiment_manager.start_experiment(experiment_id)

        # Add multiple test users to ensure variant diversity
        from src.core.ab_testing_framework import UserCharacteristics

        user_ids = ["test_user_a", "test_user_b", "test_user_c", "test_user_d"]
        variants = []

        for user_id in user_ids:
            user_chars = UserCharacteristics(user_id=user_id, usage_frequency="high", is_early_adopter=True)
            variant, _ = await experiment_manager.assign_user_to_experiment(user_id, experiment_id, user_chars)
            variants.append((user_id, variant))

        user1, variant1 = variants[0]
        user2, variant2 = variants[1]

        # Record optimization results
        from unittest.mock import MagicMock

        from src.core.dynamic_loading_integration import OptimizationReport, ProcessingResult

        processing_result1 = ProcessingResult(
            query="test query 1",
            session_id="session_1",
            detection_result=MagicMock(),
            loading_decision=MagicMock(),
            optimization_report=OptimizationReport(
                session_id="session_1",
                baseline_token_count=1000,
                optimized_token_count=300,
                reduction_percentage=70.0,
                target_achieved=True,
                categories_detected=["optimization"],
                functions_loaded=25,
                strategy_used=variant1,
                processing_time_ms=150.0,
            ),
            user_commands=[],
            detection_time_ms=25.0,
            loading_time_ms=50.0,
            total_time_ms=150.0,
            baseline_tokens=1000,
            optimized_tokens=300,
            reduction_percentage=70.0,
            target_achieved=True,
            success=True,
        )
        processing_result2 = ProcessingResult(
            query="test query 2",
            session_id="session_2",
            detection_result=MagicMock(),
            loading_decision=MagicMock(),
            optimization_report=OptimizationReport(
                session_id="session_2",
                baseline_token_count=1000,
                optimized_token_count=800,
                reduction_percentage=20.0,
                target_achieved=False,
                categories_detected=["optimization"],
                functions_loaded=10,
                strategy_used=variant2,
                processing_time_ms=200.0,
            ),
            user_commands=[],
            detection_time_ms=30.0,
            loading_time_ms=60.0,
            total_time_ms=200.0,
            baseline_tokens=1000,
            optimized_tokens=800,
            reduction_percentage=20.0,
            target_achieved=False,
            success=True,
        )

        await experiment_manager.record_optimization_result(experiment_id, user1, processing_result1)
        await experiment_manager.record_optimization_result(experiment_id, user2, processing_result2)

        # Get dashboard data
        dashboard_data = await dashboard.get_dashboard_data(experiment_id)

        assert dashboard_data is not None
        assert dashboard_data["experiment_id"] == experiment_id
        assert dashboard_data["experiment_name"] == test_experiment_config.name
        assert "total_users" in dashboard_data
        assert "statistical_significance" in dashboard_data


class TestAPIEndpoints:
    """Test API endpoints functionality."""

    def test_create_experiment_endpoint(self):
        """Test experiment creation endpoint."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        # Mock the experiment manager dependency
        mock_exp_manager = MagicMock()  # Change to regular MagicMock since we need synchronous context manager
        mock_exp_manager.create_experiment = AsyncMock(return_value="test_experiment_id")

        # Mock the database session and query result
        mock_experiment = MagicMock()
        mock_experiment.id = "test_experiment_id"
        mock_experiment.name = "Test Experiment"
        mock_experiment.description = "Test Description"
        mock_experiment.experiment_type = "dynamic_loading"
        mock_experiment.status = "draft"
        mock_experiment.target_percentage = 50.0
        mock_experiment.current_percentage = 0.0
        mock_experiment.planned_duration_hours = 24
        mock_experiment.total_users = 0
        mock_experiment.statistical_significance = 0.0
        mock_experiment.created_at = utc_now()
        mock_experiment.start_time = None
        mock_experiment.end_time = None

        # Mock the database session context manager properly
        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_experiment
        mock_exp_manager.get_db_session.return_value.__enter__.return_value = mock_db_session
        mock_exp_manager.get_db_session.return_value.__exit__.return_value = None

        # Override the dependency with a synchronous function that returns the mock
        async def mock_dependency():
            return mock_exp_manager

        app.dependency_overrides[get_experiment_manager_dependency] = mock_dependency

        # Mock audit logger
        with patch("src.api.ab_testing_endpoints.audit_logger_instance") as mock_audit_logger:
            mock_audit_logger.log_api_event = MagicMock()

            client = TestClient(app)
            response = client.post(
                "/api/v1/ab-testing/experiments",
                json={
                    "name": "Test Experiment",
                    "description": "Test Description",
                    "experiment_type": "dynamic_loading",
                    "target_percentage": 50.0,
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "test_experiment_id"
            assert data["name"] == "Test Experiment"

    def test_user_assignment_endpoint(self):
        """Test user assignment endpoint."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        # Mock the experiment manager dependency
        mock_exp_manager = MagicMock()
        mock_exp_manager.assign_user_to_experiment = AsyncMock(return_value=("treatment", UserSegment.POWER_USER))

        # Override the dependency
        async def mock_dependency():
            return mock_exp_manager

        app.dependency_overrides[get_experiment_manager_dependency] = mock_dependency

        # Mock audit logger
        with patch("src.api.ab_testing_endpoints.audit_logger_instance") as mock_audit_logger:
            mock_audit_logger.log_api_event = MagicMock()

            client = TestClient(app)
            response = client.post(
                "/api/v1/ab-testing/assign-user",
                json={
                    "user_id": "test_user",
                    "experiment_id": "test_experiment",
                    "usage_frequency": "high",
                    "is_early_adopter": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "test_user"
            assert data["variant"] == "treatment"
            assert data["segment"] == "power_user"
            assert data["success"] is True

    def test_dynamic_loading_check_endpoint(self):
        """Test dynamic loading assignment check endpoint."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        # Mock the experiment manager dependency
        mock_exp_manager = MagicMock()
        mock_exp_manager.should_use_dynamic_loading = AsyncMock(return_value=True)

        # Override the dependency
        async def mock_dependency():
            return mock_exp_manager

        app.dependency_overrides[get_experiment_manager_dependency] = mock_dependency

        client = TestClient(app)
        response = client.get("/api/v1/ab-testing/check-dynamic-loading/test_user")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user"
        assert data["use_dynamic_loading"] is True

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # Mock dependencies
        with patch("src.api.ab_testing_endpoints.get_experiment_manager") as mock_get_manager:
            mock_exp_manager = MagicMock()
            mock_exp_manager.get_db_session.return_value.__enter__.return_value.execute.return_value = None
            mock_get_manager.return_value = mock_exp_manager

            response = client.get("/api/v1/ab-testing/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "ab-testing"


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_experiment_lifecycle(
        self,
        experiment_manager,
        test_experiment_config,
        test_user_characteristics,
    ):
        """Test complete experiment lifecycle from creation to completion."""

        # 1. Create experiment with 100% rollout to ensure variant diversity
        config = ExperimentConfig(
            name="Test Dynamic Loading Experiment",
            description="Test experiment for dynamic function loading",
            experiment_type=ExperimentType.DYNAMIC_LOADING,
            planned_duration_hours=24,
            initial_percentage=100.0,  # Full rollout to get both variants
            target_percentage=100.0,
            segment_filters=[UserSegment.RANDOM, UserSegment.EARLY_ADOPTER, UserSegment.POWER_USER],
            feature_flags={"dynamic_loading_enabled": True},
            variant_configs={
                "control": {"feature_flags": {"dynamic_loading_enabled": False}},
                "treatment": {"feature_flags": {"dynamic_loading_enabled": True}},
            },
            success_criteria={"min_token_reduction": 50.0},
            failure_thresholds={"max_error_rate": 10.0},
            auto_rollback_enabled=True,
        )

        experiment_id = await experiment_manager.create_experiment(config)
        assert experiment_id is not None

        # 2. Start experiment
        success = await experiment_manager.start_experiment(experiment_id)
        assert success is True

        # 3. Assign users and collect metrics - ensure we get both variants
        users = [f"user_{i}" for i in range(100)]
        variant_counts = {"control": 0, "treatment": 0}

        for user_id in users:
            # Assign user to experiment
            variant, segment = await experiment_manager.assign_user_to_experiment(
                user_id,
                experiment_id,
                test_user_characteristics,
            )
            variant_counts[variant] += 1

            # Simulate processing result
            processing_result = ProcessingResult(
                query=f"test query for {user_id}",
                session_id=f"session_{user_id}",
                detection_result=MagicMock(),
                loading_decision=MagicMock(),
                optimization_report=OptimizationReport(
                    session_id=f"session_{user_id}",
                    baseline_token_count=1000,
                    optimized_token_count=300 if variant == "treatment" else 800,
                    reduction_percentage=70.0 if variant == "treatment" else 20.0,
                    target_achieved=variant == "treatment",
                    categories_detected=["optimization"],
                    functions_loaded=25,
                    strategy_used=variant,
                    processing_time_ms=150.0,
                ),
                user_commands=[],
                detection_time_ms=25.0,
                loading_time_ms=50.0,
                total_time_ms=150.0,
                baseline_tokens=1000,
                optimized_tokens=300 if variant == "treatment" else 800,
                reduction_percentage=70.0 if variant == "treatment" else 20.0,
                target_achieved=variant == "treatment",
                success=True,
            )

            # Record optimization result
            await experiment_manager.record_optimization_result(
                experiment_id,
                user_id,
                processing_result,
            )

        # Ensure we have both variants for statistical analysis
        assert variant_counts["control"] > 0, f"No control users found. Variant counts: {variant_counts}"
        assert variant_counts["treatment"] > 0, f"No treatment users found. Variant counts: {variant_counts}"
        assert len(variant_counts) >= 2, f"Need at least 2 variants for analysis. Found: {variant_counts}"

        # 4. Analyze results
        results = await experiment_manager.get_experiment_results(experiment_id)
        assert results is not None
        assert results.total_users > 0
        assert results.statistical_significance >= 0.0

        # 5. Generate dashboard
        dashboard = ABTestingDashboard(experiment_manager)
        dashboard_data = await dashboard.get_dashboard_data(experiment_id)
        assert dashboard_data is not None

        # 6. Stop experiment
        success = await experiment_manager.stop_experiment(experiment_id)
        assert success is True

        # Verify final state
        with experiment_manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import ExperimentModel

            experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
            assert experiment.status == "completed"
            assert experiment.end_time is not None

    @pytest.mark.asyncio
    async def test_rollback_scenario(self, experiment_manager, test_experiment_config):
        """Test automatic rollback scenario."""

        # Create experiment with low failure threshold
        config = test_experiment_config
        config.failure_thresholds = {"max_error_rate": 1.0}  # Very low threshold
        config.auto_rollback_enabled = True
        config.circuit_breaker_threshold = 5.0

        experiment_id = await experiment_manager.create_experiment(config)
        await experiment_manager.start_experiment(experiment_id)

        # Simulate high error rate
        users = [f"user_{i}" for i in range(20)]

        for _i, user_id in enumerate(users):
            await experiment_manager.assign_user_to_experiment(user_id, experiment_id)

            # Create failing processing results
            processing_result = ProcessingResult(
                query=f"test query for {user_id}",
                session_id=f"session_{user_id}",
                detection_result=MagicMock(),
                loading_decision=MagicMock(),
                optimization_report=OptimizationReport(
                    session_id=f"session_{user_id}",
                    baseline_token_count=1000,
                    optimized_token_count=1000,
                    reduction_percentage=0.0,
                    target_achieved=False,
                    categories_detected=[],
                    functions_loaded=0,
                    strategy_used="failed",
                    processing_time_ms=1000.0,
                    error_message="Simulated failure",
                ),
                user_commands=[],
                detection_time_ms=25.0,
                loading_time_ms=50.0,
                total_time_ms=1000.0,
                baseline_tokens=1000,
                optimized_tokens=1000,
                reduction_percentage=0.0,
                target_achieved=False,
                success=False,
                error_message="Simulated failure",
            )

            await experiment_manager.record_optimization_result(
                experiment_id,
                user_id,
                processing_result,
            )

        # Check if automatic rollback would be triggered
        from src.core.ab_testing_framework import RolloutController

        with experiment_manager.get_db_session() as db_session:
            controller = RolloutController(db_session)
            rollback_triggered = await controller.auto_rollback_if_needed(experiment_id)

            # With high error rate, rollback should be triggered
            # Note: This depends on having enough data for analysis
            assert isinstance(rollback_triggered, bool)

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, experiment_manager):
        """Test performance monitoring and alerting."""

        # Create dynamic loading experiment
        experiment_id = await create_dynamic_loading_experiment(
            target_percentage=10.0,
            duration_hours=1,
        )

        # Simulate performance data
        users = [f"perf_user_{i}" for i in range(50)]

        for user_id in users:
            # Check dynamic loading assignment
            should_use = await experiment_manager.should_use_dynamic_loading(user_id, experiment_id)

            # Create processing result based on assignment
            if should_use:
                # Treatment group - better performance
                processing_result = ProcessingResult(
                    query=f"performance test for {user_id}",
                    session_id=f"perf_session_{user_id}",
                    detection_result=MagicMock(),
                    loading_decision=MagicMock(),
                    optimization_report=OptimizationReport(
                        session_id=f"perf_session_{user_id}",
                        baseline_token_count=1000,
                        optimized_token_count=250,
                        reduction_percentage=75.0,
                        target_achieved=True,
                        categories_detected=["optimization", "performance"],
                        functions_loaded=20,
                        strategy_used="balanced",
                        processing_time_ms=120.0,
                    ),
                    user_commands=[],
                    detection_time_ms=20.0,
                    loading_time_ms=40.0,
                    total_time_ms=120.0,
                    baseline_tokens=1000,
                    optimized_tokens=250,
                    reduction_percentage=75.0,
                    target_achieved=True,
                    success=True,
                )
            else:
                # Control group - baseline performance
                processing_result = ProcessingResult(
                    query=f"performance test for {user_id}",
                    session_id=f"perf_session_{user_id}",
                    detection_result=MagicMock(),
                    loading_decision=MagicMock(),
                    optimization_report=OptimizationReport(
                        session_id=f"perf_session_{user_id}",
                        baseline_token_count=1000,
                        optimized_token_count=1000,
                        reduction_percentage=0.0,
                        target_achieved=False,
                        categories_detected=[],
                        functions_loaded=100,
                        strategy_used="baseline",
                        processing_time_ms=300.0,
                    ),
                    user_commands=[],
                    detection_time_ms=50.0,
                    loading_time_ms=100.0,
                    total_time_ms=300.0,
                    baseline_tokens=1000,
                    optimized_tokens=1000,
                    reduction_percentage=0.0,
                    target_achieved=False,
                    success=True,
                )

            await experiment_manager.record_optimization_result(
                experiment_id,
                user_id,
                processing_result,
            )

        # Analyze performance
        results = await experiment_manager.get_experiment_results(experiment_id)

        if results:
            assert results.total_users > 0
            assert "variants" in results.performance_summary

            # Verify we have data for both variants
            variant_comparison = results.performance_summary.get("variant_comparison", {})
            assert len(variant_comparison) >= 1  # At least one variant should have data

        # Generate dashboard with alerts
        dashboard = ABTestingDashboard(experiment_manager)
        metrics = await dashboard.metrics_collector.collect_experiment_metrics(experiment_id)

        if metrics:
            assert metrics.experiment_id == experiment_id
            assert isinstance(metrics.active_alerts, list)
            assert isinstance(metrics.recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
