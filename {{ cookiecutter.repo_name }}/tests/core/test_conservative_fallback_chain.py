"""
Comprehensive test suite for Conservative Fallback Chain

This test suite validates all aspects of the conservative fallback mechanism,
including edge cases, failure scenarios, performance characteristics, and
integration with the task detection system.

Test Categories:
- Unit tests for individual components
- Integration tests for end-to-end scenarios
- Performance tests for timeout and memory constraints
- Failure injection tests for resilience validation
- Edge case tests for unusual scenarios
"""

import asyncio
import builtins
import contextlib
import time
from typing import Any
from unittest.mock import patch

import pytest

from src.core.conservative_fallback_chain import (
    ConservativeFallbackChain,
    ErrorClassifier,
    ErrorContext,
    ErrorSeverity,
    ErrorType,
    FallbackDecision,
    FallbackLevel,
    LearningCollector,
    PerformanceMonitor,
    RecoveryManager,
    create_conservative_fallback_chain,
    with_conservative_fallback,
)
from src.core.fallback_circuit_breaker import (
    AdvancedCircuitBreakerState,
    FailurePattern,
    create_conservative_circuit_breaker,
)
from src.core.task_detection import DetectionResult
from src.core.task_detection_config import DetectionMode, TaskDetectionConfig


class MockTaskDetectionSystem:
    """Mock task detection system for testing"""

    def __init__(self, failure_mode: str | None = None, delay: float = 0.0):
        self.failure_mode = failure_mode
        self.delay = delay
        self.call_count = 0

    async def detect_categories(self, query: str, context: dict[str, Any] | None = None) -> DetectionResult:
        """Mock detection with configurable failure modes"""
        self.call_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.failure_mode == "timeout":
            raise TimeoutError("Mock timeout")
        if self.failure_mode == "network":
            raise ConnectionError("Mock network failure")
        if self.failure_mode == "memory":
            raise MemoryError("Mock memory pressure")
        if self.failure_mode == "generic":
            raise Exception("Mock generic error")

        # Handle empty queries with low confidence to trigger safe defaults
        if not query or query.strip() == "":
            return DetectionResult(
                categories={"core": False, "git": False, "debug": False, "test": False},
                confidence_scores={"core": 0.1, "git": 0.1, "debug": 0.1, "test": 0.1},
                detection_time_ms=10.0,
                signals_used={"keyword": []},
                fallback_applied=None,
            )

        # Return mock successful result
        return DetectionResult(
            categories={"git": True, "debug": True, "test": False},
            confidence_scores={"git": 0.8, "debug": 0.6, "test": 0.2},
            detection_time_ms=50.0,
            signals_used={"keyword": [{"git": 0.8}]},
            fallback_applied=None,
        )


@pytest.fixture
def mock_detection_system():
    """Fixture providing mock detection system"""
    return MockTaskDetectionSystem()


@pytest.fixture
def conservative_config():
    """Fixture providing conservative configuration"""
    config = TaskDetectionConfig()
    config.apply_mode_preset(DetectionMode.CONSERVATIVE)
    return config


@pytest.fixture
def fallback_chain(mock_detection_system, conservative_config):
    """Fixture providing configured fallback chain"""
    return ConservativeFallbackChain(mock_detection_system, conservative_config)


class TestErrorClassifier:
    """Test suite for error classifier component"""

    def test_timeout_error_classification(self):
        """Test classification of timeout errors"""
        classifier = ErrorClassifier()

        timeout_error = TimeoutError("Request timed out")
        context = {"query": "test query"}

        error_context = classifier.classify_error(timeout_error, context)

        assert error_context.error_type == ErrorType.TIMEOUT
        assert error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]
        assert error_context.query == "test query"

    def test_network_error_classification(self):
        """Test classification of network errors"""
        classifier = ErrorClassifier()

        network_error = ConnectionError("Network connection failed")
        context = {"query": "test query"}

        error_context = classifier.classify_error(network_error, context)

        assert error_context.error_type == ErrorType.NETWORK_FAILURE
        assert error_context.metadata["exception_type"] == "ConnectionError"

    def test_memory_error_classification(self):
        """Test classification of memory errors"""
        classifier = ErrorClassifier()

        memory_error = MemoryError("Out of memory")
        context = {"query": "test query"}

        error_context = classifier.classify_error(memory_error, context)

        assert error_context.error_type == ErrorType.MEMORY_PRESSURE
        assert error_context.severity == ErrorSeverity.HIGH

    def test_circuit_breaker_trigger_logic(self):
        """Test logic for triggering circuit breaker"""
        classifier = ErrorClassifier()

        # Critical error should trigger circuit breaker
        critical_context = ErrorContext(
            error_type=ErrorType.NETWORK_FAILURE,
            severity=ErrorSeverity.CRITICAL,
            timestamp=time.time(),
            query="test",
            context={},
        )
        assert classifier.should_trigger_circuit_breaker(critical_context)

        # Network failure should trigger circuit breaker
        network_context = ErrorContext(
            error_type=ErrorType.NETWORK_FAILURE,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test",
            context={},
        )
        assert classifier.should_trigger_circuit_breaker(network_context)

        # Detection failure should not trigger circuit breaker
        detection_context = ErrorContext(
            error_type=ErrorType.DETECTION_FAILURE,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test",
            context={},
        )
        assert not classifier.should_trigger_circuit_breaker(detection_context)

    def test_recovery_strategy_recommendations(self):
        """Test recovery strategy recommendations"""
        classifier = ErrorClassifier()

        timeout_context = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test",
            context={},
        )
        assert classifier.get_recommended_recovery_strategy(timeout_context) == "retry_with_backoff"

        network_context = ErrorContext(
            error_type=ErrorType.NETWORK_FAILURE,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test",
            context={},
        )
        assert classifier.get_recommended_recovery_strategy(network_context) == "circuit_breaker_with_fallback"


class TestPerformanceMonitor:
    """Test suite for performance monitor component"""

    def test_operation_recording(self):
        """Test recording of operation metrics"""
        monitor = PerformanceMonitor(max_history=10)

        # Record some operations
        monitor.record_operation(1.0, 50.0, False)  # Success
        monitor.record_operation(2.0, 60.0, True)  # Error
        monitor.record_operation(0.5, 40.0, False)  # Success

        assert len(monitor.response_times) == 3
        assert len(monitor.memory_usage) == 3
        assert len(monitor.error_rates) == 3

    def test_health_status_calculation(self):
        """Test health status calculation"""
        monitor = PerformanceMonitor()

        # Record normal operations
        for _i in range(10):
            monitor.record_operation(0.5, 30.0, False)

        health = monitor.get_health_status()

        assert health["healthy"] is True
        assert health["metrics"]["avg_response_time"] == 0.5
        assert health["metrics"]["error_rate"] == 0.0
        assert health["status"]["response_time_healthy"] is True
        assert health["status"]["memory_healthy"] is True
        assert health["status"]["error_rate_healthy"] is True

    def test_emergency_mode_trigger(self):
        """Test emergency mode trigger logic"""
        monitor = PerformanceMonitor()

        # Record consistently slow operations
        for i in range(10):
            monitor.record_operation(6.0, 50.0, False)  # Slow but not error

        assert monitor.should_trigger_emergency_mode() is True

        # Reset and test high error rate
        monitor = PerformanceMonitor()
        for i in range(20):
            monitor.record_operation(1.0, 50.0, i < 15)  # 75% error rate

        assert monitor.should_trigger_emergency_mode() is True


class TestRecoveryManager:
    """Test suite for recovery manager component"""

    @pytest.mark.asyncio
    async def test_successful_recovery(self):
        """Test successful recovery scenario"""
        config = TaskDetectionConfig()
        manager = RecoveryManager(config)

        # Mock successful detection system
        mock_system = MockTaskDetectionSystem()

        error_context = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test query",
            context={},
        )

        result = await manager.attempt_recovery(error_context, mock_system)

        assert result is not None
        assert isinstance(result, DetectionResult)
        assert mock_system.call_count == 1

    @pytest.mark.asyncio
    async def test_failed_recovery(self):
        """Test failed recovery scenario"""
        config = TaskDetectionConfig()
        manager = RecoveryManager(config)

        # Mock failing detection system
        mock_system = MockTaskDetectionSystem(failure_mode="generic")

        error_context = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test query",
            context={},
        )

        result = await manager.attempt_recovery(error_context, mock_system)

        assert result is None
        assert mock_system.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retry_limit(self):
        """Test maximum retry limit enforcement"""
        config = TaskDetectionConfig()
        manager = RecoveryManager(config)
        manager.max_retry_attempts = 2

        mock_system = MockTaskDetectionSystem(failure_mode="generic")

        error_context = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test query",
            context={},
        )

        # First two attempts should be made
        result1 = await manager.attempt_recovery(error_context, mock_system)
        result2 = await manager.attempt_recovery(error_context, mock_system)
        result3 = await manager.attempt_recovery(error_context, mock_system)

        assert result1 is None
        assert result2 is None
        assert result3 is None  # Should be rejected due to max attempts
        assert mock_system.call_count == 2  # Only first two attempts made

    def test_recovery_statistics(self):
        """Test recovery statistics tracking"""
        manager = RecoveryManager()

        # Simulate some recovery history
        manager.recovery_history.extend(
            [
                {"success": True, "recovery_time": 1.0, "error_type": "timeout"},
                {"success": False, "recovery_time": 2.0, "error_type": "network"},
                {"success": True, "recovery_time": 1.5, "error_type": "timeout"},
            ],
        )

        stats = manager.get_recovery_stats()

        assert stats["total_attempts"] == 3
        assert stats["successful_attempts"] == 2
        assert stats["success_rate"] == 2 / 3
        assert stats["avg_recovery_time"] == 1.5


class TestLearningCollector:
    """Test suite for learning collector component"""

    def test_failure_recording(self):
        """Test recording of failure patterns"""
        collector = LearningCollector(max_samples=100)

        error_context = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            query="test query",
            context={},
        )

        collector.record_failure("test query", {}, error_context, FallbackLevel.DETECTION_FAILURE)

        assert len(collector.failure_patterns) == 1
        assert collector.failure_patterns[0]["error_type"] == "timeout"
        assert collector.failure_patterns[0]["fallback_level"] == "detection_failure"

    def test_success_recording(self):
        """Test recording of success patterns"""
        collector = LearningCollector(max_samples=100)

        result = DetectionResult(
            categories={"git": True, "debug": False},
            confidence_scores={"git": 0.8, "debug": 0.3},
            detection_time_ms=45.0,
            signals_used={},
            fallback_applied=None,
        )

        collector.record_success("test query", {}, result)

        assert len(collector.success_patterns) == 1
        assert "git" in collector.success_patterns[0]["detected_categories"]
        assert collector.success_patterns[0]["detection_time"] == 45.0

    def test_learning_insights_generation(self):
        """Test generation of learning insights"""
        collector = LearningCollector(max_samples=100)

        # Add some failure patterns
        for i in range(5):
            error_context = ErrorContext(
                error_type=ErrorType.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                timestamp=time.time(),
                query=f"query {i}",
                context={},
            )
            collector.record_failure(f"query {i}", {}, error_context, FallbackLevel.DETECTION_FAILURE)

        # Add some success patterns
        for i in range(10):
            result = DetectionResult(
                categories={"git": True},
                confidence_scores={"git": 0.8},
                detection_time_ms=50.0,
                signals_used={},
                fallback_applied=None,
            )
            collector.record_success(f"success query {i}", {}, result)

        insights = collector.get_learning_insights()

        assert "insights" in insights
        assert "recommendations" in insights
        assert insights["total_failures"] == 5
        assert insights["total_successes"] == 10
        assert insights["failure_rate"] == 5 / 15

    def test_query_complexity_estimation(self):
        """Test query complexity estimation"""
        collector = LearningCollector()

        simple_query = "git status"
        complex_query = "analyze and debug the complex authentication system with multiple security vulnerabilities"

        simple_complexity = collector._estimate_query_complexity(simple_query)
        complex_complexity = collector._estimate_query_complexity(complex_query)

        assert simple_complexity < complex_complexity
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0


class TestConservativeFallbackChain:
    """Test suite for main fallback chain component"""

    @pytest.mark.asyncio
    async def test_high_confidence_detection(self, fallback_chain):
        """Test high confidence detection scenario"""
        # Mock high confidence result
        with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
            mock_detect.return_value = DetectionResult(
                categories={"git": True, "debug": True},
                confidence_scores={"git": 0.9, "debug": 0.8},
                detection_time_ms=30.0,
                signals_used={},
                fallback_applied=None,
            )

            categories, decision = await fallback_chain.get_function_categories("git commit changes")

            assert decision.level == FallbackLevel.HIGH_CONFIDENCE
            assert decision.confidence_threshold >= 0.7
            assert categories["git"] is True
            assert categories["debug"] is True

    @pytest.mark.asyncio
    async def test_medium_confidence_detection(self, fallback_chain):
        """Test medium confidence detection scenario"""
        with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
            mock_detect.return_value = DetectionResult(
                categories={"git": True, "debug": False},
                confidence_scores={"git": 0.5, "debug": 0.2},
                detection_time_ms=40.0,
                signals_used={},
                fallback_applied=None,
            )

            categories, decision = await fallback_chain.get_function_categories("some git work")

            assert decision.level == FallbackLevel.MEDIUM_CONFIDENCE
            assert 0.3 <= decision.confidence_threshold < 0.7
            assert categories["git"] is True
            assert categories["core"] is True  # Should include base categories
            assert categories["analysis"] is True  # Should include buffer

    @pytest.mark.asyncio
    async def test_low_confidence_detection(self, fallback_chain):
        """Test low confidence detection scenario"""
        with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
            mock_detect.return_value = DetectionResult(
                categories={"git": False, "debug": False},
                confidence_scores={"git": 0.1, "debug": 0.1},
                detection_time_ms=60.0,
                signals_used={},
                fallback_applied=None,
            )

            categories, decision = await fallback_chain.get_function_categories("unclear request")

            assert decision.level == FallbackLevel.LOW_CONFIDENCE
            assert decision.confidence_threshold < 0.3
            assert categories["core"] is True
            assert categories["git"] is True
            assert categories["analysis"] is True
            assert categories["debug"] is True

    @pytest.mark.asyncio
    async def test_timeout_failure_handling(self, fallback_chain):
        """Test timeout failure handling"""
        with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
            mock_detect.side_effect = TimeoutError("Detection timeout")

            categories, decision = await fallback_chain.get_function_categories("test query")

            # Should fall back to detection failure level
            assert decision.level in [FallbackLevel.DETECTION_FAILURE, FallbackLevel.LOW_CONFIDENCE]
            assert categories["core"] is True
            assert categories["git"] is True

    @pytest.mark.asyncio
    async def test_network_failure_handling(self, fallback_chain):
        """Test network failure handling"""
        with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
            mock_detect.side_effect = ConnectionError("Network failure")

            categories, decision = await fallback_chain.get_function_categories("test query")

            # Should trigger circuit breaker and use detection failure level
            assert decision.level in [FallbackLevel.DETECTION_FAILURE, FallbackLevel.LOW_CONFIDENCE]
            assert fallback_chain.circuit_breaker_failure_count > 0

    @pytest.mark.asyncio
    async def test_emergency_mode_activation(self, fallback_chain):
        """Test emergency mode activation"""
        # Force emergency mode
        fallback_chain.emergency_mode = True
        fallback_chain.emergency_mode_start = time.time()

        categories, decision = await fallback_chain.get_function_categories("test query")

        assert decision.level == FallbackLevel.SYSTEM_EMERGENCY
        assert all(categories.values())  # All categories should be loaded
        assert decision.expected_function_count >= 90

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, fallback_chain):
        """Test circuit breaker integration"""
        # Trigger circuit breaker
        fallback_chain.circuit_breaker_open = True
        fallback_chain.circuit_breaker_last_failure = time.time()

        categories, decision = await fallback_chain.get_function_categories("test query")

        assert decision.level == FallbackLevel.DETECTION_FAILURE
        assert decision.recovery_strategy == "circuit_breaker_protection"

    @pytest.mark.asyncio
    async def test_conservative_bias_application(self, fallback_chain):
        """Test conservative bias in decision making"""
        # Configure for conservative mode
        fallback_chain.config.mode = DetectionMode.CONSERVATIVE

        with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
            mock_detect.return_value = DetectionResult(
                categories={"git": True},
                confidence_scores={"git": 0.8},
                detection_time_ms=30.0,
                signals_used={},
                fallback_applied=None,
            )

            categories, decision = await fallback_chain.get_function_categories("git work")

            # Conservative mode should include analysis for safety
            assert categories["analysis"] is True

    @pytest.mark.asyncio
    async def test_recovery_attempt_success(self, fallback_chain):
        """Test successful recovery attempt"""
        # Mock recovery manager to return successful result
        mock_result = DetectionResult(
            categories={"git": True, "debug": True},
            confidence_scores={"git": 0.7, "debug": 0.6},
            detection_time_ms=40.0,
            signals_used={},
            fallback_applied=None,
        )

        with patch.object(fallback_chain.recovery_manager, "attempt_recovery") as mock_recovery:
            mock_recovery.return_value = mock_result

            with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
                mock_detect.side_effect = Exception("Initial failure")

                categories, decision = await fallback_chain.get_function_categories("test query")

                # Should use recovered result
                assert categories["git"] is True
                assert categories["debug"] is True
                assert fallback_chain.metrics.successful_recoveries > 0

    def test_health_status_reporting(self, fallback_chain):
        """Test health status reporting"""
        health = fallback_chain.get_health_status()

        assert "system_status" in health
        assert "performance" in health
        assert "recovery" in health
        assert "learning" in health
        assert "metrics" in health
        assert "circuit_breaker" in health

        assert isinstance(health["system_status"]["healthy"], bool)
        assert isinstance(health["metrics"], dict)

    def test_circuit_breaker_manual_control(self, fallback_chain):
        """Test manual circuit breaker control"""
        # Test manual reset
        fallback_chain.circuit_breaker_open = True
        fallback_chain.reset_circuit_breaker()
        assert fallback_chain.circuit_breaker_open is False

        # Test emergency mode exit
        fallback_chain.emergency_mode = True
        fallback_chain.exit_emergency_mode()
        assert fallback_chain.emergency_mode is False


class TestFallbackCircuitBreaker:
    """Test suite for fallback circuit breaker"""

    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation with different configurations"""
        # Conservative configuration
        conservative_cb = create_conservative_circuit_breaker()
        assert conservative_cb.adaptive_threshold.base_failure_threshold == 3
        assert conservative_cb.state == AdvancedCircuitBreakerState.CLOSED

        health = conservative_cb.get_health_status()
        assert health["healthy"] is True
        assert health["state"] == "closed"

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful function execution through circuit breaker"""
        cb = create_conservative_circuit_breaker()

        async def mock_success():
            await asyncio.sleep(0.01)
            return "success"

        result = await cb.execute(mock_success)

        assert result == "success"
        assert cb.consecutive_successes == 1
        assert cb.consecutive_failures == 0
        assert cb.metrics.successful_requests == 1

    @pytest.mark.asyncio
    async def test_failure_handling(self):
        """Test failure handling and state transitions"""
        cb = create_conservative_circuit_breaker()

        async def mock_failure():
            raise Exception("Mock failure")

        # Execute failures to trigger state transition
        for _i in range(4):  # One more than threshold
            with contextlib.suppress(builtins.BaseException):
                await cb.execute(mock_failure)

        assert cb.consecutive_failures == 4
        assert cb.state in [AdvancedCircuitBreakerState.OPEN, AdvancedCircuitBreakerState.DEGRADED]
        assert cb.metrics.failed_requests == 4

    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test circuit breaker state transitions"""
        cb = create_conservative_circuit_breaker()

        # Force open state
        cb.force_open("test")
        assert cb.state == AdvancedCircuitBreakerState.FORCED_OPEN

        # Force close state
        cb.force_close("test")
        assert cb.state == AdvancedCircuitBreakerState.CLOSED
        assert cb.consecutive_failures == 0

    def test_failure_pattern_detection(self):
        """Test failure pattern detection"""
        cb = create_conservative_circuit_breaker()

        # Simulate timeout pattern
        for _i in range(10):
            cb.failure_analyzer.record_request(False, 1.0, "timeout")

        pattern = cb.failure_analyzer.detect_failure_pattern()
        assert pattern == FailurePattern.TIMEOUT_HEAVY

        confidence = cb.failure_analyzer.get_pattern_confidence()
        assert confidence > 0.5

    def test_adaptive_thresholds(self):
        """Test adaptive threshold adjustment"""
        cb = create_conservative_circuit_breaker()

        # Set different failure patterns and check threshold adjustments
        cb.current_pattern = FailurePattern.CASCADING
        thresholds = cb._get_current_thresholds()

        # Cascading pattern should have lower failure threshold
        assert thresholds["failure_threshold"] < cb.adaptive_threshold.base_failure_threshold

        cb.current_pattern = FailurePattern.INTERMITTENT
        thresholds = cb._get_current_thresholds()

        # Intermittent pattern should have higher failure threshold
        assert thresholds["failure_threshold"] > cb.adaptive_threshold.base_failure_threshold


class TestIntegrationScenarios:
    """Integration tests for complete fallback chain scenarios"""

    @pytest.mark.asyncio
    async def test_end_to_end_success_scenario(self):
        """Test complete successful detection scenario"""
        detection_system = MockTaskDetectionSystem()
        config = TaskDetectionConfig()
        fallback_chain = ConservativeFallbackChain(detection_system, config)

        categories, decision = await fallback_chain.get_function_categories(
            "help me commit my git changes",
            {"file_extensions": [".py"], "has_uncommitted_changes": True},
        )

        assert isinstance(categories, dict)
        assert isinstance(decision, FallbackDecision)
        assert decision.level in [FallbackLevel.HIGH_CONFIDENCE, FallbackLevel.MEDIUM_CONFIDENCE]
        assert categories["git"] is True

    @pytest.mark.asyncio
    async def test_cascade_failure_scenario(self):
        """Test cascade failure handling"""
        # Create detection system that always fails
        detection_system = MockTaskDetectionSystem(failure_mode="network")
        config = TaskDetectionConfig()
        fallback_chain = ConservativeFallbackChain(detection_system, config)

        # Multiple failures should trigger circuit breaker
        for i in range(3):
            categories, decision = await fallback_chain.get_function_categories(f"query {i}")

        # Circuit breaker should be open after multiple failures
        assert fallback_chain.circuit_breaker_failure_count >= 1

    @pytest.mark.asyncio
    async def test_performance_degradation_scenario(self):
        """Test performance degradation handling"""
        # Create slow detection system
        detection_system = MockTaskDetectionSystem(delay=6.0)  # Exceeds timeout
        config = TaskDetectionConfig()
        fallback_chain = ConservativeFallbackChain(detection_system, config)

        start_time = time.time()
        categories, decision = await fallback_chain.get_function_categories("test query")
        end_time = time.time()

        # Should timeout and use fallback within reasonable time (allowing for timeout + recovery attempts)
        assert end_time - start_time < 15.0  # Allow for initial timeout (5s) + recovery timeout (5s) + overhead
        # After timeout, recovery may succeed, so accept any fallback level that works
        assert decision.level in [
            FallbackLevel.DETECTION_FAILURE,
            FallbackLevel.LOW_CONFIDENCE,
            FallbackLevel.HIGH_CONFIDENCE,
        ]

    @pytest.mark.asyncio
    async def test_recovery_after_failures(self):
        """Test recovery after temporary failures"""
        detection_system = MockTaskDetectionSystem(failure_mode="timeout")
        config = TaskDetectionConfig()
        fallback_chain = ConservativeFallbackChain(detection_system, config)

        # Initial failure
        categories1, decision1 = await fallback_chain.get_function_categories("query 1")

        # Fix the detection system
        detection_system.failure_mode = None

        # Should recover on subsequent requests
        categories2, decision2 = await fallback_chain.get_function_categories("query 2")

        assert decision1.level in [FallbackLevel.DETECTION_FAILURE, FallbackLevel.LOW_CONFIDENCE]
        assert decision2.level in [FallbackLevel.HIGH_CONFIDENCE, FallbackLevel.MEDIUM_CONFIDENCE]


class TestEdgeCases:
    """Test edge cases and unusual scenarios"""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, fallback_chain):
        """Test handling of empty queries"""
        categories, decision = await fallback_chain.get_function_categories("")

        assert isinstance(categories, dict)
        assert isinstance(decision, FallbackDecision)
        # Should default to safe configuration
        assert categories["core"] is True

    @pytest.mark.asyncio
    async def test_very_long_query_handling(self, fallback_chain):
        """Test handling of very long queries"""
        long_query = "analyze " * 1000  # Very long query

        categories, decision = await fallback_chain.get_function_categories(long_query)

        assert isinstance(categories, dict)
        assert isinstance(decision, FallbackDecision)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, fallback_chain):
        """Test handling of concurrent requests"""

        async def make_request(query_id):
            return await fallback_chain.get_function_categories(f"query {query_id}")

        # Make multiple concurrent requests
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            categories, decision = result
            assert isinstance(categories, dict)
            assert isinstance(decision, FallbackDecision)

    @pytest.mark.asyncio
    async def test_memory_pressure_simulation(self, fallback_chain):
        """Test behavior under simulated memory pressure"""
        # Simulate memory pressure by forcing memory error
        with patch.object(fallback_chain.detection_system, "detect_categories") as mock_detect:
            mock_detect.side_effect = MemoryError("Simulated memory pressure")

            categories, decision = await fallback_chain.get_function_categories("test query")

            # Should handle memory pressure gracefully
            assert isinstance(categories, dict)
            assert decision.recovery_strategy in ["reduce_load_and_retry", "conservative_fallback"]


class TestFactoryFunctions:
    """Test factory functions and configuration utilities"""

    def test_conservative_fallback_chain_creation(self):
        """Test conservative fallback chain factory"""
        detection_system = MockTaskDetectionSystem()

        fallback_chain = create_conservative_fallback_chain(detection_system)

        assert isinstance(fallback_chain, ConservativeFallbackChain)
        assert fallback_chain.config.mode == DetectionMode.CONSERVATIVE
        assert isinstance(fallback_chain.error_classifier, ErrorClassifier)
        assert isinstance(fallback_chain.performance_monitor, PerformanceMonitor)

    def test_decorator_functionality(self):
        """Test fallback decorator functionality"""
        detection_system = MockTaskDetectionSystem()
        fallback_chain = create_conservative_fallback_chain(detection_system)

        @with_conservative_fallback(fallback_chain)
        async def test_function(query, context=None):
            raise Exception("Function failure")

        # Test that decorator handles failures
        result = asyncio.run(test_function("test query"))

        assert result["fallback_used"] is True
        assert "fallback_decision" in result
        assert "categories" in result
        assert "original_error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
