"""Unit tests for circuit breaker implementation.

Tests the CircuitBreaker class and related functionality for handling
external service failures with configurable thresholds and recovery.
"""

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    create_circuit_breaker_config_from_settings,
    get_all_circuit_breakers,
    get_circuit_breaker,
    register_circuit_breaker,
    reset_all_circuit_breakers,
    start_all_health_monitoring,
    stop_all_health_monitoring,
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig validation and behavior."""

    def test_valid_config_creation(self):
        """Test creating config with valid parameters."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=30,
            max_retries=2,
            base_delay=0.5,
            max_delay=30.0,
            backoff_multiplier=1.5,
        )

        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.recovery_timeout == 30
        assert config.max_retries == 2
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 1.5

    def test_default_config_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.recovery_timeout == 60
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.enable_metrics is True

    def test_invalid_failure_threshold(self):
        """Test validation of failure threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_invalid_success_threshold(self):
        """Test validation of success threshold."""
        with pytest.raises(ValueError, match="success_threshold must be positive"):
            CircuitBreakerConfig(success_threshold=-1)

    def test_invalid_recovery_timeout(self):
        """Test validation of recovery timeout."""
        with pytest.raises(ValueError, match="recovery_timeout must be positive"):
            CircuitBreakerConfig(recovery_timeout=0)

    def test_invalid_max_retries(self):
        """Test validation of max retries."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            CircuitBreakerConfig(max_retries=-1)

    def test_invalid_base_delay(self):
        """Test validation of base delay."""
        with pytest.raises(ValueError, match="base_delay must be positive"):
            CircuitBreakerConfig(base_delay=0)

    def test_invalid_backoff_multiplier(self):
        """Test validation of backoff multiplier."""
        with pytest.raises(ValueError, match="backoff_multiplier must be greater than 1"):
            CircuitBreakerConfig(backoff_multiplier=1.0)


class TestCircuitBreakerMetrics:
    """Test CircuitBreakerMetrics calculations."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = CircuitBreakerMetrics()

        # No requests
        assert metrics.get_success_rate() == 100.0

        # Some requests
        metrics.total_requests = 10
        metrics.successful_requests = 8
        assert metrics.get_success_rate() == 80.0

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        metrics = CircuitBreakerMetrics()

        # No requests
        assert metrics.get_failure_rate() == 0.0

        # Some requests
        metrics.total_requests = 10
        metrics.failed_requests = 3
        assert metrics.get_failure_rate() == 30.0


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions and behavior."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=1,  # Short timeout for testing
            max_retries=2,
            base_delay=0.1,
            enable_metrics=False,  # Disable to avoid dependency issues
            enable_tracing=False,
        )
        return CircuitBreaker("test", config)

    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        metrics = circuit_breaker.metrics
        assert metrics.total_requests == 0
        assert metrics.consecutive_failures == 0

    def test_successful_call_sync(self, circuit_breaker):
        """Test successful synchronous call."""

        def success_func():
            return "success"

        result = circuit_breaker.call_sync(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        metrics = circuit_breaker.metrics
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_successful_call_async(self, circuit_breaker):
        """Test successful asynchronous call."""

        async def success_func():
            return "async_success"

        result = await circuit_breaker.call_async(success_func)
        assert result == "async_success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        metrics = circuit_breaker.metrics
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1

    def test_failing_calls_transition_to_open(self, circuit_breaker):
        """Test that multiple failures cause transition to OPEN state."""

        def failing_func():
            raise Exception("Service unavailable")

        # First 2 failures should keep circuit closed
        for _i in range(2):
            with pytest.raises(Exception, match="Service unavailable"):
                circuit_breaker.call_sync(failing_func)
            assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Third failure should open the circuit
        with pytest.raises(Exception, match="Service unavailable"):
            circuit_breaker.call_sync(failing_func)
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        metrics = circuit_breaker.metrics
        assert metrics.total_requests == 3
        assert metrics.failed_requests == 3
        assert metrics.consecutive_failures == 3

    def test_open_circuit_rejects_requests(self, circuit_breaker):
        """Test that open circuit rejects requests."""
        # Force circuit to open state
        circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        def success_func():
            return "should_not_execute"

        with pytest.raises(CircuitBreakerOpenError):
            circuit_breaker.call_sync(success_func)

        metrics = circuit_breaker.metrics
        assert metrics.rejected_requests == 1

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_half_open_recovery(self, circuit_breaker):
        """Test recovery through half-open state."""
        # Force circuit to open
        circuit_breaker.force_open()

        # Wait for recovery timeout - use longer timeout to ensure recovery
        await asyncio.sleep(2.0)  # Well beyond recovery timeout (recovery_timeout=1)

        async def success_func():
            return "recovery_success"

        # First call should transition to half-open and succeed
        result = await circuit_breaker.call_async(success_func)
        assert result == "recovery_success"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Second successful call should close the circuit
        result = await circuit_breaker.call_async(success_func)
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_decorator_functionality(self, circuit_breaker):
        """Test circuit breaker decorator."""

        @circuit_breaker.decorator
        def decorated_func(value):
            if value == "fail":
                raise Exception("Decorated failure")
            return f"decorated_{value}"

        # Test success
        result = decorated_func("success")
        assert result == "decorated_success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Test failure
        with pytest.raises(Exception, match="Decorated failure"):
            decorated_func("fail")

    @pytest.mark.asyncio
    async def test_async_decorator_functionality(self, circuit_breaker):
        """Test circuit breaker decorator with async function."""

        @circuit_breaker.decorator
        async def async_decorated_func(value):
            if value == "fail":
                raise Exception("Async decorated failure")
            return f"async_decorated_{value}"

        # Test success
        result = await async_decorated_func("success")
        assert result == "async_decorated_success"

        # Test failure
        with pytest.raises(Exception, match="Async decorated failure"):
            await async_decorated_func("fail")

    def test_thread_safety(self, circuit_breaker):
        """Test thread safety of circuit breaker."""
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):

                    def test_func():
                        return f"worker_{worker_id}_call_{i}"  # noqa: B023

                    result = circuit_breaker.call_sync(test_func)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0
        assert len(results) == 50  # 5 workers x 10 calls each

        metrics = circuit_breaker.metrics
        assert metrics.total_requests == 50
        assert metrics.successful_requests == 50

    def test_reset_functionality(self, circuit_breaker):
        """Test circuit breaker reset."""
        # Generate some activity
        circuit_breaker.call_sync(lambda: "success")

        def failing_func():
            raise Exception("Failure")

        with pytest.raises(Exception, match="Failure"):
            circuit_breaker.call_sync(failing_func)

        # Reset and verify
        circuit_breaker.reset()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        metrics = circuit_breaker.metrics
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0

    def test_force_state_changes(self, circuit_breaker):
        """Test forcing circuit breaker state changes."""
        # Test force open
        circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Test force close
        circuit_breaker.force_close()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_health_status_reporting(self, circuit_breaker):
        """Test health status reporting."""
        status = circuit_breaker.get_health_status()

        assert status["name"] == "test"
        assert status["healthy"] is True  # CLOSED state is healthy
        assert status["state"] == "closed"
        assert "metrics" in status
        assert "timing" in status
        assert "config" in status

        # Test unhealthy state
        circuit_breaker.force_open()
        status = circuit_breaker.get_health_status()
        assert status["healthy"] is False  # OPEN state is unhealthy


class TestCircuitBreakerRetries:
    """Test retry functionality with exponential backoff."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with retry configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=10,  # High threshold to test retries
            max_retries=3,
            base_delay=0.01,  # Very short delay for testing
            backoff_multiplier=2.0,
            enable_metrics=False,
            enable_tracing=False,
        )
        return CircuitBreaker("retry_test", config)

    @pytest.mark.asyncio
    async def test_retry_with_eventual_success(self, circuit_breaker):
        """Test retry mechanism with eventual success."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure {call_count}")
            return "success"

        result = await circuit_breaker.call_async(flaky_func)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, circuit_breaker):
        """Test retry exhaustion."""
        call_count = 0

        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception(f"Failure {call_count}")

        with pytest.raises(Exception, match="Failure"):
            await circuit_breaker.call_async(always_failing_func)

        # Should have tried: initial + 3 retries = 4 total
        assert call_count == 4


class TestCircuitBreakerHealthChecks:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_function(self):
        """Test health check function execution."""
        health_check_called = False

        async def health_check():
            nonlocal health_check_called
            health_check_called = True
            return True

        circuit_breaker = CircuitBreaker(
            "health_test",
            health_check_func=health_check,
        )

        result = await circuit_breaker.health_check()
        assert result is True
        assert health_check_called is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure handling."""

        async def failing_health_check():
            raise Exception("Health check failed")

        circuit_breaker = CircuitBreaker(
            "health_fail_test",
            health_check_func=failing_health_check,
        )

        result = await circuit_breaker.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_monitoring_lifecycle(self):
        """Test health monitoring start/stop lifecycle."""
        health_check_count = 0

        async def counting_health_check():
            nonlocal health_check_count
            health_check_count += 1
            return True

        config = CircuitBreakerConfig(
            health_check_interval=0.1,  # Very short interval for testing
        )

        circuit_breaker = CircuitBreaker(
            "monitoring_test",
            config=config,
            health_check_func=counting_health_check,
        )

        # Start monitoring
        await circuit_breaker.start_health_monitoring()

        # Wait for a few health checks
        await asyncio.sleep(0.25)

        # Stop monitoring
        await circuit_breaker.stop_health_monitoring()

        # Should have performed at least 2 health checks
        assert health_check_count >= 2


class TestCircuitBreakerFactory:
    """Test factory functions and configuration."""

    def test_create_config_from_settings(self):
        """Test creating config from application settings."""
        # Mock settings object
        settings = Mock()
        settings.circuit_breaker_failure_threshold = 7
        settings.circuit_breaker_success_threshold = 4
        settings.circuit_breaker_recovery_timeout = 90
        settings.circuit_breaker_max_retries = 5
        settings.circuit_breaker_base_delay = 2.0
        settings.circuit_breaker_max_delay = 120.0
        settings.circuit_breaker_backoff_multiplier = 3.0
        settings.circuit_breaker_jitter_enabled = False
        settings.circuit_breaker_health_check_interval = 45
        settings.circuit_breaker_health_check_timeout = 10.0
        settings.performance_monitoring_enabled = True

        config = create_circuit_breaker_config_from_settings(settings)

        assert config.failure_threshold == 7
        assert config.success_threshold == 4
        assert config.recovery_timeout == 90
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_multiplier == 3.0
        assert config.jitter is False
        assert config.health_check_interval == 45
        assert config.health_check_timeout == 10.0
        assert config.enable_metrics is True

    @patch("src.utils.circuit_breaker._circuit_breakers", {})
    def test_circuit_breaker_registry(self):
        """Test circuit breaker registry functionality."""
        # Create a circuit breaker
        circuit_breaker = CircuitBreaker("registry_test")

        # Register it
        register_circuit_breaker("registry_test", circuit_breaker)

        # Retrieve it
        retrieved = get_circuit_breaker("registry_test")
        assert retrieved is circuit_breaker

        # Test get all
        all_breakers = get_all_circuit_breakers()
        assert "registry_test" in all_breakers
        assert all_breakers["registry_test"] is circuit_breaker

    @patch("src.utils.circuit_breaker._circuit_breakers", {})
    def test_get_circuit_breaker_with_settings(self):
        """Test getting circuit breaker with settings."""
        settings = Mock()
        settings.circuit_breaker_failure_threshold = 5
        settings.circuit_breaker_success_threshold = 3
        settings.circuit_breaker_recovery_timeout = 60
        settings.circuit_breaker_max_retries = 3
        settings.circuit_breaker_base_delay = 1.0
        settings.circuit_breaker_max_delay = 60.0
        settings.circuit_breaker_backoff_multiplier = 2.0
        settings.circuit_breaker_jitter_enabled = True
        settings.circuit_breaker_health_check_interval = 30
        settings.circuit_breaker_health_check_timeout = 5.0
        settings.performance_monitoring_enabled = True

        # Get non-existent circuit breaker (should create it)
        circuit_breaker = get_circuit_breaker("new_breaker", settings)
        assert circuit_breaker is not None
        assert circuit_breaker.name == "new_breaker"

        # Get same circuit breaker again (should return cached)
        same_breaker = get_circuit_breaker("new_breaker")
        assert same_breaker is circuit_breaker

    @patch("src.utils.circuit_breaker._circuit_breakers", {})
    @pytest.mark.asyncio
    async def test_global_health_monitoring(self):
        """Test global health monitoring functions."""
        # Create and register a few circuit breakers
        cb1 = CircuitBreaker("global_test_1")
        cb2 = CircuitBreaker("global_test_2")

        register_circuit_breaker("global_test_1", cb1)
        register_circuit_breaker("global_test_2", cb2)

        # Test start all health monitoring
        await start_all_health_monitoring()

        # Test stop all health monitoring
        await stop_all_health_monitoring()

    @patch("src.utils.circuit_breaker._circuit_breakers", {})
    def test_reset_all_circuit_breakers(self):
        """Test resetting all circuit breakers."""
        # Create and register circuit breakers
        cb1 = CircuitBreaker("reset_test_1")
        cb2 = CircuitBreaker("reset_test_2")

        register_circuit_breaker("reset_test_1", cb1)
        register_circuit_breaker("reset_test_2", cb2)

        # Add some activity
        cb1.call_sync(lambda: "success")
        cb2.call_sync(lambda: "success")

        # Reset all
        reset_all_circuit_breakers()

        # Verify reset
        assert cb1.metrics.total_requests == 0
        assert cb2.metrics.total_requests == 0


class TestCircuitBreakerErrors:
    """Test circuit breaker error handling and exceptions."""

    def test_circuit_breaker_error_creation(self):
        """Test CircuitBreakerError creation."""

        now = datetime.now(UTC)
        error = CircuitBreakerError(
            "Test error",
            CircuitBreakerState.OPEN,
            failure_count=5,
            last_failure_time=now,
        )

        assert str(error) == "Test error"
        assert error.state == CircuitBreakerState.OPEN
        assert error.failure_count == 5
        assert error.last_failure_time == now

    def test_circuit_breaker_open_error_creation(self):
        """Test CircuitBreakerOpenError creation."""

        now = datetime.now(UTC)
        recovery_time = now + timedelta(seconds=60)

        error = CircuitBreakerOpenError(
            failure_count=3,
            last_failure_time=now,
            recovery_time=recovery_time,
        )

        assert error.state == CircuitBreakerState.OPEN
        assert error.failure_count == 3
        assert error.recovery_time == recovery_time
        assert "Circuit breaker is open" in str(error)


@pytest.mark.integration
class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with real scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_realistic_service_outage_scenario(self):
        """Test realistic service outage and recovery scenario."""
        # Configure circuit breaker for realistic scenario
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=1,  # Short for testing
            max_retries=2,
            base_delay=0.1,
            enable_metrics=False,
            enable_tracing=False,
        )

        circuit_breaker = CircuitBreaker("integration_test", config)

        # Simulate service being healthy initially
        service_healthy = True
        call_count = 0

        async def simulated_service_call():
            nonlocal call_count
            call_count += 1

            if not service_healthy:
                raise Exception(f"Service unavailable (call {call_count})")
            return f"Service response {call_count}"

        # Phase 1: Normal operation
        result = await circuit_breaker.call_async(simulated_service_call)
        assert "Service response" in result
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Phase 2: Service goes down
        service_healthy = False

        # Multiple failures should open the circuit
        for _i in range(3):
            with pytest.raises(Exception, match="Service error"):
                await circuit_breaker.call_async(simulated_service_call)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Phase 3: Circuit is open, requests are rejected
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call_async(simulated_service_call)

        # Phase 4: Wait for recovery timeout
        await asyncio.sleep(1.5)

        # Phase 5: Service comes back online
        service_healthy = True

        # First call should transition to half-open
        result = await circuit_breaker.call_async(simulated_service_call)
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Second successful call should close the circuit
        result = await circuit_breaker.call_async(simulated_service_call)
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Verify metrics
        metrics = circuit_breaker.metrics
        assert metrics.total_requests > 0
        assert metrics.successful_requests > 0
        assert metrics.failed_requests > 0
