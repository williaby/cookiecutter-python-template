"""Unit tests for Zen MCP error handling system.

Tests the modular resilience strategies and their integration
with comprehensive coverage of error scenarios and state transitions.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.core.zen_mcp_error_handling import (
    CircuitBreakerStrategy,
    MockZenMCPClient,
    RetryStrategy,
    ZenMCPConnectionError,
    ZenMCPError,
    ZenMCPIntegration,
    ZenMCPTimeoutError,
    create_default_zen_mcp_integration,
    create_fast_fail_zen_mcp_integration,
    create_high_availability_zen_mcp_integration,
)
from src.utils.resilience import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    CompositeResilienceHandler,
    RetryConfig,
    RetryExhaustedError,
)


class TestCircuitBreakerStrategy:
    """Test circuit breaker resilience strategy."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,
            success_threshold=2,
        )
        self.circuit_breaker = CircuitBreakerStrategy(self.config)

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""

        async def success_func():
            return "success"

        result = await self.circuit_breaker.execute(success_func)
        assert result == "success"
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""

        async def failure_func():
            raise ZenMCPConnectionError("Connection failed")

        # Trigger enough failures to open circuit breaker
        for _ in range(3):
            with pytest.raises(ZenMCPError):
                await self.circuit_breaker.execute(failure_func)

        assert self.circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state_blocks_calls(self):
        """Test circuit breaker blocks calls when open."""
        # Force circuit breaker to open state with recent failure
        self.circuit_breaker.state = CircuitBreakerState.OPEN
        self.circuit_breaker.last_failure_time = time.time()  # Recent failure

        async def success_func():
            return "success"

        with pytest.raises(CircuitBreakerOpenError, match="Circuit breaker is OPEN"):
            await self.circuit_breaker.execute(success_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open recovery."""
        # Force circuit breaker to half-open state
        self.circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        self.circuit_breaker.success_count = 0

        async def success_func():
            return "success"

        # Execute successful calls to close circuit breaker
        for _ in range(2):  # success_threshold = 2
            result = await self.circuit_breaker.execute(success_func)
            assert result == "success"

        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure_reopens(self):
        """Test circuit breaker reopens on failure in half-open state."""
        self.circuit_breaker.state = CircuitBreakerState.HALF_OPEN

        async def failure_func():
            raise ZenMCPConnectionError("Connection failed")

        with pytest.raises(ZenMCPError):
            await self.circuit_breaker.execute(failure_func)

        assert self.circuit_breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_should_attempt_reset(self):
        """Test circuit breaker reset timing."""
        # Set circuit breaker to open with old failure time
        self.circuit_breaker.state = CircuitBreakerState.OPEN
        self.circuit_breaker.last_failure_time = time.time() - 2  # 2 seconds ago

        assert self.circuit_breaker._should_attempt_reset() is True

        # Set recent failure time
        self.circuit_breaker.last_failure_time = time.time()
        assert self.circuit_breaker._should_attempt_reset() is False

    def test_circuit_breaker_health_status(self):
        """Test circuit breaker health status."""
        status = self.circuit_breaker.get_health_status()

        assert "healthy" in status
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert status["state"] == CircuitBreakerState.CLOSED.value


class TestRetryStrategy:
    """Test retry resilience strategy."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        self.retry_strategy = RetryStrategy(self.config)

    @pytest.mark.asyncio
    async def test_retry_strategy_success_on_first_try(self):
        """Test retry strategy with immediate success."""

        async def success_func():
            return "success"

        result = await self.retry_strategy.execute(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_strategy_success_after_retries(self):
        """Test retry strategy with success after retries."""
        call_count = 0

        async def eventually_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ZenMCPConnectionError("Temporary failure")
            return "success"

        result = await self.retry_strategy.execute(eventually_success_func)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_strategy_all_retries_fail(self):
        """Test retry strategy when all retries fail."""

        async def failure_func():
            raise ZenMCPConnectionError("Persistent failure")

        with pytest.raises(RetryExhaustedError):
            await self.retry_strategy.execute(failure_func)

    def test_retry_strategy_calculate_delay(self):
        """Test retry delay calculation."""
        # Mock secure random to return predictable values
        mock_rng = MagicMock()
        mock_rng.exponential_backoff_jitter.return_value = 1.0

        self.retry_strategy.secure_rng = mock_rng
        delay = self.retry_strategy._calculate_delay(0)

        # Verify the secure random method was called correctly
        mock_rng.exponential_backoff_jitter.assert_called_once_with(
            self.config.base_delay,
            0,
            self.config.max_delay,
        )
        assert delay == 1.0

    def test_retry_strategy_health_status(self):
        """Test retry strategy health status."""
        status = self.retry_strategy.get_health_status()

        assert status["healthy"] is True
        assert status["max_retries"] == 3
        assert "base_delay" in status
        assert "max_delay" in status

    @pytest.mark.asyncio
    async def test_retry_strategy_non_retryable_exception(self):
        """Test retry strategy with non-retryable exception."""
        config = RetryConfig(
            max_retries=3,
            retryable_exceptions=(ZenMCPConnectionError,),
        )
        retry_strategy = RetryStrategy(config)

        async def timeout_func():
            raise ZenMCPTimeoutError("Timeout error")

        # Should not retry timeout errors with this config
        with pytest.raises(ZenMCPTimeoutError):
            await retry_strategy.execute(timeout_func)


class TestCompositeResilienceHandler:
    """Test composite resilience handler."""

    def setup_method(self):
        """Setup test fixtures."""
        self.circuit_breaker = CircuitBreakerStrategy()
        self.retry_strategy = RetryStrategy()
        self.handler = CompositeResilienceHandler(
            [self.circuit_breaker, self.retry_strategy],
        )

    @pytest.mark.asyncio
    async def test_handler_primary_success(self):
        """Test handler with primary function success."""

        async def success_func():
            return "success"

        result = await self.handler.execute_with_protection(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_handler_fallback_success(self):
        """Test handler with fallback function success."""

        async def failure_func():
            raise ZenMCPConnectionError("Primary failed")

        async def fallback_func():
            return "fallback_success"

        # Configure to fail fast for quick test
        circuit_config = CircuitBreakerConfig(failure_threshold=1)
        retry_config = RetryConfig(max_retries=1)

        circuit_breaker = CircuitBreakerStrategy(circuit_config)
        retry_strategy = RetryStrategy(retry_config)
        handler = CompositeResilienceHandler([circuit_breaker, retry_strategy])

        result = await handler.execute_with_protection(failure_func, fallback_func)
        assert result == "fallback_success"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_handler_fallback_failure(self):
        """Test handler when both primary and fallback fail."""

        async def failure_func():
            raise ZenMCPConnectionError("Primary failed")

        async def fallback_failure():
            raise ZenMCPError("Fallback failed")

        # Configure to fail fast
        circuit_config = CircuitBreakerConfig(failure_threshold=1)
        retry_config = RetryConfig(max_retries=1)

        circuit_breaker = CircuitBreakerStrategy(circuit_config)
        retry_strategy = RetryStrategy(retry_config)
        handler = CompositeResilienceHandler([circuit_breaker, retry_strategy])

        with pytest.raises(Exception, match="Both primary and fallback functions failed"):
            await handler.execute_with_protection(failure_func, fallback_failure)

    def test_handler_health_status(self):
        """Test handler health status."""
        status = self.handler.get_health_status()

        assert "strategies" in status
        assert "overall_healthy" in status
        assert len(status["strategies"]) == 2


class TestMockZenMCPClient:
    """Test mock Zen MCP client."""

    def setup_method(self):
        """Setup test fixtures."""
        # Use deterministic secure random for testing
        self.mock_rng = MagicMock()
        self.client = MockZenMCPClient(failure_rate=0.0, secure_rng=self.mock_rng)

    @pytest.mark.asyncio
    async def test_mock_client_success(self):
        """Test mock client successful operation."""
        self.mock_rng.random.return_value = 0.9  # > failure_rate, so success

        result = await self.client.process_prompt("test prompt")
        assert result["enhanced_prompt"] == "Enhanced: test prompt"
        assert result["metadata"]["call_count"] == 1

    @pytest.mark.asyncio
    async def test_mock_client_failure(self):
        """Test mock client failure simulation."""
        client = MockZenMCPClient(failure_rate=1.0, secure_rng=self.mock_rng)
        self.mock_rng.random.return_value = 0.1  # < failure_rate, so failure

        with pytest.raises(ZenMCPConnectionError):
            await client.process_prompt("test prompt")

    @pytest.mark.asyncio
    async def test_mock_client_call_count(self):
        """Test mock client call counting."""
        self.mock_rng.random.return_value = 0.9  # Success

        await self.client.process_prompt("test 1")
        await self.client.process_prompt("test 2")

        assert self.client.call_count == 2

    def test_mock_client_invalid_failure_rate(self):
        """Test mock client with invalid failure rate."""
        with pytest.raises(ValueError, match="failure_rate must be between 0.0 and 1.0"):
            MockZenMCPClient(failure_rate=1.5)

        with pytest.raises(ValueError, match="failure_rate must be between 0.0 and 1.0"):
            MockZenMCPClient(failure_rate=-0.1)


class TestZenMCPIntegration:
    """Test Zen MCP integration layer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_rng = MagicMock()
        self.client = MockZenMCPClient(failure_rate=0.0, secure_rng=self.mock_rng)
        self.integration = ZenMCPIntegration(self.client)

    @pytest.mark.asyncio
    async def test_integration_enhance_prompt_success(self):
        """Test integration prompt enhancement success."""
        self.mock_rng.random.return_value = 0.9  # Success

        result = await self.integration.enhance_prompt("test prompt")
        assert "enhanced_prompt" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_integration_enhance_prompt_fallback(self):
        """Test integration prompt enhancement with fallback."""
        # Use client with 100% failure rate
        client = MockZenMCPClient(failure_rate=1.0, secure_rng=self.mock_rng)
        self.mock_rng.random.return_value = 0.1  # Failure

        # Configure integration to fail fast
        circuit_config = CircuitBreakerConfig(failure_threshold=1)
        retry_config = RetryConfig(max_retries=1)

        circuit_breaker = CircuitBreakerStrategy(circuit_config)
        retry_strategy = RetryStrategy(retry_config)
        handler = CompositeResilienceHandler([circuit_breaker, retry_strategy])

        integration = ZenMCPIntegration(client, handler)

        result = await integration.enhance_prompt("test prompt")
        assert result["enhanced_prompt"] == "[FALLBACK] test prompt"
        assert result["metadata"]["fallback"] is True

    def test_integration_get_circuit_breaker_state(self):
        """Test integration circuit breaker state access."""
        state = self.integration.get_circuit_breaker_state()
        assert state == CircuitBreakerState.CLOSED

    def test_integration_get_failure_count(self):
        """Test integration failure count access."""
        count = self.integration.get_failure_count()
        assert count == 0

    def test_integration_health_status(self):
        """Test integration health status."""
        status = self.integration.get_health_status()
        assert "strategies" in status
        assert "overall_healthy" in status


class TestFactoryFunctions:
    """Test factory functions for common configurations."""

    def test_create_default_integration(self):
        """Test default integration factory."""
        integration = create_default_zen_mcp_integration()
        assert isinstance(integration, ZenMCPIntegration)
        assert len(integration.resilience_handler.strategies) == 2

    def test_create_high_availability_integration(self):
        """Test high availability integration factory."""
        integration = create_high_availability_zen_mcp_integration()
        assert isinstance(integration, ZenMCPIntegration)

        # Check that strategies use HA configuration
        for strategy in integration.resilience_handler.strategies:
            if isinstance(strategy, CircuitBreakerStrategy):
                assert strategy.config.failure_threshold == 3
                assert strategy.config.recovery_timeout == 30
            elif isinstance(strategy, RetryStrategy):
                assert strategy.config.max_retries == 5
                assert strategy.config.base_delay == 2.0

    def test_create_fast_fail_integration(self):
        """Test fast fail integration factory."""
        integration = create_fast_fail_zen_mcp_integration()
        assert isinstance(integration, ZenMCPIntegration)

        # Check that strategies use fast-fail configuration
        for strategy in integration.resilience_handler.strategies:
            if isinstance(strategy, CircuitBreakerStrategy):
                assert strategy.config.failure_threshold == 2
                assert strategy.config.recovery_timeout == 15
            elif isinstance(strategy, RetryStrategy):
                assert strategy.config.max_retries == 1
                assert strategy.config.base_delay == 0.5


class TestConfiguration:
    """Test configuration validation."""

    def test_circuit_breaker_config_defaults(self):
        """Test CircuitBreakerConfig default values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        assert config.success_threshold == 3

    def test_circuit_breaker_config_custom(self):
        """Test CircuitBreakerConfig custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=30,
            success_threshold=5,
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 30
        assert config.success_threshold == 5

    def test_circuit_breaker_config_validation(self):
        """Test CircuitBreakerConfig validation."""
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreakerConfig(failure_threshold=0)

        with pytest.raises(ValueError, match="recovery_timeout must be positive"):
            CircuitBreakerConfig(recovery_timeout=-1)

        with pytest.raises(ValueError, match="success_threshold must be positive"):
            CircuitBreakerConfig(success_threshold=0)

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_retry_config_custom(self):
        """Test RetryConfig custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_retry_config_validation(self):
        """Test RetryConfig validation."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-1)

        with pytest.raises(ValueError, match="base_delay must be positive"):
            RetryConfig(base_delay=0)

        with pytest.raises(ValueError, match="max_delay must be positive"):
            RetryConfig(max_delay=-1)

        with pytest.raises(ValueError, match="exponential_base must be greater than 1"):
            RetryConfig(exponential_base=1.0)


class TestSecureRandomIntegration:
    """Test secure random integration."""

    def test_retry_strategy_uses_secure_random(self):
        """Test that retry strategy uses secure random for jitter."""
        mock_rng = MagicMock()
        mock_rng.exponential_backoff_jitter.return_value = 2.5

        retry_strategy = RetryStrategy(secure_rng=mock_rng)
        delay = retry_strategy._calculate_delay(1)

        assert delay == 2.5
        mock_rng.exponential_backoff_jitter.assert_called_once()

    def test_mock_client_uses_secure_random(self):
        """Test that mock client uses secure random for failure simulation."""
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.3

        client = MockZenMCPClient(failure_rate=0.5, secure_rng=mock_rng)

        # Access the random value (simulates failure check)
        should_fail = client.secure_rng.random() < client.failure_rate
        assert should_fail is True
        mock_rng.random.assert_called_once()
