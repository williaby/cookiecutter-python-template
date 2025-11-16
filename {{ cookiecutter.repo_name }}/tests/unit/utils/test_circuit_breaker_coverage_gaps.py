"""Comprehensive tests for circuit_breaker.py coverage gaps - targeting 0% coverage functions."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.utils.circuit_breaker import create_openrouter_circuit_breaker


class TestCreateOpenRouterCircuitBreakerCoverageGaps:
    """Test create_openrouter_circuit_breaker function with 0% coverage."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock ApplicationSettings for testing."""
        settings = Mock()
        settings.openrouter_api_key = "test-api-key"
        settings.openrouter_base_url = "https://openrouter.ai/api/v1"
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
        settings.health_check_enabled = True
        return settings

    @pytest.fixture
    def mock_settings_health_disabled(self):
        """Create mock ApplicationSettings with health check disabled."""
        settings = Mock()
        settings.openrouter_api_key = "test-api-key"
        settings.openrouter_base_url = "https://openrouter.ai/api/v1"
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
        settings.health_check_enabled = False
        return settings

    async def test_create_openrouter_circuit_breaker_health_enabled(self, mock_settings):
        """Test create_openrouter_circuit_breaker with health check enabled."""
        circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

        # Verify circuit breaker properties
        assert circuit_breaker.name == "openrouter"
        assert circuit_breaker.config.failure_threshold == mock_settings.circuit_breaker_failure_threshold
        assert circuit_breaker.config.recovery_timeout == mock_settings.circuit_breaker_recovery_timeout
        assert circuit_breaker.config.max_retries == mock_settings.circuit_breaker_max_retries
        assert circuit_breaker.health_check_func is not None

    async def test_create_openrouter_circuit_breaker_health_disabled(self, mock_settings_health_disabled):
        """Test create_openrouter_circuit_breaker with health check disabled."""
        circuit_breaker = create_openrouter_circuit_breaker(mock_settings_health_disabled)

        # Verify circuit breaker properties
        assert circuit_breaker.name == "openrouter"
        assert circuit_breaker.health_check_func is None

    async def test_openrouter_health_check_success(self, mock_settings):
        """Test the openrouter_health_check nested function for successful health check."""
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
            patch("src.mcp_integration.mcp_client.MCPConnectionState") as mock_connection_state,
        ):

            # Setup mocks
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock health status
            mock_health_status = Mock()
            mock_health_status.connection_state = mock_connection_state.CONNECTED
            mock_client.health_check = AsyncMock(return_value=mock_health_status)

            # Create circuit breaker with health check
            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

            # Execute health check function
            result = await circuit_breaker.health_check_func()

            # Verify health check result
            assert result is True

            # Verify client was created with correct parameters
            mock_client_class.assert_called_once_with(
                api_key=mock_settings.openrouter_api_key,
                base_url=mock_settings.openrouter_base_url,
                timeout=5.0,
            )
            mock_client.health_check.assert_called_once()

    async def test_openrouter_health_check_failure(self, mock_settings):
        """Test the openrouter_health_check nested function for failed health check."""
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
            patch("src.mcp_integration.mcp_client.MCPConnectionState") as mock_connection_state,
        ):

            # Setup mocks
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock health status indicating failure
            mock_health_status = Mock()
            mock_health_status.connection_state = mock_connection_state.DISCONNECTED
            mock_client.health_check = AsyncMock(return_value=mock_health_status)

            # Create circuit breaker with health check
            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

            # Execute health check function
            result = await circuit_breaker.health_check_func()

            # Verify health check result
            assert result is False

    async def test_openrouter_health_check_exception(self, mock_settings):
        """Test the openrouter_health_check nested function handling exceptions."""
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
            patch("src.utils.circuit_breaker.logger") as mock_logger,
        ):

            # Setup mock to raise exception
            mock_client_class.side_effect = Exception("Connection failed")

            # Create circuit breaker with health check
            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

            # Execute health check function
            result = await circuit_breaker.health_check_func()

            # Verify health check result and logging
            assert result is False
            mock_logger.warning.assert_called_once()

    async def test_openrouter_health_check_import_error(self, mock_settings):
        """Test the openrouter_health_check nested function handling import errors."""
        with (
            patch(
                "src.mcp_integration.openrouter_client.OpenRouterClient",
                side_effect=ImportError("Module not found"),
            ),
            patch("src.utils.circuit_breaker.logger") as mock_logger,
        ):

            # Create circuit breaker with health check
            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

            # Execute health check function
            result = await circuit_breaker.health_check_func()

            # Verify health check result and logging
            assert result is False
            mock_logger.warning.assert_called_once()

    def test_openrouter_circuit_breaker_config_mapping(self, mock_settings):
        """Test that settings are correctly mapped to circuit breaker config."""
        # Modify settings to test different values
        mock_settings.circuit_breaker_failure_threshold = 10
        mock_settings.circuit_breaker_recovery_timeout = 120
        mock_settings.circuit_breaker_max_retries = 5

        circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

        # Verify configuration mapping
        assert circuit_breaker.config.failure_threshold == 10
        assert circuit_breaker.config.recovery_timeout == 120
        assert circuit_breaker.config.max_retries == 5

    async def test_openrouter_health_check_client_creation_parameters(self, mock_settings):
        """Test that OpenRouterClient is created with correct parameters."""
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
            patch("src.mcp_integration.mcp_client.MCPConnectionState") as mock_connection_state,
        ):

            # Setup mocks
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_health_status = Mock()
            mock_health_status.connection_state = mock_connection_state.CONNECTED
            mock_client.health_check = AsyncMock(return_value=mock_health_status)

            # Modify settings to test parameter passing
            mock_settings.openrouter_api_key = "custom-api-key"
            mock_settings.openrouter_base_url = "https://custom.openrouter.com/api/v1"

            # Create circuit breaker and execute health check
            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)
            await circuit_breaker.health_check_func()

            # Verify client creation parameters
            mock_client_class.assert_called_once_with(
                api_key="custom-api-key",
                base_url="https://custom.openrouter.com/api/v1",
                timeout=5.0,
            )

    async def test_openrouter_health_check_multiple_calls(self, mock_settings):
        """Test multiple calls to openrouter_health_check function."""
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
            patch("src.mcp_integration.mcp_client.MCPConnectionState") as mock_connection_state,
        ):

            # Setup mocks
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_health_status = Mock()
            mock_health_status.connection_state = mock_connection_state.CONNECTED
            mock_client.health_check = AsyncMock(return_value=mock_health_status)

            # Create circuit breaker
            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

            # Execute health check multiple times
            result1 = await circuit_breaker.health_check_func()
            result2 = await circuit_breaker.health_check_func()
            result3 = await circuit_breaker.health_check_func()

            # Verify all calls succeed
            assert result1 is True
            assert result2 is True
            assert result3 is True

            # Verify client was created for each call
            assert mock_client_class.call_count == 3

    async def test_openrouter_health_check_timeout_handling(self, mock_settings):
        """Test openrouter_health_check with timeout scenarios."""
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
            patch("src.utils.circuit_breaker.logger") as mock_logger,
        ):

            # Setup mock to raise timeout exception
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.health_check = AsyncMock(side_effect=TimeoutError("Health check timed out"))

            # Create circuit breaker
            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

            # Execute health check function
            result = await circuit_breaker.health_check_func()

            # Verify timeout is handled gracefully
            assert result is False
            mock_logger.warning.assert_called_once()

    def test_create_openrouter_circuit_breaker_edge_cases(self):
        """Test create_openrouter_circuit_breaker with edge case settings."""
        # Test with minimal settings
        minimal_settings = Mock()
        minimal_settings.openrouter_api_key = ""
        minimal_settings.openrouter_base_url = ""
        minimal_settings.circuit_breaker_failure_threshold = 1
        minimal_settings.circuit_breaker_success_threshold = 1
        minimal_settings.circuit_breaker_recovery_timeout = 1
        minimal_settings.circuit_breaker_max_retries = 0
        minimal_settings.circuit_breaker_base_delay = 1.0
        minimal_settings.circuit_breaker_max_delay = 60.0
        minimal_settings.circuit_breaker_backoff_multiplier = 2.0
        minimal_settings.circuit_breaker_jitter_enabled = True
        minimal_settings.circuit_breaker_health_check_interval = 30
        minimal_settings.circuit_breaker_health_check_timeout = 5.0
        minimal_settings.performance_monitoring_enabled = True
        minimal_settings.health_check_enabled = False

        circuit_breaker = create_openrouter_circuit_breaker(minimal_settings)

        # Verify circuit breaker is created with minimal settings
        assert circuit_breaker.name == "openrouter"
        assert circuit_breaker.config.failure_threshold == 1
        assert circuit_breaker.config.recovery_timeout == 1
        assert circuit_breaker.config.max_retries == 0
        assert circuit_breaker.health_check_func is None


class TestOpenRouterHealthCheckIntegration:
    """Integration tests for openrouter_health_check function."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock ApplicationSettings for integration testing."""
        settings = Mock()
        settings.openrouter_api_key = "integration-test-key"
        settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        settings.circuit_breaker_failure_threshold = 3
        settings.circuit_breaker_success_threshold = 2
        settings.circuit_breaker_recovery_timeout = 30
        settings.circuit_breaker_max_retries = 2
        settings.circuit_breaker_base_delay = 1.0
        settings.circuit_breaker_max_delay = 60.0
        settings.circuit_breaker_backoff_multiplier = 2.0
        settings.circuit_breaker_jitter_enabled = True
        settings.circuit_breaker_health_check_interval = 30
        settings.circuit_breaker_health_check_timeout = 5.0
        settings.performance_monitoring_enabled = True
        settings.health_check_enabled = True
        return settings

    async def test_health_check_function_exists_and_callable(self, mock_settings):
        """Test that health check function is properly assigned and callable."""
        circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

        # Verify health check function exists
        assert circuit_breaker.health_check_func is not None
        assert callable(circuit_breaker.health_check_func)

    async def test_health_check_function_signature(self, mock_settings):
        """Test that health check function has correct signature."""
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
            patch("src.mcp_integration.mcp_client.MCPConnectionState") as mock_connection_state,
        ):

            # Setup mocks
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_health_status = Mock()
            mock_health_status.connection_state = mock_connection_state.CONNECTED
            mock_client.health_check = AsyncMock(return_value=mock_health_status)

            circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

            # Test function can be called without arguments
            result = await circuit_breaker.health_check_func()

            # Verify result type
            assert isinstance(result, bool)


@pytest.mark.parametrize("health_enabled", [True, False])
def test_create_openrouter_circuit_breaker_health_check_conditional(health_enabled):
    """Test create_openrouter_circuit_breaker with health check enabled/disabled."""
    mock_settings = Mock()
    mock_settings.openrouter_api_key = "test-key"
    mock_settings.openrouter_base_url = "https://test.com"
    mock_settings.circuit_breaker_failure_threshold = 5
    mock_settings.circuit_breaker_success_threshold = 3
    mock_settings.circuit_breaker_recovery_timeout = 60
    mock_settings.circuit_breaker_max_retries = 3
    mock_settings.circuit_breaker_base_delay = 1.0
    mock_settings.circuit_breaker_max_delay = 60.0
    mock_settings.circuit_breaker_backoff_multiplier = 2.0
    mock_settings.circuit_breaker_jitter_enabled = True
    mock_settings.circuit_breaker_health_check_interval = 30
    mock_settings.circuit_breaker_health_check_timeout = 5.0
    mock_settings.performance_monitoring_enabled = True
    mock_settings.health_check_enabled = health_enabled

    circuit_breaker = create_openrouter_circuit_breaker(mock_settings)

    if health_enabled:
        assert circuit_breaker.health_check_func is not None
    else:
        assert circuit_breaker.health_check_func is None


@pytest.mark.parametrize(
    ("connection_state", "expected_result"),
    [
        ("CONNECTED", True),
        ("DISCONNECTED", False),
        ("CONNECTING", False),
        ("ERROR", False),
    ],
)
async def test_openrouter_health_check_connection_states(connection_state, expected_result):
    """Test openrouter_health_check with different connection states."""
    mock_settings = Mock()
    mock_settings.openrouter_api_key = "test-key"
    mock_settings.openrouter_base_url = "https://test.com"
    mock_settings.circuit_breaker_failure_threshold = 5
    mock_settings.circuit_breaker_success_threshold = 3
    mock_settings.circuit_breaker_recovery_timeout = 60
    mock_settings.circuit_breaker_max_retries = 3
    mock_settings.circuit_breaker_base_delay = 1.0
    mock_settings.circuit_breaker_max_delay = 60.0
    mock_settings.circuit_breaker_backoff_multiplier = 2.0
    mock_settings.circuit_breaker_jitter_enabled = True
    mock_settings.circuit_breaker_health_check_interval = 30
    mock_settings.circuit_breaker_health_check_timeout = 5.0
    mock_settings.performance_monitoring_enabled = True
    mock_settings.health_check_enabled = True

    with (
        patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_client_class,
        patch("src.mcp_integration.mcp_client.MCPConnectionState") as mock_connection_state_enum,
    ):

        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock connection state
        mock_connection_state_enum.CONNECTED = "CONNECTED"

        # Mock health status
        mock_health_status = Mock()
        mock_health_status.connection_state = connection_state
        mock_client.health_check = AsyncMock(return_value=mock_health_status)

        # Create circuit breaker and test health check
        circuit_breaker = create_openrouter_circuit_breaker(mock_settings)
        result = await circuit_breaker.health_check_func()

        # Verify result matches expected
        assert result == expected_result
