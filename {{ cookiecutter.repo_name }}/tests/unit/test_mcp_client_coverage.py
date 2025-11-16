"""Comprehensive MCP Client Coverage Tests.

This test suite is specifically designed to achieve 80%+ coverage for the MCP client
by testing all missing methods, error paths, and edge cases identified in the coverage analysis.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from tenacity import RetryError

from src.mcp_integration.mcp_client import (
    MCPAuthenticationError,
    MCPClientFactory,
    MCPClientInterface,
    MCPConnectionError,
    MCPConnectionManager,
    MCPConnectionState,
    MCPError,
    MCPErrorType,
    MCPHealthStatus,
    MCPRateLimitError,
    MCPServiceUnavailableError,
    MCPTimeoutError,
    MCPValidationError,
    MockMCPClient,
    Response,
    WorkflowStep,
    ZenMCPClient,
)


class TestMCPClientFactory:
    """Test MCPClientFactory for comprehensive coverage."""

    def test_create_mock_client(self):
        """Test creating mock client."""
        client = MCPClientFactory.create_client("mock")
        assert isinstance(client, MockMCPClient)
        assert client.simulate_failures is False
        assert client.failure_rate == 0.1

    def test_create_mock_client_with_params(self):
        """Test creating mock client with custom parameters."""
        client = MCPClientFactory.create_client(
            "mock",
            simulate_failures=True,
            failure_rate=0.3,
            response_delay=0.2,
            max_agents=15,
        )
        assert isinstance(client, MockMCPClient)
        assert client.simulate_failures is True
        assert client.failure_rate == 0.3
        assert client.response_delay == 0.2
        assert client.max_agents == 15

    def test_create_zen_client(self):
        """Test creating Zen client."""
        client = MCPClientFactory.create_client(
            "zen",
            server_url="http://test:8080",
            api_key="test-key",
        )
        assert isinstance(client, ZenMCPClient)
        assert client.server_url == "http://test:8080"
        assert client.api_key == "test-key"

    def test_create_zen_client_missing_url(self):
        """Test creating Zen client without server URL."""
        with pytest.raises(ValueError, match="server_url required for Zen MCP client"):
            MCPClientFactory.create_client("zen")

    def test_create_unsupported_client_type(self):
        """Test creating unsupported client type."""
        with pytest.raises(ValueError, match="Unsupported client type: invalid"):
            MCPClientFactory.create_client("invalid")

    @patch("src.mcp_integration.mcp_client.get_settings")
    def test_create_from_settings_disabled(self, mock_get_settings):
        """Test creating client from settings when MCP is disabled."""
        mock_settings = Mock()
        mock_settings.mcp_enabled = False
        mock_get_settings.return_value = mock_settings

        client = MCPClientFactory.create_from_settings(mock_settings)
        assert isinstance(client, MockMCPClient)

    @patch("src.mcp_integration.mcp_client.get_settings")
    def test_create_from_settings_enabled(self, mock_get_settings):
        """Test creating client from settings when MCP is enabled."""
        mock_settings = Mock()
        mock_settings.mcp_enabled = True
        mock_settings.mcp_server_url = "http://zen:8080"
        mock_settings.mcp_api_key = Mock()
        mock_settings.mcp_api_key.get_secret_value.return_value = "secret-key"
        mock_settings.mcp_timeout = 60
        mock_settings.mcp_max_retries = 5
        mock_get_settings.return_value = mock_settings

        client = MCPClientFactory.create_from_settings(mock_settings)
        assert isinstance(client, ZenMCPClient)
        assert client.server_url == "http://zen:8080"
        assert client.api_key == "secret-key"
        assert client.timeout == 60
        assert client.max_retries == 5

    @patch("src.mcp_integration.mcp_client.get_settings")
    def test_create_from_settings_no_api_key(self, mock_get_settings):
        """Test creating client from settings without API key."""
        mock_settings = Mock()
        mock_settings.mcp_enabled = True
        mock_settings.mcp_server_url = "http://zen:8080"
        mock_settings.mcp_api_key = None
        mock_settings.mcp_timeout = 30
        mock_settings.mcp_max_retries = 3
        mock_get_settings.return_value = mock_settings

        client = MCPClientFactory.create_from_settings(mock_settings)
        assert isinstance(client, ZenMCPClient)
        assert client.api_key is None


class TestMockMCPClient:
    """Test MockMCPClient missing coverage."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client with failure simulation."""
        return MockMCPClient(simulate_failures=True, failure_rate=1.0)  # Always fail

    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_client):
        """Test connection failure simulation."""
        with pytest.raises(MCPConnectionError):
            await mock_client.connect()
        assert mock_client.connection_state == MCPConnectionState.FAILED

    @pytest.mark.asyncio
    async def test_validate_query_service_unavailable(self, mock_client):
        """Test query validation service unavailable."""
        with pytest.raises(MCPServiceUnavailableError):
            await mock_client.validate_query("test query")

    @pytest.mark.asyncio
    async def test_orchestrate_agents_service_unavailable(self, mock_client):
        """Test agent orchestration service unavailable."""
        steps = [WorkflowStep(step_id="test1", agent_id="agent1", input_data={"task": "test"})]
        with pytest.raises(MCPServiceUnavailableError):
            await mock_client.orchestrate_agents(steps)

    @pytest.mark.asyncio
    async def test_orchestrate_agents_too_many_agents(self):
        """Test orchestration with too many agents."""
        client = MockMCPClient(max_agents=2)
        steps = [
            WorkflowStep(step_id=f"test{i}", agent_id=f"agent{i}", input_data={})
            for i in range(5)  # More than max_agents
        ]
        with pytest.raises(MCPError, match="Too many agents requested"):
            await client.orchestrate_agents(steps)

    @pytest.mark.asyncio
    async def test_get_capabilities_failure(self, mock_client):
        """Test get capabilities failure."""
        with pytest.raises(MCPConnectionError):
            await mock_client.get_capabilities()

    @pytest.mark.asyncio
    async def test_health_check_degraded_state(self):
        """Test health check with degraded state."""
        client = MockMCPClient(simulate_failures=True)  # Enable failures
        client.connection_state = MCPConnectionState.CONNECTED  # Set connected state
        # Force some errors to get degraded state - account for increment in health_check
        client.error_count = 1  # Will become 2 after increment, which is < DEGRADED_ERROR_THRESHOLD (3)

        # Mock the _should_fail method to return True to trigger degraded path
        client._should_fail = lambda: True

        health = await client.health_check()
        assert health.connection_state == MCPConnectionState.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_failed_state(self):
        """Test health check with failed state."""
        client = MockMCPClient(simulate_failures=True)  # Enable failures
        client.connection_state = MCPConnectionState.CONNECTED  # Set connected state
        # Force many errors to get failed state
        client.error_count = 5  # Above threshold

        # Mock the _should_fail method to return True to trigger failure path
        client._should_fail = lambda: True

        health = await client.health_check()
        assert health.connection_state == MCPConnectionState.FAILED


class TestZenMCPClient:
    """Test ZenMCPClient missing coverage."""

    @pytest.fixture
    def zen_client(self):
        """Create Zen MCP client."""
        return ZenMCPClient(
            server_url="http://test:8080",
            api_key="test-key",
            timeout=10.0,
            max_retries=2,
        )

    @pytest.mark.asyncio
    async def test_connect_httpx_not_available(self, zen_client):
        """Test connection when httpx is not available."""
        with (
            patch("src.mcp_integration.mcp_client.httpx", None),
            pytest.raises(MCPConnectionError, match="Connection failed: httpx is not installed"),
        ):
            await zen_client.connect()

    @pytest.mark.asyncio
    async def test_connect_authentication_error(self, zen_client):
        """Test connection with authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": "invalid_api_key"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get.return_value = mock_response

            with pytest.raises(MCPAuthenticationError):
                await zen_client.connect()

            assert zen_client.connection_state == MCPConnectionState.FAILED

    @pytest.mark.asyncio
    async def test_connect_http_error(self, zen_client):
        """Test connection with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get.return_value = mock_response

            with pytest.raises(MCPConnectionError):
                await zen_client.connect()

    @pytest.mark.asyncio
    async def test_connect_timeout_error(self, zen_client):
        """Test connection timeout."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(MCPConnectionError):
                await zen_client.connect()

    @pytest.mark.asyncio
    async def test_connect_connection_error(self, zen_client):
        """Test connection error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = httpx.ConnectError("Cannot connect")

            with pytest.raises(MCPConnectionError):
                await zen_client.connect()

    @pytest.mark.asyncio
    async def test_connect_unexpected_error(self, zen_client):
        """Test unexpected connection error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.side_effect = ValueError("Unexpected error")

            with pytest.raises(MCPConnectionError):
                await zen_client.connect()

    @pytest.mark.asyncio
    async def test_connect_non_healthy_response(self, zen_client):
        """Test connection with non-healthy server response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"status": "degraded"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get.return_value = mock_response

            # Should still connect but log warning
            result = await zen_client.connect()
            assert result is True
            assert zen_client.connection_state == MCPConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_error(self, zen_client):
        """Test disconnection error."""
        # Set up a session that will fail to close
        zen_client.session = AsyncMock()
        zen_client.session.aclose.side_effect = Exception("Close error")

        result = await zen_client.disconnect()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_httpx_not_available(self, zen_client):
        """Test health check when httpx not available."""
        with patch("src.mcp_integration.mcp_client.httpx", None), pytest.raises(RetryError):
            await zen_client.health_check()

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, zen_client):
        """Test health check when not connected."""
        # Ensure client is in disconnected state (default)
        zen_client.connection_state = MCPConnectionState.DISCONNECTED
        zen_client.session = None
        # The health_check method has a retry decorator, so it will raise RetryError after retries
        with pytest.raises(RetryError):
            await zen_client.health_check()

    @pytest.mark.asyncio
    async def test_health_check_http_error(self, zen_client):
        """Test health check with HTTP error."""
        zen_client.session = AsyncMock()
        zen_client.connection_state = MCPConnectionState.CONNECTED

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=Mock(),
            response=mock_response,
        )
        zen_client.session.get.return_value = mock_response

        # The health_check method has a retry decorator, so it will raise RetryError after retries
        with pytest.raises(RetryError):
            await zen_client.health_check()

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, zen_client):
        """Test health check timeout."""
        zen_client.session = AsyncMock()
        zen_client.connection_state = MCPConnectionState.CONNECTED
        zen_client.session.get.side_effect = httpx.TimeoutException("Timeout")

        # The health_check method has a retry decorator, so it will raise RetryError after retries
        with pytest.raises(RetryError):
            await zen_client.health_check()

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, zen_client):
        """Test health check connection error."""
        zen_client.session = AsyncMock()
        zen_client.connection_state = MCPConnectionState.CONNECTED
        zen_client.session.get.side_effect = httpx.ConnectError("Cannot connect")

        # The health_check method has a retry decorator, so it will raise RetryError after retries
        with pytest.raises(RetryError):
            await zen_client.health_check()

    @pytest.mark.asyncio
    async def test_validate_query_empty(self, zen_client):
        """Test validating empty query."""
        result = await zen_client.validate_query("")
        assert result["is_valid"] is False
        assert result["sanitized_query"] == ""
        assert "Empty query" in result["potential_issues"]

    @pytest.mark.asyncio
    async def test_validate_query_httpx_not_available(self, zen_client):
        """Test query validation when httpx not available."""
        with (
            patch("src.mcp_integration.mcp_client.httpx", None),
            pytest.raises(
                MCPServiceUnavailableError,
                match="Validation service unavailable: httpx is not installed",
            ),
        ):
            await zen_client.validate_query("test")

    @pytest.mark.asyncio
    async def test_validate_query_not_connected(self, zen_client):
        """Test query validation when not connected."""
        # Ensure client is in disconnected state
        zen_client.connection_state = MCPConnectionState.DISCONNECTED
        zen_client.session = None
        with pytest.raises(MCPConnectionError, match="Not connected to server"):
            await zen_client.validate_query("test")

    @pytest.mark.asyncio
    async def test_validate_query_bad_request(self, zen_client):
        """Test query validation with bad request error."""
        zen_client.session = AsyncMock()

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": "Invalid query", "details": {"field": "query"}}
        mock_response.text = "Bad Request"

        error = httpx.HTTPStatusError("Bad request", request=Mock(), response=mock_response)
        zen_client.session.post.side_effect = error

        with pytest.raises(MCPValidationError):
            await zen_client.validate_query("test")

    @pytest.mark.asyncio
    async def test_validate_query_service_unavailable(self, zen_client):
        """Test query validation service unavailable."""
        zen_client.session = AsyncMock()

        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.headers = {"Retry-After": "60"}

        error = httpx.HTTPStatusError("Service unavailable", request=Mock(), response=mock_response)
        zen_client.session.post.side_effect = error

        with pytest.raises(MCPServiceUnavailableError):
            await zen_client.validate_query("test")

    @pytest.mark.asyncio
    async def test_validate_query_timeout(self, zen_client):
        """Test query validation timeout."""
        zen_client.session = AsyncMock()
        zen_client.session.post.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(MCPTimeoutError):
            await zen_client.validate_query("test")

    @pytest.mark.asyncio
    async def test_validate_query_connection_error(self, zen_client):
        """Test query validation connection error."""
        zen_client.session = AsyncMock()
        zen_client.session.post.side_effect = httpx.ConnectError("Cannot connect")

        with pytest.raises(MCPConnectionError):
            await zen_client.validate_query("test")

    @pytest.mark.asyncio
    async def test_validate_query_non_dict_response(self, zen_client):
        """Test query validation with non-dict response."""
        zen_client.session = AsyncMock()

        mock_response = Mock()
        mock_response.json.return_value = "not a dict"
        zen_client.session.post.return_value = mock_response

        result = await zen_client.validate_query("test")
        assert result["is_valid"] is False
        assert "Invalid response format" in result["potential_issues"]

    @pytest.mark.asyncio
    async def test_orchestrate_agents_httpx_not_available(self, zen_client):
        """Test orchestration when httpx not available."""
        with patch("src.mcp_integration.mcp_client.httpx", None), pytest.raises(RetryError):
            await zen_client.orchestrate_agents([])

    @pytest.mark.asyncio
    async def test_orchestrate_agents_not_connected(self, zen_client):
        """Test orchestration when not connected."""
        # Ensure client is in disconnected state
        zen_client.connection_state = MCPConnectionState.DISCONNECTED
        zen_client.session = None
        with pytest.raises(MCPConnectionError, match="Not connected to server"):
            await zen_client.orchestrate_agents([])

    @pytest.mark.asyncio
    async def test_orchestrate_agents_authentication_error(self, zen_client):
        """Test orchestration with authentication error."""
        zen_client.session = AsyncMock()

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": "invalid_token"}
        mock_response.text = "Unauthorized"

        error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=mock_response)
        zen_client.session.post.side_effect = error

        steps = [WorkflowStep(step_id="test", agent_id="agent", input_data={})]
        with pytest.raises(MCPAuthenticationError):
            await zen_client.orchestrate_agents(steps)

    @pytest.mark.asyncio
    async def test_orchestrate_agents_bad_request(self, zen_client):
        """Test orchestration with bad request."""
        zen_client.session = AsyncMock()

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"message": "Invalid request"}

        error = httpx.HTTPStatusError("Bad request", request=Mock(), response=mock_response)
        zen_client.session.post.side_effect = error

        steps = [WorkflowStep(step_id="test", agent_id="agent", input_data={})]
        with pytest.raises(MCPError):
            await zen_client.orchestrate_agents(steps)

    @pytest.mark.asyncio
    async def test_orchestrate_agents_rate_limit(self, zen_client):
        """Test orchestration with rate limit."""
        zen_client.session = AsyncMock()

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}

        error = httpx.HTTPStatusError("Rate limited", request=Mock(), response=mock_response)
        zen_client.session.post.side_effect = error

        steps = [WorkflowStep(step_id="test", agent_id="agent", input_data={})]
        with pytest.raises(MCPRateLimitError):
            await zen_client.orchestrate_agents(steps)

    @pytest.mark.asyncio
    async def test_orchestrate_agents_timeout(self, zen_client):
        """Test orchestration timeout."""
        zen_client.session = AsyncMock()
        zen_client.session.post.side_effect = httpx.TimeoutException("Timeout")

        steps = [WorkflowStep(step_id="test", agent_id="agent", input_data={})]
        with pytest.raises(RetryError):
            await zen_client.orchestrate_agents(steps)

    @pytest.mark.asyncio
    async def test_orchestrate_agents_connection_error(self, zen_client):
        """Test orchestration connection error."""
        zen_client.session = AsyncMock()
        zen_client.session.post.side_effect = httpx.ConnectError("Cannot connect")

        steps = [WorkflowStep(step_id="test", agent_id="agent", input_data={})]
        with pytest.raises(MCPConnectionError):
            await zen_client.orchestrate_agents(steps)

    @pytest.mark.asyncio
    async def test_get_capabilities_httpx_not_available(self, zen_client):
        """Test get capabilities when httpx not available."""
        with (
            patch("src.mcp_integration.mcp_client.httpx", None),
            pytest.raises(MCPConnectionError, match="Capabilities query failed: httpx is not installed"),
        ):
            await zen_client.get_capabilities()

    @pytest.mark.asyncio
    async def test_get_capabilities_not_connected(self, zen_client):
        """Test get capabilities when not connected."""
        # Ensure client is in disconnected state
        zen_client.connection_state = MCPConnectionState.DISCONNECTED
        zen_client.session = None
        with pytest.raises(MCPConnectionError, match="Not connected to server"):
            await zen_client.get_capabilities()

    @pytest.mark.asyncio
    async def test_get_capabilities_http_error(self, zen_client):
        """Test get capabilities with HTTP error (should return defaults)."""
        zen_client.session = AsyncMock()
        zen_client.connection_state = MCPConnectionState.CONNECTED

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=Mock(),
            response=mock_response,
        )
        zen_client.session.get.return_value = mock_response

        # Should return default capabilities instead of raising
        capabilities = await zen_client.get_capabilities()
        assert capabilities == ["zen_orchestration", "multi_agent", "consensus", "validation"]

    @pytest.mark.asyncio
    async def test_get_capabilities_timeout(self, zen_client):
        """Test get capabilities timeout."""
        zen_client.session = AsyncMock()
        zen_client.connection_state = MCPConnectionState.CONNECTED
        zen_client.session.get.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(MCPTimeoutError):
            await zen_client.get_capabilities()

    @pytest.mark.asyncio
    async def test_get_capabilities_connection_error(self, zen_client):
        """Test get capabilities connection error."""
        zen_client.session = AsyncMock()
        zen_client.connection_state = MCPConnectionState.CONNECTED
        zen_client.session.get.side_effect = httpx.ConnectError("Cannot connect")

        with pytest.raises(MCPConnectionError):
            await zen_client.get_capabilities()

    @pytest.mark.asyncio
    async def test_get_capabilities_non_list_response(self, zen_client):
        """Test get capabilities with non-list response."""
        zen_client.session = AsyncMock()
        zen_client.connection_state = MCPConnectionState.CONNECTED

        mock_response = Mock()
        mock_response.json.return_value = {"capabilities": "not a list"}
        zen_client.session.get.return_value = mock_response

        capabilities = await zen_client.get_capabilities()
        assert capabilities == ["zen_orchestration", "multi_agent", "consensus", "validation"]


class TestMCPConnectionManager:
    """Test MCPConnectionManager missing coverage."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MCP client."""
        return Mock(spec=MCPClientInterface)

    @pytest.fixture
    def connection_manager(self, mock_client):
        """Create connection manager."""
        return MCPConnectionManager(
            client=mock_client,
            health_check_interval=0.1,  # Fast for testing
            max_consecutive_failures=2,
            circuit_breaker_timeout=0.5,
        )

    @pytest.mark.asyncio
    async def test_start_success(self, connection_manager, mock_client):
        """Test successful start."""
        mock_client.connect = AsyncMock(return_value=True)

        result = await connection_manager.start()
        assert result is True
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_failure(self, connection_manager, mock_client):
        """Test start failure."""
        mock_client.connect = AsyncMock(side_effect=Exception("Connection failed"))

        result = await connection_manager.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_stop(self, connection_manager, mock_client):
        """Test stop."""
        mock_client.disconnect = AsyncMock()

        # Start first to create health check task
        mock_client.connect = AsyncMock(return_value=True)
        await connection_manager.start()

        # Now stop
        await connection_manager.stop()
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, connection_manager, mock_client):
        """Test successful operation execution."""
        mock_client.health_check = AsyncMock(return_value={"status": "ok"})

        result = await connection_manager.execute_with_fallback("health_check")
        assert result == {"status": "ok"}
        assert connection_manager.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_execute_with_fallback_failure(self, connection_manager, mock_client):
        """Test operation failure with fallback."""
        mock_client.health_check = AsyncMock(side_effect=Exception("Operation failed"))

        result = await connection_manager.execute_with_fallback("health_check")
        assert result["fallback"] is True
        assert result["operation"] == "health_check"
        assert connection_manager.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_execute_with_fallback_circuit_breaker_open(self, connection_manager, mock_client):
        """Test operation with circuit breaker open."""
        # Force circuit breaker open
        connection_manager.is_circuit_breaker_open = True
        connection_manager.circuit_breaker_open_time = time.time()

        result = await connection_manager.execute_with_fallback("health_check")
        assert result["fallback"] is True
        assert result["error"] == "Circuit breaker open"

    @pytest.mark.asyncio
    async def test_execute_with_fallback_circuit_breaker_reset(self, connection_manager, mock_client):
        """Test circuit breaker reset after timeout."""
        # Set circuit breaker open in the past
        connection_manager.is_circuit_breaker_open = True
        connection_manager.circuit_breaker_open_time = time.time() - 1.0  # 1 second ago

        mock_client.health_check = AsyncMock(return_value={"status": "ok"})

        result = await connection_manager.execute_with_fallback("health_check")
        assert result == {"status": "ok"}
        assert connection_manager.is_circuit_breaker_open is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, connection_manager, mock_client):
        """Test circuit breaker opens after max failures."""
        mock_client.health_check = AsyncMock(side_effect=Exception("Always fails"))

        # Execute multiple times to trigger circuit breaker
        for _ in range(3):
            await connection_manager.execute_with_fallback("health_check")

        assert connection_manager.is_circuit_breaker_open is True
        assert connection_manager.consecutive_failures >= connection_manager.max_consecutive_failures

    @pytest.mark.asyncio
    async def test_fallback_response_orchestrate_agents(self, connection_manager):
        """Test fallback response for orchestrate_agents."""
        result = connection_manager._get_fallback_response("orchestrate_agents")
        assert result["responses"] == []
        assert result["total_agents"] == 0
        assert result["fallback"] is True

    @pytest.mark.asyncio
    async def test_fallback_response_validate_query(self, connection_manager):
        """Test fallback response for validate_query."""
        result = connection_manager._get_fallback_response("validate_query")
        assert result["is_valid"] is True
        assert result["sanitized_query"] == ""
        assert "MCP validation service unavailable" in result["potential_issues"]

    @pytest.mark.asyncio
    async def test_fallback_response_get_capabilities(self, connection_manager):
        """Test fallback response for get_capabilities."""
        result = connection_manager._get_fallback_response("get_capabilities")
        assert result["capabilities"] == ["basic_processing"]
        assert result["fallback"] is True

    @pytest.mark.asyncio
    async def test_health_monitor_failed_state(self, connection_manager, mock_client):
        """Test health monitor handling failed state."""
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.disconnect = AsyncMock()

        # Mock health check to return failed state once
        failed_health = MCPHealthStatus(
            connection_state=MCPConnectionState.FAILED,
            error_count=5,
        )
        mock_client.health_check = AsyncMock(return_value=failed_health)

        # Start the manager (this starts the health monitor)
        await connection_manager.start()

        # Wait a bit for health monitor to run
        await asyncio.sleep(0.2)

        # Stop to clean up
        await connection_manager.stop()

        # Should have attempted reconnection
        assert mock_client.disconnect.call_count >= 1
        assert mock_client.connect.call_count >= 2  # Initial + reconnection


class TestMCPErrors:
    """Test MCP error classes for comprehensive coverage."""

    def test_mcp_error_base(self):
        """Test base MCP error."""
        error = MCPError(
            "Test error",
            error_type=MCPErrorType.CONNECTION_ERROR,
            details={"key": "value"},
            retry_after=60,
        )
        assert str(error) == "Test error"
        assert error.error_type == MCPErrorType.CONNECTION_ERROR
        assert error.details == {"key": "value"}
        assert error.retry_after == 60

    def test_mcp_connection_error(self):
        """Test MCP connection error."""
        error = MCPConnectionError("Connection failed", {"host": "test"})
        assert str(error) == "Connection failed"
        assert error.error_type == MCPErrorType.CONNECTION_ERROR
        assert error.details == {"host": "test"}

    def test_mcp_timeout_error(self):
        """Test MCP timeout error."""
        error = MCPTimeoutError("Timeout occurred", 30.0, {"operation": "connect"})
        assert str(error) == "Timeout occurred"
        assert error.error_type == MCPErrorType.TIMEOUT_ERROR
        assert error.timeout_seconds == 30.0
        assert error.details == {"operation": "connect"}

    def test_mcp_service_unavailable_error(self):
        """Test MCP service unavailable error."""
        error = MCPServiceUnavailableError("Service down", 120, {"service": "zen"})
        assert str(error) == "Service down"
        assert error.error_type == MCPErrorType.SERVICE_UNAVAILABLE
        assert error.retry_after == 120
        assert error.details == {"service": "zen"}

    def test_mcp_rate_limit_error(self):
        """Test MCP rate limit error."""
        error = MCPRateLimitError("Rate limited", 300, {"limit": "100/hour"})
        assert str(error) == "Rate limited"
        assert error.error_type == MCPErrorType.RATE_LIMIT_ERROR
        assert error.retry_after == 300
        assert error.details == {"limit": "100/hour"}

    def test_mcp_authentication_error(self):
        """Test MCP authentication error."""
        error = MCPAuthenticationError(
            "Auth failed",
            error_code="INVALID_TOKEN",
            details={"token": "expired"},
        )
        assert str(error) == "Auth failed"
        assert error.error_type == MCPErrorType.AUTHENTICATION_ERROR
        assert error.error_code == "INVALID_TOKEN"
        assert error.details == {"token": "expired"}

    def test_mcp_validation_error(self):
        """Test MCP validation error."""
        error = MCPValidationError(
            "Validation failed",
            validation_errors={"field": "required"},
            details={"input": "invalid"},
        )
        assert str(error) == "Validation failed"
        assert error.error_type == MCPErrorType.INVALID_REQUEST
        assert error.validation_errors == {"field": "required"}
        assert error.details == {"input": "invalid"}


class TestWorkflowModels:
    """Test workflow models for comprehensive coverage."""

    def test_workflow_step(self):
        """Test WorkflowStep model."""
        step = WorkflowStep(
            step_id="test_step",
            agent_id="test_agent",
            input_data={"key": "value"},
            dependencies=["dep1", "dep2"],
            timeout_seconds=45,
        )
        assert step.step_id == "test_step"
        assert step.agent_id == "test_agent"
        assert step.input_data == {"key": "value"}
        assert step.dependencies == ["dep1", "dep2"]
        assert step.timeout_seconds == 45

    def test_response_model(self):
        """Test Response model."""
        response = Response(
            agent_id="test_agent",
            content="Test response",
            metadata={"source": "test"},
            confidence=0.95,
            processing_time=1.5,
            success=True,
            error_message=None,
        )
        assert response.agent_id == "test_agent"
        assert response.content == "Test response"
        assert response.metadata == {"source": "test"}
        assert response.confidence == 0.95
        assert response.processing_time == 1.5
        assert response.success is True
        assert response.error_message is None

    def test_response_model_with_error(self):
        """Test Response model with error."""
        response = Response(
            agent_id="test_agent",
            content="",
            confidence=0.0,
            processing_time=0.1,
            success=False,
            error_message="Processing failed",
        )
        assert response.success is False
        assert response.error_message == "Processing failed"

    def test_mcp_health_status(self):
        """Test MCPHealthStatus model."""
        health = MCPHealthStatus(
            connection_state=MCPConnectionState.CONNECTED,
            last_successful_request=time.time(),
            error_count=0,
            response_time_ms=150.0,
            capabilities=["orchestration", "validation"],
            server_version="ZenMCP-2.0",
            metadata={"region": "us-west"},
        )
        assert health.connection_state == MCPConnectionState.CONNECTED
        assert health.error_count == 0
        assert health.response_time_ms == 150.0
        assert health.capabilities == ["orchestration", "validation"]
        assert health.server_version == "ZenMCP-2.0"
        assert health.metadata == {"region": "us-west"}
