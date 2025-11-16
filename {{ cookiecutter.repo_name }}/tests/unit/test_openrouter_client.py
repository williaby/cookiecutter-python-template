"""Unit tests for OpenRouterClient.

Tests cover OpenRouter API integration, authentication, error handling, model selection,
and MCP interface compliance. Ensures 80% coverage requirement with comprehensive
test scenarios including connection management, response parsing, and retry logic.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.config.settings import ApplicationSettings
from src.mcp_integration.mcp_client import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPConnectionState,
    MCPError,
    MCPHealthStatus,
    MCPRateLimitError,
    MCPTimeoutError,
    MCPValidationError,
    Response,
    WorkflowStep,
)
from src.mcp_integration.model_registry import ModelCapabilities, ModelRegistry
from src.mcp_integration.openrouter_client import OpenRouterClient


def create_mock_settings(api_key: str = "test-key", circuit_breaker_enabled: bool = True, **kwargs) -> Mock:
    """Create a mock ApplicationSettings with all required circuit breaker settings."""
    mock_settings = Mock(spec=ApplicationSettings)

    # OpenRouter settings
    if api_key:
        mock_settings.openrouter_api_key = Mock()
        mock_settings.openrouter_api_key.get_secret_value.return_value = api_key
    else:
        mock_settings.openrouter_api_key = None

    mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"

    # Circuit breaker settings
    mock_settings.circuit_breaker_enabled = circuit_breaker_enabled
    mock_settings.circuit_breaker_failure_threshold = 5
    mock_settings.circuit_breaker_success_threshold = 3
    mock_settings.circuit_breaker_recovery_timeout = 60
    mock_settings.circuit_breaker_max_retries = 3
    mock_settings.circuit_breaker_base_delay = 1.0
    mock_settings.circuit_breaker_max_delay = 30.0
    mock_settings.circuit_breaker_backoff_multiplier = 2.0
    mock_settings.circuit_breaker_jitter_enabled = True
    mock_settings.circuit_breaker_health_check_interval = 10
    mock_settings.circuit_breaker_health_check_timeout = 5
    mock_settings.performance_monitoring_enabled = True
    mock_settings.health_check_enabled = True

    # Apply any additional overrides
    for key, value in kwargs.items():
        setattr(mock_settings, key, value)

    return mock_settings


class TestOpenRouterClient:
    """Test OpenRouterClient initialization and basic functionality."""

    def test_init_with_default_settings(self):
        """Test OpenRouterClient initialization with default settings."""
        with patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings:
            mock_settings.return_value = create_mock_settings(api_key="test-api-key")

            with (
                patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
                patch("src.mcp_integration.openrouter_client.get_circuit_breaker") as mock_get_cb,
            ):
                mock_registry.return_value = Mock(spec=ModelRegistry)
                mock_get_cb.return_value = Mock()

                client = OpenRouterClient()

                assert client.api_key == "test-api-key"
                assert client.base_url == "https://openrouter.ai/api/v1"
                assert client.timeout == 30.0
                assert client.max_retries == 3
                assert client.site_url == "https://promptcraft.io"
                assert client.app_name == "PromptCraft-Hybrid"
                assert client.connection_state == MCPConnectionState.DISCONNECTED
                assert client.session is None
                assert client.error_count == 0
                assert client.last_successful_request is None

    def test_init_with_custom_parameters(self):
        """Test OpenRouterClient initialization with custom parameters."""
        with patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings:
            mock_settings_obj = create_mock_settings(api_key=None, circuit_breaker_enabled=False)
            mock_settings.return_value = mock_settings_obj

            with (
                patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
                patch("src.mcp_integration.openrouter_client.get_circuit_breaker") as mock_get_cb,
            ):
                mock_registry.return_value = Mock(spec=ModelRegistry)
                mock_get_cb.return_value = Mock()

                client = OpenRouterClient(
                    api_key="custom-key",
                    base_url="https://custom.api.com/v1",
                    timeout=60.0,
                    max_retries=5,
                    site_url="https://example.com",
                    app_name="CustomApp",
                )

                assert client.api_key == "custom-key"
                assert client.base_url == "https://custom.api.com/v1"
                assert client.timeout == 60.0
                assert client.max_retries == 5
                assert client.site_url == "https://example.com"
                assert client.app_name == "CustomApp"

    def test_init_without_api_key(self):
        """Test OpenRouterClient initialization without API key logs warning."""
        with patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings:
            mock_settings_obj = create_mock_settings(api_key=None, circuit_breaker_enabled=False)
            mock_settings.return_value = mock_settings_obj

            with (
                patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
                patch("src.mcp_integration.openrouter_client.get_circuit_breaker") as mock_get_cb,
            ):
                mock_registry.return_value = Mock(spec=ModelRegistry)
                mock_get_cb.return_value = Mock()

                with patch("src.mcp_integration.openrouter_client.logger") as mock_logger:
                    client = OpenRouterClient()

                    assert client.api_key is None
                    mock_logger.warning.assert_called_once_with(
                        "OpenRouter API key not configured - client will be limited",
                    )

    def test_get_headers_with_api_key(self):
        """Test _get_headers method with API key."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings(api_key="test-key")
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            headers = client._get_headers()

            expected_headers = {
                "Content-Type": "application/json",
                "User-Agent": "PromptCraft-Hybrid/1.0",
                "HTTP-Referer": "https://promptcraft.io",
                "X-Title": "PromptCraft-Hybrid",
                "Authorization": "Bearer test-key",
            }

            assert headers == expected_headers

    def test_get_headers_without_api_key(self):
        """Test _get_headers method without API key."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings(api_key=None, circuit_breaker_enabled=False)
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient(api_key=None)
            headers = client._get_headers()

            expected_headers = {
                "Content-Type": "application/json",
                "User-Agent": "PromptCraft-Hybrid/1.0",
                "HTTP-Referer": "https://promptcraft.io",
                "X-Title": "PromptCraft-Hybrid",
            }

            assert headers == expected_headers
            assert "Authorization" not in headers


class TestOpenRouterClientConnection:
    """Test OpenRouterClient connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to OpenRouter API."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            # Mock httpx.AsyncClient
            mock_session = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_session.get.return_value = mock_response

            with patch("httpx.AsyncClient", return_value=mock_session):
                result = await client.connect()

                assert result is True
                assert client.connection_state == MCPConnectionState.CONNECTED
                assert client.session == mock_session
                assert client.last_successful_request is not None

                # Verify session configuration
                mock_session.get.assert_called_once_with("/models", timeout=10.0)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_connect_authentication_failure(self):
        """Test connection failure due to authentication error."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings(api_key="invalid-key")
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            # Mock authentication failure
            mock_session = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 401
            mock_error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=mock_response)
            mock_session.get.side_effect = mock_error

            with patch("httpx.AsyncClient", return_value=mock_session):
                with pytest.raises(MCPAuthenticationError) as exc_info:
                    await client.connect()

                assert "OpenRouter API key authentication failed" in str(exc_info.value)
                assert client.connection_state == MCPConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_degraded_state(self):
        """Test connection in degraded state on HTTP error."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            # Mock HTTP error response
            mock_session = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 503  # Service Unavailable
            mock_session.get.return_value = mock_response

            with patch("httpx.AsyncClient", return_value=mock_session):
                result = await client.connect()

                assert result is False
                assert client.connection_state == MCPConnectionState.DEGRADED

    @pytest.mark.asyncio
    async def test_disconnect_success(self):
        """Test successful disconnection."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            mock_session = AsyncMock()
            client.session = mock_session
            client.connection_state = MCPConnectionState.CONNECTED

            result = await client.disconnect()

            assert result is True
            assert client.connection_state == MCPConnectionState.DISCONNECTED
            assert client.session is None
            mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_with_error(self):
        """Test disconnection with error."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            client.session = AsyncMock()
            client.session.aclose.side_effect = Exception("Close error")

            result = await client.disconnect()

            assert result is False


class TestOpenRouterClientHealthCheck:
    """Test OpenRouterClient health check functionality."""

    @pytest.mark.asyncio
    async def test_async_health_check_healthy(self):
        """Test async health check with healthy status."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            client.session = AsyncMock()
            client.connection_state = MCPConnectionState.CONNECTED
            client.error_count = 0
            client.last_successful_request = time.time()

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
            client.session.get.return_value = mock_response

            health_status = await client.async_health_check()

            assert isinstance(health_status, MCPHealthStatus)
            assert health_status.connection_state == MCPConnectionState.CONNECTED
            assert health_status.error_count == 0
            assert health_status.metadata["service"] == "openrouter"
            assert health_status.metadata["models_available"] == 2

    @pytest.mark.asyncio
    async def test_async_health_check_degraded(self):
        """Test async health check with degraded status."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            client.session = AsyncMock()
            client.connection_state = MCPConnectionState.CONNECTED
            client.error_count = 1

            # Mock HTTP error response
            mock_response = Mock()
            mock_response.status_code = 503
            client.session.get.return_value = mock_response

            health_status = await client.async_health_check()

            assert health_status.connection_state == MCPConnectionState.DEGRADED
            assert health_status.error_count == 2  # Incremented during health check
            assert health_status.metadata["error"] == "HTTP 503"

    @pytest.mark.asyncio
    async def test_async_health_check_unhealthy(self):
        """Test async health check with unhealthy status."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            client.session = AsyncMock()
            client.session.get.side_effect = Exception("Connection failed")

            health_status = await client.async_health_check()

            assert health_status.connection_state == MCPConnectionState.FAILED
            assert "Connection failed" in health_status.metadata["error"]


class TestOpenRouterClientValidation:
    """Test OpenRouterClient query validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_query_valid(self):
        """Test query validation with valid query."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            result = await client.validate_query("What is the capital of France?")

            assert result["is_valid"] is True
            assert result["sanitized_query"] == "What is the capital of France?"
            assert result["potential_issues"] == []

    @pytest.mark.asyncio
    async def test_validate_query_too_long(self):
        """Test query validation with overly long query."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            long_query = "x" * 60000  # Exceeds 50K limit

            result = await client.validate_query(long_query)

            assert result["is_valid"] is False
            assert len(result["sanitized_query"]) == 50000
            assert "Query length exceeds recommended limit" in result["potential_issues"]

    @pytest.mark.asyncio
    async def test_validate_query_suspicious_content(self):
        """Test query validation with suspicious content."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            result = await client.validate_query('<script>alert("xss")</script>')

            assert result["is_valid"] is False
            assert "Query contains potentially unsafe content" in result["potential_issues"]

    @pytest.mark.asyncio
    async def test_validate_query_excessive_repetition(self):
        """Test query validation with excessive repetition."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            repetitive_query = " ".join(["spam"] * 120)  # Exceeds repetition limit (>100 words and >50 repetitions)

            result = await client.validate_query(repetitive_query)

            assert result["is_valid"] is False
            assert "Query contains excessive repetition" in result["potential_issues"]


class TestOpenRouterClientOrchestration:
    """Test OpenRouterClient agent orchestration functionality."""

    @pytest.mark.asyncio
    async def test_orchestrate_agents_success(self):
        """Test successful agent orchestration."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            # Mock model registry
            mock_registry_obj = Mock(spec=ModelRegistry)
            mock_registry_obj.select_best_model.return_value = "deepseek/deepseek-chat-v3-0324:free"
            mock_registry.return_value = mock_registry_obj

            client = OpenRouterClient()
            client.session = AsyncMock()

            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test response content"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
            client.session.post.return_value = mock_response

            # Create test workflow step
            step = WorkflowStep(
                step_id="test-step",
                agent_id="test-agent",
                input_data={"query": "Test question"},
                timeout_seconds=30,
            )

            responses = await client.orchestrate_agents([step])

            assert len(responses) == 1
            response = responses[0]
            assert isinstance(response, Response)
            assert response.agent_id == "test-agent"
            assert response.content == "Test response content"
            assert response.success is True
            assert response.confidence > 0.0
            assert "model_id" in response.metadata

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_orchestrate_agents_connection_error(self):
        """Test orchestration with connection error."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            # Mock connect method to fail
            client.connect = AsyncMock(return_value=False)

            step = WorkflowStep(
                step_id="test-step",
                agent_id="test-agent",
                input_data={"query": "Test question"},
                timeout_seconds=30,
            )

            with pytest.raises(MCPConnectionError):
                await client.orchestrate_agents([step])

    @pytest.mark.asyncio
    async def test_execute_single_step_timeout(self):
        """Test single step execution with timeout."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            mock_registry_obj = Mock(spec=ModelRegistry)
            mock_registry_obj.select_best_model.return_value = "test-model"
            mock_registry.return_value = mock_registry_obj

            client = OpenRouterClient()
            client.session = AsyncMock()
            client.session.post.side_effect = httpx.TimeoutException("Request timeout")

            step = WorkflowStep(
                step_id="test-step",
                agent_id="test-agent",
                input_data={"query": "Test question"},
                timeout_seconds=1,
            )

            with pytest.raises(MCPTimeoutError):
                await client._execute_single_step(step)

    @pytest.mark.asyncio
    async def test_execute_single_step_no_query(self):
        """Test single step execution without query."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            client.session = AsyncMock()

            step = WorkflowStep(
                step_id="test-step",
                agent_id="test-agent",
                input_data={},
                timeout_seconds=30,  # No query
            )

            with pytest.raises(MCPValidationError) as exc_info:
                await client._execute_single_step(step)

            assert "No query provided" in str(exc_info.value)


class TestOpenRouterClientErrorHandling:
    """Test OpenRouterClient error handling functionality."""

    @pytest.mark.asyncio
    async def test_handle_api_error_authentication(self):
        """Test API error handling for authentication failure."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"error": "Invalid API key"}

            with pytest.raises(MCPAuthenticationError) as exc_info:
                await client._handle_api_error(mock_response)

            assert "OpenRouter API authentication failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_api_error_rate_limit(self):
        """Test API error handling for rate limiting."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "120"}

            with pytest.raises(MCPRateLimitError) as exc_info:
                await client._handle_api_error(mock_response)

            assert exc_info.value.retry_after == 120

    @pytest.mark.asyncio
    async def test_handle_api_error_validation(self):
        """Test API error handling for validation errors."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"message": "Invalid request format"}

            with pytest.raises(MCPValidationError) as exc_info:
                await client._handle_api_error(mock_response)

            assert "Invalid OpenRouter request" in str(exc_info.value)

    def test_calculate_confidence_basic(self):
        """Test confidence calculation with basic response."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            api_response = {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 100}}
            content = "This is a good quality response with sufficient length."

            confidence = client._calculate_confidence(api_response, content)

            assert 0.0 <= confidence <= 1.0
            assert confidence > 0.8  # Should be higher than base

    def test_calculate_confidence_short_content(self):
        """Test confidence calculation with short content."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()

            api_response = {"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 5}}
            content = "Short"

            confidence = client._calculate_confidence(api_response, content)

            assert 0.0 <= confidence <= 1.0
            assert confidence < 0.8  # Should be lower due to short content and truncation


class TestOpenRouterClientCapabilities:
    """Test OpenRouterClient capabilities functionality."""

    @pytest.mark.asyncio
    async def test_get_capabilities_success(self):
        """Test successful capabilities retrieval."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            # Mock model registry with capabilities
            mock_registry_obj = Mock(spec=ModelRegistry)
            mock_capabilities = Mock(spec=ModelCapabilities)
            mock_capabilities.supports_function_calling = True
            mock_capabilities.supports_vision = True
            mock_capabilities.supports_reasoning = False
            mock_registry_obj.get_model_capabilities.return_value = mock_capabilities
            mock_registry.return_value = mock_registry_obj

            client = OpenRouterClient()
            client.session = AsyncMock()

            # Mock API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
            client.session.get.return_value = mock_response
            mock_response.raise_for_status.return_value = None

            capabilities = await client.get_capabilities()

            assert "chat_completion" in capabilities
            assert "text_generation" in capabilities
            assert "model_routing" in capabilities
            assert "fallback_chains" in capabilities
            assert "function_calling" in capabilities
            assert "vision" in capabilities
            assert "reasoning" not in capabilities  # Should not be added

    @pytest.mark.asyncio
    async def test_get_capabilities_error(self):
        """Test capabilities retrieval with error."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            client = OpenRouterClient()
            client.session = AsyncMock()
            client.session.get.side_effect = Exception("API Error")

            with pytest.raises(MCPError) as exc_info:
                await client.get_capabilities()

            assert "Capability query failed" in str(exc_info.value)


class TestOpenRouterClientIntegration:
    """Test OpenRouterClient integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """Test complete workflow from connection to orchestration."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
        ):
            mock_settings_obj = create_mock_settings()
            mock_settings.return_value = mock_settings_obj

            mock_registry_obj = Mock(spec=ModelRegistry)
            mock_registry_obj.select_best_model.return_value = "test-model"
            mock_registry.return_value = mock_registry_obj

            client = OpenRouterClient()

            # Mock session and responses
            mock_session = AsyncMock()

            # Connection test response
            connect_response = Mock()
            connect_response.status_code = 200

            # Chat completion response
            chat_response = Mock()
            chat_response.status_code = 200
            chat_response.json.return_value = {
                "choices": [{"message": {"content": "Integration test response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

            mock_session.get.return_value = connect_response
            mock_session.post.return_value = chat_response

            with patch("httpx.AsyncClient", return_value=mock_session):
                # Connect
                connected = await client.connect()
                assert connected is True

                # Validate query
                validation = await client.validate_query("Test query")
                assert validation["is_valid"] is True

                # Orchestrate
                step = WorkflowStep(
                    step_id="integration-test",
                    agent_id="test-agent",
                    input_data={"query": "Test integration"},
                    timeout_seconds=30,
                )

                responses = await client.orchestrate_agents([step])
                assert len(responses) == 1
                assert responses[0].content == "Integration test response"

                # Disconnect
                disconnected = await client.disconnect()
                assert disconnected is True


# Additional comprehensive tests to improve coverage beyond 80%


@pytest.mark.unit
class TestOpenRouterClientAdditionalCoverage:
    """Additional tests to ensure comprehensive coverage of OpenRouterClient."""

    @pytest.mark.asyncio
    async def test_query_validation_max_length(self):
        """Test query validation for maximum length."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()
            client = OpenRouterClient()

            # Create query that exceeds max length
            long_query = "x" * 60000  # Exceeds MAX_QUERY_LENGTH of 50000

            validation = await client.validate_query(long_query)
            assert validation["is_valid"] is False
            assert "error" in validation
            assert "too long" in validation["error"].lower()

    @pytest.mark.asyncio
    async def test_query_validation_empty_query(self):
        """Test query validation for empty query."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()
            client = OpenRouterClient()

            validation = await client.validate_query("")
            assert validation["is_valid"] is False
            assert "error" in validation
            assert "empty" in validation["error"].lower()

    @pytest.mark.asyncio
    async def test_query_validation_none_query(self):
        """Test query validation for None query."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()
            client = OpenRouterClient()

            validation = await client.validate_query(None)
            assert validation["is_valid"] is False
            assert "error" in validation
            assert "empty" in validation["error"].lower()

    async def test_health_check_healthy(self):
        """Test health check when client is healthy."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()
            client = OpenRouterClient()
            client.connection_state = MCPConnectionState.CONNECTED
            client.error_count = 0
            client.last_successful_request = time.time() - 30  # 30 seconds ago

            health = await client.health_check()

            assert health.connection_state == MCPConnectionState.CONNECTED
            assert health.error_count == 0
            assert "HEALTHY" in health.metadata["status"]
            assert "healthy" in health.metadata["message"].lower()

    async def test_health_check_unhealthy_disconnected(self):
        """Test health check when client is disconnected."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()
            client = OpenRouterClient()
            client.connection_state = MCPConnectionState.DISCONNECTED

            health = await client.health_check()

            assert health.connection_state == MCPConnectionState.DISCONNECTED
            assert "UNHEALTHY" in health.metadata["status"]
            assert "not connected" in health.metadata["message"].lower()

    async def test_health_check_degraded_high_errors(self):
        """Test health check when client has high error count."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()
            client = OpenRouterClient()
            client.connection_state = MCPConnectionState.CONNECTED
            client.error_count = 15  # High error count

            health = await client.health_check()

            assert health.connection_state == MCPConnectionState.CONNECTED
            assert health.error_count == 15
            assert "DEGRADED" in health.metadata["status"]
            assert "error count" in health.metadata["message"].lower()

    async def test_health_check_degraded_old_request(self):
        """Test health check when last successful request is old."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()
            client = OpenRouterClient()
            client.connection_state = MCPConnectionState.CONNECTED
            client.error_count = 0
            client.last_successful_request = time.time() - 3700  # Over an hour ago

            health = await client.health_check()

            assert health.connection_state == MCPConnectionState.CONNECTED
            assert health.error_count == 0
            assert "DEGRADED" in health.metadata["status"]
            assert "hour ago" in health.metadata["message"].lower()

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
            patch("src.mcp_integration.openrouter_client.get_circuit_breaker") as mock_get_cb,
        ):
            mock_settings.return_value = create_mock_settings(circuit_breaker_enabled=True)
            mock_circuit_breaker = Mock()
            mock_get_cb.return_value = mock_circuit_breaker

            client = OpenRouterClient()

            # Verify circuit breaker is set up
            assert client.circuit_breaker == mock_circuit_breaker
            mock_get_cb.assert_called_once_with("openrouter", mock_settings.return_value)

    def test_no_circuit_breaker_when_disabled(self):
        """Test no circuit breaker when disabled in settings."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings(circuit_breaker_enabled=False)

            client = OpenRouterClient()

            # Verify no circuit breaker
            assert client.circuit_breaker is None

    def test_base_url_strip_trailing_slash(self):
        """Test that base URL strips trailing slashes."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry"),
        ):
            mock_settings.return_value = create_mock_settings()

            client = OpenRouterClient(base_url="https://api.example.com/v1/")

            # Should strip trailing slash
            assert client.base_url == "https://api.example.com/v1"

    def test_model_capabilities_integration(self):
        """Test integration with model capabilities."""
        with (
            patch("src.mcp_integration.openrouter_client.get_settings") as mock_settings,
            patch("src.mcp_integration.openrouter_client.get_model_registry") as mock_registry,
        ):
            mock_settings.return_value = create_mock_settings()

            # Mock model registry with capabilities using the correct interface
            mock_registry_instance = Mock()
            mock_capabilities = ModelCapabilities(
                model_id="test_model",
                display_name="Test Model",
                provider="test",
                category="free_general",
                context_window=4096,
                max_tokens_per_request=2048,
                rate_limit_requests_per_minute=100,
                supports_function_calling=True,
                supports_vision=False,
            )
            mock_registry_instance.get_model_capabilities.return_value = mock_capabilities
            mock_registry.return_value = mock_registry_instance

            client = OpenRouterClient()

            # Test capabilities retrieval
            capabilities = client.model_registry.get_model_capabilities("test_model")
            assert capabilities.context_window == 4096
            assert capabilities.supports_function_calling is True
            assert capabilities.supports_vision is False
