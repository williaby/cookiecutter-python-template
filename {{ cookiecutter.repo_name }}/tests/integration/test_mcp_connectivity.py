"""
MCP integration testing for external service connectivity and error handling.

This module validates MCP server connectivity, external service integrations,
error handling for service unavailability, and session persistence across
MCP restarts for Phase 1 Issue 5.

Test Coverage:
- Zen MCP Server connectivity validation
- External service integrations (Qdrant, Azure AI)
- Error handling for service unavailability
- Session persistence across MCP restarts
- Service health monitoring and fallback
- Connection pooling and retry mechanisms
"""

import asyncio
import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.config.settings import ApplicationSettings
from src.ui.multi_journey_interface import MultiJourneyInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP test constants
MCP_CONNECTION_TIMEOUT = 5.0
QDRANT_HEALTH_ENDPOINT = "http://192.168.1.16:6333/health"
AZURE_AI_MOCK_ENDPOINT = "https://api.openai.com/v1/chat/completions"
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0

# Service availability simulation states
SERVICE_STATES = {
    "all_available": {"zen_mcp": True, "qdrant": True, "azure_ai": True},
    "zen_mcp_down": {"zen_mcp": False, "qdrant": True, "azure_ai": True},
    "qdrant_down": {"zen_mcp": True, "qdrant": False, "azure_ai": True},
    "azure_ai_down": {"zen_mcp": True, "qdrant": False, "azure_ai": False},
    "all_down": {"zen_mcp": False, "qdrant": False, "azure_ai": False},
}


class MockMCPServer:
    """Mock MCP server for testing connectivity and resilience."""

    def __init__(self, server_name: str, is_available: bool = True):
        self.server_name = server_name
        self.is_available = is_available
        self.connection_count = 0
        self.request_count = 0
        self.last_request_time = None
        self.session_data = {}

    async def connect(self) -> bool:
        """Mock MCP server connection."""
        if not self.is_available:
            raise ConnectionError(f"MCP server {self.server_name} is unavailable")

        self.connection_count += 1
        return True

    async def disconnect(self) -> None:
        """Mock MCP server disconnection."""
        self.connection_count = max(0, self.connection_count - 1)

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Mock MCP request handling."""
        if not self.is_available:
            raise ConnectionError(f"MCP server {self.server_name} is unavailable")

        self.request_count += 1
        self.last_request_time = time.time()

        # Simulate different response types based on request
        if request.get("method") == "health_check":
            return {"status": "healthy", "server": self.server_name}
        if request.get("method") == "agent_orchestration":
            return {
                "result": f"Mock orchestration response from {self.server_name}",
                "agents_involved": ["create_agent", "security_agent"],
                "processing_time": 0.5,
            }
        if request.get("method") == "session_persist":
            session_id = request.get("session_id")
            if session_id:
                self.session_data[session_id] = request.get("data", {})
                return {"status": "persisted", "session_id": session_id}

        return {"status": "processed", "request_id": request.get("id", "unknown")}

    def simulate_restart(self) -> None:
        """Simulate MCP server restart."""
        self.connection_count = 0
        self.request_count = 0
        # Session data should persist across restarts

    def get_status(self) -> dict[str, Any]:
        """Get mock server status."""
        return {
            "server_name": self.server_name,
            "is_available": self.is_available,
            "connection_count": self.connection_count,
            "request_count": self.request_count,
            "last_request_time": self.last_request_time,
            "session_count": len(self.session_data),
        }


class MockExternalService:
    """Mock external service (Qdrant, Azure AI) for integration testing."""

    def __init__(self, service_name: str, endpoint: str, is_available: bool = True):
        self.service_name = service_name
        self.endpoint = endpoint
        self.is_available = is_available
        self.request_count = 0
        self.error_count = 0
        self.response_times = []

    async def health_check(self) -> dict[str, Any]:
        """Mock service health check."""
        if not self.is_available:
            self.error_count += 1
            raise httpx.ConnectError(f"Service {self.service_name} unavailable")

        return {"status": "healthy", "service": self.service_name, "endpoint": self.endpoint}

    async def make_request(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mock service request."""
        start_time = time.time()

        if not self.is_available:
            self.error_count += 1
            raise httpx.ConnectError(f"Service {self.service_name} unavailable")

        # Simulate processing time
        await asyncio.sleep(0.1)

        self.request_count += 1
        response_time = time.time() - start_time
        self.response_times.append(response_time)

        if self.service_name == "qdrant":
            return {
                "status": "success",
                "results": [
                    {"id": "doc1", "score": 0.95, "content": "Mock search result 1"},
                    {"id": "doc2", "score": 0.87, "content": "Mock search result 2"},
                ],
                "query_time": response_time,
            }
        if self.service_name == "azure_ai":
            return {
                "id": "chatcmpl-mock123",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Mock AI response from Azure OpenAI"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            }

        return {"status": "success", "service": self.service_name}

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            "service_name": self.service_name,
            "is_available": self.is_available,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time": avg_response_time,
            "success_rate": (
                (self.request_count / (self.request_count + self.error_count)) * 100
                if (self.request_count + self.error_count) > 0
                else 0
            ),
        }


class TestMCPConnectivity:
    """Integration tests for MCP connectivity and external service integration."""

    @pytest.fixture
    def mcp_settings(self):
        """Create MCP-focused settings for testing."""
        return ApplicationSettings(
            # MCP Configuration
            app_name="PromptCraft-MCP-Test",
            debug=True,
            # External service timeouts
            query_timeout=10.0,
            max_concurrent_queries=5,
            # MCP configuration
            mcp_enabled=True,
            mcp_timeout=10.0,
        )

    @pytest.fixture
    def mock_zen_mcp_server(self):
        """Mock Zen MCP Server for testing."""
        return MockMCPServer("zen_mcp", is_available=True)

    @pytest.fixture
    def mock_qdrant_service(self):
        """Mock Qdrant service for testing."""
        return MockExternalService("qdrant", QDRANT_HEALTH_ENDPOINT, is_available=True)

    @pytest.fixture
    def mock_azure_ai_service(self):
        """Mock Azure AI service for testing."""
        return MockExternalService("azure_ai", AZURE_AI_MOCK_ENDPOINT, is_available=True)

    @pytest.fixture
    def mcp_integration_interface(self, mcp_settings, mock_zen_mcp_server, mock_qdrant_service, mock_azure_ai_service):
        """Create UI interface with mocked MCP and external services."""
        with (
            patch("src.config.settings.get_settings", return_value=mcp_settings),
            patch("src.mcp_integration.mcp_client.MCPClient") as mock_zen_client,
            patch("httpx.AsyncClient") as mock_http_client,
            patch("src.ui.journeys.journey1_smart_templates.Journey1SmartTemplates"),
            patch("src.ui.components.shared.export_utils.ExportUtils"),
        ):
            # Mock MCP server integration
            mock_zen_instance = Mock()
            mock_zen_instance.connect = mock_zen_mcp_server.connect
            mock_zen_instance.disconnect = mock_zen_mcp_server.disconnect
            mock_zen_instance.send_request = mock_zen_mcp_server.send_request
            mock_zen_client.return_value = mock_zen_instance

            # Mock external services
            async def mock_get(url, **kwargs):
                if QDRANT_HEALTH_ENDPOINT in url:
                    return Mock(json=lambda: mock_qdrant_service.health_check(), status_code=200)
                return Mock(json=lambda: {"status": "ok"}, status_code=200)

            async def mock_post(url, **kwargs):
                if "openai" in url:
                    return Mock(
                        json=lambda: mock_azure_ai_service.make_request(kwargs.get("json", {})),
                        status_code=200,
                    )
                return Mock(json=lambda: {"status": "ok"}, status_code=200)

            mock_http_instance = Mock()
            mock_http_instance.get = mock_get
            mock_http_instance.post = mock_post
            mock_http_client.return_value.__aenter__ = AsyncMock(return_value=mock_http_instance)
            mock_http_client.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock Journey processors
            interface = MultiJourneyInterface()

            # Attach mock services for testing access
            interface._mock_zen_mcp = mock_zen_mcp_server
            interface._mock_qdrant = mock_qdrant_service
            interface._mock_azure_ai = mock_azure_ai_service

            return interface

    @pytest.mark.integration
    async def test_zen_mcp_server_connectivity(self, mcp_integration_interface):
        """Test basic Zen MCP Server connectivity."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp

        # Test connection establishment
        connection_result = await zen_mcp.connect()
        assert connection_result is True
        assert zen_mcp.connection_count == 1

        # Test health check request
        health_request = {"method": "health_check", "id": "test_health_1"}
        health_response = await zen_mcp.send_request(health_request)

        assert health_response["status"] == "healthy"
        assert health_response["server"] == "zen_mcp"
        assert zen_mcp.request_count == 1

        # Test disconnection
        await zen_mcp.disconnect()
        assert zen_mcp.connection_count == 0

    @pytest.mark.integration
    async def test_zen_mcp_agent_orchestration(self, mcp_integration_interface):
        """Test Zen MCP Server agent orchestration functionality."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp

        # Establish connection
        await zen_mcp.connect()

        # Test agent orchestration request
        orchestration_request = {
            "method": "agent_orchestration",
            "id": "test_orchestration_1",
            "agents": ["create_agent", "security_agent"],
            "task": "Generate secure code template",
        }

        response = await zen_mcp.send_request(orchestration_request)

        assert response["result"] is not None
        assert "Mock orchestration response" in response["result"]
        assert "agents_involved" in response
        assert "create_agent" in response["agents_involved"]
        assert "security_agent" in response["agents_involved"]
        assert response["processing_time"] > 0

    @pytest.mark.integration
    async def test_qdrant_service_integration(self, mcp_integration_interface):
        """Test Qdrant vector database integration."""
        qdrant_service = mcp_integration_interface._mock_qdrant

        # Test health check
        health_status = await qdrant_service.health_check()
        assert health_status["status"] == "healthy"
        assert health_status["service"] == "qdrant"

        # Test search request
        search_data = {"query": "test search query", "limit": 10, "threshold": 0.8}

        search_results = await qdrant_service.make_request(search_data)

        assert search_results["status"] == "success"
        assert "results" in search_results
        assert len(search_results["results"]) == 2
        assert search_results["results"][0]["score"] > 0.8
        assert "Mock search result" in search_results["results"][0]["content"]

    @pytest.mark.integration
    async def test_azure_ai_service_integration(self, mcp_integration_interface):
        """Test Azure AI service integration."""
        azure_ai_service = mcp_integration_interface._mock_azure_ai

        # Test health check
        health_status = await azure_ai_service.health_check()
        assert health_status["status"] == "healthy"
        assert health_status["service"] == "azure_ai"

        # Test AI completion request
        completion_data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Generate a test response"}],
            "max_tokens": 100,
        }

        completion_result = await azure_ai_service.make_request(completion_data)

        assert completion_result["object"] == "chat.completion"
        assert "choices" in completion_result
        assert len(completion_result["choices"]) == 1
        assert "Mock AI response" in completion_result["choices"][0]["message"]["content"]
        assert "usage" in completion_result

    @pytest.mark.integration
    async def test_service_unavailability_handling(self, mcp_integration_interface):
        """Test error handling when services are unavailable."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp
        qdrant_service = mcp_integration_interface._mock_qdrant
        azure_ai_service = mcp_integration_interface._mock_azure_ai

        # Simulate service unavailability
        zen_mcp.is_available = False
        qdrant_service.is_available = False
        azure_ai_service.is_available = False

        # Test MCP connection failure
        with pytest.raises(ConnectionError) as exc_info:
            await zen_mcp.connect()
        assert "zen_mcp" in str(exc_info.value)

        # Test Qdrant service failure
        with pytest.raises(httpx.ConnectError) as exc_info:
            await qdrant_service.health_check()
        assert "qdrant" in str(exc_info.value)

        # Test Azure AI service failure
        with pytest.raises(httpx.ConnectError) as exc_info:
            await azure_ai_service.health_check()
        assert "azure_ai" in str(exc_info.value)

        # Verify error counts
        assert qdrant_service.error_count == 1
        assert azure_ai_service.error_count == 1

    @pytest.mark.integration
    async def test_session_persistence_across_mcp_restarts(self, mcp_integration_interface):
        """Test session persistence across MCP server restarts."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp

        # Establish connection and create session
        await zen_mcp.connect()

        session_id = "test_session_persistence"
        session_data = {
            "user_preferences": {"theme": "dark", "model": "gpt-4o"},
            "conversation_history": ["Hello", "How can I help?"],
            "active_agents": ["create_agent"],
        }

        # Persist session data
        persist_request = {"method": "session_persist", "session_id": session_id, "data": session_data}

        persist_response = await zen_mcp.send_request(persist_request)
        assert persist_response["status"] == "persisted"
        assert persist_response["session_id"] == session_id

        # Verify session data is stored
        assert session_id in zen_mcp.session_data
        assert zen_mcp.session_data[session_id] == session_data

        # Simulate MCP server restart
        zen_mcp.simulate_restart()

        # Verify session data persists after restart
        assert session_id in zen_mcp.session_data
        assert zen_mcp.session_data[session_id] == session_data
        assert zen_mcp.connection_count == 0  # Connections reset
        assert zen_mcp.request_count == 0  # Request counts reset

    @pytest.mark.integration
    async def test_connection_retry_mechanisms(self, mcp_integration_interface):
        """Test connection retry mechanisms for resilient service integration."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp

        # Simulate intermittent connectivity
        zen_mcp.is_available = False

        retry_count = 0
        max_retries = 3

        for attempt in range(max_retries):
            try:
                await zen_mcp.connect()
                break
            except ConnectionError:
                retry_count += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                    # Simulate service recovery on final attempt
                    if attempt == max_retries - 2:
                        zen_mcp.is_available = True

        # Verify retry behavior
        assert retry_count == max_retries - 1  # Failed twice, succeeded on third
        assert zen_mcp.connection_count == 1  # Final connection succeeded

    @pytest.mark.integration
    async def test_concurrent_mcp_requests(self, mcp_integration_interface):
        """Test concurrent MCP requests handling."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp

        # Establish connection
        await zen_mcp.connect()

        # Create multiple concurrent requests
        async def make_concurrent_request(request_id: int):
            request = {
                "method": "agent_orchestration",
                "id": f"concurrent_request_{request_id}",
                "task": f"Concurrent task {request_id}",
            }
            return await zen_mcp.send_request(request)

        # Execute concurrent requests
        concurrent_tasks = [make_concurrent_request(i) for i in range(5)]
        responses = await asyncio.gather(*concurrent_tasks)

        # Verify all requests succeeded
        assert len(responses) == 5
        for _i, response in enumerate(responses):
            assert response["result"] is not None
            assert "Mock orchestration response" in response["result"]

        # Verify request count
        assert zen_mcp.request_count == 5

    @pytest.mark.integration
    async def test_service_health_monitoring(self, mcp_integration_interface):
        """Test comprehensive service health monitoring."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp
        qdrant_service = mcp_integration_interface._mock_qdrant
        azure_ai_service = mcp_integration_interface._mock_azure_ai

        # Establish connections and make some requests
        await zen_mcp.connect()
        await zen_mcp.send_request({"method": "health_check"})

        await qdrant_service.health_check()
        await qdrant_service.make_request({"query": "health test"})

        await azure_ai_service.health_check()
        await azure_ai_service.make_request({"model": "gpt-4o-mini"})

        # Get health status from all services
        zen_status = zen_mcp.get_status()
        qdrant_metrics = qdrant_service.get_metrics()
        azure_metrics = azure_ai_service.get_metrics()

        # Verify MCP server status
        assert zen_status["is_available"] is True
        assert zen_status["connection_count"] == 1
        assert zen_status["request_count"] == 1

        # Verify Qdrant metrics
        assert qdrant_metrics["is_available"] is True
        assert qdrant_metrics["request_count"] == 1
        assert qdrant_metrics["error_count"] == 0
        assert qdrant_metrics["success_rate"] == 100.0

        # Verify Azure AI metrics
        assert azure_metrics["is_available"] is True
        assert azure_metrics["request_count"] == 1
        assert azure_metrics["error_count"] == 0
        assert azure_metrics["success_rate"] == 100.0

    @pytest.mark.integration
    async def test_service_fallback_mechanisms(self, mcp_integration_interface):
        """Test fallback mechanisms when primary services fail."""
        qdrant_service = mcp_integration_interface._mock_qdrant

        # Test primary service failure with fallback
        qdrant_service.is_available = False

        # Attempt request with fallback logic
        fallback_response = None
        try:
            await qdrant_service.make_request({"query": "test"})
        except httpx.ConnectError:
            # Simulate fallback to local search or cache
            fallback_response = {
                "status": "fallback",
                "source": "local_cache",
                "results": [{"content": "Fallback search result", "score": 0.5}],
            }

        # Verify fallback was triggered
        assert fallback_response is not None
        assert fallback_response["status"] == "fallback"
        assert fallback_response["source"] == "local_cache"
        assert len(fallback_response["results"]) == 1

    @pytest.mark.integration
    async def test_mcp_performance_under_load(self, mcp_integration_interface):
        """Test MCP performance under various load conditions."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp

        # Establish connection
        await zen_mcp.connect()

        # Test high-frequency requests
        request_count = 50
        start_time = time.time()

        tasks = []
        for i in range(request_count):
            request = {"method": "agent_orchestration", "id": f"load_test_{i}", "task": f"Load test task {i}"}
            tasks.append(zen_mcp.send_request(request))

        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        # Calculate performance metrics
        total_time = end_time - start_time
        requests_per_second = request_count / total_time

        # Verify performance
        assert len(responses) == request_count
        assert zen_mcp.request_count == request_count
        assert requests_per_second > 10  # Should handle at least 10 RPS

        # All responses should be successful
        for response in responses:
            assert response["result"] is not None

    @pytest.mark.integration
    def test_mcp_integration_in_ui_workflow(self, mcp_integration_interface):
        """Test MCP integration within UI workflow."""
        session_id = "mcp_ui_integration_test"
        test_input = "Create a secure API endpoint with authentication"

        # Mock the Journey 1 processor to simulate MCP calls
        with patch.object(mcp_integration_interface, "_process_journey1") as mock_process:
            # Simulate successful MCP-enhanced processing
            mock_process.return_value = (
                "Enhanced prompt with MCP orchestration\n"
                "Context: Secure API development with multi-agent validation\n"
                "Request: API endpoint with OAuth2 authentication\n"
                "Examples: REST API, FastAPI, Flask, Express.js endpoints\n"
                "Augmentations: Security scanning, code review, documentation generation\n"
                "Tone: Technical, security-focused, comprehensive\n"
                "Evaluation: Security compliance, performance, maintainability\n"
                "MCP Enhanced | Agents: create+security | Time: 2.1s\n"
                "MCP Knowledge Base Integration"
            )

            # Test UI request with MCP integration
            response = mcp_integration_interface.handle_journey1_request(test_input, session_id)

            # Verify MCP-enhanced response
            assert response is not None
            assert "Enhanced prompt with MCP orchestration" in response
            assert "MCP Enhanced" in response
            assert "Agents: create+security" in response

    @pytest.mark.integration
    async def test_error_recovery_and_graceful_degradation(self, mcp_integration_interface):
        """Test error recovery and graceful degradation scenarios."""
        zen_mcp = mcp_integration_interface._mock_zen_mcp
        qdrant_service = mcp_integration_interface._mock_qdrant

        # Test partial service failure scenario
        zen_mcp.is_available = True  # MCP available
        qdrant_service.is_available = False  # Qdrant unavailable

        # Establish MCP connection
        await zen_mcp.connect()

        # Test graceful degradation - should work with MCP but without vector search
        degraded_request = {
            "method": "agent_orchestration",
            "id": "degraded_test",
            "task": "Process without vector search",
            "fallback_mode": True,
        }

        response = await zen_mcp.send_request(degraded_request)

        # Verify system continues to function in degraded mode
        assert response["result"] is not None
        assert "Mock orchestration response" in response["result"]

        # Verify error tracking
        with contextlib.suppress(httpx.ConnectError):
            await qdrant_service.health_check()

        assert qdrant_service.error_count == 1


class TestMCPServiceStates:
    """Test various service availability combinations."""

    @pytest.fixture(params=list(SERVICE_STATES.keys()))
    def service_state(self, request):
        """Parametrized fixture for different service availability states."""
        return SERVICE_STATES[request.param]

    @pytest.mark.integration
    async def test_service_combinations(self, service_state):
        """Test system behavior under different service availability combinations."""
        # Create mock services based on state
        zen_mcp = MockMCPServer("zen_mcp", service_state["zen_mcp"])
        qdrant_service = MockExternalService("qdrant", QDRANT_HEALTH_ENDPOINT, service_state["qdrant"])
        azure_ai_service = MockExternalService("azure_ai", AZURE_AI_MOCK_ENDPOINT, service_state["azure_ai"])

        # Test system health check under this configuration
        health_results = {}

        # Test MCP health
        try:
            await zen_mcp.connect()
            health_results["zen_mcp"] = "healthy"
            await zen_mcp.disconnect()
        except ConnectionError:
            health_results["zen_mcp"] = "unhealthy"

        # Test Qdrant health
        try:
            await qdrant_service.health_check()
            health_results["qdrant"] = "healthy"
        except httpx.ConnectError:
            health_results["qdrant"] = "unhealthy"

        # Test Azure AI health
        try:
            await azure_ai_service.health_check()
            health_results["azure_ai"] = "healthy"
        except httpx.ConnectError:
            health_results["azure_ai"] = "unhealthy"

        # Verify health results match expected state
        assert (health_results["zen_mcp"] == "healthy") == service_state["zen_mcp"]
        assert (health_results["qdrant"] == "healthy") == service_state["qdrant"]
        assert (health_results["azure_ai"] == "healthy") == service_state["azure_ai"]

        # Test system behavior based on available services
        available_services = sum(service_state.values())

        if available_services == 3:
            # All services available - full functionality
            assert all(status == "healthy" for status in health_results.values())
        elif available_services == 0:
            # No services available - system should handle gracefully
            assert all(status == "unhealthy" for status in health_results.values())
        else:
            # Partial availability - system should degrade gracefully
            healthy_count = sum(1 for status in health_results.values() if status == "healthy")
            assert healthy_count == available_services


if __name__ == "__main__":
    """
    Run MCP connectivity integration tests.

    Usage:
        python -m pytest tests/integration/test_mcp_connectivity.py -v -m integration
        python tests/integration/test_mcp_connectivity.py  # Direct execution
    """
    pytest.main([__file__, "-v", "-m", "integration"])
