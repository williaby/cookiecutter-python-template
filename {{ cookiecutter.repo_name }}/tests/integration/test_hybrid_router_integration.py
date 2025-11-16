"""
Integration tests for HybridRouter - Phase 1 Issue NEW-11.

Tests the complete integration of HybridRouter with real OpenRouter and MCP clients,
demonstrating end-to-end routing behavior, circuit breaker protection, and gradual
rollout functionality in realistic scenarios.

Integration Test Coverage:
    - Real client integration with mocked external services
    - Circuit breaker behavior with actual failure scenarios
    - Performance monitoring under load
    - Gradual rollout with realistic traffic patterns
    - Service recovery and fallback scenarios
    - Configuration changes and dynamic routing
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import ApplicationSettings
from src.mcp_integration.hybrid_router import (
    HybridRouter,
    RoutingStrategy,
)
from src.mcp_integration.mcp_client import (
    MCPConnectionState,
    MCPHealthStatus,
    Response,
    WorkflowStep,
)
from src.utils.circuit_breaker import CircuitBreakerOpenError


@pytest.fixture
def mock_settings():
    """Create realistic mock settings for integration testing."""
    settings = MagicMock(spec=ApplicationSettings)
    settings.openrouter_traffic_percentage = 50
    settings.circuit_breaker_enabled = True
    settings.openrouter_api_key = MagicMock()
    settings.openrouter_api_key.get_secret_value.return_value = "test_key"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    settings.openrouter_timeout = 30.0
    settings.openrouter_max_retries = 3
    settings.mcp_server_url = "http://localhost:3000"
    settings.mcp_timeout = 30.0
    settings.mcp_max_retries = 3
    return settings


@pytest.fixture
async def mock_openrouter_responses():
    """Mock OpenRouter API responses for integration testing."""
    return {
        "models": {
            "data": [
                {"id": "anthropic/claude-3-sonnet", "name": "Claude 3 Sonnet"},
                {"id": "openai/gpt-4", "name": "GPT-4"},
            ],
        },
        "chat": {
            "choices": [{"message": {"content": "OpenRouter integration test response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    }


@pytest.fixture
async def mock_mcp_responses():
    """Mock MCP server responses for integration testing."""
    return {
        "health": {
            "status": "healthy",
            "response_time": 0.1,
            "capabilities": ["agent_orchestration", "multi_agent_coordination"],
        },
        "orchestrate": [
            {
                "agent_id": "test_agent",
                "content": "MCP integration test response",
                "metadata": {"service": "mcp", "confidence": 0.9},
                "success": True,
            },
        ],
    }


class TestHybridRouterIntegration:
    """Integration tests for HybridRouter with realistic scenarios."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("src.mcp_integration.hybrid_router.get_settings")
    @patch("src.mcp_integration.hybrid_router.get_circuit_breaker")
    async def test_end_to_end_orchestration_flow(
        self,
        mock_get_cb,
        mock_get_settings,
        mock_httpx_client,
        mock_settings,
        mock_openrouter_responses,
        mock_mcp_responses,
    ):
        """Test complete end-to-end orchestration flow."""
        mock_get_settings.return_value = mock_settings

        # Mock circuit breaker
        circuit_breaker = MagicMock()
        circuit_breaker.is_available.return_value = True
        circuit_breaker.call_async = AsyncMock(side_effect=lambda func: func())
        mock_get_cb.return_value = circuit_breaker

        # Mock HTTP client responses for OpenRouter
        mock_client_instance = MagicMock()
        mock_httpx_client.return_value = mock_client_instance

        # Mock successful connection test
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openrouter_responses["models"]
        mock_client_instance.get = AsyncMock(return_value=mock_response)

        # Mock chat completion
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.return_value = mock_openrouter_responses["chat"]
        mock_client_instance.post = AsyncMock(return_value=mock_chat_response)

        # Mock MCP client
        mock_mcp_client = MagicMock()
        mock_mcp_client.connect = AsyncMock(return_value=True)
        mock_mcp_client.connection_state = MCPConnectionState.CONNECTED
        mock_mcp_client.health_check = AsyncMock(
            return_value=MCPHealthStatus(
                connection_state=MCPConnectionState.CONNECTED,
                response_time_ms=0.1,
                error_count=0,
                metadata=mock_mcp_responses["health"],
            ),
        )
        mock_mcp_client.orchestrate_agents = AsyncMock(
            return_value=[
                Response(
                    agent_id="test_agent",
                    content="MCP integration test response",
                    metadata={"service": "mcp", "confidence": 0.9},
                    confidence=0.8,
                    processing_time=0.2,
                    success=True,
                ),
            ],
        )

        # Create HybridRouter with mocked clients
        with (
            patch("src.mcp_integration.openrouter_client.OpenRouterClient") as mock_or_class,
            patch("src.mcp_integration.mcp_client.ZenMCPClient") as mock_mcp_class,
        ):
            mock_or_instance = MagicMock()
            mock_or_instance.connect = AsyncMock(return_value=True)
            mock_or_instance.connection_state = MCPConnectionState.CONNECTED
            mock_or_instance.orchestrate_agents = AsyncMock(
                return_value=[
                    Response(
                        agent_id="test_agent",
                        content="OpenRouter response",
                        metadata={"service": "openrouter"},
                        confidence=0.9,
                        processing_time=0.1,
                        success=True,
                    ),
                ],
            )
            mock_or_class.return_value = mock_or_instance
            mock_mcp_class.return_value = mock_mcp_client

            router = HybridRouter(
                openrouter_client=mock_or_instance,
                mcp_client=mock_mcp_client,
                strategy=RoutingStrategy.OPENROUTER_PRIMARY,
            )

            # Test connection
            await router.connect()
            assert router.connection_state == MCPConnectionState.CONNECTED

            # Test orchestration
            workflow_steps = [
                WorkflowStep(
                    step_id="integration_test_step",
                    agent_id="test_agent",
                    input_data={"query": "Integration test query", "task_type": "general"},
                    timeout_seconds=30.0,
                ),
            ]

            responses = await router.orchestrate_agents(workflow_steps)

            # Verify results
            assert len(responses) == 1
            assert responses[0].success is True
            assert responses[0].agent_id == "test_agent"
            assert router.metrics.total_requests == 1
            assert router.metrics.successful_routes == 1

    @pytest.mark.asyncio
    @patch("src.mcp_integration.hybrid_router.get_settings")
    @patch("src.mcp_integration.hybrid_router.get_circuit_breaker")
    async def test_gradual_rollout_traffic_distribution(self, mock_get_cb, mock_get_settings, mock_settings):
        """Test gradual rollout traffic distribution over multiple requests."""
        mock_settings.openrouter_traffic_percentage = 30  # 30% to OpenRouter
        mock_settings.circuit_breaker_enabled = False  # Disable circuit breaker for consistent testing
        mock_get_settings.return_value = mock_settings

        # Create mock clients that track calls
        openrouter_calls = []
        mcp_calls = []

        mock_openrouter = MagicMock()
        mock_openrouter.connect = AsyncMock(return_value=True)
        mock_openrouter.connection_state = MCPConnectionState.CONNECTED
        mock_openrouter.orchestrate_agents = AsyncMock(
            side_effect=lambda steps: openrouter_calls.append(len(steps))
            or [
                Response(
                    agent_id=step.agent_id,
                    content=f"OpenRouter response {step.step_id}",
                    metadata={"service": "openrouter"},
                    confidence=0.9,
                    processing_time=0.1,
                    success=True,
                )
                for step in steps
            ],
        )

        mock_mcp = MagicMock()
        mock_mcp.connect = AsyncMock(return_value=True)
        mock_mcp.connection_state = MCPConnectionState.CONNECTED
        mock_mcp.orchestrate_agents = AsyncMock(
            side_effect=lambda steps: mcp_calls.append(len(steps))
            or [
                Response(
                    agent_id=step.agent_id,
                    content=f"MCP response {step.step_id}",
                    metadata={"service": "mcp"},
                    confidence=0.8,
                    processing_time=0.2,
                    success=True,
                )
                for step in steps
            ],
        )

        router = HybridRouter(
            openrouter_client=mock_openrouter,
            mcp_client=mock_mcp,
            strategy=RoutingStrategy.OPENROUTER_PRIMARY,  # Use primary strategy for gradual rollout testing
            enable_gradual_rollout=True,  # Explicitly enable gradual rollout
        )

        await router.connect()

        # Simulate multiple requests
        num_requests = 100
        for i in range(num_requests):
            workflow_steps = [
                WorkflowStep(
                    step_id=f"step_{i}",
                    agent_id="test_agent",
                    input_data={"query": f"Test query {i}"},
                    timeout_seconds=30.0,
                ),
            ]

            await router.orchestrate_agents(workflow_steps)

        # Verify traffic distribution
        total_calls = len(openrouter_calls) + len(mcp_calls)
        assert total_calls == num_requests

        openrouter_percentage = (len(openrouter_calls) / total_calls) * 100

        # Allow for some variance due to hash-based routing
        # With 30% traffic percentage, expect roughly 30% to OpenRouter with some variance
        # Hash-based routing should be deterministic but allow for reasonable variance in test runs
        assert 20 <= openrouter_percentage <= 40  # Allow ±10% variance for 30% target

        # Verify metrics
        assert router.metrics.total_requests == num_requests
        assert router.metrics.successful_routes == num_requests
        assert router.metrics.openrouter_requests == len(openrouter_calls)
        assert router.metrics.mcp_requests == len(mcp_calls)

    @pytest.mark.asyncio
    @pytest.mark.integration
    @patch("src.mcp_integration.hybrid_router.get_settings")
    @patch("src.mcp_integration.hybrid_router.get_circuit_breaker")
    async def test_circuit_breaker_protection_and_recovery(self, mock_get_cb, mock_get_settings, mock_settings):
        """Test circuit breaker protection and service recovery."""
        mock_settings.openrouter_traffic_percentage = 100  # Force OpenRouter
        mock_get_settings.return_value = mock_settings

        # Create a circuit breaker that can be controlled
        circuit_breaker_state = {"is_open": False}

        def mock_is_available():
            return not circuit_breaker_state["is_open"]

        async def mock_call_async(func):
            if circuit_breaker_state["is_open"]:
                raise CircuitBreakerOpenError("Circuit breaker is open")
            return await func()

        circuit_breaker = MagicMock()
        circuit_breaker.is_available = mock_is_available
        circuit_breaker.call_async = mock_call_async
        mock_get_cb.return_value = circuit_breaker

        # Create mock clients
        openrouter_failures = 0

        mock_openrouter = MagicMock()
        mock_openrouter.connect = AsyncMock(return_value=True)
        mock_openrouter.connection_state = MCPConnectionState.CONNECTED

        def mock_orchestrate_openrouter(steps):
            nonlocal openrouter_failures
            if openrouter_failures > 0:
                openrouter_failures -= 1
                raise Exception("OpenRouter service failure")
            return [
                Response(
                    agent_id=step.agent_id,
                    content=f"OpenRouter response {step.step_id}",
                    metadata={"service": "openrouter"},
                    confidence=0.9,
                    processing_time=0.1,
                    success=True,
                )
                for step in steps
            ]

        mock_openrouter.orchestrate_agents = AsyncMock(side_effect=mock_orchestrate_openrouter)

        mock_mcp = MagicMock()
        mock_mcp.connect = AsyncMock(return_value=True)
        mock_mcp.connection_state = MCPConnectionState.CONNECTED
        mock_mcp.orchestrate_agents = AsyncMock(
            return_value=[
                Response(
                    agent_id="test_agent",
                    content="MCP fallback response",
                    metadata={"service": "mcp"},
                    confidence=0.8,
                    processing_time=0.2,
                    success=True,
                ),
            ],
        )

        router = HybridRouter(
            openrouter_client=mock_openrouter,
            mcp_client=mock_mcp,
            strategy=RoutingStrategy.OPENROUTER_PRIMARY,
        )

        await router.connect()

        workflow_steps = [
            WorkflowStep(
                step_id="circuit_test",
                agent_id="test_agent",
                input_data={"query": "Circuit breaker test"},
                timeout_seconds=30.0,
            ),
        ]

        # Phase 1: Normal operation
        responses = await router.orchestrate_agents(workflow_steps)
        assert "OpenRouter" in responses[0].content
        assert router.metrics.fallback_uses == 0

        # Phase 2: OpenRouter fails, trigger circuit breaker
        openrouter_failures = 3
        circuit_breaker_state["is_open"] = True

        responses = await router.orchestrate_agents(workflow_steps)
        assert "MCP" in responses[0].content
        assert router.metrics.fallback_uses == 1

        # Phase 3: Circuit breaker recovery
        circuit_breaker_state["is_open"] = False
        openrouter_failures = 0

        responses = await router.orchestrate_agents(workflow_steps)
        assert "OpenRouter" in responses[0].content

    @pytest.mark.asyncio
    @patch("src.mcp_integration.hybrid_router.get_settings")
    async def test_dynamic_configuration_changes(self, mock_get_settings, mock_settings):
        """Test dynamic configuration changes during operation."""
        mock_settings.openrouter_traffic_percentage = 0  # Start with MCP only
        mock_settings.circuit_breaker_enabled = False
        mock_get_settings.return_value = mock_settings

        # Track service usage
        openrouter_usage = []
        mcp_usage = []

        mock_openrouter = MagicMock()
        mock_openrouter.connect = AsyncMock(return_value=True)
        mock_openrouter.connection_state = MCPConnectionState.CONNECTED
        mock_openrouter.orchestrate_agents = AsyncMock(
            side_effect=lambda steps: openrouter_usage.append(len(steps))
            or [
                Response(
                    agent_id=step.agent_id,
                    content="OpenRouter response",
                    metadata={"service": "openrouter"},
                    confidence=0.9,
                    processing_time=0.1,
                    success=True,
                )
                for step in steps
            ],
        )

        mock_mcp = MagicMock()
        mock_mcp.connect = AsyncMock(return_value=True)
        mock_mcp.connection_state = MCPConnectionState.CONNECTED
        mock_mcp.orchestrate_agents = AsyncMock(
            side_effect=lambda steps: mcp_usage.append(len(steps))
            or [
                Response(
                    agent_id=step.agent_id,
                    content="MCP response",
                    metadata={"service": "mcp"},
                    confidence=0.8,
                    processing_time=0.2,
                    success=True,
                )
                for step in steps
            ],
        )

        router = HybridRouter(
            openrouter_client=mock_openrouter,
            mcp_client=mock_mcp,
            strategy=RoutingStrategy.OPENROUTER_PRIMARY,
        )

        await router.connect()

        workflow_steps = [
            WorkflowStep(
                step_id="config_test",
                agent_id="test_agent",
                input_data={"query": "Configuration test"},
                timeout_seconds=30.0,
            ),
        ]

        # Phase 1: 0% OpenRouter traffic (should use OpenRouter due to primary strategy)
        for _ in range(5):
            await router.orchestrate_agents(workflow_steps)

        phase1_openrouter = len(openrouter_usage)
        phase1_mcp = len(mcp_usage)

        # Phase 2: Increase to 100% OpenRouter traffic
        router.set_traffic_percentage(100)

        for _ in range(5):
            await router.orchestrate_agents(workflow_steps)

        phase2_openrouter = len(openrouter_usage) - phase1_openrouter
        phase2_mcp = len(mcp_usage) - phase1_mcp

        # Verify configuration change took effect
        assert router.openrouter_traffic_percentage == 100

        # Phase 1 should use OpenRouter due to primary strategy
        assert phase1_openrouter == 5
        assert phase1_mcp == 0

        # Phase 2 should definitely use OpenRouter (100% traffic)
        assert phase2_openrouter == 5
        assert phase2_mcp == 0

    @pytest.mark.asyncio
    @patch("src.mcp_integration.hybrid_router.get_settings")
    async def test_performance_under_concurrent_load(self, mock_get_settings, mock_settings):
        """Test HybridRouter performance under concurrent load."""
        mock_settings.openrouter_traffic_percentage = 50
        mock_settings.circuit_breaker_enabled = False
        mock_get_settings.return_value = mock_settings

        # Create mock clients with realistic delays
        async def mock_openrouter_orchestrate(steps):
            await asyncio.sleep(0.01)  # Simulate 10ms latency
            return [
                Response(
                    agent_id=step.agent_id,
                    content="OpenRouter response",
                    metadata={"service": "openrouter"},
                    confidence=0.9,
                    processing_time=0.01,
                    success=True,
                )
                for step in steps
            ]

        async def mock_mcp_orchestrate(steps):
            await asyncio.sleep(0.02)  # Simulate 20ms latency
            return [
                Response(
                    agent_id=step.agent_id,
                    content="MCP response",
                    metadata={"service": "mcp"},
                    confidence=0.8,
                    processing_time=0.02,
                    success=True,
                )
                for step in steps
            ]

        mock_openrouter = MagicMock()
        mock_openrouter.connect = AsyncMock(return_value=True)
        mock_openrouter.connection_state = MCPConnectionState.CONNECTED
        mock_openrouter.orchestrate_agents = AsyncMock(side_effect=mock_openrouter_orchestrate)

        mock_mcp = MagicMock()
        mock_mcp.connect = AsyncMock(return_value=True)
        mock_mcp.connection_state = MCPConnectionState.CONNECTED
        mock_mcp.orchestrate_agents = AsyncMock(side_effect=mock_mcp_orchestrate)

        router = HybridRouter(
            openrouter_client=mock_openrouter,
            mcp_client=mock_mcp,
            strategy=RoutingStrategy.ROUND_ROBIN,
        )

        await router.connect()

        # Create concurrent orchestration tasks
        async def orchestrate_task(task_id):
            workflow_steps = [
                WorkflowStep(
                    step_id=f"concurrent_step_{task_id}",
                    agent_id="test_agent",
                    input_data={"query": f"Concurrent query {task_id}"},
                    timeout_seconds=30.0,
                ),
            ]
            return await router.orchestrate_agents(workflow_steps)

        # Run concurrent orchestrations
        start_time = time.time()
        num_concurrent = 20

        tasks = [orchestrate_task(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all requests completed successfully
        assert len(results) == num_concurrent
        for result in results:
            assert len(result) == 1
            assert result[0].success is True

        # Verify performance metrics
        assert router.metrics.total_requests == num_concurrent
        assert router.metrics.successful_routes == num_concurrent
        assert router.metrics.failed_routes == 0

        # Should complete in reasonable time (less than sequential execution)
        max_expected_time = num_concurrent * 0.02  # If all were sequential at max latency
        assert total_time < max_expected_time

        # Verify average response time is reasonable
        assert 0 < router.metrics.average_response_time < 0.1  # Less than 100ms average

    @pytest.mark.asyncio
    @patch("src.mcp_integration.hybrid_router.get_settings")
    async def test_service_failure_and_recovery_scenarios(self, mock_get_settings, mock_settings):
        """Test various service failure and recovery scenarios."""
        mock_settings.openrouter_traffic_percentage = 100  # Prefer OpenRouter
        mock_settings.circuit_breaker_enabled = False
        mock_get_settings.return_value = mock_settings

        # Create controllable mock clients
        openrouter_state = {"should_fail": False, "calls": 0}
        mcp_state = {"should_fail": False, "calls": 0}

        async def mock_openrouter_orchestrate(steps):
            openrouter_state["calls"] += 1
            if openrouter_state["should_fail"]:
                raise Exception("OpenRouter service temporarily unavailable")
            return [
                Response(
                    agent_id=step.agent_id,
                    content="OpenRouter response",
                    metadata={"service": "openrouter"},
                    confidence=0.9,
                    processing_time=0.1,
                    success=True,
                )
                for step in steps
            ]

        async def mock_mcp_orchestrate(steps):
            mcp_state["calls"] += 1
            if mcp_state["should_fail"]:
                raise Exception("MCP service temporarily unavailable")
            return [
                Response(
                    agent_id=step.agent_id,
                    content="MCP response",
                    metadata={"service": "mcp"},
                    confidence=0.8,
                    processing_time=0.2,
                    success=True,
                )
                for step in steps
            ]

        mock_openrouter = MagicMock()
        mock_openrouter.connect = AsyncMock(return_value=True)
        mock_openrouter.connection_state = MCPConnectionState.CONNECTED
        mock_openrouter.orchestrate_agents = AsyncMock(side_effect=mock_openrouter_orchestrate)

        mock_mcp = MagicMock()
        mock_mcp.connect = AsyncMock(return_value=True)
        mock_mcp.connection_state = MCPConnectionState.CONNECTED
        mock_mcp.orchestrate_agents = AsyncMock(side_effect=mock_mcp_orchestrate)

        router = HybridRouter(
            openrouter_client=mock_openrouter,
            mcp_client=mock_mcp,
            strategy=RoutingStrategy.OPENROUTER_PRIMARY,
        )

        await router.connect()

        workflow_steps = [
            WorkflowStep(
                step_id="failure_test",
                agent_id="test_agent",
                input_data={"query": "Failure recovery test"},
                timeout_seconds=30.0,
            ),
        ]

        # Scenario 1: Both services healthy
        responses = await router.orchestrate_agents(workflow_steps)
        assert "OpenRouter" in responses[0].content
        assert router.metrics.fallback_uses == 0

        # Scenario 2: OpenRouter fails, MCP fallback
        openrouter_state["should_fail"] = True
        responses = await router.orchestrate_agents(workflow_steps)
        assert "MCP" in responses[0].content
        assert router.metrics.fallback_uses == 1

        # Scenario 3: OpenRouter recovers
        openrouter_state["should_fail"] = False
        responses = await router.orchestrate_agents(workflow_steps)
        assert "OpenRouter" in responses[0].content

        # Scenario 4: Both services fail
        openrouter_state["should_fail"] = True
        mcp_state["should_fail"] = True

        with pytest.raises(Exception, match="Orchestration failed on all services"):
            await router.orchestrate_agents(workflow_steps)

        assert router.metrics.failed_routes == 1

        # Scenario 5: Both services recover
        openrouter_state["should_fail"] = False
        mcp_state["should_fail"] = False

        responses = await router.orchestrate_agents(workflow_steps)
        assert responses[0].success is True

        # Verify call distribution shows fallback behavior
        assert openrouter_state["calls"] >= 2  # Initial + recovery attempts
        assert mcp_state["calls"] >= 1  # Fallback usage
