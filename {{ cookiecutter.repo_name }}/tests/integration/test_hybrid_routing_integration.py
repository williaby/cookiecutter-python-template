"""
Integration tests for HybridRouter integration with QueryCounselor and HydeProcessor.

This module tests the complete integration of the HybridRouter system with
QueryCounselor and HydeProcessor, demonstrating end-to-end hybrid routing
functionality with OpenRouter API fallback to MCP services.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / ".." / ".." / "src"))

from src.core.hyde_processor import HydeProcessor
from src.core.query_counselor import QueryCounselor, QueryType
from src.mcp_integration.hybrid_router import HybridRouter, RoutingStrategy
from src.mcp_integration.mcp_client import (
    MCPConnectionState,
    MCPHealthStatus,
    WorkflowStep,
)
from src.mcp_integration.mcp_client import (
    Response as MCPResponse,
)


class TestHybridRoutingIntegration:
    """Integration tests for hybrid routing functionality."""

    @pytest.fixture
    def mock_openrouter_client(self):
        """Create mock OpenRouter client."""
        mock_client = Mock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.disconnect = AsyncMock(return_value=True)
        mock_client.health_check = AsyncMock(
            return_value=MCPHealthStatus(
                connection_state=MCPConnectionState.CONNECTED,
                last_successful_request=1234567890.0,
                error_count=0,
                response_time_ms=100.0,
                capabilities=["text_generation", "reasoning", "analysis"],
                server_version="1.0.0",
                metadata={"service": "openrouter"},
            ),
        )
        mock_client.validate_query = AsyncMock(
            return_value={"is_valid": True, "sanitized_query": "test query", "potential_issues": []},
        )
        mock_client.orchestrate_agents = AsyncMock(
            return_value=[
                MCPResponse(
                    agent_id="test_agent",
                    content="OpenRouter response",
                    confidence=0.9,
                    processing_time=0.2,
                    success=True,
                    metadata={"service": "openrouter"},
                ),
            ],
        )
        mock_client.get_capabilities = AsyncMock(return_value=["text_generation", "reasoning", "analysis"])
        mock_client.connection_state = MCPConnectionState.CONNECTED
        return mock_client

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        mock_client = Mock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.disconnect = AsyncMock(return_value=True)
        mock_client.health_check = AsyncMock(
            return_value=MCPHealthStatus(
                connection_state=MCPConnectionState.CONNECTED,
                last_successful_request=1234567890.0,
                error_count=0,
                response_time_ms=150.0,
                capabilities=["agent_orchestration", "workflow_management"],
                server_version="1.0.0",
                metadata={"service": "mcp"},
            ),
        )
        mock_client.validate_query = AsyncMock(
            return_value={"is_valid": True, "sanitized_query": "test query", "potential_issues": []},
        )
        mock_client.orchestrate_agents = AsyncMock(
            return_value=[
                MCPResponse(
                    agent_id="test_agent",
                    content="MCP response",
                    confidence=0.8,
                    processing_time=0.3,
                    success=True,
                    metadata={"service": "mcp"},
                ),
            ],
        )
        mock_client.get_capabilities = AsyncMock(return_value=["agent_orchestration", "workflow_management"])
        mock_client.connection_state = MCPConnectionState.CONNECTED
        return mock_client

    @pytest.fixture
    def hybrid_router(self, mock_openrouter_client, mock_mcp_client):
        """Create HybridRouter with mocked clients."""
        return HybridRouter(
            openrouter_client=mock_openrouter_client,
            mcp_client=mock_mcp_client,
            strategy=RoutingStrategy.OPENROUTER_PRIMARY,
            enable_gradual_rollout=True,
        )

    @pytest.fixture
    def query_counselor_with_hybrid_routing(self, hybrid_router):
        """Create QueryCounselor with hybrid routing enabled."""
        return QueryCounselor(
            mcp_client=hybrid_router,
            hyde_processor=None,  # Will be created with defaults
            enable_hybrid_routing=True,
            routing_strategy=RoutingStrategy.OPENROUTER_PRIMARY,
        )

    @pytest.fixture
    def hyde_processor_with_hybrid_routing(self, hybrid_router):
        """Create HydeProcessor with hybrid routing enabled."""
        return HydeProcessor(
            vector_store=None,  # Will use mock
            hybrid_router=hybrid_router,
            enable_openrouter=True,
        )

    @pytest.mark.asyncio
    async def test_query_counselor_hybrid_routing_initialization(self, query_counselor_with_hybrid_routing):
        """Test QueryCounselor initialization with hybrid routing."""
        counselor = query_counselor_with_hybrid_routing

        # Verify hybrid routing is enabled
        assert counselor.hybrid_routing_enabled is True
        assert counselor.mcp_client is not None
        assert counselor.model_registry is not None
        assert isinstance(counselor.mcp_client, HybridRouter)

    @pytest.mark.asyncio
    async def test_query_counselor_model_selection(self, query_counselor_with_hybrid_routing):
        """Test model selection functionality in QueryCounselor."""
        counselor = query_counselor_with_hybrid_routing

        # Test different query types
        test_cases = [
            (QueryType.IMPLEMENTATION, "complex", "reasoning"),
            (QueryType.ANALYSIS_REQUEST, "medium", "analysis"),
            (QueryType.GENERAL_QUERY, "simple", "general"),
        ]

        for query_type, complexity, _expected_task_type in test_cases:
            selected_model = counselor._select_model_for_task(query_type, complexity)

            assert selected_model is not None
            assert isinstance(selected_model, str)
            # Model should be from our registry
            assert "free" in selected_model  # Should select free models by default

    @pytest.mark.asyncio
    async def test_query_counselor_intent_analysis_with_hybrid_routing(self, query_counselor_with_hybrid_routing):
        """Test intent analysis with hybrid routing metadata."""
        counselor = query_counselor_with_hybrid_routing

        query = "How to implement secure authentication in Python web applications?"
        intent = await counselor.analyze_intent(query)

        # Verify intent includes hybrid routing metadata
        assert hasattr(intent, "metadata")
        assert intent.metadata is not None
        assert "hybrid_routing_enabled" in intent.metadata
        assert intent.metadata["hybrid_routing_enabled"] is True
        assert "selected_model" in intent.metadata
        assert "task_type" in intent.metadata

    @pytest.mark.asyncio
    async def test_query_counselor_workflow_orchestration_with_model_selection(
        self,
        query_counselor_with_hybrid_routing,
    ):
        """Test workflow orchestration passes model selection to HybridRouter."""
        counselor = query_counselor_with_hybrid_routing

        # Create test intent with metadata
        from src.core.query_counselor import QueryIntent

        intent = QueryIntent(
            query_type=QueryType.IMPLEMENTATION,
            confidence=0.8,
            complexity="complex",
            requires_agents=["create_agent"],
            hyde_recommended=False,
            original_query="test query",
            metadata={
                "selected_model": "deepseek/deepseek-chat-v3-0324:free",
                "task_type": "reasoning",
                "hybrid_routing_enabled": True,
                "allow_premium": True,
            },
        )

        # Create test agents
        from src.core.query_counselor import Agent

        agents = [
            Agent(
                agent_id="create_agent",
                agent_type="create",
                capabilities=["implementation"],
            ),
        ]

        # Execute workflow
        responses = await counselor.orchestrate_workflow(agents, "test query", intent)

        # Verify response
        assert len(responses) == 1
        assert responses[0].content == "OpenRouter response"  # Should use OpenRouter primary
        assert responses[0].success is True

    @pytest.mark.asyncio
    async def test_hyde_processor_hybrid_routing_initialization(self, hyde_processor_with_hybrid_routing):
        """Test HydeProcessor initialization with hybrid routing."""
        processor = hyde_processor_with_hybrid_routing

        # Verify hybrid routing is enabled
        assert processor.enable_openrouter is True
        assert processor.hybrid_router is not None
        assert processor.model_registry is not None
        assert isinstance(processor.hybrid_router, HybridRouter)

    @pytest.mark.asyncio
    async def test_hyde_processor_openrouter_document_generation(self, hyde_processor_with_hybrid_routing):
        """Test hypothetical document generation using OpenRouter."""
        processor = hyde_processor_with_hybrid_routing

        # Mock the OpenRouter response for document generation
        mock_responses = [
            MCPResponse(
                agent_id="hyde_generator",
                content="Generated comprehensive guide about Python authentication",
                confidence=0.9,
                processing_time=0.5,
                success=True,
                metadata={"model": "deepseek/deepseek-chat-v3-0324:free"},
            ),
            MCPResponse(
                agent_id="hyde_generator",
                content="Technical documentation with code examples",
                confidence=0.85,
                processing_time=0.4,
                success=True,
                metadata={"model": "deepseek/deepseek-chat-v3-0324:free"},
            ),
        ]

        processor.hybrid_router.orchestrate_agents = AsyncMock(return_value=mock_responses)

        # Generate hypothetical documents
        docs = await processor._generate_hypothetical_docs_with_openrouter("Python authentication")

        # Verify documents were generated
        assert len(docs) == 2
        assert docs[0].content == "Generated comprehensive guide about Python authentication"
        assert docs[0].generation_method == "openrouter"
        assert docs[0].metadata["model_used"] is not None

    @pytest.mark.asyncio
    async def test_hyde_processor_fallback_to_mock_on_openrouter_failure(self, hyde_processor_with_hybrid_routing):
        """Test fallback to mock generation when OpenRouter fails."""
        processor = hyde_processor_with_hybrid_routing

        # Mock OpenRouter failure
        processor.hybrid_router.orchestrate_agents = AsyncMock(side_effect=Exception("OpenRouter failed"))

        # Generate hypothetical documents (should fallback to mock)
        docs = await processor._generate_hypothetical_docs("Python authentication")

        # Verify fallback to mock generation
        assert len(docs) == 3  # Mock generates 3 documents
        assert all(doc.generation_method == "mock_template" for doc in docs)

    @pytest.mark.asyncio
    async def test_hybrid_router_routing_decision_logic(self, hybrid_router):
        """Test HybridRouter routing decision logic."""
        router = hybrid_router

        # Test routing decisions
        request_id = "test_request_123"

        # Test with OpenRouter primary strategy
        decision = router._make_routing_decision(request_id, "orchestration")

        assert decision.service in ["openrouter", "mcp"]
        assert decision.request_id == request_id
        assert decision.confidence > 0.0
        assert isinstance(decision.fallback_available, bool)
        assert decision.reason is not None

    @pytest.mark.asyncio
    async def test_hybrid_router_fallback_behavior(self, hybrid_router):
        """Test HybridRouter fallback behavior when primary service fails."""
        router = hybrid_router

        # Mock OpenRouter failure
        router.openrouter_client.orchestrate_agents = AsyncMock(side_effect=Exception("OpenRouter service unavailable"))

        # Create test workflow
        workflow_steps = [
            WorkflowStep(
                step_id="test_step",
                agent_id="test_agent",
                input_data={"query": "test query"},
            ),
        ]

        # Execute workflow (should fallback to MCP)
        responses = await router.orchestrate_agents(workflow_steps)

        # Verify fallback to MCP
        assert len(responses) == 1
        assert responses[0].content == "MCP response"
        assert router.metrics.fallback_uses == 1

    @pytest.mark.asyncio
    async def test_end_to_end_hybrid_routing_integration(self, query_counselor_with_hybrid_routing):
        """Test end-to-end hybrid routing integration."""
        counselor = query_counselor_with_hybrid_routing

        # Process a complex query that should use hybrid routing
        query = "How to implement secure JWT authentication with refresh tokens in a Python FastAPI application?"

        response = await counselor.process_query(query)

        # Verify response
        assert response.success is True
        assert response.confidence > 0.0
        assert response.processing_time > 0.0
        assert response.response is not None
        assert len(response.agents_used) > 0

    @pytest.mark.asyncio
    async def test_hybrid_routing_performance_monitoring(self, hybrid_router):
        """Test performance monitoring in hybrid routing."""
        router = hybrid_router

        # Execute multiple requests to test metrics
        workflow_steps = [
            WorkflowStep(
                step_id="perf_test",
                agent_id="test_agent",
                input_data={"query": "test query"},
            ),
        ]

        # Execute multiple workflows
        for _i in range(5):
            await router.orchestrate_agents(workflow_steps)

        # Verify metrics collection
        metrics = router.get_routing_metrics()
        assert metrics["total_requests"] == 5
        assert metrics["successful_routes"] == 5
        assert metrics["average_response_time"] > 0.0
        assert metrics["success_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_hybrid_routing_with_gradual_rollout(self, mock_openrouter_client, mock_mcp_client):
        """Test gradual rollout functionality."""
        # Create router with 50% OpenRouter traffic
        router = HybridRouter(
            openrouter_client=mock_openrouter_client,
            mcp_client=mock_mcp_client,
            strategy=RoutingStrategy.OPENROUTER_PRIMARY,
            enable_gradual_rollout=True,
        )

        # Set traffic percentage
        router.set_traffic_percentage(50)

        # Test routing decisions for multiple requests
        openrouter_count = 0
        mcp_count = 0

        for i in range(100):
            decision = router._make_routing_decision(f"request_{i}", "orchestration")
            if decision.service == "openrouter":
                openrouter_count += 1
            else:
                mcp_count += 1

        # Should be approximately 50/50 split (allowing for some variance)
        assert 30 <= openrouter_count <= 70
        assert 30 <= mcp_count <= 70

    @pytest.mark.asyncio
    async def test_model_registry_integration(self, query_counselor_with_hybrid_routing):
        """Test ModelRegistry integration with hybrid routing."""
        counselor = query_counselor_with_hybrid_routing

        # Test model selection for different scenarios
        model_registry = counselor.model_registry

        # Test free model selection
        free_model = model_registry.select_best_model("general", allow_premium=False)
        assert "free" in free_model

        # Test model capabilities lookup
        capabilities = model_registry.get_model_capabilities(free_model)
        assert capabilities is not None
        assert capabilities.is_free is True

        # Test fallback chains
        fallback_chain = model_registry.get_fallback_chain("free_general")
        assert len(fallback_chain) > 0
        assert all("free" in model for model in fallback_chain)
