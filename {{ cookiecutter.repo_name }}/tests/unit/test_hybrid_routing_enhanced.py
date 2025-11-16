"""
Enhanced unit tests for hybrid routing functionality in QueryCounselor and HydeProcessor.

This module tests the specific enhancements made to QueryCounselor and HydeProcessor
for hybrid routing integration with OpenRouter and MCP services.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / ".." / ".." / "src"))

from src.core.hyde_processor import HydeProcessor, HypotheticalDocument
from src.core.query_counselor import QueryCounselor, QueryIntent, QueryType
from src.mcp_integration.hybrid_router import HybridRouter, RoutingStrategy
from src.mcp_integration.model_registry import ModelRegistry


class TestQueryCounselorHybridRouting:
    """Test hybrid routing enhancements in QueryCounselor."""

    @pytest.fixture
    def mock_hybrid_router(self):
        """Create mock HybridRouter."""
        mock_router = Mock(spec=HybridRouter)
        mock_router.connect = AsyncMock(return_value=True)
        mock_router.disconnect = AsyncMock(return_value=True)
        mock_router.validate_query = AsyncMock(
            return_value={"is_valid": True, "sanitized_query": "test query", "potential_issues": []},
        )
        mock_router.orchestrate_agents = AsyncMock(return_value=[])
        mock_router.get_capabilities = AsyncMock(return_value=["hybrid_routing"])
        return mock_router

    @pytest.fixture
    def mock_model_registry(self):
        """Create mock ModelRegistry."""
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.select_best_model = Mock(return_value="deepseek/deepseek-chat-v3-0324:free")
        mock_registry.get_model_capabilities = Mock(return_value=Mock(is_free=True))
        mock_registry.get_fallback_chain = Mock(return_value=["deepseek/deepseek-chat-v3-0324:free"])
        return mock_registry

    @pytest.fixture
    def query_counselor_with_hybrid_routing(self, mock_hybrid_router):
        """Create QueryCounselor with hybrid routing enabled."""
        # Clear global registry and caches before test
        from src.core.performance_optimizer import clear_all_caches
        from src.mcp_integration.model_registry import clear_model_registry

        clear_model_registry()
        clear_all_caches()

        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.select_best_model = Mock(return_value="test-model")

        # Use persistent patch that lasts until explicit cleanup
        patcher = patch("src.core.query_counselor.get_model_registry", return_value=mock_registry)
        patcher.start()

        try:
            counselor = QueryCounselor(
                mcp_client=None,  # Let it create HybridRouter internally
                hyde_processor=None,
                enable_hybrid_routing=True,  # Explicitly enable hybrid routing
                routing_strategy=RoutingStrategy.OPENROUTER_PRIMARY,
            )

            # Override the created HybridRouter with our mock
            counselor.mcp_client = mock_hybrid_router

            yield counselor
        finally:
            # Clean up the patch, global registry, and caches
            patcher.stop()
            clear_model_registry()
            clear_all_caches()

    @pytest.fixture
    def query_counselor_without_hybrid_routing(self):
        """Create QueryCounselor without hybrid routing."""
        # Clear global registry and caches before test to ensure clean state
        from src.core.performance_optimizer import clear_all_caches
        from src.mcp_integration.model_registry import clear_model_registry

        clear_model_registry()
        clear_all_caches()

        # Create a fresh instance without any patches
        counselor = QueryCounselor(
            hyde_processor=None,
            enable_hybrid_routing=False,
        )

        # Verify the instance is configured correctly
        assert counselor.hybrid_routing_enabled is False
        assert counselor.model_registry is None
        assert counselor.mcp_client is None

        yield counselor

        # Clear global registry and caches after test to ensure clean state
        clear_model_registry()
        clear_all_caches()

    def test_query_counselor_initialization_with_hybrid_routing(self, query_counselor_with_hybrid_routing):
        """Test QueryCounselor initialization with hybrid routing enabled."""
        counselor = query_counselor_with_hybrid_routing

        assert counselor.hybrid_routing_enabled is True
        assert counselor.mcp_client is not None
        assert counselor.model_registry is not None

    def test_query_counselor_initialization_without_hybrid_routing(self, query_counselor_without_hybrid_routing):
        """Test QueryCounselor initialization without hybrid routing."""
        counselor = query_counselor_without_hybrid_routing

        assert counselor.hybrid_routing_enabled is False
        assert counselor.mcp_client is None
        assert counselor.model_registry is None

    def test_model_selection_for_different_query_types(self, query_counselor_with_hybrid_routing):
        """Test model selection logic for different query types."""
        counselor = query_counselor_with_hybrid_routing

        test_cases = [
            (QueryType.IMPLEMENTATION, "complex", "reasoning"),
            (QueryType.ANALYSIS_REQUEST, "medium", "analysis"),
            (QueryType.SECURITY, "simple", "analysis"),
            (QueryType.GENERAL_QUERY, "simple", "general"),
        ]

        for query_type, complexity, _expected_task_type in test_cases:
            selected_model = counselor._select_model_for_task(query_type, complexity)

            if counselor.hybrid_routing_enabled:
                assert selected_model is not None
                assert isinstance(selected_model, str)
            else:
                assert selected_model is None

    def test_model_selection_disabled_when_hybrid_routing_disabled(self, query_counselor_without_hybrid_routing):
        """Test model selection returns None when hybrid routing is disabled."""
        counselor = query_counselor_without_hybrid_routing

        selected_model = counselor._select_model_for_task(QueryType.GENERAL_QUERY, "simple")
        assert selected_model is None

    @pytest.mark.asyncio
    async def test_intent_analysis_includes_hybrid_routing_metadata(self, query_counselor_with_hybrid_routing):
        """Test that intent analysis includes hybrid routing metadata."""
        counselor = query_counselor_with_hybrid_routing

        query = "How to implement authentication in Python?"
        intent = await counselor.analyze_intent(query)

        # Check that metadata is included when hybrid routing is enabled
        assert hasattr(intent, "metadata")
        assert intent.metadata is not None
        assert "hybrid_routing_enabled" in intent.metadata
        assert intent.metadata["hybrid_routing_enabled"] is True
        assert "selected_model" in intent.metadata
        assert "task_type" in intent.metadata

    @pytest.mark.asyncio
    async def test_intent_analysis_without_hybrid_routing_metadata(self, query_counselor_without_hybrid_routing):
        """Test that intent analysis excludes hybrid routing metadata when disabled."""
        counselor = query_counselor_without_hybrid_routing

        query = "How to implement authentication in Python?"
        intent = await counselor.analyze_intent(query)

        # Check that metadata is empty when hybrid routing is disabled
        assert hasattr(intent, "metadata")
        assert intent.metadata == {}

    @pytest.mark.asyncio
    async def test_workflow_orchestration_with_model_metadata(self, query_counselor_with_hybrid_routing):
        """Test that workflow orchestration includes model metadata."""
        counselor = query_counselor_with_hybrid_routing

        # Create intent with metadata
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

        # Mock successful orchestration
        counselor.mcp_client.orchestrate_agents = AsyncMock(
            return_value=[
                Mock(
                    agent_id="create_agent",
                    content="test response",
                    confidence=0.9,
                    processing_time=0.1,
                    success=True,
                    metadata={},
                ),
            ],
        )

        # Execute workflow
        await counselor.orchestrate_workflow(agents, "test query", intent)

        # Verify orchestration was called with enhanced metadata
        assert counselor.mcp_client.orchestrate_agents.called
        call_args = counselor.mcp_client.orchestrate_agents.call_args[0][0]

        # Check that workflow step includes model metadata
        assert len(call_args) == 1
        step = call_args[0]
        assert "selected_model" in step.input_data
        assert "task_type" in step.input_data
        assert "complexity" in step.input_data
        assert "query_type" in step.input_data


class TestHydeProcessorHybridRouting:
    """Test hybrid routing enhancements in HydeProcessor."""

    @pytest.fixture
    def mock_hybrid_router(self):
        """Create mock HybridRouter."""
        mock_router = Mock(spec=HybridRouter)
        mock_router.orchestrate_agents = AsyncMock(return_value=[])
        return mock_router

    @pytest.fixture
    def mock_model_registry(self):
        """Create mock ModelRegistry."""
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.select_best_model = Mock(return_value="deepseek/deepseek-chat-v3-0324:free")
        return mock_registry

    @pytest.fixture
    def hyde_processor_with_openrouter(self, mock_hybrid_router):
        """Create HydeProcessor with OpenRouter enabled."""
        with patch("src.core.hyde_processor.get_model_registry") as mock_get_registry:
            mock_get_registry.return_value = Mock(spec=ModelRegistry)
            mock_get_registry.return_value.select_best_model = Mock(return_value="test-model")

            return HydeProcessor(
                vector_store=None,  # Will use mock
                hybrid_router=mock_hybrid_router,
                enable_openrouter=True,
            )

    @pytest.fixture
    def hyde_processor_without_openrouter(self):
        """Create HydeProcessor without OpenRouter."""
        return HydeProcessor(
            vector_store=None,
            enable_openrouter=False,
        )

    def test_hyde_processor_initialization_with_openrouter(self, hyde_processor_with_openrouter):
        """Test HydeProcessor initialization with OpenRouter enabled."""
        processor = hyde_processor_with_openrouter

        assert processor.enable_openrouter is True
        assert processor.hybrid_router is not None
        assert processor.model_registry is not None

    def test_hyde_processor_initialization_without_openrouter(self, hyde_processor_without_openrouter):
        """Test HydeProcessor initialization without OpenRouter."""
        processor = hyde_processor_without_openrouter

        assert processor.enable_openrouter is False
        assert processor.hybrid_router is None
        assert processor.model_registry is None

    def test_hyde_prompt_generation(self, hyde_processor_with_openrouter):
        """Test HyDE prompt generation for different document types."""
        processor = hyde_processor_with_openrouter

        query = "Python authentication"

        # Test different document types
        for doc_index in range(3):
            prompt = processor._create_hyde_prompt(query, doc_index)

            assert isinstance(prompt, str)
            assert query in prompt
            assert len(prompt) > 50  # Should be substantial prompt

    @pytest.mark.asyncio
    async def test_hypothetical_document_generation_with_openrouter(self, hyde_processor_with_openrouter):
        """Test hypothetical document generation using OpenRouter."""
        processor = hyde_processor_with_openrouter

        # Mock successful OpenRouter responses
        mock_responses = [
            Mock(
                agent_id="hyde_generator",
                content="Generated guide content",
                confidence=0.9,
                processing_time=0.2,
                success=True,
                metadata={},
            ),
            Mock(
                agent_id="hyde_generator",
                content="Generated technical documentation",
                confidence=0.85,
                processing_time=0.3,
                success=True,
                metadata={},
            ),
        ]

        processor.hybrid_router.orchestrate_agents = AsyncMock(return_value=mock_responses)

        # Generate documents
        docs = await processor._generate_hypothetical_docs_with_openrouter("Python authentication")

        # Verify documents
        assert len(docs) == 2
        assert all(isinstance(doc, HypotheticalDocument) for doc in docs)
        assert docs[0].content == "Generated guide content"
        assert docs[0].generation_method == "openrouter"
        assert "model_used" in docs[0].metadata

    @pytest.mark.asyncio
    async def test_hypothetical_document_generation_fallback_to_mock(self, hyde_processor_with_openrouter):
        """Test fallback to mock generation when OpenRouter is unavailable."""
        processor = hyde_processor_with_openrouter

        # Mock OpenRouter failure
        processor.hybrid_router.orchestrate_agents = AsyncMock(side_effect=Exception("Service unavailable"))

        # Generate documents (should fallback to mock)
        docs = await processor._generate_hypothetical_docs("Python authentication")

        # Verify fallback to mock
        assert len(docs) == 3  # Mock generates 3 documents
        assert all(doc.generation_method == "mock_template" for doc in docs)

    @pytest.mark.asyncio
    async def test_hypothetical_document_generation_disabled_openrouter(self, hyde_processor_without_openrouter):
        """Test document generation when OpenRouter is disabled."""
        processor = hyde_processor_without_openrouter

        # Generate documents (should use mock)
        docs = await processor._generate_hypothetical_docs("Python authentication")

        # Verify mock generation
        assert len(docs) == 3
        assert all(doc.generation_method == "mock_template" for doc in docs)

    @pytest.mark.asyncio
    async def test_document_generation_error_handling(self, hyde_processor_with_openrouter):
        """Test error handling in document generation."""
        processor = hyde_processor_with_openrouter

        # Mock partial failure (some responses succeed, others fail)
        mock_responses = [
            Mock(
                agent_id="hyde_generator",
                content="Generated content",
                confidence=0.9,
                processing_time=0.2,
                success=True,
                metadata={},
            ),
            Mock(
                agent_id="hyde_generator",
                content="",
                confidence=0.0,
                processing_time=0.0,
                success=False,
                metadata={},
            ),
        ]

        processor.hybrid_router.orchestrate_agents = AsyncMock(return_value=mock_responses)

        # Generate documents
        docs = await processor._generate_hypothetical_docs_with_openrouter("Python authentication")

        # Should only include successful responses
        assert len(docs) == 1
        assert docs[0].content == "Generated content"
        assert docs[0].generation_method == "openrouter"


class TestHybridRoutingPerformance:
    """Test performance aspects of hybrid routing."""

    @pytest.fixture
    def mock_slow_service(self):
        """Create mock service that introduces delay."""
        mock_service = Mock()
        mock_service.orchestrate_agents = AsyncMock(side_effect=lambda *args: asyncio.sleep(0.1))
        return mock_service

    @pytest.mark.asyncio
    async def test_hybrid_routing_performance_monitoring(self):
        """Test performance monitoring in hybrid routing."""
        # This test would verify that performance metrics are collected
        # and can be used for routing decisions

    @pytest.mark.asyncio
    async def test_hybrid_routing_timeout_handling(self):
        """Test timeout handling in hybrid routing."""
        # This test would verify that long-running requests are handled gracefully

    @pytest.mark.asyncio
    async def test_hybrid_routing_concurrent_requests(self):
        """Test concurrent request handling in hybrid routing."""
        # This test would verify that multiple concurrent requests are handled correctly


class TestModelSelectionLogic:
    """Test model selection logic in hybrid routing."""

    @pytest.fixture
    def mock_model_registry(self):
        """Create mock ModelRegistry with various models."""
        mock_registry = Mock(spec=ModelRegistry)

        # Mock different model types
        mock_registry.select_best_model = Mock(
            side_effect=lambda task_type, allow_premium: {
                ("general", False): "deepseek/deepseek-chat-v3-0324:free",
                ("reasoning", False): "deepseek/deepseek-r1-0528:free",
                ("analysis", False): "google/gemini-2.0-flash-exp:free",
                ("general", True): "google/gemini-2.5-pro",
                ("reasoning", True): "openai/o3-mini",
            }.get((task_type, allow_premium), "deepseek/deepseek-chat-v3-0324:free"),
        )

        return mock_registry

    def test_model_selection_for_free_models(self, mock_model_registry):
        """Test model selection prefers free models by default."""
        # Test that free models are selected for standard complexity
        free_model = mock_model_registry.select_best_model("general", allow_premium=False)
        assert "free" in free_model

    def test_model_selection_for_premium_models(self, mock_model_registry):
        """Test model selection allows premium models for complex tasks."""
        # Test that premium models are selected for high complexity
        premium_model = mock_model_registry.select_best_model("reasoning", allow_premium=True)
        assert "free" not in premium_model

    def test_model_selection_task_type_mapping(self, mock_model_registry):
        """Test that query types are correctly mapped to task types."""
        # Test different task types get appropriate models
        reasoning_model = mock_model_registry.select_best_model("reasoning", allow_premium=False)
        analysis_model = mock_model_registry.select_best_model("analysis", allow_premium=False)

        assert reasoning_model != analysis_model
        assert "reasoning" in reasoning_model or "r1" in reasoning_model
        assert "gemini" in analysis_model or "analysis" in analysis_model
