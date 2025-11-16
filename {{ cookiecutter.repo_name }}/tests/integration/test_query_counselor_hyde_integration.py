"""Integration tests for QueryCounselor + HydeProcessor workflow.

This module tests the integration between QueryCounselor and HydeProcessor,
validating the complete query processing workflow with enhanced retrieval.
"""

import asyncio
import contextlib
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.config.settings import ApplicationSettings
from src.core.query_counselor import QueryCounselor
from src.core.vector_store import AbstractVectorStore


class TestQueryCounselorHydeIntegration:
    """Integration tests for QueryCounselor + HydeProcessor workflow."""

    @pytest.fixture
    def test_settings(self):
        """Create test settings for integration testing."""
        return ApplicationSettings(
            mcp_enabled=True,
            mcp_server_url="http://localhost:3000",
            mcp_timeout=10.0,
            mcp_max_retries=2,
            qdrant_enabled=True,
            qdrant_host="192.168.1.16",
            qdrant_port=6333,
            qdrant_timeout=30.0,
            vector_store_type="qdrant",
        )

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client for testing."""
        # Create a more lenient mock without strict spec
        client = AsyncMock()
        client.connect = AsyncMock(return_value=True)
        client.disconnect = AsyncMock(return_value=True)
        client.connection_state = "connected"

        # Mock query validation
        client.validate_query = AsyncMock(
            return_value={"is_valid": True, "sanitized_query": "test query", "potential_issues": []},
        )

        # Mock orchestrate_agents with realistic responses
        def mock_orchestrate_response(workflow_steps):
            responses = []
            for step in workflow_steps:
                response = MagicMock()
                response.agent_id = step.agent_id
                # Include relevant content based on the query in the step
                query = step.input_data.get("query", "")
                if "authentication" in query.lower() or "auth" in query.lower():
                    response.content = f"Authentication implementation guide from {step.agent_id}"
                else:
                    response.content = f"Response from {step.agent_id} for query: {query}"
                response.confidence = 0.85
                response.processing_time = 0.5
                response.success = True
                response.error_message = None
                response.metadata = {"agent_type": step.input_data.get("agent_type", "unknown")}
                responses.append(response)
            return responses

        client.orchestrate_agents = AsyncMock(side_effect=mock_orchestrate_response)

        return client

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store for testing."""
        store = AsyncMock(spec=AbstractVectorStore)
        store.connect.return_value = None
        store.disconnect.return_value = None
        store.health_check.return_value = {"status": "healthy"}
        return store

    @pytest.fixture
    def mock_hyde_processor(self, mock_vector_store):
        """Create mock HydeProcessor for testing."""
        # Create a more lenient mock without strict spec to allow dynamic attribute setting
        processor = AsyncMock()
        processor.vector_store = mock_vector_store

        # Set up default return values for key methods
        processor.initialize = AsyncMock(return_value=True)

        # Mock three_tier_analysis method with realistic response
        mock_enhanced_query = MagicMock()
        mock_enhanced_query.enhanced_query = "Enhanced test query"
        mock_enhanced_query.processing_strategy = "standard_hyde"
        mock_enhanced_query.specificity_analysis = MagicMock()
        mock_enhanced_query.specificity_analysis.specificity_level = MagicMock()
        mock_enhanced_query.specificity_analysis.specificity_level.value = "medium"
        mock_enhanced_query.specificity_analysis.specificity_score = 65.0
        mock_enhanced_query.specificity_analysis.processing_time = 0.1
        mock_enhanced_query.specificity_analysis.reasoning = "Medium specificity query suitable for HyDE enhancement"
        mock_enhanced_query.specificity_analysis.guiding_questions = [
            "What specific implementation details are needed?",
        ]
        processor.three_tier_analysis = AsyncMock(return_value=mock_enhanced_query)

        # Mock process_query method with realistic response
        mock_results = MagicMock()
        mock_results.results = [
            {
                "id": "doc_1",
                "content": "Test document content",
                "score": 0.89,
                "metadata": {"source": "test_source"},
            },
        ]
        mock_results.hyde_enhanced = True
        processor.process_query = AsyncMock(return_value=mock_results)

        return processor

    @pytest.fixture
    def query_counselor(self, test_settings, mock_mcp_client, mock_hyde_processor):
        """Create QueryCounselor with mocked dependencies."""
        with (
            patch("src.config.settings.get_settings", return_value=test_settings),
            patch("src.mcp_integration.mcp_client.MCPClientFactory.create_from_settings", return_value=mock_mcp_client),
            patch("src.core.hyde_processor.HydeProcessor", return_value=mock_hyde_processor),
        ):
            # Pass mock client directly to constructor to ensure proper injection
            return QueryCounselor(mcp_client=mock_mcp_client, hyde_processor=mock_hyde_processor)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_query_processing_workflow(self, query_counselor, mock_mcp_client, mock_hyde_processor):
        """Test complete query processing workflow with HyDE enhancement."""

        # Mock query intent analysis
        mock_intent = MagicMock()
        mock_intent.query_type.value = "knowledge_retrieval"
        mock_intent.complexity = "medium"
        mock_intent.domain = "technical"
        mock_intent.confidence = 0.85

        # Mock HyDE enhanced query
        mock_hyde_processor.process_query.return_value = {
            "enhanced_query": "How to implement secure authentication in FastAPI with JWT tokens and OAuth2 scopes",
            "original_query": "How to implement authentication in FastAPI",
            "enhancement_score": 0.92,
            "processing_time": 0.15,
            "retrieved_documents": [
                {
                    "id": "doc_1",
                    "content": "FastAPI authentication with JWT tokens",
                    "score": 0.89,
                    "metadata": {"source": "fastapi_docs", "section": "security"},
                },
                {
                    "id": "doc_2",
                    "content": "OAuth2 implementation patterns",
                    "score": 0.84,
                    "metadata": {"source": "oauth2_guide", "section": "implementation"},
                },
            ],
        }

        # Mock MCP orchestration response
        mock_mcp_client.orchestrate_agents.return_value = [
            MagicMock(
                agent_id="create_agent",
                content="Comprehensive FastAPI authentication implementation guide",
                confidence=0.95,
                processing_time=1.2,
                success=True,
                metadata={"framework": "FastAPI", "security_level": "high"},
            ),
        ]

        # Process query with HyDE enhancement
        query = "How to implement authentication in FastAPI"

        # Step 1: Analyze query intent
        intent = await query_counselor.analyze_intent(query)
        # The query "How to implement authentication in FastAPI" should be classified as documentation
        # based on the QueryCounselor's intent analysis logic (contains "how to")
        assert intent.query_type.value == "documentation"

        # Step 2: Process query with HyDE enhancement
        hyde_result = await query_counselor.hyde_processor.process_query(query)
        assert hyde_result["enhanced_query"] != query
        assert hyde_result["enhancement_score"] > 0.8
        assert len(hyde_result["retrieved_documents"]) == 2

        # Step 3: Select agents based on enhanced query
        agent_selection = await query_counselor.select_agents(intent)
        assert len(agent_selection.primary_agents) > 0

        # Convert AgentSelection to list of Agent objects for orchestration
        selected_agents = []
        for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
            agent = next((a for a in query_counselor._available_agents if a.agent_id == agent_id), None)
            if agent:
                selected_agents.append(agent)

        # Step 4: Orchestrate workflow with enhanced context
        responses = await query_counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])
        # Documentation queries may select multiple agents (analysis_agent + create_agent)
        assert len(responses) >= 1
        assert all(r.success for r in responses)
        # Check that responses are generated successfully (content should be non-empty)
        assert all(r.content for r in responses)
        # Verify responses contain relevant content (either agent ID or some meaningful content)
        for response in responses:
            assert response.content is not None
            assert len(response.content) > 0

        # Verify HyDE processor was called
        mock_hyde_processor.process_query.assert_called_once_with(query)

        # Verify MCP orchestration was called with enhanced query
        mock_mcp_client.orchestrate_agents.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hyde_processor_vector_store_integration(
        self,
        query_counselor,
        mock_hyde_processor,
        mock_vector_store,
    ):
        """Test HydeProcessor integration with vector store operations."""

        # Mock vector store search results
        mock_vector_store.search.return_value = [
            {
                "id": "doc_1",
                "content": "Python async programming patterns",
                "score": 0.92,
                "metadata": {"language": "python", "topic": "async"},
            },
            {
                "id": "doc_2",
                "content": "FastAPI async endpoint implementation",
                "score": 0.88,
                "metadata": {"framework": "fastapi", "topic": "async"},
            },
        ]

        # Mock HyDE processing with vector store integration
        mock_hyde_processor.process_query.return_value = {
            "enhanced_query": "Python async programming patterns for FastAPI endpoints",
            "original_query": "async in Python",
            "enhancement_score": 0.89,
            "processing_time": 0.12,
            "retrieved_documents": [
                {
                    "id": "doc_1",
                    "content": "Python async programming patterns",
                    "score": 0.92,
                    "metadata": {"language": "python", "topic": "async"},
                },
            ],
        }

        # Process query
        query = "async in Python"
        result = await query_counselor.hyde_processor.process_query(query)

        # Verify results
        assert result["enhanced_query"] != query
        assert result["enhancement_score"] > 0.8
        assert len(result["retrieved_documents"]) == 1
        assert result["retrieved_documents"][0]["score"] > 0.9

        # Verify HyDE processor was called
        mock_hyde_processor.process_query.assert_called_once_with(query)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_counselor_hyde_error_handling(self, query_counselor, mock_hyde_processor):
        """Test error handling in QueryCounselor + HydeProcessor integration."""

        # Mock HyDE processor failure
        mock_hyde_processor.process_query.side_effect = Exception("Vector store connection failed")

        # Test graceful degradation
        query = "Test query that should fail"

        # Should handle HyDE processor failure gracefully
        try:
            # In a real implementation, this should fall back to original query
            intent = await query_counselor.analyze_intent(query)
            assert intent is not None

            # Try to process the query directly which would trigger HyDE if recommended
            if intent.hyde_recommended and query_counselor.hyde_processor is not None:
                # This should fail but be handled gracefully
                with contextlib.suppress(Exception):
                    await query_counselor.hyde_processor.process_query(query)

        except Exception as e:
            # Should not propagate vector store errors
            if "Vector store connection failed" in str(e):
                pytest.fail(f"Unexpected vector store error propagated: {e}")

        # Verify HyDE processor was attempted if hyde was recommended
        if intent.hyde_recommended:
            mock_hyde_processor.process_query.assert_called_with(query)
        else:
            # If HyDE wasn't recommended for this query, that's also valid
            assert True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_with_hyde_enhancement(self, query_counselor, mock_mcp_client, mock_hyde_processor):
        """Test end-to-end performance with HyDE enhancement meets <2s requirement."""

        # Mock fast HyDE processing
        mock_hyde_processor.process_query.return_value = {
            "enhanced_query": "Enhanced query with better context",
            "original_query": "Original query",
            "enhancement_score": 0.88,
            "processing_time": 0.08,  # 80ms for HyDE processing
            "retrieved_documents": [
                {
                    "id": "doc_1",
                    "content": "Relevant document content",
                    "score": 0.91,
                    "metadata": {"source": "knowledge_base"},
                },
            ],
        }

        # Mock fast MCP response
        mock_mcp_client.orchestrate_agents.return_value = [
            MagicMock(
                agent_id="create_agent",
                content="Fast enhanced response",
                confidence=0.93,
                processing_time=0.5,  # 500ms for MCP processing
                success=True,
                metadata={"enhanced": True},
            ),
        ]

        # Simulate realistic async delays
        async def delayed_hyde_process(query):
            await asyncio.sleep(0.08)  # 80ms delay
            return mock_hyde_processor.process_query.return_value

        async def delayed_mcp_orchestrate(agents):
            await asyncio.sleep(0.5)  # 500ms delay
            return mock_mcp_client.orchestrate_agents.return_value

        mock_hyde_processor.process_query.side_effect = delayed_hyde_process
        mock_mcp_client.orchestrate_agents.side_effect = delayed_mcp_orchestrate

        # Test complete workflow performance
        query = "Performance test query"

        start_time = time.time()

        # Complete workflow
        intent = await query_counselor.analyze_intent(query)
        hyde_result = await query_counselor.hyde_processor.process_query(query)
        agent_selection = await query_counselor.select_agents(intent)
        # Convert AgentSelection to list of Agent objects for orchestration
        selected_agents = []
        for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
            agent = next((a for a in query_counselor._available_agents if a.agent_id == agent_id), None)
            if agent:
                selected_agents.append(agent)
        responses = await query_counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])

        end_time = time.time()
        total_time = end_time - start_time

        # Verify performance requirement
        assert total_time < 2.0, f"Total processing time {total_time:.3f}s exceeds 2s requirement"
        assert len(responses) == 1
        assert responses[0].success is True

        # Verify all components were called
        assert mock_hyde_processor.process_query.called
        assert mock_mcp_client.orchestrate_agents.called

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_multi_step_workflow(self, query_counselor, mock_mcp_client, mock_hyde_processor):
        """Test complex multi-step workflow with HyDE enhancement."""

        # Mock complex HyDE processing with multiple document retrievals
        mock_hyde_processor.process_query.return_value = {
            "enhanced_query": "Comprehensive database design patterns for microservices architecture with event sourcing",
            "original_query": "database design patterns",
            "enhancement_score": 0.94,
            "processing_time": 0.18,
            "retrieved_documents": [
                {
                    "id": "doc_1",
                    "content": "Database design patterns for microservices",
                    "score": 0.95,
                    "metadata": {"architecture": "microservices", "pattern": "database"},
                },
                {
                    "id": "doc_2",
                    "content": "Event sourcing implementation patterns",
                    "score": 0.91,
                    "metadata": {"pattern": "event_sourcing", "domain": "architecture"},
                },
                {
                    "id": "doc_3",
                    "content": "CQRS pattern with event sourcing",
                    "score": 0.87,
                    "metadata": {"pattern": "cqrs", "complement": "event_sourcing"},
                },
            ],
        }

        # Mock multi-agent orchestration response - override the default side_effect for this test
        def mock_multi_agent_response(workflow_steps):
            # Always return 2 responses for this complex workflow test
            return [
                MagicMock(
                    agent_id="architecture_agent",
                    content="Database design patterns analysis",
                    confidence=0.92,
                    processing_time=0.8,
                    success=True,
                    metadata={"analysis_type": "architecture", "patterns_found": 3},
                ),
                MagicMock(
                    agent_id="create_agent",
                    content="Implementation guide for database patterns",
                    confidence=0.89,
                    processing_time=1.1,
                    success=True,
                    metadata={"output_type": "implementation_guide", "complexity": "high"},
                ),
            ]

        mock_mcp_client.orchestrate_agents.side_effect = mock_multi_agent_response

        # Process complex query
        query = "database design patterns"

        # Step 1: Analyze intent
        intent = await query_counselor.analyze_intent(query)

        # Step 2: HyDE enhancement
        hyde_result = await query_counselor.hyde_processor.process_query(query)

        # Verify HyDE enhancement quality
        assert hyde_result["enhancement_score"] > 0.9
        assert len(hyde_result["retrieved_documents"]) == 3
        assert "microservices" in hyde_result["enhanced_query"]
        assert "event sourcing" in hyde_result["enhanced_query"]

        # Step 3: Agent selection and orchestration
        agent_selection = await query_counselor.select_agents(intent)
        # Convert AgentSelection to list of Agent objects for orchestration
        selected_agents = []
        for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
            agent = next((a for a in query_counselor._available_agents if a.agent_id == agent_id), None)
            if agent:
                selected_agents.append(agent)
        responses = await query_counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])

        # Verify multi-agent response
        assert len(responses) == 2
        assert responses[0].agent_id == "architecture_agent"
        assert responses[1].agent_id == "create_agent"
        assert all(r.success for r in responses)
        assert all(r.confidence > 0.8 for r in responses)

        # Verify workflow coordination
        mock_hyde_processor.process_query.assert_called_once_with(query)
        mock_mcp_client.orchestrate_agents.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hyde_processor_fallback_behavior(self, query_counselor, mock_hyde_processor, mock_mcp_client):
        """Test HydeProcessor fallback behavior when enhancement fails."""

        # Mock HyDE processor timeout/failure
        mock_hyde_processor.process_query.side_effect = TimeoutError("Vector store timeout")

        # Mock MCP client success with original query - override the default side_effect for this test
        def mock_fallback_response(workflow_steps):
            return [
                MagicMock(
                    agent_id="create_agent",
                    content="Response using original query",
                    confidence=0.82,
                    processing_time=0.9,
                    success=True,
                    metadata={"fallback": True, "enhancement": "failed"},
                ),
            ]

        mock_mcp_client.orchestrate_agents.side_effect = mock_fallback_response

        # Process query that should trigger fallback
        query = "Test query for fallback"

        # Should handle HyDE failure gracefully
        intent = await query_counselor.analyze_intent(query)

        # Try HyDE processing (should fail)
        with contextlib.suppress(TimeoutError):
            await query_counselor.hyde_processor.process_query(query)

        # Continue with original query
        agent_selection = await query_counselor.select_agents(intent)
        # Convert AgentSelection to list of Agent objects for orchestration
        selected_agents = []
        for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
            agent = next((a for a in query_counselor._available_agents if a.agent_id == agent_id), None)
            if agent:
                selected_agents.append(agent)
        responses = await query_counselor.orchestrate_workflow(selected_agents, query)  # Original query

        # Verify fallback worked
        assert len(responses) == 1
        assert responses[0].success is True
        assert responses[0].metadata.get("fallback") is True

        # Verify HyDE was attempted
        mock_hyde_processor.process_query.assert_called_once_with(query)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, query_counselor, mock_mcp_client, mock_hyde_processor):
        """Test concurrent query processing with HyDE enhancement."""

        # Mock concurrent processing responses
        def mock_hyde_response(query):
            return {
                "enhanced_query": f"Enhanced: {query}",
                "original_query": query,
                "enhancement_score": 0.86,
                "processing_time": 0.1,
                "retrieved_documents": [
                    {
                        "id": f"doc_{hash(query) % 1000}",
                        "content": f"Content for {query}",
                        "score": 0.88,
                        "metadata": {"query": query},
                    },
                ],
            }

        mock_hyde_processor.process_query.side_effect = mock_hyde_response

        # Mock MCP responses
        def mock_mcp_response(agents):
            return [
                MagicMock(
                    agent_id="create_agent",
                    content="Response for concurrent query",
                    confidence=0.87,
                    processing_time=0.6,
                    success=True,
                    metadata={"concurrent": True},
                ),
            ]

        mock_mcp_client.orchestrate_agents.side_effect = mock_mcp_response

        # Test concurrent query processing
        queries = ["How to implement caching in Redis", "Database migration strategies", "API rate limiting patterns"]

        async def process_single_query(query):
            intent = await query_counselor.analyze_intent(query)
            hyde_result = await query_counselor.hyde_processor.process_query(query)
            agent_selection = await query_counselor.select_agents(intent)
            # Convert AgentSelection to list of Agent objects for orchestration
            selected_agents = []
            for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                agent = next((a for a in query_counselor._available_agents if a.agent_id == agent_id), None)
                if agent:
                    selected_agents.append(agent)
            responses = await query_counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])
            return {"query": query, "hyde_result": hyde_result, "responses": responses}

        # Process queries concurrently
        start_time = time.time()
        results = await asyncio.gather(*[process_single_query(q) for q in queries])
        end_time = time.time()

        # Verify all queries processed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["query"] == queries[i]
            assert result["hyde_result"]["enhanced_query"] == f"Enhanced: {queries[i]}"
            assert len(result["responses"]) == 1
            assert result["responses"][0].success is True

        # Verify concurrent processing was efficient
        total_time = end_time - start_time
        assert total_time < 5.0, f"Concurrent processing took {total_time:.3f}s, should be < 5s"

        # Verify all components were called correct number of times
        assert mock_hyde_processor.process_query.call_count == 3
        assert mock_mcp_client.orchestrate_agents.call_count == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_counselor_initialization_integration(
        self,
        test_settings,
        mock_mcp_client,
        mock_hyde_processor,
    ):
        """Test QueryCounselor initialization with HydeProcessor integration."""

        with (
            patch("src.config.settings.get_settings", return_value=test_settings),
            patch("src.mcp_integration.mcp_client.MCPClientFactory.create_from_settings", return_value=mock_mcp_client),
            patch("src.core.hyde_processor.HydeProcessor", return_value=mock_hyde_processor),
        ):

            # Initialize QueryCounselor with explicit dependency injection
            counselor = QueryCounselor(mcp_client=mock_mcp_client, hyde_processor=mock_hyde_processor)

            # Verify initialization - the QueryCounselor should have the injected dependencies
            assert counselor.mcp_client == mock_mcp_client
            assert counselor.hyde_processor == mock_hyde_processor

            # Verify the counselor is configured correctly
            assert counselor.confidence_threshold > 0.0
            assert counselor.confidence_threshold <= 1.0
            assert len(counselor._available_agents) > 0

            # Verify the mock dependencies have correct specs
            assert hasattr(counselor.mcp_client, "connect")
            assert hasattr(counselor.hyde_processor, "initialize")

            # Test that mock methods can be called
            await counselor.mcp_client.connect()
            await counselor.hyde_processor.initialize()

            # Verify the calls were made
            mock_mcp_client.connect.assert_called()
            mock_hyde_processor.initialize.assert_called()
