"""
Comprehensive unit tests for QueryCounselor class.

This module provides comprehensive unit test coverage for the QueryCounselor
class and its methods, including intent analysis, agent selection, workflow
orchestration, and HyDE integration.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / ".." / ".." / "src"))

from src.core.hyde_processor import EnhancedQuery, HydeProcessor, QueryAnalysis, RankedResults, SpecificityLevel
from src.core.query_counselor import (
    Agent,
    AgentSelection,
    FinalResponse,
    QueryCounselor,
    QueryIntent,
    QueryResponse,
    QueryType,
)
from src.core.vector_store import VectorStore
from src.mcp_integration.mcp_client import MCPClient
from src.mcp_integration.mcp_client import Response as MCPResponse


class TestQueryCounselor:
    """Test suite for QueryCounselor class."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        mock_client = Mock(spec=MCPClient)
        mock_client.initialize = AsyncMock()
        mock_client.call_agent = AsyncMock()
        mock_client.get_available_agents = AsyncMock(
            return_value=["create_agent", "security_agent", "performance_agent"],
        )
        mock_client.is_connected = Mock(return_value=True)
        mock_client.validate_query = AsyncMock(
            return_value={"is_valid": True, "sanitized_query": "test query", "potential_issues": []},
        )

        mock_client.orchestrate_agents = AsyncMock(
            return_value=[
                MCPResponse(
                    agent_id="create_agent",
                    content="Response content",
                    confidence=0.85,
                    processing_time=0.3,
                    success=True,
                    metadata={"source": "test"},
                ),
            ],
        )
        return mock_client

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock(spec=VectorStore)
        mock_store.search = AsyncMock()
        mock_store.health_check = AsyncMock(return_value=True)
        return mock_store

    @pytest.fixture
    def mock_hyde_processor(self, mock_vector_store):
        """Create mock HyDE processor."""
        mock_processor = Mock(spec=HydeProcessor)
        mock_processor.three_tier_analysis = AsyncMock()
        mock_processor.process_query = AsyncMock()
        mock_processor.vector_store = mock_vector_store
        return mock_processor

    @pytest.fixture
    def query_counselor(self, mock_mcp_client, mock_hyde_processor):
        """Create QueryCounselor instance with mocked dependencies."""
        return QueryCounselor(
            mcp_client=mock_mcp_client,
            hyde_processor=mock_hyde_processor,
            enable_hybrid_routing=False,  # Disable hybrid routing for tests
        )

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return [
            "How to implement user authentication in Python?",
            "What are the best practices for API security?",
            "Explain database optimization techniques",
            "How to handle errors in async Python code?",
            "What are microservices and when to use them?",
            "Implement caching strategies for web applications",
            "How to secure API endpoints against attacks?",
            "Design patterns for scalable applications",
            "Performance monitoring best practices",
            "Container orchestration with Kubernetes",
        ]

    # Test QueryCounselor initialization
    def test_query_counselor_initialization(self, query_counselor):
        """Test QueryCounselor initialization."""
        assert query_counselor is not None
        assert hasattr(query_counselor, "mcp_client")
        assert hasattr(query_counselor, "hyde_processor")
        assert hasattr(query_counselor, "confidence_threshold")
        assert query_counselor.confidence_threshold == 0.7

    def test_query_counselor_initialization_with_custom_threshold(self, mock_mcp_client, mock_hyde_processor):
        """Test QueryCounselor initialization with custom confidence threshold."""
        custom_threshold = 0.8
        counselor = QueryCounselor(
            mcp_client=mock_mcp_client,
            hyde_processor=mock_hyde_processor,
            confidence_threshold=custom_threshold,
        )
        assert counselor.confidence_threshold == custom_threshold

    # Test analyze_intent method
    @pytest.mark.asyncio
    async def test_analyze_intent_basic(self, query_counselor):
        """Test basic intent analysis."""
        query = "How to implement authentication in Python?"

        intent = await query_counselor.analyze_intent(query)

        assert isinstance(intent, QueryIntent)
        assert intent.original_query == query
        assert isinstance(intent.query_type, QueryType)
        assert 0.0 <= intent.confidence <= 1.0
        assert len(intent.keywords) > 0
        assert len(intent.context_requirements) >= 0

    @pytest.mark.asyncio
    async def test_analyze_intent_different_query_types(self, query_counselor, sample_queries):
        """Test intent analysis with different query types."""
        for query in sample_queries[:5]:  # Test first 5 queries
            intent = await query_counselor.analyze_intent(query)

            assert isinstance(intent, QueryIntent)
            assert intent.original_query == query
            assert isinstance(intent.query_type, QueryType)
            assert 0.0 <= intent.confidence <= 1.0
            assert len(intent.keywords) > 0

    @pytest.mark.asyncio
    async def test_analyze_intent_empty_query(self, query_counselor):
        """Test intent analysis with empty query."""
        intent = await query_counselor.analyze_intent("")

        assert isinstance(intent, QueryIntent)
        assert intent.original_query == ""
        assert intent.query_type == QueryType.UNKNOWN
        assert intent.confidence < 0.5

    @pytest.mark.asyncio
    async def test_analyze_intent_very_short_query(self, query_counselor):
        """Test intent analysis with very short query."""
        intent = await query_counselor.analyze_intent("help")

        assert isinstance(intent, QueryIntent)
        assert intent.original_query == "help"
        assert isinstance(intent.query_type, QueryType)
        assert intent.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_intent_long_query(self, query_counselor):
        """Test intent analysis with long, complex query."""
        long_query = (
            "I need to implement a comprehensive authentication system for a web application "
            "that includes user registration, login, logout, password reset, two-factor "
            "authentication, session management, and integration with OAuth providers like "
            "Google and GitHub. The system should be secure, scalable, and follow best "
            "practices for web security including protection against common attacks like "
            "SQL injection, XSS, and CSRF."
        )

        intent = await query_counselor.analyze_intent(long_query)

        assert isinstance(intent, QueryIntent)
        assert intent.original_query == long_query
        assert isinstance(intent.query_type, QueryType)
        assert intent.confidence >= 0.5  # Should be confident with detailed query
        assert len(intent.keywords) > 5  # Should extract multiple keywords

    # Test select_agents method
    @pytest.mark.asyncio
    async def test_select_agents_basic(self, query_counselor):
        """Test basic agent selection."""
        intent = QueryIntent(
            original_query="How to implement authentication?",
            query_type=QueryType.IMPLEMENTATION,
            confidence=0.8,
            complexity="medium",
            keywords=["authentication", "implement", "security"],
            context_requirements=["python", "web"],
        )

        selection = await query_counselor.select_agents(intent)

        assert isinstance(selection, AgentSelection)
        assert len(selection.primary_agents) > 0
        assert len(selection.secondary_agents) >= 0
        assert selection.reasoning is not None
        assert selection.confidence > 0.0

    @pytest.mark.asyncio
    async def test_select_agents_security_query(self, query_counselor):
        """Test agent selection for security-related query."""
        intent = QueryIntent(
            original_query="How to protect against SQL injection?",
            query_type=QueryType.SECURITY,
            confidence=0.9,
            complexity="medium",
            keywords=["sql", "injection", "security", "protection"],
            context_requirements=["database", "web"],
            requires_agents=["security_agent"],
        )

        selection = await query_counselor.select_agents(intent)

        assert isinstance(selection, AgentSelection)
        assert "security_agent" in selection.primary_agents
        assert selection.confidence > 0.7

    @pytest.mark.asyncio
    async def test_select_agents_performance_query(self, query_counselor):
        """Test agent selection for performance-related query."""
        intent = QueryIntent(
            original_query="How to optimize database queries?",
            query_type=QueryType.PERFORMANCE,
            confidence=0.85,
            complexity="medium",
            keywords=["optimize", "database", "performance", "queries"],
            context_requirements=["database", "sql"],
            requires_agents=["performance_agent"],
        )

        selection = await query_counselor.select_agents(intent)

        assert isinstance(selection, AgentSelection)
        assert "performance_agent" in selection.primary_agents
        assert selection.confidence > 0.7

    @pytest.mark.asyncio
    async def test_select_agents_mcp_client_failure(self, query_counselor, mock_mcp_client):
        """Test agent selection when MCP client fails."""
        mock_mcp_client.get_available_agents.side_effect = Exception("MCP connection failed")

        intent = QueryIntent(
            original_query="How to implement authentication?",
            query_type=QueryType.IMPLEMENTATION,
            confidence=0.8,
            complexity="medium",
            keywords=["authentication", "implement"],
            context_requirements=["python"],
            requires_agents=["create_agent"],
        )

        selection = await query_counselor.select_agents(intent)

        # Should fall back to default agents
        assert isinstance(selection, AgentSelection)
        assert "create_agent" in selection.primary_agents
        assert selection.confidence > 0.0

    # Test orchestrate_workflow method
    @pytest.mark.asyncio
    async def test_orchestrate_workflow_basic(self, query_counselor, mock_mcp_client):
        """Test basic workflow orchestration."""
        # Mock MCP client responses
        mock_mcp_client.call_agent.return_value = {
            "content": "Here's how to implement authentication...",
            "confidence": 0.85,
            "sources": ["auth_best_practices.md"],
        }

        # Create list of Agent objects for orchestration
        agents = [
            Agent(
                agent_id="create_agent",
                agent_type="create",
                capabilities=["prompt_enhancement", "template_generation"],
            ),
            Agent(
                agent_id="security_agent",
                agent_type="security",
                capabilities=["security_analysis", "vulnerability_assessment"],
            ),
        ]

        query = "How to implement authentication?"
        result = await query_counselor.orchestrate_workflow(agents, query)

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].agent_id == "create_agent"
        assert result[0].success is True
        assert result[0].content == "Response content"

    @pytest.mark.asyncio
    async def test_orchestrate_workflow_with_multiple_agents(self, query_counselor, mock_mcp_client):
        """Test workflow orchestration with multiple agents."""

        # Mock orchestrate_agents to return multiple responses
        mock_mcp_client.orchestrate_agents = AsyncMock(
            return_value=[
                MCPResponse(
                    agent_id="create_agent",
                    content="Implementation approach for authentication...",
                    confidence=0.8,
                    processing_time=0.3,
                    success=True,
                    metadata={"source": "implementation_guide.md"},
                ),
                MCPResponse(
                    agent_id="security_agent",
                    content="Security considerations for authentication...",
                    confidence=0.9,
                    processing_time=0.4,
                    success=True,
                    metadata={"source": "security_best_practices.md"},
                ),
                MCPResponse(
                    agent_id="performance_agent",
                    content="Performance optimization for auth systems...",
                    confidence=0.75,
                    processing_time=0.5,
                    success=True,
                    metadata={"source": "performance_guide.md"},
                ),
            ],
        )

        # Create list of Agent objects for orchestration
        agents = [
            Agent(
                agent_id="create_agent",
                agent_type="create",
                capabilities=["prompt_enhancement", "template_generation"],
            ),
            Agent(
                agent_id="security_agent",
                agent_type="security",
                capabilities=["security_analysis", "vulnerability_assessment"],
            ),
            Agent(
                agent_id="performance_agent",
                agent_type="performance",
                capabilities=["performance_optimization", "monitoring"],
            ),
        ]

        query = "How to implement secure and fast authentication?"
        result = await query_counselor.orchestrate_workflow(agents, query)

        assert isinstance(result, list)
        assert len(result) >= 2  # At least primary agents
        assert result[0].success is True
        assert result[0].content is not None
        assert "Implementation approach" in result[0].content

    @pytest.mark.asyncio
    async def test_orchestrate_workflow_agent_failure(self, query_counselor, mock_mcp_client):
        """Test workflow orchestration when an agent fails."""

        # Mock one agent failing
        def mock_agent_call(agent_name, query, context=None):
            if agent_name == "failing_agent":
                raise Exception("Agent failed")
            return {
                "content": f"Response from {agent_name}...",
                "confidence": 0.8,
                "sources": [f"{agent_name}_guide.md"],
            }

        mock_mcp_client.call_agent.side_effect = mock_agent_call

        # Create list of Agent objects for orchestration
        agents = [
            Agent(
                agent_id="create_agent",
                agent_type="create",
                capabilities=["prompt_enhancement", "template_generation"],
            ),
            Agent(
                agent_id="failing_agent",
                agent_type="failing",
                capabilities=["failing_capability"],
            ),
        ]

        query = "Test query"
        result = await query_counselor.orchestrate_workflow(agents, query)

        assert isinstance(result, list)
        assert len(result) >= 1  # At least one response (even if failed)
        # Should have both successful and failed responses
        successful_responses = [r for r in result if r.success]
        assert len(successful_responses) >= 1  # At least one successful response

    # Test synthesize_response method
    @pytest.mark.asyncio
    async def test_synthesize_response_basic(self, query_counselor):
        """Test basic response synthesis."""
        # Create list of MCPResponse objects
        agent_outputs = [
            MCPResponse(
                agent_id="create_agent",
                content="Here's how to implement authentication...",
                confidence=0.85,
                processing_time=0.5,
                success=True,
                metadata={"source": "implementation_guide.md"},
            ),
            MCPResponse(
                agent_id="security_agent",
                content="Security best practices include...",
                confidence=0.9,
                processing_time=0.3,
                success=True,
                metadata={"source": "security_guide.md"},
            ),
        ]

        response = await query_counselor.synthesize_response(agent_outputs)

        assert isinstance(response, FinalResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert 0.0 <= response.confidence <= 1.0
        assert len(response.agents_used) > 0
        assert response.processing_time > 0.0
        assert len(response.metadata) >= 0

    @pytest.mark.asyncio
    async def test_synthesize_response_empty_workflow(self, query_counselor):
        """Test response synthesis with empty workflow."""
        # Empty list of MCPResponse objects
        agent_outputs = []

        response = await query_counselor.synthesize_response(agent_outputs)

        assert isinstance(response, FinalResponse)
        assert response.content is not None
        assert response.confidence < 0.5
        assert len(response.agents_used) == 0
        assert response.processing_time >= 0.0

    # Test process_query method (main entry point)
    @pytest.mark.asyncio
    async def test_process_query_basic(self, query_counselor, mock_mcp_client):
        """Test basic query processing."""
        mock_mcp_client.call_agent.return_value = {
            "content": "Response content",
            "confidence": 0.85,
            "sources": ["test_source.md"],
        }

        query = "How to implement authentication in Python?"
        response = await query_counselor.process_query(query)

        assert isinstance(response, QueryResponse)
        assert response.response is not None
        assert len(response.response) > 0
        assert response.confidence > 0.0
        assert len(response.agents_used) > 0
        assert response.processing_time > 0.0

    @pytest.mark.asyncio
    async def test_process_query_with_different_types(self, query_counselor, mock_mcp_client, sample_queries):
        """Test query processing with different query types."""
        mock_mcp_client.call_agent.return_value = {
            "content": "Response content",
            "confidence": 0.8,
            "sources": ["test_source.md"],
        }

        for query in sample_queries[:3]:  # Test first 3 queries
            response = await query_counselor.process_query(query)

            assert isinstance(response, QueryResponse)
            assert response.response is not None
            assert response.confidence > 0.0
            assert len(response.agents_used) > 0
            assert response.processing_time > 0.0

    # Test process_query_with_hyde method
    @pytest.mark.asyncio
    async def test_process_query_with_hyde_basic(self, query_counselor, mock_mcp_client, mock_hyde_processor):
        """Test query processing with HyDE integration."""
        # Create the QueryAnalysis that will be inside the EnhancedQuery
        query_analysis = QueryAnalysis(
            original_query="How to implement secure user authentication in Python web applications with detailed analysis of best practices and security considerations?",
            specificity_score=75,
            specificity_level=SpecificityLevel.MEDIUM,
            enhanced_query="How to implement secure user authentication in Python web applications with detailed analysis of best practices and security considerations?",
            processing_strategy="enhanced",
            confidence=0.85,
            reasoning="Query demonstrates good specificity with clear domain and requirements",
            guiding_questions=[],
            processing_time=0.1,
        )

        # Create the EnhancedQuery that three_tier_analysis should actually return
        enhanced_query = EnhancedQuery(
            original_query="How to implement secure user authentication in Python web applications with detailed analysis of best practices and security considerations?",
            enhanced_query="How to implement secure user authentication in Python web applications with detailed analysis of best practices and security considerations?",
            embeddings=[],
            hypothetical_docs=[],
            specificity_analysis=query_analysis,
            processing_strategy="enhanced",
        )

        mock_hyde_processor.three_tier_analysis.return_value = enhanced_query

        mock_hyde_processor.process_query.return_value = RankedResults(
            results=[],
            total_found=1,
            processing_time=0.3,
            ranking_method="similarity",
            hyde_enhanced=True,
        )

        # Mock MCP client response
        mock_mcp_client.call_agent.return_value = {
            "content": "Enhanced response with HyDE context",
            "confidence": 0.9,
            "sources": ["auth_guide.md"],
        }

        query = "How to implement secure user authentication in Python web applications with detailed analysis of best practices and security considerations?"
        response = await query_counselor.process_query_with_hyde(query)

        assert isinstance(response, QueryResponse)
        assert response.response is not None
        assert response.confidence > 0.0
        assert len(response.agents_used) > 0
        assert response.processing_time > 0.0

        # Verify HyDE processor was called
        mock_hyde_processor.three_tier_analysis.assert_called_once_with(query)
        mock_hyde_processor.process_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_with_hyde_different_specificity_levels(
        self,
        query_counselor,
        mock_mcp_client,
        mock_hyde_processor,
    ):
        """Test HyDE processing with different specificity levels."""
        specificity_levels = [(SpecificityLevel.LOW, 30), (SpecificityLevel.MEDIUM, 65), (SpecificityLevel.HIGH, 90)]

        for level, score in specificity_levels:
            mock_hyde_processor.three_tier_analysis.return_value = QueryAnalysis(
                original_query="Test query",
                specificity_score=score,
                specificity_level=level,
                enhanced_query="Enhanced test query",
                processing_strategy=level.value.lower(),
                confidence=0.8,
            )

            mock_hyde_processor.process_query.return_value = RankedResults(
                results=[],
                total_found=1,
                processing_time=0.2,
                ranking_method="similarity",
                hyde_enhanced=True,
            )

            mock_mcp_client.call_agent.return_value = {
                "content": f"Response for {level.value} specificity",
                "confidence": 0.8,
                "sources": [],
            }

            response = await query_counselor.process_query_with_hyde("Test query")

            assert isinstance(response, QueryResponse)
            assert response.response is not None
            assert response.confidence > 0.0

    # Test error handling
    @pytest.mark.asyncio
    async def test_process_query_mcp_failure(self, query_counselor, mock_mcp_client):
        """Test query processing when MCP client fails."""
        mock_mcp_client.orchestrate_agents.side_effect = Exception("MCP connection failed")

        query = "How to implement authentication?"
        response = await query_counselor.process_query(query)

        assert isinstance(response, QueryResponse)
        assert response.response is not None
        assert response.confidence < 0.5  # Should have low confidence due to failure
        assert response.processing_time > 0.0

    @pytest.mark.asyncio
    async def test_process_query_with_hyde_processor_failure(
        self,
        query_counselor,
        mock_mcp_client,
        mock_hyde_processor,
    ):
        """Test HyDE processing when processor fails."""
        mock_hyde_processor.three_tier_analysis.side_effect = Exception("HyDE processor failed")

        # Should fall back to regular processing
        mock_mcp_client.call_agent.return_value = {"content": "Fallback response", "confidence": 0.7, "sources": []}

        query = "How to implement authentication?"
        response = await query_counselor.process_query_with_hyde(query)

        assert isinstance(response, QueryResponse)
        assert response.response is not None
        assert response.confidence > 0.0

    # Test performance with decorators
    @pytest.mark.asyncio
    async def test_analyze_intent_performance_decorator(self, query_counselor):
        """Test that performance decorators are applied to analyze_intent."""
        # This test verifies the decorator is applied without breaking functionality
        query = "How to implement authentication?"

        intent = await query_counselor.analyze_intent(query)

        assert isinstance(intent, QueryIntent)
        assert intent.original_query == query

        # Test caching by running same query again
        intent2 = await query_counselor.analyze_intent(query)
        assert intent2.original_query == query

    # Test concurrent processing
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, query_counselor, mock_mcp_client):
        """Test concurrent query processing."""
        mock_mcp_client.call_agent.return_value = {"content": "Concurrent response", "confidence": 0.8, "sources": []}

        queries = [
            "How to implement authentication?",
            "What are REST API best practices?",
            "How to optimize database queries?",
        ]

        # Process queries concurrently
        tasks = [query_counselor.process_query(query) for query in queries]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == len(queries)
        for response in responses:
            assert isinstance(response, QueryResponse)
            assert response.response is not None
            assert response.confidence > 0.0

    # Test edge cases
    @pytest.mark.asyncio
    async def test_process_query_none_input(self, query_counselor):
        """Test processing None input."""
        response = await query_counselor.process_query(None)

        assert isinstance(response, QueryResponse)
        assert response.confidence < 0.5

    @pytest.mark.asyncio
    async def test_process_query_whitespace_input(self, query_counselor, mock_mcp_client):
        """Test processing whitespace-only input."""
        mock_mcp_client.orchestrate_agents.side_effect = Exception("Invalid query")

        response = await query_counselor.process_query("   \n\t  ")

        assert isinstance(response, QueryResponse)
        assert response.confidence < 0.5

    @pytest.mark.asyncio
    async def test_process_query_very_long_input(self, query_counselor, mock_mcp_client):
        """Test processing very long input."""
        mock_mcp_client.call_agent.return_value = {"content": "Long query response", "confidence": 0.8, "sources": []}

        very_long_query = "How to implement authentication? " * 100
        response = await query_counselor.process_query(very_long_query)

        assert isinstance(response, QueryResponse)
        assert response.response is not None
        assert response.confidence > 0.0

    # Test configuration and initialization variations
    def test_query_counselor_with_none_dependencies(self):
        """Test QueryCounselor with None dependencies."""
        counselor = QueryCounselor(mcp_client=None, hyde_processor=None)

        assert counselor.mcp_client is None
        assert counselor.hyde_processor is None
        assert counselor.confidence_threshold == 0.7

    def test_query_counselor_with_invalid_threshold(self, mock_mcp_client, mock_hyde_processor):
        """Test QueryCounselor with invalid confidence threshold."""
        # Should handle invalid threshold gracefully
        counselor = QueryCounselor(
            mcp_client=mock_mcp_client,
            hyde_processor=mock_hyde_processor,
            confidence_threshold=1.5,  # Invalid threshold
        )

        # Should clamp to valid range
        assert 0.0 <= counselor.confidence_threshold <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
