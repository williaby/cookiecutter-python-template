"""Comprehensive unit tests for src/core/query_counselor.py.

This module tests the QueryCounselor system including:
- Intent analysis and classification
- Agent selection and orchestration
- Multi-agent workflow coordination
- Response synthesis and aggregation
- HyDE processor integration
- Model registry integration
- Error handling and resilience

Test Categories:
- Data models and validation
- Intent analysis with various query types
- Agent selection logic
- Workflow orchestration
- Response synthesis
- HyDE integration workflows
- Error conditions and edge cases
- Performance monitoring integration
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.query_counselor import (
    Agent,
    AgentSelection,
    FinalResponse,
    QueryCounselor,
    QueryIntent,
    QueryResponse,
    QueryType,
    Response,
    WorkflowResult,
    WorkflowStep,
)
from src.mcp_integration.hybrid_router import RoutingStrategy
from src.mcp_integration.mcp_client import MCPError
from src.mcp_integration.mcp_client import Response as MCPResponse


class TestDataModels:
    """Test data model validation and creation."""

    @pytest.mark.unit
    def test_query_intent_creation(self):
        """Test QueryIntent model creation and validation."""
        intent = QueryIntent(
            query_type=QueryType.CREATE_ENHANCEMENT,
            confidence=0.8,
            complexity="medium",
            requires_agents=["create_agent"],
            context_needed=True,
            hyde_recommended=True,
            original_query="Create a better prompt",
            keywords=["create", "prompt"],
            context_requirements=["creative"],
        )

        assert intent.query_type == QueryType.CREATE_ENHANCEMENT
        assert intent.confidence == 0.8
        assert intent.complexity == "medium"
        assert intent.requires_agents == ["create_agent"]
        assert intent.context_needed is True
        assert intent.hyde_recommended is True
        assert intent.original_query == "Create a better prompt"
        assert intent.keywords == ["create", "prompt"]
        assert intent.context_requirements == ["creative"]
        assert intent.metadata == {}

    @pytest.mark.unit
    def test_query_intent_validation_complexity(self):
        """Test QueryIntent complexity validation."""
        # Valid complexity
        intent = QueryIntent(
            query_type=QueryType.GENERAL_QUERY,
            confidence=0.5,
            complexity="simple",
            original_query="test",
        )
        assert intent.complexity == "simple"

        # Invalid complexity should raise ValidationError
        with pytest.raises(ValueError, match="Complexity must be simple, medium, or complex"):
            QueryIntent(
                query_type=QueryType.GENERAL_QUERY,
                confidence=0.5,
                complexity="invalid",
                original_query="test",
            )

    @pytest.mark.unit
    def test_agent_model_creation(self):
        """Test Agent model creation and validation."""
        agent = Agent(
            agent_id="test_agent",
            agent_type="analysis",
            capabilities=["analyze", "review"],
            availability=True,
            load_factor=0.3,
        )

        assert agent.agent_id == "test_agent"
        assert agent.agent_type == "analysis"
        assert agent.capabilities == ["analyze", "review"]
        assert agent.availability is True
        assert agent.load_factor == 0.3

    @pytest.mark.unit
    def test_agent_selection_model(self):
        """Test AgentSelection model creation."""
        selection = AgentSelection(
            primary_agents=["agent1", "agent2"],
            secondary_agents=["agent3"],
            reasoning="Selected based on capabilities",
            confidence=0.9,
        )

        assert selection.primary_agents == ["agent1", "agent2"]
        assert selection.secondary_agents == ["agent3"]
        assert selection.reasoning == "Selected based on capabilities"
        assert selection.confidence == 0.9

    @pytest.mark.unit
    def test_query_response_model(self):
        """Test QueryResponse model creation."""
        response = QueryResponse(
            response="Test response content",
            agents_used=["agent1", "agent2"],
            processing_time=1.5,
            success=True,
            confidence=0.85,
            metadata={"test": "data"},
        )

        assert response.response == "Test response content"
        assert response.agents_used == ["agent1", "agent2"]
        assert response.processing_time == 1.5
        assert response.success is True
        assert response.confidence == 0.85
        assert response.metadata == {"test": "data"}

    @pytest.mark.unit
    def test_workflow_step_model(self):
        """Test WorkflowStep model creation."""
        step = WorkflowStep(
            step_id="step_1",
            agent_id="test_agent",
            input_data={"query": "test query"},
            dependencies=["step_0"],
            timeout_seconds=60,
        )

        assert step.step_id == "step_1"
        assert step.agent_id == "test_agent"
        assert step.input_data == {"query": "test query"}
        assert step.dependencies == ["step_0"]
        assert step.timeout_seconds == 60

    @pytest.mark.unit
    def test_workflow_result_model(self):
        """Test WorkflowResult model creation."""
        step = WorkflowStep(
            step_id="step_1",
            agent_id="test_agent",
            input_data={"query": "test"},
            dependencies=[],
        )

        result = WorkflowResult(
            steps=[step],
            final_response="Final result",
            success=True,
            total_time=2.0,
            agents_used=["test_agent"],
            metadata={"workflow": "complete"},
        )

        assert len(result.steps) == 1
        assert result.steps[0].step_id == "step_1"
        assert result.final_response == "Final result"
        assert result.success is True
        assert result.total_time == 2.0
        assert result.agents_used == ["test_agent"]
        assert result.metadata == {"workflow": "complete"}

    @pytest.mark.unit
    def test_response_model(self):
        """Test Response model creation."""
        response = Response(
            agent_id="test_agent",
            content="Response content",
            metadata={"source": "test"},
            confidence=0.9,
            processing_time=1.0,
            success=True,
            error_message=None,
        )

        assert response.agent_id == "test_agent"
        assert response.content == "Response content"
        assert response.metadata == {"source": "test"}
        assert response.confidence == 0.9
        assert response.processing_time == 1.0
        assert response.success is True
        assert response.error_message is None

    @pytest.mark.unit
    def test_final_response_model(self):
        """Test FinalResponse model creation."""
        response = FinalResponse(
            content="Final response content",
            sources=["agent1", "agent2"],
            confidence=0.8,
            processing_time=2.5,
            query_type=QueryType.CREATE_ENHANCEMENT,
            agents_used=["agent1", "agent2"],
            metadata={"synthesis": "complete"},
        )

        assert response.content == "Final response content"
        assert response.sources == ["agent1", "agent2"]
        assert response.confidence == 0.8
        assert response.processing_time == 2.5
        assert response.query_type == QueryType.CREATE_ENHANCEMENT
        assert response.agents_used == ["agent1", "agent2"]
        assert response.metadata == {"synthesis": "complete"}


class TestQueryCounselorInitialization:
    """Test QueryCounselor initialization and configuration."""

    @pytest.mark.unit
    def test_initialization_default(self):
        """Test default QueryCounselor initialization."""
        counselor = QueryCounselor()

        assert counselor.mcp_client is None
        assert counselor.model_registry is None
        assert counselor.hyde_processor is None
        assert counselor.confidence_threshold == 0.7
        assert counselor.hybrid_routing_enabled is False
        assert len(counselor._available_agents) == 5  # Default mock agents

    @pytest.mark.unit
    def test_initialization_with_mcp_client(self):
        """Test QueryCounselor initialization with MCP client."""
        mock_client = Mock()
        counselor = QueryCounselor(mcp_client=mock_client)

        assert counselor.mcp_client == mock_client
        assert counselor.model_registry is None
        assert counselor.hybrid_routing_enabled is False

    @pytest.mark.unit
    @patch("src.core.query_counselor.HybridRouter")
    @patch("src.core.query_counselor.get_model_registry")
    def test_initialization_with_hybrid_routing(self, mock_get_registry, mock_hybrid_router):
        """Test QueryCounselor initialization with hybrid routing enabled."""
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        mock_router = Mock()
        mock_hybrid_router.return_value = mock_router

        counselor = QueryCounselor(
            enable_hybrid_routing=True,
            routing_strategy=RoutingStrategy.OPENROUTER_PRIMARY,
        )

        assert counselor.mcp_client == mock_router
        assert counselor.model_registry == mock_registry
        assert counselor.hybrid_routing_enabled is True

        mock_hybrid_router.assert_called_once_with(
            strategy=RoutingStrategy.OPENROUTER_PRIMARY,
            enable_gradual_rollout=True,
        )

    @pytest.mark.unit
    def test_initialization_with_hybrid_router_instance(self):
        """Test QueryCounselor initialization with HybridRouter instance."""
        from src.core.query_counselor import HybridRouter

        mock_hybrid_router = Mock(spec=HybridRouter)

        with patch("src.core.query_counselor.get_model_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_get_registry.return_value = mock_registry

            counselor = QueryCounselor(mcp_client=mock_hybrid_router)

            assert counselor.mcp_client == mock_hybrid_router
            assert counselor.model_registry == mock_registry
            assert counselor.hybrid_routing_enabled is True

    @pytest.mark.unit
    def test_initialization_with_hyde_processor(self):
        """Test QueryCounselor initialization with HyDE processor."""
        mock_hyde = Mock()
        counselor = QueryCounselor(hyde_processor=mock_hyde)

        assert counselor.hyde_processor == mock_hyde

    @pytest.mark.unit
    def test_initialization_confidence_threshold_clamping(self):
        """Test confidence threshold is clamped to valid range."""
        # Test below minimum
        counselor = QueryCounselor(confidence_threshold=-0.5)
        assert counselor.confidence_threshold == 0.0

        # Test above maximum
        counselor = QueryCounselor(confidence_threshold=1.5)
        assert counselor.confidence_threshold == 1.0

        # Test valid range
        counselor = QueryCounselor(confidence_threshold=0.8)
        assert counselor.confidence_threshold == 0.8


class TestIntentAnalysis:
    """Test query intent analysis and classification."""

    @pytest.fixture
    def counselor(self):
        """Create QueryCounselor instance for testing."""
        return QueryCounselor()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_empty_query(self, counselor):
        """Test intent analysis with empty query."""
        intent = await counselor.analyze_intent("")

        assert intent.query_type == QueryType.UNKNOWN
        assert intent.confidence == 0.0
        assert intent.complexity == "simple"
        assert intent.requires_agents == []
        assert intent.context_needed is False
        assert intent.hyde_recommended is False
        assert intent.original_query == ""
        assert intent.keywords == []

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_whitespace_only_query(self, counselor):
        """Test intent analysis with whitespace-only query."""
        intent = await counselor.analyze_intent("   \n\t  ")

        assert intent.query_type == QueryType.UNKNOWN
        assert intent.confidence == 0.0
        assert intent.complexity == "simple"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_create_enhancement(self, counselor):
        """Test intent analysis for create enhancement queries."""
        queries = [
            "Create a better prompt for marketing",
            "Generate an improved version of this text",
            "Enhance my writing style",
            "Improve this content",
        ]

        for query in queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.CREATE_ENHANCEMENT
            assert intent.confidence == 0.8
            assert intent.requires_agents == ["create_agent"]
            assert intent.original_query == query

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_template_generation(self, counselor):
        """Test intent analysis for template generation queries."""
        # Test queries that should match template first (no "create" or "generate")
        pure_template_queries = [
            "Template for email marketing",
            "Pattern for user stories",
            "Format this as a standard report",
        ]

        for query in pure_template_queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.TEMPLATE_GENERATION
            assert intent.confidence == 0.85
            assert intent.requires_agents == ["create_agent"]

        # Test queries with "create" + "template" - should match CREATE_ENHANCEMENT first
        create_template_queries = [
            "Create a template for email marketing",
            "Generate a pattern for user stories",
        ]

        for query in create_template_queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.CREATE_ENHANCEMENT
            assert intent.confidence == 0.8
            assert intent.requires_agents == ["create_agent"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_analysis_request(self, counselor):
        """Test intent analysis for analysis queries."""
        queries = [
            "Analyze this code for bugs",
            "Review my document structure",
            "Examine the performance metrics",
            "Evaluate the user experience",
        ]

        for query in queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.ANALYSIS_REQUEST
            assert intent.confidence == 0.8
            assert intent.requires_agents == ["analysis_agent"]
            assert intent.complexity == "medium"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_documentation(self, counselor):
        """Test intent analysis for documentation queries."""
        queries = [
            "Document this API endpoint",
            "Explain how this function works",
            "Describe the architecture",
            "How to set up the environment",
        ]

        for query in queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.DOCUMENTATION
            assert intent.confidence == 0.75
            assert intent.requires_agents == ["analysis_agent", "create_agent"]
            assert intent.complexity == "medium"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_implementation(self, counselor):
        """Test intent analysis for implementation queries."""
        queries = [
            "Implement a user authentication system",
            "Build a REST API endpoint",
            "Develop a data processing pipeline",
            "Code a sorting algorithm",
        ]

        for query in queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.IMPLEMENTATION
            assert intent.confidence == 0.8
            assert intent.requires_agents == ["create_agent"]
            assert intent.context_requirements == ["python"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_security(self, counselor):
        """Test intent analysis for security queries."""
        # Pure security queries (without analyze keywords)
        pure_security_queries = [
            "Check for security vulnerabilities",
            "Secure this authentication flow",
            "Attack surface assessment needed",
        ]

        for query in pure_security_queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.SECURITY
            assert intent.confidence == 0.85
            assert intent.requires_agents == ["security_agent"]
            assert intent.context_requirements == ["security", "web"]

        # Analyze+security queries should return ANALYSIS_REQUEST due to keyword precedence
        analyze_security_queries = [
            "Analyze potential attack vectors",
            "Review auth implementation",
        ]

        for query in analyze_security_queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.ANALYSIS_REQUEST  # analyze keyword has precedence
            assert intent.confidence == 0.8
            assert intent.requires_agents == ["analysis_agent"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_performance(self, counselor):
        """Test intent analysis for performance queries."""
        # Pure performance queries (without create/improve keywords)
        pure_performance_queries = [
            "Optimize this database query",
            "Fix slow loading times",
            "Performance tune the system",
            "Speed up the application",
        ]

        for query in pure_performance_queries:
            intent = await counselor.analyze_intent(query)

            assert intent.query_type == QueryType.PERFORMANCE
            assert intent.confidence == 0.8
            assert intent.requires_agents == ["performance_agent"]
            assert intent.context_requirements == ["performance"]

        # Improve+performance queries should return CREATE_ENHANCEMENT due to keyword precedence
        improve_performance_query = "Improve the application speed"
        intent = await counselor.analyze_intent(improve_performance_query)

        assert intent.query_type == QueryType.CREATE_ENHANCEMENT  # improve keyword has precedence
        assert intent.confidence == 0.8
        assert intent.requires_agents == ["create_agent"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_complexity_analysis(self, counselor):
        """Test query complexity analysis."""
        # Simple query
        simple_intent = await counselor.analyze_intent("Hello")
        assert simple_intent.complexity == "simple"
        assert simple_intent.hyde_recommended is False

        # Medium complexity query (analysis requests have medium complexity set initially)
        # But this query is 9 words and doesn't contain "analysis", "compare", "evaluate"
        # so the general complexity logic keeps it as "simple"
        medium_query = "Analyze this code and provide recommendations for improvement"
        medium_intent = await counselor.analyze_intent(medium_query)
        assert medium_intent.complexity == "simple"  # Overridden by general complexity logic

        # Query that actually gets medium complexity - contains "analysis" keyword
        actual_medium_query = "Provide analysis and comparison of these options"  # 8 words + "analysis" keyword
        actual_medium_intent = await counselor.analyze_intent(actual_medium_query)
        assert actual_medium_intent.complexity == "medium"

        # Complex query (word count threshold)
        complex_query = " ".join(["word"] * 25)  # More than COMPLEX_QUERY_WORD_THRESHOLD
        complex_intent = await counselor.analyze_intent(complex_query)
        assert complex_intent.complexity == "complex"
        assert complex_intent.hyde_recommended is True

        # Complex query (keyword-based)
        complex_keyword_query = "Provide a detailed complex analysis"
        complex_keyword_intent = await counselor.analyze_intent(complex_keyword_query)
        assert complex_keyword_intent.complexity == "complex"
        assert complex_keyword_intent.hyde_recommended is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_keyword_extraction(self, counselor):
        """Test keyword extraction from queries."""
        intent = await counselor.analyze_intent("Create a comprehensive analysis of user behavior patterns")

        # Keywords should be extracted (length > MIN_KEYWORD_LENGTH)
        expected_keywords = ["create", "comprehensive", "analysis", "user", "behavior", "patterns"]
        assert set(intent.keywords) == set(expected_keywords)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_context_needed(self, counselor):
        """Test context_needed determination."""
        # Simple query - no context needed
        simple_intent = await counselor.analyze_intent("Hello")
        assert simple_intent.context_needed is False

        # Medium complexity - context needed
        medium_intent = await counselor.analyze_intent("Analyze this data structure")
        assert medium_intent.context_needed is True

        # Analysis request - context needed
        analysis_intent = await counselor.analyze_intent("Evaluate the performance")
        assert analysis_intent.context_needed is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_with_hybrid_routing(self):
        """Test intent analysis with hybrid routing enabled."""
        with (
            patch("src.core.query_counselor.HybridRouter") as mock_hybrid_router,
            patch("src.core.query_counselor.get_model_registry") as mock_get_registry,
        ):

            mock_registry = Mock()
            mock_registry.select_best_model.return_value = "test-model-id"
            mock_get_registry.return_value = mock_registry
            mock_router = Mock()
            mock_hybrid_router.return_value = mock_router

            counselor = QueryCounselor(enable_hybrid_routing=True)
            intent = await counselor.analyze_intent("Create a complex analysis")

            # Should have hybrid routing metadata
            assert "selected_model" in intent.metadata
            assert intent.metadata["hybrid_routing_enabled"] is True
            assert intent.metadata["task_type"] == "general"
            assert intent.metadata["allow_premium"] is True  # Complex query

            # Model registry should be called
            mock_registry.select_best_model.assert_called_once_with(
                task_type="general",
                allow_premium=True,
            )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_model_selection_failure(self):
        """Test intent analysis when model selection fails."""
        with (
            patch("src.core.query_counselor.HybridRouter") as mock_hybrid_router,
            patch("src.core.query_counselor.get_model_registry") as mock_get_registry,
        ):

            mock_registry = Mock()
            mock_registry.select_best_model.side_effect = Exception("Model selection failed")
            mock_get_registry.return_value = mock_registry
            mock_router = Mock()
            mock_hybrid_router.return_value = mock_router

            counselor = QueryCounselor(enable_hybrid_routing=True)
            intent = await counselor.analyze_intent("Create something")

            # Should still work but with None model
            assert intent.metadata["selected_model"] is None
            assert intent.metadata["hybrid_routing_enabled"] is True


class TestAgentSelection:
    """Test agent selection logic."""

    @pytest.fixture
    def counselor(self):
        """Create QueryCounselor instance for testing."""
        return QueryCounselor()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_select_agents_basic(self, counselor):
        """Test basic agent selection."""
        intent = QueryIntent(
            query_type=QueryType.CREATE_ENHANCEMENT,
            confidence=0.8,
            complexity="medium",
            requires_agents=["create_agent"],
            original_query="Create something",
        )

        selection = await counselor.select_agents(intent)

        assert "create_agent" in selection.primary_agents
        assert selection.confidence == 0.8
        assert "Selected 1 agents" in selection.reasoning

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_select_agents_multiple(self, counselor):
        """Test selecting multiple agents."""
        intent = QueryIntent(
            query_type=QueryType.DOCUMENTATION,
            confidence=0.75,
            complexity="medium",
            requires_agents=["analysis_agent", "create_agent"],
            original_query="Document this code",
        )

        selection = await counselor.select_agents(intent)

        assert len(selection.primary_agents) == 2
        assert "analysis_agent" in selection.primary_agents
        assert "create_agent" in selection.primary_agents
        assert selection.confidence == 0.75

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_select_agents_unavailable_agent(self, counselor):
        """Test agent selection when requested agent is unavailable."""
        # Make create_agent unavailable
        for agent in counselor._available_agents:
            if agent.agent_id == "create_agent":
                agent.availability = False
                break

        intent = QueryIntent(
            query_type=QueryType.CREATE_ENHANCEMENT,
            confidence=0.8,
            complexity="medium",
            requires_agents=["create_agent"],
            original_query="Create something",
        )

        selection = await counselor.select_agents(intent)

        # Should fallback to general agent
        assert "general_agent" in selection.primary_agents
        assert "create_agent" not in selection.primary_agents

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_select_agents_nonexistent_agent(self, counselor):
        """Test agent selection with non-existent agent."""
        intent = QueryIntent(
            query_type=QueryType.UNKNOWN,
            confidence=0.5,
            complexity="simple",
            requires_agents=["nonexistent_agent"],
            original_query="Test query",
        )

        selection = await counselor.select_agents(intent)

        # Should fallback to general agent
        assert "general_agent" in selection.primary_agents
        assert "nonexistent_agent" not in selection.primary_agents

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_select_agents_primary_secondary_split(self, counselor):
        """Test splitting agents into primary and secondary."""
        intent = QueryIntent(
            query_type=QueryType.ANALYSIS_REQUEST,
            confidence=0.8,
            complexity="complex",
            requires_agents=["analysis_agent", "create_agent", "general_agent"],
            original_query="Complex analysis task",
        )

        selection = await counselor.select_agents(intent)

        # First 2 should be primary, rest secondary
        assert len(selection.primary_agents) == 2
        assert len(selection.secondary_agents) >= 0  # Could be 1 or 0 depending on availability


class TestWorkflowOrchestration:
    """Test workflow orchestration logic."""

    @pytest.fixture
    def counselor(self):
        """Create QueryCounselor instance for testing."""
        mock_client = AsyncMock()
        return QueryCounselor(mcp_client=mock_client)

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing."""
        return [
            Agent(agent_id="agent1", agent_type="create", capabilities=["enhance"]),
            Agent(agent_id="agent2", agent_type="analysis", capabilities=["analyze"]),
        ]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_success(self, counselor, sample_agents):
        """Test successful workflow orchestration."""
        # Mock successful responses
        mock_responses = [
            MCPResponse(
                agent_id="agent1",
                content="Response from agent1",
                confidence=0.8,
                processing_time=1.0,
                success=True,
            ),
            MCPResponse(
                agent_id="agent2",
                content="Response from agent2",
                confidence=0.9,
                processing_time=1.2,
                success=True,
            ),
        ]

        counselor.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "test query",
        }
        counselor.mcp_client.orchestrate_agents.return_value = mock_responses

        responses = await counselor.orchestrate_workflow(sample_agents, "test query")

        assert len(responses) == 2
        assert responses[0].agent_id == "agent1"
        assert responses[1].agent_id == "agent2"
        assert all(r.success for r in responses)

        # Verify MCP client calls
        counselor.mcp_client.validate_query.assert_called_once_with("test query")
        counselor.mcp_client.orchestrate_agents.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_with_intent_metadata(self, counselor, sample_agents):
        """Test workflow orchestration with intent metadata."""
        intent = QueryIntent(
            query_type=QueryType.CREATE_ENHANCEMENT,
            confidence=0.8,
            complexity="complex",
            requires_agents=["agent1"],
            original_query="test query",
            metadata={
                "selected_model": "test-model",
                "task_type": "general",
                "hybrid_routing_enabled": True,
            },
        )

        counselor.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "test query",
        }
        counselor.mcp_client.orchestrate_agents.return_value = [
            MCPResponse(
                agent_id="agent1",
                content="Response",
                confidence=0.8,
                processing_time=1.0,
                success=True,
            ),
        ]

        await counselor.orchestrate_workflow(sample_agents, "test query", intent)

        # Verify workflow steps include metadata
        call_args = counselor.mcp_client.orchestrate_agents.call_args[0][0]
        step_input = call_args[0].input_data

        assert step_input["selected_model"] == "test-model"
        assert step_input["task_type"] == "general"
        assert step_input["complexity"] == "complex"
        assert step_input["query_type"] == "create_enhancement"
        assert step_input["hybrid_routing_enabled"] is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_validation_failure(self, counselor, sample_agents):
        """Test workflow orchestration with query validation failure."""
        counselor.mcp_client.validate_query.return_value = {
            "is_valid": False,
            "potential_issues": ["contains malicious content"],
        }

        with pytest.raises(ValueError, match="Query validation failed"):
            await counselor.orchestrate_workflow(sample_agents, "malicious query")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_mcp_error(self, counselor, sample_agents):
        """Test workflow orchestration with MCP error."""
        counselor.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "test query",
        }
        counselor.mcp_client.orchestrate_agents.side_effect = MCPError("Test error", "CONNECTION_ERROR")

        responses = await counselor.orchestrate_workflow(sample_agents, "test query")

        # Should return error responses
        assert len(responses) == 2
        assert all(not r.success for r in responses)
        assert all("MCP error" in r.content for r in responses)
        assert all(r.error_message == "CONNECTION_ERROR: Test error" for r in responses)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_generic_exception(self, counselor, sample_agents):
        """Test workflow orchestration with generic exception."""
        counselor.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "test query",
        }
        counselor.mcp_client.orchestrate_agents.side_effect = Exception("Generic error")

        responses = await counselor.orchestrate_workflow(sample_agents, "test query")

        # Should return error responses
        assert len(responses) == 2
        assert all(not r.success for r in responses)
        assert all("unavailable" in r.content for r in responses)
        assert all(r.error_message == "Generic error" for r in responses)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_no_mcp_client(self, sample_agents):
        """Test workflow orchestration without MCP client."""
        counselor = QueryCounselor()  # No MCP client

        responses = await counselor.orchestrate_workflow(sample_agents, "test query")

        # Should return error responses for all agents
        assert len(responses) == 2
        assert all(not r.success for r in responses)
        assert all("no MCP client" in r.content for r in responses)
        assert all(r.error_message == "No MCP client available" for r in responses)


class TestResponseSynthesis:
    """Test response synthesis logic."""

    @pytest.fixture
    def counselor(self):
        """Create QueryCounselor instance for testing."""
        return QueryCounselor()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_synthesize_response_single_successful(self, counselor):
        """Test synthesizing response from single successful agent."""
        agent_outputs = [
            MCPResponse(
                agent_id="agent1",
                content="Single agent response",
                confidence=0.8,
                processing_time=1.0,
                success=True,
                metadata={"query_type": QueryType.CREATE_ENHANCEMENT},
            ),
        ]

        final_response = await counselor.synthesize_response(agent_outputs)

        assert final_response.content == "Single agent response"
        assert final_response.confidence == 0.8
        assert final_response.sources == ["agent1"]
        assert final_response.agents_used == ["agent1"]
        assert final_response.query_type == QueryType.CREATE_ENHANCEMENT
        assert final_response.metadata["successful_agents"] == 1
        assert final_response.metadata["failed_agents"] == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_synthesize_response_multiple_successful(self, counselor):
        """Test synthesizing response from multiple successful agents."""
        agent_outputs = [
            MCPResponse(
                agent_id="agent1",
                content="Response from agent 1",
                confidence=0.8,
                processing_time=1.0,
                success=True,
            ),
            MCPResponse(
                agent_id="agent2",
                content="Response from agent 2",
                confidence=0.9,
                processing_time=1.2,
                success=True,
            ),
        ]

        final_response = await counselor.synthesize_response(agent_outputs)

        assert "Combined response:" in final_response.content
        assert "Response from agent 1" in final_response.content
        assert "Response from agent 2" in final_response.content
        assert abs(final_response.confidence - 0.85) < 1e-10  # Average of 0.8 and 0.9 (floating point precision)
        assert final_response.sources == ["agent1", "agent2"]
        assert final_response.agents_used == ["agent1", "agent2"]
        assert final_response.metadata["successful_agents"] == 2
        assert final_response.metadata["failed_agents"] == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_synthesize_response_mixed_success_failure(self, counselor):
        """Test synthesizing response from mixed successful and failed agents."""
        agent_outputs = [
            MCPResponse(
                agent_id="agent1",
                content="Successful response",
                confidence=0.8,
                processing_time=1.0,
                success=True,
            ),
            MCPResponse(
                agent_id="agent2",
                content="Failed response",
                confidence=0.0,
                processing_time=0.5,
                success=False,
                error_message="Agent failed",
            ),
        ]

        final_response = await counselor.synthesize_response(agent_outputs)

        assert final_response.content == "Successful response"
        assert final_response.confidence == 0.8
        assert final_response.sources == ["agent1"]
        assert final_response.agents_used == ["agent1"]
        assert final_response.metadata["successful_agents"] == 1
        assert final_response.metadata["failed_agents"] == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_synthesize_response_all_failed(self, counselor):
        """Test synthesizing response when all agents failed."""
        agent_outputs = [
            MCPResponse(
                agent_id="agent1",
                content="",
                confidence=0.0,
                processing_time=0.5,
                success=False,
                error_message="Connection failed",
            ),
            MCPResponse(
                agent_id="agent2",
                content="",
                confidence=0.0,
                processing_time=0.3,
                success=False,
                error_message="Timeout occurred",
            ),
        ]

        final_response = await counselor.synthesize_response(agent_outputs)

        assert "Unable to process query - all agents unavailable" in final_response.content
        assert "Connection failed; Timeout occurred" in final_response.content
        assert final_response.confidence == 0.0
        assert final_response.sources == []
        assert final_response.agents_used == []
        assert final_response.metadata["error"] is True
        assert final_response.metadata["failed_agents"] == 2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_synthesize_response_empty_agent_outputs(self, counselor):
        """Test synthesizing response with empty agent outputs."""
        final_response = await counselor.synthesize_response([])

        assert "Unable to process query - all agents unavailable" in final_response.content
        assert final_response.confidence == 0.0
        assert final_response.sources == []
        assert final_response.agents_used == []
        assert final_response.metadata["error"] is True
        assert final_response.metadata["failed_agents"] == 0


class TestQueryProcessing:
    """Test complete query processing workflows."""

    @pytest.fixture
    def counselor(self):
        """Create QueryCounselor instance for testing."""
        mock_client = AsyncMock()
        return QueryCounselor(mcp_client=mock_client)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_success(self, counselor):
        """Test successful query processing."""
        # Mock all the steps
        counselor.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "test query",
        }
        counselor.mcp_client.orchestrate_agents.return_value = [
            MCPResponse(
                agent_id="general_agent",
                content="Query processed successfully",
                confidence=0.8,
                processing_time=1.0,
                success=True,
            ),
        ]

        response = await counselor.process_query("test query")

        assert response.success is True
        assert response.confidence == 0.8
        assert response.response == "Query processed successfully"
        assert response.agents_used == ["general_agent"]
        assert response.processing_time > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_failure(self, counselor):
        """Test query processing failure."""
        counselor.mcp_client.validate_query.side_effect = Exception("Validation failed")

        response = await counselor.process_query("test query")

        assert response.success is False
        assert response.confidence == 0.0
        assert "Query processing failed" in response.response
        assert response.agents_used == []
        assert response.metadata["error"] is True
        assert response.metadata["error_message"] == "Validation failed"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_with_create_enhancement(self, counselor):
        """Test processing CREATE_ENHANCEMENT query type."""
        counselor.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "create better content",
        }
        counselor.mcp_client.orchestrate_agents.return_value = [
            MCPResponse(
                agent_id="create_agent",
                content="Enhanced content created",
                confidence=0.9,
                processing_time=1.5,
                success=True,
            ),
        ]

        response = await counselor.process_query("create better content")

        assert response.success is True
        assert response.confidence == 0.9
        assert response.agents_used == ["create_agent"]


class TestHydeIntegration:
    """Test HyDE processor integration."""

    @pytest.fixture
    def counselor_with_hyde(self):
        """Create QueryCounselor with HyDE processor for testing."""
        mock_client = AsyncMock()
        mock_hyde = AsyncMock()
        return QueryCounselor(mcp_client=mock_client, hyde_processor=mock_hyde)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_with_hyde_success(self, counselor_with_hyde):
        """Test successful query processing with HyDE integration."""
        # Mock HyDE processor
        mock_enhanced_query = Mock()
        mock_enhanced_query.processing_strategy = "enhanced_search"
        mock_enhanced_query.enhanced_query = "enhanced version of query"
        mock_enhanced_query.specificity_analysis.specificity_level.value = "high"
        mock_enhanced_query.specificity_analysis.specificity_score = 0.9
        mock_enhanced_query.specificity_analysis.processing_time = 0.5
        mock_enhanced_query.specificity_analysis.reasoning = "Query is highly specific"
        mock_enhanced_query.specificity_analysis.guiding_questions = ["What specific aspect?"]

        mock_hyde_results = Mock()
        mock_hyde_results.results = [{"content": "result1"}, {"content": "result2"}]
        mock_hyde_results.hyde_enhanced = True

        counselor_with_hyde.hyde_processor.three_tier_analysis.return_value = mock_enhanced_query
        counselor_with_hyde.hyde_processor.process_query.return_value = mock_hyde_results

        # Mock MCP client
        counselor_with_hyde.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "complex query requiring analysis",
        }
        counselor_with_hyde.mcp_client.orchestrate_agents.return_value = [
            MCPResponse(
                agent_id="analysis_agent",
                content="Complex analysis completed",
                confidence=0.8,
                processing_time=2.0,
                success=True,
            ),
        ]

        # Process complex query (should trigger HyDE)
        response = await counselor_with_hyde.process_query_with_hyde("provide detailed complex analysis of this system")

        assert response.success is True
        assert response.confidence > 0.8  # Should be boosted by HyDE results
        assert "hyde_integration" in response.metadata
        assert response.metadata["hyde_integration"]["hyde_applied"] is True
        assert response.metadata["hyde_integration"]["processing_strategy"] == "enhanced_search"
        assert response.metadata["hyde_integration"]["hyde_results_count"] == 2

        # Verify HyDE processor was called
        counselor_with_hyde.hyde_processor.three_tier_analysis.assert_called_once()
        counselor_with_hyde.hyde_processor.process_query.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_with_hyde_clarification_needed(self, counselor_with_hyde):
        """Test HyDE processing when clarification is needed."""
        # Mock HyDE processor returning clarification_needed
        mock_enhanced_query = Mock()
        mock_enhanced_query.processing_strategy = "clarification_needed"
        mock_enhanced_query.enhanced_query = "needs clarification"
        mock_enhanced_query.specificity_analysis.specificity_level.value = "low"
        mock_enhanced_query.specificity_analysis.specificity_score = 0.2
        mock_enhanced_query.specificity_analysis.processing_time = 0.3

        counselor_with_hyde.hyde_processor.three_tier_analysis.return_value = mock_enhanced_query

        # Mock MCP client
        counselor_with_hyde.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "complex query",
        }
        counselor_with_hyde.mcp_client.orchestrate_agents.return_value = [
            MCPResponse(
                agent_id="general_agent",
                content="General response",
                confidence=0.7,
                processing_time=1.0,
                success=True,
            ),
        ]

        response = await counselor_with_hyde.process_query_with_hyde("complex query")

        assert response.success is True
        assert "hyde_integration" in response.metadata
        assert response.metadata["hyde_integration"]["processing_strategy"] == "clarification_needed"
        assert response.metadata["hyde_integration"]["hyde_results_count"] == 0

        # HyDE search should not be called when clarification is needed
        counselor_with_hyde.hyde_processor.process_query.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_with_hyde_no_hyde_processor(self):
        """Test HyDE processing without HyDE processor."""
        mock_client = AsyncMock()
        counselor = QueryCounselor(mcp_client=mock_client)  # No HyDE processor

        counselor.mcp_client.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "complex query",
        }
        counselor.mcp_client.orchestrate_agents.return_value = [
            MCPResponse(
                agent_id="general_agent",
                content="Response without HyDE",
                confidence=0.7,
                processing_time=1.0,
                success=True,
            ),
        ]

        response = await counselor.process_query_with_hyde("simple query without hyde triggers")

        assert response.success is True
        assert "hyde_integration" in response.metadata
        assert response.metadata["hyde_integration"]["hyde_applied"] is False
        assert response.metadata["hyde_integration"]["processing_strategy"] == "direct"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_with_hyde_failure(self, counselor_with_hyde):
        """Test HyDE processing failure."""
        counselor_with_hyde.hyde_processor.three_tier_analysis.side_effect = Exception("HyDE failed")

        # Use a complex query that will trigger HyDE recommendation
        response = await counselor_with_hyde.process_query_with_hyde(
            "complex detailed analysis with comprehensive examination",
        )

        assert response.success is False
        assert "Query processing failed" in response.response
        assert response.metadata["error"] is True
        assert "hyde_integration" in response.metadata
        assert response.metadata["hyde_integration"]["error"] == "Processing failed before HyDE integration"


class TestProcessingRecommendations:
    """Test processing recommendation functionality."""

    @pytest.fixture
    def counselor_with_hyde(self):
        """Create QueryCounselor with HyDE processor for testing."""
        mock_hyde = AsyncMock()
        return QueryCounselor(hyde_processor=mock_hyde)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_processing_recommendation_success(self, counselor_with_hyde):
        """Test successful processing recommendation."""
        # Mock HyDE processor
        mock_enhanced_query = Mock()
        mock_enhanced_query.processing_strategy = "enhanced_search"
        mock_enhanced_query.specificity_analysis.specificity_level.value = "high"
        mock_enhanced_query.specificity_analysis.specificity_score = 0.9
        mock_enhanced_query.specificity_analysis.reasoning = "Specific query"
        mock_enhanced_query.specificity_analysis.guiding_questions = ["What aspect?"]

        counselor_with_hyde.hyde_processor.three_tier_analysis.return_value = mock_enhanced_query

        recommendation = await counselor_with_hyde.get_processing_recommendation(
            "analyze this complex system in detail",
        )

        assert "query_analysis" in recommendation
        assert recommendation["query_analysis"]["query_type"] == "analysis_request"
        assert recommendation["query_analysis"]["complexity"] == "complex"
        assert recommendation["query_analysis"]["hyde_recommended"] is True

        assert "hyde_analysis" in recommendation
        assert recommendation["hyde_analysis"]["processing_strategy"] == "enhanced_search"
        assert recommendation["hyde_analysis"]["specificity_level"] == "high"

        assert "agent_recommendations" in recommendation
        assert len(recommendation["agent_recommendations"]) > 0

        assert "processing_strategy" in recommendation
        assert recommendation["processing_strategy"]["use_hyde"] is True
        assert recommendation["processing_strategy"]["expected_complexity"] == "complex"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_processing_recommendation_simple_query(self, counselor_with_hyde):
        """Test processing recommendation for simple query."""
        recommendation = await counselor_with_hyde.get_processing_recommendation("hello")

        assert recommendation["query_analysis"]["query_type"] == "general_query"
        assert recommendation["query_analysis"]["complexity"] == "simple"
        assert recommendation["query_analysis"]["hyde_recommended"] is False
        assert recommendation["hyde_analysis"] is None
        assert recommendation["processing_strategy"]["use_hyde"] is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_processing_recommendation_no_hyde_processor(self):
        """Test processing recommendation without HyDE processor."""
        counselor = QueryCounselor()  # No HyDE processor

        recommendation = await counselor.get_processing_recommendation("complex analysis")

        assert recommendation["query_analysis"]["complexity"] == "complex"
        assert recommendation["query_analysis"]["hyde_recommended"] is True
        assert recommendation["hyde_analysis"] is None  # No HyDE processor available

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_processing_recommendation_failure(self, counselor_with_hyde):
        """Test processing recommendation failure."""
        counselor_with_hyde.hyde_processor.three_tier_analysis.side_effect = Exception("Analysis failed")

        # Use a complex query that triggers HyDE processing
        recommendation = await counselor_with_hyde.get_processing_recommendation(
            "complex detailed analysis with comprehensive examination",
        )

        assert recommendation["error"] is True
        assert recommendation["error_message"] == "Analysis failed"
        assert recommendation["error_type"] == "Exception"


class TestModelSelection:
    """Test model selection for hybrid routing."""

    @pytest.mark.unit
    def test_select_model_for_task_no_hybrid_routing(self):
        """Test model selection when hybrid routing is disabled."""
        counselor = QueryCounselor()  # No hybrid routing

        model = counselor._select_model_for_task(QueryType.CREATE_ENHANCEMENT, "complex")
        assert model is None

    @pytest.mark.unit
    def test_select_model_for_task_no_model_registry(self):
        """Test model selection without model registry."""
        counselor = QueryCounselor()
        counselor.hybrid_routing_enabled = True
        counselor.model_registry = None

        model = counselor._select_model_for_task(QueryType.CREATE_ENHANCEMENT, "complex")
        assert model is None

    @pytest.mark.unit
    def test_select_model_for_task_success(self):
        """Test successful model selection."""
        counselor = QueryCounselor()
        counselor.hybrid_routing_enabled = True
        mock_registry = Mock()
        mock_registry.select_best_model.return_value = "selected-model-id"
        counselor.model_registry = mock_registry

        # Test different query types
        test_cases = [
            (QueryType.CREATE_ENHANCEMENT, "simple", "general", False),
            (QueryType.TEMPLATE_GENERATION, "medium", "general", False),
            (QueryType.ANALYSIS_REQUEST, "complex", "analysis", True),
            (QueryType.IMPLEMENTATION, "complex", "reasoning", True),
            (QueryType.SECURITY, "medium", "analysis", False),
            (QueryType.PERFORMANCE, "simple", "analysis", False),
            (QueryType.UNKNOWN, "medium", "general", False),
        ]

        for query_type, complexity, expected_task_type, expected_premium in test_cases:
            mock_registry.reset_mock()

            model = counselor._select_model_for_task(query_type, complexity)

            assert model == "selected-model-id"
            mock_registry.select_best_model.assert_called_once_with(
                task_type=expected_task_type,
                allow_premium=expected_premium,
            )

    @pytest.mark.unit
    def test_select_model_for_task_registry_failure(self):
        """Test model selection when registry fails."""
        counselor = QueryCounselor()
        counselor.hybrid_routing_enabled = True
        mock_registry = Mock()
        mock_registry.select_best_model.side_effect = Exception("Registry failed")
        counselor.model_registry = mock_registry

        model = counselor._select_model_for_task(QueryType.CREATE_ENHANCEMENT, "complex")
        assert model is None


class TestPerformanceMonitoring:
    """Test performance monitoring integration."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_intent_performance_monitoring(self):
        """Test that analyze_intent works correctly with performance monitoring."""
        counselor = QueryCounselor()

        # Test that the method executes successfully (decorators don't break it)
        result = await counselor.analyze_intent("test query for performance monitoring")

        # Verify basic functionality still works
        assert result.query_type
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        assert result.complexity in ["simple", "medium", "complex"]
        assert isinstance(result.requires_agents, list)
        assert isinstance(result.keywords, list)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_performance_monitoring(self):
        """Test that orchestrate_workflow works correctly with performance monitoring."""
        counselor = QueryCounselor()  # No MCP client
        agents = [Agent(agent_id="test", agent_type="test", capabilities=[])]

        # Test that the method executes successfully (decorators don't break it)
        result = await counselor.orchestrate_workflow(agents, "test")

        # Verify basic functionality still works
        assert isinstance(result, list)
        # Since no MCP client, should return error responses
        for response in result:
            assert hasattr(response, "agent_id")
            assert hasattr(response, "success")
            assert hasattr(response, "processing_time")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_synthesize_response_performance_monitoring(self):
        """Test that synthesize_response works correctly with performance monitoring."""
        counselor = QueryCounselor()

        # Test that the method executes successfully (decorators don't break it)
        result = await counselor.synthesize_response([])

        # Verify basic functionality still works
        assert hasattr(result, "content")
        assert hasattr(result, "confidence")
        assert hasattr(result, "processing_time")
        assert hasattr(result, "query_type")
        assert hasattr(result, "agents_used")
        assert isinstance(result.agents_used, list)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_orchestrate_workflow_query_validation_timeout(self):
        """Test handling of query validation timeout."""
        mock_client = AsyncMock()
        counselor = QueryCounselor(mcp_client=mock_client)

        # Simulate timeout during validation
        mock_client.validate_query.side_effect = TimeoutError("Validation timeout")

        agents = [Agent(agent_id="test", agent_type="test", capabilities=[])]

        with pytest.raises(asyncio.TimeoutError):
            await counselor.orchestrate_workflow(agents, "test query")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_process_query_with_none_query(self):
        """Test processing None as query."""
        counselor = QueryCounselor()

        # This should handle None gracefully through analyze_intent
        response = await counselor.process_query(None)

        assert response.success is False
        assert "Query processing failed" in response.response

    @pytest.mark.unit
    def test_agent_load_factor_boundary_values(self):
        """Test Agent model with boundary load factor values."""
        # Valid boundary values
        agent_min = Agent(
            agent_id="test",
            agent_type="test",
            capabilities=[],
            load_factor=0.0,
        )
        assert agent_min.load_factor == 0.0

        agent_max = Agent(
            agent_id="test",
            agent_type="test",
            capabilities=[],
            load_factor=1.0,
        )
        assert agent_max.load_factor == 1.0

    @pytest.mark.unit
    def test_confidence_boundary_values(self):
        """Test confidence field boundary values in various models."""
        # QueryIntent confidence boundaries
        intent_min = QueryIntent(
            query_type=QueryType.GENERAL_QUERY,
            confidence=0.0,
            complexity="simple",
            original_query="test",
        )
        assert intent_min.confidence == 0.0

        intent_max = QueryIntent(
            query_type=QueryType.GENERAL_QUERY,
            confidence=1.0,
            complexity="simple",
            original_query="test",
        )
        assert intent_max.confidence == 1.0

        # AgentSelection confidence boundaries
        selection_min = AgentSelection(
            primary_agents=["test"],
            reasoning="test",
            confidence=0.0,
        )
        assert selection_min.confidence == 0.0

        selection_max = AgentSelection(
            primary_agents=["test"],
            reasoning="test",
            confidence=1.0,
        )
        assert selection_max.confidence == 1.0


class TestQueryTypeMapping:
    """Test query type mapping and constants."""

    @pytest.mark.unit
    def test_query_type_enum_values(self):
        """Test QueryType enum has expected values."""
        expected_types = {
            "CREATE_ENHANCEMENT",
            "TEMPLATE_GENERATION",
            "ANALYSIS_REQUEST",
            "DOCUMENTATION",
            "GENERAL_QUERY",
            "UNKNOWN",
            "IMPLEMENTATION",
            "SECURITY",
            "PERFORMANCE",
        }

        actual_types = {item.name for item in QueryType}
        assert actual_types == expected_types

    @pytest.mark.unit
    def test_constants_values(self):
        """Test module constants have expected values."""
        from src.core.query_counselor import (
            COMPLEX_QUERY_WORD_THRESHOLD,
            MEDIUM_QUERY_WORD_THRESHOLD,
            MIN_KEYWORD_LENGTH,
        )

        assert COMPLEX_QUERY_WORD_THRESHOLD == 20
        assert MEDIUM_QUERY_WORD_THRESHOLD == 10
        assert MIN_KEYWORD_LENGTH == 3

        # Ensure thresholds make sense
        assert COMPLEX_QUERY_WORD_THRESHOLD > MEDIUM_QUERY_WORD_THRESHOLD
        assert MEDIUM_QUERY_WORD_THRESHOLD > MIN_KEYWORD_LENGTH
