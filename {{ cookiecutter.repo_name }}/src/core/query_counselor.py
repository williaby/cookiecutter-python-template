"""
Query Counselor for PromptCraft-Hybrid.

This module implements the query counseling and routing system that analyzes user queries
and determines the most appropriate processing path within the PromptCraft ecosystem.
It serves as the intelligent dispatcher that understands query intent and routes requests
to the optimal combination of agents and processing pipelines.

The Query Counselor provides:
- Query intent analysis and classification
- Agent selection and routing logic
- Query preprocessing and enhancement
- Multi-agent orchestration coordination
- Response aggregation and post-processing

Architecture:
    The QueryCounselor acts as the central intelligence hub that bridges user queries
    with the appropriate agents and processing pipelines. It integrates with the
    HyDE processor for enhanced retrieval and coordinates with the Zen MCP system
    for robust error handling and resilience.

Key Components:
    - Intent classification using NLP techniques
    - Agent capability matching and selection
    - Query enhancement and preprocessing
    - Multi-agent workflow orchestration
    - Response synthesis and quality assurance

Dependencies:
    - src.agents.registry: For agent discovery and selection
    - src.core.hyde_processor: For enhanced query processing
    - src.core.zen_mcp_error_handling: For resilient execution
    - External AI services: For intent analysis and classification

Called by:
    - src/main.py: Primary FastAPI endpoint handlers
    - Agent orchestration systems
    - Query processing pipelines
    - Multi-agent coordination workflows

Complexity: O(n*m) where n is number of available agents and m is query complexity
"""

import logging
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.core.hyde_processor import HydeProcessor
from src.core.performance_optimizer import (
    cache_query_analysis,
    monitor_performance,
)
from src.mcp_integration.hybrid_router import HybridRouter, RoutingStrategy
from src.mcp_integration.mcp_client import (
    MCPClientInterface,
    MCPError,
)
from src.mcp_integration.mcp_client import (
    Response as MCPResponse,
)
from src.mcp_integration.mcp_client import (
    WorkflowStep as MCPWorkflowStep,
)
from src.mcp_integration.model_registry import ModelRegistry, get_model_registry

# Constants for query complexity analysis
COMPLEX_QUERY_WORD_THRESHOLD = 20
MEDIUM_QUERY_WORD_THRESHOLD = 10
MIN_KEYWORD_LENGTH = 3


class QueryType(str, Enum):
    """Enumeration of supported query types."""

    CREATE_ENHANCEMENT = "create_enhancement"
    TEMPLATE_GENERATION = "template_generation"
    ANALYSIS_REQUEST = "analysis_request"
    DOCUMENTATION = "documentation"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"
    IMPLEMENTATION = "implementation"
    SECURITY = "security"
    PERFORMANCE = "performance"


class QueryIntent(BaseModel):
    """Data model for query intent analysis results."""

    query_type: QueryType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    complexity: str = Field(description="Query complexity: simple, medium, complex")
    requires_agents: list[str] = Field(default_factory=list, description="Required agent types")
    context_needed: bool = Field(default=False, description="Whether additional context is needed")
    hyde_recommended: bool = Field(default=False, description="Whether HyDE processing recommended")
    original_query: str = Field(description="Original query text")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords from query")
    context_requirements: list[str] = Field(default_factory=list, description="Context requirements for the query")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata for hybrid routing")

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        if v not in ["simple", "medium", "complex"]:
            raise ValueError("Complexity must be simple, medium, or complex")
        return v


class Agent(BaseModel):
    """Data model representing an available agent."""

    agent_id: str
    agent_type: str
    capabilities: list[str]
    availability: bool = True
    load_factor: float = Field(ge=0.0, le=1.0, default=0.0)


class AgentSelection(BaseModel):
    """Data model for agent selection results."""

    primary_agents: list[str] = Field(description="Primary agents selected for the query")
    secondary_agents: list[str] = Field(default_factory=list, description="Secondary/fallback agents")
    reasoning: str = Field(description="Reasoning behind the agent selection")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Confidence in selection")


class QueryResponse(BaseModel):
    """Data model for the final query response."""

    response: str = Field(description="The response content")
    agents_used: list[str] = Field(default_factory=list, description="List of agents that contributed")
    processing_time: float = Field(description="Total processing time in seconds")
    success: bool = Field(default=True, description="Whether the query processing was successful")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the response")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WorkflowStep(BaseModel):
    """Individual step in a multi-agent workflow."""

    step_id: str
    agent_id: str
    input_data: dict[str, Any]
    dependencies: list[str] = Field(default_factory=list)
    timeout_seconds: int = 30


class WorkflowResult(BaseModel):
    """Data model for workflow execution results."""

    steps: list[WorkflowStep] = Field(description="List of workflow steps executed")
    final_response: str = Field(description="The final response from the workflow")
    success: bool = Field(description="Whether the workflow completed successfully")
    total_time: float = Field(description="Total execution time in seconds")
    agents_used: list[str] = Field(default_factory=list, description="List of agents involved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional workflow metadata")


class Response(BaseModel):
    """Response from individual agent."""

    agent_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    success: bool = True
    error_message: str | None = None


class FinalResponse(BaseModel):
    """Final synthesized response from QueryCounselor."""

    content: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    query_type: QueryType
    agents_used: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryCounselor:
    """
    Central query processing and agent orchestration system.

    The QueryCounselor analyzes user queries, determines intent, selects appropriate
    agents, orchestrates multi-agent workflows, and synthesizes final responses.
    """

    # Class attribute type annotations
    mcp_client: MCPClientInterface | None
    model_registry: ModelRegistry | None

    def __init__(
        self,
        mcp_client: MCPClientInterface | None = None,
        hyde_processor: HydeProcessor | None = None,
        confidence_threshold: float = 0.7,
        routing_strategy: RoutingStrategy = RoutingStrategy.OPENROUTER_PRIMARY,
        enable_hybrid_routing: bool | None = None,
    ) -> None:
        """Initialize QueryCounselor with optional MCP client and HyDE processor.

        Args:
            mcp_client: MCP client interface (if None and enable_hybrid_routing is True, creates HybridRouter)
            hyde_processor: HyDE processor instance
            confidence_threshold: Confidence threshold for decisions (0.0-1.0)
            routing_strategy: Routing strategy for HybridRouter
            enable_hybrid_routing: Whether to enable hybrid OpenRouter/MCP routing
                                  (None = auto-detect based on mcp_client parameter)
        """
        self.logger = logging.getLogger(__name__)

        # Auto-detect hybrid routing if not explicitly specified
        if enable_hybrid_routing is None:
            # Default to False for backward compatibility
            # Users must explicitly enable hybrid routing
            enable_hybrid_routing = False

        # Initialize MCP client or HybridRouter
        if mcp_client is not None:
            # Use explicitly provided MCP client
            self.mcp_client = mcp_client
            # Check if the provided client is a HybridRouter
            if isinstance(mcp_client, HybridRouter):
                self.model_registry = get_model_registry()
                self.hybrid_routing_enabled = True
            else:
                # Regular MCP client (for testing)
                self.model_registry = None
                self.hybrid_routing_enabled = False
        # Create HybridRouter for production use if enabled
        elif enable_hybrid_routing:
            self.mcp_client = HybridRouter(
                strategy=routing_strategy,
                enable_gradual_rollout=True,
            )
            self.model_registry = get_model_registry()
            self.hybrid_routing_enabled = True
            self.logger.info("Initialized QueryCounselor with HybridRouter")
        else:
            # Fallback to None - will create error responses
            self.mcp_client = None
            self.model_registry = None
            self.hybrid_routing_enabled = False
            self.logger.warning("QueryCounselor initialized without MCP client")

        # Initialize other components
        self.hyde_processor = hyde_processor
        # Clamp confidence threshold to valid range
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))

        # Initialize agent registry mock data
        self._available_agents = [
            Agent(
                agent_id="create_agent",
                agent_type="create",
                capabilities=["prompt_enhancement", "template_generation"],
            ),
            Agent(agent_id="analysis_agent", agent_type="analysis", capabilities=["code_analysis", "documentation"]),
            Agent(agent_id="general_agent", agent_type="general", capabilities=["general_query", "assistance"]),
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

    def _select_model_for_task(self, query_type: QueryType, complexity: str) -> str | None:
        """Select optimal model for query type and complexity.

        Args:
            query_type: Type of query being processed
            complexity: Query complexity level (simple, medium, complex)

        Returns:
            Model ID string if hybrid routing enabled, None otherwise
        """
        if not self.hybrid_routing_enabled or not self.model_registry:
            return None

        # Map query types to task types for model selection
        task_type_mapping = {
            QueryType.CREATE_ENHANCEMENT: "general",
            QueryType.TEMPLATE_GENERATION: "general",
            QueryType.ANALYSIS_REQUEST: "analysis",
            QueryType.DOCUMENTATION: "general",
            QueryType.GENERAL_QUERY: "general",
            QueryType.IMPLEMENTATION: "reasoning",
            QueryType.SECURITY: "analysis",
            QueryType.PERFORMANCE: "analysis",
            QueryType.UNKNOWN: "general",
        }

        task_type = task_type_mapping.get(query_type, "general")

        # Determine if premium models are allowed (for complex queries)
        allow_premium = complexity == "complex"

        try:
            selected_model = self.model_registry.select_best_model(
                task_type=task_type,
                allow_premium=allow_premium,
            )
            self.logger.debug("Selected model %s for %s task", selected_model, query_type)
            return selected_model
        except Exception as e:
            self.logger.warning("Model selection failed: %s", e)
            return None

    @cache_query_analysis
    @monitor_performance("analyze_intent")
    async def analyze_intent(self, query: str) -> QueryIntent:
        """
        Analyze query intent and classify the type of request.

        Args:
            query: User query string

        Returns:
            QueryIntent: Analysis results with type, confidence, and requirements
        """
        start_time = time.time()

        # Input validation - handle empty queries
        if not query or not query.strip():
            # Return UNKNOWN type for empty queries
            intent = QueryIntent(
                query_type=QueryType.UNKNOWN,
                confidence=0.0,
                complexity="simple",
                requires_agents=[],
                context_needed=False,
                hyde_recommended=False,
                original_query=query,
                keywords=[],
                context_requirements=[],
            )

            # Add hybrid routing metadata if enabled
            if self.hybrid_routing_enabled:
                intent.metadata = {
                    "selected_model": None,
                    "hybrid_routing_enabled": True,
                    "task_type": "general",
                    "allow_premium": False,
                }

            return intent

        query = query.strip()

        # Simple rule-based intent analysis (to be enhanced with ML later)
        query_lower = query.lower()

        # Determine query type
        query_type = QueryType.GENERAL_QUERY
        confidence = 0.6
        complexity = "simple"
        requires_agents = ["general_agent"]
        hyde_recommended = False
        context_requirements = []

        if any(keyword in query_lower for keyword in ["create", "generate", "enhance", "improve"]):
            query_type = QueryType.CREATE_ENHANCEMENT
            requires_agents = ["create_agent"]
            confidence = 0.8

        elif any(keyword in query_lower for keyword in ["template", "pattern", "format"]):
            query_type = QueryType.TEMPLATE_GENERATION
            requires_agents = ["create_agent"]
            confidence = 0.85

        elif any(keyword in query_lower for keyword in ["analyze", "review", "examine", "evaluate"]):
            query_type = QueryType.ANALYSIS_REQUEST
            requires_agents = ["analysis_agent"]
            confidence = 0.8
            complexity = "medium"

        elif any(keyword in query_lower for keyword in ["document", "explain", "describe", "how to"]):
            query_type = QueryType.DOCUMENTATION
            requires_agents = ["analysis_agent", "create_agent"]
            confidence = 0.75
            complexity = "medium"

        elif any(keyword in query_lower for keyword in ["implement", "build", "develop", "code"]):
            query_type = QueryType.IMPLEMENTATION
            requires_agents = ["create_agent"]
            confidence = 0.8
            context_requirements = ["python"]

        elif any(keyword in query_lower for keyword in ["security", "secure", "auth", "attack", "vulnerability"]):
            query_type = QueryType.SECURITY
            requires_agents = ["security_agent"]
            confidence = 0.85
            context_requirements = ["security", "web"]

        elif any(keyword in query_lower for keyword in ["performance", "optimize", "speed", "fast", "slow"]):
            query_type = QueryType.PERFORMANCE
            requires_agents = ["performance_agent"]
            confidence = 0.8
            context_requirements = ["performance"]

        # Determine complexity
        if len(query.split()) > COMPLEX_QUERY_WORD_THRESHOLD or "complex" in query_lower or "detailed" in query_lower:
            complexity = "complex"
            hyde_recommended = True
        elif len(query.split()) > MEDIUM_QUERY_WORD_THRESHOLD or any(
            keyword in query_lower for keyword in ["analysis", "compare", "evaluate"]
        ):
            complexity = "medium"

        # Context needed for complex queries
        context_needed = complexity in ["medium", "complex"] or query_type == QueryType.ANALYSIS_REQUEST

        # Extract simple keywords (basic word splitting)
        keywords = [word.lower() for word in query.split() if len(word) > MIN_KEYWORD_LENGTH]

        processing_time = time.time() - start_time

        # Select optimal model for this query type and complexity
        selected_model = self._select_model_for_task(query_type, complexity)

        self.logger.info(
            "Intent analysis completed in %.3fs: %s (confidence: %.2f, model: %s)",
            processing_time,
            query_type,
            confidence,
            selected_model or "default",
        )

        # Create QueryIntent with model selection metadata
        intent = QueryIntent(
            query_type=query_type,
            confidence=confidence,
            complexity=complexity,
            requires_agents=requires_agents,
            context_needed=context_needed,
            hyde_recommended=hyde_recommended,
            original_query=query,
            keywords=keywords,
            context_requirements=context_requirements,
        )

        # Add hybrid routing metadata if enabled
        if self.hybrid_routing_enabled:
            # Map query types to task types for model selection
            task_type_mapping = {
                QueryType.CREATE_ENHANCEMENT: "general",
                QueryType.TEMPLATE_GENERATION: "general",
                QueryType.ANALYSIS_REQUEST: "analysis",
                QueryType.DOCUMENTATION: "general",
                QueryType.GENERAL_QUERY: "general",
                QueryType.IMPLEMENTATION: "reasoning",
                QueryType.SECURITY: "analysis",
                QueryType.PERFORMANCE: "analysis",
                QueryType.UNKNOWN: "general",
            }

            # Store model selection in a way that can be accessed by orchestration
            intent.metadata = {
                "selected_model": selected_model,
                "hybrid_routing_enabled": self.hybrid_routing_enabled,
                "task_type": task_type_mapping.get(query_type, "general"),
                "allow_premium": complexity == "complex",
            }

        return intent

    async def select_agents(self, intent: QueryIntent) -> AgentSelection:
        """
        Select appropriate agents based on query intent.

        Args:
            intent: QueryIntent from analyze_intent()

        Returns:
            AgentSelection: Selected agents for processing
        """
        selected_agents = []

        for agent_id in intent.requires_agents:
            # Find agent by ID
            agent = next((a for a in self._available_agents if a.agent_id == agent_id), None)
            if agent and agent.availability:
                selected_agents.append(agent)

        # Fallback to general agent if no specific agents found
        if not selected_agents:
            general_agent = next((a for a in self._available_agents if a.agent_type == "general"), None)
            if general_agent:
                selected_agents.append(general_agent)

        self.logger.info(
            "Selected %d agents: %s",
            len(selected_agents),
            [a.agent_id for a in selected_agents],
        )

        # Create AgentSelection object
        primary_agents = [a.agent_id for a in selected_agents[:2]]  # First 2 as primary
        secondary_agents = [a.agent_id for a in selected_agents[2:]]  # Rest as secondary

        return AgentSelection(
            primary_agents=primary_agents,
            secondary_agents=secondary_agents,
            reasoning=f"Selected {len(selected_agents)} agents based on query requirements: {intent.requires_agents}",
            confidence=intent.confidence,
        )

    @monitor_performance("orchestrate_workflow")
    async def orchestrate_workflow(
        self,
        agents: list[Agent],
        query: str,
        intent: QueryIntent | None = None,
    ) -> list[MCPResponse]:
        """
        Orchestrate multi-agent workflow for query processing.

        Args:
            agents: List of selected agents
            query: Original user query
            intent: Optional query intent for model routing

        Returns:
            List[Response]: Responses from all agents
        """
        start_time = time.time()

        # Validate query first if MCP client is available
        if self.mcp_client is not None:
            validation_result = await self.mcp_client.validate_query(query)
            if not validation_result["is_valid"]:
                raise ValueError(f"Query validation failed: {validation_result.get('potential_issues', [])}")
            sanitized_query = validation_result["sanitized_query"]
        else:
            # If no MCP client, use query as-is
            sanitized_query = query

        # Create workflow steps with enhanced metadata for hybrid routing
        workflow_steps = []
        for i, agent in enumerate(agents):
            step_input = {
                "query": sanitized_query,
                "agent_type": agent.agent_type,
                "capabilities": agent.capabilities,
            }

            # Add model selection metadata if available
            if intent and intent.metadata:
                step_input.update(
                    {
                        "selected_model": intent.metadata.get("selected_model"),
                        "task_type": intent.metadata.get("task_type", "general"),
                        "complexity": intent.complexity,
                        "query_type": intent.query_type.value,
                        "hybrid_routing_enabled": intent.metadata.get("hybrid_routing_enabled", False),
                    },
                )

            step = MCPWorkflowStep(
                step_id=f"step_{i}",
                agent_id=agent.agent_id,
                input_data=step_input,
            )
            workflow_steps.append(step)

        # Execute workflow through MCP if available
        if self.mcp_client is not None:
            try:
                responses = await self.mcp_client.orchestrate_agents(workflow_steps)

                processing_time = time.time() - start_time
                self.logger.info("Workflow orchestration completed in %.3fs", processing_time)

                return responses

            except MCPError as e:
                self.logger.error("MCP orchestration failed: %s (type: %s)", str(e), e.error_type)
                # Return error responses for graceful fallback
                error_responses = []
                for agent in agents:
                    error_response = MCPResponse(
                        agent_id=agent.agent_id,
                        content=f"Agent {agent.agent_id} unavailable due to MCP error",
                        confidence=0.0,
                        processing_time=0.0,
                        success=False,
                        error_message=f"{e.error_type}: {e!s}",
                    )
                    error_responses.append(error_response)
                return error_responses
            except Exception as e:
                self.logger.error("Workflow orchestration failed: %s", str(e))
                # Return error responses for graceful fallback
                error_responses = []
                for agent in agents:
                    error_response = MCPResponse(
                        agent_id=agent.agent_id,
                        content=f"Agent {agent.agent_id} unavailable",
                        confidence=0.0,
                        processing_time=0.0,
                        success=False,
                        error_message=str(e),
                    )
                    error_responses.append(error_response)
                return error_responses
        else:
            # If no MCP client, return error responses for all agents
            self.logger.warning("No MCP client available for workflow orchestration")
            error_responses = []
            for agent in agents:
                error_response = MCPResponse(
                    agent_id=agent.agent_id,
                    content=f"Agent {agent.agent_id} unavailable - no MCP client",
                    confidence=0.0,
                    processing_time=0.0,
                    success=False,
                    error_message="No MCP client available",
                )
                error_responses.append(error_response)
            return error_responses

    @monitor_performance("synthesize_response")
    async def synthesize_response(self, agent_outputs: list[MCPResponse]) -> FinalResponse:
        """
        Synthesize final response from multiple agent outputs.

        Args:
            agent_outputs: List of responses from agents

        Returns:
            FinalResponse: Synthesized final response
        """
        start_time = time.time()

        # Filter successful responses
        successful_responses = [r for r in agent_outputs if r.success]

        if not successful_responses:
            # All agents failed - return error response
            error_content = "Unable to process query - all agents unavailable"
            if agent_outputs:
                error_messages = [r.error_message for r in agent_outputs if r.error_message]
                if error_messages:
                    error_content += f": {'; '.join(error_messages)}"

            return FinalResponse(
                content=error_content,
                confidence=0.0,
                processing_time=time.time() - start_time,
                query_type=QueryType.GENERAL_QUERY,
                agents_used=[],
                metadata={"error": True, "failed_agents": len(agent_outputs)},
            )

        # Synthesize content from successful responses
        content_parts = []
        sources = []
        total_confidence = 0.0
        agents_used = []

        for response in successful_responses:
            content_parts.append(response.content)
            sources.append(response.agent_id)
            total_confidence += response.confidence
            agents_used.append(response.agent_id)

        # Calculate average confidence
        avg_confidence = total_confidence / len(successful_responses)

        # Combine content
        if len(content_parts) == 1:
            final_content = content_parts[0]
        else:
            final_content = "Combined response:\n\n" + "\n\n---\n\n".join(content_parts)

        processing_time = time.time() - start_time

        # Determine query type from metadata (simplified)
        query_type = QueryType.GENERAL_QUERY
        if successful_responses and "query_type" in successful_responses[0].metadata:
            query_type = successful_responses[0].metadata["query_type"]

        self.logger.info("Response synthesis completed in %.3fs", processing_time)

        return FinalResponse(
            content=final_content,
            sources=sources,
            confidence=avg_confidence,
            processing_time=processing_time,
            query_type=query_type,
            agents_used=agents_used,
            metadata={
                "synthesis_method": "simple_concatenation",
                "successful_agents": len(successful_responses),
                "failed_agents": len(agent_outputs) - len(successful_responses),
            },
        )

    async def process_query(self, query: str) -> QueryResponse:
        """
        Main entry point for query processing.

        This method provides the standard query processing pipeline without
        automatic HyDE integration, suitable for most use cases.

        Args:
            query: User query string

        Returns:
            QueryResponse: Processed query response
        """
        start_time = time.time()

        try:
            # Step 1: Intent analysis
            intent = await self.analyze_intent(query)

            # Step 2: Agent selection and orchestration
            agent_selection = await self.select_agents(intent)
            # Convert AgentSelection to list of Agent objects for orchestration
            selected_agents = []
            for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                agent = next((a for a in self._available_agents if a.agent_id == agent_id), None)
                if agent:
                    selected_agents.append(agent)
            agent_responses = await self.orchestrate_workflow(selected_agents, query, intent)

            # Step 3: Response synthesis
            final_response = await self.synthesize_response(agent_responses)

            # Update processing time
            total_processing_time = time.time() - start_time

            # Convert FinalResponse to QueryResponse
            return QueryResponse(
                response=final_response.content,
                agents_used=final_response.agents_used,
                processing_time=total_processing_time,
                success=final_response.confidence > 0.0,
                confidence=final_response.confidence,
                metadata=final_response.metadata,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error("Query processing failed: %s", str(e))

            return QueryResponse(
                response=f"Query processing failed: {e!s}",
                agents_used=[],
                processing_time=processing_time,
                success=False,
                confidence=0.0,
                metadata={
                    "error": True,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def process_query_with_hyde(self, query: str) -> QueryResponse:
        """
        Complete query processing pipeline with HyDE enhancement integration.

        This method implements the full Week 1 integration workflow between
        QueryCounselor and HydeProcessor, providing end-to-end query processing
        with enhanced retrieval capabilities.

        Args:
            query: User query string

        Returns:
            FinalResponse: Comprehensive response with HyDE enhancement metadata

        Raises:
            ValueError: If query is invalid
            Exception: For processing errors
        """
        start_time = time.time()

        try:
            # Step 1: Intent analysis
            self.logger.info("Starting enhanced query processing with HyDE integration")
            intent = await self.analyze_intent(query)

            # Step 2: HyDE processing if recommended
            enhanced_query = None
            hyde_results = None
            if intent.hyde_recommended and self.hyde_processor is not None:
                self.logger.info("Applying HyDE enhancement for complex query")
                enhanced_query = await self.hyde_processor.three_tier_analysis(query)

                # Only proceed with HyDE search if strategy allows it
                if enhanced_query.processing_strategy != "clarification_needed":
                    hyde_results = await self.hyde_processor.process_query(query)

            # Step 3: Agent selection and orchestration
            agent_selection = await self.select_agents(intent)
            # Convert AgentSelection to list of Agent objects for orchestration
            selected_agents = []
            for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                agent = next((a for a in self._available_agents if a.agent_id == agent_id), None)
                if agent:
                    selected_agents.append(agent)

            # Use enhanced query for agent processing if available
            processing_query = enhanced_query.enhanced_query if enhanced_query else query
            agent_responses = await self.orchestrate_workflow(selected_agents, processing_query, intent)

            # Step 4: Enhanced response synthesis with HyDE context
            final_response = await self.synthesize_response(agent_responses)

            # Step 5: Enhance final response with HyDE metadata
            total_processing_time = time.time() - start_time

            # Augment metadata with HyDE information
            enhanced_metadata = final_response.metadata.copy()
            enhanced_metadata.update(
                {
                    "hyde_integration": {
                        "hyde_applied": intent.hyde_recommended,
                        "processing_strategy": enhanced_query.processing_strategy if enhanced_query else "direct",
                        "specificity_level": (
                            enhanced_query.specificity_analysis.specificity_level.value if enhanced_query else "unknown"
                        ),
                        "specificity_score": (
                            enhanced_query.specificity_analysis.specificity_score if enhanced_query else 0.0
                        ),
                        "hyde_results_count": len(hyde_results.results) if hyde_results else 0,
                        "hyde_enhanced_search": hyde_results.hyde_enhanced if hyde_results else False,
                    },
                    "query_analysis": {
                        "original_query": query,
                        "enhanced_query": enhanced_query.enhanced_query if enhanced_query else query,
                        "intent_confidence": intent.confidence,
                        "complexity": intent.complexity,
                        "context_needed": intent.context_needed,
                    },
                    "performance": {
                        "total_processing_time": total_processing_time,
                        "hyde_processing_time": (
                            enhanced_query.specificity_analysis.processing_time if enhanced_query else 0.0
                        ),
                        "agent_processing_time": final_response.processing_time,
                    },
                },
            )

            # Enhance confidence based on HyDE results
            enhanced_confidence = final_response.confidence
            if hyde_results and hyde_results.results:
                # Boost confidence if HyDE found relevant results
                hyde_confidence_boost = min(0.1, len(hyde_results.results) * 0.02)
                enhanced_confidence = min(1.0, enhanced_confidence + hyde_confidence_boost)

            # Create enhanced final response
            enhanced_final_response = FinalResponse(
                content=final_response.content,
                sources=final_response.sources,
                confidence=enhanced_confidence,
                processing_time=total_processing_time,
                query_type=final_response.query_type,
                agents_used=final_response.agents_used,
                metadata=enhanced_metadata,
            )

            self.logger.info(
                "Enhanced query processing completed in %.3fs (HyDE: %s, Strategy: %s)",
                total_processing_time,
                "applied" if intent.hyde_recommended else "not applied",
                enhanced_query.processing_strategy if enhanced_query else "direct",
            )

            # Convert FinalResponse to QueryResponse
            return QueryResponse(
                response=enhanced_final_response.content,
                agents_used=enhanced_final_response.agents_used,
                processing_time=enhanced_final_response.processing_time,
                success=enhanced_final_response.confidence > 0.0,
                confidence=enhanced_final_response.confidence,
                metadata=enhanced_final_response.metadata,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error("Enhanced query processing failed: %s", str(e))

            # Return error response with diagnostic information
            return QueryResponse(
                response=f"Query processing failed: {e!s}",
                agents_used=[],
                processing_time=processing_time,
                success=False,
                confidence=0.0,
                metadata={
                    "error": True,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "hyde_integration": {"error": "Processing failed before HyDE integration"},
                },
            )

    async def get_processing_recommendation(self, query: str) -> dict[str, Any]:
        """
        Get processing recommendations for a query without full execution.

        This method provides analysis and recommendations for query processing
        strategy without executing the full pipeline, useful for performance
        planning and debugging.

        Args:
            query: User query string

        Returns:
            Dict containing processing recommendations and analysis
        """
        try:
            # Analyze intent
            intent = await self.analyze_intent(query)

            # Get HyDE analysis if recommended
            hyde_analysis = None
            if intent.hyde_recommended and self.hyde_processor is not None:
                enhanced_query = await self.hyde_processor.three_tier_analysis(query)
                hyde_analysis = {
                    "processing_strategy": enhanced_query.processing_strategy,
                    "specificity_level": enhanced_query.specificity_analysis.specificity_level.value,
                    "specificity_score": enhanced_query.specificity_analysis.specificity_score,
                    "reasoning": enhanced_query.specificity_analysis.reasoning,
                    "guiding_questions": enhanced_query.specificity_analysis.guiding_questions,
                }

            # Get agent recommendations
            agent_selection = await self.select_agents(intent)
            # Convert AgentSelection to list of Agent objects for recommendations
            selected_agents = []
            for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                agent = next((a for a in self._available_agents if a.agent_id == agent_id), None)
                if agent:
                    selected_agents.append(agent)

            return {
                "query_analysis": {
                    "query_type": intent.query_type.value,
                    "confidence": intent.confidence,
                    "complexity": intent.complexity,
                    "context_needed": intent.context_needed,
                    "hyde_recommended": intent.hyde_recommended,
                },
                "hyde_analysis": hyde_analysis,
                "agent_recommendations": [
                    {
                        "agent_id": agent.agent_id,
                        "agent_type": agent.agent_type,
                        "capabilities": agent.capabilities,
                        "availability": agent.availability,
                    }
                    for agent in selected_agents
                ],
                "processing_strategy": {
                    "use_hyde": intent.hyde_recommended,
                    "expected_complexity": intent.complexity,
                    "estimated_agents": len(selected_agents),
                    "recommended_timeout": 30 if intent.complexity == "complex" else 15,
                },
            }

        except Exception as e:
            return {
                "error": True,
                "error_message": str(e),
                "error_type": type(e).__name__,
            }
