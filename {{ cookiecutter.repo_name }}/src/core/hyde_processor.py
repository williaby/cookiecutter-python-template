"""
HyDE (Hypothetical Document Embeddings) Processor for PromptCraft-Hybrid.

This module implements the HyDE-enhanced retrieval system that improves semantic search
accuracy through three-tier query analysis and hypothetical document generation.
HyDE enhances traditional RAG (Retrieval-Augmented Generation) by generating hypothetical
documents that better match the embedding space of relevant knowledge.

The HyDE processor provides:
- Three-tier query analysis system
- Hypothetical document generation
- Enhanced semantic embedding creation
- Improved retrieval accuracy
- Multi-modal query processing

Architecture:
    The HyDE system processes queries through three progressive tiers:
    1. Direct query embedding for simple semantic matches
    2. Query expansion and reformulation for complex queries
    3. Hypothetical document generation for advanced retrieval

    This tiered approach ensures optimal retrieval performance across different
    query types and complexity levels.

Key Components:
    - Query classification and tier selection
    - Hypothetical document generation using AI models
    - Enhanced embedding creation and indexing
    - Multi-tier retrieval strategy implementation
    - Result ranking and relevance scoring

Dependencies:
    - External AI services: For document generation and embedding
    - src.config.settings: For HyDE configuration parameters
    - Qdrant vector database: For enhanced semantic search
    - src.core.zen_mcp_error_handling: For resilient processing

Called by:
    - src.core.query_counselor: For enhanced query processing
    - Agent implementations: For improved knowledge retrieval
    - RAG pipeline components: For semantic search enhancement
    - Knowledge ingestion systems: For embedding optimization

Complexity: O(k*n) where k is number of tiers and n is query/document complexity
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.core.performance_optimizer import (
    cache_hyde_processing,
    monitor_performance,
)
from src.core.vector_store import (
    AbstractVectorStore,
    EnhancedMockVectorStore,
    SearchParameters,
    SearchStrategy,
    VectorStoreFactory,
)
from src.core.vector_store import (
    SearchResult as VectorSearchResult,
)
from src.mcp_integration.hybrid_router import HybridRouter, RoutingStrategy
from src.mcp_integration.mcp_client import WorkflowStep
from src.mcp_integration.model_registry import ModelRegistry, get_model_registry

# Constants for HyDE processing thresholds (per hyde-processor.md)
HIGH_SPECIFICITY_THRESHOLD = 85
LOW_SPECIFICITY_THRESHOLD = 40
MAX_HYPOTHETICAL_DOCS = 3
DEFAULT_EMBEDDING_DIMENSIONS = 384

# Constants for query analysis
HIGH_WORD_COUNT_THRESHOLD = 15
MEDIUM_WORD_COUNT_THRESHOLD = 8


class SpecificityLevel(str, Enum):
    """Query specificity levels for HyDE processing."""

    HIGH = "high"  # Score > 85 - Skip HyDE, direct retrieval
    MEDIUM = "medium"  # Score 40-85 - Apply Standard HyDE
    LOW = "low"  # Score < 40 - Return clarifying questions


class ProcessingStrategy(str, Enum):
    """Processing strategies for HyDE query handling."""

    DIRECT = "direct_retrieval"
    ENHANCED = "standard_hyde"
    HYPOTHETICAL = "clarification_needed"


class QueryAnalysis(BaseModel):
    """Query analysis results from the Query Counselor."""

    original_query: str = Field(description="Original query string")
    specificity_score: float = Field(ge=0.0, le=100.0, description="Specificity score 0-100")
    specificity_level: SpecificityLevel
    enhanced_query: str = Field(description="Enhanced query for processing")
    processing_strategy: str = Field(description="Processing strategy to use")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")
    reasoning: str = Field(default="", description="Brief explanation of the scoring")
    guiding_questions: list[str] = Field(default_factory=list, description="Questions for low-specificity queries")
    processing_time: float = Field(default=0.0, description="Analysis processing time")


class HypotheticalDocument(BaseModel):
    """Generated hypothetical document for enhanced retrieval."""

    content: str = Field(description="Generated document content")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    embedding: list[float] = Field(default_factory=list, description="Document embedding vector")
    generation_method: str = Field(default="standard", description="Generation strategy used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HydeSearchResult(BaseModel):
    """Search result from vector database."""

    document_id: str
    content: str
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(default="unknown", description="Result source")


class RankedResults(BaseModel):
    """Ranked and filtered search results."""

    results: list[VectorSearchResult]
    total_found: int
    processing_time: float
    ranking_method: str = Field(default="similarity", description="Ranking strategy used")
    hyde_enhanced: bool = Field(default=False, description="Whether HyDE was applied")


class EnhancedQuery(BaseModel):
    """Enhanced query with HyDE processing applied."""

    original_query: str
    enhanced_query: str = Field(description="Processed query for search")
    embeddings: list[list[float]] = Field(default_factory=list, description="Query embeddings")
    hypothetical_docs: list[HypotheticalDocument] = Field(
        default_factory=list,
        description="Generated hypothetical documents",
    )
    specificity_analysis: QueryAnalysis
    processing_strategy: str = Field(description="HyDE strategy applied")


# MockVectorStore removed - now using real AbstractVectorStore implementations
# through VectorStoreFactory which provides both EnhancedMockVectorStore and QdrantVectorStore


class MockQueryCounselor:
    """Mock Query Counselor for HyDE analysis."""

    async def analyze_query_specificity(self, query: str) -> QueryAnalysis:
        """Mock query specificity analysis."""
        start_time = time.time()

        # Simple rule-based specificity scoring
        query_lower = query.lower()
        word_count = len(query.split())
        specificity_score = 50.0  # Default medium specificity

        # Increase score for specific technical terms
        technical_terms = ["implement", "configure", "install", "error", "debug", "optimize"]
        if any(term in query_lower for term in technical_terms):
            specificity_score += 20

        # Increase score for longer, more detailed queries
        if word_count > HIGH_WORD_COUNT_THRESHOLD:
            specificity_score += 15
        elif word_count > MEDIUM_WORD_COUNT_THRESHOLD:
            specificity_score += 10

        # Decrease score for vague queries
        vague_terms = ["help", "how", "what", "general", "basic", "simple"]
        if any(term in query_lower for term in vague_terms):
            specificity_score -= 10

        # Ensure score is within bounds
        specificity_score = max(0.0, min(100.0, specificity_score))

        # Determine specificity level
        if specificity_score > HIGH_SPECIFICITY_THRESHOLD:
            level = SpecificityLevel.HIGH
        elif specificity_score >= LOW_SPECIFICITY_THRESHOLD:
            level = SpecificityLevel.MEDIUM
        else:
            level = SpecificityLevel.LOW

        # Generate guiding questions for low specificity
        guiding_questions = []
        if level == SpecificityLevel.LOW:
            guiding_questions = [
                "What specific technology or framework are you working with?",
                "What is the exact problem you're trying to solve?",
                "What have you already tried?",
            ]

        processing_time = time.time() - start_time

        return QueryAnalysis(
            original_query=query,
            specificity_score=specificity_score,
            specificity_level=level,
            enhanced_query=query,
            processing_strategy="mock_analysis",
            confidence=specificity_score / 100.0,
            reasoning=f"Score based on technical terms, query length ({word_count} words), and specificity indicators",
            guiding_questions=guiding_questions,
            processing_time=processing_time,
        )


@dataclass
class HydeProcessorConfig:
    """Configuration for HydeProcessor initialization."""

    vector_store: "AbstractVectorStore | None" = None
    query_counselor: "MockQueryCounselor | None" = None
    specificity_threshold_high: float | None = None
    specificity_threshold_low: float | None = None
    hybrid_router: "HybridRouter | None" = None
    enable_openrouter: bool = True


class HydeProcessor:
    """
    HyDE (Hypothetical Document Embeddings) processor for enhanced retrieval.

    Implements three-tier query analysis and processing strategy:
    - High specificity: Direct retrieval without HyDE
    - Medium specificity: Standard HyDE with hypothetical document generation
    - Low specificity: Return clarifying questions to user
    """

    # Class attribute type annotations
    hybrid_router: HybridRouter | None
    model_registry: ModelRegistry | None

    def __init__(
        self,
        config: HydeProcessorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HydeProcessor with optional dependencies.

        Args:
            config: Configuration object with all dependencies and settings
            **kwargs: Additional configuration parameters (for backward compatibility)
        """
        self.logger = logging.getLogger(__name__)

        # Use config or create from kwargs for backward compatibility
        if config is None:
            config = HydeProcessorConfig(**kwargs)

        # Initialize vector store using factory if not provided
        if config.vector_store is None:
            try:
                settings = get_settings()
                vector_config = {
                    "type": settings.vector_store_type,
                    "host": settings.qdrant_host,
                    "port": settings.qdrant_port,
                    "timeout": settings.qdrant_timeout,
                    "api_key": settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
                    "simulate_latency": settings.environment == "dev",
                    "error_rate": 0.05 if settings.environment == "dev" else 0.0,
                }
                self.vector_store = VectorStoreFactory.create_vector_store(vector_config)
                self.logger.info("Initialized vector store: %s", type(self.vector_store).__name__)
            except Exception as e:
                self.logger.warning("Failed to create real vector store, using mock: %s", str(e))
                # Fallback to enhanced mock
                self.vector_store = EnhancedMockVectorStore(
                    {"simulate_latency": True, "error_rate": 0.0, "base_latency": 0.05},
                )
        else:
            self.vector_store = config.vector_store

        self.query_counselor = config.query_counselor or MockQueryCounselor()

        # Store thresholds
        self.specificity_threshold_high = config.specificity_threshold_high or HIGH_SPECIFICITY_THRESHOLD
        self.specificity_threshold_low = config.specificity_threshold_low or LOW_SPECIFICITY_THRESHOLD

        # Initialize OpenRouter integration
        self.enable_openrouter = config.enable_openrouter
        if config.enable_openrouter:
            self.hybrid_router = config.hybrid_router or HybridRouter(
                strategy=RoutingStrategy.OPENROUTER_PRIMARY,
                enable_gradual_rollout=True,
            )
            self.model_registry = get_model_registry()
            self.logger.info("Initialized HydeProcessor with OpenRouter integration")
        else:
            self.hybrid_router = None
            self.model_registry = None

        # Initialize vector store connection
        self._vector_store_connected = False

    @cache_hyde_processing
    @monitor_performance("three_tier_analysis")
    async def three_tier_analysis(self, query: str) -> EnhancedQuery:
        """
        Perform three-tier analysis and determine processing strategy.

        Args:
            query: User query string

        Returns:
            EnhancedQuery: Analysis results with processing strategy
        """
        start_time = time.time()

        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        # Analyze query specificity
        analysis = await self.query_counselor.analyze_query_specificity(query)

        # Determine processing strategy based on specificity
        if analysis.specificity_level == SpecificityLevel.HIGH:
            strategy = "direct_retrieval"
            enhanced_query = query
            hypothetical_docs = []
        elif analysis.specificity_level == SpecificityLevel.MEDIUM:
            strategy = "standard_hyde"
            enhanced_query = query
            hypothetical_docs = await self._generate_hypothetical_docs(query)
        else:  # LOW specificity
            strategy = "clarification_needed"
            enhanced_query = query
            hypothetical_docs = []

        processing_time = time.time() - start_time

        self.logger.info(
            "Three-tier analysis completed in %.3fs: %s strategy for specificity %.1f",
            processing_time,
            strategy,
            analysis.specificity_score,
        )

        return EnhancedQuery(
            original_query=query,
            enhanced_query=enhanced_query,
            embeddings=[],  # Will be populated by embedding service
            hypothetical_docs=hypothetical_docs,
            specificity_analysis=analysis,
            processing_strategy=strategy,
        )

    async def generate_hypothetical_docs(self, query: str) -> list[HypotheticalDocument]:
        """
        Generate hypothetical documents for medium-specificity queries.

        Args:
            query: User query string

        Returns:
            List[HypotheticalDocument]: Generated hypothetical documents
        """
        return await self._generate_hypothetical_docs(query)

    async def _generate_hypothetical_docs(self, query: str) -> list[HypotheticalDocument]:
        """Internal method to generate hypothetical documents."""
        start_time = time.time()

        # Try to use OpenRouter for document generation if available
        if self.enable_openrouter and self.hybrid_router and self.model_registry:
            try:
                return await self._generate_hypothetical_docs_with_openrouter(query)
            except Exception as e:
                self.logger.warning("OpenRouter document generation failed, falling back to mock: %s", str(e))

        # Fallback to mock document generation
        docs = []

        # Generate 1-3 hypothetical documents based on query
        doc_templates = [
            f"A comprehensive guide to {query} with step-by-step instructions and best practices.",
            f"Technical documentation explaining {query} with code examples and troubleshooting tips.",
            f"Expert analysis of {query} including common pitfalls and recommended solutions.",
        ]

        for i, template in enumerate(doc_templates[:MAX_HYPOTHETICAL_DOCS]):
            # Simulate document generation processing time
            await asyncio.sleep(0.02)

            # Create mock embedding (to be replaced with real embedding service)
            mock_embedding = [0.1 * (i + 1)] * DEFAULT_EMBEDDING_DIMENSIONS

            doc = HypotheticalDocument(
                content=template,
                relevance_score=0.9 - (i * 0.1),  # Decreasing relevance
                embedding=mock_embedding,
                generation_method="mock_template",
                metadata={
                    "generated_at": time.time(),
                    "query_hash": hash(query),
                    "doc_index": i,
                },
            )
            docs.append(doc)

        processing_time = time.time() - start_time
        self.logger.info("Generated %d hypothetical documents in %.3fs", len(docs), processing_time)

        return docs

    async def _generate_hypothetical_docs_with_openrouter(self, query: str) -> list[HypotheticalDocument]:
        """Generate hypothetical documents using OpenRouter API."""
        start_time = time.time()

        # Select appropriate model for document generation
        if not self.model_registry:
            raise ValueError("Model registry not available for HyDE processing")

        selected_model = self.model_registry.select_best_model(
            task_type="general",
            allow_premium=False,  # Use free models for HyDE
        )

        # Create workflow steps for document generation
        workflow_steps = []
        for i in range(MAX_HYPOTHETICAL_DOCS):
            step = WorkflowStep(
                step_id=f"hyde_doc_{i}",
                agent_id="hyde_generator",
                input_data={
                    "query": query,
                    "selected_model": selected_model,
                    "task_type": "document_generation",
                    "document_type": ["guide", "technical_doc", "expert_analysis"][i],
                    "generation_prompt": self._create_hyde_prompt(query, i),
                },
                timeout_seconds=30,
            )
            workflow_steps.append(step)

        # Execute document generation through HybridRouter
        if not self.hybrid_router:
            raise ValueError("Hybrid router not available for HyDE processing")

        responses = await self.hybrid_router.orchestrate_agents(workflow_steps)

        # Convert responses to HypotheticalDocument objects
        docs = []
        for i, response in enumerate(responses):
            if response.success and response.content:
                # Create embedding from the generated content
                embedding = self._create_embeddings(response.content)

                doc = HypotheticalDocument(
                    content=response.content,
                    relevance_score=response.confidence,
                    embedding=embedding,
                    generation_method="openrouter",
                    metadata={
                        "generated_at": time.time(),
                        "query_hash": hash(query),
                        "doc_index": i,
                        "model_used": selected_model,
                        "processing_time": response.processing_time,
                    },
                )
                docs.append(doc)

        processing_time = time.time() - start_time
        self.logger.info(
            "Generated %d hypothetical documents via OpenRouter in %.3fs (model: %s)",
            len(docs),
            processing_time,
            selected_model,
        )

        return docs

    def _create_hyde_prompt(self, query: str, doc_index: int) -> str:
        """Create specialized prompt for hypothetical document generation."""
        prompt_templates = [
            f"Write a comprehensive guide that would answer the query '{query}'. "
            f"Include step-by-step instructions, best practices, and practical examples. "
            f"Format as a well-structured technical document.",
            f"Create technical documentation that explains '{query}' with detailed code examples, "
            f"troubleshooting tips, and common implementation patterns. "
            f"Focus on practical implementation details.",
            f"Provide expert analysis of '{query}' including common pitfalls, "
            f"recommended solutions, performance considerations, and real-world examples. "
            f"Write from the perspective of an experienced practitioner.",
        ]

        return prompt_templates[doc_index % len(prompt_templates)]

    def _analyze_query_specificity(self, query: str) -> float:
        """
        Analyze query specificity and return a score.

        Args:
            query: User query string

        Returns:
            float: Specificity score between 0-100
        """
        if not query or not query.strip():
            return 0.0

        query_lower = query.strip().lower()
        word_count = len(query.split())
        specificity_score = 50.0  # Default medium specificity

        # Increase score for specific technical terms
        technical_terms = ["implement", "configure", "install", "error", "debug", "optimize"]
        if any(term in query_lower for term in technical_terms):
            specificity_score += 20

        # Increase score for longer, more detailed queries
        if word_count > HIGH_WORD_COUNT_THRESHOLD:
            specificity_score += 15
        elif word_count > MEDIUM_WORD_COUNT_THRESHOLD:
            specificity_score += 10

        # Decrease score for vague queries
        vague_terms = ["help", "how", "what", "general", "basic", "simple"]
        if any(term in query_lower for term in vague_terms):
            specificity_score -= 20

        # Extra penalty for very short vague queries
        # Create constant for minimum word count
        min_word_count = 2
        if word_count <= min_word_count and any(term in query_lower for term in vague_terms):
            specificity_score -= 20

        # Ensure score is within bounds
        return max(0.0, min(100.0, specificity_score))

    def _determine_processing_strategy(self, analysis: QueryAnalysis) -> str:
        """
        Determine processing strategy based on query analysis.

        Args:
            analysis: Query analysis results

        Returns:
            str: Processing strategy to use
        """
        if analysis.specificity_score > self.specificity_threshold_high:
            return ProcessingStrategy.HYPOTHETICAL.value
        if analysis.specificity_score >= self.specificity_threshold_low:
            return ProcessingStrategy.ENHANCED.value
        return ProcessingStrategy.DIRECT.value

    def _enhance_query(self, query: str, strategy: ProcessingStrategy) -> str:
        """
        Enhance query based on processing strategy.

        Args:
            query: Original query string
            strategy: Processing strategy to apply

        Returns:
            str: Enhanced query string
        """
        if strategy == ProcessingStrategy.DIRECT:
            # Direct strategy - return query as-is
            return query
        if strategy == ProcessingStrategy.ENHANCED:
            # Enhanced strategy - add context and specificity
            return f"Detailed explanation and implementation guide for: {query}"
        # HYPOTHETICAL
        # Hypothetical strategy - add clarifying context
        return f"Please provide more specific details about: {query}"

    def _create_embeddings(self, text: str) -> list[float]:
        """
        Create embeddings for text.

        Args:
            text: Text to create embeddings for

        Returns:
            list[float]: Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * DEFAULT_EMBEDDING_DIMENSIONS

        # Mock embedding generation based on text hash
        text_hash = hash(text.strip())
        base_value = (text_hash % 1000) / 1000.0  # 0.0 to 0.999

        # Create variant embeddings based on text characteristics
        embeddings = []
        for i in range(DEFAULT_EMBEDDING_DIMENSIONS):
            value = base_value + (i * 0.001)
            # Normalize to [-1, 1] range typical for embeddings
            embeddings.append((value % 2.0) - 1.0)

        return embeddings

    def _create_search_parameters(
        self,
        enhanced_query: str,
        embeddings: list[float],
        limit: int = 10,
        collection: str = "default",
    ) -> SearchParameters:
        """
        Create search parameters for vector store query.

        Args:
            enhanced_query: Enhanced query string
            embeddings: Query embeddings
            limit: Maximum number of results
            collection: Collection name to search

        Returns:
            SearchParameters: Search parameters object
        """
        return SearchParameters(
            embeddings=[embeddings],  # SearchParameters expects list of embeddings
            limit=limit,
            collection=collection,
            strategy=SearchStrategy.SEMANTIC,
            score_threshold=0.3,
        )

    async def enhance_embeddings(self, query: str, docs: list[HypotheticalDocument]) -> list[list[float]]:
        """
        Create enhanced embeddings from query and hypothetical documents.

        Args:
            query: Original user query
            docs: Generated hypothetical documents

        Returns:
            List[List[float]]: Enhanced embedding vectors
        """
        start_time = time.time()

        embeddings = []

        # Create embedding for original query (mock implementation)
        # Use query parameter to create variant embeddings
        query_embedding = [0.5 + hash(query) % 100 / 1000] * DEFAULT_EMBEDDING_DIMENSIONS
        embeddings.append(query_embedding)

        # Add embeddings from hypothetical documents
        for doc in docs:
            if doc.embedding:
                embeddings.append(doc.embedding)

        processing_time = time.time() - start_time
        self.logger.info("Enhanced embeddings created in %.3fs: %d vectors", processing_time, len(embeddings))

        return embeddings

    async def rank_results(self, results: list[VectorSearchResult]) -> RankedResults:
        """
        Rank and filter search results based on relevance and quality.

        Args:
            results: Raw search results from vector database

        Returns:
            RankedResults: Ranked and filtered results
        """
        start_time = time.time()

        if not results:
            return RankedResults(
                results=[],
                total_found=0,
                processing_time=time.time() - start_time,
                ranking_method="no_results",
                hyde_enhanced=False,
            )

        # Sort by similarity score (descending)
        ranked_results = sorted(results, key=lambda x: x.score, reverse=True)

        # Apply quality filtering (remove results below threshold)
        quality_threshold = 0.3
        filtered_results = [r for r in ranked_results if r.score >= quality_threshold]

        processing_time = time.time() - start_time

        self.logger.info(
            "Ranked %d results in %.3fs (filtered from %d)",
            len(filtered_results),
            processing_time,
            len(results),
        )

        return RankedResults(
            results=filtered_results,
            total_found=len(results),
            processing_time=processing_time,
            ranking_method="similarity_score",
            hyde_enhanced=bool(filtered_results),
        )

    @monitor_performance("process_query")
    async def process_query(self, query: str) -> RankedResults:
        """
        Complete HyDE processing pipeline for a query.

        Args:
            query: User query string

        Returns:
            RankedResults: Final processed and ranked results
        """
        start_time = time.time()
        try:
            # Step 1: Three-tier analysis
            enhanced_query = await self.three_tier_analysis(query)

            # Step 2: Handle based on processing strategy
            if enhanced_query.processing_strategy == "clarification_needed":
                # Return empty results with clarifying questions in metadata
                return RankedResults(
                    results=[],
                    total_found=0,
                    processing_time=0.0,
                    ranking_method="clarification_needed",
                    hyde_enhanced=False,
                )

            # Step 3: Create embeddings for search
            if enhanced_query.hypothetical_docs:
                # HyDE-enhanced search
                embeddings = await self.enhance_embeddings(
                    enhanced_query.enhanced_query,
                    enhanced_query.hypothetical_docs,
                )
                hyde_enhanced = True
            else:
                # Direct search
                embeddings = [[0.5] * DEFAULT_EMBEDDING_DIMENSIONS]  # Mock query embedding
                hyde_enhanced = False

            # Step 4: Perform vector search
            search_params = SearchParameters(
                embeddings=embeddings,
                limit=10,
                collection="default",
                strategy=SearchStrategy.SEMANTIC,
                score_threshold=0.3,
            )
            search_results = await self.vector_store.search(search_params)

            # Step 5: Rank and filter results
            ranked_results = await self.rank_results(search_results)
            ranked_results.hyde_enhanced = hyde_enhanced

            return ranked_results

        except Exception as e:
            self.logger.error("HyDE processing failed: %s", str(e))
            # Return empty results with error indication
            processing_time = time.time() - start_time
            return RankedResults(
                results=[],
                total_found=0,
                processing_time=processing_time,
                ranking_method="error",
                hyde_enhanced=False,
            )
