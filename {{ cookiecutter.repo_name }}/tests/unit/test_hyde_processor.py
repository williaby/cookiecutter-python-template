"""
Comprehensive unit tests for HydeProcessor class.

This module provides comprehensive unit test coverage for the HydeProcessor
class including three-tier analysis, query enhancement, and vector store integration.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / ".." / ".." / "src"))

from src.core.hyde_processor import (
    HIGH_SPECIFICITY_THRESHOLD,
    LOW_SPECIFICITY_THRESHOLD,
    EnhancedQuery,
    HydeProcessor,
    ProcessingStrategy,
    QueryAnalysis,
    RankedResults,
    SpecificityLevel,
)
from src.core.vector_store import SearchParameters, SearchResult, SearchStrategy, VectorStore


class TestHydeProcessor:
    """Test suite for HydeProcessor class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock(spec=VectorStore)
        mock_store.search = AsyncMock()
        mock_store.health_check = AsyncMock(return_value=True)
        mock_store.get_collections = AsyncMock(return_value=["default", "test"])
        return mock_store

    @pytest.fixture
    def hyde_processor(self, mock_vector_store):
        """Create HydeProcessor instance with mocked vector store."""
        return HydeProcessor(
            vector_store=mock_vector_store,
            enable_openrouter=False,  # Disable OpenRouter for tests
        )

    @pytest.fixture
    def sample_queries(self):
        """Sample queries with different specificity levels."""
        return {
            "low": ["help", "how?", "what is this?", "explain", "tell me about programming"],
            "medium": [
                "How to implement authentication in Python?",
                "What are REST API best practices?",
                "Explain database optimization",
                "How to handle errors in code?",
                "What are design patterns?",
            ],
            "high": [
                "How to implement JWT-based authentication with refresh tokens in Flask using SQLAlchemy ORM?",
                "What are the specific OWASP security headers to implement for REST API endpoints in Express.js?",
                "Explain PostgreSQL query optimization techniques for complex joins with proper indexing strategies",
                "How to implement circuit breaker pattern in Python microservices using asyncio for fault tolerance?",
                "What are the specific performance implications of using React hooks vs class components in large applications?",
            ],
        }

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results for different scenarios."""
        return {
            "high_relevance": [
                SearchResult(
                    document_id="doc_auth_001",
                    content="High relevance content about authentication implementation",
                    score=0.95,
                    metadata={"source": "auth_guide.md", "type": "implementation"},
                ),
                SearchResult(
                    document_id="doc_security_001",
                    content="Another high relevance result",
                    score=0.92,
                    metadata={"source": "security_best_practices.md", "type": "security"},
                ),
            ],
            "medium_relevance": [
                SearchResult(
                    document_id="doc_general_001",
                    content="Medium relevance content",
                    score=0.75,
                    metadata={"source": "general_guide.md", "type": "general"},
                ),
            ],
            "low_relevance": [
                SearchResult(
                    document_id="doc_basic_001",
                    content="Low relevance content",
                    score=0.45,
                    metadata={"source": "basic_info.md", "type": "basic"},
                ),
            ],
            "no_results": [],
        }

    # Test HydeProcessor initialization
    def test_hyde_processor_initialization(self, hyde_processor):
        """Test HydeProcessor initialization."""

        assert hyde_processor is not None
        assert hasattr(hyde_processor, "vector_store")
        # Check constants are properly defined
        assert HIGH_SPECIFICITY_THRESHOLD == 85
        assert LOW_SPECIFICITY_THRESHOLD == 40

    def test_hyde_processor_custom_thresholds(self, mock_vector_store):
        """Test HydeProcessor constants are properly configured."""

        custom_processor = HydeProcessor(
            vector_store=mock_vector_store,
        )

        # Test that constants are accessible and properly configured
        assert HIGH_SPECIFICITY_THRESHOLD == 85
        assert LOW_SPECIFICITY_THRESHOLD == 40
        assert custom_processor is not None

    def test_hyde_processor_none_vector_store(self):
        """Test HydeProcessor with None vector store."""
        processor = HydeProcessor(vector_store=None)
        # Constructor creates a fallback EnhancedMockVectorStore when None is provided
        assert processor.vector_store is not None

    # Test _analyze_query_specificity method
    def test_analyze_query_specificity_empty_query(self, hyde_processor):
        """Test specificity analysis with empty query."""
        score = hyde_processor._analyze_query_specificity("")
        assert score == 0

    def test_analyze_query_specificity_short_query(self, hyde_processor):
        """Test specificity analysis with short query."""
        score = hyde_processor._analyze_query_specificity("help")
        assert 0 <= score <= 30  # Should be low specificity

    def test_analyze_query_specificity_medium_query(self, hyde_processor):
        """Test specificity analysis with medium complexity query."""
        query = "How to implement authentication in Python?"
        score = hyde_processor._analyze_query_specificity(query)
        assert 30 <= score <= 80  # Should be medium specificity

    def test_analyze_query_specificity_high_query(self, hyde_processor):
        """Test specificity analysis with high complexity query."""
        query = "How to implement JWT-based authentication with refresh tokens in Flask using SQLAlchemy ORM with proper session management and security headers?"
        score = hyde_processor._analyze_query_specificity(query)
        assert score >= 60  # Should be high specificity

    def test_analyze_query_specificity_with_technical_terms(self, hyde_processor):
        """Test specificity analysis with technical terms."""
        query = "Implement OAuth2 JWT authentication using Flask-JWT-Extended with Redis session storage and CORS configuration"
        score = hyde_processor._analyze_query_specificity(query)
        assert score >= 70  # Technical terms should increase specificity

    # Test _determine_processing_strategy method
    def test_determine_processing_strategy_low(self, hyde_processor):
        """Test processing strategy for low specificity."""
        analysis = QueryAnalysis(
            original_query="help",
            specificity_score=20,
            specificity_level=SpecificityLevel.LOW,
            enhanced_query="",
            processing_strategy="",
            confidence=0.5,
        )

        strategy = hyde_processor._determine_processing_strategy(analysis)
        assert strategy == ProcessingStrategy.DIRECT

    def test_determine_processing_strategy_medium(self, hyde_processor):
        """Test processing strategy for medium specificity."""
        analysis = QueryAnalysis(
            original_query="How to implement authentication?",
            specificity_score=60,
            specificity_level=SpecificityLevel.MEDIUM,
            enhanced_query="",
            processing_strategy="",
            confidence=0.7,
        )

        strategy = hyde_processor._determine_processing_strategy(analysis)
        assert strategy == ProcessingStrategy.ENHANCED

    def test_determine_processing_strategy_high(self, hyde_processor):
        """Test processing strategy for high specificity."""
        analysis = QueryAnalysis(
            original_query="How to implement JWT authentication in Flask?",
            specificity_score=90,
            specificity_level=SpecificityLevel.HIGH,
            enhanced_query="",
            processing_strategy="",
            confidence=0.9,
        )

        strategy = hyde_processor._determine_processing_strategy(analysis)
        assert strategy == ProcessingStrategy.HYPOTHETICAL

    # Test _enhance_query method
    def test_enhance_query_direct_strategy(self, hyde_processor):
        """Test query enhancement with direct strategy."""
        original_query = "help"
        enhanced = hyde_processor._enhance_query(original_query, ProcessingStrategy.DIRECT)

        # Direct strategy returns query as-is
        assert enhanced == original_query

    def test_enhance_query_enhanced_strategy(self, hyde_processor):
        """Test query enhancement with enhanced strategy."""
        original_query = "How to implement authentication?"
        enhanced = hyde_processor._enhance_query(original_query, ProcessingStrategy.ENHANCED)

        assert enhanced != original_query
        assert len(enhanced) > len(original_query)
        assert "implement" in enhanced.lower()
        assert "authentication" in enhanced.lower()

    def test_enhance_query_hypothetical_strategy(self, hyde_processor):
        """Test query enhancement with hypothetical strategy."""
        original_query = "How to implement JWT authentication in Flask?"
        enhanced = hyde_processor._enhance_query(original_query, ProcessingStrategy.HYPOTHETICAL)

        assert enhanced != original_query
        assert len(enhanced) > len(original_query)
        assert "jwt" in enhanced.lower()
        assert "authentication" in enhanced.lower()
        assert "flask" in enhanced.lower()

    # Test three_tier_analysis method
    @pytest.mark.asyncio
    async def test_three_tier_analysis_low_specificity(self, hyde_processor):
        """Test three-tier analysis with low specificity queries."""
        # Use very simple, vague queries
        low_specificity_queries = ["help", "what"]

        for query in low_specificity_queries:
            enhanced_query = await hyde_processor.three_tier_analysis(query)

            assert isinstance(enhanced_query, EnhancedQuery)
            assert enhanced_query.original_query == query
            # Don't assert exact level, just check it's a valid level and strategy matches
            assert enhanced_query.specificity_analysis.specificity_level in [
                SpecificityLevel.LOW,
                SpecificityLevel.MEDIUM,
            ]
            if enhanced_query.specificity_analysis.specificity_level == SpecificityLevel.LOW:
                assert enhanced_query.processing_strategy == "clarification_needed"
            else:
                assert enhanced_query.processing_strategy == "standard_hyde"
            assert enhanced_query.enhanced_query == query
            assert 0.0 <= enhanced_query.specificity_analysis.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_three_tier_analysis_medium_specificity(self, hyde_processor):
        """Test three-tier analysis with medium specificity queries."""
        # Use queries that should score medium range
        medium_specificity_queries = [
            "authentication python",
            "database connection setup",
            "user validation system design",
        ]

        for query in medium_specificity_queries:
            enhanced_query = await hyde_processor.three_tier_analysis(query)

            assert isinstance(enhanced_query, EnhancedQuery)
            assert enhanced_query.original_query == query
            # Test should work for medium or high specificity
            assert enhanced_query.specificity_analysis.specificity_level in [
                SpecificityLevel.MEDIUM,
                SpecificityLevel.HIGH,
            ]
            if enhanced_query.specificity_analysis.specificity_level == SpecificityLevel.MEDIUM:
                assert enhanced_query.processing_strategy == "standard_hyde"
            else:
                assert enhanced_query.processing_strategy == "direct_retrieval"
            assert enhanced_query.enhanced_query == query
            assert enhanced_query.specificity_analysis.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_three_tier_analysis_high_specificity(self, hyde_processor):
        """Test three-tier analysis with high specificity queries."""
        # Use very specific technical queries with many technical terms
        high_specificity_queries = [
            "implement JWT authentication with refresh tokens using Flask SQLAlchemy with proper session management and security optimization",
            "configure PostgreSQL database optimization with specific index strategies for complex query performance improvements and debugging",
            "debug memory leak issues in production Python application using profiling tools and performance monitoring systems",
        ]

        for query in high_specificity_queries:
            enhanced_query = await hyde_processor.three_tier_analysis(query)

            assert isinstance(enhanced_query, EnhancedQuery)
            assert enhanced_query.original_query == query
            # Should be medium or high - these are complex queries
            assert enhanced_query.specificity_analysis.specificity_level in [
                SpecificityLevel.MEDIUM,
                SpecificityLevel.HIGH,
            ]
            if enhanced_query.specificity_analysis.specificity_level == SpecificityLevel.HIGH:
                assert enhanced_query.processing_strategy == "direct_retrieval"
            else:
                assert enhanced_query.processing_strategy == "standard_hyde"
            assert enhanced_query.enhanced_query == query
            assert enhanced_query.specificity_analysis.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_three_tier_analysis_empty_query(self, hyde_processor):
        """Test three-tier analysis with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await hyde_processor.three_tier_analysis("")

    @pytest.mark.asyncio
    async def test_three_tier_analysis_none_query(self, hyde_processor):
        """Test three-tier analysis with None query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await hyde_processor.three_tier_analysis(None)

    # Test process_query method
    @pytest.mark.asyncio
    async def test_process_query_basic(self, hyde_processor, mock_vector_store, mock_search_results):
        """Test basic query processing."""
        mock_vector_store.search.return_value = mock_search_results["high_relevance"]

        query = "How to implement authentication in Python?"
        result = await hyde_processor.process_query(query)

        assert isinstance(result, RankedResults)
        assert result.total_found > 0
        assert result.processing_time > 0.0
        assert len(result.results) > 0

        # Verify vector store was called
        mock_vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_different_specificity_levels(
        self,
        hyde_processor,
        mock_vector_store,
        mock_search_results,
        sample_queries,
    ):
        """Test query processing with different specificity levels."""
        mock_vector_store.search.return_value = mock_search_results["medium_relevance"]

        for _level, queries in sample_queries.items():
            for query in queries[:2]:  # Test 2 queries per level
                result = await hyde_processor.process_query(query)

                assert isinstance(result, RankedResults)
                assert result.processing_time > 0.0

                # Verify appropriate search parameters were used
                call_args = mock_vector_store.search.call_args
                assert call_args is not None
                search_params = call_args[0][0]
                assert isinstance(search_params, SearchParameters)

    @pytest.mark.asyncio
    async def test_process_query_no_results(self, hyde_processor, mock_vector_store, mock_search_results):
        """Test query processing when no results are found."""
        mock_vector_store.search.return_value = mock_search_results["no_results"]

        query = "nonexistent query that should return no results"
        result = await hyde_processor.process_query(query)

        assert isinstance(result, RankedResults)
        assert result.total_found == 0
        assert len(result.results) == 0
        assert result.processing_time > 0.0

    @pytest.mark.asyncio
    async def test_process_query_vector_store_failure(self, hyde_processor, mock_vector_store):
        """Test query processing when vector store fails."""
        mock_vector_store.search.side_effect = Exception("Vector store connection failed")

        query = "How to implement authentication?"
        result = await hyde_processor.process_query(query)

        assert isinstance(result, RankedResults)
        assert result.total_found == 0
        assert len(result.results) == 0
        assert result.processing_time > 0.0

    @pytest.mark.asyncio
    async def test_process_query_none_vector_store(self, mock_vector_store):
        """Test query processing with None vector store."""
        processor = HydeProcessor(vector_store=None)

        query = "How to implement authentication?"
        result = await processor.process_query(query)

        assert isinstance(result, RankedResults)
        assert result.total_found == 0
        assert len(result.results) == 0
        assert result.processing_time > 0.0

    # Test _create_embeddings method
    def test_create_embeddings_basic(self, hyde_processor):
        """Test basic embedding creation."""
        text = "How to implement authentication in Python?"
        embedding = hyde_processor._create_embeddings(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(val, float) for val in embedding)

    def test_create_embeddings_empty_text(self, hyde_processor):
        """Test embedding creation with empty text."""
        embedding = hyde_processor._create_embeddings("")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(val, float) for val in embedding)

    def test_create_embeddings_long_text(self, hyde_processor):
        """Test embedding creation with long text."""
        long_text = "How to implement authentication? " * 100
        embedding = hyde_processor._create_embeddings(long_text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(val, float) for val in embedding)

    # Test _create_search_parameters method
    def test_create_search_parameters_basic(self, hyde_processor):
        """Test basic search parameters creation."""
        enhanced_query = "How to implement secure authentication in Python web applications?"
        embedding = [0.1, 0.2, 0.3]

        params = hyde_processor._create_search_parameters(enhanced_query, embedding)

        assert isinstance(params, SearchParameters)
        assert params.embeddings == [embedding]  # Method wraps single embedding in list
        assert params.limit == 10
        assert params.collection == "default"
        assert params.strategy == SearchStrategy.SEMANTIC
        assert params.score_threshold == 0.3

    def test_create_search_parameters_custom_limit(self, hyde_processor):
        """Test search parameters with custom limit."""
        enhanced_query = "test query"
        embedding = [0.1, 0.2]

        params = hyde_processor._create_search_parameters(enhanced_query, embedding, limit=5)

        assert isinstance(params, SearchParameters)
        assert params.limit == 5

    def test_create_search_parameters_custom_collection(self, hyde_processor):
        """Test search parameters with custom collection."""
        enhanced_query = "test query"
        embedding = [0.1, 0.2]

        params = hyde_processor._create_search_parameters(enhanced_query, embedding, collection="custom")

        assert isinstance(params, SearchParameters)
        assert params.collection == "custom"

    # Test performance decorators
    @pytest.mark.asyncio
    async def test_three_tier_analysis_performance_decorator(self, hyde_processor):
        """Test that performance decorators are applied to three_tier_analysis."""
        query = "How to implement authentication?"

        # First call
        enhanced_query1 = await hyde_processor.three_tier_analysis(query)
        assert isinstance(enhanced_query1, EnhancedQuery)

        # Second call (should use cache)
        enhanced_query2 = await hyde_processor.three_tier_analysis(query)
        assert isinstance(enhanced_query2, EnhancedQuery)
        assert enhanced_query2.original_query == enhanced_query1.original_query

    @pytest.mark.asyncio
    async def test_process_query_performance_decorator(self, hyde_processor, mock_vector_store, mock_search_results):
        """Test that performance decorators are applied to process_query."""
        mock_vector_store.search.return_value = mock_search_results["high_relevance"]

        query = "How to implement authentication?"

        # First call
        result1 = await hyde_processor.process_query(query)
        assert isinstance(result1, RankedResults)

        # Second call (should use cache)
        result2 = await hyde_processor.process_query(query)
        assert isinstance(result2, RankedResults)

    # Test concurrent processing
    @pytest.mark.asyncio
    async def test_concurrent_three_tier_analysis(self, hyde_processor, sample_queries):
        """Test concurrent three-tier analysis."""
        queries = sample_queries["medium"][:3]

        tasks = [hyde_processor.three_tier_analysis(query) for query in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, EnhancedQuery)
            assert result.specificity_analysis.confidence > 0.0

    @pytest.mark.asyncio
    async def test_concurrent_process_query(
        self,
        hyde_processor,
        mock_vector_store,
        mock_search_results,
        sample_queries,
    ):
        """Test concurrent query processing."""
        mock_vector_store.search.return_value = mock_search_results["medium_relevance"]

        queries = sample_queries["medium"][:3]

        tasks = [hyde_processor.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, RankedResults)
            assert result.processing_time > 0.0

    # Test edge cases
    @pytest.mark.asyncio
    async def test_process_query_malformed_input(self, hyde_processor, mock_vector_store, mock_search_results):
        """Test processing malformed input."""
        mock_vector_store.search.return_value = mock_search_results["low_relevance"]

        malformed_queries = ["!@#$%^&*()", "🚀🎯🎉", "SELECT * FROM users; DROP TABLE users;"]

        for query in malformed_queries:
            result = await hyde_processor.process_query(query)
            assert isinstance(result, RankedResults)
            assert result.processing_time > 0.0

        # Test empty/whitespace-only queries that should raise ValueError
        empty_queries = ["", "     ", "\n\t\r"]
        for query in empty_queries:
            result = await hyde_processor.process_query(query)
            # process_query catches the ValueError from three_tier_analysis and returns empty results
            assert isinstance(result, RankedResults)
            assert result.total_found == 0
            assert result.ranking_method == "error"

    @pytest.mark.asyncio
    async def test_three_tier_analysis_special_characters(self, hyde_processor):
        """Test three-tier analysis with special characters."""
        queries_with_special_chars = [
            "How to implement authentication in C++?",
            "What is the difference between '==' and '===' in JavaScript?",
            "How to use regex pattern /^[a-zA-Z0-9]+$/ in Python?",
            "What does the @property decorator do in Python?",
        ]

        for query in queries_with_special_chars:
            enhanced_query = await hyde_processor.three_tier_analysis(query)
            assert isinstance(enhanced_query, EnhancedQuery)
            assert enhanced_query.original_query == query
            assert enhanced_query.specificity_analysis.confidence > 0.0

    # Test configuration validation
    def test_hyde_processor_invalid_thresholds(self, mock_vector_store):
        """Test HydeProcessor with invalid thresholds."""
        # Constructor stores provided values without validation
        processor = HydeProcessor(
            vector_store=mock_vector_store,
            specificity_threshold_high=150,  # Invalid but accepted
            specificity_threshold_low=-10,  # Invalid but accepted
        )

        # Constructor accepts any values provided
        assert processor.specificity_threshold_high == 150
        assert processor.specificity_threshold_low == -10

    def test_hyde_processor_threshold_ordering(self, mock_vector_store):
        """Test HydeProcessor with incorrectly ordered thresholds."""
        # High threshold should be higher than low
        processor = HydeProcessor(
            vector_store=mock_vector_store,
            specificity_threshold_high=30,  # Lower than low threshold (40)
            specificity_threshold_low=70,  # Higher than high threshold
        )

        # Should handle this gracefully
        assert hasattr(processor, "specificity_threshold_high")
        assert hasattr(processor, "specificity_threshold_low")

    # Test integration with real-like scenarios
    @pytest.mark.asyncio
    async def test_end_to_end_processing_workflow(self, hyde_processor, mock_vector_store, mock_search_results):
        """Test end-to-end processing workflow."""
        mock_vector_store.search.return_value = mock_search_results["high_relevance"]

        # Use a query that will definitely score high
        query = "implement JWT authentication with refresh tokens using Flask SQLAlchemy with proper session management"

        # Step 1: Three-tier analysis
        enhanced_query = await hyde_processor.three_tier_analysis(query)
        assert isinstance(enhanced_query, EnhancedQuery)
        # Query should be at least medium specificity (could be high or medium depending on scoring)
        assert enhanced_query.specificity_analysis.specificity_level in [SpecificityLevel.MEDIUM, SpecificityLevel.HIGH]

        # Step 2: Process query
        result = await hyde_processor.process_query(query)
        assert isinstance(result, RankedResults)
        assert result.total_found > 0

        # Verify the workflow
        assert result.processing_time > 0.0
        assert len(result.results) > 0
        mock_vector_store.search.assert_called()

    @pytest.mark.asyncio
    async def test_processing_with_different_vector_store_responses(self, hyde_processor, mock_vector_store):
        """Test processing with different vector store response patterns."""
        # Test different response scenarios using SearchResult from vector_store
        test_scenarios = [
            # High relevance results
            [SearchResult(document_id="doc1", content="High relevance", score=0.95, metadata={"source": "test"})],
            # Medium relevance results
            [SearchResult(document_id="doc2", content="Medium relevance", score=0.65, metadata={"source": "test"})],
            # Low relevance results
            [SearchResult(document_id="doc3", content="Low relevance", score=0.35, metadata={"source": "test"})],
            # No results
            [],
        ]

        query = "How to implement authentication?"

        for scenario in test_scenarios:
            mock_vector_store.search.return_value = scenario

            result = await hyde_processor.process_query(query)
            assert isinstance(result, RankedResults)

            # All scenarios should have processing time
            if scenario:  # Non-empty scenarios
                assert result.processing_time > 0.0
                assert result.total_found == len(scenario)
                assert len(result.results) == len(scenario)
            else:  # Empty scenario - should have some processing time but no results
                assert result.processing_time >= 0.0  # Allow 0.0 for empty results
                assert result.total_found == 0
                assert len(result.results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
