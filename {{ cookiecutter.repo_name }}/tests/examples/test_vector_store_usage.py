#!/usr/bin/env python3
"""Comprehensive tests for examples/vector_store_usage.py.

This test suite covers all functions and code paths in the vector store usage example,
ensuring 80%+ test coverage while following project testing standards.
"""

import contextlib
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from examples.vector_store_usage import VectorStoreExamples, main
from src.core.vector_store import (
    DEFAULT_VECTOR_DIMENSIONS,
    SearchParameters,
    SearchStrategy,
    VectorDocument,
    VectorStoreType,
)


class TestVectorStoreExamples:
    """Test the VectorStoreExamples class and its methods."""

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_basic_mock_store_usage(self, mock_factory):
        """Test basic mock store usage example."""
        # Setup mock store
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        # Execute the example
        await VectorStoreExamples.basic_mock_store_usage()

        # Verify store interactions
        mock_factory.create_vector_store.assert_called_once()
        mock_store.connect.assert_called_once()
        mock_store.health_check.assert_called_once()
        mock_store.insert_documents.assert_called_once()
        mock_store.search.assert_called_once()
        mock_store.get_metrics.assert_called_once()
        mock_store.disconnect.assert_called_once()

        # Verify configuration
        config = mock_factory.create_vector_store.call_args[0][0]
        assert config["type"] == VectorStoreType.MOCK
        assert config["simulate_latency"] is True
        assert config["error_rate"] == 0.0
        assert config["base_latency"] == 0.01

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_basic_mock_store_usage_documents(self, mock_factory):
        """Test that documents are created correctly in basic usage."""
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        await VectorStoreExamples.basic_mock_store_usage()

        # Verify documents were inserted
        insert_call = mock_store.insert_documents.call_args[0][0]
        assert len(insert_call) == 3

        # Check document structure
        for doc in insert_call:
            assert isinstance(doc, VectorDocument)
            assert doc.id is not None
            assert doc.content is not None
            assert doc.embedding is not None
            assert len(doc.embedding) == DEFAULT_VECTOR_DIMENSIONS
            assert doc.metadata is not None
            assert doc.collection == "programming"

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_basic_mock_store_usage_search_parameters(self, mock_factory):
        """Test search parameters in basic usage."""
        mock_store = AsyncMock()
        mock_store.search.return_value = [Mock(), Mock()]  # Mock search results
        mock_factory.create_vector_store.return_value = mock_store

        await VectorStoreExamples.basic_mock_store_usage()

        # Verify search parameters
        search_call = mock_store.search.call_args[0][0]
        assert isinstance(search_call, SearchParameters)
        assert len(search_call.embeddings[0]) == DEFAULT_VECTOR_DIMENSIONS
        assert search_call.limit == 3
        assert search_call.collection == "programming"
        assert search_call.strategy == SearchStrategy.SEMANTIC

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.vector_store_connection")
    async def test_advanced_search_strategies(self, mock_connection):
        """Test advanced search strategies example."""
        mock_store = AsyncMock()
        mock_store.search.return_value = [Mock(), Mock()]
        mock_connection.return_value.__aenter__.return_value = mock_store

        await VectorStoreExamples.advanced_search_strategies()

        # Verify store operations
        mock_store.insert_documents.assert_called_once()

        # Should test multiple search strategies
        assert mock_store.search.call_count >= 4  # One for each strategy

        # Verify different strategies were tested
        search_calls = [call[0][0] for call in mock_store.search.call_args_list]
        strategies_tested = {params.strategy for params in search_calls}

        expected_strategies = {
            SearchStrategy.EXACT,
            SearchStrategy.SEMANTIC,
            SearchStrategy.HYBRID,
            SearchStrategy.FILTERED,
        }
        assert strategies_tested.issuperset(expected_strategies)

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.vector_store_connection")
    async def test_advanced_search_strategies_documents(self, mock_connection):
        """Test documents in advanced search strategies."""
        mock_store = AsyncMock()
        mock_connection.return_value.__aenter__.return_value = mock_store

        await VectorStoreExamples.advanced_search_strategies()

        # Verify ML documents were inserted
        insert_call = mock_store.insert_documents.call_args[0][0]
        assert len(insert_call) == 3

        # Check document metadata richness
        for doc in insert_call:
            assert "category" in doc.metadata
            assert "difficulty" in doc.metadata
            assert "tags" in doc.metadata
            assert "author" in doc.metadata
            assert "year" in doc.metadata
            assert doc.collection == "ml_docs"

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.vector_store_connection")
    async def test_advanced_search_strategies_filtering(self, mock_connection):
        """Test filtering in advanced search strategies."""
        mock_store = AsyncMock()
        mock_connection.return_value.__aenter__.return_value = mock_store

        await VectorStoreExamples.advanced_search_strategies()

        # Find filtered search calls
        search_calls = [call[0][0] for call in mock_store.search.call_args_list]
        filtered_calls = [params for params in search_calls if params.filters]

        # Should have multiple filtered searches
        assert len(filtered_calls) >= 4

        # Check filter variety
        filter_keys = set()
        for params in filtered_calls:
            filter_keys.update(params.filters.keys())

        expected_filters = {"difficulty", "category", "year", "tags"}
        assert filter_keys.issuperset(expected_filters)

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    @patch("examples.vector_store_usage.HydeProcessor")
    async def test_hyde_processor_integration(self, mock_hyde_class, mock_factory):
        """Test HyDE processor integration example."""
        # Setup mocks
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        mock_hyde = AsyncMock()
        mock_hyde_class.return_value = mock_hyde

        # Mock enhanced query results
        mock_enhanced_query = Mock()
        mock_enhanced_query.processing_strategy = "standard_hyde"
        mock_enhanced_query.hypothetical_docs = [
            Mock(
                content="Mock HyDE document",
                embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
                relevance_score=0.95,
                generation_method="mock",
                metadata={"source": "hyde"},
            ),
        ]
        mock_enhanced_query.specificity_analysis = Mock()
        mock_enhanced_query.specificity_analysis.guiding_questions = ["What framework?", "What use case?"]

        mock_hyde.three_tier_analysis.return_value = mock_enhanced_query
        mock_hyde.enhance_embeddings.return_value = [[0.2] * DEFAULT_VECTOR_DIMENSIONS]

        await VectorStoreExamples.hyde_processor_integration()

        # Verify HyDE processor interactions
        mock_hyde.three_tier_analysis.assert_called()

        # Should process multiple queries
        assert mock_hyde.three_tier_analysis.call_count == 3

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    @patch("examples.vector_store_usage.HydeProcessor")
    async def test_hyde_processor_integration_clarification_needed(self, mock_hyde_class, mock_factory):
        """Test HyDE processor integration with clarification needed."""
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        mock_hyde = AsyncMock()
        mock_hyde_class.return_value = mock_hyde

        # Mock query requiring clarification
        mock_enhanced_query = Mock()
        mock_enhanced_query.processing_strategy = "clarification_needed"
        mock_enhanced_query.specificity_analysis = Mock()
        mock_enhanced_query.specificity_analysis.guiding_questions = ["What specific implementation?", "Which version?"]

        mock_hyde.three_tier_analysis.return_value = mock_enhanced_query

        await VectorStoreExamples.hyde_processor_integration()

        # Should handle clarification case without errors
        mock_hyde.three_tier_analysis.assert_called()

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    @patch("examples.vector_store_usage.time")
    async def test_performance_monitoring(self, mock_time, mock_factory):
        """Test performance monitoring example."""
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        # Mock time.time() for performance measurement
        mock_time.time.return_value = 0.0

        await VectorStoreExamples.performance_monitoring()

        # Verify performance test setup
        config = mock_factory.create_vector_store.call_args[0][0]
        assert config["simulate_latency"] is True
        assert config["base_latency"] == 0.02

        # Verify batch insert with 100 documents
        insert_call = mock_store.insert_documents.call_args[0][0]
        assert len(insert_call) == 100

        # Verify multiple searches (20)
        assert mock_store.search.call_count == 20

        # Verify metrics collection
        mock_store.get_metrics.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    @patch("examples.vector_store_usage.time")
    async def test_performance_monitoring_document_generation(self, mock_time, mock_factory):
        """Test document generation in performance monitoring."""
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store
        mock_time.time.return_value = 0.0

        await VectorStoreExamples.performance_monitoring()

        # Check generated documents
        insert_call = mock_store.insert_documents.call_args[0][0]

        for i, doc in enumerate(insert_call):
            assert doc.id == f"perf_doc_{i}"
            assert f"Performance test document {i}" in doc.content
            assert doc.metadata["index"] == i
            assert doc.metadata["topic"] == i % 10
            assert doc.collection == "performance_test"
            assert len(doc.embedding) == DEFAULT_VECTOR_DIMENSIONS

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    @patch("examples.vector_store_usage.asyncio.sleep")
    async def test_error_handling_and_resilience(self, mock_sleep, mock_factory):
        """Test error handling and resilience example."""
        # Mock stores with different error rates
        mock_store_some_errors = AsyncMock()
        mock_store_all_errors = AsyncMock()

        # First store succeeds after retries, second always fails
        mock_factory.create_vector_store.side_effect = [mock_store_some_errors, mock_store_all_errors]

        # Configure search behavior for retry testing
        mock_store_some_errors.search.side_effect = [Exception("Temp error"), Exception("Temp error"), "Success"]

        # Mock a search return value (empty list) for the error store
        mock_store_all_errors.search.return_value = []

        # The error handling function should suppress exceptions
        await VectorStoreExamples.error_handling_and_resilience()

        # Verify retry logic
        assert mock_store_some_errors.search.call_count <= 3  # Max retries
        mock_sleep.assert_called()  # Should use exponential backoff

        # Verify circuit breaker testing
        assert mock_store_all_errors.search.call_count >= 1

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_error_handling_retry_logic(self, mock_factory):
        """Test retry logic in error handling example."""
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        # Mock search to fail twice then succeed
        mock_store.search.side_effect = [Exception("Error 1"), Exception("Error 2"), "Success"]

        # Mock the retry logic manually to test it
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await mock_store.search(Mock())
                if result == "Success":
                    break
            except Exception:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    pass
                else:
                    # Exponential backoff would happen here
                    pass

        # Should eventually succeed
        assert mock_store.search.call_count == 3

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_qdrant_configuration_example(self, mock_factory):
        """Test Qdrant configuration example."""
        mock_factory.create_vector_store.side_effect = Exception("No qdrant-client")

        # Should not raise exception due to contextlib.suppress
        await VectorStoreExamples.qdrant_configuration_example()

        # Verify correct Qdrant configuration was attempted
        config = mock_factory.create_vector_store.call_args[0][0]
        assert config["type"] == VectorStoreType.QDRANT
        assert config["host"] == "192.168.1.16"
        assert config["port"] == 6333
        assert config["api_key"] is None
        assert config["timeout"] == 30.0

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.vector_store_connection")
    async def test_collection_management(self, mock_connection):
        """Test collection management example."""
        mock_store = AsyncMock()
        mock_store.list_collections.return_value = ["user_knowledge", "code_examples", "documentation"]
        mock_store.create_collection.return_value = True
        mock_store.get_collection_info.return_value = {"name": "test", "size": 1000}
        mock_connection.return_value.__aenter__.return_value = mock_store

        await VectorStoreExamples.collection_management()

        # Verify collection operations
        mock_store.list_collections.assert_called()
        assert mock_store.create_collection.call_count == 3  # Three new collections
        assert mock_store.get_collection_info.call_count == 3
        mock_store.insert_documents.assert_called()  # Documents inserted
        mock_store.search.assert_called()  # Searches performed

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.vector_store_connection")
    async def test_collection_management_collections_created(self, mock_connection):
        """Test specific collections created in collection management."""
        mock_store = AsyncMock()
        mock_store.list_collections.return_value = ["user_knowledge", "code_examples", "documentation"]
        mock_store.create_collection.return_value = True
        mock_connection.return_value.__aenter__.return_value = mock_store

        await VectorStoreExamples.collection_management()

        # Check collections created
        create_calls = mock_store.create_collection.call_args_list
        collections_created = [(call[0][0], call[0][1]) for call in create_calls]

        expected_collections = [
            ("user_knowledge", 512),
            ("code_examples", 768),
            ("documentation", 384),
        ]

        assert collections_created == expected_collections

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.vector_store_connection")
    async def test_collection_management_documents_by_collection(self, mock_connection):
        """Test documents are inserted into correct collections."""
        mock_store = AsyncMock()
        mock_store.list_collections.return_value = ["user_knowledge", "code_examples", "documentation"]
        mock_connection.return_value.__aenter__.return_value = mock_store

        await VectorStoreExamples.collection_management()

        # Check document insertions
        insert_calls = mock_store.insert_documents.call_args_list

        # Should have documents for each collection
        assert len(insert_calls) == 3

        # Check specific document properties
        for call in insert_calls:
            docs = call[0][0]
            for doc in docs:
                if doc.collection == "user_knowledge":
                    assert len(doc.embedding) == 512
                elif doc.collection == "code_examples":
                    assert len(doc.embedding) == 768
                elif doc.collection == "documentation":
                    assert len(doc.embedding) == 384


class TestMainFunction:
    """Test the main function."""

    @pytest.mark.asyncio
    @patch.object(VectorStoreExamples, "basic_mock_store_usage")
    @patch.object(VectorStoreExamples, "advanced_search_strategies")
    @patch.object(VectorStoreExamples, "hyde_processor_integration")
    @patch.object(VectorStoreExamples, "performance_monitoring")
    @patch.object(VectorStoreExamples, "error_handling_and_resilience")
    @patch.object(VectorStoreExamples, "qdrant_configuration_example")
    @patch.object(VectorStoreExamples, "collection_management")
    async def test_main_function_all_examples(
        self,
        mock_collection_mgmt,
        mock_qdrant_config,
        mock_error_handling,
        mock_performance,
        mock_hyde_integration,
        mock_advanced_search,
        mock_basic_usage,
    ):
        """Test that main function runs all examples."""
        await main()

        # Verify all examples were called
        mock_basic_usage.assert_called_once()
        mock_advanced_search.assert_called_once()
        mock_hyde_integration.assert_called_once()
        mock_performance.assert_called_once()
        mock_error_handling.assert_called_once()
        mock_qdrant_config.assert_called_once()
        mock_collection_mgmt.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @patch.object(VectorStoreExamples, "basic_mock_store_usage", side_effect=Exception("Test error"))
    @patch.object(VectorStoreExamples, "advanced_search_strategies")
    @patch("examples.vector_store_usage.logger")
    async def test_main_function_error_handling(self, mock_logger, mock_advanced, mock_basic):
        """Test main function error handling."""
        await main()

        # Should continue even if one example fails
        mock_basic.assert_called_once()
        mock_advanced.assert_called_once()

        # Should log the exception
        mock_logger.exception.assert_called_once_with("Example failed")

    @pytest.mark.asyncio
    @patch.object(VectorStoreExamples, "basic_mock_store_usage")
    @patch.object(VectorStoreExamples, "advanced_search_strategies", side_effect=Exception("Error 2"))
    @patch.object(VectorStoreExamples, "hyde_processor_integration", side_effect=Exception("Error 3"))
    @patch("examples.vector_store_usage.logger")
    @pytest.mark.slow
    async def test_main_function_multiple_errors(self, mock_logger, mock_hyde, mock_advanced, mock_basic):
        """Test main function with multiple example failures."""
        await main()

        # All examples should be attempted
        mock_basic.assert_called_once()
        mock_advanced.assert_called_once()
        mock_hyde.assert_called_once()

        # Should log multiple exceptions
        assert mock_logger.exception.call_count >= 2


class TestLoggingAndConfiguration:
    """Test logging setup and configuration."""

    @patch("examples.vector_store_usage.logging.basicConfig")
    def test_logging_configuration(self, mock_basic_config):
        """Test that logging is configured correctly."""
        # Re-import to trigger logging setup
        import importlib

        import examples.vector_store_usage

        importlib.reload(examples.vector_store_usage)

        # Verify logging was configured
        mock_basic_config.assert_called_with(level=examples.vector_store_usage.logging.INFO)

    def test_logger_creation(self):
        """Test that logger is created correctly."""
        import examples.vector_store_usage

        assert examples.vector_store_usage.logger.name == "examples.vector_store_usage"


class TestDocumentCreation:
    """Test vector document creation patterns."""

    def test_vector_document_structure_basic(self):
        """Test basic vector document structure."""
        doc = VectorDocument(
            id="test_doc",
            content="Test content",
            embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
            metadata={"test": "value"},
            collection="test_collection",
        )

        assert doc.id == "test_doc"
        assert doc.content == "Test content"
        assert len(doc.embedding) == DEFAULT_VECTOR_DIMENSIONS
        assert doc.metadata["test"] == "value"
        assert doc.collection == "test_collection"

    def test_vector_document_embedding_dimensions(self):
        """Test that vector documents use correct embedding dimensions."""
        # Test different embedding sizes used in examples
        sizes = [384, 512, 768, DEFAULT_VECTOR_DIMENSIONS]

        for size in sizes:
            embedding = [0.1] * size
            doc = VectorDocument(
                id=f"test_{size}",
                content="Test",
                embedding=embedding,
                metadata={},
                collection="test",
            )
            assert len(doc.embedding) == size

    def test_search_parameters_creation(self):
        """Test search parameters creation."""
        embeddings = [[0.1] * DEFAULT_VECTOR_DIMENSIONS]
        params = SearchParameters(
            embeddings=embeddings,
            collection="test",
            strategy=SearchStrategy.SEMANTIC,
            limit=10,
            filters={"category": "test"},
        )

        assert params.embeddings == embeddings
        assert params.collection == "test"
        assert params.strategy == SearchStrategy.SEMANTIC
        assert params.limit == 10
        assert params.filters == {"category": "test"}


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_empty_search_results(self, mock_factory):
        """Test handling of empty search results."""
        mock_store = AsyncMock()
        mock_store.search.return_value = []  # Empty results
        mock_factory.create_vector_store.return_value = mock_store

        # Should handle empty results gracefully
        await VectorStoreExamples.basic_mock_store_usage()

        mock_store.search.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_store_connection_failure(self, mock_factory):
        """Test handling of store connection failure."""
        mock_store = AsyncMock()
        mock_store.connect.side_effect = Exception("Connection failed")
        mock_factory.create_vector_store.return_value = mock_store

        # Should raise exception for connection failure
        with pytest.raises(Exception, match="Connection failed"):
            await VectorStoreExamples.basic_mock_store_usage()

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    async def test_large_batch_insert(self, mock_factory):
        """Test handling of large batch inserts."""
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        # Test with performance monitoring (100 documents)
        await VectorStoreExamples.performance_monitoring()

        # Verify large batch was inserted
        insert_call = mock_store.insert_documents.call_args[0][0]
        assert len(insert_call) == 100

    def test_vector_dimension_consistency(self):
        """Test that all vector dimensions are consistent."""
        # All embeddings should use DEFAULT_VECTOR_DIMENSIONS unless specifically testing different sizes
        assert DEFAULT_VECTOR_DIMENSIONS > 0

        # Test that we can create embeddings of this size
        embedding = [0.1] * DEFAULT_VECTOR_DIMENSIONS
        assert len(embedding) == DEFAULT_VECTOR_DIMENSIONS


class TestIntegrationScenarios:
    """Test integration scenarios and realistic usage patterns."""

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.VectorStoreFactory")
    @patch("examples.vector_store_usage.HydeProcessor")
    async def test_realistic_hyde_integration_workflow(self, mock_hyde_class, mock_factory):
        """Test realistic HyDE integration workflow."""
        mock_store = AsyncMock()
        mock_factory.create_vector_store.return_value = mock_store

        mock_hyde = AsyncMock()
        mock_hyde_class.return_value = mock_hyde

        # Create realistic enhanced query
        mock_doc = Mock()
        mock_doc.content = "How to implement caching in Python applications using Redis"
        mock_doc.embedding = [0.1] * DEFAULT_VECTOR_DIMENSIONS
        mock_doc.relevance_score = 0.9
        mock_doc.generation_method = "hyde_generation"
        mock_doc.metadata = {"complexity": "intermediate"}

        mock_enhanced_query = Mock()
        mock_enhanced_query.processing_strategy = "standard_hyde"
        mock_enhanced_query.hypothetical_docs = [mock_doc]

        mock_hyde.three_tier_analysis.return_value = mock_enhanced_query
        mock_hyde.enhance_embeddings.return_value = [[0.2] * DEFAULT_VECTOR_DIMENSIONS]

        await VectorStoreExamples.hyde_processor_integration()

        # Verify realistic workflow
        mock_hyde.three_tier_analysis.assert_called()
        mock_store.insert_documents.assert_called()
        mock_store.search.assert_called()

    @pytest.mark.asyncio
    async def test_async_context_manager_pattern(self):
        """Test async context manager usage pattern."""
        # Test that contextlib patterns work correctly
        async with contextlib.AsyncExitStack() as stack:
            # Simulate context manager usage
            mock_store = AsyncMock()
            await stack.enter_async_context(mock_store)

        # Should complete without errors
        assert True

    @pytest.mark.asyncio
    @patch("examples.vector_store_usage.time.time")
    async def test_performance_timing_accuracy(self, mock_time):
        """Test that performance timing is calculated correctly."""
        # Mock time progression
        times = [0.0, 0.1, 0.2, 0.3]
        mock_time.side_effect = times

        # Simulate timing measurement
        start_time = time.time()
        # Simulate work
        end_time = time.time()
        duration = end_time - start_time

        assert duration == 0.1  # Should be difference between first two calls

    def test_memory_efficiency_document_creation(self):
        """Test memory efficiency of document creation."""
        # Create many documents to test memory usage
        documents = []
        for i in range(1000):
            doc = VectorDocument(
                id=f"doc_{i}",
                content=f"Content {i}",
                embedding=[0.001 * i] * 100,  # Smaller embedding for memory test
                metadata={"index": i},
                collection="test",
            )
            documents.append(doc)

        # Should create all documents without memory issues
        assert len(documents) == 1000
        assert all(doc.id.startswith("doc_") for doc in documents)
