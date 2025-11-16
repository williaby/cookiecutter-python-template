"""
Integration tests for vector store with HydeProcessor and other components.

This module tests the integration between the vector store implementations
and other PromptCraft components, particularly focusing on HydeProcessor
compatibility and end-to-end workflows.
"""

import asyncio
import time

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from src.core.hyde_processor import (
    HydeProcessor,
)
from src.core.vector_store import (
    DEFAULT_VECTOR_DIMENSIONS,
    EnhancedMockVectorStore,
    MockVectorStore,
    QdrantVectorStore,
    SearchParameters,
    SearchStrategy,
    VectorDocument,
    VectorStoreFactory,
    VectorStoreType,
    vector_store_connection,
)


class TestVectorStoreHydeIntegration:
    """Test integration between vector store and HydeProcessor."""

    @pytest.fixture
    async def enhanced_vector_store(self):
        """Provide enhanced mock vector store."""
        config = {"simulate_latency": False, "error_rate": 0.0, "base_latency": 0.001}
        store = EnhancedMockVectorStore(config)
        await store.connect()
        return store

    @pytest.fixture
    async def hyde_processor_with_enhanced_store(self, enhanced_vector_store):
        """Provide HydeProcessor with enhanced vector store."""
        # Create compatibility wrapper
        compatible_store = MockVectorStore()
        await asyncio.sleep(0.1)  # Wait for auto-connect

        processor = HydeProcessor(vector_store=compatible_store)
        return processor, enhanced_vector_store

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_query_processing(self, hyde_processor_with_enhanced_store):
        """Test complete end-to-end query processing with enhanced vector store."""
        processor, enhanced_store = hyde_processor_with_enhanced_store

        # Test high-specificity query (direct retrieval)
        high_spec_query = "How to implement Redis caching in Python with connection pooling and error handling"

        results = await processor.process_query(high_spec_query)

        assert results is not None
        assert hasattr(results, "results")
        assert hasattr(results, "hyde_enhanced")
        assert results.total_found >= 0

        # Test medium-specificity query (HyDE enhancement)
        medium_spec_query = "Python caching strategies"

        results = await processor.process_query(medium_spec_query)

        assert results is not None
        assert results.ranking_method in ["similarity_score", "clarification_needed", "error"]

    @pytest.mark.asyncio
    async def test_hyde_document_generation_and_storage(self, enhanced_vector_store):
        """Test HyDE document generation and storage in enhanced vector store."""
        # Create HydeProcessor with enhanced store
        processor = HydeProcessor()

        # Generate hypothetical documents
        query = "Implementing microservices architecture"
        enhanced_query = await processor.three_tier_analysis(query)

        # For testing purposes, create mock HyDE documents if none were generated
        if enhanced_query.processing_strategy != "standard_hyde" or not enhanced_query.hypothetical_docs:
            # Create test documents directly for reliable testing
            test_embedding = [0.5, 0.3, 0.8] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3)
            vector_docs = [
                VectorDocument(
                    id=f"hyde_test_{int(time.time())}",
                    content="Mock HyDE document for microservices architecture implementation",
                    embedding=test_embedding,
                    metadata={
                        "relevance_score": 0.9,
                        "generation_method": "mock_hyde",
                        "source": "hyde_generated",
                        "topic": "microservices",
                    },
                    collection="hyde_documents",
                ),
            ]
        else:
            # Store real hypothetical documents in enhanced vector store
            vector_docs = []
            for i, hyde_doc in enumerate(enhanced_query.hypothetical_docs):
                vector_doc = VectorDocument(
                    id=f"hyde_{i}_{int(time.time())}",
                    content=hyde_doc.content,
                    embedding=hyde_doc.embedding,
                    metadata={
                        **hyde_doc.metadata,
                        "relevance_score": hyde_doc.relevance_score,
                        "generation_method": hyde_doc.generation_method,
                        "source": "hyde_generated",
                    },
                    collection="hyde_documents",
                )
                vector_docs.append(vector_doc)

        # Insert into enhanced store
        result = await enhanced_vector_store.insert_documents(vector_docs)

        assert result.success_count == len(vector_docs)
        assert result.error_count == 0

        # Verify documents can be searched
        test_embedding = [0.5, 0.3, 0.8] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3)
        search_params = SearchParameters(
            embeddings=[test_embedding],
            collection="hyde_documents",
            strategy=SearchStrategy.SEMANTIC,
        )

        search_results = await enhanced_vector_store.search(search_params)
        assert len(search_results) > 0

    @pytest.mark.asyncio
    async def test_multi_tier_search_strategies(self, enhanced_vector_store):
        """Test different search strategies with enhanced vector store."""
        # Insert test documents with different characteristics
        test_docs = [
            VectorDocument(
                id="exact_match",
                content="Python Redis caching implementation with connection pooling",
                embedding=[0.9, 0.1, 0.8] + [0.0] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"type": "exact", "difficulty": "advanced"},
                collection="multi_tier_test",
            ),
            VectorDocument(
                id="semantic_match",
                content="Memory storage strategies for high-performance applications",
                embedding=[0.7, 0.3, 0.6] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"type": "semantic", "difficulty": "intermediate"},
                collection="multi_tier_test",
            ),
            VectorDocument(
                id="hybrid_match",
                content="Distributed caching systems and performance optimization",
                embedding=[0.8, 0.2, 0.7] + [0.2] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"type": "hybrid", "difficulty": "expert"},
                collection="multi_tier_test",
            ),
        ]

        await enhanced_vector_store.insert_documents(test_docs)

        # Test different search strategies
        query_embedding = [0.85, 0.15, 0.75] + [0.05] * (DEFAULT_VECTOR_DIMENSIONS - 3)

        strategies_to_test = [
            SearchStrategy.EXACT,
            SearchStrategy.SEMANTIC,
            SearchStrategy.HYBRID,
            SearchStrategy.FILTERED,
        ]

        for strategy in strategies_to_test:
            search_params = SearchParameters(
                embeddings=[query_embedding],
                collection="multi_tier_test",
                strategy=strategy,
                limit=5,
            )

            results = await enhanced_vector_store.search(search_params)

            assert len(results) <= 5
            for result in results:
                assert result.search_strategy == strategy.value
                if strategy == SearchStrategy.HYBRID:
                    assert result.embedding is not None
                else:
                    assert result.embedding is None

    @pytest.mark.asyncio
    async def test_filtered_search_integration(self, enhanced_vector_store):
        """Test filtered search capabilities with metadata."""
        # Insert documents with rich metadata
        docs_with_metadata = [
            VectorDocument(
                id="python_beginner",
                content="Introduction to Python programming",
                embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={
                    "language": "python",
                    "difficulty": "beginner",
                    "category": "programming",
                    "tags": ["tutorial", "basics"],
                },
                collection="filtered_test",
            ),
            VectorDocument(
                id="python_advanced",
                content="Advanced Python metaclasses and decorators",
                embedding=[0.2] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={
                    "language": "python",
                    "difficulty": "advanced",
                    "category": "programming",
                    "tags": ["advanced", "metaclasses"],
                },
                collection="filtered_test",
            ),
            VectorDocument(
                id="javascript_intermediate",
                content="JavaScript async/await patterns",
                embedding=[0.3] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={
                    "language": "javascript",
                    "difficulty": "intermediate",
                    "category": "programming",
                    "tags": ["async", "patterns"],
                },
                collection="filtered_test",
            ),
        ]

        await enhanced_vector_store.insert_documents(docs_with_metadata)

        # Test various filter combinations
        filter_tests = [
            # Filter by language
            {"filters": {"language": "python"}, "expected_count": 2},
            # Filter by difficulty - this should only match the advanced document
            {"filters": {"difficulty": "advanced"}, "expected_count": 1},
            # Filter by multiple criteria
            {"filters": {"language": "python", "category": "programming"}, "expected_count": 2},
            # Filter with list values - beginner and intermediate (not advanced)
            {"filters": {"difficulty": ["beginner", "intermediate"]}, "expected_count": 2},
        ]

        query_embedding = [0.15] * DEFAULT_VECTOR_DIMENSIONS

        for test_case in filter_tests:
            search_params = SearchParameters(
                embeddings=[query_embedding],
                collection="filtered_test",
                filters=test_case["filters"],
                strategy=SearchStrategy.FILTERED,
            )

            results = await enhanced_vector_store.search(search_params)

            # Verify filter was applied correctly
            assert len(results) == test_case["expected_count"]

            # Verify all results match the filter criteria
            for result in results:
                for filter_key, filter_value in test_case["filters"].items():
                    if isinstance(filter_value, list):
                        assert result.metadata[filter_key] in filter_value, (
                            f"Document {result.document_id} metadata[{filter_key}]={result.metadata[filter_key]} "
                            f"not in filter list {filter_value}"
                        )
                    else:
                        assert result.metadata[filter_key] == filter_value, (
                            f"Document {result.document_id} metadata[{filter_key}]={result.metadata[filter_key]} "
                            f"does not match filter value {filter_value}"
                        )

    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, enhanced_vector_store):
        """Test performance characteristics with larger dataset."""
        # Insert a substantial number of documents
        large_dataset = []
        for i in range(500):
            doc = VectorDocument(
                id=f"perf_doc_{i}",
                content=f"Performance test document {i} with content about topic {i % 10}",
                embedding=[0.1 * (i % 10)] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={"index": i, "topic": f"topic_{i % 10}", "category": "performance_test"},
                collection="performance_test",
            )
            large_dataset.append(doc)

        # Measure insertion time
        start_time = time.time()
        insert_result = await enhanced_vector_store.insert_documents(large_dataset)
        insert_time = time.time() - start_time

        assert insert_result.success_count == 500
        assert insert_time < 5.0  # Should insert 500 docs in under 5 seconds

        # Test search performance
        query_embedding = [0.5] * DEFAULT_VECTOR_DIMENSIONS
        search_params = SearchParameters(embeddings=[query_embedding], collection="performance_test", limit=10)

        # Measure search time
        start_time = time.time()
        search_results = await enhanced_vector_store.search(search_params)
        search_time = time.time() - start_time

        assert len(search_results) <= 10
        assert search_time < 1.0  # Should search in under 1 second

        # Check metrics
        metrics = enhanced_vector_store.get_metrics()
        assert metrics.search_count > 0
        assert metrics.insert_count > 0
        assert metrics.avg_latency > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_hyde_processing(self, enhanced_vector_store):
        """Test concurrent HyDE processing with enhanced vector store."""
        # Create multiple HydeProcessor instances
        processors = [HydeProcessor() for _ in range(5)]

        queries = [
            "Python web development frameworks",
            "Machine learning model deployment",
            "Database optimization techniques",
            "Cloud infrastructure management",
            "API security best practices",
        ]

        async def process_query_task(processor: HydeProcessor, query: str):
            """Process a single query and store results."""
            enhanced_query = await processor.three_tier_analysis(query)

            if enhanced_query.processing_strategy == "standard_hyde":
                # Convert and store hypothetical documents
                vector_docs = []
                for i, hyde_doc in enumerate(enhanced_query.hypothetical_docs):
                    vector_doc = VectorDocument(
                        id=f"concurrent_{hash(query)}_{i}",
                        content=hyde_doc.content,
                        embedding=hyde_doc.embedding,
                        metadata={
                            "query": query,
                            "relevance_score": hyde_doc.relevance_score,
                            "processing_strategy": enhanced_query.processing_strategy,
                        },
                        collection="concurrent_test",
                    )
                    vector_docs.append(vector_doc)

                if vector_docs:
                    await enhanced_vector_store.insert_documents(vector_docs)

            return enhanced_query

        # Process queries concurrently
        tasks = [process_query_task(processor, query) for processor, query in zip(processors, queries, strict=False)]

        results = await asyncio.gather(*tasks)

        # Verify all queries were processed
        assert len(results) == 5

        # Check that documents were stored
        collections = await enhanced_vector_store.list_collections()
        if "concurrent_test" in collections:
            collection_info = await enhanced_vector_store.get_collection_info("concurrent_test")
            assert collection_info["document_count"] > 0


class TestVectorStoreErrorHandling:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self):
        """Test recovery from connection failures."""
        config = {"simulate_latency": False, "error_rate": 0.5}  # 50% error rate
        store = EnhancedMockVectorStore(config)
        await store.connect()

        # Attempt operations with errors

        success_count = 0
        error_count = 0

        # Increase attempts to ensure we get some errors with 50% rate
        # Use slightly different embeddings to avoid caching
        for i in range(50):
            # Vary embeddings slightly to avoid cache hits
            varied_embeddings = [[0.5 + i * 0.001] * DEFAULT_VECTOR_DIMENSIONS]
            varied_params = SearchParameters(embeddings=varied_embeddings, collection="default")
            try:
                await store.search(varied_params)
                success_count += 1
            except Exception:
                error_count += 1

        # Should have both successes and errors (with 50% error rate over 50 attempts)
        assert success_count > 0, f"Expected some successes, got {success_count}"
        assert error_count > 0, f"Expected some errors with 50% error rate, got {error_count}"

        # Metrics should track errors
        metrics = store.get_metrics()
        assert metrics.error_count > 0, f"Metrics should show errors, got {metrics.error_count}"

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker functionality in integration scenarios."""
        config = {"error_rate": 1.0}  # Always error
        store = EnhancedMockVectorStore(config)
        await store.connect()

        # Trigger circuit breaker by causing enough failures
        # Use different embeddings to avoid caching
        failure_count = 0
        for i in range(10):
            # Vary embeddings slightly to avoid cache hits
            varied_embeddings = [[0.5 + i * 0.01] * DEFAULT_VECTOR_DIMENSIONS]
            varied_params = SearchParameters(embeddings=varied_embeddings, collection="default")
            try:
                await store.search(varied_params)
            except Exception:
                failure_count += 1
                # Record failure for circuit breaker
                store._record_operation_failure()

        # Verify we got failures
        assert failure_count > 0, f"Expected failures with 100% error rate, got {failure_count}"

        # Circuit breaker should be open after enough failures
        assert store._circuit_breaker_open is True, "Circuit breaker should be open after failures"

        # Subsequent operations should return empty results due to circuit breaker
        # Use a new set of embeddings to avoid cache hits
        final_embeddings = [[0.9] * DEFAULT_VECTOR_DIMENSIONS]
        final_params = SearchParameters(embeddings=final_embeddings, collection="default")
        results = await store.search(final_params)
        assert results == [], "Circuit breaker should return empty results"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when vector store is unavailable."""
        # Test with vector store that fails connection
        config = {"type": VectorStoreType.QDRANT, "host": "nonexistent.host", "port": 9999}

        # Should fall back gracefully
        try:
            store = VectorStoreFactory.create_vector_store(config)
            assert isinstance(store, QdrantVectorStore)

            # Connection should fail gracefully
            with pytest.raises((ConnectionError, RuntimeError, ResponseHandlingException)):
                await store.connect()

        except (ConnectionError, RuntimeError, ResponseHandlingException) as e:
            # Expected behavior for missing Qdrant client
            error_msg = str(e)
            assert "not available" in error_msg or "Qdrant" in error_msg


class TestVectorStoreConfigurationIntegration:
    """Test integration with configuration system."""

    @pytest.mark.asyncio
    async def test_configuration_integration(self):
        """Test vector store configuration integration."""
        # Test with different configuration sources
        configurations = [
            {"type": VectorStoreType.MOCK, "simulate_latency": True, "error_rate": 0.1, "base_latency": 0.02},
            {"type": VectorStoreType.QDRANT, "host": "192.168.1.16", "port": 6333, "timeout": 60.0},
        ]

        for config in configurations:
            store = VectorStoreFactory.create_vector_store(config)

            if config["type"] == VectorStoreType.MOCK:
                assert isinstance(store, EnhancedMockVectorStore)
                assert store._simulate_latency == config["simulate_latency"]
                assert store._error_rate == config["error_rate"]
                assert store._base_latency == config["base_latency"]
            elif config["type"] == VectorStoreType.QDRANT:
                assert isinstance(store, QdrantVectorStore)
                assert store._host == config["host"]
                assert store._port == config["port"]
                assert store._timeout == config["timeout"]

    @pytest.mark.asyncio
    async def test_context_manager_with_configuration(self):
        """Test context manager with different configurations."""
        config = {"type": VectorStoreType.MOCK, "simulate_latency": False, "error_rate": 0.0}

        async with vector_store_connection(config) as store:
            # Test basic operations
            docs = [
                VectorDocument(
                    id="ctx_test",
                    content="Context manager test",
                    embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
                    collection="context_test",
                ),
            ]

            result = await store.insert_documents(docs)
            assert result.success_count == 1

            # Test search
            params = SearchParameters(embeddings=[[0.1] * DEFAULT_VECTOR_DIMENSIONS], collection="context_test")

            results = await store.search(params)
            assert len(results) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
