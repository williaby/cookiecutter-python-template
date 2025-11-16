"""
Unit tests for vector store implementations.

This module provides comprehensive test coverage for the vector store abstract
interfaces and concrete implementations, including mock and Qdrant clients.
Tests cover functionality, performance, error handling, and integration scenarios.
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.performance_optimizer import clear_all_caches
from src.core.vector_store import (
    CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TIMEOUT,
    DEFAULT_VECTOR_DIMENSIONS,
    BatchOperationResult,
    ConnectionStatus,
    EnhancedMockVectorStore,
    HealthCheckResult,
    MockVectorStore,
    QdrantVectorStore,
    SearchParameters,
    SearchResult,
    SearchStrategy,
    VectorDocument,
    VectorStoreFactory,
    VectorStoreMetrics,
    VectorStoreType,
    vector_store_connection,
)

# Import for integration tests
try:
    from src.core.hyde_processor import HydeProcessor, HypotheticalDocument
except ImportError:
    HydeProcessor = None
    HypotheticalDocument = None


class TestVectorStoreModels:
    """Test Pydantic models used in vector store operations."""

    def test_vector_document_creation(self):
        """Test VectorDocument model creation and validation."""
        embedding = [0.1, 0.2, 0.3]
        metadata = {"category": "test", "difficulty": "easy"}

        doc = VectorDocument(
            id="test_doc_1",
            content="Test document content",
            embedding=embedding,
            metadata=metadata,
            collection="test_collection",
        )

        assert doc.id == "test_doc_1"
        assert doc.content == "Test document content"
        assert doc.embedding == embedding
        assert doc.metadata == metadata
        assert doc.collection == "test_collection"
        assert isinstance(doc.timestamp, float)

    def test_search_parameters_validation(self):
        """Test SearchParameters model validation."""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        params = SearchParameters(
            embeddings=embeddings,
            limit=5,
            collection="test",
            strategy=SearchStrategy.SEMANTIC,
            score_threshold=0.7,
        )

        assert params.embeddings == embeddings
        assert params.limit == 5
        assert params.collection == "test"
        assert params.strategy == SearchStrategy.SEMANTIC
        assert params.score_threshold == 0.7

    def test_search_parameters_defaults(self):
        """Test SearchParameters default values."""
        embeddings = [[0.1, 0.2]]
        params = SearchParameters(embeddings=embeddings)

        assert params.limit == DEFAULT_SEARCH_LIMIT
        assert params.collection == "default"
        assert params.strategy == SearchStrategy.SEMANTIC
        assert params.score_threshold == 0.0
        assert params.timeout == DEFAULT_TIMEOUT

    def test_search_result_creation(self):
        """Test SearchResult model creation."""
        result = SearchResult(
            document_id="doc_1",
            content="Test content",
            score=0.85,
            metadata={"type": "test"},
            source="mock_store",
        )

        assert result.document_id == "doc_1"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.metadata == {"type": "test"}
        assert result.source == "mock_store"

    def test_batch_operation_result(self):
        """Test BatchOperationResult model."""
        result = BatchOperationResult(
            success_count=8,
            error_count=2,
            total_count=10,
            errors=["Error 1", "Error 2"],
            processing_time=1.5,
            batch_id="batch_123",
        )

        assert result.success_count == 8
        assert result.error_count == 2
        assert result.total_count == 10
        assert len(result.errors) == 2
        assert result.processing_time == 1.5
        assert result.batch_id == "batch_123"

    def test_health_check_result(self):
        """Test HealthCheckResult model."""
        result = HealthCheckResult(
            status=ConnectionStatus.HEALTHY,
            latency=0.05,
            details={"connections": 5},
            error_message=None,
        )

        assert result.status == ConnectionStatus.HEALTHY
        assert result.latency == 0.05
        assert result.details == {"connections": 5}
        assert result.error_message is None
        assert isinstance(result.timestamp, float)


class TestVectorStoreMetrics:
    """Test performance metrics functionality."""

    def test_metrics_initialization(self):
        """Test VectorStoreMetrics initialization."""
        metrics = VectorStoreMetrics()

        assert metrics.search_count == 0
        assert metrics.insert_count == 0
        assert metrics.total_latency == 0.0
        assert metrics.avg_latency == 0.0
        assert metrics.error_count == 0
        assert isinstance(metrics.last_operation_time, float)

    def test_search_metrics_update(self):
        """Test search metrics update."""
        metrics = VectorStoreMetrics()

        metrics.update_search_metrics(0.1)
        assert metrics.search_count == 1
        assert metrics.total_latency == 0.1
        assert metrics.avg_latency == 0.1

        metrics.update_search_metrics(0.2)
        assert metrics.search_count == 2
        assert metrics.total_latency == pytest.approx(0.3)
        assert metrics.avg_latency == pytest.approx(0.15)

    def test_insert_metrics_update(self):
        """Test insert metrics update."""
        metrics = VectorStoreMetrics()

        metrics.update_insert_metrics(0.5)
        assert metrics.insert_count == 1
        assert metrics.total_latency == 0.5
        assert metrics.avg_latency == 0.5

    def test_combined_metrics_update(self):
        """Test combined search and insert metrics."""
        metrics = VectorStoreMetrics()

        metrics.update_search_metrics(0.1)
        metrics.update_insert_metrics(0.3)

        assert metrics.search_count == 1
        assert metrics.insert_count == 1
        assert metrics.total_latency == 0.4
        assert metrics.avg_latency == 0.2

    def test_error_count_increment(self):
        """Test error count increment."""
        metrics = VectorStoreMetrics()

        metrics.increment_error_count()
        assert metrics.error_count == 1

        metrics.increment_error_count()
        assert metrics.error_count == 2


class TestEnhancedMockVectorStore:
    """Test EnhancedMockVectorStore implementation."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear all caches before each test."""
        clear_all_caches()

    @pytest.fixture
    def mock_config(self):
        """Provide mock configuration."""
        return {"simulate_latency": False, "error_rate": 0.0, "base_latency": 0.001}  # Disable for faster tests

    @pytest.fixture
    async def mock_store(self, mock_config):
        """Provide connected mock vector store."""
        store = EnhancedMockVectorStore(mock_config)
        await store.connect()
        return store

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_config):
        """Test connection and disconnection."""
        store = EnhancedMockVectorStore(mock_config)

        # Initial state
        assert store.get_connection_status() == ConnectionStatus.UNKNOWN

        # Connect
        await store.connect()
        assert store.get_connection_status() == ConnectionStatus.HEALTHY

        # Disconnect
        await store.disconnect()
        assert store.get_connection_status() == ConnectionStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_health_check_connected(self, mock_store):
        """Test health check when connected."""
        result = await mock_store.health_check()

        assert result.status == ConnectionStatus.HEALTHY
        assert result.latency >= 0
        assert result.details["connected"] is True
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, mock_config):
        """Test health check when disconnected."""
        store = EnhancedMockVectorStore(mock_config)
        result = await store.health_check()

        assert result.status == ConnectionStatus.UNHEALTHY
        assert result.error_message == "Not connected"
        assert result.details["connected"] is False

    @pytest.mark.asyncio
    async def test_basic_search(self, mock_store):
        """Test basic search functionality."""
        embeddings = [[0.8, 0.2, 0.9] + [0.0] * (DEFAULT_VECTOR_DIMENSIONS - 3)]
        params = SearchParameters(embeddings=embeddings, limit=3)

        results = await mock_store.search(params)

        assert isinstance(results, list)
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, SearchResult)
            assert 0.0 <= result.score <= 1.0
            assert result.source == "enhanced_mock_vector_store"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_store):
        """Test search with metadata filters."""
        embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]
        filters = {"category": "performance"}
        params = SearchParameters(embeddings=embeddings, filters=filters)

        results = await mock_store.search(params)

        # Should find at least one result with performance category
        performance_results = [r for r in results if r.metadata.get("category") == "performance"]
        assert len(performance_results) > 0

    @pytest.mark.asyncio
    async def test_search_score_threshold(self, mock_store):
        """Test search with score threshold."""
        embeddings = [[0.1] * DEFAULT_VECTOR_DIMENSIONS]
        params = SearchParameters(embeddings=embeddings, score_threshold=0.8)

        results = await mock_store.search(params)

        # All results should meet threshold
        for result in results:
            assert result.score >= 0.8

    @pytest.mark.asyncio
    async def test_search_different_strategies(self, mock_store):
        """Test different search strategies."""
        embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]

        for strategy in SearchStrategy:
            params = SearchParameters(embeddings=embeddings, strategy=strategy)
            results = await mock_store.search(params)

            for result in results:
                assert result.search_strategy == strategy.value
                if strategy == SearchStrategy.HYBRID:
                    assert result.embedding is not None
                else:
                    assert result.embedding is None

    @pytest.mark.asyncio
    async def test_insert_documents(self, mock_store):
        """Test document insertion."""
        docs = [
            VectorDocument(
                id="test_1",
                content="Test document 1",
                embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={"type": "test"},
                collection="test_collection",
            ),
            VectorDocument(
                id="test_2",
                content="Test document 2",
                embedding=[0.2] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={"type": "test"},
                collection="test_collection",
            ),
        ]

        result = await mock_store.insert_documents(docs)

        assert result.success_count == 2
        assert result.error_count == 0
        assert result.total_count == 2
        assert len(result.errors) == 0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_insert_creates_collection(self, mock_store):
        """Test that inserting documents creates collection if needed."""
        docs = [
            VectorDocument(
                id="new_test_1",
                content="New test document",
                embedding=[0.3] * DEFAULT_VECTOR_DIMENSIONS,
                collection="new_collection",
            ),
        ]

        # Collection shouldn't exist initially
        collections = await mock_store.list_collections()
        assert "new_collection" not in collections

        # Insert should create collection
        await mock_store.insert_documents(docs)
        collections = await mock_store.list_collections()
        assert "new_collection" in collections

    @pytest.mark.asyncio
    async def test_update_document(self, mock_store):
        """Test document update."""
        # Insert initial document
        doc = VectorDocument(
            id="update_test",
            content="Original content",
            embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
            collection="default",
        )
        await mock_store.insert_documents([doc])

        # Update document
        updated_doc = VectorDocument(
            id="update_test",
            content="Updated content",
            embedding=[0.2] * DEFAULT_VECTOR_DIMENSIONS,
            collection="default",
        )

        result = await mock_store.update_document(updated_doc)
        assert result is True

    @pytest.mark.asyncio
    async def test_update_nonexistent_document(self, mock_store):
        """Test updating non-existent document."""
        doc = VectorDocument(
            id="nonexistent",
            content="Content",
            embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
            collection="default",
        )

        result = await mock_store.update_document(doc)
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_documents(self, mock_store):
        """Test document deletion."""
        # Insert test documents
        docs = [
            VectorDocument(
                id="delete_test_1",
                content="Delete test 1",
                embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
                collection="delete_test",
            ),
            VectorDocument(
                id="delete_test_2",
                content="Delete test 2",
                embedding=[0.2] * DEFAULT_VECTOR_DIMENSIONS,
                collection="delete_test",
            ),
        ]
        await mock_store.insert_documents(docs)

        # Delete documents
        result = await mock_store.delete_documents(["delete_test_1", "delete_test_2"], "delete_test")

        assert result.success_count == 2
        assert result.error_count == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_documents(self, mock_store):
        """Test deleting non-existent documents."""
        result = await mock_store.delete_documents(["nonexistent_1", "nonexistent_2"], "default")

        assert result.success_count == 0
        assert result.error_count == 2
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_create_collection(self, mock_store):
        """Test collection creation."""
        result = await mock_store.create_collection("new_test_collection", 512)
        assert result is True

        collections = await mock_store.list_collections()
        assert "new_test_collection" in collections

    @pytest.mark.asyncio
    async def test_create_existing_collection(self, mock_store):
        """Test creating existing collection."""
        await mock_store.create_collection("duplicate_collection")
        result = await mock_store.create_collection("duplicate_collection")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_store):
        """Test listing collections."""
        collections = await mock_store.list_collections()
        assert isinstance(collections, list)
        assert "default" in collections

    @pytest.mark.asyncio
    async def test_get_collection_info(self, mock_store):
        """Test getting collection information."""
        info = await mock_store.get_collection_info("default")

        assert isinstance(info, dict)
        assert "vector_size" in info
        assert "document_count" in info
        assert info["vector_size"] == DEFAULT_VECTOR_DIMENSIONS

    @pytest.mark.asyncio
    async def test_get_nonexistent_collection_info(self, mock_store):
        """Test getting info for non-existent collection."""
        with pytest.raises(ValueError, match="Collection not found"):
            await mock_store.get_collection_info("nonexistent")

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, mock_store):
        """Test that operations update metrics."""
        initial_metrics = mock_store.get_metrics()
        initial_search_count = initial_metrics.search_count
        initial_insert_count = initial_metrics.insert_count

        # Perform operations
        embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]
        params = SearchParameters(embeddings=embeddings)
        await mock_store.search(params)

        docs = [VectorDocument(id="metrics_test", content="Metrics test", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS)]
        await mock_store.insert_documents(docs)

        # Check metrics updated
        updated_metrics = mock_store.get_metrics()
        assert updated_metrics.search_count == initial_search_count + 1
        assert updated_metrics.insert_count == initial_insert_count + 1
        assert updated_metrics.avg_latency > 0

    @pytest.mark.asyncio
    async def test_error_simulation(self):
        """Test error simulation functionality."""
        config = {"simulate_latency": False, "error_rate": 1.0}  # Always error
        store = EnhancedMockVectorStore(config)
        await store.connect()

        embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]
        params = SearchParameters(embeddings=embeddings)

        with pytest.raises(RuntimeError, match="Simulated error"):
            await store.search(params)

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self):
        """Test circuit breaker behavior."""
        config = {"error_rate": 1.0}  # Always error
        store = EnhancedMockVectorStore(config)
        await store.connect()

        embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]
        params = SearchParameters(embeddings=embeddings)

        # Trigger circuit breaker
        for _ in range(CIRCUIT_BREAKER_THRESHOLD + 1):
            with contextlib.suppress(RuntimeError):
                await store.search(params)

        # Circuit breaker should now be open
        assert store._circuit_breaker_open is True


class TestMockVectorStoreCompatibility:
    """Test backward compatibility with existing HydeProcessor."""

    @pytest.mark.asyncio
    async def test_mock_vector_store_creation(self):
        """Test MockVectorStore creation for backward compatibility."""
        store = MockVectorStore()

        # Should auto-connect
        await asyncio.sleep(0.2)  # Give time for auto-connect
        assert store.get_connection_status() == ConnectionStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_legacy_search_interface(self):
        """Test legacy search interface."""
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Wait for auto-connect

        embeddings = [[0.5, 0.3, 0.8] + [0.0] * (DEFAULT_VECTOR_DIMENSIONS - 3)]
        results = await store.search(embeddings, limit=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_legacy_add_documents_interface(self):
        """Test legacy add_documents interface."""
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Wait for auto-connect

        # Mock HypotheticalDocument objects
        mock_docs = []
        for i in range(3):
            mock_doc = Mock()
            mock_doc.content = f"Hypothetical document {i}"
            mock_doc.embedding = [0.1 * i] * DEFAULT_VECTOR_DIMENSIONS
            mock_doc.metadata = {"type": "hypothetical", "index": i}
            mock_docs.append(mock_doc)

        result = await store.add_documents(mock_docs)
        assert result is True


class TestVectorStoreFactory:
    """Test VectorStoreFactory functionality."""

    def test_create_mock_store(self):
        """Test creating mock vector store."""
        config = {"type": VectorStoreType.MOCK}
        store = VectorStoreFactory.create_vector_store(config)

        assert isinstance(store, EnhancedMockVectorStore)

    def test_create_qdrant_store(self):
        """Test creating Qdrant vector store."""
        config = {"type": VectorStoreType.QDRANT, "host": "localhost", "port": 6333}
        store = VectorStoreFactory.create_vector_store(config)

        assert isinstance(store, QdrantVectorStore)

    def test_auto_detect_mock(self):
        """Test auto-detection falls back to mock."""
        config = {"type": VectorStoreType.AUTO}
        store = VectorStoreFactory.create_vector_store(config)

        # Should fall back to mock when Qdrant not available
        assert isinstance(store, EnhancedMockVectorStore)

    def test_invalid_store_type(self):
        """Test invalid store type raises error."""
        config = {"type": "invalid"}

        with pytest.raises(ValueError, match="Unknown vector store type"):
            VectorStoreFactory.create_vector_store(config)


class TestVectorStoreConnection:
    """Test async context manager for vector store connections."""

    @pytest.mark.asyncio
    async def test_connection_context_manager(self):
        """Test vector store connection context manager."""
        config = {"type": VectorStoreType.MOCK, "simulate_latency": False}

        async with vector_store_connection(config) as store:
            assert isinstance(store, EnhancedMockVectorStore)
            assert store.get_connection_status() == ConnectionStatus.HEALTHY

            # Test basic functionality
            embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]
            params = SearchParameters(embeddings=embeddings)
            results = await store.search(params)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_connection_cleanup(self):
        """Test that connection is properly cleaned up."""
        config = {"type": VectorStoreType.MOCK, "simulate_latency": False}
        store_ref = None

        async with vector_store_connection(config) as store:
            store_ref = store
            assert store.get_connection_status() == ConnectionStatus.HEALTHY

        # After context exit, should be disconnected
        assert store_ref.get_connection_status() == ConnectionStatus.UNKNOWN


class TestQdrantVectorStore:
    """Test QdrantVectorStore implementation (mocked)."""

    @pytest.fixture
    def qdrant_config(self):
        """Provide Qdrant configuration."""
        return {"host": "192.168.1.16", "port": 6333, "api_key": "test_key", "timeout": 30.0}

    @pytest.mark.asyncio
    @patch("src.core.vector_store.QDRANT_AVAILABLE", False)
    async def test_qdrant_connection_without_client(self, qdrant_config):
        """Test Qdrant connection when client library not available."""
        store = QdrantVectorStore(qdrant_config)

        # Should raise error when qdrant-client not available
        with pytest.raises(RuntimeError, match="Qdrant client not available"):
            await store.connect()

    @pytest.mark.skip(reason="Tracemalloc issue with pytest internal error handling")
    @pytest.mark.asyncio
    @patch("src.core.vector_store.QdrantClient")
    async def test_qdrant_connection_success(self, mock_qdrant_client, qdrant_config):
        """Test successful Qdrant connection."""
        # Mock successful connection
        mock_client_instance = AsyncMock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        store = QdrantVectorStore(qdrant_config)

        # Mock QDRANT_AVAILABLE as True for this test
        with patch("src.core.vector_store.QDRANT_AVAILABLE", True), patch("builtins.__import__", return_value=Mock()):
            await store.connect()

            assert store.get_connection_status() == ConnectionStatus.HEALTHY
            mock_qdrant_client.assert_called_once_with(
                host="192.168.1.16",
                port=6333,
                api_key="test_key",
                timeout=30.0,
            )

    @pytest.mark.asyncio
    async def test_qdrant_health_check_healthy(self, qdrant_config):
        """Test Qdrant health check when healthy."""
        store = QdrantVectorStore(qdrant_config)

        # Mock the client directly after creation
        mock_client_instance = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(), Mock()]
        mock_client_instance.get_collections.return_value = mock_collections

        # Set up the store with mocked client
        store._client = mock_client_instance
        store._connection_status = ConnectionStatus.HEALTHY

        result = await store.health_check()

        assert result.status == ConnectionStatus.HEALTHY
        assert result.latency > 0
        assert result.details["collections_count"] == 2
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_qdrant_health_check_no_client(self, qdrant_config):
        """Test Qdrant health check when not connected."""
        store = QdrantVectorStore(qdrant_config)

        result = await store.health_check()

        assert result.status == ConnectionStatus.UNHEALTHY
        assert result.error_message == "Not connected"

    @pytest.mark.skip(reason="Interface delegation to factory causes test interference")
    @pytest.mark.asyncio
    @patch("src.core.vector_store.QDRANT_AVAILABLE", True)
    async def test_qdrant_search_no_client(self, qdrant_config):
        """Test Qdrant search when not connected."""
        store = QdrantVectorStore(qdrant_config)
        # Ensure no client is set
        store._client = None

        # Mock circuit breaker to return True
        store._handle_circuit_breaker = AsyncMock(return_value=True)

        embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]
        params = SearchParameters(embeddings=embeddings)

        results = await store.search(params)
        assert results == []

    def test_qdrant_configuration(self, qdrant_config):
        """Test Qdrant store configuration."""
        store = QdrantVectorStore(qdrant_config)

        assert store._host == "192.168.1.16"
        assert store._port == 6333
        assert store._api_key == "test_key"
        assert store._timeout == 30.0


class TestIntegrationWithHydeProcessor:
    """Test integration between vector store and HydeProcessor."""

    @pytest.mark.asyncio
    async def test_hyde_processor_integration(self):
        """Test vector store integration with HydeProcessor."""
        if HydeProcessor is None:
            pytest.skip("HydeProcessor not available")

        # Create enhanced mock store
        config = {"simulate_latency": False, "error_rate": 0.0}
        vector_store = EnhancedMockVectorStore(config)
        await vector_store.connect()

        # Create compatible mock store for HydeProcessor
        mock_store = MockVectorStore()
        await asyncio.sleep(0.1)  # Wait for auto-connect

        # Create HydeProcessor with mock store
        processor = HydeProcessor(vector_store=mock_store)

        # Test complete processing pipeline
        query = "How to implement caching in Python applications"
        results = await processor.process_query(query)

        assert results is not None
        assert hasattr(results, "results")
        assert hasattr(results, "hyde_enhanced")

    @pytest.mark.asyncio
    async def test_hypothetical_document_conversion(self):
        """Test conversion between HypotheticalDocument and VectorDocument."""
        if HypotheticalDocument is None:
            pytest.skip("HypotheticalDocument not available")

        # Create HypotheticalDocument
        hyde_doc = HypotheticalDocument(
            content="Test hypothetical document",
            relevance_score=0.9,
            embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
            generation_method="test",
            metadata={"type": "hypothetical"},
        )

        # Test conversion in MockVectorStore
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Wait for auto-connect

        result = await store.add_documents([hyde_doc])
        assert result is True


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""

    @pytest.mark.asyncio
    async def test_large_batch_insert(self):
        """Test inserting large batch of documents."""
        config = {"simulate_latency": False}
        store = EnhancedMockVectorStore(config)
        await store.connect()

        # Create large batch
        docs = []
        for i in range(1000):
            doc = VectorDocument(
                id=f"large_batch_{i}",
                content=f"Large batch document {i}",
                embedding=[0.1 * (i % 10)] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={"batch": "large", "index": i},
            )
            docs.append(doc)

        start_time = time.time()
        result = await store.insert_documents(docs)
        duration = time.time() - start_time

        assert result.success_count == 1000
        assert result.error_count == 0
        assert duration < 10.0  # Should complete in reasonable time

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent vector store operations."""
        config = {"simulate_latency": False}
        store = EnhancedMockVectorStore(config)
        await store.connect()

        async def search_task(task_id: int):
            embeddings = [[0.1 * task_id] * DEFAULT_VECTOR_DIMENSIONS]
            params = SearchParameters(embeddings=embeddings, limit=5)
            return await store.search(params)

        async def insert_task(task_id: int):
            docs = [
                VectorDocument(
                    id=f"concurrent_{task_id}",
                    content=f"Concurrent document {task_id}",
                    embedding=[0.1 * task_id] * DEFAULT_VECTOR_DIMENSIONS,
                ),
            ]
            return await store.insert_documents(docs)

        # Run concurrent operations
        search_tasks = [search_task(i) for i in range(10)]
        insert_tasks = [insert_task(i) for i in range(10)]

        search_results = await asyncio.gather(*search_tasks)
        insert_results = await asyncio.gather(*insert_tasks)

        # All operations should succeed
        assert len(search_results) == 10
        assert len(insert_results) == 10

        for result in insert_results:
            assert result.success_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
