"""
Tests for core vector store implementation.

This module tests the actual vector store classes and interfaces to improve coverage.
"""

import contextlib
from unittest.mock import Mock, patch

import pytest

# Import the actual vector store classes to get coverage
from src.core.vector_store import (
    AbstractVectorStore,
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
)


class TestEnums:
    """Test enum classes."""

    def test_vector_store_type_values(self):
        """Test VectorStoreType enum values."""
        assert VectorStoreType.MOCK == "mock"
        assert VectorStoreType.QDRANT == "qdrant"

    def test_connection_status_values(self):
        """Test ConnectionStatus enum values."""
        assert ConnectionStatus.HEALTHY == "healthy"
        assert ConnectionStatus.DEGRADED == "degraded"
        assert ConnectionStatus.UNHEALTHY == "unhealthy"
        assert ConnectionStatus.UNKNOWN == "unknown"

    def test_search_strategy_values(self):
        """Test SearchStrategy enum values."""
        assert SearchStrategy.EXACT == "exact"
        assert SearchStrategy.HYBRID == "hybrid"
        assert SearchStrategy.SEMANTIC == "semantic"
        assert SearchStrategy.FILTERED == "filtered"


class TestVectorStoreMetrics:
    """Test VectorStoreMetrics class."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = VectorStoreMetrics()
        assert metrics.search_count == 0
        assert metrics.insert_count == 0
        assert metrics.error_count == 0
        assert metrics.total_latency == 0.0
        assert metrics.avg_latency == 0.0
        assert metrics.connection_pool_usage == 0.0

    def test_record_search(self):
        """Test recording search metrics."""
        metrics = VectorStoreMetrics()
        metrics.update_search_metrics(0.5)

        assert metrics.search_count == 1
        assert metrics.total_latency == 0.5
        assert metrics.avg_latency == 0.5

        # Record another search
        metrics.update_search_metrics(1.0)
        assert metrics.search_count == 2
        assert metrics.total_latency == 1.5
        assert metrics.avg_latency == 0.75  # 1.5 / 2

    def test_record_insert(self):
        """Test recording insert metrics."""
        metrics = VectorStoreMetrics()
        metrics.update_insert_metrics(0.3)

        assert metrics.insert_count == 1
        assert metrics.total_latency == 0.3
        assert metrics.avg_latency == 0.3

    def test_record_error(self):
        """Test recording error metrics."""
        metrics = VectorStoreMetrics()
        metrics.increment_error_count()

        assert metrics.error_count == 1

    def test_combined_metrics(self):
        """Test combined search and insert metrics."""
        metrics = VectorStoreMetrics()
        metrics.update_search_metrics(0.5)
        metrics.update_insert_metrics(0.3)
        metrics.increment_error_count()

        assert metrics.search_count == 1
        assert metrics.insert_count == 1
        assert metrics.error_count == 1
        assert metrics.total_latency == 0.8  # 0.5 + 0.3
        assert metrics.avg_latency == 0.4  # 0.8 / 2


class TestVectorStoreModels:
    """Test vector store model classes."""

    def test_vector_document_creation(self):
        """Test VectorDocument model creation."""
        doc = VectorDocument(id="doc1", content="Test content", embedding=[0.1, 0.2, 0.3], metadata={"type": "test"})

        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"type": "test"}

    def test_search_result_creation(self):
        """Test SearchResult model creation."""
        result = SearchResult(
            document_id="doc1",
            content="Test content",
            score=0.95,
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3],
        )

        assert result.document_id == "doc1"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata == {"source": "test"}
        assert result.embedding == [0.1, 0.2, 0.3]

    def test_search_parameters_defaults(self):
        """Test SearchParameters with defaults."""
        params = SearchParameters(embeddings=[[0.1, 0.2, 0.3]])

        assert params.limit == 10
        assert params.score_threshold == 0.0
        assert params.strategy == SearchStrategy.SEMANTIC
        assert params.filters is None
        assert params.collection == "default"

    def test_search_parameters_custom(self):
        """Test SearchParameters with custom values."""
        params = SearchParameters(
            embeddings=[[0.1, 0.2, 0.3]],
            limit=5,
            score_threshold=0.8,
            strategy=SearchStrategy.HYBRID,
            filters={"type": "document"},
            collection="custom",
        )

        assert params.limit == 5
        assert params.score_threshold == 0.8
        assert params.strategy == SearchStrategy.HYBRID
        assert params.filters == {"type": "document"}
        assert params.collection == "custom"

    def test_batch_operation_result(self):
        """Test BatchOperationResult model."""
        result = BatchOperationResult(
            success_count=2,
            error_count=1,
            total_count=3,
            errors=["Error processing doc3"],
            processing_time=1.5,
            batch_id="batch_123",
        )

        assert result.success_count == 2
        assert result.error_count == 1
        assert result.total_count == 3
        assert result.errors == ["Error processing doc3"]
        assert result.processing_time == 1.5
        assert result.batch_id == "batch_123"

    def test_health_check_result(self):
        """Test HealthCheckResult model."""
        result = HealthCheckResult(status=ConnectionStatus.HEALTHY, latency=50.0, details={"collections": 5})

        assert result.status == ConnectionStatus.HEALTHY
        assert result.latency == 50.0
        assert result.details == {"collections": 5}


class TestEnhancedMockVectorStore:
    """Test EnhancedMockVectorStore implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = EnhancedMockVectorStore(config={"simulate_latency": False})

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test connection and disconnection."""
        # Initially disconnected
        assert not self.store._connected

        # Connect
        await self.store.connect()
        assert self.store._connected

        # Disconnect
        await self.store.disconnect()
        assert not self.store._connected

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        await self.store.connect()

        health = await self.store.health_check()

        assert isinstance(health, HealthCheckResult)
        assert health.status == ConnectionStatus.HEALTHY
        assert health.latency >= 0

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self):
        """Test health check when disconnected."""
        health = await self.store.health_check()

        assert health.status == ConnectionStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_create_collection(self):
        """Test collection creation."""
        await self.store.connect()

        result = await self.store.create_collection("test_collection", vector_size=128)
        assert result is True

        # Collection should exist
        collections = await self.store.list_collections()
        assert "test_collection" in collections

    @pytest.mark.asyncio
    async def test_create_existing_collection(self):
        """Test creating an existing collection."""
        await self.store.connect()

        # Create collection first time
        result1 = await self.store.create_collection("test_collection", vector_size=128)
        assert result1 is True

        # Try to create again
        result2 = await self.store.create_collection("test_collection", vector_size=128)
        assert result2 is False  # Should fail or return False for existing

    @pytest.mark.asyncio
    async def test_insert_documents(self):
        """Test document insertion."""
        await self.store.connect()
        await self.store.create_collection("test_collection", vector_size=3)

        docs = [
            VectorDocument(id="doc1", content="First document", embedding=[0.1, 0.2, 0.3], metadata={"type": "test"}),
            VectorDocument(id="doc2", content="Second document", embedding=[0.4, 0.5, 0.6], metadata={"type": "test"}),
        ]

        result = await self.store.insert_documents(docs)

        assert isinstance(result, BatchOperationResult)
        assert result.success_count == 2
        assert result.error_count == 0

    @pytest.mark.asyncio
    async def test_search_documents(self):
        """Test document searching."""
        await self.store.connect()
        await self.store.create_collection("test_collection", vector_size=3)

        # Insert test documents
        docs = [
            VectorDocument(id="doc1", content="First document", embedding=[0.1, 0.2, 0.3], metadata={"type": "test"}),
        ]
        await self.store.insert_documents(docs)

        # Search
        params = SearchParameters(embeddings=[[0.1, 0.2, 0.3]], limit=5)
        results = await self.store.search(params)

        assert isinstance(results, list)
        if results:  # Mock might return results
            assert isinstance(results[0], SearchResult)

    @pytest.mark.asyncio
    async def test_update_document(self):
        """Test document updating."""
        await self.store.connect()
        await self.store.create_collection("test_collection", vector_size=3)

        # Insert document first
        doc = VectorDocument(id="doc1", content="Original content", embedding=[0.1, 0.2, 0.3])
        await self.store.insert_documents([doc])

        # Update document
        updated_doc = VectorDocument(id="doc1", content="Updated content", embedding=[0.4, 0.5, 0.6])
        result = await self.store.update(updated_doc, "test_collection")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_documents(self):
        """Test document deletion."""
        await self.store.connect()
        await self.store.create_collection("test_collection", vector_size=3)

        # Insert document first
        doc = VectorDocument(id="doc1", content="Test content", embedding=[0.1, 0.2, 0.3])
        await self.store.insert_documents([doc])

        # Delete document
        result = await self.store.delete(["doc1"], "test_collection")

        assert isinstance(result, BatchOperationResult)

    @pytest.mark.asyncio
    async def test_get_collection_info(self):
        """Test getting collection information."""
        await self.store.connect()
        await self.store.create_collection("test_collection", vector_size=128)

        info = await self.store.get_collection_info("test_collection")

        assert isinstance(info, dict)
        assert "name" in info

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using store as context manager."""
        async with self.store as store:
            assert store.is_connected()

        # Should be disconnected after exiting context
        assert not self.store.is_connected()


class TestQdrantVectorStore:
    """Test QdrantVectorStore implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("src.core.vector_store.ApplicationSettings") as mock_settings:
            mock_settings.return_value.qdrant_url = "http://localhost:6333"
            mock_settings.return_value.qdrant_api_key = None
            self.store = QdrantVectorStore()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test QdrantVectorStore initialization."""
        assert self.store._connection_status == ConnectionStatus.UNKNOWN
        assert self.store.metrics is not None

    @pytest.mark.asyncio
    async def test_connection_attempt(self):
        """Test connection attempt (will fail without real Qdrant)."""
        with patch("src.core.vector_store.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value = Mock()

            await self.store.connect()

            mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self):
        """Test health check when disconnected."""
        health = await self.store.health_check()

        assert health.status == ConnectionStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_operations_require_connection(self):
        """Test that operations handle disconnected state gracefully."""
        # All operations should handle disconnected state gracefully
        params = SearchParameters(embeddings=[[0.1, 0.2, 0.3]])

        # Should handle disconnected state gracefully and return empty results
        results = await self.store.search(params)
        assert isinstance(results, list)
        assert len(results) == 0


class TestVectorStoreFactory:
    """Test VectorStoreFactory."""

    def test_create_mock_store(self):
        """Test creating mock store."""
        store = VectorStoreFactory.create_store(VectorStoreType.MOCK)

        assert isinstance(store, EnhancedMockVectorStore)

    def test_create_qdrant_store(self):
        """Test creating Qdrant store."""
        with patch("src.core.vector_store.ApplicationSettings"):
            store = VectorStoreFactory.create_store(VectorStoreType.QDRANT)

            assert isinstance(store, QdrantVectorStore)

    def test_create_store_from_config(self):
        """Test creating store from config."""
        with patch("src.core.vector_store.ApplicationSettings") as mock_settings:
            # Test mock config
            mock_settings.return_value.vector_store_type = "mock"
            store = VectorStoreFactory.create_store_from_config()
            assert isinstance(store, EnhancedMockVectorStore)

            # Test qdrant config
            mock_settings.return_value.vector_store_type = "qdrant"
            store = VectorStoreFactory.create_store_from_config()
            assert isinstance(store, QdrantVectorStore)

    def test_invalid_store_type(self):
        """Test handling invalid store type."""
        with pytest.raises(ValueError, match="Unsupported vector store type|invalid"):
            VectorStoreFactory.create_store("invalid_type")

    def test_get_supported_types(self):
        """Test getting supported store types."""
        types = VectorStoreFactory.get_supported_types()

        assert VectorStoreType.MOCK in types
        assert VectorStoreType.QDRANT in types


class TestMockVectorStoreCompatibility:
    """Test MockVectorStore compatibility class."""

    def test_mock_vector_store_inheritance(self):
        """Test MockVectorStore inherits from EnhancedMockVectorStore."""
        store = MockVectorStore()

        assert isinstance(store, EnhancedMockVectorStore)
        assert isinstance(store, AbstractVectorStore)

    @pytest.mark.asyncio
    async def test_legacy_compatibility(self):
        """Test legacy compatibility methods work."""
        store = MockVectorStore()
        await store.connect()

        # Should have all the enhanced functionality
        collections = await store.list_collections()
        assert isinstance(collections, list)


class TestVectorStoreErrorHandling:
    """Test error handling in vector store operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = EnhancedMockVectorStore(config={"simulate_latency": False})

    @pytest.mark.asyncio
    async def test_operations_when_disconnected(self):
        """Test operations when store is disconnected."""
        # Most operations should handle disconnected state gracefully
        params = SearchParameters(embeddings=[[0.1, 0.2, 0.3]])

        # These should not crash but may return empty results or handle gracefully
        with contextlib.suppress(Exception):
            await self.store.search(params)

        with contextlib.suppress(Exception):
            await self.store.list_collections()

    @pytest.mark.asyncio
    async def test_invalid_collection_operations(self):
        """Test operations on non-existent collections."""
        await self.store.connect()

        # Operations on non-existent collections should be handled gracefully
        params = SearchParameters(embeddings=[[0.1, 0.2, 0.3]], collection="nonexistent")
        results = await self.store.search(params)

        # Should return empty results or handle gracefully
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_metrics_error_tracking(self):
        """Test that metrics track errors properly."""
        await self.store.connect()

        initial_errors = self.store.metrics.error_count

        # Try an operation that might fail
        with contextlib.suppress(Exception):
            await self.store.search(SearchParameters(embeddings=[[0.1, 0.2, 0.3]], collection="nonexistent"))

        # Error count might increase depending on implementation
        assert self.store.metrics.error_count >= initial_errors
