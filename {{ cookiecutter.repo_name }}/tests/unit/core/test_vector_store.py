"""
Unit tests for vector store implementations.

This module provides comprehensive test coverage for the vector store system,
testing abstract interfaces, mock implementations, Qdrant integration,
performance metrics, connection management, and error handling.
Uses proper pytest markers for codecov integration per codecov.yml config component.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.performance_optimizer import clear_all_caches
from src.core.vector_store import (
    CIRCUIT_BREAKER_THRESHOLD,
    CONNECTION_POOL_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TIMEOUT,
    DEFAULT_VECTOR_DIMENSIONS,
    DEGRADED_LATENCY_THRESHOLD,
    HEALTH_CHECK_INTERVAL,
    MAX_RETRIES,
    QDRANT_AVAILABLE,
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
    VectorStore,
    VectorStoreFactory,
    VectorStoreMetrics,
    VectorStoreType,
    vector_store_connection,
)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all caches between tests to prevent test pollution."""
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.mark.unit
class TestVectorStoreMetrics:
    """Test cases for VectorStoreMetrics class."""

    def test_init_default_values(self):
        """Test VectorStoreMetrics initialization with default values."""
        metrics = VectorStoreMetrics()

        assert metrics.search_count == 0
        assert metrics.insert_count == 0
        assert metrics.total_latency == 0.0
        assert metrics.avg_latency == 0.0
        assert metrics.error_count == 0
        assert metrics.connection_pool_usage == 0.0
        assert isinstance(metrics.last_operation_time, float)

    def test_update_search_metrics(self):
        """Test updating search operation metrics."""
        metrics = VectorStoreMetrics()
        initial_time = metrics.last_operation_time

        # Update with first search
        metrics.update_search_metrics(0.5)

        assert metrics.search_count == 1
        assert metrics.insert_count == 0
        assert metrics.total_latency == 0.5
        assert metrics.avg_latency == 0.5
        assert metrics.last_operation_time > initial_time

    def test_update_insert_metrics(self):
        """Test updating insert operation metrics."""
        metrics = VectorStoreMetrics()
        initial_time = metrics.last_operation_time

        # Update with first insert
        metrics.update_insert_metrics(1.2)

        assert metrics.search_count == 0
        assert metrics.insert_count == 1
        assert metrics.total_latency == 1.2
        assert metrics.avg_latency == 1.2
        assert metrics.last_operation_time > initial_time

    def test_update_mixed_metrics(self):
        """Test updating both search and insert metrics."""
        metrics = VectorStoreMetrics()

        # Add search and insert operations
        metrics.update_search_metrics(0.3)
        metrics.update_insert_metrics(0.7)

        assert metrics.search_count == 1
        assert metrics.insert_count == 1
        assert metrics.total_latency == 1.0
        assert metrics.avg_latency == 0.5  # (0.3 + 0.7) / (1 + 1)

    def test_increment_error_count(self):
        """Test incrementing error counter."""
        metrics = VectorStoreMetrics()

        assert metrics.error_count == 0

        metrics.increment_error_count()
        assert metrics.error_count == 1

        metrics.increment_error_count()
        assert metrics.error_count == 2


@pytest.mark.unit
class TestVectorDocument:
    """Test cases for VectorDocument model."""

    def test_create_basic_document(self):
        """Test creating a basic vector document."""
        embedding = [0.1, 0.2, 0.3]

        doc = VectorDocument(id="test_doc_1", content="Test document content", embedding=embedding)

        assert doc.id == "test_doc_1"
        assert doc.content == "Test document content"
        assert doc.embedding == embedding
        assert doc.metadata == {}
        assert doc.collection == "default"
        assert isinstance(doc.timestamp, float)

    def test_create_document_with_metadata(self):
        """Test creating document with metadata."""
        metadata = {"category": "test", "priority": "high"}
        embedding = [0.4, 0.5, 0.6]

        doc = VectorDocument(
            id="test_doc_2",
            content="Document with metadata",
            embedding=embedding,
            metadata=metadata,
            collection="test_collection",
        )

        assert doc.metadata == metadata
        assert doc.collection == "test_collection"

    def test_document_timestamp_automatic(self):
        """Test that document timestamp is set automatically."""
        before_time = time.time()

        doc = VectorDocument(id="test_doc_3", content="Timestamp test", embedding=[0.1, 0.2])

        after_time = time.time()

        assert before_time <= doc.timestamp <= after_time


@pytest.mark.unit
class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_create_basic_search_result(self):
        """Test creating a basic search result."""
        result = SearchResult(document_id="doc_1", content="Result content", score=0.95)

        assert result.document_id == "doc_1"
        assert result.content == "Result content"
        assert result.score == 0.95
        assert result.metadata == {}
        assert result.source == "unknown"
        assert result.embedding is None
        assert result.search_strategy == "default"

    def test_create_complete_search_result(self):
        """Test creating a complete search result with all fields."""
        metadata = {"category": "ai", "type": "technical"}
        embedding = [0.1, 0.2, 0.3, 0.4]

        result = SearchResult(
            document_id="doc_2",
            content="Complete result content",
            score=0.87,
            metadata=metadata,
            source="enhanced_mock",
            embedding=embedding,
            search_strategy="hybrid",
        )

        assert result.metadata == metadata
        assert result.source == "enhanced_mock"
        assert result.embedding == embedding
        assert result.search_strategy == "hybrid"

    def test_score_validation(self):
        """Test that score validation works correctly."""
        # Valid scores
        result1 = SearchResult(document_id="doc_1", content="Test", score=0.0)
        assert result1.score == 0.0

        result2 = SearchResult(document_id="doc_2", content="Test", score=1.0)
        assert result2.score == 1.0

        result3 = SearchResult(document_id="doc_3", content="Test", score=0.5)
        assert result3.score == 0.5


@pytest.mark.unit
class TestSearchParameters:
    """Test cases for SearchParameters model."""

    def test_create_basic_parameters(self):
        """Test creating basic search parameters."""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        params = SearchParameters(embeddings=embeddings)

        assert params.embeddings == embeddings
        assert params.limit == DEFAULT_SEARCH_LIMIT
        assert params.collection == "default"
        assert params.filters is None
        assert params.strategy == SearchStrategy.SEMANTIC
        assert params.score_threshold == 0.0
        assert params.timeout == DEFAULT_TIMEOUT

    def test_create_custom_parameters(self):
        """Test creating search parameters with custom values."""
        embeddings = [[0.5, 0.6, 0.7]]
        filters = {"category": "tech"}

        params = SearchParameters(
            embeddings=embeddings,
            limit=25,
            collection="custom_collection",
            filters=filters,
            strategy=SearchStrategy.HYBRID,
            score_threshold=0.7,
            timeout=60.0,
        )

        assert params.limit == 25
        assert params.collection == "custom_collection"
        assert params.filters == filters
        assert params.strategy == SearchStrategy.HYBRID
        assert params.score_threshold == 0.7
        assert params.timeout == 60.0

    def test_limit_validation(self):
        """Test limit validation in search parameters."""
        embeddings = [[0.1, 0.2]]

        # Valid limits
        params1 = SearchParameters(embeddings=embeddings, limit=1)
        assert params1.limit == 1

        params2 = SearchParameters(embeddings=embeddings, limit=100)
        assert params2.limit == 100

    def test_score_threshold_validation(self):
        """Test score threshold validation."""
        embeddings = [[0.1, 0.2]]

        # Valid thresholds
        params1 = SearchParameters(embeddings=embeddings, score_threshold=0.0)
        assert params1.score_threshold == 0.0

        params2 = SearchParameters(embeddings=embeddings, score_threshold=1.0)
        assert params2.score_threshold == 1.0


@pytest.mark.unit
class TestBatchOperationResult:
    """Test cases for BatchOperationResult model."""

    def test_create_successful_result(self):
        """Test creating a successful batch operation result."""
        result = BatchOperationResult(
            success_count=10,
            error_count=0,
            total_count=10,
            processing_time=1.5,
            batch_id="batch_123",
        )

        assert result.success_count == 10
        assert result.error_count == 0
        assert result.total_count == 10
        assert result.errors == []
        assert result.processing_time == 1.5
        assert result.batch_id == "batch_123"

    def test_create_partial_failure_result(self):
        """Test creating a batch result with partial failures."""
        errors = ["Error 1", "Error 2"]

        result = BatchOperationResult(
            success_count=8,
            error_count=2,
            total_count=10,
            errors=errors,
            processing_time=2.1,
            batch_id="batch_456",
        )

        assert result.success_count == 8
        assert result.error_count == 2
        assert result.errors == errors


@pytest.mark.unit
class TestHealthCheckResult:
    """Test cases for HealthCheckResult model."""

    def test_create_healthy_result(self):
        """Test creating a healthy health check result."""
        result = HealthCheckResult(status=ConnectionStatus.HEALTHY, latency=0.05)

        assert result.status == ConnectionStatus.HEALTHY
        assert result.latency == 0.05
        assert result.details == {}
        assert isinstance(result.timestamp, float)
        assert result.error_message is None

    def test_create_unhealthy_result(self):
        """Test creating an unhealthy health check result."""
        details = {"connected": False, "error_code": 500}

        result = HealthCheckResult(
            status=ConnectionStatus.UNHEALTHY,
            latency=10.0,
            details=details,
            error_message="Connection failed",
        )

        assert result.status == ConnectionStatus.UNHEALTHY
        assert result.latency == 10.0
        assert result.details == details
        assert result.error_message == "Connection failed"


@pytest.mark.unit
class TestAbstractVectorStore:
    """Test cases for AbstractVectorStore base class."""

    def test_init_abstract_store(self):
        """Test initialization of abstract vector store."""
        config = {"host": "localhost", "port": 6333}

        # Create a concrete implementation for testing
        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore(config)

        assert store.config == config
        assert isinstance(store.metrics, VectorStoreMetrics)
        assert store._connection_status == ConnectionStatus.UNKNOWN
        assert store._circuit_breaker_failures == 0
        assert store._circuit_breaker_open is False

    def test_get_metrics(self):
        """Test getting metrics from abstract store."""

        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore({})
        metrics = store.get_metrics()

        assert isinstance(metrics, VectorStoreMetrics)
        assert metrics is store.metrics

    def test_get_connection_status(self):
        """Test getting connection status."""

        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore({})
        status = store.get_connection_status()

        assert status == ConnectionStatus.UNKNOWN

    async def test_circuit_breaker_open(self):
        """Test circuit breaker when open."""

        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore({})
        store._circuit_breaker_open = True
        store._circuit_breaker_failures = CIRCUIT_BREAKER_THRESHOLD + 1

        result = await store._handle_circuit_breaker("test_operation")
        assert result is False

    async def test_circuit_breaker_reset(self):
        """Test circuit breaker reset attempt."""

        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore({})
        store._circuit_breaker_open = True
        store._circuit_breaker_failures = 2  # Below threshold

        result = await store._handle_circuit_breaker("test_operation")
        assert result is True
        assert store._circuit_breaker_open is False

    def test_record_operation_success(self):
        """Test recording successful operation."""

        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore({})
        store._circuit_breaker_failures = 3
        store._circuit_breaker_open = True

        store._record_operation_success()

        assert store._circuit_breaker_failures == 2
        assert store._circuit_breaker_open is False

    def test_record_operation_failure(self):
        """Test recording failed operation."""

        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore({})
        initial_failures = store._circuit_breaker_failures

        store._record_operation_failure()

        assert store._circuit_breaker_failures == initial_failures + 1

    def test_record_failure_opens_circuit_breaker(self):
        """Test that enough failures open the circuit breaker."""

        class TestVectorStore(AbstractVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self):
                pass

            async def search(self, parameters):
                pass

            async def insert_documents(self, documents):
                pass

            async def update_document(self, document):
                pass

            async def delete_documents(self, document_ids, collection="default"):
                pass

            async def create_collection(self, collection_name, vector_size=DEFAULT_VECTOR_DIMENSIONS):
                pass

            async def list_collections(self):
                pass

            async def get_collection_info(self, collection_name):
                pass

        store = TestVectorStore({})
        store._circuit_breaker_failures = CIRCUIT_BREAKER_THRESHOLD - 1

        store._record_operation_failure()

        assert store._circuit_breaker_failures == CIRCUIT_BREAKER_THRESHOLD
        assert store._circuit_breaker_open is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestEnhancedMockVectorStore:
    """Test cases for EnhancedMockVectorStore class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        store = EnhancedMockVectorStore({"initialize_sample_data": False})

        assert store.config == {"initialize_sample_data": False}
        assert store._documents == {}
        assert "default" in store._collections
        assert store._connected is False
        assert store._simulate_latency is True
        assert store._error_rate == 0.0

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = {"simulate_latency": False, "error_rate": 0.1, "base_latency": 0.1}

        store = EnhancedMockVectorStore(config)

        assert store._simulate_latency is False
        assert store._error_rate == 0.1
        assert store._base_latency == 0.1

    def test_sample_data_initialization(self):
        """Test that sample data is initialized correctly."""
        store = EnhancedMockVectorStore({})

        # Should have sample documents
        assert len(store._documents) > 0

        # Check for specific sample documents
        sample_ids = [
            "doc_python_caching",
            "doc_async_error_handling",
            "doc_cicd_github_actions",
            "doc_vector_search",
            "doc_api_design",
        ]

        for doc_id in sample_ids:
            assert doc_id in store._documents
            doc = store._documents[doc_id]
            assert isinstance(doc, VectorDocument)
            assert len(doc.embedding) == DEFAULT_VECTOR_DIMENSIONS

    async def test_connect(self):
        """Test connection establishment."""
        store = EnhancedMockVectorStore({})

        assert store._connected is False
        assert store._connection_status == ConnectionStatus.UNKNOWN

        await store.connect()

        assert store._connected is True
        assert store._connection_status == ConnectionStatus.HEALTHY

    async def test_disconnect(self):
        """Test connection closure."""
        store = EnhancedMockVectorStore({})
        await store.connect()

        assert store._connected is True

        await store.disconnect()

        assert store._connected is False
        assert store._connection_status == ConnectionStatus.UNKNOWN

    async def test_health_check_connected(self):
        """Test health check when connected."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        result = await store.health_check()

        assert isinstance(result, HealthCheckResult)
        assert result.status in [ConnectionStatus.HEALTHY, ConnectionStatus.DEGRADED]
        assert result.latency >= 0
        assert result.details["connected"] is True
        assert result.details["documents_count"] == len(store._documents)

    async def test_health_check_not_connected(self):
        """Test health check when not connected."""
        store = EnhancedMockVectorStore({"simulate_latency": False})

        result = await store.health_check()

        assert result.status == ConnectionStatus.UNHEALTHY
        assert result.error_message == "Not connected"
        assert result.details["connected"] is False

    async def test_search_basic(self):
        """Test basic search functionality."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        # Create search parameters
        embeddings = [[0.8, 0.2, 0.9] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3)]
        params = SearchParameters(embeddings=embeddings, limit=5, collection="default")

        results = await store.search(params)

        assert isinstance(results, list)
        assert len(results) <= params.limit

        for result in results:
            assert isinstance(result, SearchResult)
            assert result.score >= 0.0
            assert result.score <= 1.0
            assert result.source == "enhanced_mock_vector_store"

    async def test_search_with_filters(self):
        """Test search with metadata filters."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        embeddings = [[0.5] * DEFAULT_VECTOR_DIMENSIONS]
        filters = {"category": "performance"}

        params = SearchParameters(embeddings=embeddings, filters=filters, collection="default")

        results = await store.search(params)

        # Should find documents matching the filter
        for result in results:
            assert "category" in result.metadata
            assert result.metadata["category"] == "performance"

    async def test_search_with_score_threshold(self):
        """Test search with score threshold."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        embeddings = [[0.1] * DEFAULT_VECTOR_DIMENSIONS]

        params = SearchParameters(embeddings=embeddings, score_threshold=0.8, collection="default")

        results = await store.search(params)

        # All results should meet the threshold
        for result in results:
            assert result.score >= 0.8

    async def test_search_empty_collection(self):
        """Test search in empty collection."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        embeddings = [[0.1] * DEFAULT_VECTOR_DIMENSIONS]

        params = SearchParameters(embeddings=embeddings, collection="nonexistent")

        results = await store.search(params)

        assert results == []

    async def test_search_circuit_breaker_open(self):
        """Test search when circuit breaker is open."""
        store = EnhancedMockVectorStore({"simulate_latency": False, "initialize_sample_data": False})
        await store.connect()

        # Force circuit breaker open with failures above threshold
        store._circuit_breaker_open = True
        store._circuit_breaker_failures = CIRCUIT_BREAKER_THRESHOLD + 1

        # Verify circuit breaker state before search
        assert store._circuit_breaker_open is True
        assert store._circuit_breaker_failures == CIRCUIT_BREAKER_THRESHOLD + 1

        embeddings = [[0.1] * DEFAULT_VECTOR_DIMENSIONS]
        params = SearchParameters(embeddings=embeddings)

        results = await store.search(params)

        # Circuit breaker should prevent search and return empty results
        assert results == []

    async def test_insert_documents_success(self):
        """Test successful document insertion."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        docs = [
            VectorDocument(
                id="test_doc_1",
                content="Test document 1",
                embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS,
                collection="test_collection",
            ),
            VectorDocument(
                id="test_doc_2",
                content="Test document 2",
                embedding=[0.2] * DEFAULT_VECTOR_DIMENSIONS,
                collection="test_collection",
            ),
        ]

        result = await store.insert_documents(docs)

        assert isinstance(result, BatchOperationResult)
        assert result.success_count == 2
        assert result.error_count == 0
        assert result.total_count == 2
        assert len(result.errors) == 0

        # Check documents were inserted
        assert "test_doc_1" in store._documents
        assert "test_doc_2" in store._documents

    async def test_insert_documents_circuit_breaker(self):
        """Test insert when circuit breaker is open."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        # Force circuit breaker open
        store._circuit_breaker_open = True
        store._circuit_breaker_failures = CIRCUIT_BREAKER_THRESHOLD + 1

        docs = [VectorDocument(id="test", content="test", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS)]

        result = await store.insert_documents(docs)

        assert result.success_count == 0
        assert result.error_count == 1
        assert "Circuit breaker open" in result.errors

    async def test_insert_documents_with_errors(self):
        """Test insert with simulated errors."""
        store = EnhancedMockVectorStore({"simulate_latency": False, "error_rate": 1.0})  # Always error
        await store.connect()

        docs = [VectorDocument(id="test", content="test", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS)]

        # Mock the error simulation to always raise
        with patch.object(store, "_maybe_simulate_error", side_effect=RuntimeError("Simulated error")):
            result = await store.insert_documents(docs)

        assert result.success_count == 0
        assert result.error_count == 1
        assert len(result.errors) > 0

    async def test_update_document_success(self):
        """Test successful document update."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        # Insert initial document
        doc = VectorDocument(id="update_test", content="Original content", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS)
        await store.insert_documents([doc])

        # Update document
        updated_doc = VectorDocument(
            id="update_test",
            content="Updated content",
            embedding=[0.2] * DEFAULT_VECTOR_DIMENSIONS,
        )

        result = await store.update_document(updated_doc)

        assert result is True
        assert store._documents["update_test"].content == "Updated content"

    async def test_update_document_not_found(self):
        """Test updating non-existent document."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        doc = VectorDocument(id="nonexistent", content="Content", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS)

        result = await store.update_document(doc)

        assert result is False

    async def test_delete_documents_success(self):
        """Test successful document deletion."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        # Insert test documents
        docs = [
            VectorDocument(id="delete_1", content="Test 1", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS),
            VectorDocument(id="delete_2", content="Test 2", embedding=[0.2] * DEFAULT_VECTOR_DIMENSIONS),
        ]
        await store.insert_documents(docs)

        # Delete documents
        result = await store.delete_documents(["delete_1", "delete_2"])

        assert isinstance(result, BatchOperationResult)
        assert result.success_count == 2
        assert result.error_count == 0

        # Check documents were deleted
        assert "delete_1" not in store._documents
        assert "delete_2" not in store._documents

    async def test_delete_documents_not_found(self):
        """Test deleting non-existent documents."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        result = await store.delete_documents(["nonexistent1", "nonexistent2"])

        assert result.success_count == 0
        assert result.error_count == 2
        assert len(result.errors) == 2

    async def test_create_collection_success(self):
        """Test successful collection creation."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        result = await store.create_collection("new_collection", 512)

        assert result is True
        assert "new_collection" in store._collections
        assert store._collections["new_collection"]["vector_size"] == 512

    async def test_create_collection_already_exists(self):
        """Test creating collection that already exists."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        # Create collection first time
        await store.create_collection("test_collection")

        # Try to create again
        result = await store.create_collection("test_collection")

        assert result is False

    async def test_list_collections(self):
        """Test listing collections."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        # Create additional collections
        await store.create_collection("collection_1")
        await store.create_collection("collection_2")

        collections = await store.list_collections()

        assert isinstance(collections, list)
        assert "default" in collections
        assert "collection_1" in collections
        assert "collection_2" in collections

    async def test_get_collection_info_success(self):
        """Test getting collection information."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        info = await store.get_collection_info("default")

        assert isinstance(info, dict)
        assert "vector_size" in info
        assert "document_count" in info
        assert info["vector_size"] == DEFAULT_VECTOR_DIMENSIONS

    async def test_get_collection_info_not_found(self):
        """Test getting info for non-existent collection."""
        store = EnhancedMockVectorStore({"simulate_latency": False})
        await store.connect()

        with pytest.raises(ValueError, match="Collection not found"):
            await store.get_collection_info("nonexistent")

    def test_matches_filters_exact_match(self):
        """Test filter matching with exact values."""
        store = EnhancedMockVectorStore({})

        doc = VectorDocument(
            id="test",
            content="Test",
            embedding=[0.1],
            metadata={"category": "tech", "priority": "high"},
        )

        # Exact match filters
        filters = {"category": "tech"}
        assert store._matches_filters(doc, filters) is True

        filters = {"category": "business"}
        assert store._matches_filters(doc, filters) is False

    def test_matches_filters_list_match(self):
        """Test filter matching with list values."""
        store = EnhancedMockVectorStore({})

        doc = VectorDocument(id="test", content="Test", embedding=[0.1], metadata={"category": "tech"})

        # List match filters
        filters = {"category": ["tech", "business"]}
        assert store._matches_filters(doc, filters) is True

        filters = {"category": ["business", "marketing"]}
        assert store._matches_filters(doc, filters) is False

    def test_matches_filters_range_match(self):
        """Test filter matching with range values."""
        store = EnhancedMockVectorStore({})

        doc = VectorDocument(id="test", content="Test", embedding=[0.1], metadata={"score": 75})

        # Range filters
        filters = {"score": {"gte": 70}}
        assert store._matches_filters(doc, filters) is True

        filters = {"score": {"lte": 80}}
        assert store._matches_filters(doc, filters) is True

        filters = {"score": {"gte": 80}}
        assert store._matches_filters(doc, filters) is False

    def test_matches_filters_missing_field(self):
        """Test filter matching when field is missing."""
        store = EnhancedMockVectorStore({})

        doc = VectorDocument(id="test", content="Test", embedding=[0.1], metadata={"category": "tech"})

        filters = {"priority": "high"}
        assert store._matches_filters(doc, filters) is False

    async def test_simulate_operation_delay(self):
        """Test operation delay simulation."""
        store = EnhancedMockVectorStore({"simulate_latency": True, "base_latency": 0.01})

        start_time = time.time()
        await store._simulate_operation_delay()
        end_time = time.time()

        assert end_time - start_time >= 0.01

    async def test_simulate_operation_no_delay(self):
        """Test no delay when latency simulation is disabled."""
        store = EnhancedMockVectorStore({"simulate_latency": False})

        start_time = time.time()
        await store._simulate_operation_delay()
        end_time = time.time()

        assert end_time - start_time < 0.01

    async def test_maybe_simulate_error_no_error(self):
        """Test error simulation when probability is 0."""
        store = EnhancedMockVectorStore({"error_rate": 0.0})

        # Should not raise any exception
        await store._maybe_simulate_error("test_operation")

    async def test_maybe_simulate_error_with_error(self):
        """Test error simulation when probability is 1."""
        store = EnhancedMockVectorStore({"error_rate": 1.0})

        # Mock random to always return a value that triggers error
        with (
            patch("src.utils.secure_random.secure_random.random", return_value=0.0),
            pytest.raises(RuntimeError, match="Simulated error in test_operation"),
        ):
            await store._maybe_simulate_error("test_operation", probability=1.0)


@pytest.mark.unit
class TestQdrantVectorStore:
    """Test cases for QdrantVectorStore class."""

    @patch("src.core.vector_store.ApplicationSettings")
    def test_init_default_config(self, mock_settings_class):
        """Test initialization with default configuration."""
        # Make ApplicationSettings raise an exception to trigger fallback values
        mock_settings_class.side_effect = Exception("Settings not available")

        store = QdrantVectorStore({})

        assert store._client is None
        assert store._connection_pool == []
        assert store._host == "192.168.1.16"
        assert store._port == 6333
        assert store._api_key is None
        assert store._timeout == DEFAULT_TIMEOUT

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = {"host": "localhost", "port": 6334, "api_key": "test_key", "timeout": 60.0}

        store = QdrantVectorStore(config)

        assert store._host == "localhost"
        assert store._port == 6334
        assert store._api_key == "test_key"
        assert store._timeout == 60.0

    @pytest.mark.asyncio
    async def test_connect_qdrant_not_available(self):
        """Test connection when Qdrant is not available."""
        store = QdrantVectorStore({})

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", False),
            pytest.raises(RuntimeError, match="Qdrant client not available"),
        ):
            await store.connect()

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to Qdrant."""
        store = QdrantVectorStore({})

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_client),
        ):
            await store.connect()

            assert store._client is mock_client
            assert store._connection_status == ConnectionStatus.HEALTHY
            mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure to Qdrant."""
        store = QdrantVectorStore({})

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", side_effect=Exception("Connection failed")),
            pytest.raises(Exception, match="Connection failed"),
        ):
            await store.connect()

        assert store._connection_status == ConnectionStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection from Qdrant."""
        store = QdrantVectorStore({})
        mock_client = Mock()
        store._client = mock_client
        store._connection_status = ConnectionStatus.HEALTHY

        await store.disconnect()

        mock_client.close.assert_called_once()
        assert store._client is None
        assert store._connection_status == ConnectionStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_health_check_no_client(self):
        """Test health check when no client is connected."""
        store = QdrantVectorStore({})

        result = await store.health_check()

        assert result.status == ConnectionStatus.UNHEALTHY
        assert result.error_message == "Not connected"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        store = QdrantVectorStore({})
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test_collection")]
        mock_client.get_collections.return_value = mock_collections
        store._client = mock_client

        result = await store.health_check()

        assert result.status in [ConnectionStatus.HEALTHY, ConnectionStatus.DEGRADED]
        assert result.latency >= 0
        assert result.details["collections_count"] == 1

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        store = QdrantVectorStore({})
        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Health check failed")
        store._client = mock_client

        result = await store.health_check()

        assert result.status == ConnectionStatus.UNHEALTHY
        assert result.error_message == "Health check failed"

    @pytest.mark.asyncio
    async def test_search_no_client(self):
        """Test search when no client is available."""
        store = QdrantVectorStore({})

        params = SearchParameters(embeddings=[[0.1, 0.2]])
        results = await store.search(params)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_circuit_breaker_open(self):
        """Test search when circuit breaker is open."""
        store = QdrantVectorStore({})
        store._client = Mock()
        store._circuit_breaker_open = True
        store._circuit_breaker_failures = CIRCUIT_BREAKER_THRESHOLD + 1

        params = SearchParameters(embeddings=[[0.1, 0.2]])
        results = await store.search(params)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful Qdrant search."""
        store = QdrantVectorStore({})
        mock_client = AsyncMock()

        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "doc_1"
        mock_hit.score = 0.95
        mock_hit.payload = {"content": "Test content", "metadata": {"category": "test"}}
        mock_hit.vector = [0.1, 0.2, 0.3]

        mock_client.search.return_value = [mock_hit]
        store._client = mock_client

        params = SearchParameters(
            embeddings=[[0.1, 0.2, 0.3]],
            collection="test_collection",
            strategy=SearchStrategy.HYBRID,
        )

        results = await store.search(params)

        assert len(results) == 1
        result = results[0]
        assert result.document_id == "doc_1"
        assert result.score == 0.95
        assert result.content == "Test content"
        assert result.source == "qdrant"
        assert result.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_insert_documents_no_client(self):
        """Test insert when no client is available."""
        store = QdrantVectorStore({})

        docs = [VectorDocument(id="test", content="test", embedding=[0.1])]
        result = await store.insert_documents(docs)

        assert result.success_count == 0
        assert result.error_count == 1
        assert "Client not available or circuit breaker open" in result.errors

    @pytest.mark.asyncio
    async def test_insert_documents_success(self):
        """Test successful document insertion."""
        store = QdrantVectorStore({})
        mock_client = Mock()

        # Mock successful upsert
        mock_result = Mock()
        mock_result.status = "completed"
        mock_client.upsert.return_value = mock_result

        store._client = mock_client

        # Mock collection existence check
        with (
            patch.object(store, "_ensure_collection_exists", return_value=None),
            patch("src.core.vector_store.PointStruct"),
        ):
            docs = [
                VectorDocument(
                    id="test_doc",
                    content="Test content",
                    embedding=[0.1, 0.2],
                    collection="test_collection",
                ),
            ]

            result = await store.insert_documents(docs)

            assert result.success_count == 1
            assert result.error_count == 0
            mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_document_no_client(self):
        """Test update when no client is available."""
        store = QdrantVectorStore({})

        doc = VectorDocument(id="test", content="test", embedding=[0.1])
        result = await store.update_document(doc)

        assert result is False

    @pytest.mark.asyncio
    async def test_update_document_success(self):
        """Test successful document update."""
        store = QdrantVectorStore({})
        mock_client = Mock()

        mock_result = Mock()
        mock_result.status = "completed"
        mock_client.upsert.return_value = mock_result

        store._client = mock_client

        with patch("src.core.vector_store.PointStruct"):
            doc = VectorDocument(id="test", content="test", embedding=[0.1])
            result = await store.update_document(doc)

            assert result is True
            mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_documents_no_client(self):
        """Test delete when no client is available."""
        store = QdrantVectorStore({})

        result = await store.delete_documents(["doc_1", "doc_2"])

        assert result.success_count == 0
        assert result.error_count == 2
        assert "Client not available" in result.errors

    @pytest.mark.asyncio
    async def test_delete_documents_success(self):
        """Test successful document deletion."""
        store = QdrantVectorStore({})
        mock_client = Mock()

        mock_result = Mock()
        mock_result.status = "completed"
        mock_client.delete.return_value = mock_result

        store._client = mock_client

        result = await store.delete_documents(["doc_1", "doc_2"])

        assert result.success_count == 2
        assert result.error_count == 0
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_no_client(self):
        """Test create collection when no client is available."""
        store = QdrantVectorStore({})

        result = await store.create_collection("test_collection")

        assert result is False

    @pytest.mark.asyncio
    async def test_create_collection_success(self):
        """Test successful collection creation."""
        store = QdrantVectorStore({})
        mock_client = Mock()
        store._client = mock_client

        with (
            patch("src.core.vector_store.VectorParams"),
            patch("src.core.vector_store.Distance"),
        ):
            result = await store.create_collection("test_collection", 512)

            assert result is True
            mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_collections_no_client(self):
        """Test list collections when no client is available."""
        store = QdrantVectorStore({})

        result = await store.list_collections()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_collections_success(self):
        """Test successful collection listing."""
        store = QdrantVectorStore({})
        mock_client = Mock()

        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        store._client = mock_client

        result = await store.list_collections()

        assert result == ["test_collection"]

    @pytest.mark.asyncio
    async def test_get_collection_info_no_client(self):
        """Test get collection info when no client is available."""
        store = QdrantVectorStore({})

        with pytest.raises(ValueError, match="Client not connected"):
            await store.get_collection_info("test_collection")

    @pytest.mark.asyncio
    async def test_get_collection_info_success(self):
        """Test successful collection info retrieval."""
        store = QdrantVectorStore({})
        mock_client = Mock()

        mock_info = Mock()
        mock_info.config.params.vectors.size = 384
        mock_info.config.params.vectors.distance.value = "cosine"
        mock_info.points_count = 100
        mock_info.segments_count = 1
        mock_info.status.value = "green"

        mock_client.get_collection.return_value = mock_info
        store._client = mock_client

        result = await store.get_collection_info("test_collection")

        assert result["vector_size"] == 384
        assert result["distance"] == "cosine"
        assert result["points_count"] == 100

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_no_client(self):
        """Test ensure collection exists when no client is available."""
        store = QdrantVectorStore({})

        # Should not raise exception
        await store._ensure_collection_exists("test", 384)

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_create_new(self):
        """Test ensuring collection exists by creating new one."""
        store = QdrantVectorStore({})
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        store._client = mock_client

        with patch.object(store, "create_collection") as mock_create:
            await store._ensure_collection_exists("test_collection", 384)

            mock_create.assert_called_once_with("test_collection", 384)


@pytest.mark.unit
class TestVectorStoreFactory:
    """Test cases for VectorStoreFactory class."""

    def test_create_mock_store(self):
        """Test creating mock vector store."""
        config = {"type": VectorStoreType.MOCK}

        store = VectorStoreFactory.create_vector_store(config)

        assert isinstance(store, EnhancedMockVectorStore)

    def test_create_qdrant_store(self):
        """Test creating Qdrant vector store."""
        config = {"type": VectorStoreType.QDRANT}

        store = VectorStoreFactory.create_vector_store(config)

        assert isinstance(store, QdrantVectorStore)

    def test_create_auto_detect_qdrant(self):
        """Test auto-detection of Qdrant store."""
        config = {"type": VectorStoreType.AUTO, "host": "localhost", "port": 6333}

        with patch("src.core.vector_store.QDRANT_AVAILABLE", True):
            store = VectorStoreFactory.create_vector_store(config)
            assert isinstance(store, QdrantVectorStore)

    def test_create_auto_detect_mock(self):
        """Test auto-detection falling back to mock."""
        config = {"type": VectorStoreType.AUTO}

        with patch("src.core.vector_store.QDRANT_AVAILABLE", False):
            store = VectorStoreFactory.create_vector_store(config)
            assert isinstance(store, EnhancedMockVectorStore)

    def test_create_invalid_type(self):
        """Test creating store with invalid type."""
        config = {"type": "invalid_type"}

        with pytest.raises(ValueError, match="Unknown vector store type"):
            VectorStoreFactory.create_vector_store(config)

    def test_detect_store_type_qdrant_available(self):
        """Test store type detection when Qdrant is available."""
        config = {"host": "localhost", "port": 6333}

        with patch("src.core.vector_store.QDRANT_AVAILABLE", True):
            store_type = VectorStoreFactory._detect_store_type(config)
            assert store_type == VectorStoreType.QDRANT

    def test_detect_store_type_fallback_mock(self):
        """Test store type detection fallback to mock."""
        config = {}

        store_type = VectorStoreFactory._detect_store_type(config)
        assert store_type == VectorStoreType.MOCK


@pytest.mark.unit
@pytest.mark.asyncio
class TestVectorStoreConnection:
    """Test cases for vector_store_connection context manager."""

    async def test_connection_context_manager(self):
        """Test vector store connection context manager."""
        config = {"type": VectorStoreType.MOCK, "simulate_latency": False}

        async with vector_store_connection(config) as store:
            assert isinstance(store, EnhancedMockVectorStore)
            assert store._connected is True

        # Store should be disconnected after context exit
        assert store._connected is False

    async def test_connection_context_manager_exception(self):
        """Test context manager with exception during use."""
        config = {"type": VectorStoreType.MOCK, "simulate_latency": False}

        try:
            async with vector_store_connection(config) as store:
                assert store._connected is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Store should still be disconnected after exception
        assert store._connected is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestMockVectorStore:
    """Test cases for MockVectorStore compatibility wrapper."""

    def test_init_compatibility_wrapper(self):
        """Test initialization of compatibility wrapper."""
        store = MockVectorStore()

        assert isinstance(store, EnhancedMockVectorStore)
        assert store._simulate_latency is True
        assert store._error_rate == 0.0
        assert hasattr(store, "_auto_connect_task")

    async def test_search_legacy_interface(self):
        """Test search with legacy interface (embeddings, limit)."""
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Allow auto-connect to complete

        embeddings = [[0.8, 0.2, 0.9] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3)]
        results = await store.search(embeddings, limit=3)

        assert isinstance(results, list)
        assert len(results) <= 3

        for result in results:
            assert isinstance(result, SearchResult)

    async def test_search_new_interface(self):
        """Test search with new SearchParameters interface."""
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Allow auto-connect to complete

        params = SearchParameters(embeddings=[[0.5] * DEFAULT_VECTOR_DIMENSIONS], limit=5, collection="default")

        results = await store.search(params)

        assert isinstance(results, list)
        assert len(results) <= 5

    async def test_search_embeddings_method(self):
        """Test backward compatible search_embeddings method."""
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Allow auto-connect to complete

        embeddings = [[0.7] * DEFAULT_VECTOR_DIMENSIONS]
        results = await store.search_embeddings(embeddings, limit=2)

        assert isinstance(results, list)
        assert len(results) <= 2

    async def test_add_documents_compatibility(self):
        """Test backward compatible add_documents method."""
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Allow auto-connect to complete

        # Mock HypotheticalDocument objects
        mock_doc1 = Mock()
        mock_doc1.content = "Test document 1"
        mock_doc1.embedding = [0.1] * DEFAULT_VECTOR_DIMENSIONS
        mock_doc1.metadata = {"type": "test"}

        mock_doc2 = Mock()
        mock_doc2.content = "Test document 2"
        mock_doc2.embedding = [0.2] * DEFAULT_VECTOR_DIMENSIONS
        mock_doc2.metadata = {"type": "test2"}

        docs = [mock_doc1, mock_doc2]

        result = await store.add_documents(docs)

        assert isinstance(result, bool)
        assert result is True

    async def test_add_documents_with_error(self):
        """Test add_documents error handling."""
        store = MockVectorStore()
        await asyncio.sleep(0.1)  # Allow auto-connect to complete

        # Mock documents that will cause errors
        with patch.object(store, "insert_documents", side_effect=Exception("Insert failed")):
            result = await store.add_documents([Mock()])

            assert result is False


@pytest.mark.unit
class TestEnums:
    """Test cases for enum classes."""

    def test_vector_store_type_values(self):
        """Test VectorStoreType enum values."""
        assert VectorStoreType.MOCK == "mock"
        assert VectorStoreType.QDRANT == "qdrant"
        assert VectorStoreType.AUTO == "auto"

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


@pytest.mark.unit
class TestConstants:
    """Test cases for module constants."""

    def test_default_constants(self):
        """Test default constant values."""
        assert DEFAULT_VECTOR_DIMENSIONS == 384
        assert DEFAULT_SEARCH_LIMIT == 10
        assert DEFAULT_BATCH_SIZE == 100
        assert MAX_RETRIES == 3
        assert DEFAULT_TIMEOUT == 30.0
        assert CONNECTION_POOL_SIZE == 5
        assert HEALTH_CHECK_INTERVAL == 60.0
        assert CIRCUIT_BREAKER_THRESHOLD == 5
        assert DEGRADED_LATENCY_THRESHOLD == 0.1

    def test_qdrant_availability(self):
        """Test QDRANT_AVAILABLE constant."""
        assert isinstance(QDRANT_AVAILABLE, bool)


@pytest.mark.unit
class TestModuleExports:
    """Test cases for module exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from src.core.vector_store import __all__

        expected_exports = [
            "AbstractVectorStore",
            "BatchOperationResult",
            "ConnectionStatus",
            "DEFAULT_VECTOR_DIMENSIONS",
            "EnhancedMockVectorStore",
            "HealthCheckResult",
            "MockVectorStore",
            "QdrantVectorStore",
            "SearchParameters",
            "SearchResult",
            "SearchStrategy",
            "VectorDocument",
            "VectorStore",
            "VectorStoreFactory",
            "VectorStoreMetrics",
            "VectorStoreType",
            "vector_store_connection",
        ]

        for export in expected_exports:
            assert export in __all__

    def test_vector_store_alias(self):
        """Test that VectorStore is an alias for MockVectorStore."""
        assert VectorStore is MockVectorStore


@pytest.mark.unit
@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration test cases for complete workflows."""

    async def test_complete_mock_workflow(self):
        """Test complete workflow with mock vector store."""
        config = {"type": VectorStoreType.MOCK, "simulate_latency": False}

        async with vector_store_connection(config) as store:
            # Create collection
            collection_created = await store.create_collection("test_workflow", 128)
            assert collection_created is True

            # Insert documents
            docs = [
                VectorDocument(
                    id="workflow_doc_1",
                    content="First test document for workflow",
                    embedding=[0.1] * 128,
                    collection="test_workflow",
                    metadata={"type": "test", "priority": 1},
                ),
                VectorDocument(
                    id="workflow_doc_2",
                    content="Second test document for workflow",
                    embedding=[0.2] * 128,
                    collection="test_workflow",
                    metadata={"type": "test", "priority": 2},
                ),
            ]

            insert_result = await store.insert_documents(docs)
            assert insert_result.success_count == 2

            # Search documents
            search_params = SearchParameters(embeddings=[[0.15] * 128], collection="test_workflow", limit=5)

            search_results = await store.search(search_params)
            assert len(search_results) <= 5

            # Update document
            updated_doc = VectorDocument(
                id="workflow_doc_1",
                content="Updated first document",
                embedding=[0.3] * 128,
                collection="test_workflow",
                metadata={"type": "test", "priority": 1, "updated": True},
            )

            update_success = await store.update_document(updated_doc)
            assert update_success is True

            # Get collection info
            collection_info = await store.get_collection_info("test_workflow")
            assert collection_info["vector_size"] == 128
            assert collection_info["document_count"] == 2

            # Delete documents
            delete_result = await store.delete_documents(
                ["workflow_doc_1", "workflow_doc_2"],
                collection="test_workflow",
            )
            assert delete_result.success_count == 2

            # Health check
            health = await store.health_check()
            assert health.status == ConnectionStatus.HEALTHY

    async def test_error_handling_workflow(self):
        """Test error handling throughout workflow."""
        config = {
            "type": VectorStoreType.MOCK,
            "simulate_latency": False,
            "error_rate": 0.5,  # 50% error rate for testing
        }

        store = VectorStoreFactory.create_vector_store(config)
        await store.connect()

        # Test with circuit breaker
        store._circuit_breaker_failures = CIRCUIT_BREAKER_THRESHOLD
        store._circuit_breaker_open = True

        # Search should return empty results due to circuit breaker
        params = SearchParameters(embeddings=[[0.1] * DEFAULT_VECTOR_DIMENSIONS])
        results = await store.search(params)
        assert results == []

        # Insert should return error result due to circuit breaker
        docs = [VectorDocument(id="test", content="test", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS)]
        insert_result = await store.insert_documents(docs)
        assert insert_result.success_count == 0

        await store.disconnect()

    async def test_performance_metrics_workflow(self):
        """Test performance metrics collection."""
        config = {"type": VectorStoreType.MOCK, "simulate_latency": False}

        async with vector_store_connection(config) as store:
            initial_metrics = store.get_metrics()
            assert initial_metrics.search_count == 0
            assert initial_metrics.insert_count == 0

            # Perform search operation
            params = SearchParameters(embeddings=[[0.1] * DEFAULT_VECTOR_DIMENSIONS])
            await store.search(params)

            # Check metrics updated
            search_metrics = store.get_metrics()
            assert search_metrics.search_count == 1
            assert search_metrics.avg_latency > 0

            # Perform insert operation
            docs = [VectorDocument(id="metrics_test", content="test", embedding=[0.1] * DEFAULT_VECTOR_DIMENSIONS)]
            await store.insert_documents(docs)

            # Check metrics updated
            final_metrics = store.get_metrics()
            assert final_metrics.insert_count == 1
            assert final_metrics.search_count == 1
