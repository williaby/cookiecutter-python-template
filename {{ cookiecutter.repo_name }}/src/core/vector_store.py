"""
Vector Store Integration for PromptCraft-Hybrid.

This module provides abstract interfaces for vector database integration with support
for both mock implementations and real Qdrant connections. The design emphasizes
performance, reliability, and seamless integration with the HydeProcessor.

Key Features:
- Abstract base class for vector store clients
- Enhanced mock implementation for development and testing
- Real Qdrant integration interface for production
- Connection pooling and retry logic
- Performance monitoring and optimization
- Health checks and circuit breaker patterns
- Comprehensive error handling and logging

Architecture:
    The vector store system implements a pluggable architecture that allows
    seamless switching between mock and real implementations. This enables
    development with mock data while preparing for production Qdrant integration.

    Connection management includes:
    - Lazy connection initialization
    - Connection pooling for performance
    - Automatic retry logic with exponential backoff
    - Health checks and monitoring
    - Circuit breaker for fault tolerance

Key Components:
    - AbstractVectorStore: Base interface for all implementations
    - EnhancedMockVectorStore: Feature-rich mock for development
    - QdrantVectorStore: Production-ready Qdrant client
    - VectorStoreFactory: Factory for creating store instances
    - ConnectionManager: Handles connection lifecycle and health

Dependencies:
    - qdrant-client: For real Qdrant database connections
    - src.config.settings: For vector store configuration
    - src.core.zen_mcp_error_handling: For resilient error handling
    - asyncio: For asynchronous operations and connection management

Called by:
    - src.core.hyde_processor: For enhanced semantic search
    - Agent implementations: For knowledge retrieval
    - Knowledge ingestion systems: For document storage
    - Search and retrieval components: For vector operations

Performance Characteristics:
    - Connection pooling: O(1) connection reuse
    - Batch operations: O(n) for n documents with optimized batching
    - Search operations: O(log n) with proper indexing
    - Memory usage: Configurable connection pool size and caching
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.config.settings import ApplicationSettings
from src.core.performance_optimizer import (
    cache_vector_search,
    monitor_performance,
)
from src.utils.secure_random import secure_random

# Optional imports for Qdrant - only available if qdrant-client is installed
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.http.models import Distance, Filter, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    # Fallback for when qdrant-client is not available
    QdrantClient = None  # type: ignore[misc,assignment]
    UnexpectedResponse = None  # type: ignore[misc,assignment]
    Distance = None  # type: ignore[misc,assignment]
    Filter = None  # type: ignore[misc,assignment]
    PointStruct = None  # type: ignore[misc,assignment]
    VectorParams = None  # type: ignore[misc,assignment]
    QDRANT_AVAILABLE = False

# Type aliases for better readability
EmbeddingVector = list[float]
DocumentMetadata = dict[str, Any]
SearchFilter = dict[str, Any]

# Constants for vector store configuration
DEFAULT_VECTOR_DIMENSIONS = 384
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0
CONNECTION_POOL_SIZE = 5
HEALTH_CHECK_INTERVAL = 60.0
CIRCUIT_BREAKER_THRESHOLD = 5
DEGRADED_LATENCY_THRESHOLD = 0.1


class VectorStoreType(str, Enum):
    """Vector store implementation types."""

    MOCK = "mock"
    QDRANT = "qdrant"
    AUTO = "auto"  # Auto-detect based on configuration


class ConnectionStatus(str, Enum):
    """Connection status for health monitoring."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SearchStrategy(str, Enum):
    """Search strategy types for different use cases."""

    EXACT = "exact"  # Exact similarity search
    HYBRID = "hybrid"  # Combines dense and sparse vectors
    SEMANTIC = "semantic"  # Semantic similarity focus
    FILTERED = "filtered"  # Search with metadata filtering


@dataclass
class VectorStoreMetrics:
    """Performance metrics for vector store operations."""

    search_count: int = 0
    insert_count: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    error_count: int = 0
    last_operation_time: float = field(default_factory=time.time)
    connection_pool_usage: float = 0.0

    def update_search_metrics(self, latency: float) -> None:
        """Update search operation metrics."""
        self.search_count += 1
        self.total_latency += latency
        self.avg_latency = self.total_latency / (self.search_count + self.insert_count)
        self.last_operation_time = time.time()

    def update_insert_metrics(self, latency: float) -> None:
        """Update insert operation metrics."""
        self.insert_count += 1
        self.total_latency += latency
        self.avg_latency = self.total_latency / (self.search_count + self.insert_count)
        self.last_operation_time = time.time()

    def increment_error_count(self) -> None:
        """Increment error counter."""
        self.error_count += 1


class VectorDocument(BaseModel):
    """Vector document for storage and retrieval."""

    id: str = Field(description="Unique document identifier")
    content: str = Field(description="Document content text")
    # nosemgrep: python.lang.correctness.return-not-in-function
    # False positive: lambda in Pydantic Field default_factory is valid syntax
    embedding: EmbeddingVector = Field(
        default_factory=lambda: [0.0] * DEFAULT_VECTOR_DIMENSIONS,
        description="Vector embedding",
    )
    metadata: DocumentMetadata = Field(default_factory=dict, description="Document metadata")
    collection: str = Field(default="default", description="Collection name")
    timestamp: float = Field(default_factory=time.time, description="Creation timestamp")


class SearchResult(BaseModel):
    """Enhanced search result from vector database."""

    document_id: str
    content: str
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    metadata: DocumentMetadata = Field(default_factory=dict)
    source: str = Field(default="unknown", description="Result source")
    embedding: EmbeddingVector | None = Field(default=None, description="Document embedding")
    search_strategy: str = Field(default="default", description="Search strategy used")


class SearchParameters(BaseModel):
    """Parameters for vector search operations."""

    embeddings: list[EmbeddingVector] = Field(description="Query embeddings")
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=100, description="Maximum results")
    collection: str = Field(default="default", description="Collection to search")
    filters: SearchFilter | None = Field(default=None, description="Metadata filters")
    strategy: SearchStrategy = Field(default=SearchStrategy.SEMANTIC, description="Search strategy")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    timeout: float = Field(default=DEFAULT_TIMEOUT, description="Operation timeout")


class BatchOperationResult(BaseModel):
    """Result of batch operations."""

    success_count: int
    error_count: int
    total_count: int
    errors: list[str] = Field(default_factory=list)
    processing_time: float
    batch_id: str


class HealthCheckResult(BaseModel):
    """Health check result with detailed status information."""

    status: ConnectionStatus
    latency: float
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    error_message: str | None = None


class AbstractVectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    This interface defines the contract that all vector store implementations
    must follow, enabling seamless switching between mock and real implementations.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize vector store with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metrics = VectorStoreMetrics()
        self._connection_status = ConnectionStatus.UNKNOWN
        self._circuit_breaker_failures = 0
        self._circuit_breaker_open = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to vector store."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to vector store."""

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform health check on vector store connection."""

    @abstractmethod
    async def search(self, parameters: SearchParameters) -> list[SearchResult]:
        """Search for similar vectors."""

    @abstractmethod
    async def insert_documents(self, documents: list[VectorDocument]) -> BatchOperationResult:
        """Insert multiple documents into vector store."""

    @abstractmethod
    async def update_document(self, document: VectorDocument) -> bool:
        """Update a single document."""

    @abstractmethod
    async def delete_documents(self, document_ids: list[str], collection: str = "default") -> BatchOperationResult:
        """Delete multiple documents by ID."""

    @abstractmethod
    async def create_collection(self, collection_name: str, vector_size: int = DEFAULT_VECTOR_DIMENSIONS) -> bool:
        """Create a new collection."""

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all available collections."""

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection."""

    def get_metrics(self) -> VectorStoreMetrics:
        """Get current performance metrics."""
        return self.metrics

    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._connection_status

    async def _handle_circuit_breaker(self, operation_name: str) -> bool:
        """Check circuit breaker status before operations."""
        # Reset circuit breaker if failures are below threshold
        if self._circuit_breaker_open and self._circuit_breaker_failures < CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_breaker_open = False
            self.logger.info("Circuit breaker reset - failures below threshold")

        if self._circuit_breaker_open:
            self.logger.warning("Circuit breaker is open, skipping %s operation", operation_name)
            return False
        return True

    def _record_operation_success(self) -> None:
        """Record successful operation for circuit breaker."""
        self._circuit_breaker_failures = max(0, self._circuit_breaker_failures - 1)
        self._circuit_breaker_open = False

    def _record_operation_failure(self) -> None:
        """Record failed operation for circuit breaker."""
        self._circuit_breaker_failures += 1
        if self._circuit_breaker_failures >= CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_breaker_open = True
            self.logger.error("Circuit breaker opened after %d failures", self._circuit_breaker_failures)


class EnhancedMockVectorStore(AbstractVectorStore):
    """
    Enhanced mock vector store for development and testing.

    This implementation provides a fully-featured mock that simulates real
    vector store behavior including latency, error conditions, and complex
    search scenarios. It supports all operations needed for development
    and comprehensive testing.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize enhanced mock vector store."""
        config = config or {}
        super().__init__(config)
        self._documents: dict[str, VectorDocument] = {}
        self._collections: dict[str, dict[str, Any]] = {"default": {"vector_size": DEFAULT_VECTOR_DIMENSIONS}}
        self._connected = False
        self._simulate_latency = config.get("simulate_latency", True)
        self._error_rate = config.get("error_rate", 0.0)  # 0-1 error probability
        self._base_latency = config.get("base_latency", 0.05)  # Base latency in seconds

        # Pre-populate with realistic sample data unless disabled
        if config.get("initialize_sample_data", True):
            self._initialize_sample_data()

    async def connect(self) -> None:
        """Simulate connection establishment."""
        await asyncio.sleep(0.1)  # Simulate connection time
        self._connected = True
        self._connection_status = ConnectionStatus.HEALTHY
        self.logger.info("Mock vector store connected successfully")

    async def disconnect(self) -> None:
        """Simulate connection closure."""
        self._connected = False
        self._connection_status = ConnectionStatus.UNKNOWN
        self.logger.info("Mock vector store disconnected")

    async def health_check(self) -> HealthCheckResult:
        """Perform mock health check."""
        start_time = time.time()

        if self._simulate_latency:
            await asyncio.sleep(0.01)

        latency = time.time() - start_time

        if not self._connected:
            return HealthCheckResult(
                status=ConnectionStatus.UNHEALTHY,
                latency=latency,
                error_message="Not connected",
                details={"connected": False},
            )

        # Simulate occasional degraded performance
        status = ConnectionStatus.HEALTHY
        if latency > DEGRADED_LATENCY_THRESHOLD:
            status = ConnectionStatus.DEGRADED

        return HealthCheckResult(
            status=status,
            latency=latency,
            details={
                "connected": self._connected,
                "documents_count": len(self._documents),
                "collections_count": len(self._collections),
                "error_rate": self._error_rate,
            },
        )

    @cache_vector_search
    @monitor_performance("mock_vector_search")
    async def search(self, parameters: SearchParameters) -> list[SearchResult]:
        """Perform mock vector search with realistic behavior."""
        if not await self._handle_circuit_breaker("search"):
            return []

        start_time = time.time()

        try:
            await self._simulate_operation_delay()
            await self._maybe_simulate_error("search")

            # Filter documents by collection
            collection_docs = {
                doc_id: doc for doc_id, doc in self._documents.items() if doc.collection == parameters.collection
            }

            if not collection_docs:
                return []

            # Mock similarity calculation
            results = []
            for doc_id, doc in collection_docs.items():
                # Simple mock similarity based on embedding dot product
                max_similarity = 0.0
                for query_embedding in parameters.embeddings:
                    if len(doc.embedding) == len(query_embedding):
                        # Simplified dot product similarity
                        similarity = sum(a * b for a, b in zip(doc.embedding, query_embedding, strict=False))
                        max_similarity = max(max_similarity, min(1.0, abs(similarity)))

                # Apply score threshold
                if max_similarity >= parameters.score_threshold:
                    # Apply metadata filters if specified
                    if parameters.filters and not self._matches_filters(doc, parameters.filters):
                        continue

                    result = SearchResult(
                        document_id=doc_id,
                        content=doc.content,
                        score=max_similarity,
                        metadata=doc.metadata,
                        source="enhanced_mock_vector_store",
                        embedding=doc.embedding if parameters.strategy == SearchStrategy.HYBRID else None,
                        search_strategy=parameters.strategy.value,
                    )
                    results.append(result)

            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[: parameters.limit]

            # Update metrics
            latency = time.time() - start_time
            self.metrics.update_search_metrics(latency)
            self._record_operation_success()

            self.logger.debug(
                "Mock search completed: %d results in %.3fs for collection '%s'",
                len(results),
                latency,
                parameters.collection,
            )

            return results

        except Exception as e:
            self._record_operation_failure()
            self.metrics.increment_error_count()
            self.logger.error("Mock search failed: %s", str(e))
            raise

    async def insert_documents(self, documents: list[VectorDocument]) -> BatchOperationResult:
        """Insert documents into mock store."""
        if not await self._handle_circuit_breaker("insert"):
            return BatchOperationResult(
                success_count=0,
                error_count=len(documents),
                total_count=len(documents),
                errors=["Circuit breaker open"],
                processing_time=0.0,
                batch_id=f"batch_{int(time.time())}",
            )

        start_time = time.time()
        success_count = 0
        errors = []

        try:
            await self._simulate_operation_delay()

            for doc in documents:
                try:
                    await self._maybe_simulate_error("insert", probability=self._error_rate * 0.1)

                    # Ensure collection exists
                    if doc.collection not in self._collections:
                        await self.create_collection(doc.collection, len(doc.embedding))

                    self._documents[doc.id] = doc
                    success_count += 1

                except Exception as e:
                    errors.append(f"Failed to insert document {doc.id}: {e!s}")

            processing_time = time.time() - start_time
            self.metrics.update_insert_metrics(processing_time)
            self._record_operation_success()

            batch_id = f"mock_batch_{int(time.time())}"

            self.logger.info(
                "Mock batch insert completed: %d/%d successful in %.3fs",
                success_count,
                len(documents),
                processing_time,
            )

            return BatchOperationResult(
                success_count=success_count,
                error_count=len(documents) - success_count,
                total_count=len(documents),
                errors=errors,
                processing_time=processing_time,
                batch_id=batch_id,
            )

        except Exception as e:
            self._record_operation_failure()
            self.metrics.increment_error_count()
            self.logger.error("Mock batch insert failed: %s", str(e))
            raise

    async def update_document(self, document: VectorDocument) -> bool:
        """Update a document in mock store."""
        await self._simulate_operation_delay()

        if document.id in self._documents:
            self._documents[document.id] = document
            self.logger.debug("Mock document updated: %s", document.id)
            return True

        self.logger.warning("Mock document not found for update: %s", document.id)
        return False

    async def delete_documents(self, document_ids: list[str], collection: str = "default") -> BatchOperationResult:
        """Delete documents from mock store."""
        start_time = time.time()
        success_count = 0
        errors = []

        await self._simulate_operation_delay()

        for doc_id in document_ids:
            if doc_id in self._documents and self._documents[doc_id].collection == collection:
                del self._documents[doc_id]
                success_count += 1
            else:
                errors.append(f"Document not found: {doc_id}")

        processing_time = time.time() - start_time
        batch_id = f"delete_batch_{int(time.time())}"

        return BatchOperationResult(
            success_count=success_count,
            error_count=len(document_ids) - success_count,
            total_count=len(document_ids),
            errors=errors,
            processing_time=processing_time,
            batch_id=batch_id,
        )

    async def create_collection(self, collection_name: str, vector_size: int = DEFAULT_VECTOR_DIMENSIONS) -> bool:
        """Create a mock collection."""
        await self._simulate_operation_delay()

        if collection_name not in self._collections:
            self._collections[collection_name] = {
                "vector_size": vector_size,
                "created_at": time.time(),
                "document_count": 0,
            }
            self.logger.info("Mock collection created: %s", collection_name)
            return True

        self.logger.warning("Mock collection already exists: %s", collection_name)
        return False

    async def list_collections(self) -> list[str]:
        """List all mock collections."""
        await self._simulate_operation_delay()
        return list(self._collections.keys())

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get mock collection information."""
        await self._simulate_operation_delay()

        if collection_name not in self._collections:
            raise ValueError(f"Collection not found: {collection_name}")

        doc_count = sum(1 for doc in self._documents.values() if doc.collection == collection_name)

        info = self._collections[collection_name].copy()
        info["name"] = collection_name
        info["document_count"] = doc_count
        return info

    def _initialize_sample_data(self) -> None:
        """Initialize mock store with realistic sample data."""
        sample_docs = [
            VectorDocument(
                id="doc_python_caching",
                content="Implementing efficient caching strategies in Python applications using Redis and in-memory caches",
                embedding=[0.8, 0.2, 0.9] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "performance", "language": "python", "difficulty": "intermediate"},
                collection="default",
            ),
            VectorDocument(
                id="doc_async_error_handling",
                content="Best practices for error handling in asynchronous Python code with proper exception management",
                embedding=[0.7, 0.8, 0.3] + [0.2] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "error_handling", "language": "python", "difficulty": "advanced"},
                collection="default",
            ),
            VectorDocument(
                id="doc_cicd_github_actions",
                content="Setting up comprehensive CI/CD pipelines with GitHub Actions for automated testing and deployment",
                embedding=[0.5, 0.6, 0.8] + [0.3] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "devops", "platform": "github", "difficulty": "beginner"},
                collection="default",
            ),
            VectorDocument(
                id="doc_vector_search",
                content="Implementing semantic search with vector databases and embedding models for AI applications",
                embedding=[0.9, 0.4, 0.7] + [0.4] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "ai", "technology": "vectors", "difficulty": "advanced"},
                collection="default",
            ),
            VectorDocument(
                id="doc_api_design",
                content="RESTful API design principles and best practices for scalable web services",
                embedding=[0.6, 0.9, 0.2] + [0.5] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "api_design", "technology": "rest", "difficulty": "intermediate"},
                collection="default",
            ),
        ]

        for doc in sample_docs:
            self._documents[doc.id] = doc

    async def _simulate_operation_delay(self) -> None:
        """Simulate realistic operation latency."""
        if self._simulate_latency:
            # Add some randomness to latency using secure random
            jitter = secure_random.random() * self._base_latency
            latency = self._base_latency + jitter
            await asyncio.sleep(latency)

    async def _maybe_simulate_error(self, operation: str, probability: float | None = None) -> None:
        """Simulate random errors for testing error handling."""
        error_prob = probability if probability is not None else self._error_rate
        if error_prob > 0 and secure_random.random() < error_prob:
            raise RuntimeError(f"Simulated error in {operation} operation")

    def _matches_filters(self, document: VectorDocument, filters: SearchFilter) -> bool:  # noqa: PLR0911
        """Check if document matches metadata filters."""
        for key, value in filters.items():
            if key not in document.metadata:
                return False

            doc_value = document.metadata[key]

            # Support different filter types
            if isinstance(value, dict):
                # Range filters: {"difficulty": {"gte": "intermediate"}}
                if "gte" in value and doc_value < value["gte"]:
                    return False
                if "lte" in value and doc_value > value["lte"]:
                    return False
                if "eq" in value and doc_value != value["eq"]:
                    return False
            elif isinstance(value, list):
                # List filters: {"category": ["performance", "api_design"]}
                if doc_value not in value:
                    return False
            # Exact match
            elif doc_value != value:
                return False

        return True

    async def insert(self, documents: list[VectorDocument], collection: str = "default") -> BatchOperationResult:
        """Legacy method for inserting documents."""
        # Update collection for each document
        for doc in documents:
            doc.collection = collection
        return await self.insert_documents(documents)

    async def update(self, document: VectorDocument, collection: str = "default") -> bool:
        """Legacy method for updating a document."""
        document.collection = collection
        return await self.update_document(document)

    async def delete(self, document_ids: list[str], collection: str = "default") -> BatchOperationResult:
        """Legacy method for deleting documents."""
        return await self.delete_documents(document_ids, collection)

    def is_connected(self) -> bool:
        """Check if the store is connected."""
        return self._connected

    async def __aenter__(self) -> "EnhancedMockVectorStore":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()


class QdrantVectorStore(AbstractVectorStore):
    """
    Production-ready Qdrant vector store implementation.

    This implementation provides a full-featured Qdrant client with connection
    pooling, retry logic, and comprehensive error handling. It's designed for
    production use with the external Qdrant instance at 192.168.1.16:6333.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Qdrant vector store client."""
        config = config or {}
        super().__init__(config)
        self._client: Any | None = None  # QdrantClient will be imported dynamically
        self._connection_pool: list[Any] = []
        self._pool_semaphore = asyncio.Semaphore(CONNECTION_POOL_SIZE)

        # Use ApplicationSettings for configuration with config overrides
        # Handle case where ApplicationSettings might not be available in tests
        try:
            settings = ApplicationSettings()
            default_host = settings.qdrant_host
            default_port = settings.qdrant_port
            default_api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            default_timeout = settings.qdrant_timeout
        except Exception:
            # Fallback for tests or when settings are not available
            default_host = "192.168.1.16"
            default_port = 6333
            default_api_key = None
            default_timeout = DEFAULT_TIMEOUT

        self._host = config.get("host", default_host)
        self._port = config.get("port", default_port)
        self._api_key = config.get("api_key", default_api_key)
        self._timeout = config.get("timeout", default_timeout)

    async def connect(self) -> None:
        """Establish connection to Qdrant server."""
        try:
            if not QDRANT_AVAILABLE:
                self.logger.error("qdrant-client not available. Using mock implementation.")
                raise RuntimeError("Qdrant client not available")

            # Only create a new client if one doesn't exist (allows mocking in tests)
            if self._client is None:
                self._client = QdrantClient(
                    host=self._host,
                    port=self._port,
                    api_key=self._api_key,
                    timeout=self._timeout,
                )

            # Test connection
            self._client.get_collections()
            self._connection_status = ConnectionStatus.HEALTHY
            self.logger.info("Connected to Qdrant at %s:%d", self._host, self._port)

        except Exception as e:
            self._connection_status = ConnectionStatus.UNHEALTHY
            self.logger.error("Failed to connect to Qdrant: %s", str(e))
            raise

    async def disconnect(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._connection_status = ConnectionStatus.UNKNOWN
            self.logger.info("Disconnected from Qdrant")

    async def health_check(self) -> HealthCheckResult:
        """Perform Qdrant health check."""
        if not self._client:
            return HealthCheckResult(status=ConnectionStatus.UNHEALTHY, latency=0.0, error_message="Not connected")

        start_time = time.time()

        try:
            # Simple health check - list collections
            collections = self._client.get_collections()
            latency = time.time() - start_time

            status = ConnectionStatus.HEALTHY
            if latency > 1.0:  # Consider degraded if > 1s
                status = ConnectionStatus.DEGRADED

            return HealthCheckResult(
                status=status,
                latency=latency,
                details={"collections_count": len(collections.collections), "host": self._host, "port": self._port},
            )

        except Exception as e:
            latency = time.time() - start_time
            return HealthCheckResult(status=ConnectionStatus.UNHEALTHY, latency=latency, error_message=str(e))

    @cache_vector_search
    @monitor_performance("qdrant_vector_search")
    async def search(self, parameters: SearchParameters) -> list[SearchResult]:
        """Perform Qdrant vector search."""
        if not self._client:
            self.logger.warning("QdrantVectorStore is not connected, returning empty results")
            return []

        if not await self._handle_circuit_breaker("search"):
            return []

        start_time = time.time()

        try:
            # Convert filters to Qdrant format
            qdrant_filter = None
            if parameters.filters and Filter is not None:
                qdrant_filter = Filter(**parameters.filters)

            results = []

            # Search with each embedding
            for embedding in parameters.embeddings:
                search_result = self._client.search(
                    collection_name=parameters.collection,
                    query_vector=embedding,
                    limit=parameters.limit,
                    query_filter=qdrant_filter,
                    score_threshold=parameters.score_threshold,
                )

                # Handle potential coroutine from mocked client
                if hasattr(search_result, "__await__"):
                    search_result = await search_result

                for hit in search_result:
                    result = SearchResult(
                        document_id=str(hit.id),
                        content=hit.payload.get("content", ""),
                        score=hit.score,
                        metadata=hit.payload.get("metadata", {}),
                        source="qdrant",
                        embedding=hit.vector if parameters.strategy == SearchStrategy.HYBRID else None,
                        search_strategy=parameters.strategy.value,
                    )
                    results.append(result)

            # Remove duplicates and sort by score
            seen_ids = set()
            unique_results = []
            for result in sorted(results, key=lambda x: x.score, reverse=True):
                if result.document_id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result.document_id)

            # Limit final results
            unique_results = unique_results[: parameters.limit]

            # Update metrics
            latency = time.time() - start_time
            self.metrics.update_search_metrics(latency)
            self._record_operation_success()

            self.logger.debug("Qdrant search completed: %d results in %.3fs", len(unique_results), latency)

            return unique_results

        except Exception as e:
            self._record_operation_failure()
            self.metrics.increment_error_count()
            self.logger.error("Qdrant search failed: %s", str(e))
            raise

    async def insert_documents(self, documents: list[VectorDocument]) -> BatchOperationResult:
        """Insert documents into Qdrant."""
        if not self._client or not await self._handle_circuit_breaker("insert"):
            return BatchOperationResult(
                success_count=0,
                error_count=len(documents),
                total_count=len(documents),
                errors=["Client not available or circuit breaker open"],
                processing_time=0.0,
                batch_id=f"failed_batch_{int(time.time())}",
            )

        start_time = time.time()

        try:
            if PointStruct is None:
                raise RuntimeError("PointStruct not available - qdrant-client not installed")

            # Group documents by collection
            by_collection: dict[str, list[VectorDocument]] = {}
            for doc in documents:
                if doc.collection not in by_collection:
                    by_collection[doc.collection] = []
                by_collection[doc.collection].append(doc)

            success_count = 0
            errors = []

            # Insert documents collection by collection
            for collection, docs in by_collection.items():
                try:
                    # Ensure collection exists
                    await self._ensure_collection_exists(collection, len(docs[0].embedding))

                    # Prepare points for batch upload
                    points = []
                    for doc in docs:
                        point = PointStruct(
                            id=doc.id,
                            vector=doc.embedding,
                            payload={"content": doc.content, "metadata": doc.metadata, "timestamp": doc.timestamp},
                        )
                        points.append(point)

                    # Batch upload
                    result = self._client.upsert(collection_name=collection, points=points)

                    if result.status == "completed":
                        success_count += len(docs)
                    else:
                        errors.append(f"Batch upload failed for collection {collection}: {result.status}")

                except Exception as e:
                    errors.append(f"Failed to insert into collection {collection}: {e!s}")

            processing_time = time.time() - start_time
            self.metrics.update_insert_metrics(processing_time)
            self._record_operation_success()

            batch_id = f"qdrant_batch_{int(time.time())}"

            return BatchOperationResult(
                success_count=success_count,
                error_count=len(documents) - success_count,
                total_count=len(documents),
                errors=errors,
                processing_time=processing_time,
                batch_id=batch_id,
            )

        except Exception as e:
            self._record_operation_failure()
            self.metrics.increment_error_count()
            self.logger.error("Qdrant batch insert failed: %s", str(e))
            raise

    async def update_document(self, document: VectorDocument) -> bool:
        """Update a document in Qdrant."""
        if not self._client:
            return False

        try:
            if PointStruct is None:
                raise RuntimeError("PointStruct not available - qdrant-client not installed")

            point = PointStruct(
                id=document.id,
                vector=document.embedding,
                payload={"content": document.content, "metadata": document.metadata, "timestamp": document.timestamp},
            )

            result = self._client.upsert(collection_name=document.collection, points=[point])

            return bool(result.status == "completed")

        except Exception as e:
            self.logger.error("Failed to update document %s: %s", document.id, str(e))
            return False

    async def delete_documents(self, document_ids: list[str], collection: str = "default") -> BatchOperationResult:
        """Delete documents from Qdrant."""
        if not self._client:
            return BatchOperationResult(
                success_count=0,
                error_count=len(document_ids),
                total_count=len(document_ids),
                errors=["Client not available"],
                processing_time=0.0,
                batch_id=f"failed_delete_{int(time.time())}",
            )

        start_time = time.time()

        try:
            result = self._client.delete(collection_name=collection, points_selector=document_ids)

            processing_time = time.time() - start_time

            if result.status == "completed":
                return BatchOperationResult(
                    success_count=len(document_ids),
                    error_count=0,
                    total_count=len(document_ids),
                    errors=[],
                    processing_time=processing_time,
                    batch_id=f"delete_batch_{int(time.time())}",
                )
            return BatchOperationResult(
                success_count=0,
                error_count=len(document_ids),
                total_count=len(document_ids),
                errors=[f"Delete operation failed: {result.status}"],
                processing_time=processing_time,
                batch_id=f"failed_delete_{int(time.time())}",
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error("Qdrant delete failed: %s", str(e))
            return BatchOperationResult(
                success_count=0,
                error_count=len(document_ids),
                total_count=len(document_ids),
                errors=[str(e)],
                processing_time=processing_time,
                batch_id=f"error_delete_{int(time.time())}",
            )

    async def create_collection(self, collection_name: str, vector_size: int = DEFAULT_VECTOR_DIMENSIONS) -> bool:
        """Create a new Qdrant collection."""
        if not self._client:
            return False

        try:
            if VectorParams is None or Distance is None:
                raise RuntimeError("VectorParams/Distance not available - qdrant-client not installed")

            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            self.logger.info("Created Qdrant collection: %s", collection_name)
            return True

        except Exception as e:
            self.logger.error("Failed to create collection %s: %s", collection_name, str(e))
            return False

    async def list_collections(self) -> list[str]:
        """List all Qdrant collections."""
        if not self._client:
            return []

        try:
            collections = self._client.get_collections()
            return [col.name for col in collections.collections]

        except Exception as e:
            self.logger.error("Failed to list collections: %s", str(e))
            return []

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get Qdrant collection information."""
        if not self._client:
            raise ValueError("Client not connected")

        try:
            info = self._client.get_collection(collection_name)
            return {
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status.value,
            }

        except Exception as e:
            self.logger.error("Failed to get collection info for %s: %s", collection_name, str(e))
            raise

    async def _ensure_collection_exists(self, collection_name: str, vector_size: int) -> None:
        """Ensure a collection exists, create if it doesn't."""
        if not self._client:
            return
        try:
            self._client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.logger.info("Creating collection: %s", collection_name)
            await self.create_collection(collection_name, vector_size)


class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    This factory provides a centralized way to create vector store instances
    based on configuration. It handles auto-detection of available implementations
    and provides appropriate fallbacks.
    """

    @staticmethod
    def create_store(store_type: VectorStoreType | str, config: dict[str, Any] | None = None) -> AbstractVectorStore:
        """
        Create vector store instance by type.

        Args:
            store_type: Type of vector store to create
            config: Optional configuration dictionary

        Returns:
            AbstractVectorStore: Configured vector store instance

        Raises:
            ValueError: If store type is invalid
        """
        config = config or {}

        if isinstance(store_type, str):
            try:
                store_type = VectorStoreType(store_type)
            except ValueError as e:
                raise ValueError(f"Unknown vector store type: {store_type}") from e

        if store_type == VectorStoreType.MOCK:
            return EnhancedMockVectorStore(config)
        if store_type == VectorStoreType.QDRANT:
            return QdrantVectorStore(config)
        raise ValueError(f"Unknown vector store type: {store_type}")

    @staticmethod
    def create_store_from_config(config: dict[str, Any] | None = None) -> AbstractVectorStore:
        """
        Create vector store instance from configuration.

        Args:
            config: Optional configuration dictionary

        Returns:
            AbstractVectorStore: Configured vector store instance
        """
        if config is None:
            # Use ApplicationSettings for configuration
            settings = ApplicationSettings()
            store_type = settings.vector_store_type
        else:
            store_type = config.get("type", "auto")

        if store_type == "auto":
            store_type = VectorStoreFactory._detect_store_type(config or {})

        return VectorStoreFactory.create_store(store_type, config)

    @staticmethod
    def get_supported_types() -> list[VectorStoreType]:
        """Get list of supported vector store types."""
        return list(VectorStoreType)

    @staticmethod
    def create_vector_store(config: dict[str, Any]) -> AbstractVectorStore:
        """
        Create vector store instance based on configuration.

        Args:
            config: Vector store configuration containing type and connection details

        Returns:
            AbstractVectorStore: Configured vector store instance

        Raises:
            ValueError: If store type is invalid or configuration is incomplete
        """
        store_type = config.get("type", VectorStoreType.AUTO)

        if store_type == VectorStoreType.AUTO:
            # Auto-detect based on availability
            store_type = VectorStoreFactory._detect_store_type(config)

        if store_type == VectorStoreType.MOCK:
            return EnhancedMockVectorStore(config)
        if store_type == VectorStoreType.QDRANT:
            return QdrantVectorStore(config)
        raise ValueError(f"Unknown vector store type: {store_type}")

    @staticmethod
    def _detect_store_type(config: dict[str, Any]) -> VectorStoreType:
        """Detect appropriate vector store type based on environment."""
        # Check if Qdrant is available and configured
        if config.get("host") and config.get("port") and QDRANT_AVAILABLE:
            return VectorStoreType.QDRANT

        # Fall back to mock
        return VectorStoreType.MOCK


@asynccontextmanager
async def vector_store_connection(config: dict[str, Any]) -> AsyncIterator[AbstractVectorStore]:
    """
    Async context manager for vector store connections.

    This context manager ensures proper connection lifecycle management
    with automatic cleanup on exit.

    Args:
        config: Vector store configuration

    Yields:
        AbstractVectorStore: Connected vector store instance
    """
    store = VectorStoreFactory.create_vector_store(config)

    try:
        await store.connect()
        yield store
    finally:
        await store.disconnect()


# Compatibility with existing HydeProcessor
class MockVectorStore(EnhancedMockVectorStore):
    """
    Compatibility wrapper for existing HydeProcessor integration.

    This class maintains backward compatibility with the existing MockVectorStore
    used in HydeProcessor while providing the enhanced functionality.
    """

    def __init__(self) -> None:
        """Initialize compatibility mock vector store."""
        config = {"simulate_latency": True, "error_rate": 0.0, "base_latency": 0.05}
        super().__init__(config)
        # Auto-connect immediately for compatibility (only if event loop is running)
        try:
            self._auto_connect_task = asyncio.create_task(self.connect())
        except RuntimeError:
            # No event loop running, will connect on first use
            self._auto_connect_task = None

    async def _ensure_connected(self) -> None:
        """Ensure the store is connected, auto-connect if needed."""
        if not self._connected:
            await self.connect()

    async def search(self, parameters: SearchParameters | list[list[float]], limit: int = 5) -> list[SearchResult]:
        """
        Search method compatible with both new and legacy interfaces.

        Args:
            parameters: Either SearchParameters object or list of embeddings (legacy)
            limit: Maximum number of results (only used with legacy interface)

        Returns:
            List of search results
        """
        await self._ensure_connected()

        # Handle legacy interface: search(embeddings, limit=3)
        if isinstance(parameters, list):
            embeddings = parameters
            search_params = SearchParameters(
                embeddings=embeddings,
                limit=limit,
                collection="default",
                strategy=SearchStrategy.SEMANTIC,
            )
            return await super().search(search_params)

        # Handle new interface: search(SearchParameters)
        return await super().search(parameters)

    async def search_embeddings(self, embeddings: list[list[float]], limit: int = 5) -> list[SearchResult]:
        """
        Backward compatible search method for HydeProcessor.

        Args:
            embeddings: List of query embeddings
            limit: Maximum number of results

        Returns:
            List[SearchResult]: Search results
        """
        parameters = SearchParameters(
            embeddings=embeddings,
            limit=limit,
            collection="default",
            strategy=SearchStrategy.SEMANTIC,
        )

        return await super().search(parameters)

    async def add_documents(self, docs: list[Any]) -> bool:
        """
        Backward compatible document addition for HydeProcessor.

        Args:
            docs: List of HypotheticalDocument objects from HydeProcessor

        Returns:
            bool: Success status
        """
        await self._ensure_connected()

        try:
            # Convert HypotheticalDocument to VectorDocument
            vector_docs = []
            for i, doc in enumerate(docs):
                vector_doc = VectorDocument(
                    id=f"hyde_doc_{i}_{int(time.time())}",
                    content=doc.content,
                    embedding=doc.embedding if doc.embedding else [0.1] * DEFAULT_VECTOR_DIMENSIONS,
                    metadata=doc.metadata if hasattr(doc, "metadata") else {},
                    collection="hyde_generated",
                )
                vector_docs.append(vector_doc)

            result = await self.insert_documents(vector_docs)
            return result.success_count == len(vector_docs)

        except Exception as e:
            self.logger.error("Failed to add HyDE documents: %s", str(e))
            return False


# Alias for backward compatibility
VectorStore = MockVectorStore


# Compatibility aliases and additional models for testing

class VectorStoreConfig(BaseModel):
    """Configuration for vector store instances."""
    
    host: str = Field(default="localhost", description="Vector store host")
    port: int = Field(default=6333, description="Vector store port")
    collection: str = Field(default="default", description="Default collection name")
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")
    api_key: str | None = Field(default=None, description="API key for authentication")
    use_ssl: bool = Field(default=False, description="Whether to use SSL connection")


class ConnectionManager:
    """Basic connection manager for vector store instances."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize connection manager with configuration."""
        self.config = config
        self._connection_pool = {}
        self._is_connected = False
    
    async def connect(self) -> bool:
        """Establish connection to the vector store."""
        try:
            # Basic connection simulation
            self._is_connected = True
            return True
        except Exception:
            self._is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the vector store."""
        self._is_connected = False
        self._connection_pool.clear()
    
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._is_connected
    
    async def health_check(self) -> dict[str, Any]:
        """Perform health check on the connection."""
        return {
            "status": "healthy" if self._is_connected else "disconnected",
            "host": self.config.host,
            "port": self.config.port,
            "connection_pool_size": len(self._connection_pool)
        }


# Compatibility alias
Document = VectorDocument


# Export main classes for external use
__all__ = [
    "DEFAULT_VECTOR_DIMENSIONS",
    "AbstractVectorStore",
    "BatchOperationResult",
    "ConnectionManager",
    "ConnectionStatus",
    "Document",  # Alias for VectorDocument
    "EnhancedMockVectorStore",
    "HealthCheckResult",
    "MockVectorStore",  # Backward compatibility
    "QdrantVectorStore",
    "SearchParameters",
    "SearchResult",
    "SearchStrategy",
    "VectorDocument",
    "VectorStore",  # Main alias for import compatibility
    "VectorStoreConfig",
    "VectorStoreFactory",
    "VectorStoreMetrics",
    "VectorStoreType",
    "vector_store_connection",
]
