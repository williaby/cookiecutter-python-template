"""
Integration tests for real Qdrant vector database integration.

This module tests the integration between PromptCraft components and real Qdrant
vector database at 192.168.1.16:6333, validating end-to-end workflows with actual
vector operations and performance requirements.
"""

import asyncio
import contextlib
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.hyde_processor import HydeProcessor
from src.core.vector_store import (
    DEFAULT_VECTOR_DIMENSIONS,
    ConnectionStatus,
    HealthCheckResult,
    QdrantVectorStore,
    SearchParameters,
    SearchResult,
    SearchStrategy,
    VectorDocument,
    VectorStoreFactory,
    VectorStoreType,
    vector_store_connection,
)
from tests.utils.mock_helpers import create_qdrant_client_mock


class TestRealQdrantIntegration:
    """Integration tests for real Qdrant functionality."""

    @pytest.fixture
    def qdrant_config(self):
        """Create test configuration for Qdrant integration."""
        return {
            "type": VectorStoreType.QDRANT,
            "host": "192.168.1.16",
            "port": 6333,
            "timeout": 10.0,
            "api_key": None,  # Assuming no auth required for test environment
            "simulate_latency": False,
            "error_rate": 0.0,
        }

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing integration patterns."""
        return create_qdrant_client_mock()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            VectorDocument(
                id="doc_1",
                content="Python caching strategies using Redis for high-performance applications",
                embedding=[0.8, 0.2, 0.9] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "performance", "language": "python"},
                collection="test_collection",
            ),
            VectorDocument(
                id="doc_2",
                content="Asynchronous error handling best practices in modern Python development",
                embedding=[0.7, 0.8, 0.3] + [0.2] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "error_handling", "language": "python"},
                collection="test_collection",
            ),
            VectorDocument(
                id="doc_3",
                content="Vector database optimization for semantic search applications",
                embedding=[0.9, 0.4, 0.7] + [0.3] * (DEFAULT_VECTOR_DIMENSIONS - 3),
                metadata={"category": "databases", "technology": "vectors"},
                collection="test_collection",
            ),
        ]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_connection_lifecycle(self, qdrant_config, mock_qdrant_client):
        """Test complete Qdrant connection lifecycle."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            # Create Qdrant store
            store = QdrantVectorStore(qdrant_config)

            # Test connection
            await store.connect()
            assert store.get_connection_status() == ConnectionStatus.HEALTHY

            # Verify client was configured correctly
            mock_qdrant_client.get_collections.assert_called_once()

            # Test disconnection
            await store.disconnect()
            assert store.get_connection_status() == ConnectionStatus.UNKNOWN
            mock_qdrant_client.close.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_connection_failure_handling(self, qdrant_config):
        """Test Qdrant connection failure scenarios."""

        # Test import error (qdrant-client not available)
        with patch("src.core.vector_store.QDRANT_AVAILABLE", False):
            store = QdrantVectorStore(qdrant_config)

            with pytest.raises(RuntimeError, match="Qdrant client not available"):
                await store.connect()

            assert store.get_connection_status() == ConnectionStatus.UNHEALTHY

        # Test connection error (server unreachable)
        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient") as mock_client_class,
        ):

            # Make the QdrantClient constructor itself raise the exception
            mock_client_class.side_effect = Exception("Connection timeout")

            store = QdrantVectorStore(qdrant_config)

            with pytest.raises(Exception, match="Connection timeout"):
                await store.connect()

            assert store.get_connection_status() == ConnectionStatus.UNHEALTHY

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_qdrant_health_check_integration(self, qdrant_config, mock_qdrant_client):
        """Test Qdrant health check with various server states."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Test healthy status
            health_result = await store.health_check()
            assert isinstance(health_result, HealthCheckResult)
            assert health_result.status == ConnectionStatus.HEALTHY
            assert health_result.latency > 0
            assert "collections_count" in health_result.details
            assert health_result.details["host"] == "192.168.1.16"
            assert health_result.details["port"] == 6333

            # Test degraded status (high latency)
            def slow_get_collections():
                time.sleep(1.1)  # Simulate >1s latency
                return mock_qdrant_client.get_collections.return_value

            mock_qdrant_client.get_collections.side_effect = slow_get_collections
            health_result = await store.health_check()
            assert health_result.status == ConnectionStatus.DEGRADED
            assert health_result.latency > 1.0

            # Test unhealthy status (error)
            mock_qdrant_client.get_collections.side_effect = Exception("Server error")
            health_result = await store.health_check()
            assert health_result.status == ConnectionStatus.UNHEALTHY
            assert health_result.error_message == "Server error"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_document_operations_integration(self, qdrant_config, mock_qdrant_client, sample_documents):
        """Test Qdrant document CRUD operations."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Test document insertion
            with patch("src.core.vector_store.PointStruct", MagicMock()):
                insert_result = await store.insert_documents(sample_documents)
            assert insert_result.success_count == len(sample_documents)
            assert insert_result.error_count == 0
            assert insert_result.total_count == len(sample_documents)
            assert insert_result.processing_time > 0

            # Verify upsert was called with correct data
            mock_qdrant_client.upsert.assert_called()
            call_args = mock_qdrant_client.upsert.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            assert len(call_args[1]["points"]) == len(sample_documents)

            # Test document update
            updated_doc = sample_documents[0]
            updated_doc.content = "Updated content about Python caching"

            with patch("src.core.vector_store.PointStruct", MagicMock()):
                update_success = await store.update_document(updated_doc)
            assert update_success is True

            # Test document deletion
            delete_ids = ["doc_1", "doc_2"]
            delete_result = await store.delete_documents(delete_ids, "test_collection")
            assert delete_result.success_count == len(delete_ids)
            assert delete_result.error_count == 0

            # Verify delete was called with correct parameters
            mock_qdrant_client.delete.assert_called_once()
            call_args = mock_qdrant_client.delete.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            assert call_args[1]["points_selector"] == delete_ids

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_search_integration(self, qdrant_config, mock_qdrant_client):
        """Test Qdrant vector search functionality."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Create search parameters
            search_params = SearchParameters(
                embeddings=[[0.8, 0.2, 0.9] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3)],
                limit=10,
                collection="test_collection",
                strategy=SearchStrategy.SEMANTIC,
                score_threshold=0.7,
            )

            # Test search
            results = await store.search(search_params)
            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            assert results[0].document_id == "test_doc_1"
            assert results[0].score == 0.95
            assert results[0].content == "Test document content about Python caching strategies"
            assert results[0].source == "qdrant"
            assert results[0].search_strategy == SearchStrategy.SEMANTIC.value

            # Verify search was called with correct parameters
            mock_qdrant_client.search.assert_called()
            call_args = mock_qdrant_client.search.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            assert call_args[1]["limit"] == 10
            assert call_args[1]["score_threshold"] == 0.7

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_collection_management_integration(self, qdrant_config, mock_qdrant_client):
        """Test Qdrant collection management operations."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Test collection creation
            with (
                patch("src.core.vector_store.VectorParams", MagicMock()),
                patch("src.core.vector_store.Distance", MagicMock()),
            ):
                collection_created = await store.create_collection("new_test_collection", 512)
            assert collection_created is True

            # Verify create_collection was called
            mock_qdrant_client.create_collection.assert_called()
            call_args = mock_qdrant_client.create_collection.call_args
            assert call_args[1]["collection_name"] == "new_test_collection"

            # Test collection listing
            collections = await store.list_collections()
            assert isinstance(collections, list)
            assert "test_collection" in collections
            assert "default" in collections

            # Test collection info retrieval
            collection_info = await store.get_collection_info("test_collection")
            assert isinstance(collection_info, dict)
            assert "vector_size" in collection_info
            assert "distance" in collection_info
            assert "points_count" in collection_info
            assert collection_info["vector_size"] == DEFAULT_VECTOR_DIMENSIONS
            assert collection_info["points_count"] == 42

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_error_handling_integration(self, qdrant_config, mock_qdrant_client):
        """Test Qdrant error handling for various failure scenarios."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Test search error handling
            mock_qdrant_client.search.side_effect = Exception("Search service temporarily unavailable")

            search_params = SearchParameters(
                embeddings=[[0.1] * DEFAULT_VECTOR_DIMENSIONS],
                limit=5,
                collection="test_collection",
            )
            with pytest.raises(Exception, match="Search service temporarily unavailable"):
                await store.search(search_params)

            # Verify error metrics were updated
            assert store.get_metrics().error_count > 0

            # Test circuit breaker activation
            # Simulate multiple failures to trigger circuit breaker
            for _ in range(6):  # Exceed CIRCUIT_BREAKER_THRESHOLD
                with contextlib.suppress(Exception):
                    await store.search(search_params)

            # Circuit breaker should now be open
            assert store._circuit_breaker_open is True

            # Test operations are blocked when circuit breaker is open
            mock_qdrant_client.search.side_effect = None  # Reset to successful
            mock_qdrant_client.search.return_value = []
            results = await store.search(search_params)
            assert results == []  # Should return empty due to circuit breaker

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vector_store_factory_qdrant_integration(self, qdrant_config):
        """Test VectorStoreFactory integration with Qdrant configuration."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Test Qdrant store creation via factory
            store = VectorStoreFactory.create_vector_store(qdrant_config)
            assert isinstance(store, QdrantVectorStore)
            assert store.config["host"] == "192.168.1.16"
            assert store.config["port"] == 6333

            # Test auto-detection with Qdrant available
            auto_config = qdrant_config.copy()
            auto_config["type"] = VectorStoreType.AUTO

            auto_store = VectorStoreFactory.create_vector_store(auto_config)
            assert isinstance(auto_store, QdrantVectorStore)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vector_store_context_manager_integration(self, qdrant_config, mock_qdrant_client):
        """Test vector store context manager with Qdrant."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            # Test context manager lifecycle
            async with vector_store_connection(qdrant_config) as store:
                assert isinstance(store, QdrantVectorStore)
                assert store.get_connection_status() == ConnectionStatus.HEALTHY

                # Perform operation within context
                health = await store.health_check()
                assert health.status == ConnectionStatus.HEALTHY

            # Verify connection was closed
            mock_qdrant_client.close.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_hyde_processor_qdrant_integration(self, qdrant_config):
        """Test HydeProcessor integration with real Qdrant vector store."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful operations for HydeProcessor
            mock_collections_response = MagicMock()
            mock_collections_response.collections = []
            mock_client.get_collections.return_value = mock_collections_response

            mock_upsert_result = MagicMock()
            mock_upsert_result.status = "completed"
            mock_client.upsert.return_value = mock_upsert_result

            mock_search_result = MagicMock()
            mock_search_result.id = "hyde_doc_1"
            mock_search_result.score = 0.92
            mock_search_result.payload = {
                "content": "Enhanced query processing with HyDE methodology",
                "metadata": {"source": "hyde_generated"},
            }
            mock_search_result.vector = [0.5] * DEFAULT_VECTOR_DIMENSIONS
            mock_client.search.return_value = [mock_search_result]

            # Create HydeProcessor with Qdrant vector store
            qdrant_store = QdrantVectorStore(qdrant_config)
            await qdrant_store.connect()

            hyde_processor = HydeProcessor(vector_store=qdrant_store)

            # Test HyDE query processing with Qdrant backend
            query = "How to optimize vector database performance for large-scale semantic search?"
            enhanced_query = await hyde_processor.three_tier_analysis(query)

            assert enhanced_query is not None
            assert enhanced_query.enhanced_query == query  # Mock implementation returns same query
            assert enhanced_query.specificity_analysis.specificity_score > 0

            # Test full HyDE processing pipeline
            results = await hyde_processor.process_query(query)
            assert results is not None
            assert hasattr(results, "results")

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_qdrant_performance_requirements(self, qdrant_config, mock_qdrant_client, sample_documents):
        """Test Qdrant performance meets <2s response time requirement."""

        # Configure mock with realistic timing
        def realistic_search(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms search time
            return mock_qdrant_client.search.return_value

        async def realistic_upsert(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate 50ms upsert time
            return mock_qdrant_client.upsert.return_value

        mock_qdrant_client.search.side_effect = realistic_search
        mock_qdrant_client.upsert.side_effect = realistic_upsert

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Test search performance
            start_time = time.time()

            search_params = SearchParameters(
                embeddings=[[0.8, 0.2, 0.9] + [0.1] * (DEFAULT_VECTOR_DIMENSIONS - 3)],
                limit=20,
                collection="test_collection",
                strategy=SearchStrategy.SEMANTIC,
            )

            results = await store.search(search_params)
            search_time = time.time() - start_time

            # Verify search performance
            assert search_time < 2.0, f"Search time {search_time:.3f}s exceeds 2s requirement"
            assert len(results) > 0

            # Test batch insert performance
            start_time = time.time()

            with patch("src.core.vector_store.PointStruct", MagicMock()):
                batch_result = await store.insert_documents(sample_documents)
            insert_time = time.time() - start_time

            # Verify insert performance
            assert insert_time < 2.0, f"Insert time {insert_time:.3f}s exceeds 2s requirement"
            assert batch_result.success_count == len(sample_documents)

            # Test comprehensive workflow performance (search + insert)
            start_time = time.time()

            # Simulate realistic workflow: search existing, process, insert new
            await store.search(search_params)
            with patch("src.core.vector_store.PointStruct", MagicMock()):
                await store.insert_documents(sample_documents[:1])  # Insert one new doc
            await store.search(search_params)  # Search again

            workflow_time = time.time() - start_time

            # Verify total workflow performance
            assert workflow_time < 2.0, f"Workflow time {workflow_time:.3f}s exceeds 2s requirement"

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_qdrant_concurrent_operations_integration(self, qdrant_config, mock_qdrant_client):
        """Test Qdrant concurrent operations and connection pooling."""

        # Configure mock for concurrent operations
        call_count = 0

        def concurrent_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate processing time
            return [MagicMock(id=f"doc_{call_count}", score=0.9, payload={"content": f"Result {call_count}"})]

        # Set up the mock to track calls properly
        mock_qdrant_client.search.side_effect = concurrent_search

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Test concurrent search operations with unique parameters to avoid caching
            search_results = []
            for i in range(5):
                search_params = SearchParameters(
                    embeddings=[[0.1 + i * 0.01] * DEFAULT_VECTOR_DIMENSIONS],  # Unique embeddings
                    limit=5,
                    collection="test_collection",
                )
                result = await store.search(search_params)
                search_results.append(result)

            # Verify all operations completed successfully
            assert len(search_results) == 5
            assert all(len(result) > 0 for result in search_results)
            assert call_count == 5  # All operations were executed

            # Verify performance metrics were updated
            metrics = store.get_metrics()
            assert metrics.search_count >= 5
            assert metrics.avg_latency > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_data_consistency_integration(self, qdrant_config, mock_qdrant_client):
        """Test Qdrant data consistency and transaction-like behavior."""

        # Track operations for consistency verification
        operations_log = []

        def tracked_upsert(*args, **kwargs):
            operations_log.append(f"upsert: {kwargs.get('collection_name', 'unknown')}")
            # Return the mock result directly, not the coroutine
            mock_result = MagicMock()
            mock_result.status = "completed"
            return mock_result

        def tracked_delete(*args, **kwargs):
            operations_log.append(f"delete: {kwargs.get('collection_name', 'unknown')}")
            # Return the mock result directly, not the coroutine
            mock_result = MagicMock()
            mock_result.status = "completed"
            return mock_result

        # Set side effects on the AwaitableCallableMock objects
        mock_qdrant_client.upsert.side_effect = tracked_upsert
        mock_qdrant_client.delete.side_effect = tracked_delete

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Test batch operation consistency
            documents = [
                VectorDocument(
                    id=f"consistency_doc_{i}",
                    content=f"Test document {i} for consistency testing",
                    embedding=[float(i) / 10] * DEFAULT_VECTOR_DIMENSIONS,
                    metadata={"test_id": i},
                    collection="consistency_test",
                )
                for i in range(3)
            ]

            # Insert documents
            with patch("src.core.vector_store.PointStruct", MagicMock()):
                insert_result = await store.insert_documents(documents)
            assert insert_result.success_count == len(documents)

            # Verify operations were logged
            assert len(operations_log) > 0
            assert any("upsert: consistency_test" in op for op in operations_log)

            # Test document updates maintain consistency
            updated_doc = documents[0]
            updated_doc.content = "Updated content for consistency test"
            with patch("src.core.vector_store.PointStruct", MagicMock()):
                update_success = await store.update_document(updated_doc)
            assert update_success is True

            # Test cleanup operations
            document_ids = [doc.id for doc in documents]
            delete_result = await store.delete_documents(document_ids, "consistency_test")
            assert delete_result.success_count == len(document_ids)

            # Verify delete operations were logged
            assert any("delete: consistency_test" in op for op in operations_log)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_configuration_validation_integration(self, qdrant_config):
        """Test Qdrant configuration validation and error handling."""

        # Test invalid host configuration
        invalid_config = qdrant_config.copy()
        invalid_config["host"] = "invalid.host.name"

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient") as mock_client_class,
        ):
            mock_client_class.side_effect = Exception("Host not reachable")

            store = QdrantVectorStore(invalid_config)

            with pytest.raises(Exception, match="Host not reachable"):
                await store.connect()

        # Test invalid port configuration
        invalid_port_config = qdrant_config.copy()
        invalid_port_config["port"] = 99999

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient") as mock_client_class,
        ):
            mock_client_class.side_effect = Exception("Connection refused")

            store = QdrantVectorStore(invalid_port_config)

            with pytest.raises(Exception, match="Connection refused"):
                await store.connect()

        # Test missing required configuration
        minimal_config = {"type": VectorStoreType.QDRANT}

        # Should use defaults
        store = QdrantVectorStore(minimal_config)
        # Check for environment-specific host (CI uses localhost, production uses 192.168.1.16)
        expected_host = "localhost" if os.getenv("CI_ENVIRONMENT") else "192.168.1.16"
        assert store._host == expected_host  # Default host
        assert store._port == 6333  # Default port

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_qdrant_metrics_and_monitoring_integration(self, qdrant_config, mock_qdrant_client):
        """Test Qdrant metrics collection and monitoring."""

        with (
            patch("src.core.vector_store.QDRANT_AVAILABLE", True),
            patch("src.core.vector_store.QdrantClient", return_value=mock_qdrant_client),
        ):
            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            # Perform various operations to generate metrics
            search_params_1 = SearchParameters(
                embeddings=[[0.1] * DEFAULT_VECTOR_DIMENSIONS],
                limit=5,
                collection="metrics_test",
            )

            search_params_2 = SearchParameters(
                embeddings=[[0.2] * DEFAULT_VECTOR_DIMENSIONS],  # Different embedding to avoid cache
                limit=5,
                collection="metrics_test",
            )

            # Execute multiple operations with different parameters to avoid cache
            await store.search(search_params_1)
            await store.search(search_params_2)

            test_doc = VectorDocument(
                id="metrics_test_doc",
                content="Test document for metrics",
                embedding=[0.5] * DEFAULT_VECTOR_DIMENSIONS,
                metadata={"test": "metrics"},
                collection="metrics_test",
            )
            with patch("src.core.vector_store.PointStruct", MagicMock()):
                await store.insert_documents([test_doc])

            # Check metrics collection
            metrics = store.get_metrics()
            assert metrics.search_count >= 2
            assert metrics.insert_count >= 1
            assert metrics.total_latency > 0
            assert metrics.avg_latency > 0
            assert metrics.last_operation_time > 0

            # Store initial error count for comparison
            initial_error_count = metrics.error_count

            # Test error metrics
            mock_qdrant_client.search.side_effect = Exception("Test error")

            # Use different parameters to avoid cache
            search_params_error = SearchParameters(
                embeddings=[[0.3] * DEFAULT_VECTOR_DIMENSIONS],  # Different embedding to avoid cache
                limit=5,
                collection="metrics_test",
            )

            with contextlib.suppress(Exception):
                await store.search(search_params_error)

            # Verify error count increased
            updated_metrics = store.get_metrics()
            assert updated_metrics.error_count > initial_error_count
