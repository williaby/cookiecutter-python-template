"""
Comprehensive unit tests for Performance Optimizer components.

This module provides comprehensive unit test coverage for the performance
optimization system including caching, monitoring, and optimization strategies.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / ".." / ".." / "src"))

from src.core.performance_optimizer import (
    AsyncBatcher,
    ConnectionPool,
    LRUCache,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceOptimizer,
    _hyde_cache,
    _performance_monitor,
    _query_cache,
    _vector_cache,
    cache_hyde_processing,
    cache_query_analysis,
    cache_vector_search,
    clear_all_caches,
    get_performance_stats,
    monitor_performance,
    warm_up_system,
)


class TestLRUCache:
    """Test suite for LRUCache implementation."""

    @pytest.fixture
    def lru_cache(self):
        """Create LRUCache instance for testing."""
        return LRUCache(max_size=5, ttl_seconds=60)

    def test_lru_cache_initialization(self, lru_cache):
        """Test LRUCache initialization."""
        assert lru_cache.max_size == 5
        assert lru_cache.ttl_seconds == 60
        assert lru_cache.size == 0
        assert lru_cache.hits == 0
        assert lru_cache.misses == 0

    def test_lru_cache_basic_operations(self, lru_cache):
        """Test basic put and get operations."""
        # Test put and get
        lru_cache.put("key1", "value1")
        assert lru_cache.get("key1") == "value1"
        assert lru_cache.size == 1
        assert lru_cache.hits == 1
        assert lru_cache.misses == 0

        # Test cache miss
        assert lru_cache.get("nonexistent") is None
        assert lru_cache.hits == 1
        assert lru_cache.misses == 1

    def test_lru_cache_capacity_limit(self, lru_cache):
        """Test LRU cache capacity limit."""
        # Fill cache to capacity
        for i in range(5):
            lru_cache.put(f"key{i}", f"value{i}")

        assert lru_cache.size == 5

        # Add one more item, should evict oldest
        lru_cache.put("key5", "value5")
        assert lru_cache.size == 5
        assert lru_cache.get("key0") is None  # Should be evicted
        assert lru_cache.get("key5") == "value5"  # Should be present

    def test_lru_cache_access_order(self, lru_cache):
        """Test LRU access order maintenance."""
        # Add items
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        lru_cache.put("key3", "value3")

        # Access key1 to make it most recently used
        lru_cache.get("key1")

        # Fill cache to capacity
        lru_cache.put("key4", "value4")
        lru_cache.put("key5", "value5")

        # Add one more, should evict key2 (least recently used)
        lru_cache.put("key6", "value6")
        assert lru_cache.get("key2") is None  # Should be evicted
        assert lru_cache.get("key1") == "value1"  # Should still be present

    def test_lru_cache_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(0.2)
        assert cache.get("key1") is None
        assert cache.misses == 1

    def test_lru_cache_update_existing_key(self, lru_cache):
        """Test updating existing key."""
        lru_cache.put("key1", "value1")
        assert lru_cache.get("key1") == "value1"

        # Update same key
        lru_cache.put("key1", "new_value1")
        assert lru_cache.get("key1") == "new_value1"
        assert lru_cache.size == 1  # Size should not change

    def test_lru_cache_clear(self, lru_cache):
        """Test cache clearing."""
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        assert lru_cache.size == 2

        lru_cache.clear()
        assert lru_cache.size == 0
        assert lru_cache.get("key1") is None
        assert lru_cache.get("key2") is None

    def test_lru_cache_statistics(self, lru_cache):
        """Test cache statistics."""
        # Generate some hits and misses
        lru_cache.put("key1", "value1")
        lru_cache.get("key1")  # hit
        lru_cache.get("key1")  # hit
        lru_cache.get("nonexistent")  # miss

        stats = lru_cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_lru_cache_contains(self, lru_cache):
        """Test contains functionality."""
        lru_cache.put("key1", "value1")
        assert lru_cache.contains("key1") is True
        assert lru_cache.contains("nonexistent") is False

    def test_lru_cache_with_none_values(self, lru_cache):
        """Test cache with None values."""
        lru_cache.put("key1", None)
        assert lru_cache.get("key1") is None
        assert lru_cache.contains("key1") is True  # Should still be in cache

    def test_lru_cache_edge_cases(self):
        """Test edge cases."""
        # Zero size cache
        cache = LRUCache(max_size=0, ttl_seconds=60)
        cache.put("key1", "value1")
        assert cache.get("key1") is None
        assert cache.size == 0

        # Very small TTL
        cache = LRUCache(max_size=10, ttl_seconds=0.001)
        cache.put("key1", "value1")
        time.sleep(0.002)
        assert cache.get("key1") is None


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor implementation."""

    @pytest.fixture
    def monitor(self):
        """Create PerformanceMonitor instance for testing."""
        return PerformanceMonitor()

    def test_performance_monitor_initialization(self, monitor):
        """Test PerformanceMonitor initialization."""
        assert monitor.operation_count == 0
        assert monitor.total_duration == 0.0
        assert monitor.max_duration == 0.0
        assert monitor.min_duration == float("inf")
        assert monitor.error_count == 0
        assert len(monitor.operations) == 0

    def test_performance_monitor_start_operation(self, monitor):
        """Test starting operation monitoring."""
        metric = monitor.start_operation("test_operation")

        assert isinstance(metric, PerformanceMetric)
        assert metric.operation_name == "test_operation"
        assert metric.start_time > 0
        assert metric.end_time is None
        assert metric.duration is None
        assert metric.success is None

    def test_performance_monitor_complete_operation(self, monitor):
        """Test completing operation monitoring."""
        metric = monitor.start_operation("test_operation")
        time.sleep(0.01)  # Small delay
        monitor.complete_operation(metric)

        assert metric.end_time > metric.start_time
        assert metric.duration > 0
        assert metric.success is True
        assert monitor.operation_count == 1
        assert monitor.total_duration > 0

    def test_performance_monitor_error_operation(self, monitor):
        """Test error operation monitoring."""
        metric = monitor.start_operation("test_operation")
        time.sleep(0.01)
        monitor.complete_operation(metric, success=False, error="Test error")

        assert metric.success is False
        assert metric.error == "Test error"
        assert monitor.error_count == 1

    def test_performance_monitor_multiple_operations(self, monitor):
        """Test multiple operation monitoring."""
        # Start and complete multiple operations
        for i in range(3):
            metric = monitor.start_operation(f"operation_{i}")
            time.sleep(0.01)
            monitor.complete_operation(metric)

        assert monitor.operation_count == 3
        assert monitor.total_duration > 0
        assert monitor.max_duration > 0
        assert monitor.min_duration < float("inf")

    def test_performance_monitor_statistics(self, monitor):
        """Test performance statistics."""
        # Generate some operations
        for i in range(5):
            metric = monitor.start_operation(f"operation_{i}")
            time.sleep(0.01)
            success = i < 4  # Make last operation fail
            monitor.complete_operation(metric, success=success)

        stats = monitor.get_performance_summary()

        assert stats["total_operations"] == 5
        assert stats["avg_duration_ms"] > 0
        assert stats["max_duration_ms"] > 0
        assert stats["min_duration_ms"] > 0
        assert stats["error_rate"] == 0.2  # 1 error out of 5
        assert stats["success_rate"] == 0.8  # 4 successes out of 5

    def test_performance_monitor_get_recent_operations(self, monitor):
        """Test getting recent operations."""
        # Generate operations
        for i in range(10):
            metric = monitor.start_operation(f"operation_{i}")
            time.sleep(0.001)
            monitor.complete_operation(metric)

        # Get recent operations
        recent = monitor.get_recent_operations(5)
        assert len(recent) == 5

        # Should be most recent operations
        assert recent[0].operation_name == "operation_9"
        assert recent[4].operation_name == "operation_5"

    def test_performance_monitor_clear_metrics(self, monitor):
        """Test clearing metrics."""
        # Generate some operations
        metric = monitor.start_operation("test_operation")
        monitor.complete_operation(metric)

        assert monitor.operation_count == 1

        monitor.clear_metrics()

        assert monitor.operation_count == 0
        assert monitor.total_duration == 0.0
        assert monitor.max_duration == 0.0
        assert monitor.min_duration == float("inf")
        assert monitor.error_count == 0
        assert len(monitor.operations) == 0

    def test_performance_monitor_slow_operation_detection(self, monitor):
        """Test slow operation detection."""
        # Set threshold for slow operations
        monitor.slow_operation_threshold = 0.05  # 50ms

        # Fast operation
        metric = monitor.start_operation("fast_operation")
        time.sleep(0.01)
        monitor.complete_operation(metric)

        # Slow operation
        metric = monitor.start_operation("slow_operation")
        time.sleep(0.06)
        monitor.complete_operation(metric)

        slow_ops = monitor.get_slow_operations()
        assert len(slow_ops) == 1
        assert slow_ops[0].operation_name == "slow_operation"

    def test_performance_monitor_concurrent_operations(self, monitor):
        """Test concurrent operation monitoring."""

        async def async_operation(name: str):
            metric = monitor.start_operation(name)
            await asyncio.sleep(0.01)
            monitor.complete_operation(metric)

        # Run concurrent operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        tasks = [async_operation(f"concurrent_{i}") for i in range(5)]
        loop.run_until_complete(asyncio.gather(*tasks))

        assert monitor.operation_count == 5
        assert monitor.total_duration > 0

        loop.close()


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer implementation."""

    @pytest.fixture
    def optimizer(self):
        """Create PerformanceOptimizer instance for testing."""
        opt = PerformanceOptimizer()
        # Clear all caches and reset monitor for clean test state
        opt.clear_all_caches()
        opt.monitor.clear_metrics()
        return opt

    def test_performance_optimizer_initialization(self, optimizer):
        """Test PerformanceOptimizer initialization."""
        assert optimizer is not None
        assert hasattr(optimizer, "query_cache")
        assert hasattr(optimizer, "hyde_cache")
        assert hasattr(optimizer, "vector_cache")
        assert hasattr(optimizer, "monitor")

    def test_performance_optimizer_cache_access(self, optimizer):
        """Test cache access through optimizer."""
        # Test query cache
        optimizer.query_cache.put("test_query", "result")
        assert optimizer.query_cache.get("test_query") == "result"

        # Test HyDE cache
        optimizer.hyde_cache.put("test_hyde", "hyde_result")
        assert optimizer.hyde_cache.get("test_hyde") == "hyde_result"

        # Test vector cache
        optimizer.vector_cache.put("test_vector", "vector_result")
        assert optimizer.vector_cache.get("test_vector") == "vector_result"

    def test_performance_optimizer_monitoring(self, optimizer):
        """Test performance monitoring through optimizer."""
        metric = optimizer.monitor.start_operation("test_operation")
        time.sleep(0.01)
        optimizer.monitor.complete_operation(metric)

        assert optimizer.monitor.operation_count == 1
        assert optimizer.monitor.total_duration > 0

    def test_performance_optimizer_statistics(self, optimizer):
        """Test getting optimization statistics."""
        # Generate some cache activity
        optimizer.query_cache.put("query1", "result1")
        optimizer.query_cache.get("query1")
        optimizer.query_cache.get("nonexistent")

        # Generate some monitoring activity
        metric = optimizer.monitor.start_operation("test_operation")
        time.sleep(0.01)
        optimizer.monitor.complete_operation(metric)

        stats = optimizer.get_optimization_stats()

        assert "query_cache" in stats
        assert "hyde_cache" in stats
        assert "vector_cache" in stats
        assert "performance_monitor" in stats

        # Check cache stats
        assert stats["query_cache"]["hits"] == 1
        assert stats["query_cache"]["misses"] == 1
        assert stats["query_cache"]["size"] == 1

        # Check monitor stats
        assert stats["performance_monitor"]["total_operations"] == 1

    def test_performance_optimizer_clear_all_caches(self, optimizer):
        """Test clearing all caches."""
        # Add items to all caches
        optimizer.query_cache.put("query1", "result1")
        optimizer.hyde_cache.put("hyde1", "hyde_result1")
        optimizer.vector_cache.put("vector1", "vector_result1")

        assert optimizer.query_cache.size > 0
        assert optimizer.hyde_cache.size > 0
        assert optimizer.vector_cache.size > 0

        optimizer.clear_all_caches()

        assert optimizer.query_cache.size == 0
        assert optimizer.hyde_cache.size == 0
        assert optimizer.vector_cache.size == 0

    @pytest.mark.asyncio
    async def test_performance_optimizer_warm_up(self, optimizer):
        """Test cache warm-up functionality."""
        warm_up_data = {
            "query_cache": {"common_query": "common_result"},
            "hyde_cache": {"common_hyde": "common_hyde_result"},
            "vector_cache": {"common_vector": "common_vector_result"},
        }

        await optimizer.warm_up_caches(warm_up_data)

        assert optimizer.query_cache.get("common_query") == "common_result"
        assert optimizer.hyde_cache.get("common_hyde") == "common_hyde_result"
        assert optimizer.vector_cache.get("common_vector") == "common_vector_result"


class TestAsyncBatcher:
    """Test suite for AsyncBatcher implementation."""

    @pytest.fixture
    def batcher(self):
        """Create AsyncBatcher instance for testing."""
        return AsyncBatcher(batch_size=3, flush_interval=0.1)

    def test_async_batcher_initialization(self, batcher):
        """Test AsyncBatcher initialization."""
        assert batcher.batch_size == 3
        assert batcher.flush_interval == 0.1
        assert len(batcher.batches) == 0

    @pytest.mark.asyncio
    async def test_async_batcher_add_items(self, batcher):
        """Test adding items to batcher."""
        # Mock batch processor
        processed_batches = []

        async def process_batch(batch):
            processed_batches.append(batch)

        batcher.set_batch_processor(process_batch)

        # Add items
        await batcher.add_item("item1")
        await batcher.add_item("item2")
        await batcher.add_item("item3")  # Should trigger batch processing

        # Give time for processing
        await asyncio.sleep(0.01)

        assert len(processed_batches) == 1
        assert processed_batches[0] == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_async_batcher_flush_interval(self, batcher):
        """Test flush interval functionality."""
        processed_batches = []

        async def process_batch(batch):
            processed_batches.append(batch)

        batcher.set_batch_processor(process_batch)

        # Add items but don't reach batch size
        await batcher.add_item("item1")
        await batcher.add_item("item2")

        # Wait for flush interval
        await asyncio.sleep(0.15)

        assert len(processed_batches) == 1
        assert processed_batches[0] == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_async_batcher_manual_flush(self, batcher):
        """Test manual flush functionality."""
        processed_batches = []

        async def process_batch(batch):
            processed_batches.append(batch)

        batcher.set_batch_processor(process_batch)

        # Add items
        await batcher.add_item("item1")
        await batcher.add_item("item2")

        # Manual flush
        await batcher.flush()

        assert len(processed_batches) == 1
        assert processed_batches[0] == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_async_batcher_shutdown(self, batcher):
        """Test batcher shutdown."""
        processed_batches = []

        async def process_batch(batch):
            processed_batches.append(batch)

        batcher.set_batch_processor(process_batch)

        # Add items
        await batcher.add_item("item1")
        await batcher.add_item("item2")

        # Shutdown should flush remaining items
        await batcher.shutdown()

        assert len(processed_batches) == 1
        assert processed_batches[0] == ["item1", "item2"]


class TestConnectionPool:
    """Test suite for ConnectionPool implementation."""

    @pytest.fixture
    def connection_pool(self):
        """Create ConnectionPool instance for testing."""
        return ConnectionPool(max_size=5, timeout=1.0)

    def test_connection_pool_initialization(self, connection_pool):
        """Test ConnectionPool initialization."""
        assert connection_pool.max_size == 5
        assert connection_pool.timeout == 1.0
        assert connection_pool.active_connections == 0
        assert connection_pool.available_connections == 0

    @pytest.mark.asyncio
    async def test_connection_pool_acquire_release(self, connection_pool):
        """Test acquiring and releasing connections."""
        # Mock connection factory
        connection_count = 0

        async def create_connection():
            nonlocal connection_count
            connection_count += 1
            return Mock(id=connection_count)

        connection_pool.set_connection_factory(create_connection)

        # Acquire connection
        conn = await connection_pool.acquire()
        assert conn is not None
        assert connection_pool.active_connections == 1

        # Release connection
        await connection_pool.release(conn)
        assert connection_pool.active_connections == 0
        assert connection_pool.available_connections == 1

    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self, connection_pool):
        """Test connection reuse."""

        async def create_connection():
            return Mock(id=1)

        connection_pool.set_connection_factory(create_connection)

        # Acquire and release
        conn1 = await connection_pool.acquire()
        await connection_pool.release(conn1)

        # Acquire again, should reuse
        conn2 = await connection_pool.acquire()
        assert conn2 is conn1  # Should be same connection

    @pytest.mark.asyncio
    async def test_connection_pool_max_size(self, connection_pool):
        """Test connection pool max size limit."""

        async def create_connection():
            return Mock()

        connection_pool.set_connection_factory(create_connection)

        # Acquire max connections
        connections = []
        for _ in range(5):
            conn = await connection_pool.acquire()
            connections.append(conn)

        assert connection_pool.active_connections == 5

        # Try to acquire one more (should timeout)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(connection_pool.acquire(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_connection_pool_context_manager(self, connection_pool):
        """Test connection pool context manager."""

        async def create_connection():
            return Mock()

        connection_pool.set_connection_factory(create_connection)

        # Use context manager
        async with connection_pool.get_connection() as conn:
            assert conn is not None
            assert connection_pool.active_connections == 1

        # Connection should be released
        assert connection_pool.active_connections == 0
        assert connection_pool.available_connections == 1

    @pytest.mark.asyncio
    async def test_connection_pool_close_all(self, connection_pool):
        """Test closing all connections."""

        async def create_connection():
            return Mock()

        connection_pool.set_connection_factory(create_connection)

        # Acquire connections
        conn1 = await connection_pool.acquire()
        _conn2 = await connection_pool.acquire()
        await connection_pool.release(conn1)

        assert connection_pool.active_connections == 1
        assert connection_pool.available_connections == 1

        # Close all
        await connection_pool.close_all()

        assert connection_pool.active_connections == 0
        assert connection_pool.available_connections == 0


class TestPerformanceDecorators:
    """Test suite for performance decorators."""

    def test_cache_query_analysis_decorator(self):
        """Test cache_query_analysis decorator."""
        call_count = 0

        @cache_query_analysis
        async def analyze_query(query: str):
            nonlocal call_count
            call_count += 1
            return f"analyzed_{query}"

        # Run async test
        async def test_caching():
            # First call
            result1 = await analyze_query("test_query")
            assert result1 == "analyzed_test_query"
            assert call_count == 1

            # Second call (should use cache)
            result2 = await analyze_query("test_query")
            assert result2 == "analyzed_test_query"
            assert call_count == 1  # Should not increment

            # Different query
            result3 = await analyze_query("different_query")
            assert result3 == "analyzed_different_query"
            assert call_count == 2

        asyncio.run(test_caching())

    def test_cache_hyde_processing_decorator(self):
        """Test cache_hyde_processing decorator."""
        call_count = 0

        @cache_hyde_processing
        async def process_hyde(query: str):
            nonlocal call_count
            call_count += 1
            return f"processed_{query}"

        async def test_caching():
            # First call
            result1 = await process_hyde("test_query")
            assert result1 == "processed_test_query"
            assert call_count == 1

            # Second call (should use cache)
            result2 = await process_hyde("test_query")
            assert result2 == "processed_test_query"
            assert call_count == 1

        asyncio.run(test_caching())

    def test_cache_vector_search_decorator(self):
        """Test cache_vector_search decorator."""
        call_count = 0

        @cache_vector_search
        async def search_vectors(embeddings: list[list[float]]):
            nonlocal call_count
            call_count += 1
            return f"search_results_{len(embeddings)}"

        async def test_caching():
            embeddings = [[0.1, 0.2], [0.3, 0.4]]

            # First call
            result1 = await search_vectors(embeddings)
            assert result1 == "search_results_2"
            assert call_count == 1

            # Second call (should use cache)
            result2 = await search_vectors(embeddings)
            assert result2 == "search_results_2"
            assert call_count == 1

        asyncio.run(test_caching())

    def test_monitor_performance_decorator(self):
        """Test monitor_performance decorator."""

        @monitor_performance("test_operation")
        async def test_function():
            await asyncio.sleep(0.01)
            return "test_result"

        async def test_monitoring():
            initial_count = _performance_monitor.operation_count

            result = await test_function()
            assert result == "test_result"
            assert _performance_monitor.operation_count == initial_count + 1

        asyncio.run(test_monitoring())

    def test_decorator_error_handling(self):
        """Test decorator error handling."""

        @cache_query_analysis
        @monitor_performance("error_operation")
        async def error_function():
            raise ValueError("Test error")

        async def test_error():
            initial_error_count = _performance_monitor.error_count

            with pytest.raises(ValueError, match="Test error"):
                await error_function()

            assert _performance_monitor.error_count == initial_error_count + 1

        asyncio.run(test_error())


class TestGlobalFunctions:
    """Test suite for global utility functions."""

    def test_get_performance_stats(self):
        """Test get_performance_stats function."""
        # Generate some activity
        _query_cache.put("test_query", "result")
        _query_cache.get("test_query")
        _query_cache.get("nonexistent")

        stats = get_performance_stats()

        assert "query_cache" in stats
        assert "hyde_cache" in stats
        assert "vector_cache" in stats
        assert "performance_monitor" in stats

        # Check structure
        assert "hits" in stats["query_cache"]
        assert "misses" in stats["query_cache"]
        assert "size" in stats["query_cache"]
        assert "hit_rate" in stats["query_cache"]

    def test_clear_all_caches(self):
        """Test clear_all_caches function."""
        # Add items to caches
        _query_cache.put("query1", "result1")
        _hyde_cache.put("hyde1", "hyde_result1")
        _vector_cache.put("vector1", "vector_result1")

        assert _query_cache.size > 0
        assert _hyde_cache.size > 0
        assert _vector_cache.size > 0

        clear_all_caches()

        assert _query_cache.size == 0
        assert _hyde_cache.size == 0
        assert _vector_cache.size == 0

    @pytest.mark.asyncio
    async def test_warm_up_system(self):
        """Test warm_up_system function."""
        # Clear caches first
        clear_all_caches()

        await warm_up_system()

        # Should have some pre-populated cache entries
        stats = get_performance_stats()
        total_items = stats["query_cache"]["size"] + stats["hyde_cache"]["size"] + stats["vector_cache"]["size"]

        assert total_items > 0

    def test_performance_metric_model(self):
        """Test PerformanceMetric model."""
        metric = PerformanceMetric(operation_name="test_operation", start_time=time.time())

        assert metric.operation_name == "test_operation"
        assert metric.start_time > 0
        assert metric.end_time is None
        assert metric.duration is None
        assert metric.success is None
        assert metric.error is None

        # Complete the metric
        metric.end_time = time.time()
        metric.duration = metric.end_time - metric.start_time
        metric.success = True

        assert metric.duration > 0
        assert metric.success is True


class TestPerformanceIntegration:
    """Integration tests for performance optimization components."""

    @pytest.mark.asyncio
    async def test_end_to_end_performance_optimization(self):
        """Test end-to-end performance optimization."""
        # Clear caches
        clear_all_caches()

        # Define test function with all optimizations
        @cache_query_analysis
        @monitor_performance("integrated_operation")
        async def integrated_function(query: str):
            await asyncio.sleep(0.01)
            return f"processed_{query}"

        # First call
        result1 = await integrated_function("test_query")
        assert result1 == "processed_test_query"

        # Second call (should use cache)
        start_time = time.time()
        result2 = await integrated_function("test_query")
        cache_time = time.time() - start_time

        assert result2 == "processed_test_query"
        assert cache_time < 0.005  # Should be much faster due to caching

        # Check statistics
        stats = get_performance_stats()
        assert stats["query_cache"]["hits"] >= 1
        assert stats["performance_monitor"]["total_operations"] >= 2

    @pytest.mark.asyncio
    async def test_concurrent_performance_optimization(self):
        """Test performance optimization under concurrent load."""

        # Clear all metrics and caches first to avoid interference from other tests
        _performance_monitor.clear_metrics()
        clear_all_caches()

        @monitor_performance("concurrent_operation")
        async def concurrent_function(query: str):
            await asyncio.sleep(0.01)
            return f"concurrent_{query}"

        # Run concurrent operations
        tasks = []
        for i in range(10):
            query = f"query_{i % 3}"  # Cycle through 3 different queries
            tasks.append(concurrent_function(query))

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for i, result in enumerate(results):
            expected_query = f"query_{i % 3}"
            assert result == f"concurrent_{expected_query}"

        # Check that performance monitoring tracked all operations
        stats = get_performance_stats()
        assert stats["performance_monitor"]["total_operations"] == 10  # All operations monitored

        # Test demonstrates concurrent execution capability
        assert all("concurrent_" in result for result in results)

    def test_performance_configuration_validation(self):
        """Test performance configuration validation."""
        # Test invalid cache configuration
        with pytest.raises(ValueError, match="max_size must be non-negative"):
            LRUCache(max_size=-1, ttl_seconds=60)

        with pytest.raises(ValueError, match="ttl_seconds must be non-negative"):
            LRUCache(max_size=100, ttl_seconds=-1)

        # Test invalid batcher configuration
        with pytest.raises(ValueError, match="batch_size must be positive"):
            AsyncBatcher(batch_size=0, flush_interval=0.1)

        with pytest.raises(ValueError, match="flush_interval must be non-negative"):
            AsyncBatcher(batch_size=10, flush_interval=-1)

        # Test invalid connection pool configuration
        with pytest.raises(ValueError, match="max_size must be positive"):
            ConnectionPool(max_size=0, timeout=1.0)

        with pytest.raises(ValueError, match="timeout must be non-negative"):
            ConnectionPool(max_size=5, timeout=-1)


class TestPerformanceMemoryManagement:
    """Test memory management in performance optimization."""

    def test_cache_memory_limit(self):
        """Test cache memory limit enforcement."""
        cache = LRUCache(max_size=3, ttl_seconds=60)

        # Add items up to limit
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")

        # Should only have 3 items (most recent)
        assert cache.size == 3
        assert cache.get("key_0") is None  # Should be evicted
        assert cache.get("key_1") is None  # Should be evicted
        assert cache.get("key_2") is not None  # Should be present
        assert cache.get("key_3") is not None  # Should be present
        assert cache.get("key_4") is not None  # Should be present

    def test_monitor_memory_efficiency(self):
        """Test monitor memory efficiency."""
        monitor = PerformanceMonitor(max_operations=100)

        # Add many operations
        for i in range(200):
            metric = monitor.start_operation(f"operation_{i}")
            monitor.complete_operation(metric)

        # Should only keep latest 100 operations
        assert len(monitor.operations) == 100
        assert monitor.operations[0].operation_name == "operation_100"
        assert monitor.operations[-1].operation_name == "operation_199"

    def test_cache_cleanup_on_ttl(self):
        """Test cache cleanup on TTL expiration."""
        cache = LRUCache(max_size=10, ttl_seconds=0.05)

        # Add items
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")

        assert cache.size == 5

        # Wait for TTL expiration
        time.sleep(0.1)

        # Access cache to trigger cleanup
        cache.get("key_0")

        # All items should be expired and cleaned up
        assert cache.size == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
