"""Comprehensive tests for performance_optimizer.py coverage gaps - targeting 0% coverage functions."""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.core.performance_optimizer import (
    AsyncBatcher,
    PerformanceOptimizer,
)


class TestAsyncBatcherCoverageGaps:
    """Test AsyncBatcher methods with 0% coverage."""

    @pytest.fixture
    async def async_batcher(self):
        """Create AsyncBatcher for testing."""
        batcher = AsyncBatcher(batch_size=3, flush_interval=0.1)
        yield batcher
        await batcher.shutdown()

    async def test_add_operation_method(self, async_batcher):
        """Test add_operation method with 0% coverage."""

        # Create async operations for testing
        async def mock_op1(*args, **kwargs):
            return f"result1-{args[0] if args else 'no-args'}"

        async def mock_op2(*args, **kwargs):
            return f"result2-{args[0] if args else 'no-args'}"

        # Test adding single operation (should timeout and execute)
        result1 = await async_batcher.add_operation(mock_op1, "arg1")

        # Verify first operation result (batch executed due to timeout)
        assert result1 == ["result1-arg1"]

        # Test adding operation that triggers batch size limit
        # Add operations up to batch size
        async def simple_op(*args, **kwargs):
            return f"batch-{args[0] if args else 'default'}"

        # Fill up to batch size - 1
        for i in range(async_batcher.batch_size - 1):
            async_batcher.pending_operations.append((simple_op, (f"item{i}",), {}))

        # This should trigger immediate batch execution
        result2 = await async_batcher.add_operation(mock_op2, "trigger")

        # Verify batch was executed
        assert len(result2) == async_batcher.batch_size  # batch_size operations total

    async def test_execute_batch_method(self, async_batcher):
        """Test _execute_batch method with 0% coverage."""

        # Create async mock operations that accept arguments
        async def mock_op1(*args, **kwargs):
            return f"result1-{args[0] if args else 'no-args'}"

        async def mock_op2(*args, **kwargs):
            return f"result2-{args[0] if args else 'no-args'}"

        # Add operations directly to pending list
        async_batcher.pending_operations = [(mock_op1, ("arg1",), {}), (mock_op2, ("arg2",), {})]

        # Execute batch
        results = await async_batcher._execute_batch()

        # Verify batch execution
        assert len(results) == 2
        assert results[0] == "result1-arg1"
        assert results[1] == "result2-arg2"
        assert async_batcher.pending_operations == []

    async def test_execute_batch_empty(self, async_batcher):
        """Test _execute_batch with empty operations."""
        # Test with no pending operations
        results = await async_batcher._execute_batch()

        # Should return empty list
        assert results == []

    async def test_add_operation_batch_size_trigger(self, async_batcher):
        """Test add_operation triggering batch execution when batch_size reached."""

        # Create async mock operations
        async def mock_op_async():
            return "async_result"

        # Add operations up to batch size
        for _ in range(3):  # batch_size is 3
            async_batcher.pending_operations.append((mock_op_async, (), {}))

        # This should trigger batch execution
        await async_batcher.add_operation(mock_op_async)

        # Verify batch was executed (pending operations cleared)
        assert len(async_batcher.pending_operations) == 0

    async def test_add_operation_timeout_trigger(self, async_batcher):
        """Test add_operation triggering batch execution on timeout."""

        # Create async mock operation
        async def mock_op_async():
            return "timeout_result"

        # Add single operation (below batch_size)
        result = await async_batcher.add_operation(mock_op_async)

        # Should execute after timeout (max_wait_time)
        assert result is not None or len(async_batcher.pending_operations) == 0


class TestPerformanceOptimizerCoverageGaps:
    """Test PerformanceOptimizer methods with 0% coverage."""

    @pytest.fixture
    def performance_optimizer(self):
        """Create PerformanceOptimizer for testing."""
        return PerformanceOptimizer()

    async def test_optimize_query_processing_method(self, performance_optimizer):
        """Test optimize_query_processing method with 0% coverage."""
        test_query = "test query for optimization"

        # Test query processing
        result = await performance_optimizer.optimize_query_processing(test_query)

        # Verify result structure
        assert isinstance(result, dict)
        assert result["query"] == test_query
        assert result["optimized"] is True
        assert "timestamp" in result

        # Test cache hit on second call
        result2 = await performance_optimizer.optimize_query_processing(test_query)

        # Should be same result from cache
        assert result == result2

    async def test_optimize_query_processing_cache_miss(self, performance_optimizer):
        """Test optimize_query_processing with cache miss."""
        # Clear cache first
        performance_optimizer.clear_all_caches()

        test_query = "unique query for cache miss test"

        # First call should be cache miss
        result = await performance_optimizer.optimize_query_processing(test_query)

        # Verify cache miss indicators
        assert result["cache_miss"] is True
        assert result["optimized"] is True

    async def test_optimize_query_processing_error_handling(self, performance_optimizer):
        """Test optimize_query_processing error handling."""
        # Patch the cache to raise an exception
        with patch.object(performance_optimizer.query_cache, "get", side_effect=Exception("Cache error")):
            test_query = "error query"

            # Should handle error gracefully
            with pytest.raises(Exception, match="Cache error"):
                await performance_optimizer.optimize_query_processing(test_query)

    async def test_warm_up_with_default_data_method(self, performance_optimizer):
        """Test _warm_up_with_default_data method with 0% coverage."""
        # Test warm-up with default data
        await performance_optimizer._warm_up_with_default_data()

        # Verify some queries were processed
        cache_stats = performance_optimizer.query_cache.get_stats()
        assert cache_stats["size"] > 0

    async def test_warm_up_with_default_data_error_handling(self, performance_optimizer):
        """Test _warm_up_with_default_data with error handling."""
        # Mock optimize_query_processing to raise error for some queries
        original_method = performance_optimizer.optimize_query_processing

        async def mock_optimize_query_processing(query):
            if "error" in query.lower():
                raise Exception("Processing error")
            return await original_method(query)

        with patch.object(
            performance_optimizer,
            "optimize_query_processing",
            side_effect=mock_optimize_query_processing,
        ):
            # Should handle errors and continue with other queries
            await performance_optimizer._warm_up_with_default_data()

    async def test_warm_up_with_default_data_performance(self, performance_optimizer):
        """Test performance of _warm_up_with_default_data."""
        start_time = time.time()

        # Warm up with default data
        await performance_optimizer._warm_up_with_default_data()

        duration = time.time() - start_time

        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds max


class TestPerformanceOptimizerIntegration:
    """Test integration scenarios for PerformanceOptimizer."""

    @pytest.fixture
    def performance_optimizer(self):
        """Create PerformanceOptimizer for testing."""
        return PerformanceOptimizer()

    async def test_full_optimization_workflow(self, performance_optimizer):
        """Test complete optimization workflow."""
        # Clear caches initially
        performance_optimizer.clear_all_caches()

        # Warm up system
        await performance_optimizer._warm_up_with_default_data()

        # Process some queries
        queries = ["test query 1", "test query 2", "test query 3"]
        results = []

        for query in queries:
            result = await performance_optimizer.optimize_query_processing(query)
            results.append(result)

        # Verify all queries were processed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["query"] == queries[i]
            assert result["optimized"] is True

        # Get optimization stats
        stats = performance_optimizer.get_optimization_stats()
        assert "cache_stats" in stats
        assert "connection_pool_size" in stats
        assert "batcher_pending" in stats

    async def test_concurrent_query_processing(self, performance_optimizer):
        """Test concurrent query processing performance."""
        # Create multiple concurrent queries
        queries = [f"concurrent query {i}" for i in range(10)]

        # Process all queries concurrently
        tasks = [performance_optimizer.optimize_query_processing(query) for query in queries]
        results = await asyncio.gather(*tasks)

        # Verify all results
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["query"] == queries[i]
            assert result["optimized"] is True

    async def test_cache_efficiency(self, performance_optimizer):
        """Test cache efficiency with repeated queries."""
        query = "repeated query for cache test"

        # Clear cache first
        performance_optimizer.clear_all_caches()

        # First call - should be cache miss
        result1 = await performance_optimizer.optimize_query_processing(query)
        assert result1["cache_miss"] is True

        # Subsequent calls - should be cache hits
        start_time = time.time()
        for _ in range(100):
            result = await performance_optimizer.optimize_query_processing(query)
            assert result["query"] == query

        cache_duration = time.time() - start_time

        # Cache hits should be very fast
        assert cache_duration < 0.1  # 100ms for 100 cache hits


class TestAsyncBatcherAdvanced:
    """Test advanced AsyncBatcher functionality."""

    async def test_batcher_with_exceptions(self):
        """Test AsyncBatcher handling operations that raise exceptions."""
        batcher = AsyncBatcher(batch_size=2, flush_interval=0.1)

        # Create operations that raise exceptions
        async def failing_op():
            raise ValueError("Operation failed")

        async def succeeding_op():
            return "success"

        try:
            # Add operations to pending list
            batcher.pending_operations = [(failing_op, (), {}), (succeeding_op, (), {})]

            # Execute batch
            results = await batcher._execute_batch()

            # Should handle both success and failure
            assert len(results) == 2
            assert isinstance(results[0], ValueError)
            assert results[1] == "success"

        finally:
            await batcher.shutdown()

    async def test_batcher_large_batch(self):
        """Test AsyncBatcher with large batch size."""
        batcher = AsyncBatcher(batch_size=100, flush_interval=0.1)

        async def mock_operation(value):
            return f"processed_{value}"

        try:
            # Add many operations
            operations = [(mock_operation, (i,), {}) for i in range(50)]
            batcher.pending_operations = operations

            # Execute batch
            results = await batcher._execute_batch()

            # Verify all operations processed
            assert len(results) == 50
            for i, result in enumerate(results):
                assert result == f"processed_{i}"

        finally:
            await batcher.shutdown()


@pytest.mark.parametrize(
    "query",
    [
        "simple query",
        "query with special characters: !@#$%^&*()",
        "very long query " * 100,
        "",  # empty query
        "unicode query: 测试查询 🚀",
    ],
)
async def test_optimize_query_processing_various_inputs(query):
    """Test optimize_query_processing with various input types."""
    optimizer = PerformanceOptimizer()

    result = await optimizer.optimize_query_processing(query)

    assert result["query"] == query
    assert result["optimized"] is True
    assert "timestamp" in result


@pytest.mark.parametrize(
    ("batch_size", "flush_interval"),
    [
        (1, 0.01),
        (10, 0.1),
        (100, 0.5),
    ],
)
async def test_async_batcher_configurations(batch_size, flush_interval):
    """Test AsyncBatcher with different configurations."""
    batcher = AsyncBatcher(batch_size=batch_size, flush_interval=flush_interval)

    async def test_operation():
        return "test_result"

    try:
        # Add operation to pending list
        batcher.pending_operations = [(test_operation, (), {})]

        # Execute batch
        results = await batcher._execute_batch()

        # Verify configuration works
        assert len(results) == 1
        assert results[0] == "test_result"

    finally:
        await batcher.shutdown()
