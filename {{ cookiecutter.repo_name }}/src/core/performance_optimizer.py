"""
Performance optimization utilities for PromptCraft-Hybrid core components.

This module provides performance enhancements to ensure the <2s response time
requirement is met consistently across all operations. It includes caching,
connection pooling, async optimizations, and monitoring.

Key optimizations:
- LRU caching for frequent operations
- Connection pooling for vector stores
- Async batching for multiple operations
- Query result caching
- Performance monitoring and alerting
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

# Type variables for generic caching
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Performance constants
CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_CACHE_SIZE = 1000
BATCH_SIZE = 50
PERFORMANCE_THRESHOLD_MS = 2000  # 2 seconds
MAX_METRICS_COUNT = 10000
METRICS_TRIMMED_COUNT = 5000
RECENT_METRICS_SECONDS = 300  # 5 minutes


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""

    operation_name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    cache_hit: bool = False
    batch_size: int = 1
    error_occurred: bool = False
    success: bool | None = None
    error: str | None = None

    @property
    def duration(self) -> float | None:
        """Get duration in seconds (for compatibility)."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @duration.setter
    def duration(self, value: float) -> None:
        """Set duration (for compatibility) - calculates end_time."""
        if value is not None:
            self.end_time = self.start_time + value

    def complete(self) -> None:
        """Mark operation as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if self.success is None:
            self.success = True

    def is_slow(self) -> bool:
        """Check if operation exceeded performance threshold."""
        return self.duration_ms > PERFORMANCE_THRESHOLD_MS


# Alias for backward compatibility
PerformanceMetric = PerformanceMetrics


class LRUCache:
    """High-performance LRU cache with TTL support."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    @property
    def hits(self) -> int:
        """Get number of cache hits."""
        return self.stats["hits"]

    @property
    def misses(self) -> int:
        """Get number of cache misses."""
        return self.stats["misses"]

    def contains(self, key: str) -> bool:
        """Check if key exists in cache (without affecting stats)."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            return not self._is_expired(timestamp)
        return False

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items() if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
            self.stats["evictions"] += 1

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        self._evict_expired()

        if key in self.cache:
            value, timestamp = self.cache[key]
            if not self._is_expired(timestamp):
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return value
            del self.cache[key]
            self.stats["evictions"] += 1

        self.stats["misses"] += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        current_time = time.time()

        # Handle zero-size cache
        if self.max_size == 0:
            return

        if key in self.cache:
            # Update existing entry
            self.cache[key] = (value, current_time)
            self.cache.move_to_end(key)
        else:
            # Add new entry
            if len(self.cache) >= self.max_size and self.cache:
                # Remove least recently used
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1

            self.cache[key] = (value, current_time)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
        }

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class PerformanceMonitor:
    """Performance monitoring and alerting system."""

    def __init__(self, max_operations: int | None = None) -> None:
        self.metrics: list[PerformanceMetrics] = []
        self.alerts: list[str] = []
        self.logger = logging.getLogger(__name__)
        self.max_operations = max_operations
        self.operation_count = 0
        self.error_count = 0
        self._total_duration = 0.0
        self._max_duration = 0.0
        self._min_duration = float("inf")
        self.slow_operation_threshold = 2.0  # Default 2 seconds

    @property
    def total_duration(self) -> float:
        """Get total duration of all operations."""
        return self._total_duration

    @property
    def max_duration(self) -> float:
        """Get maximum duration of any operation."""
        return self._max_duration

    @property
    def min_duration(self) -> float:
        """Get minimum duration of any operation."""
        return self._min_duration

    @property
    def operations(self) -> list[PerformanceMetrics]:
        """Get list of all operations."""
        return self.metrics

    def start_operation(self, operation_name: str) -> PerformanceMetrics:
        """Start tracking an operation."""
        return PerformanceMetrics(operation_name=operation_name, start_time=time.time())

    def complete_operation(
        self,
        metric: PerformanceMetrics,
        **kwargs: Any,
    ) -> None:
        """Complete operation tracking.

        Args:
            metric: Performance metrics object to complete
            **kwargs: Additional parameters including:
                - cache_hit: Whether cache was hit (default: False)
                - batch_size: Size of the batch processed (default: 1)
                - error_occurred: Whether an error occurred (default: False)
                - success: Whether operation was successful
                - error: Error message if any
        """
        metric.complete()
        metric.cache_hit = kwargs.get("cache_hit", False)
        metric.batch_size = kwargs.get("batch_size", 1)
        metric.error_occurred = kwargs.get("error_occurred", False)

        # Handle success/error parameters
        success = kwargs.get("success")
        error = kwargs.get("error")
        if success is not None:
            metric.success = success
        if error is not None:
            metric.error = error
            metric.success = False
            metric.error_occurred = True

        self.metrics.append(metric)
        self.operation_count += 1

        # Update duration statistics
        duration_seconds = metric.duration or 0.0
        self._total_duration += duration_seconds
        self._max_duration = max(self._max_duration, duration_seconds)
        self._min_duration = min(self._min_duration, duration_seconds)

        if kwargs.get("error_occurred", False) or metric.error_occurred:
            self.error_count += 1

        # Check for performance issues
        if metric.is_slow():
            alert = f"Slow operation detected: {metric.operation_name} took {metric.duration_ms:.0f}ms"
            self.alerts.append(alert)
            self.logger.warning(alert)

        # Keep only recent metrics based on max_operations or default
        max_count = self.max_operations if self.max_operations is not None else MAX_METRICS_COUNT
        if len(self.metrics) > max_count:
            trim_count = self.max_operations if self.max_operations is not None else METRICS_TRIMMED_COUNT
            self.metrics = self.metrics[-trim_count:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {"total_operations": 0}

        recent_metrics = [m for m in self.metrics if m.end_time and time.time() - m.end_time < RECENT_METRICS_SECONDS]

        durations = [m.duration_ms for m in recent_metrics]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        errors = sum(1 for m in recent_metrics if m.error_occurred or not m.success)
        successes = sum(1 for m in recent_metrics if m.success)
        slow_operations = sum(1 for m in recent_metrics if m.is_slow())

        return {
            "total_operations": len(recent_metrics),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "cache_hit_rate": cache_hits / len(recent_metrics) if recent_metrics else 0,
            "error_rate": errors / len(recent_metrics) if recent_metrics else 0,
            "success_rate": successes / len(recent_metrics) if recent_metrics else 0,
            "slow_operation_rate": slow_operations / len(recent_metrics) if recent_metrics else 0,
            "recent_alerts": self.alerts[-10:] if self.alerts else [],
        }

    def get_recent_operations(self, count: int) -> list[PerformanceMetrics]:
        """Get most recent operations."""
        if not self.metrics:
            return []
        # Return most recent first (reverse order)
        return list(reversed(self.metrics[-count:]))

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.alerts.clear()
        self.operation_count = 0
        self.error_count = 0
        self._total_duration = 0.0
        self._max_duration = 0.0
        self._min_duration = float("inf")

    def get_slow_operations(self) -> list[PerformanceMetrics]:
        """Get list of slow operations."""
        return [m for m in self.metrics if m.duration and m.duration * 1000 > self.slow_operation_threshold * 1000]


# Global instances
_query_cache = LRUCache(max_size=500, ttl_seconds=300)
_hyde_cache = LRUCache(max_size=200, ttl_seconds=600)
_vector_cache = LRUCache(max_size=1000, ttl_seconds=180)
_performance_monitor = PerformanceMonitor()


def cache_query_analysis(func: F) -> F:
    """Decorator to cache query analysis results."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create cache key from query content - handle different argument positions
        query = None
        if args and len(args) > 1:
            query = args[1]  # Second argument (self, query)
        elif args and len(args) > 0 and isinstance(args[0], str):
            query = args[0]  # First argument is query
        else:
            query = kwargs.get("query", "")

        if not isinstance(query, str):
            query = str(query)

        cache_key = f"query_analysis:{hashlib.sha256(query.encode()).hexdigest()}"

        # Check cache first
        cached_result = _query_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Execute function and cache result
        result = await func(*args, **kwargs)
        _query_cache.put(cache_key, result)

        return result

    return wrapper  # type: ignore[return-value]


def cache_hyde_processing(func: F) -> F:
    """Decorator to cache HyDE processing results."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create cache key from query and parameters
        query = args[1] if args and len(args) > 1 else kwargs.get("query", "")

        # Handle None query by converting to empty string
        if query is None:
            query = ""

        cache_key = f"hyde_processing:{hashlib.sha256(query.encode()).hexdigest()}"

        # Check cache first
        cached_result = _hyde_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Execute function and cache result
        result = await func(*args, **kwargs)
        _hyde_cache.put(cache_key, result)

        return result

    return wrapper  # type: ignore[return-value]


def cache_vector_search(func: F) -> F:
    """Decorator to cache vector search results."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # For simple test cases, create cache key from first argument (embeddings list)
        embeddings = args[0] if args else kwargs.get("embeddings")

        if embeddings and isinstance(embeddings, list):
            # Create hash from embeddings
            embedding_str = str(embeddings)
            cache_key = f"vector_search:{hashlib.sha256(embedding_str.encode()).hexdigest()}"
        else:
            # Fallback to original approach for complex parameters
            params = args[1] if args and len(args) > 1 else kwargs.get("parameters")

            if (
                params is not None
                and hasattr(params, "embeddings")
                and hasattr(params, "limit")
                and hasattr(params, "collection")
                and hasattr(params, "strategy")
                and params.embeddings
            ):
                # Create hash from embeddings and parameters
                embedding_str = str(params.embeddings[0][:10])  # First 10 dimensions
                strategy_str = params.strategy.value if hasattr(params.strategy, "value") else str(params.strategy)
                filters_str = str(sorted(params.filters.items())) if params.filters else "no_filters"
                cache_key = f"vector_search:{hashlib.sha256(embedding_str.encode()).hexdigest()}:{params.limit}:{params.collection}:{strategy_str}:{hashlib.sha256(filters_str.encode()).hexdigest()}"
            else:
                return await func(*args, **kwargs)

        # Check cache first
        cached_result = _vector_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Execute function and cache result
        result = await func(*args, **kwargs)
        _vector_cache.put(cache_key, result)

        return result

    return wrapper  # type: ignore[return-value]


def monitor_performance(operation_name: str) -> Callable[[F], F]:
    """Decorator to monitor operation performance."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            metric = _performance_monitor.start_operation(operation_name)

            try:
                result = await func(*args, **kwargs)
                _performance_monitor.complete_operation(metric, error_occurred=False)
                return result
            except Exception:
                _performance_monitor.complete_operation(metric, error_occurred=True)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


class AsyncBatcher:
    """Batches async operations for better performance."""

    def __init__(self, batch_size: int = BATCH_SIZE, flush_interval: float = 0.1) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if flush_interval < 0:
            raise ValueError("flush_interval must be non-negative")
        self.batch_size = batch_size
        self.flush_interval = flush_interval  # Use flush_interval instead of max_wait_time
        self.max_wait_time = flush_interval  # Keep for backwards compatibility
        self.pending_operations: list[tuple[Callable, tuple, dict]] = []
        self.batches: list[Any] = []  # Add batches attribute expected by tests
        self.batch_lock = asyncio.Lock()
        self._batch_processor: Callable | None = None
        self._flush_task: asyncio.Task | None = None
        self._shutdown = False

    def set_batch_processor(self, processor: Callable) -> None:
        """Set the batch processor function."""
        self._batch_processor = processor
        # Start the flush timer if not already running
        if self._flush_task is None and not self._shutdown:
            self._flush_task = asyncio.create_task(self._flush_timer())

    async def _flush_timer(self) -> None:
        """Timer-based flush mechanism."""
        while not self._shutdown:
            await asyncio.sleep(self.flush_interval)
            if not self._shutdown and self.batches and self._batch_processor:
                async with self.batch_lock:
                    if self.batches:  # Double-check with lock
                        batch_to_process = self.batches.copy()
                        self.batches.clear()
                        await self._batch_processor(batch_to_process)

    async def add_item(self, item: Any) -> None:
        """Add an item to the batch."""
        async with self.batch_lock:
            self.batches.append(item)

            if len(self.batches) >= self.batch_size and self._batch_processor:
                batch_to_process = self.batches.copy()
                self.batches.clear()
                await self._batch_processor(batch_to_process)

    async def flush(self) -> None:
        """Manually flush pending items."""
        async with self.batch_lock:
            if self.batches and self._batch_processor:
                batch_to_process = self.batches.copy()
                self.batches.clear()
                await self._batch_processor(batch_to_process)

    async def shutdown(self) -> None:
        """Shutdown the batcher and flush remaining items."""
        self._shutdown = True
        if self._flush_task:
            self._flush_task.cancel()
            # Wait for task to complete (ignore cancellation)
            with suppress(asyncio.CancelledError):
                await self._flush_task
        await self.flush()

    async def add_operation(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Add operation to batch."""
        async with self.batch_lock:
            self.pending_operations.append((operation, args, kwargs))

            if len(self.pending_operations) >= self.batch_size:
                return await self._execute_batch()

            # Wait for more operations or timeout
            await asyncio.sleep(self.max_wait_time)

            if self.pending_operations:
                return await self._execute_batch()

        return None

    async def _execute_batch(self) -> list[Any]:
        """Execute batched operations."""
        if not self.pending_operations:
            return []

        operations = self.pending_operations.copy()
        self.pending_operations.clear()

        # Execute all operations concurrently
        tasks = [op(*args, **kwargs) for op, args, kwargs in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)


class ConnectionPool:
    """Connection pool for vector store operations."""

    def __init__(self, max_size: int = 10, timeout: float = 5.0) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        self.max_size = max_size  # Use max_size instead of max_connections
        self.max_connections = max_size  # Keep for backwards compatibility
        self.timeout = timeout
        self.available_connections_queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.active_connections = 0
        self.connection_lock = asyncio.Lock()
        self._connection_factory: Callable | None = None

    def set_connection_factory(self, factory: Callable) -> None:
        """Set the connection factory function."""
        self._connection_factory = factory

    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        return await self.acquire_connection()

    async def release(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        await self.release_connection(connection)

    def get_connection(self) -> "ConnectionContext":
        """Context manager for getting a connection."""
        return ConnectionContext(self)

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self.connection_lock:
            # Clear the queue
            while not self.available_connections_queue.empty():
                try:
                    self.available_connections_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self.active_connections = 0

    @property
    def available_connections(self) -> int:
        """Get number of available connections."""
        return self.available_connections_queue.qsize()

    async def acquire_connection(self) -> Any:
        """Acquire a connection from the pool."""
        try:
            # Try to get existing connection
            return self.available_connections_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Create new connection if under limit
            async with self.connection_lock:
                if self.active_connections < self.max_size:
                    self.active_connections += 1
                    return await self._create_connection()
                # Wait for available connection
                return await self.available_connections_queue.get()

    async def release_connection(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        if connection:
            async with self.connection_lock:
                await self.available_connections_queue.put(connection)
                # Decrement active_connections when releasing
                if self.active_connections > 0:
                    self.active_connections -= 1

    async def _create_connection(self) -> Any:
        """Create a new connection using the factory if available."""
        if self._connection_factory:
            return await self._connection_factory()
        # Default implementation for backwards compatibility
        return {"connection_id": time.time()}


class ConnectionContext:
    """Context manager for connection pool."""

    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool
        self.connection = None

    async def __aenter__(self) -> Any:
        """Acquire connection when entering context."""
        self.connection = await self.pool.acquire()
        return self.connection

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release connection when exiting context."""
        if self.connection:
            await self.pool.release(self.connection)  # type: ignore[unreachable]


def get_performance_stats() -> dict[str, Any]:
    """Get comprehensive performance statistics."""
    return {
        "query_cache": _query_cache.get_stats(),
        "hyde_cache": _hyde_cache.get_stats(),
        "vector_cache": _vector_cache.get_stats(),
        "performance_monitor": _performance_monitor.get_performance_summary(),
    }


def clear_all_caches() -> None:
    """Clear all performance caches."""
    _query_cache.clear()
    _hyde_cache.clear()
    _vector_cache.clear()


async def warm_up_system() -> None:
    """Warm up the system for better performance."""
    logger = logging.getLogger(__name__)
    logger.info("Starting system warm-up...")

    # Pre-populate caches with common queries
    common_queries = [
        "How to implement authentication in Python?",
        "Best practices for API design",
        "Database optimization techniques",
        "Error handling strategies",
        "Performance monitoring setup",
    ]

    # This would normally call actual query processing
    # For now, just log the warm-up process
    for query in common_queries:
        cache_key = f"warmup:{hashlib.sha256(query.encode()).hexdigest()}"
        _query_cache.put(cache_key, {"warmed_up": True})

    logger.info("System warm-up completed")


# Performance optimization utilities
class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self) -> None:
        self.connection_pool = ConnectionPool(max_size=20)  # Use max_size instead of max_connections
        self.batcher = AsyncBatcher(batch_size=25, flush_interval=0.05)  # Use flush_interval
        self.logger = logging.getLogger(__name__)
        self.query_cache = _query_cache  # Reference to the global query cache
        self.hyde_cache = _hyde_cache  # Reference to the global HyDE cache
        self.vector_cache = _vector_cache  # Reference to the global vector cache
        self.monitor = _performance_monitor  # Reference to the global performance monitor

    def clear_all_caches(self) -> None:
        """Clear all caches through the optimizer."""
        clear_all_caches()

    async def optimize_query_processing(self, query: str) -> dict[str, Any]:
        """Optimize query processing with caching and batching."""
        metric = _performance_monitor.start_operation("optimize_query_processing")

        try:
            # Check cache first
            cache_key = f"optimized_query:{hashlib.sha256(query.encode()).hexdigest()}"
            cached_result = _query_cache.get(cache_key)

            if cached_result:
                _performance_monitor.complete_operation(metric, cache_hit=True)
                return cached_result  # type: ignore[no-any-return]

            # Process query with optimizations
            result = {"query": query, "optimized": True, "cache_miss": True, "timestamp": time.time()}

            # Cache the result
            _query_cache.put(cache_key, result)

            _performance_monitor.complete_operation(metric, cache_hit=False)
            return result

        except Exception:
            _performance_monitor.complete_operation(metric, error_occurred=True)
            raise

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        cache_stats = get_performance_stats()
        return {
            "cache_stats": cache_stats,
            "connection_pool_size": self.connection_pool.active_connections,
            "batcher_pending": len(self.batcher.pending_operations),
            # Also include individual cache stats at top level for backward compatibility
            **cache_stats,
        }

    async def warm_up_caches(self, warm_up_data: dict[str, dict[str, str]] | None = None) -> None:
        """Warm up caches with common queries and operations."""
        self.logger.info("Warming up performance caches...")

        if warm_up_data:
            self._warm_up_with_provided_data(warm_up_data)
        else:
            await self._warm_up_with_default_data()

    def _warm_up_with_provided_data(self, warm_up_data: dict[str, dict[str, str]]) -> None:
        """Warm up caches with provided data."""
        for cache_name, cache_data in warm_up_data.items():
            if cache_name == "query_cache" and hasattr(self, "query_cache"):
                for key, value in cache_data.items():
                    self.query_cache.put(key, value)
                    self.logger.debug("Warmed up query_cache with key: %s", key)
            elif cache_name == "hyde_cache" and hasattr(self, "hyde_cache"):
                for key, value in cache_data.items():
                    self.hyde_cache.put(key, value)
                    self.logger.debug("Warmed up hyde_cache with key: %s", key)
            elif cache_name == "vector_cache" and hasattr(self, "vector_cache"):
                for key, value in cache_data.items():
                    self.vector_cache.put(key, value)
                    self.logger.debug("Warmed up vector_cache with key: %s", key)

    async def _warm_up_with_default_data(self) -> None:
        """Warm up caches with default data."""
        # Pre-populate query cache with common queries
        common_queries = [
            "help",
            "create prompt",
            "analyze code",
            "documentation",
            "security best practices",
        ]

        for query in common_queries:
            try:
                result = await self.optimize_query_processing(query)
                # Ensure result is cached by accessing the cache
                if result:
                    self.logger.debug("Warmed up cache for query: %s", query)
            except Exception as e:
                self.logger.warning("Failed to warm up cache for query '%s': %s", query, e)

        self.logger.info("Cache warm-up completed")
