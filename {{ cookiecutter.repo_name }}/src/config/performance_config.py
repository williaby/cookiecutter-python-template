"""
Performance configuration for PromptCraft-Hybrid.

This module defines performance-related configuration settings and constants
to ensure optimal system performance and meet the <2s response time requirement.
"""

from dataclasses import dataclass
from typing import Any

from src.config.settings import get_settings

# Performance validation constants
MIN_CACHE_TTL_SECONDS = 60  # Minimum cache TTL in seconds
MIN_VECTOR_CONNECTIONS = 5  # Minimum number of vector connections


@dataclass
class PerformanceConfig:
    """Performance configuration settings."""

    # Response time requirements
    max_response_time_ms: int = 2000  # 2 seconds
    target_response_time_ms: int = 1000  # 1 second target
    cache_response_time_ms: int = 100  # 100ms for cached responses

    # Cache configuration
    query_cache_size: int = 500
    query_cache_ttl_seconds: int = 300  # 5 minutes
    hyde_cache_size: int = 200
    hyde_cache_ttl_seconds: int = 600  # 10 minutes
    vector_cache_size: int = 1000
    vector_cache_ttl_seconds: int = 180  # 3 minutes

    # Connection pooling
    max_vector_connections: int = 20
    max_mcp_connections: int = 10
    connection_timeout_seconds: int = 30

    # Batching configuration
    batch_size: int = 25
    max_batch_wait_time_ms: int = 50  # 50ms max wait for batching

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_timeout_seconds: int = 60

    # Monitoring and alerting
    performance_monitoring_enabled: bool = True
    slow_query_threshold_ms: int = 1500  # Alert if > 1.5s
    memory_usage_threshold_mb: int = 512  # Alert if > 512MB

    # Async optimization
    max_concurrent_queries: int = 100
    semaphore_limit: int = 50

    # Vector store optimization
    vector_search_timeout_seconds: int = 10
    vector_batch_size: int = 100
    vector_max_retries: int = 3

    # MCP optimization
    mcp_request_timeout_seconds: int = 15
    mcp_max_retries: int = 3
    mcp_backoff_multiplier: float = 1.5


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration based on environment."""
    settings = get_settings()

    # Base configuration
    config = PerformanceConfig()

    # Environment-specific optimizations
    if settings.environment == "dev":
        # Development optimizations
        config.max_response_time_ms = 3000  # Allow slightly longer for dev
        config.query_cache_size = 100  # Smaller cache for dev
        config.performance_monitoring_enabled = True
        config.max_concurrent_queries = 50

    elif settings.environment == "staging":
        # Staging optimizations (production-like)
        config.max_response_time_ms = 2000
        config.query_cache_size = 300
        config.performance_monitoring_enabled = True
        config.max_concurrent_queries = 75

    elif settings.environment == "prod":
        # Production optimizations (strict requirements)
        config.max_response_time_ms = 2000
        config.target_response_time_ms = 800  # Stricter target
        config.query_cache_size = 500
        config.hyde_cache_size = 300
        config.vector_cache_size = 1500
        config.max_concurrent_queries = 100
        config.performance_monitoring_enabled = True
        config.slow_query_threshold_ms = 1000  # Stricter alert threshold

    return config


def get_cache_config() -> dict[str, Any]:
    """Get cache-specific configuration."""
    config = get_performance_config()

    return {
        "query_cache": {
            "max_size": config.query_cache_size,
            "ttl_seconds": config.query_cache_ttl_seconds,
        },
        "hyde_cache": {
            "max_size": config.hyde_cache_size,
            "ttl_seconds": config.hyde_cache_ttl_seconds,
        },
        "vector_cache": {
            "max_size": config.vector_cache_size,
            "ttl_seconds": config.vector_cache_ttl_seconds,
        },
    }


def get_connection_pool_config() -> dict[str, Any]:
    """Get connection pool configuration."""
    config = get_performance_config()

    return {
        "vector_store": {
            "max_connections": config.max_vector_connections,
            "timeout_seconds": config.connection_timeout_seconds,
        },
        "mcp_client": {
            "max_connections": config.max_mcp_connections,
            "timeout_seconds": config.mcp_request_timeout_seconds,
        },
    }


def get_async_config() -> dict[str, Any]:
    """Get async optimization configuration."""
    config = get_performance_config()

    return {
        "max_concurrent_queries": config.max_concurrent_queries,
        "semaphore_limit": config.semaphore_limit,
        "batch_size": config.batch_size,
        "max_batch_wait_time_ms": config.max_batch_wait_time_ms,
    }


def get_monitoring_config() -> dict[str, Any]:
    """Get performance monitoring configuration."""
    config = get_performance_config()

    return {
        "enabled": config.performance_monitoring_enabled,
        "slow_query_threshold_ms": config.slow_query_threshold_ms,
        "memory_usage_threshold_mb": config.memory_usage_threshold_mb,
        "max_response_time_ms": config.max_response_time_ms,
        "target_response_time_ms": config.target_response_time_ms,
    }


# Performance thresholds for different operation types
OPERATION_THRESHOLDS = {
    "query_analysis": 500,  # 500ms
    "hyde_processing": 800,  # 800ms
    "vector_search": 400,  # 400ms
    "agent_orchestration": 1000,  # 1s
    "response_synthesis": 300,  # 300ms
    "end_to_end": 2000,  # 2s
}


def validate_performance_requirements() -> bool:
    """Validate that performance requirements are achievable."""
    config = get_performance_config()

    # Check if target response time is reasonable
    if config.target_response_time_ms >= config.max_response_time_ms:
        return False

    # Check if cache TTL is reasonable
    if config.query_cache_ttl_seconds < MIN_CACHE_TTL_SECONDS:  # Less than 1 minute
        return False

    # Check if connection pool sizes are reasonable
    return not config.max_vector_connections < MIN_VECTOR_CONNECTIONS


def get_optimization_recommendations() -> dict[str, str]:
    """Get performance optimization recommendations."""
    config = get_performance_config()
    settings = get_settings()

    recommendations = {}

    # Environment-specific recommendations
    if settings.environment == "dev":
        recommendations["caching"] = "Enable all caches for realistic testing"
        recommendations["monitoring"] = "Use detailed performance monitoring"
        recommendations["connections"] = "Use smaller connection pools"

    elif settings.environment == "prod":
        recommendations["caching"] = "Maximize cache sizes and TTL"
        recommendations["monitoring"] = "Enable alerts for performance degradation"
        recommendations["connections"] = "Use maximum connection pooling"
        recommendations["optimization"] = "Enable all performance optimizations"

    # General recommendations
    recommendations["response_time"] = (
        f"Target <{config.target_response_time_ms}ms, max <{config.max_response_time_ms}ms"
    )
    recommendations["batching"] = f"Use batching with size {config.batch_size}"
    recommendations["concurrency"] = f"Limit to {config.max_concurrent_queries} concurrent queries"

    return recommendations
