"""
Tests specifically for performance config module coverage.

This test file ensures that performance_config.py gets proper coverage measurement.
"""

from unittest.mock import Mock, patch

import pytest

from src.config.performance_config import (
    MIN_CACHE_TTL_SECONDS,
    MIN_VECTOR_CONNECTIONS,
    OPERATION_THRESHOLDS,
    PerformanceConfig,
    get_async_config,
    get_cache_config,
    get_connection_pool_config,
    get_monitoring_config,
    get_optimization_recommendations,
    get_performance_config,
    validate_performance_requirements,
)


class TestPerformanceConfigDataclass:
    """Test PerformanceConfig dataclass."""

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_performance_config_defaults(self):
        """Test default performance configuration values."""
        config = PerformanceConfig()

        assert config.max_response_time_ms == 2000
        assert config.target_response_time_ms == 1000
        assert config.cache_response_time_ms == 100
        assert config.query_cache_size == 500
        assert config.query_cache_ttl_seconds == 300
        assert config.hyde_cache_size == 200
        assert config.hyde_cache_ttl_seconds == 600
        assert config.vector_cache_size == 1000
        assert config.vector_cache_ttl_seconds == 180
        assert config.max_vector_connections == 20
        assert config.max_mcp_connections == 10
        assert config.connection_timeout_seconds == 30
        assert config.batch_size == 25
        assert config.max_batch_wait_time_ms == 50
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_reset_timeout_seconds == 60
        assert config.performance_monitoring_enabled is True
        assert config.slow_query_threshold_ms == 1500
        assert config.memory_usage_threshold_mb == 512
        assert config.max_concurrent_queries == 100
        assert config.semaphore_limit == 50
        assert config.vector_search_timeout_seconds == 10
        assert config.vector_batch_size == 100
        assert config.vector_max_retries == 3
        assert config.mcp_request_timeout_seconds == 15
        assert config.mcp_max_retries == 3
        assert config.mcp_backoff_multiplier == 1.5

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_performance_config_custom_values(self):
        """Test performance configuration with custom values."""
        config = PerformanceConfig(max_response_time_ms=3000, target_response_time_ms=1500, query_cache_size=200)

        assert config.max_response_time_ms == 3000
        assert config.target_response_time_ms == 1500
        assert config.query_cache_size == 200
        # Defaults should remain unchanged
        assert config.batch_size == 25


class TestPerformanceConfigConstants:
    """Test module constants."""

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_constants_values(self):
        """Test that constants have expected values."""
        assert MIN_CACHE_TTL_SECONDS == 60
        assert MIN_VECTOR_CONNECTIONS == 5

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_operation_thresholds(self):
        """Test operation thresholds dictionary."""
        assert OPERATION_THRESHOLDS["query_analysis"] == 500
        assert OPERATION_THRESHOLDS["hyde_processing"] == 800
        assert OPERATION_THRESHOLDS["vector_search"] == 400
        assert OPERATION_THRESHOLDS["agent_orchestration"] == 1000
        assert OPERATION_THRESHOLDS["response_synthesis"] == 300
        assert OPERATION_THRESHOLDS["end_to_end"] == 2000


class TestGetPerformanceConfig:
    """Test get_performance_config function."""

    @patch("src.config.performance_config.get_settings")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_performance_config_dev(self, mock_get_settings):
        """Test performance config for dev environment."""
        mock_settings = Mock()
        mock_settings.environment = "dev"
        mock_get_settings.return_value = mock_settings

        config = get_performance_config()

        assert config.max_response_time_ms == 3000
        assert config.query_cache_size == 100
        assert config.performance_monitoring_enabled is True
        assert config.max_concurrent_queries == 50

    @patch("src.config.performance_config.get_settings")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_performance_config_staging(self, mock_get_settings):
        """Test performance config for staging environment."""
        mock_settings = Mock()
        mock_settings.environment = "staging"
        mock_get_settings.return_value = mock_settings

        config = get_performance_config()

        assert config.max_response_time_ms == 2000
        assert config.query_cache_size == 300
        assert config.performance_monitoring_enabled is True
        assert config.max_concurrent_queries == 75

    @patch("src.config.performance_config.get_settings")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_performance_config_prod(self, mock_get_settings):
        """Test performance config for prod environment."""
        mock_settings = Mock()
        mock_settings.environment = "prod"
        mock_get_settings.return_value = mock_settings

        config = get_performance_config()

        assert config.max_response_time_ms == 2000
        assert config.target_response_time_ms == 800
        assert config.query_cache_size == 500
        assert config.hyde_cache_size == 300
        assert config.vector_cache_size == 1500
        assert config.max_concurrent_queries == 100
        assert config.performance_monitoring_enabled is True
        assert config.slow_query_threshold_ms == 1000

    @patch("src.config.performance_config.get_settings")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_performance_config_unknown_env(self, mock_get_settings):
        """Test performance config for unknown environment (should use defaults)."""
        mock_settings = Mock()
        mock_settings.environment = "unknown"
        mock_get_settings.return_value = mock_settings

        config = get_performance_config()

        # Should use default values
        assert config.max_response_time_ms == 2000
        assert config.query_cache_size == 500
        assert config.max_concurrent_queries == 100


class TestCacheConfig:
    """Test get_cache_config function."""

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_cache_config(self, mock_get_performance_config):
        """Test cache configuration retrieval."""
        mock_config = PerformanceConfig(
            query_cache_size=300,
            query_cache_ttl_seconds=400,
            hyde_cache_size=150,
            hyde_cache_ttl_seconds=700,
            vector_cache_size=800,
            vector_cache_ttl_seconds=200,
        )
        mock_get_performance_config.return_value = mock_config

        cache_config = get_cache_config()

        expected = {
            "query_cache": {
                "max_size": 300,
                "ttl_seconds": 400,
            },
            "hyde_cache": {
                "max_size": 150,
                "ttl_seconds": 700,
            },
            "vector_cache": {
                "max_size": 800,
                "ttl_seconds": 200,
            },
        }

        assert cache_config == expected


class TestConnectionPoolConfig:
    """Test get_connection_pool_config function."""

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_connection_pool_config(self, mock_get_performance_config):
        """Test connection pool configuration retrieval."""
        mock_config = PerformanceConfig(
            max_vector_connections=25,
            connection_timeout_seconds=45,
            max_mcp_connections=15,
            mcp_request_timeout_seconds=20,
        )
        mock_get_performance_config.return_value = mock_config

        pool_config = get_connection_pool_config()

        expected = {
            "vector_store": {
                "max_connections": 25,
                "timeout_seconds": 45,
            },
            "mcp_client": {
                "max_connections": 15,
                "timeout_seconds": 20,
            },
        }

        assert pool_config == expected


class TestAsyncConfig:
    """Test get_async_config function."""

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_async_config(self, mock_get_performance_config):
        """Test async configuration retrieval."""
        mock_config = PerformanceConfig(
            max_concurrent_queries=75,
            semaphore_limit=40,
            batch_size=30,
            max_batch_wait_time_ms=60,
        )
        mock_get_performance_config.return_value = mock_config

        async_config = get_async_config()

        expected = {
            "max_concurrent_queries": 75,
            "semaphore_limit": 40,
            "batch_size": 30,
            "max_batch_wait_time_ms": 60,
        }

        assert async_config == expected


class TestMonitoringConfig:
    """Test get_monitoring_config function."""

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_monitoring_config(self, mock_get_performance_config):
        """Test monitoring configuration retrieval."""
        mock_config = PerformanceConfig(
            performance_monitoring_enabled=False,
            slow_query_threshold_ms=1200,
            memory_usage_threshold_mb=256,
            max_response_time_ms=1800,
            target_response_time_ms=900,
        )
        mock_get_performance_config.return_value = mock_config

        monitoring_config = get_monitoring_config()

        expected = {
            "enabled": False,
            "slow_query_threshold_ms": 1200,
            "memory_usage_threshold_mb": 256,
            "max_response_time_ms": 1800,
            "target_response_time_ms": 900,
        }

        assert monitoring_config == expected


class TestValidatePerformanceRequirements:
    """Test validate_performance_requirements function."""

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_validate_performance_requirements_valid(self, mock_get_performance_config):
        """Test validation with valid performance requirements."""
        mock_config = PerformanceConfig(
            target_response_time_ms=1000,
            max_response_time_ms=2000,
            query_cache_ttl_seconds=300,
            max_vector_connections=10,
        )
        mock_get_performance_config.return_value = mock_config

        assert validate_performance_requirements() is True

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_validate_performance_requirements_target_too_high(self, mock_get_performance_config):
        """Test validation with target response time >= max response time."""
        mock_config = PerformanceConfig(
            target_response_time_ms=2000,
            max_response_time_ms=2000,
            query_cache_ttl_seconds=300,
            max_vector_connections=10,
        )
        mock_get_performance_config.return_value = mock_config

        assert validate_performance_requirements() is False

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_validate_performance_requirements_cache_ttl_too_low(self, mock_get_performance_config):
        """Test validation with cache TTL below minimum."""
        mock_config = PerformanceConfig(
            target_response_time_ms=1000,
            max_response_time_ms=2000,
            query_cache_ttl_seconds=30,  # Below MIN_CACHE_TTL_SECONDS (60)
            max_vector_connections=10,
        )
        mock_get_performance_config.return_value = mock_config

        assert validate_performance_requirements() is False

    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_validate_performance_requirements_connections_too_low(self, mock_get_performance_config):
        """Test validation with vector connections below minimum."""
        mock_config = PerformanceConfig(
            target_response_time_ms=1000,
            max_response_time_ms=2000,
            query_cache_ttl_seconds=300,
            max_vector_connections=3,  # Below MIN_VECTOR_CONNECTIONS (5)
        )
        mock_get_performance_config.return_value = mock_config

        assert validate_performance_requirements() is False


class TestGetOptimizationRecommendations:
    """Test get_optimization_recommendations function."""

    @patch("src.config.performance_config.get_settings")
    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_optimization_recommendations_dev(self, mock_get_performance_config, mock_get_settings):
        """Test optimization recommendations for dev environment."""
        mock_settings = Mock()
        mock_settings.environment = "dev"
        mock_get_settings.return_value = mock_settings

        mock_config = PerformanceConfig(
            target_response_time_ms=1000,
            max_response_time_ms=2000,
            batch_size=25,
            max_concurrent_queries=50,
        )
        mock_get_performance_config.return_value = mock_config

        recommendations = get_optimization_recommendations()

        assert "caching" in recommendations
        assert "monitoring" in recommendations
        assert "connections" in recommendations
        assert "response_time" in recommendations
        assert "batching" in recommendations
        assert "concurrency" in recommendations

        assert "Enable all caches for realistic testing" in recommendations["caching"]
        assert "Use detailed performance monitoring" in recommendations["monitoring"]
        assert "Use smaller connection pools" in recommendations["connections"]
        assert "<1000ms" in recommendations["response_time"]
        assert "size 25" in recommendations["batching"]
        assert "50 concurrent" in recommendations["concurrency"]

    @patch("src.config.performance_config.get_settings")
    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_optimization_recommendations_prod(self, mock_get_performance_config, mock_get_settings):
        """Test optimization recommendations for prod environment."""
        mock_settings = Mock()
        mock_settings.environment = "prod"
        mock_get_settings.return_value = mock_settings

        mock_config = PerformanceConfig(
            target_response_time_ms=800,
            max_response_time_ms=2000,
            batch_size=25,
            max_concurrent_queries=100,
        )
        mock_get_performance_config.return_value = mock_config

        recommendations = get_optimization_recommendations()

        assert "caching" in recommendations
        assert "monitoring" in recommendations
        assert "connections" in recommendations
        assert "optimization" in recommendations
        assert "response_time" in recommendations
        assert "batching" in recommendations
        assert "concurrency" in recommendations

        assert "Maximize cache sizes and TTL" in recommendations["caching"]
        assert "Enable alerts for performance degradation" in recommendations["monitoring"]
        assert "Use maximum connection pooling" in recommendations["connections"]
        assert "Enable all performance optimizations" in recommendations["optimization"]
        assert "<800ms" in recommendations["response_time"]
        assert "size 25" in recommendations["batching"]
        assert "100 concurrent" in recommendations["concurrency"]

    @patch("src.config.performance_config.get_settings")
    @patch("src.config.performance_config.get_performance_config")
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.stress
    def test_get_optimization_recommendations_staging(self, mock_get_performance_config, mock_get_settings):
        """Test optimization recommendations for staging environment."""
        mock_settings = Mock()
        mock_settings.environment = "staging"
        mock_get_settings.return_value = mock_settings

        mock_config = PerformanceConfig(
            target_response_time_ms=1000,
            max_response_time_ms=2000,
            batch_size=25,
            max_concurrent_queries=75,
        )
        mock_get_performance_config.return_value = mock_config

        recommendations = get_optimization_recommendations()

        # Should have general recommendations but not dev/prod specific ones
        assert "response_time" in recommendations
        assert "batching" in recommendations
        assert "concurrency" in recommendations
        assert "caching" not in recommendations  # No dev/prod specific caching recommendations
        assert "optimization" not in recommendations  # No prod-specific optimization recommendations
