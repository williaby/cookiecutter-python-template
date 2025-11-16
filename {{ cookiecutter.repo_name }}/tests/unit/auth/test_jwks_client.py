"""
Unit tests for JWKSClient class.

This module provides comprehensive test coverage for the JWKSClient class,
testing JWKS retrieval, caching, error handling, and HTTP interactions.
Uses proper pytest markers for codecov integration per codecov.yml auth component.
"""

from unittest.mock import Mock, patch

import httpx
import pytest
from cachetools import TTLCache

from src.auth.jwks_client import JWKSClient
from src.auth.models import JWKSError


@pytest.mark.auth
class TestJWKSClientInitialization:
    """Test cases for JWKSClient initialization."""

    def test_init_default_values(self):
        """Test initialization with default configuration values."""
        jwks_url = "https://test.cloudflareaccess.com/cdn-cgi/access/certs"
        client = JWKSClient(jwks_url)

        assert client.jwks_url == jwks_url
        assert client.timeout == 10
        assert isinstance(client._cache, TTLCache)
        assert client._cache.maxsize == 10
        assert client._cache.ttl == 3600
        assert isinstance(client._client, httpx.Client)

    def test_init_custom_values(self):
        """Test initialization with custom configuration values."""
        jwks_url = "https://custom.example.com/jwks"
        client = JWKSClient(jwks_url=jwks_url, cache_ttl=7200, max_cache_size=5, timeout=30)

        assert client.jwks_url == jwks_url
        assert client.timeout == 30
        assert client._cache.maxsize == 5
        assert client._cache.ttl == 7200

    def test_init_creates_http_client(self):
        """Test that initialization creates HTTP client with correct timeout."""
        client = JWKSClient("https://example.com/jwks", timeout=25)

        assert isinstance(client._client, httpx.Client)
        # httpx.Client.timeout returns a Timeout object with individual timeout values
        assert client._client.timeout.connect == 25
        assert client._client.timeout.read == 25

    def test_init_creates_cache_with_correct_settings(self):
        """Test that initialization creates cache with correct settings."""
        client = JWKSClient("https://example.com/jwks", cache_ttl=1800, max_cache_size=3)

        assert client._cache.maxsize == 3
        assert client._cache.ttl == 1800
        assert len(client._cache) == 0  # Should start empty


@pytest.mark.auth
class TestJWKSClientGetJWKS:
    """Test cases for JWKS retrieval functionality."""

    def test_get_jwks_success_first_time(self):
        """Test successful JWKS retrieval on first call."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {
            "keys": [{"kid": "key1", "kty": "RSA", "alg": "RS256"}, {"kid": "key2", "kty": "RSA", "alg": "RS256"}],
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)
            result = client.get_jwks()

            assert result == mock_jwks
            mock_client.get.assert_called_once_with(jwks_url)
            mock_response.raise_for_status.assert_called_once()

    def test_get_jwks_from_cache(self):
        """Test JWKS retrieval from cache on subsequent calls."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            # First call should fetch from endpoint
            result1 = client.get_jwks()
            assert result1 == mock_jwks

            # Second call should use cache
            result2 = client.get_jwks()
            assert result2 == mock_jwks

            # HTTP client should only be called once
            assert mock_client.get.call_count == 1

    def test_get_jwks_invalid_format_missing_keys(self):
        """Test JWKS validation fails when 'keys' field is missing."""
        jwks_url = "https://example.com/jwks"
        invalid_jwks = {"invalid": "format"}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = invalid_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Invalid JWKS format: missing 'keys' field" in str(exc_info.value)
            assert exc_info.value.error_type == "jwks_error"

    def test_get_jwks_invalid_format_keys_not_list(self):
        """Test JWKS validation fails when 'keys' is not a list."""
        jwks_url = "https://example.com/jwks"
        invalid_jwks = {"keys": "not-a-list"}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = invalid_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Invalid JWKS format: 'keys' must be a list" in str(exc_info.value)

    def test_get_jwks_invalid_format_empty_keys(self):
        """Test JWKS validation fails when 'keys' list is empty."""
        jwks_url = "https://example.com/jwks"
        invalid_jwks = {"keys": []}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = invalid_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Invalid JWKS format: 'keys' list is empty" in str(exc_info.value)

    def test_get_jwks_timeout_error(self):
        """Test JWKS retrieval handles timeout errors."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Timeout occurred")

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Timeout fetching JWKS" in str(exc_info.value)
            assert exc_info.value.error_type == "timeout_error"

    def test_get_jwks_http_error(self):
        """Test JWKS retrieval handles HTTP errors."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = httpx.HTTPError("HTTP error occurred")

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Failed to fetch JWKS" in str(exc_info.value)
            assert exc_info.value.error_type == "http_error"

    def test_get_jwks_http_status_error(self):
        """Test JWKS retrieval handles HTTP status errors."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=Mock(),
                response=Mock(),
            )
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Failed to fetch JWKS" in str(exc_info.value)
            assert exc_info.value.error_type == "http_error"

    def test_get_jwks_json_decode_error(self):
        """Test JWKS retrieval handles JSON decode errors."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Unexpected error fetching JWKS" in str(exc_info.value)
            assert exc_info.value.error_type == "unknown_error"

    def test_get_jwks_unexpected_error(self):
        """Test JWKS retrieval handles unexpected errors."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = RuntimeError("Unexpected runtime error")

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_jwks()

            assert "Unexpected error fetching JWKS" in str(exc_info.value)
            assert exc_info.value.error_type == "unknown_error"


@pytest.mark.auth
class TestJWKSClientGetKeyByKid:
    """Test cases for key retrieval by key ID."""

    def test_get_key_by_kid_found(self):
        """Test successful key retrieval by key ID."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {
            "keys": [
                {"kid": "key1", "kty": "RSA", "alg": "RS256", "use": "sig"},
                {"kid": "key2", "kty": "RSA", "alg": "RS256", "use": "sig"},
                {"kid": "key3", "kty": "RSA", "alg": "RS256", "use": "sig"},
            ],
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)
            result = client.get_key_by_kid("key2")

            expected_key = {"kid": "key2", "kty": "RSA", "alg": "RS256", "use": "sig"}
            assert result == expected_key

    def test_get_key_by_kid_not_found(self):
        """Test key retrieval when key ID is not found."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {
            "keys": [{"kid": "key1", "kty": "RSA", "alg": "RS256"}, {"kid": "key2", "kty": "RSA", "alg": "RS256"}],
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)
            result = client.get_key_by_kid("nonexistent")

            assert result is None

    def test_get_key_by_kid_empty_keys(self):
        """Test key retrieval with empty keys list."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": []}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock the get_jwks method to avoid validation error for empty keys
            client = JWKSClient(jwks_url)
            with patch.object(client, "get_jwks", return_value=mock_jwks):
                result = client.get_key_by_kid("any_key")
                assert result is None

    def test_get_key_by_kid_key_without_kid(self):
        """Test key retrieval with keys missing 'kid' field."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {
            "keys": [{"kty": "RSA", "alg": "RS256"}, {"kid": "key2", "kty": "RSA", "alg": "RS256"}],  # Missing 'kid'
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            # Should find key2
            result = client.get_key_by_kid("key2")
            assert result == {"kid": "key2", "kty": "RSA", "alg": "RS256"}

            # Should not find key without kid
            result = client.get_key_by_kid("missing")
            assert result is None

    def test_get_key_by_kid_uses_cache(self):
        """Test that get_key_by_kid uses cached JWKS."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA", "alg": "RS256"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            # First call should fetch and cache
            result1 = client.get_key_by_kid("key1")
            assert result1 is not None

            # Second call should use cache
            result2 = client.get_key_by_kid("key1")
            assert result2 == result1

            # HTTP client should only be called once
            assert mock_client.get.call_count == 1

    def test_get_key_by_kid_propagates_jwks_error(self):
        """Test that get_key_by_kid propagates JWKS retrieval errors."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError) as exc_info:
                client.get_key_by_kid("any_key")

            assert "Timeout fetching JWKS" in str(exc_info.value)


@pytest.mark.auth
class TestJWKSClientCacheManagement:
    """Test cases for cache management functionality."""

    def test_refresh_cache_clears_and_refetches(self):
        """Test that refresh_cache clears cache and fetches fresh JWKS."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url)

            # Initial fetch
            client.get_jwks()
            assert mock_client.get.call_count == 1

            # Refresh cache should clear and refetch
            client.refresh_cache()
            assert mock_client.get.call_count == 2

    def test_refresh_cache_propagates_errors(self):
        """Test that refresh_cache propagates JWKS retrieval errors."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")

            client = JWKSClient(jwks_url)

            with pytest.raises(JWKSError):
                client.refresh_cache()

    def test_get_cache_info_empty_cache(self):
        """Test cache info when cache is empty."""
        client = JWKSClient("https://example.com/jwks", cache_ttl=1800, max_cache_size=5)

        info = client.get_cache_info()

        assert info["cache_size"] == 0
        assert info["max_size"] == 5
        assert info["ttl"] == 1800
        assert info["has_cached_jwks"] is False
        assert info["cached_keys_count"] == 0

    def test_get_cache_info_with_cached_jwks(self):
        """Test cache info when JWKS is cached."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {
            "keys": [{"kid": "key1", "kty": "RSA"}, {"kid": "key2", "kty": "RSA"}, {"kid": "key3", "kty": "RSA"}],
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            client = JWKSClient(jwks_url, cache_ttl=7200)
            client.get_jwks()  # This will cache the JWKS

            info = client.get_cache_info()

            assert info["cache_size"] == 1
            assert info["ttl"] == 7200
            assert info["has_cached_jwks"] is True
            assert info["cached_keys_count"] == 3

    def test_cache_ttl_behavior(self):
        """Test that cache respects TTL settings."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            # Create client with very short TTL for testing
            client = JWKSClient(jwks_url, cache_ttl=1)

            # Initial fetch
            client.get_jwks()
            assert mock_client.get.call_count == 1

            # Mock the cache to simulate TTL expiration
            client._cache.clear()

            # Next call should fetch again
            client.get_jwks()
            assert mock_client.get.call_count == 2


@pytest.mark.auth
class TestJWKSClientContextManager:
    """Test cases for context manager functionality."""

    def test_context_manager_enter_returns_self(self):
        """Test that context manager __enter__ returns self."""
        client = JWKSClient("https://example.com/jwks")

        with client as context_client:
            assert context_client is client

    def test_context_manager_exit_closes_client(self):
        """Test that context manager __exit__ closes HTTP client."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = JWKSClient("https://example.com/jwks")

            with client:
                pass  # Do nothing in context

            mock_client.close.assert_called_once()

    def test_context_manager_exit_with_exception(self):
        """Test that context manager __exit__ handles exceptions properly."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = JWKSClient("https://example.com/jwks")

            try:
                with client:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            mock_client.close.assert_called_once()

    def test_close_method(self):
        """Test that close method closes HTTP client."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = JWKSClient("https://example.com/jwks")
            client.close()

            mock_client.close.assert_called_once()


@pytest.mark.auth
class TestJWKSClientLogging:
    """Test cases for logging functionality."""

    def test_logging_jwks_fetch_success(self):
        """Test that successful JWKS fetch is logged."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            with patch("src.auth.jwks_client.logger") as mock_logger:
                client = JWKSClient(jwks_url)
                client.get_jwks()

                # Check info logs
                mock_logger.info.assert_any_call(f"Fetching JWKS from {jwks_url}")
                mock_logger.info.assert_any_call("Successfully cached JWKS with 1 keys")

    def test_logging_jwks_cache_hit(self):
        """Test that cache hit is logged."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            with patch("src.auth.jwks_client.logger") as mock_logger:
                client = JWKSClient(jwks_url)

                # First call to populate cache
                client.get_jwks()
                mock_logger.reset_mock()

                # Second call should hit cache
                client.get_jwks()
                mock_logger.debug.assert_called_with("Retrieved JWKS from cache")

    def test_logging_key_found(self):
        """Test that finding a key by kid is logged."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            with patch("src.auth.jwks_client.logger") as mock_logger:
                client = JWKSClient(jwks_url)
                client.get_key_by_kid("key1")

                mock_logger.debug.assert_any_call("Found key with kid: key1")

    def test_logging_key_not_found(self):
        """Test that not finding a key by kid is logged."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            with patch("src.auth.jwks_client.logger") as mock_logger:
                client = JWKSClient(jwks_url)
                client.get_key_by_kid("nonexistent")

                mock_logger.warning.assert_called_with("Key with kid 'nonexistent' not found in JWKS")

    def test_logging_refresh_cache(self):
        """Test that cache refresh is logged."""
        jwks_url = "https://example.com/jwks"
        mock_jwks = {"keys": [{"kid": "key1", "kty": "RSA"}]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            with patch("src.auth.jwks_client.logger") as mock_logger:
                client = JWKSClient(jwks_url)
                client.refresh_cache()

                mock_logger.info.assert_any_call("Forcing JWKS cache refresh")

    def test_logging_errors(self):
        """Test that errors are logged properly."""
        jwks_url = "https://example.com/jwks"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Timeout occurred")

            with patch("src.auth.jwks_client.logger") as mock_logger:
                client = JWKSClient(jwks_url)

                with pytest.raises(JWKSError):
                    client.get_jwks()

                mock_logger.error.assert_called_with("Timeout fetching JWKS: Timeout occurred")


@pytest.mark.auth
class TestJWKSClientIntegration:
    """Integration test cases for complete JWKS workflows."""

    def test_complete_jwks_workflow(self):
        """Test complete workflow of fetching, caching, and retrieving keys."""
        jwks_url = "https://test.cloudflareaccess.com/cdn-cgi/access/certs"
        mock_jwks = {
            "keys": [
                {
                    "kid": "signing-key-1",
                    "kty": "RSA",
                    "alg": "RS256",
                    "use": "sig",
                    "n": "example_modulus",
                    "e": "AQAB",
                },
                {
                    "kid": "signing-key-2",
                    "kty": "RSA",
                    "alg": "RS256",
                    "use": "sig",
                    "n": "another_modulus",
                    "e": "AQAB",
                },
            ],
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_client.get.return_value = mock_response

            with JWKSClient(jwks_url, cache_ttl=3600, max_cache_size=10) as client:
                # Initial JWKS fetch
                jwks = client.get_jwks()
                assert len(jwks["keys"]) == 2

                # Retrieve specific keys
                key1 = client.get_key_by_kid("signing-key-1")
                assert key1 is not None
                assert key1["kid"] == "signing-key-1"
                assert key1["n"] == "example_modulus"

                key2 = client.get_key_by_kid("signing-key-2")
                assert key2 is not None
                assert key2["kid"] == "signing-key-2"

                # Non-existent key
                key3 = client.get_key_by_kid("nonexistent")
                assert key3 is None

                # Check cache info
                cache_info = client.get_cache_info()
                assert cache_info["has_cached_jwks"] is True
                assert cache_info["cached_keys_count"] == 2

                # Refresh cache
                client.refresh_cache()

                # Should still work after refresh
                refreshed_key = client.get_key_by_kid("signing-key-1")
                assert refreshed_key == key1

            # Context manager should close client
            mock_client.close.assert_called_once()
