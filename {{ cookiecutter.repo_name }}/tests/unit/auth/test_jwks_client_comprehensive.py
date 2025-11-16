"""
Comprehensive unit tests for JWKS client to achieve 80%+ coverage.

This module provides extensive test coverage for the JWKSClient class:
- JWKS fetching and caching with TTLCache
- Key retrieval by kid
- Error handling and timeout scenarios
- Cache management and statistics
- HTTP request handling
- Context manager usage
- Integration scenarios
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from src.auth.jwks_client import JWKSClient
from src.auth.models import JWKSError


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSClientInitialization:
    """Test JWKS client initialization and configuration."""

    def test_init_basic(self):
        """Test basic initialization."""
        client = JWKSClient(jwks_url="https://example.com/jwks")

        assert client.jwks_url == "https://example.com/jwks"
        assert client.timeout == 10  # Default 10 seconds
        assert hasattr(client, "_cache")
        assert hasattr(client, "_client")
        assert isinstance(client._client, httpx.Client)

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        client = JWKSClient(jwks_url="https://custom.com/certs", cache_ttl=7200, max_cache_size=5, timeout=30)

        assert client.jwks_url == "https://custom.com/certs"
        assert client.timeout == 30
        assert client._cache.maxsize == 5
        assert client._cache.ttl == 7200


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSClientHTTPRequests:
    """Test HTTP request handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = JWKSClient(jwks_url="https://example.com/jwks", timeout=10)

        self.mock_jwks_response = {
            "keys": [
                {"kty": "RSA", "kid": "key1", "use": "sig", "n": "test_n_value_1", "e": "AQAB"},
                {"kty": "RSA", "kid": "key2", "use": "sig", "n": "test_n_value_2", "e": "AQAB"},
            ],
        }

    @patch("httpx.Client.get")
    def test_get_jwks_success(self, mock_get):
        """Test successful JWKS fetching."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks_response
        mock_get.return_value = mock_response

        result = self.client.get_jwks()

        assert result == self.mock_jwks_response
        mock_get.assert_called_once_with("https://example.com/jwks")
        mock_response.raise_for_status.assert_called_once()

    @patch("httpx.Client.get")
    def test_get_jwks_http_error(self, mock_get):
        """Test JWKS fetching with HTTP error."""
        mock_get.side_effect = httpx.HTTPStatusError("404 Not Found", request=Mock(), response=Mock())

        with pytest.raises(JWKSError, match="Failed to fetch JWKS"):
            self.client.get_jwks()

    @patch("httpx.Client.get")
    def test_get_jwks_timeout_error(self, mock_get):
        """Test JWKS fetching with timeout error."""
        mock_get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(JWKSError, match="Timeout fetching JWKS"):
            self.client.get_jwks()

    @patch("httpx.Client.get")
    def test_get_jwks_connection_error(self, mock_get):
        """Test JWKS fetching with connection error."""
        mock_get.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(JWKSError, match="Failed to fetch JWKS"):
            self.client.get_jwks()

    @patch("httpx.Client.get")
    def test_get_jwks_unexpected_error(self, mock_get):
        """Test JWKS fetching with unexpected error."""
        mock_get.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(JWKSError, match="Unexpected error fetching JWKS"):
            self.client.get_jwks()


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSClientValidation:
    """Test JWKS response validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = JWKSClient(jwks_url="https://example.com/jwks")

    @patch("httpx.Client.get")
    def test_get_jwks_invalid_format_not_dict(self, mock_get):
        """Test JWKS fetching with non-dict response."""
        mock_response = Mock()
        mock_response.json.return_value = "not a dict"
        mock_get.return_value = mock_response

        with pytest.raises(JWKSError, match="Invalid JWKS format: missing 'keys' field"):
            self.client.get_jwks()

    @patch("httpx.Client.get")
    def test_get_jwks_missing_keys_field(self, mock_get):
        """Test JWKS fetching when response is missing keys field."""
        mock_response = Mock()
        mock_response.json.return_value = {"not_keys": []}
        mock_get.return_value = mock_response

        with pytest.raises(JWKSError, match="Invalid JWKS format: missing 'keys' field"):
            self.client.get_jwks()

    @patch("httpx.Client.get")
    def test_get_jwks_keys_not_list(self, mock_get):
        """Test JWKS fetching when keys field is not a list."""
        mock_response = Mock()
        mock_response.json.return_value = {"keys": "not a list"}
        mock_get.return_value = mock_response

        with pytest.raises(JWKSError, match="Invalid JWKS format: 'keys' must be a list"):
            self.client.get_jwks()

    @patch("httpx.Client.get")
    def test_get_jwks_empty_keys_list(self, mock_get):
        """Test JWKS fetching with empty keys list."""
        mock_response = Mock()
        mock_response.json.return_value = {"keys": []}
        mock_get.return_value = mock_response

        with pytest.raises(JWKSError, match="Invalid JWKS format: 'keys' list is empty"):
            self.client.get_jwks()


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSClientCaching:
    """Test caching functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = JWKSClient(jwks_url="https://example.com/jwks")

        self.mock_jwks_response = {
            "keys": [{"kty": "RSA", "kid": "key1", "use": "sig"}, {"kty": "RSA", "kid": "key2", "use": "sig"}],
        }

    @patch("httpx.Client.get")
    def test_get_jwks_caches_response(self, mock_get):
        """Test that JWKS response is cached."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks_response
        mock_get.return_value = mock_response

        # First call should fetch
        result1 = self.client.get_jwks()
        assert result1 == self.mock_jwks_response

        # Second call should use cache
        result2 = self.client.get_jwks()
        assert result2 == self.mock_jwks_response

        # Should only have made one HTTP request
        mock_get.assert_called_once()

    @patch("httpx.Client.get")
    def test_refresh_cache(self, mock_get):
        """Test forcing cache refresh."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks_response
        mock_get.return_value = mock_response

        # Initial call
        self.client.get_jwks()

        # Force refresh
        self.client.refresh_cache()

        # Should have made two HTTP requests
        assert mock_get.call_count == 2

    def test_get_cache_info_empty(self):
        """Test cache info when cache is empty."""
        info = self.client.get_cache_info()

        assert info["cache_size"] == 0
        assert info["has_cached_jwks"] is False
        assert info["cached_keys_count"] == 0
        assert "max_size" in info
        assert "ttl" in info

    @patch("httpx.Client.get")
    def test_get_cache_info_populated(self, mock_get):
        """Test cache info when cache is populated."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks_response
        mock_get.return_value = mock_response

        # Populate cache
        self.client.get_jwks()

        info = self.client.get_cache_info()

        assert info["cache_size"] == 1  # One JWKS cached
        assert info["has_cached_jwks"] is True
        assert info["cached_keys_count"] == 2  # Two keys in the JWKS
        assert info["max_size"] == 10  # Default max size
        assert info["ttl"] == 3600  # Default TTL


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSClientKeyRetrieval:
    """Test key retrieval by kid."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = JWKSClient(jwks_url="https://example.com/jwks")

        self.mock_jwks_response = {
            "keys": [
                {"kty": "RSA", "kid": "key1", "use": "sig", "n": "value1"},
                {"kty": "RSA", "kid": "key2", "use": "sig", "n": "value2"},
            ],
        }

    @patch("httpx.Client.get")
    def test_get_key_by_kid_found(self, mock_get):
        """Test successful key retrieval by kid."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks_response
        mock_get.return_value = mock_response

        result = self.client.get_key_by_kid("key1")

        assert result == {"kty": "RSA", "kid": "key1", "use": "sig", "n": "value1"}

    @patch("httpx.Client.get")
    def test_get_key_by_kid_not_found(self, mock_get):
        """Test key retrieval when kid is not found."""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks_response
        mock_get.return_value = mock_response

        result = self.client.get_key_by_kid("missing_key")

        assert result is None

    @patch("httpx.Client.get")
    def test_get_key_by_kid_key_without_kid(self, mock_get):
        """Test key retrieval with keys missing kid field."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "keys": [{"kty": "RSA", "use": "sig"}, {"kty": "RSA", "kid": "key2", "use": "sig"}],  # Missing kid
        }
        mock_get.return_value = mock_response

        result = self.client.get_key_by_kid("key2")

        assert result == {"kty": "RSA", "kid": "key2", "use": "sig"}

    @patch("httpx.Client.get")
    def test_get_key_by_kid_jwks_error_propagation(self, mock_get):
        """Test that JWKS fetch errors are propagated in key retrieval."""
        mock_get.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(JWKSError, match="Timeout fetching JWKS"):
            self.client.get_key_by_kid("any_key")


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSClientContextManager:
    """Test context manager functionality."""

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with JWKSClient(jwks_url="https://example.com/jwks") as client:
            assert isinstance(client, JWKSClient)
            assert client.jwks_url == "https://example.com/jwks"

    @patch("httpx.Client.close")
    def test_context_manager_closes_client(self, mock_close):
        """Test that context manager closes HTTP client."""
        client = JWKSClient(jwks_url="https://example.com/jwks")

        with client:
            pass  # Do nothing in context

        mock_close.assert_called_once()

    def test_close_method(self):
        """Test explicit close method."""
        client = JWKSClient(jwks_url="https://example.com/jwks")

        with patch.object(client._client, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()

    @patch("httpx.Client.get")
    def test_context_manager_with_operations(self, mock_get):
        """Test context manager with actual operations."""
        mock_response = Mock()
        mock_response.json.return_value = {"keys": [{"kty": "RSA", "kid": "key1", "use": "sig"}]}
        mock_get.return_value = mock_response

        with JWKSClient(jwks_url="https://example.com/jwks") as client:
            jwks = client.get_jwks()
            key = client.get_key_by_kid("key1")

            assert jwks["keys"][0]["kid"] == "key1"
            assert key["kid"] == "key1"


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSClientIntegration:
    """Test integration scenarios and workflows."""

    @patch("httpx.Client.get")
    def test_full_workflow_success(self, mock_get):
        """Test complete workflow from HTTP request to key retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "production_key",
                    "use": "sig",
                    "n": "production_n_value",
                    "e": "AQAB",
                    "alg": "RS256",
                },
            ],
        }
        mock_get.return_value = mock_response

        client = JWKSClient(jwks_url="https://example.com/jwks")

        # First call should fetch and cache
        key = client.get_key_by_kid("production_key")
        assert key["kid"] == "production_key"
        assert key["kty"] == "RSA"

        # Second call should use cache
        key2 = client.get_key_by_kid("production_key")
        assert key2 == key

        # Third call for different operation
        jwks = client.get_jwks()
        assert len(jwks["keys"]) == 1

        # Should only have made one HTTP request due to caching
        mock_get.assert_called_once()

    @patch("httpx.Client.get")
    def test_multiple_keys_management(self, mock_get):
        """Test managing multiple keys."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "keys": [
                {"kty": "RSA", "kid": "key1", "use": "sig"},
                {"kty": "RSA", "kid": "key2", "use": "sig"},
                {"kty": "RSA", "kid": "key3", "use": "sig"},
            ],
        }
        mock_get.return_value = mock_response

        client = JWKSClient(jwks_url="https://example.com/jwks")

        # Retrieve different keys
        key1 = client.get_key_by_kid("key1")
        key2 = client.get_key_by_kid("key2")
        key3 = client.get_key_by_kid("key3")
        missing = client.get_key_by_kid("key4")

        assert key1["kid"] == "key1"
        assert key2["kid"] == "key2"
        assert key3["kid"] == "key3"
        assert missing is None

        # Check cache info
        info = client.get_cache_info()
        assert info["cached_keys_count"] == 3

        # Should only have made one HTTP request (cached after first call)
        mock_get.assert_called_once()

    @patch("httpx.Client.get")
    def test_error_recovery_workflow(self, mock_get):
        """Test error recovery and retry workflow."""
        # First call fails
        mock_get.side_effect = httpx.TimeoutException("Timeout")

        client = JWKSClient(jwks_url="https://example.com/jwks")

        with pytest.raises(JWKSError):
            client.get_key_by_kid("any_key")

        # Second call succeeds (simulating recovery)
        mock_response = Mock()
        mock_response.json.return_value = {"keys": [{"kty": "RSA", "kid": "key1", "use": "sig"}]}
        mock_get.side_effect = None
        mock_get.return_value = mock_response

        key = client.get_key_by_kid("key1")
        assert key["kid"] == "key1"

    def test_custom_configuration_workflow(self):
        """Test client with custom configuration."""
        client = JWKSClient(
            jwks_url="https://custom.endpoint.com/certs",
            cache_ttl=1800,  # 30 minutes
            max_cache_size=5,
            timeout=20,  # 20 seconds
        )

        assert client.jwks_url == "https://custom.endpoint.com/certs"
        assert client.timeout == 20
        assert client._cache.maxsize == 5
        assert client._cache.ttl == 1800

        # Test cache info with custom values
        info = client.get_cache_info()
        assert info["max_size"] == 5
        assert info["ttl"] == 1800

    @patch("httpx.Client.get")
    def test_cache_refresh_workflow(self, mock_get):
        """Test cache refresh workflow."""
        # First response
        mock_response1 = Mock()
        mock_response1.json.return_value = {"keys": [{"kty": "RSA", "kid": "old_key", "use": "sig"}]}

        # Second response after refresh
        mock_response2 = Mock()
        mock_response2.json.return_value = {"keys": [{"kty": "RSA", "kid": "new_key", "use": "sig"}]}

        mock_get.side_effect = [mock_response1, mock_response2]

        client = JWKSClient(jwks_url="https://example.com/jwks")

        # Initial fetch
        key1 = client.get_key_by_kid("old_key")
        assert key1["kid"] == "old_key"

        # Refresh cache
        client.refresh_cache()

        # Verify new data
        key2 = client.get_key_by_kid("new_key")
        assert key2["kid"] == "new_key"

        # Old key should not be found
        old_key_after_refresh = client.get_key_by_kid("old_key")
        assert old_key_after_refresh is None

        # Should have made two HTTP requests
        assert mock_get.call_count == 2
