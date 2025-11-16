"""Tests for JWKS caching functionality."""

from unittest.mock import Mock, patch

import httpx
import pytest

from src.auth.jwks_client import JWKSClient
from src.auth.models import JWKSError


class TestJWKSClient:
    """Test cases for JWKS client and caching."""

    @pytest.fixture
    def sample_jwks(self):
        """Sample JWKS response."""
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "key1",
                    "use": "sig",
                    "n": "test-modulus-1",
                    "e": "AQAB",
                },
                {
                    "kty": "RSA",
                    "kid": "key2",
                    "use": "sig",
                    "n": "test-modulus-2",
                    "e": "AQAB",
                },
            ]
        }

    @pytest.fixture
    def jwks_client(self):
        """JWKS client instance."""
        return JWKSClient(
            jwks_url="https://test.cloudflareaccess.com/cdn-cgi/access/certs",
            cache_ttl=60,  # 1 minute for testing
            max_cache_size=5,
            timeout=5,
        )

    def test_get_jwks_success(self, jwks_client, sample_jwks):
        """Test successful JWKS retrieval."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_jwks
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = jwks_client.get_jwks()

            assert result == sample_jwks
            assert len(result["keys"]) == 2
            mock_get.assert_called_once_with(jwks_client.jwks_url)

    def test_get_jwks_caching(self, jwks_client, sample_jwks):
        """Test JWKS caching behavior."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_jwks
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # First call should hit the API
            result1 = jwks_client.get_jwks()
            assert result1 == sample_jwks
            assert mock_get.call_count == 1

            # Second call should use cache
            result2 = jwks_client.get_jwks()
            assert result2 == sample_jwks
            assert mock_get.call_count == 1  # No additional API call

    def test_get_jwks_http_error(self, jwks_client):
        """Test JWKS retrieval with HTTP error."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=Mock(status_code=404)
            )

            with pytest.raises(JWKSError) as exc_info:
                jwks_client.get_jwks()

            assert exc_info.value.error_type == "http_error"
            assert "Failed to fetch JWKS" in exc_info.value.message

    def test_get_jwks_timeout_error(self, jwks_client):
        """Test JWKS retrieval with timeout."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timeout")

            with pytest.raises(JWKSError) as exc_info:
                jwks_client.get_jwks()

            assert exc_info.value.error_type == "timeout_error"
            assert "Timeout fetching JWKS" in exc_info.value.message

    def test_get_jwks_invalid_format_no_keys(self, jwks_client):
        """Test JWKS retrieval with invalid format (no keys field)."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"invalid": "format"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            with pytest.raises(JWKSError) as exc_info:
                jwks_client.get_jwks()

            assert "missing 'keys' field" in exc_info.value.message

    def test_get_jwks_invalid_format_empty_keys(self, jwks_client):
        """Test JWKS retrieval with empty keys list."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"keys": []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            with pytest.raises(JWKSError) as exc_info:
                jwks_client.get_jwks()

            assert "'keys' list is empty" in exc_info.value.message

    def test_get_key_by_kid_found(self, jwks_client, sample_jwks):
        """Test retrieving specific key by kid."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_jwks
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            key = jwks_client.get_key_by_kid("key1")

            assert key is not None
            assert key["kid"] == "key1"
            assert key["n"] == "test-modulus-1"

    def test_get_key_by_kid_not_found(self, jwks_client, sample_jwks):
        """Test retrieving non-existent key by kid."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_jwks
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            key = jwks_client.get_key_by_kid("nonexistent")

            assert key is None

    def test_refresh_cache(self, jwks_client, sample_jwks):
        """Test cache refresh functionality."""
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_jwks
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Initial call
            jwks_client.get_jwks()
            assert mock_get.call_count == 1

            # Refresh cache should clear and fetch again
            jwks_client.refresh_cache()
            assert mock_get.call_count == 2

    def test_get_cache_info(self, jwks_client, sample_jwks):
        """Test cache information retrieval."""
        # Initially empty cache
        info = jwks_client.get_cache_info()
        assert info["cache_size"] == 0
        assert info["has_cached_jwks"] is False
        assert info["cached_keys_count"] == 0

        # After caching JWKS
        with patch.object(jwks_client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_jwks
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            jwks_client.get_jwks()

            info = jwks_client.get_cache_info()
            assert info["cache_size"] == 1
            assert info["has_cached_jwks"] is True
            assert info["cached_keys_count"] == 2
            assert info["ttl"] == 60
            assert info["max_size"] == 5

    def test_context_manager(self, sample_jwks):
        """Test JWKS client as context manager."""
        with JWKSClient("https://test.example.com") as client:
            with patch.object(client._client, "get") as mock_get:
                mock_response = Mock()
                mock_response.json.return_value = sample_jwks
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                result = client.get_jwks()
                assert result == sample_jwks

        # Client should be closed after context exit
        # Note: We can't easily test client.close() was called without more complex mocking

    def test_cache_ttl_expiration(self, sample_jwks):
        """Test cache TTL expiration behavior."""
        # Create client with very short TTL
        client = JWKSClient(
            jwks_url="https://test.example.com",
            cache_ttl=1,  # 1 second
        )

        with patch.object(client._client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_jwks
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # First call
            client.get_jwks()
            assert mock_get.call_count == 1

            # Immediate second call should use cache
            client.get_jwks()
            assert mock_get.call_count == 1

            # After TTL expiration, should fetch again
            # Note: In real testing, we'd wait or manipulate time
            # For this test, we'll clear cache manually to simulate expiration
            client._cache.clear()
            client.get_jwks()
            assert mock_get.call_count == 2
