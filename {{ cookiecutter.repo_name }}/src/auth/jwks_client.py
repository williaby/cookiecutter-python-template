"""JWKS (JSON Web Key Set) client for Cloudflare Access authentication.

This module provides secure JWKS retrieval and caching with TTL support.
It handles key rotation and network failures gracefully.
"""

import logging
from typing import Any

import httpx
from cachetools import TTLCache

from .models import JWKSError

logger = logging.getLogger(__name__)


class JWKSClient:
    """Client for retrieving and caching JWKS from Cloudflare Access."""

    def __init__(
        self,
        jwks_url: str,
        cache_ttl: int = 3600,  # 1 hour default TTL
        max_cache_size: int = 10,
        timeout: int = 10,
    ) -> None:
        """Initialize JWKS client.

        Args:
            jwks_url: Cloudflare JWKS endpoint URL
            cache_ttl: Cache TTL in seconds (default 1 hour)
            max_cache_size: Maximum number of cached JWKS
            timeout: HTTP request timeout in seconds
        """
        self.jwks_url = jwks_url
        self.timeout = timeout
        self._cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=max_cache_size, ttl=cache_ttl)
        self._client = httpx.Client(timeout=timeout)

    def get_jwks(self) -> dict[str, Any]:
        """Retrieve JWKS with caching.

        Returns:
            JWKS dictionary with keys

        Raises:
            JWKSError: If JWKS retrieval fails
        """
        # Check cache first
        cached_jwks = self._cache.get("jwks")
        if cached_jwks is not None:
            logger.debug("Retrieved JWKS from cache")
            return dict(cached_jwks)  # Cast to dict to satisfy type checker

        # Fetch from endpoint
        try:
            logger.info(f"Fetching JWKS from {self.jwks_url}")
            response = self._client.get(self.jwks_url)
            response.raise_for_status()

            jwks = response.json()

            # Validate JWKS structure
            if not isinstance(jwks, dict) or "keys" not in jwks:
                raise JWKSError("Invalid JWKS format: missing 'keys' field", "jwks_error")

            if not isinstance(jwks["keys"], list):
                raise JWKSError("Invalid JWKS format: 'keys' must be a list", "jwks_error")

            if not jwks["keys"]:
                raise JWKSError("Invalid JWKS format: 'keys' list is empty", "jwks_error")

            # Cache the JWKS
            self._cache["jwks"] = jwks
            logger.info(f"Successfully cached JWKS with {len(jwks['keys'])} keys")

            return jwks

        except JWKSError:
            # Re-raise JWKSError without modification
            raise

        except httpx.TimeoutException as e:
            logger.error(f"Timeout fetching JWKS: {e}")
            raise JWKSError(f"Timeout fetching JWKS: {e}", "timeout_error") from e

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching JWKS: {e}")
            raise JWKSError(f"Failed to fetch JWKS: {e}", "http_error") from e

        except Exception as e:
            logger.error(f"Unexpected error fetching JWKS: {e}")
            raise JWKSError(f"Unexpected error fetching JWKS: {e}", "unknown_error") from e

    def get_key_by_kid(self, kid: str) -> dict[str, Any] | None:
        """Get specific key by key ID.

        Args:
            kid: Key ID to search for

        Returns:
            Key dictionary if found, None otherwise

        Raises:
            JWKSError: If JWKS retrieval fails
        """
        jwks = self.get_jwks()

        for key in jwks["keys"]:
            if key.get("kid") == kid:
                logger.debug(f"Found key with kid: {kid}")
                return dict(key)  # Cast to dict to satisfy type checker

        logger.warning(f"Key with kid '{kid}' not found in JWKS")
        return None

    def refresh_cache(self) -> None:
        """Force refresh of JWKS cache."""
        logger.info("Forcing JWKS cache refresh")
        self._cache.clear()
        self.get_jwks()  # This will fetch fresh JWKS

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache information
        """
        cached_jwks = self._cache.get("jwks")
        return {
            "cache_size": len(self._cache),
            "max_size": self._cache.maxsize,
            "ttl": self._cache.ttl,
            "has_cached_jwks": cached_jwks is not None,
            "cached_keys_count": len(cached_jwks["keys"]) if cached_jwks else 0,
        }

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "JWKSClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
