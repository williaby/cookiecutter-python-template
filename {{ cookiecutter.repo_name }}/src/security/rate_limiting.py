"""Rate limiting for FastAPI applications using slowapi.

This module provides comprehensive rate limiting capabilities to protect
against abuse and ensure fair resource usage across all API endpoints.
It implements client identification, storage backends, and custom error
handling for production-grade rate limiting.

The module provides:
- Client identification with proxy header support
- Configurable storage backends (Redis for production, memory for development)
- Custom rate limit exceeded error handling
- Predefined rate limits for different endpoint types
- Utility functions for rate limit configuration

Architecture:
    The rate limiting system uses slowapi for sliding window rate limiting
    with configurable storage backends. It supports distributed rate limiting
    in production environments through Redis storage.

Key Components:
    - get_client_identifier(): Client IP extraction with proxy support
    - create_limiter(): Limiter configuration with environment-specific storage
    - rate_limit_exceeded_handler(): Custom error response handling
    - RateLimits class: Predefined limits for different endpoint types
    - setup_rate_limiting(): FastAPI integration configuration

Dependencies:
    - slowapi: For sliding window rate limiting implementation
    - fastapi: For HTTP exception handling
    - src.config.settings: For environment-specific configuration
    - Redis: For distributed rate limiting storage (production)

Called by:
    - src/main.py: During FastAPI application initialization
    - FastAPI decorators: For endpoint-specific rate limiting
    - Middleware stack: For request processing

Complexity: O(1) for rate limit checks, O(log n) for Redis operations where n is rate limit window size
"""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.auth.exceptions import AuthExceptionHandler
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def get_client_identifier(request: Request) -> str:
    """Extract client identifier for rate limiting.

    This function determines how to identify clients for rate limiting purposes.
    It checks for forwarded headers first (for reverse proxy setups) and falls
    back to the direct client IP.

    Args:
        request: The incoming request

    Returns:
        Client identifier string for rate limiting

    Time Complexity: O(1) - Simple header lookup and string operations
    Space Complexity: O(1) - Fixed memory for IP address strings

    Called by:
        - slowapi.Limiter: For client identification during rate limiting
        - rate_limit_exceeded_handler(): For error logging
        - Rate limit middleware: During request processing

    Calls:
        - request.headers.get(): HTTP header retrieval
        - str.split(): Header parsing for X-Forwarded-For
        - get_remote_address(): Fallback client IP extraction
    """
    # Check for forwarded headers (common in reverse proxy setups)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        client_ip = forwarded_for.split(",")[0].strip()
        logger.debug("Using X-Forwarded-For IP for rate limiting: %s", client_ip)
        return client_ip

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        client_ip = real_ip.strip()
        logger.debug("Using X-Real-IP for rate limiting: %s", client_ip)
        return client_ip

    # Fall back to slowapi's default method
    client_ip = get_remote_address(request)
    logger.debug("Using direct client IP for rate limiting: %s", client_ip)
    return client_ip


def create_limiter() -> Limiter:
    """Create and configure the rate limiter instance.

    Returns:
        Configured Limiter instance
    """
    settings = get_settings(validate_on_startup=False)

    # Configure storage backend based on environment
    if settings.environment == "prod":
        # Production: Use Redis for distributed rate limiting
        redis_host = getattr(settings, "redis_host", "localhost")
        redis_port = getattr(settings, "redis_port", 6379)
        redis_db = getattr(settings, "redis_db", 0)
        storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"
        logger.info("Using Redis storage for production rate limiting: %s", storage_uri)
    else:
        # Development/staging: in-memory storage is fine
        storage_uri = "memory://"
        logger.info("Using in-memory storage for %s environment", settings.environment)

    # Create limiter with custom key function
    limiter = Limiter(
        key_func=get_client_identifier,
        storage_uri=storage_uri,
        default_limits=["100 per minute"],  # Default fallback limit for unspecified endpoints
    )

    logger.info(
        "Rate limiter configured with storage: %s (Environment: %s)",
        storage_uri,
        settings.environment,
    )

    return limiter


# Global limiter instance
limiter = create_limiter()


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Any:
    """Custom handler for rate limit exceeded errors.

    This handler provides detailed rate limiting information while maintaining
    security by not exposing internal implementation details.

    Args:
        request: The incoming request
        exc: The rate limit exceeded exception

    Returns:
        JSON response with rate limit information

    Time Complexity: O(1) - Simple error response generation
    Space Complexity: O(1) - Fixed memory for error response dictionary

    Called by:
        - FastAPI exception handler: When RateLimitExceeded is raised
        - slowapi middleware: During rate limit enforcement
        - setup_rate_limiting(): Exception handler registration

    Calls:
        - get_client_identifier(): For client IP extraction
        - HTTPException(): For structured error response
        - logger.warning(): For security event logging
    """
    client_ip = get_client_identifier(request)

    # Log rate limit violation
    logger.warning(
        "Rate limit exceeded for client %s on %s %s (Limit: %s)",
        client_ip,
        request.method,
        request.url.path,
        exc.detail,
    )

    # Calculate retry-after header
    default_retry_after = 60  # Default retry time in seconds
    retry_after = exc.retry_after if hasattr(exc, "retry_after") else default_retry_after

    # Create detailed error response
    error_detail = {
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please slow down and try again later.",
        "retry_after": retry_after,
        "limit": exc.detail if hasattr(exc, "detail") else f"{default_retry_after} per minute",
    }

    # Return HTTP 429 with rate limit headers
    raise AuthExceptionHandler.handle_rate_limit_error(
        retry_after=retry_after,
        detail=error_detail,
        client_identifier=client_ip,
    )


def setup_rate_limiting(app: Any) -> None:
    """Configure rate limiting for the FastAPI application.

    This function sets up the rate limiter and its error handler.

    Args:
        app: The FastAPI application instance
    """
    # Add rate limiter to app state for access in routes
    app.state.limiter = limiter

    # Add custom rate limit exceeded handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    logger.info("Rate limiting configured for application")


# Common rate limit decorators for different endpoint types
class RateLimits:
    """Predefined rate limits for common endpoint types."""

    # API endpoints (60 requests per minute per IP)
    API_DEFAULT = "60/minute"

    # Health check endpoints (higher limit for monitoring)
    HEALTH_CHECK = "300/minute"

    # Authentication endpoints (stricter limits)
    AUTH = "10/minute"

    # File upload endpoints (very strict)
    UPLOAD = "5/minute"

    # Administrative endpoints (very strict)
    ADMIN = "10/minute"

    # Public read-only endpoints (moderate limits)
    PUBLIC_READ = "100/minute"
    # Slow API endpoints that require more processing time
    API_SLOW = "20/minute"


def get_rate_limit_for_endpoint(endpoint_type: str) -> str:
    """Get appropriate rate limit for endpoint type.

    Args:
        endpoint_type: Type of endpoint (api, health, auth, upload, admin, public)

    Returns:
        Rate limit string for the endpoint type
    """
    limits_map = {
        "api": RateLimits.API_DEFAULT,
        "health": RateLimits.HEALTH_CHECK,
        "auth": RateLimits.AUTH,
        "upload": RateLimits.UPLOAD,
        "admin": RateLimits.ADMIN,
        "public": RateLimits.PUBLIC_READ,
    }

    return limits_map.get(endpoint_type, RateLimits.API_DEFAULT)


# Utility function to create rate limit decorator
def rate_limit(limit: str) -> Callable:
    """Create a rate limit decorator for endpoints.

    Args:
        limit: Rate limit string (e.g., "60/minute")

    Returns:
        Decorator function for applying rate limits
    """
    return limiter.limit(limit)
