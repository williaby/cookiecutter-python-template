"""Security middleware for FastAPI applications.

This module provides comprehensive security middleware including headers,
rate limiting, and request logging for enhanced application security.
It implements defense-in-depth security patterns with configurable policies
for different deployment environments.

The module provides:
- Security headers middleware for web vulnerability protection
- Request logging middleware for audit trails and monitoring
- Environment-specific security configurations
- Sensitive data masking for secure logging

Architecture:
    The middleware follows the ASGI middleware pattern with proper request/response
    processing chains. Each middleware layer adds specific security enhancements
    while maintaining performance and observability.

Key Components:
    - SecurityHeadersMiddleware: Adds security headers (CSP, HSTS, etc.)
    - RequestLoggingMiddleware: Provides audit logging with data masking
    - setup_security_middleware(): Configures middleware stack

Dependencies:
    - fastapi: For Request/Response handling
    - starlette: For BaseHTTPMiddleware
    - src.config.settings: For environment-specific configuration
    - logging: For structured audit logging

Called by:
    - src/main.py: During FastAPI application initialization
    - setup_security_middleware(): For middleware configuration
    - FastAPI middleware stack: For request processing

Complexity: O(1) for header operations, O(n) for header masking where n is header count
"""

import logging
import time
from collections.abc import Callable
from typing import Any, cast

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all HTTP responses.

    This middleware adds essential security headers to protect against
    common web vulnerabilities including XSS, clickjacking, and MIME
    type confusion attacks.
    """

    def __init__(self, app: Any, csp_policy: str | None = None) -> None:
        """Initialize security headers middleware.

        Args:
            app: The ASGI application
            csp_policy: Custom Content Security Policy (optional)
        """
        super().__init__(app)
        self.csp_policy = csp_policy or self._default_csp_policy()

    def _default_csp_policy(self) -> str:
        """Generate default Content Security Policy.

        Returns:
            CSP policy string appropriate for the current environment
        """
        settings = get_settings(validate_on_startup=False)

        if settings.environment == "dev":
            # More permissive CSP for development
            return (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
                "https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' "
                "https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss: http://localhost:* https://localhost:*; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )

        # Strict CSP for production
        return (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "upgrade-insecure-requests"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add security headers to response.

        This method implements the core middleware functionality, adding security
        headers to protect against common web vulnerabilities.

        Args:
            request: The incoming request
            call_next: The next middleware/endpoint in the chain

        Returns:
            Response with security headers added

        Time Complexity: O(1) - Fixed number of header operations
        Space Complexity: O(1) - Fixed memory for header dictionary

        Called by:
            - FastAPI middleware stack during request processing
            - ASGI application layer

        Calls:
            - call_next(): Next middleware or endpoint in chain
            - _get_security_headers(): Security header generation
        """
        # Store request timestamp for error handling
        request.state.timestamp = time.time()

        # Process the request
        response = await call_next(request)

        # Add security headers
        security_headers = self._get_security_headers()

        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value

        # Cast to Response type for MyPy
        return cast(StarletteResponse, response)

    def _get_security_headers(self) -> dict[str, str]:
        """Get security headers based on environment.

        Returns:
            Dictionary of security headers to add
        """
        settings = get_settings(validate_on_startup=False)

        headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # XSS protection (disabled - rely on CSP instead)
            "X-XSS-Protection": "0",
            # Content Security Policy
            "Content-Security-Policy": self.csp_policy,
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions policy
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=()"
            ),
        }

        # Add HSTS only in production or when using HTTPS
        if settings.environment in ("staging", "prod"):
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # Add server identification removal
        headers["Server"] = "PromptCraft-Hybrid"

        return headers


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests for security monitoring and audit trails.

    This middleware logs all incoming requests with relevant security
    information for monitoring and incident response.
    """

    def _mask_sensitive_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Mask sensitive headers to prevent credential leakage in logs.

        This method implements data masking for security-sensitive headers
        to ensure credentials and secrets are not exposed in audit logs.

        Args:
            headers: Original headers dictionary

        Returns:
            Headers dictionary with sensitive values masked

        Time Complexity: O(n) where n is the number of headers
        Space Complexity: O(n) for the new masked headers dictionary

        Called by:
            - _log_request(): During request logging
            - Security audit logging systems

        Calls:
            - dict.items(): Header iteration
            - str.lower(): Case-insensitive header matching
        """
        sensitive_keys = {"authorization", "cookie", "proxy-authorization", "x-api-key"}
        masked_headers = {}

        for key, value in headers.items():
            if key.lower() in sensitive_keys:
                masked_headers[key] = "***REDACTED***"
            else:
                masked_headers[key] = value

        return masked_headers

    def __init__(self, app: Any, log_body: bool = False) -> None:
        """Initialize request logging middleware.

        Args:
            app: The ASGI application
            log_body: Whether to log request bodies (be careful with sensitive data)
        """
        super().__init__(app)
        self.log_body = log_body

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security logging.

        Args:
            request: The incoming request
            call_next: The next middleware/endpoint in the chain

        Returns:
            Response with timing information logged
        """
        start_time = time.time()

        # Log incoming request
        self._log_request(request)

        # Process the request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        self._log_response(request, response, process_time)

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        # Cast to Response type for MyPy
        return cast(StarletteResponse, response)

    def _log_request(self, request: Request) -> None:
        """Log incoming request details for security monitoring.

        Args:
            request: The incoming request
        """
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")

        # Log request details
        logger.info(
            "Request: %s %s from %s (User-Agent: %s)",
            request.method,
            request.url.path,
            client_ip,
            user_agent,
            extra={
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query) if request.url.query else None,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "headers": self._mask_sensitive_headers(dict(request.headers)),
            },
        )

    def _log_response(self, request: Request, response: Response, process_time: float) -> None:
        """Log response details for security monitoring.

        Args:
            request: The original request
            response: The response
            process_time: Time taken to process the request
        """
        client_ip = self._get_client_ip(request)

        logger.info(
            "Response: %s %s -> %d (%.3fs, %s)",
            request.method,
            request.url.path,
            response.status_code,
            process_time,
            client_ip,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": process_time,
                "client_ip": client_ip,
                "response_headers": dict(response.headers),
            },
        )

        # Log slow requests as warnings
        slow_request_threshold = 2.0  # Seconds
        if process_time > slow_request_threshold:
            logger.warning(
                "Slow request detected: %s %s took %.3fs (IP: %s)",
                request.method,
                request.url.path,
                process_time,
                client_ip,
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request.

        Args:
            request: The incoming request

        Returns:
            Client IP address
        """
        # Check for forwarded headers (common in reverse proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"


def setup_security_middleware(app: Any) -> None:
    """Configure security middleware for the FastAPI application.

    This function adds all necessary security middleware in the correct order.

    Args:
        app: The FastAPI application instance
    """
    settings = get_settings(validate_on_startup=False)

    # Add request logging middleware (outermost layer)
    app.add_middleware(RequestLoggingMiddleware, log_body=False)

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    logger.info(
        "Security middleware configured for %s environment",
        settings.environment,
    )
