"""Secure error handling for FastAPI applications.

This module provides production-safe error handlers that prevent information
disclosure through stack traces while maintaining useful error responses
for debugging in development environments.
It implements defense-in-depth error handling with environment-specific
response sanitization and comprehensive security headers.

The module provides:
- Secure error response generation with information disclosure prevention
- Environment-specific error detail levels (verbose dev, minimal prod)
- Comprehensive exception handler coverage (HTTP, validation, general)
- Security headers on all error responses
- Structured logging for monitoring and debugging
- Utility functions for manual exception creation

Architecture:
    The error handling system follows a layered approach with specialized
    handlers for different exception types, all routing through a central
    secure response generator that sanitizes output based on environment.

Key Components:
    - create_secure_error_response(): Core security function for response sanitization
    - Exception handlers: Specialized handlers for different error types
    - setup_secure_error_handlers(): Configuration function for FastAPI integration
    - create_secure_http_exception(): Utility for manual exception creation

Dependencies:
    - fastapi: For Request/Response handling and exception types
    - starlette: For low-level HTTP exception handling
    - src.config.settings: For environment-specific behavior
    - logging: For structured error logging and monitoring

Called by:
    - src/main.py: During FastAPI application initialization
    - FastAPI exception handling system: For automatic error processing
    - Application code: For manual exception creation
    - Middleware stack: For error response processing

Complexity: O(1) for error response generation, O(n) for validation error processing where n is error count
"""

import logging
import traceback
from collections.abc import Callable
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from src.auth.exceptions import AuthExceptionHandler
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def create_secure_error_response(
    request: Request,
    error: Exception,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail: str = "Internal server error",
) -> Response:
    """Create a secure error response that prevents information disclosure.

    Core security function that sanitizes error responses to prevent sensitive
    information leakage while maintaining debugging capabilities in development.

    **Security Features:**
    - Prevents stack trace disclosure in production
    - Sanitizes error messages to avoid information leakage
    - Adds security headers to all error responses
    - Provides detailed logging for monitoring

    **Environment Behavior:**
    - Development: Includes debug info, error types, and stack traces
    - Production: Returns minimal, safe error information

    Args:
        request: The incoming FastAPI request object
        error: The exception that occurred
        status_code: HTTP status code to return (default: 500)
        detail: Error detail message (production-safe)

    Returns:
        JSONResponse with sanitized error information and security headers

    Example:
        >>> response = create_secure_error_response(
        ...     request,
        ...     ValueError("Invalid input"),
        ...     status_code=400,
        ...     detail="Bad request"
        ... )

    Note:
        Always logs the full error details for monitoring regardless of environment.
    """
    settings = get_settings(validate_on_startup=False)

    # Base response with safe information
    # Only includes non-sensitive data that's safe for client consumption
    response_data: dict[str, Any] = {
        "error": detail,  # Production-safe error message
        "status_code": status_code,  # HTTP status code
        "timestamp": request.state.timestamp if hasattr(request.state, "timestamp") else None,  # Request timestamp
        "path": str(request.url.path),  # Request path (no query params)
    }

    # Add debug information only in development
    # SECURITY: Never expose internal details in production
    if settings.debug and settings.environment == "dev":
        response_data["debug"] = {
            "error_type": type(error).__name__,  # Exception class name
            "error_message": str(error),  # Exception message
        }

        # Include traceback only in development with explicit debug flag
        # Skip traceback for HTTP exceptions to avoid noise
        if isinstance(error, Exception) and not isinstance(error, HTTPException | StarletteHTTPException):
            response_data["debug"]["traceback"] = traceback.format_exc()

    # Log the actual error for monitoring (with full details)
    # SECURITY: Always log full error details for monitoring/debugging
    # regardless of what's returned to the client
    logger.error(
        "Application error: %s - %s (Path: %s, IP: %s)",
        type(error).__name__,  # Exception class name
        str(error),  # Exception message
        request.url.path,  # Request path
        request.client.host if request.client else "unknown",  # Client IP
        exc_info=True,  # Include full stack trace in logs
    )

    return JSONResponse(
        status_code=status_code,
        content=response_data,
        headers={
            "X-Content-Type-Options": "nosniff",  # Prevent MIME type sniffing attacks
            "X-Frame-Options": "DENY",  # Prevent clickjacking attacks
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """Handle HTTP exceptions with secure error responses.

    FastAPI exception handler for HTTPException instances. Routes all HTTP
    exceptions through the secure error response system to ensure consistent
    security posture.

    Args:
        request: The incoming FastAPI request object
        exc: The HTTP exception that was raised

    Returns:
        Secure JSON error response with appropriate status code and headers

    Example:
        This handler is automatically invoked for exceptions like:
        >>> raise HTTPException(status_code=404, detail="Not found")

    Note:
        Registered automatically via setup_secure_error_handlers()
    """
    return create_secure_error_response(
        request=request,
        error=exc,
        status_code=exc.status_code,
        detail=exc.detail if isinstance(exc.detail, str) else "HTTP error",
    )


async def starlette_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> Response:
    """Handle Starlette HTTP exceptions with secure error responses.

    Starlette-level exception handler that catches HTTP exceptions from the
    underlying ASGI framework. Ensures consistent error handling across all
    HTTP exceptions regardless of their origin.

    Args:
        request: The incoming FastAPI request object
        exc: The Starlette HTTP exception that was raised

    Returns:
        Secure JSON error response with appropriate status code and headers

    Example:
        Handles low-level HTTP exceptions like 404 errors from routing
        or middleware-level exceptions.

    Note:
        Provides a safety net for HTTP exceptions not caught by FastAPI handlers.
    """
    return create_secure_error_response(
        request=request,
        error=exc,
        status_code=exc.status_code,
        detail=exc.detail if isinstance(exc.detail, str) else "HTTP error",
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    """Handle request validation errors with secure responses.

    Specialized handler for Pydantic request validation errors. Provides
    detailed validation information in development while maintaining security
    in production environments.

    **Security Behavior:**
    - Development: Detailed validation errors with field paths and messages
    - Production: Minimal validation error information

    Args:
        request: The incoming FastAPI request object
        exc: The request validation error containing field-level details

    Returns:
        Secure JSON error response with validation details (environment-dependent)

    Example:
        Handles validation errors from Pydantic models:
        >>> # Input: {"email": "invalid-email", "age": "not-a-number"}
        >>> # Development response includes field paths and specific error messages
        >>> # Production response shows generic "Invalid request data"

    Note:
        Logs all validation errors for monitoring regardless of environment.
    """
    settings = get_settings(validate_on_startup=False)

    # Create sanitized validation error details
    if settings.debug and settings.environment == "dev":
        # In development, provide detailed validation errors
        detail = "Request validation failed"
        validation_errors = []

        # Process each validation error for detailed feedback
        for error in exc.errors():
            validation_errors.append(
                {
                    "field": " -> ".join(str(loc) for loc in error["loc"]),  # Field path (e.g., "user -> email")
                    "message": error["msg"],  # Error message
                    "type": error["type"],  # Error type
                },
            )

        response_data = {
            "error": detail,
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_errors": validation_errors,
        }
    else:
        # In production, provide minimal validation error information
        response_data = {
            "error": "Invalid request data",
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "message": "Please check your request parameters and try again",
        }

    # Log validation failures for monitoring potential attack patterns
    logger.warning(
        "Request validation failed: %s (Path: %s, IP: %s)",
        exc.errors(),  # Full validation error details
        request.url.path,  # Request path
        request.client.host if request.client else "unknown",  # Client IP
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data,
        headers={
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> Response:
    """Handle all other exceptions with secure error responses.

    Ultimate catch-all handler that prevents any stack traces from leaking
    to clients in production environments. This is the security backstop
    for all unhandled exceptions.

    **Critical Security Function:**
    - Prevents information disclosure from unhandled exceptions
    - Always returns generic error messages in production
    - Logs full exception details for debugging
    - Adds security headers to all responses

    Args:
        request: The incoming FastAPI request object
        exc: The unhandled exception that occurred

    Returns:
        Secure JSON error response with minimal information

    Example:
        Catches any exception not handled by other handlers:
        >>> # Database connection error, import error, etc.
        >>> # Returns: {"error": "An unexpected error occurred", "status_code": 500}

    Note:
        This handler should never be bypassed to ensure security.
    """
    return create_secure_error_response(
        request=request,
        error=exc,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred",
    )


def setup_secure_error_handlers(app: FastAPI) -> None:
    """Configure secure error handlers for the FastAPI application.

    Registers all necessary error handlers to prevent information disclosure
    through stack traces and error messages. This is a critical security
    configuration that must be called during application startup.

    **Handlers Registered:**
    1. HTTPException -> http_exception_handler
    2. StarletteHTTPException -> starlette_http_exception_handler
    3. RequestValidationError -> validation_exception_handler
    4. Exception -> general_exception_handler (catch-all)

    Args:
        app: The FastAPI application instance to configure

    Side Effects:
        - Registers exception handlers with the FastAPI app
        - Logs configuration success message

    Example:
        >>> app = FastAPI()
        >>> setup_secure_error_handlers(app)
        >>> # All exceptions now handled securely

    Note:
        Must be called before application startup to ensure security coverage.
    """
    # Register exception handlers with proper type casting
    app.add_exception_handler(HTTPException, cast(Callable, http_exception_handler))
    app.add_exception_handler(StarletteHTTPException, cast(Callable, starlette_http_exception_handler))
    app.add_exception_handler(RequestValidationError, cast(Callable, validation_exception_handler))
    app.add_exception_handler(Exception, cast(Callable, general_exception_handler))

    logger.info("Secure error handlers configured for application")


# Utility function for manual error creation
def create_secure_http_exception(
    status_code: int,
    detail: str,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """Create an HTTPException with security headers.

    Utility function for manually creating HTTP exceptions with consistent
    security headers. Ensures all manually raised exceptions follow the
    same security standards as automatically handled ones.

    **Security Headers Added:**
    - X-Content-Type-Options: nosniff (prevents MIME type sniffing)
    - X-Frame-Options: DENY (prevents clickjacking)

    Args:
        status_code: HTTP status code for the exception
        detail: Error detail message (should be production-safe)
        headers: Additional headers to include (optional)

    Returns:
        HTTPException with secure headers pre-configured

    Example:
        >>> raise create_secure_http_exception(
        ...     status_code=403,
        ...     detail="Access denied",
        ...     headers={"X-Custom-Header": "value"}
        ... )

    Note:
        Custom headers are merged with security headers (security headers take precedence).

        **DEPRECATED**: Consider using AuthExceptionHandler methods instead for
        standardized authentication, authorization, and validation errors.
        This function remains for backward compatibility and non-auth errors.
    """
    secure_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }

    if headers:
        secure_headers.update(headers)

    return HTTPException(
        status_code=status_code,
        detail=detail,
        headers=secure_headers,
    )


def create_auth_aware_http_exception(
    status_code: int,
    detail: str,
    headers: dict[str, str] | None = None,
    user_identifier: str = "",
    log_message: str = "",
) -> HTTPException:
    """Create HTTP exception using AuthExceptionHandler for common auth scenarios.

    Wrapper that routes common HTTP errors through AuthExceptionHandler while
    maintaining backward compatibility with create_secure_http_exception.

    Args:
        status_code: HTTP status code for the exception
        detail: Error detail message (should be production-safe)
        headers: Additional headers to include (optional)
        user_identifier: User identifier for logging (optional)
        log_message: Custom log message (optional)

    Returns:
        HTTPException created via AuthExceptionHandler or fallback

    Example:
        >>> raise create_auth_aware_http_exception(
        ...     status_code=401,
        ...     detail="Authentication required",
        ...     user_identifier="service_token_xyz"
        ... )
    """
    # Route common auth errors through AuthExceptionHandler
    if status_code == 401:
        return AuthExceptionHandler.handle_authentication_error(
            detail=detail,
            log_message=log_message or detail,
            user_identifier=user_identifier,
        )
    if status_code == 403:
        return AuthExceptionHandler.handle_permission_error(
            permission_name="access",
            user_identifier=user_identifier,
            detail=detail,
        )
    if status_code == 422:
        return AuthExceptionHandler.handle_validation_error(
            detail,
            field_name="request_data",
        )
    if status_code == 429:
        return AuthExceptionHandler.handle_rate_limit_error(
            detail=detail,
            client_identifier=user_identifier,
        )
    if status_code == 503:
        return AuthExceptionHandler.handle_service_unavailable(
            service_name="application",
            detail=detail,
        )
    # Fallback to original secure exception for non-auth errors
    return create_secure_http_exception(status_code, detail, headers)
