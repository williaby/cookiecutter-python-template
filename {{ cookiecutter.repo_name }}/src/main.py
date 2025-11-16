"""Main FastAPI application for PromptCraft-Hybrid.

This module defines the main FastAPI application with health check endpoints
and configuration status monitoring. It serves as the primary entry point
for the application with comprehensive security hardening.

Note: Security configuration uses environment variables for flexible deployment.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.ab_testing_endpoints import router as ab_testing_router

# Authentication and authorization imports
from src.api.auth_endpoints import audit_router, auth_router, system_router
from src.api.role_endpoints import role_router
from src.auth.exceptions import AuthExceptionHandler
from src.config.constants import (
    CORS_ORIGINS_BY_ENVIRONMENT,
    HEALTH_CHECK_ERROR_LIMIT,
    HEALTH_CHECK_SUGGESTION_LIMIT,
)
from src.config.health import (
    ConfigurationStatusModel,
    get_configuration_health_summary,
    get_configuration_status,
    get_mcp_configuration_health,
)
from src.config.settings import (
    ApplicationSettings,
    ConfigurationValidationError,
    get_settings,
)

# Security imports
from src.security.audit_logging import AuditEventSeverity, AuditEventType, audit_logger_instance
from src.security.error_handlers import setup_secure_error_handlers
from src.security.input_validation import SecureQueryParams, SecureTextInput
from src.security.middleware import setup_security_middleware
from src.security.rate_limiting import RateLimits, rate_limit, setup_rate_limiting
from src.utils.circuit_breaker import get_all_circuit_breakers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown.

    This function handles application startup and shutdown events,
    including configuration validation, security setup, and resource cleanup.

    Args:
        app: The FastAPI application instance

    Yields:
        None during application lifetime
    """
    logger.info("Starting PromptCraft-Hybrid application...")

    try:
        # Load and validate configuration on startup
        settings = get_settings(validate_on_startup=True)
        logger.info(
            "Application started successfully: %s v%s (%s)",
            settings.app_name,
            settings.version,
            settings.environment,
        )

        # Store settings in app state for access in endpoints
        app.state.settings = settings

        # Log application startup audit event
        audit_logger_instance.log_security_event(
            AuditEventType.ADMIN_SYSTEM_STARTUP,
            f"Application startup completed: {settings.app_name} v{settings.version}",
            severity=AuditEventSeverity.MEDIUM,
            additional_data={
                "environment": settings.environment,
                "debug_mode": settings.debug,
                "api_host": settings.api_host,
                "api_port": settings.api_port,
            },
        )

        yield

    except ConfigurationValidationError as e:
        logger.error("Configuration validation failed during startup: %s", e)

        # Log configuration failure as security event
        audit_logger_instance.log_security_event(
            AuditEventType.SECURITY_VALIDATION_FAILURE,
            "Configuration validation failed during startup",
            severity=AuditEventSeverity.CRITICAL,
            additional_data={
                "field_errors": e.field_errors,
                "suggestions": e.suggestions,
            },
        )

        # Log detailed errors for debugging
        for error in e.field_errors:
            logger.error("  Configuration error: %s", error)
        for suggestion in e.suggestions:
            logger.info("  Suggestion: %s", suggestion)
        raise
    except Exception as e:
        logger.error("Unexpected error during application startup: %s", e)

        # Log startup failure as critical security event
        audit_logger_instance.log_security_event(
            AuditEventType.ADMIN_SYSTEM_STARTUP,
            f"Application startup failed: {type(e).__name__}",
            severity=AuditEventSeverity.CRITICAL,
            additional_data={"error_message": str(e)},
        )
        raise
    finally:
        # Log application shutdown
        audit_logger_instance.log_security_event(
            AuditEventType.ADMIN_SYSTEM_SHUTDOWN,
            "Application shutdown initiated",
            severity=AuditEventSeverity.MEDIUM,
        )
        logger.info("PromptCraft-Hybrid application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance with security hardening
    """
    # Get settings for app metadata
    try:
        settings = get_settings(validate_on_startup=False)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning("Settings format error for app creation: %s", type(e).__name__)
        # Use defaults if settings unavailable
        settings = ApplicationSettings()
    except Exception:
        logger.exception("Failed to load settings for app creation")
        # Use defaults if settings unavailable
        settings = ApplicationSettings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="A Zen-powered AI workbench for transforming queries into accurate, context-aware outputs",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Setup security components
    logger.info("Configuring security hardening for %s environment", settings.environment)

    # 1. Setup secure error handlers (prevents stack trace leakage)
    setup_secure_error_handlers(app)

    # 2. Setup rate limiting (60 requests/minute per IP)
    setup_rate_limiting(app)

    # 3. Setup security middleware (headers, logging)
    setup_security_middleware(app)

    # 4. Add CORS middleware with enhanced security
    cors_origins = CORS_ORIGINS_BY_ENVIRONMENT.get(settings.environment, CORS_ORIGINS_BY_ENVIRONMENT["dev"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],  # Explicit methods for security
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
        ],  # Explicit headers for security
        expose_headers=["X-Process-Time", "X-RateLimit-Remaining"],
        max_age=600,  # Cache preflight requests for 10 minutes
    )

    logger.info("Security hardening configuration completed")
    return app


# Create the FastAPI application instance
app = create_app()

# Include authentication and authorization routers
app.include_router(auth_router)
app.include_router(role_router)
app.include_router(system_router)
app.include_router(audit_router)
app.include_router(ab_testing_router)


@app.get("/health", response_model=dict[str, Any])
@rate_limit(RateLimits.HEALTH_CHECK)
async def health_check(request: Request) -> dict[str, Any]:  # noqa: ARG001
    """Simple health check endpoint for basic monitoring.

    This endpoint provides a quick health status without detailed
    configuration information. It's suitable for load balancers
    and simple monitoring systems.

    Returns:
        Basic health status information

    Raises:
        HTTPException: If health check fails
    """
    try:
        health_summary = get_configuration_health_summary()

        # Add circuit breaker status
        circuit_breakers = get_all_circuit_breakers()
        circuit_breaker_status = {
            "circuit_breakers": {name: breaker.get_health_status() for name, breaker in circuit_breakers.items()},
        }

        if health_summary["healthy"]:
            return {
                "status": "healthy",
                "service": "promptcraft-hybrid",
                **health_summary,
                **circuit_breaker_status,
            }
        logger.warning("Health check failed - configuration unhealthy")
        raise AuthExceptionHandler.handle_service_unavailable(
            "promptcraft-hybrid",
            f"Health check failed: configuration unhealthy - {health_summary}",
        )
    except HTTPException:
        # Re-raise HTTPExceptions to preserve status codes
        raise
    except Exception as e:  # Catch-all for unhandled endpoint errors
        raise AuthExceptionHandler.handle_internal_error(
            "Health check",
            e,
            "Health check endpoint failed",
            expose_error=True,
        ) from e


@app.get("/health/config", response_model=ConfigurationStatusModel)
@rate_limit(RateLimits.HEALTH_CHECK)
async def configuration_health(request: Request) -> ConfigurationStatusModel:  # noqa: ARG001
    """Detailed configuration health check endpoint.

    This endpoint provides comprehensive configuration status information
    for operational monitoring and debugging. It includes validation status,
    configuration sources, and secret field counts without exposing
    sensitive values.

    Returns:
        Detailed configuration status information

    Raises:
        HTTPException: If configuration status cannot be determined
    """
    try:
        settings = get_settings(validate_on_startup=False)
        return get_configuration_status(settings)
    except ConfigurationValidationError as e:
        logger.error("Configuration validation failed in health check: %s", e)

        # Try to get settings for debug mode check, fallback to production mode
        try:
            settings = get_settings(validate_on_startup=False)
            debug_mode = settings.debug
        except Exception:
            debug_mode = False

        # Only expose detailed errors in debug mode
        if debug_mode:
            detail = {
                "error": "Configuration validation failed",
                "field_errors": e.field_errors[:HEALTH_CHECK_ERROR_LIMIT],
                "suggestions": e.suggestions[:HEALTH_CHECK_SUGGESTION_LIMIT],
            }
        else:
            detail = {
                "error": "Configuration validation failed",
                "details": "Contact system administrator",
            }

        raise AuthExceptionHandler.handle_internal_error(
            "Configuration validation",
            e,
            str(detail),
            expose_error=debug_mode,
        ) from e
    except Exception as e:  # Catch-all for unhandled endpoint errors
        raise AuthExceptionHandler.handle_internal_error(
            "Configuration health check",
            e,
            "Configuration health check failed - see application logs",
            expose_error=True,
        ) from e


@app.get("/health/mcp", response_model=dict[str, Any])
@rate_limit(RateLimits.HEALTH_CHECK)
async def mcp_health_check(request: Request) -> dict[str, Any]:  # noqa: ARG001
    """MCP configuration and parallel execution health check endpoint.

    This endpoint provides health status for MCP server configuration,
    client connections, and parallel subagent execution capabilities.
    Designed for monitoring MCP integration health and performance.

    Returns:
        MCP integration health status information

    Raises:
        HTTPException: If MCP health check fails
    """
    try:
        mcp_health = await get_mcp_configuration_health()

        if mcp_health["healthy"]:
            return {
                "status": "healthy",
                "service": "mcp-integration",
                **mcp_health,
            }

        logger.warning("MCP health check failed - integration unhealthy")
        raise AuthExceptionHandler.handle_service_unavailable(
            "mcp-integration",
            f"MCP integration unhealthy - {mcp_health}",
        )
    except ImportError as e:
        logger.warning("MCP integration not available")
        raise AuthExceptionHandler.handle_service_unavailable("mcp-integration", "MCP integration not available") from e
    except HTTPException:
        # Re-raise HTTPExceptions to preserve status codes
        raise
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error(
            "MCP health check",
            e,
            "MCP health check endpoint failed",
            expose_error=True,
        ) from e


@app.get("/health/circuit-breakers", response_model=dict[str, Any])
@rate_limit(RateLimits.HEALTH_CHECK)
async def circuit_breaker_health_check(request: Request) -> dict[str, Any]:  # noqa: ARG001
    """Circuit breaker health check endpoint.

    This endpoint provides detailed status information for all registered
    circuit breakers, including state, metrics, and configuration details.
    Useful for monitoring service resilience and failure patterns.

    Returns:
        Circuit breaker health status information

    Raises:
        HTTPException: If circuit breaker status cannot be determined
    """
    try:
        circuit_breakers = get_all_circuit_breakers()

        if not circuit_breakers:
            return {
                "status": "no_circuit_breakers",
                "service": "circuit-breaker-monitoring",
                "message": "No circuit breakers registered",
                "circuit_breakers": {},
            }

        # Get status for all circuit breakers
        circuit_breaker_details = {}
        overall_healthy = True

        for name, breaker in circuit_breakers.items():
            status_info = breaker.get_health_status()
            circuit_breaker_details[name] = status_info

            # Check if any circuit breaker is unhealthy
            if not status_info.get("healthy", True):
                overall_healthy = False

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "service": "circuit-breaker-monitoring",
            "overall_healthy": overall_healthy,
            "circuit_breaker_count": len(circuit_breakers),
            "circuit_breakers": circuit_breaker_details,
        }

    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error(
            "Circuit breaker health check",
            e,
            "Circuit breaker health check failed",
            expose_error=True,
        ) from e


@app.get("/")
@rate_limit(RateLimits.PUBLIC_READ)
async def root(request: Request) -> dict[str, str]:  # noqa: ARG001
    """Root endpoint providing basic application information.

    Returns:
        Basic application information
    """
    try:
        settings: ApplicationSettings = app.state.settings
        return {
            "service": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "status": "running",
            "docs_url": "/docs" if settings.debug else "disabled",
        }
    except AttributeError:
        # Fallback if settings not available in app state
        return {
            "service": "PromptCraft-Hybrid",
            "version": "unknown",
            "environment": "unknown",
            "status": "running",
        }


@app.get("/ping")
@rate_limit(RateLimits.HEALTH_CHECK)
async def ping(request: Request) -> dict[str, str]:  # noqa: ARG001
    """Simple ping endpoint for load balancer checks.

    Returns:
        Simple pong response
    """
    return {"message": "pong"}


@app.post("/api/v1/validate")
@rate_limit(RateLimits.API_DEFAULT)
async def validate_input(request: Request, data: SecureTextInput) -> dict[str, Any]:
    """Test endpoint for input validation demonstration.

    This endpoint demonstrates the input sanitization capabilities
    by accepting text input and returning the sanitized version.

    Args:
        request: The incoming request
        data: Input data to validate and sanitize

    Returns:
        Sanitized input data with validation results
    """
    # Log the API request
    audit_logger_instance.log_api_event(
        request=request,
        response_status=status.HTTP_200_OK,
        processing_time=0.1,  # Placeholder timing in seconds
    )

    return {
        "status": "success",
        "message": "Input validation successful",
        "sanitized_text": data.text,
        "original_length": len(data.text),
        "validation_applied": "XSS protection, length validation, null byte protection",
    }


@app.get("/api/v1/search")
@rate_limit(RateLimits.API_DEFAULT)
async def search_endpoint(request: Request, params: SecureQueryParams = Depends()) -> dict[str, Any]:  # noqa: ARG001
    """Test endpoint for query parameter validation.

    This endpoint demonstrates secure query parameter handling
    with automatic validation and sanitization.

    Args:
        request: The incoming request
        params: Query parameters to validate

    Returns:
        Validated query parameters
    """
    return {
        "status": "success",
        "query_params": {
            "search": params.search,
            "page": params.page,
            "limit": params.limit,
            "sort": params.sort,
        },
        "validation_notes": "All parameters sanitized and validated",
    }


if __name__ == "__main__":
    # Development server - not for production use
    import sys

    import uvicorn

    try:
        settings = get_settings()
        logger.info(
            "Starting development server on %s:%d",
            settings.api_host,
            settings.api_port,
        )
        uvicorn.run(
            "src.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level="info" if not settings.debug else "debug",
        )
    except ConfigurationValidationError as e:
        logger.error("Cannot start server due to configuration errors: %s", e)
        sys.exit(1)
    except (OSError, RuntimeError) as e:
        logger.error("Failed to start development server: %s", e)
        sys.exit(1)
    except Exception:  # Catch-all for startup errors
        logger.exception("Unexpected error starting development server")
        sys.exit(1)
