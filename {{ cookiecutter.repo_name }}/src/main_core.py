"""Core FastAPI application entry point for C.R.E.A.T.E. framework.

This module provides the essential FastAPI application for Phase 1 Issue 4,
implementing the basic C.R.E.A.T.E. framework API without advanced features.
"""

import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers.create_core import router as create_router
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management.

    Args:
        _app: The FastAPI application instance.
    """
    # Startup
    logger.info("Starting C.R.E.A.T.E. framework API")
    logger.info("Environment: %s", settings.environment)
    logger.info("Debug mode: %s", settings.debug)

    yield

    # Shutdown
    logger.info("Shutting down C.R.E.A.T.E. framework API")


# Create FastAPI application
app = FastAPI(
    title="PromptCraft-Hybrid C.R.E.A.T.E. Framework API",
    description="Core API for C.R.E.A.T.E. framework prompt enhancement",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Any:
    """Log all HTTP requests.

    Args:
        request: The incoming request.
        call_next: The next middleware or endpoint.

    Returns:
        The response from the next middleware or endpoint.
    """
    start_time = time.time()

    # Log request
    logger.info(
        "Request: %s %s",
        request.method,
        request.url.path,
        extra={
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else None,
        },
    )

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        "Response: %s in %.4fs",
        response.status_code,
        process_time,
        extra={
            "status_code": response.status_code,
            "process_time": process_time,
            "method": request.method,
            "path": request.url.path,
        },
    )

    return response


# Include routers
app.include_router(create_router)


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> dict[str, Any]:
    """Root endpoint providing API information.

    Returns:
        Dict containing API information.
    """
    return {
        "message": "PromptCraft-Hybrid C.R.E.A.T.E. Framework API",
        "version": "1.0.0",
        "framework": "C.R.E.A.T.E.",
        "environment": settings.environment,
        "debug": settings.debug,
        "docs_url": "/docs" if settings.debug else None,
        "endpoints": {
            "create": "/api/v1/create/",
            "health": "/api/v1/create/health",
            "domains": "/api/v1/create/domains",
            "framework": "/api/v1/create/framework",
        },
        "timestamp": time.time(),
    }


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check() -> dict[str, Any]:
    """Application health check endpoint.

    Returns:
        Dict containing health status.
    """
    return {
        "status": "healthy",
        "service": "promptcraft-hybrid-create-api",
        "version": "1.0.0",
        "environment": settings.environment,
        "debug": settings.debug,
        "timestamp": time.time(),
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, _exc: Exception) -> JSONResponse:
    """Handle 404 errors.

    Args:
        request: The request that caused the error.
        exc: The exception.

    Returns:
        JSONResponse with error details.
    """
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": f"The requested resource {request.url.path} was not found",
            "timestamp": time.time(),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Handle 500 errors.

    Args:
        request: The request that caused the error.
        exc: The exception.

    Returns:
        JSONResponse with error details.
    """
    logger.error("Internal server error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": time.time(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main_core:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        reload=settings.debug,
    )
