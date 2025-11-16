"""FastAPI authentication middleware for Cloudflare Access integration.

This module provides FastAPI middleware for JWT-based authentication with:
- JWT token extraction from Cloudflare Access headers
- Service token validation for non-interactive API access
- Email-based user identification and role mapping
- Rate limiting to prevent DOS attacks
- User context injection for downstream processing
- Database tracking for usage analytics and audit logging
"""

import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import func, select, text, update
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth.exceptions import AuthExceptionHandler
from src.database import DatabaseError, get_database_manager
from src.database.connection import get_db
from src.database.models import AuthenticationEvent, UserSession

from .config import AuthenticationConfig
from .constants import (
    AUTH_EVENT_GENERAL,
    AUTH_EVENT_JWT,
    AUTH_EVENT_SERVICE_TOKEN,
    ERROR_CODE_TOKEN_EXPIRED,
    ERROR_CODE_TOKEN_INACTIVE,
    ERROR_CODE_TOKEN_NOT_FOUND,
    ERROR_CODE_VALIDATION_EXCEPTION,
    PERMISSION_ADMIN,
    RATE_LIMIT_KEY_EMAIL,
    RATE_LIMIT_KEY_IP,
    RATE_LIMIT_KEY_USER,
    SERVICE_TOKEN_PREFIX,
    USER_TYPE_SERVICE_TOKEN,
)
from .jwks_client import JWKSClient
from .jwt_validator import JWTValidator
from .models import AuthenticatedUser, AuthenticationError, JWTValidationError

logger = logging.getLogger(__name__)


class ServiceTokenUser:
    """Represents an authenticated service token user."""

    def __init__(self, token_id: str, token_name: str, metadata: dict, usage_count: int = 0) -> None:
        """Initialize service token user.

        Args:
            token_id: Unique token identifier
            token_name: Human-readable token name
            metadata: Token metadata including permissions
            usage_count: Current usage count
        """
        self.token_id = token_id
        self.token_name = token_name
        self.metadata = metadata
        self.usage_count = usage_count
        self.user_type = USER_TYPE_SERVICE_TOKEN

    def has_permission(self, permission: str) -> bool:
        """Check if token has a specific permission.

        Args:
            permission: Permission to check

        Returns:
            True if token has permission, False otherwise
        """
        permissions = self.metadata.get("permissions", [])
        return permission in permissions or PERMISSION_ADMIN in permissions


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Cloudflare Access authentication."""

    def __init__(
        self,
        app: FastAPI,
        config: AuthenticationConfig,
        jwt_validator: JWTValidator,
        excluded_paths: list[str] | None = None,
        database_enabled: bool = True,
    ) -> None:
        """Initialize authentication middleware.

        Args:
            app: FastAPI application instance
            config: Authentication configuration
            jwt_validator: JWT validator instance
            excluded_paths: List of paths to exclude from authentication
            database_enabled: Whether database integration is enabled
        """
        super().__init__(app)
        self.config = config
        self.jwt_validator = jwt_validator
        self.database_enabled = database_enabled
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Process request through authentication middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler

        Returns:
            HTTP response
        """
        start_time = time.time()

        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        # Skip authentication if disabled
        if not self.config.cloudflare_access_enabled:
            logger.debug("Cloudflare Access authentication is disabled")
            return await call_next(request)

        try:
            # Extract and validate JWT token
            jwt_start = time.time()
            authenticated_user = await self._authenticate_request(request)
            (time.time() - jwt_start) * 1000  # Convert to milliseconds

            # Database session tracking (if enabled)
            db_start = time.time()
            if self.database_enabled:
                await self._update_user_session(authenticated_user, request)
            (time.time() - db_start) * 1000  # Convert to milliseconds

            # Inject user context into request state
            request.state.authenticated_user = authenticated_user

            # Handle different user types
            if isinstance(authenticated_user, ServiceTokenUser):
                request.state.user_email = None  # Service tokens don't have email
                request.state.user_role = None  # Service tokens don't have roles
                request.state.token_metadata = authenticated_user.metadata
            else:
                request.state.user_email = authenticated_user.email
                request.state.user_role = authenticated_user.role

            # Log successful authentication
            if self.config.auth_logging_enabled:
                if isinstance(authenticated_user, ServiceTokenUser):
                    logger.info(f"Authenticated service token: {authenticated_user.token_name}")
                else:
                    logger.info(f"Authenticated user: {authenticated_user.email} with role: {authenticated_user.role}")

            # Database event logging (if enabled)
            if self.database_enabled:
                (time.time() - start_time) * 1000
                user_email = None
                service_token_name = None
                if isinstance(authenticated_user, ServiceTokenUser):
                    service_token_name = authenticated_user.token_name
                else:
                    user_email = authenticated_user.email

                await self._log_authentication_event(
                    request,
                    user_email=user_email,
                    service_token_name=service_token_name,
                    event_type=AUTH_EVENT_GENERAL,
                    success=True,
                )

            # Proceed to next middleware/endpoint
            return await call_next(request)

        except AuthenticationError as e:
            # Log authentication failure
            if self.config.auth_logging_enabled:
                logger.warning(f"Authentication failed: {e.message}")

            # Database event logging for failures (if enabled)
            if self.database_enabled:
                await self._log_authentication_event(
                    request,
                    user_email=None,
                    service_token_name=None,
                    event_type=AUTH_EVENT_GENERAL,
                    success=False,
                    error_details={"error": str(e)},
                )

            # Use appropriate error handling based on status code
            if e.status_code == 500:
                # Internal server error (e.g., database connection failure)
                auth_exception = AuthExceptionHandler.handle_internal_error(
                    operation_name="Authentication middleware",
                    error=e,
                    detail=e.message,
                    expose_error=False,  # Never expose internal errors in production
                )
            else:
                # Standard authentication error (401, etc.)
                auth_exception = AuthExceptionHandler.handle_authentication_error(
                    detail=e.message,
                    user_identifier=getattr(request.state, "user_email", None)
                    or getattr(request.state, "authenticated_user", {}).get("email", ""),
                )
            return JSONResponse(
                status_code=auth_exception.status_code,
                content={"error": auth_exception.detail},
                headers=auth_exception.headers or {},
            )

        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in authentication middleware: {e}", exc_info=True)

            # Database event logging for errors (if enabled)
            if self.database_enabled:
                await self._log_authentication_event(
                    request,
                    user_email=None,
                    service_token_name=None,
                    event_type=AUTH_EVENT_GENERAL,
                    success=False,
                    error_details={"error": str(e)},
                )

            # Use standardized error handling for internal errors
            internal_exception = AuthExceptionHandler.handle_internal_error(
                operation_name="Authentication middleware",
                error=e,
                detail="Authentication system error",
                expose_error=False,  # Never expose internal errors in production
            )
            return JSONResponse(
                status_code=internal_exception.status_code,
                content={"error": internal_exception.detail},
            )

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path should be excluded from authentication.

        Args:
            path: Request path

        Returns:
            True if path should be excluded, False otherwise
        """
        for excluded in self.excluded_paths:
            # Exact match or prefix match with trailing slash
            if path == excluded or path.startswith(excluded + "/"):
                return True
        return False

    async def _authenticate_request(self, request: Request) -> AuthenticatedUser | ServiceTokenUser:
        """Authenticate the request and return user information.

        Args:
            request: HTTP request to authenticate

        Returns:
            AuthenticatedUser or ServiceTokenUser with validated information

        Raises:
            AuthenticationError: If authentication fails
        """
        # Extract authentication token (JWT or Service Token)
        token = self._extract_auth_token(request)
        if not token:
            raise AuthenticationError("Missing authentication token", 401)

        # Check if this is a service token (starts with service token prefix)
        if token.startswith(SERVICE_TOKEN_PREFIX):
            return await self._validate_service_token(request, token)
        # Handle JWT token validation (existing flow)
        return await self._validate_jwt_token(request, token)

    async def _validate_jwt_token(self, request: Request, token: str) -> AuthenticatedUser:
        """Validate JWT token using existing flow.

        Args:
            request: HTTP request
            token: JWT token string

        Returns:
            AuthenticatedUser with validated user information

        Raises:
            AuthenticationError: If validation fails
        """
        try:
            # Validate JWT token
            authenticated_user = self.jwt_validator.validate_token(
                token,
                email_whitelist=self.config.email_whitelist if self.config.email_whitelist_enabled else None,
            )

            # Log authentication event
            await self._log_authentication_event(
                request,
                user_email=authenticated_user.email,
                event_type=AUTH_EVENT_JWT,
                success=True,
            )

            return authenticated_user

        except JWTValidationError as e:
            # Log failed authentication
            await self._log_authentication_event(
                request,
                event_type=AUTH_EVENT_JWT,
                success=False,
                error_details={"error": str(e), "message": e.message},
            )
            # Convert JWT validation errors to authentication errors
            raise AuthenticationError(f"Token validation failed: {e.message}", 401) from e

    async def _validate_service_token(self, request: Request, token: str) -> ServiceTokenUser:
        """Validate service token against database.

        Args:
            request: HTTP request
            token: Service token string

        Returns:
            ServiceTokenUser with validated token information

        Raises:
            AuthenticationError: If validation fails
        """
        try:
            # Hash the token for database lookup
            token_hash = hashlib.sha256(token.encode()).hexdigest()

            # Get database session and validate token
            async for session in get_db():
                # Query for active, non-expired token
                result = await session.execute(
                    text(
                        """
                        SELECT id, token_name, token_metadata, usage_count, is_active,
                               CASE
                                   WHEN expires_at IS NULL THEN FALSE
                                   WHEN expires_at > NOW() THEN FALSE
                                   ELSE TRUE
                               END as is_expired
                        FROM service_tokens
                        WHERE token_hash = :token_hash
                    """,
                    ),
                    {"token_hash": token_hash},
                )

                token_record = result.fetchone()

                if not token_record:
                    await self._log_authentication_event(
                        request,
                        event_type=AUTH_EVENT_SERVICE_TOKEN,
                        success=False,
                        error_details={"error": ERROR_CODE_TOKEN_NOT_FOUND},
                    )
                    raise AuthenticationError("Invalid service token", 401)

                # Check if token is active and not expired
                if not token_record.is_active:
                    await self._log_authentication_event(
                        request,
                        service_token_name=token_record.token_name,
                        event_type=AUTH_EVENT_SERVICE_TOKEN,
                        success=False,
                        error_details={"error": ERROR_CODE_TOKEN_INACTIVE},
                    )
                    raise AuthenticationError("Service token is inactive", 401)

                if token_record.is_expired:
                    await self._log_authentication_event(
                        request,
                        service_token_name=token_record.token_name,
                        event_type=AUTH_EVENT_SERVICE_TOKEN,
                        success=False,
                        error_details={"error": ERROR_CODE_TOKEN_EXPIRED},
                    )
                    raise AuthenticationError("Service token has expired", 401)

                # Update usage count and last_used timestamp
                await session.execute(
                    text(
                        """
                        UPDATE service_tokens
                        SET usage_count = usage_count + 1, last_used = NOW()
                        WHERE token_hash = :token_hash
                    """,
                    ),
                    {"token_hash": token_hash},
                )
                await session.commit()

                # Log successful authentication
                await self._log_authentication_event(
                    request,
                    service_token_name=token_record.token_name,
                    event_type=AUTH_EVENT_SERVICE_TOKEN,
                    success=True,
                )

                # Create ServiceTokenUser
                service_user = ServiceTokenUser(
                    token_id=str(token_record.id),
                    token_name=token_record.token_name,
                    metadata=token_record.token_metadata or {},
                    usage_count=token_record.usage_count + 1,
                )

                return service_user

        except AuthenticationError:
            # Re-raise authentication errors as-is
            raise
        except Exception as e:
            logger.error(f"Service token validation error: {e}")
            await self._log_authentication_event(
                request,
                event_type=AUTH_EVENT_SERVICE_TOKEN,
                success=False,
                error_details={"error": ERROR_CODE_VALIDATION_EXCEPTION, "details": str(e)},
            )
            raise AuthenticationError("Service token validation failed", 500) from e

    def _extract_auth_token(self, request: Request) -> str | None:
        """Extract authentication token (JWT or Service Token) from headers.

        Args:
            request: HTTP request

        Returns:
            Authentication token string if found, None otherwise
        """
        # Primary: Cloudflare Access JWT header
        cf_access_jwt = request.headers.get("CF-Access-Jwt-Assertion")
        if cf_access_jwt:
            logger.debug("Found JWT token in CF-Access-Jwt-Assertion header")
            return cf_access_jwt

        # Standard Authorization header (supports both JWT and Service Tokens)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            token_type = "service token" if token.startswith(SERVICE_TOKEN_PREFIX) else "JWT token"
            logger.debug(f"Found {token_type} in Authorization header")
            return token

        # Fallback: Custom header for testing/development
        custom_jwt = request.headers.get("X-JWT-Token")
        if custom_jwt:
            logger.debug("Found JWT token in X-JWT-Token header")
            return custom_jwt

        # Service token specific headers (for non-interactive access)
        service_token = request.headers.get("X-Service-Token")
        if service_token:
            logger.debug("Found service token in X-Service-Token header")
            return service_token

        logger.debug("No authentication token found in request headers")
        return None

    def _extract_jwt_token(self, request: Request) -> str | None:
        """Extract JWT token from Cloudflare Access headers (legacy method).

        Args:
            request: HTTP request

        Returns:
            JWT token string if found, None otherwise
        """
        return self._extract_auth_token(request)

    async def _log_authentication_event(
        self,
        request: Request,
        user_email: str | None = None,
        service_token_name: str | None = None,
        event_type: str = AUTH_EVENT_GENERAL,
        success: bool = True,
        error_details: dict | None = None,
    ) -> None:
        """Log authentication event to database.

        Args:
            request: HTTP request
            user_email: User email (for JWT auth)
            service_token_name: Service token name (for service token auth)
            event_type: Type of authentication event
            success: Whether authentication was successful
            error_details: Error details for failed authentication
        """
        try:
            async for session in get_db():
                # Create authentication event record
                auth_event = AuthenticationEvent(
                    user_email=user_email,
                    service_token_name=service_token_name,
                    event_type=event_type,
                    success=success,
                    ip_address=(
                        getattr(request.client, "host", None) if hasattr(request, "client") and request.client else None
                    ),
                    user_agent=request.headers.get("user-agent"),
                    cloudflare_ray_id=request.headers.get("cf-ray"),
                    error_details=error_details,
                    created_at=datetime.now(timezone.utc),  # noqa: UP017
                )

                session.add(auth_event)
                await session.commit()

                break  # Only need first session

        except Exception as e:
            # Don't fail authentication due to logging errors
            logger.warning(f"Failed to log authentication event: {e}")

    def _create_auth_error_response(self, error: AuthenticationError) -> JSONResponse:
        """Create JSON error response for authentication failures.

        Args:
            error: Authentication error

        Returns:
            JSON response with error details
        """
        content = {
            "error": "Authentication failed",
            "message": error.message if self.config.auth_error_detail_enabled else "Authentication required",
        }

        return JSONResponse(
            status_code=error.status_code,
            content=content,
        )

    async def _update_user_session(self, authenticated_user: AuthenticatedUser, request: Request) -> None:
        """Update user session in database with graceful degradation.

        Args:
            authenticated_user: Authenticated user information
            request: HTTP request for context
        """
        if not self.database_enabled:
            return

        try:
            db_manager = get_database_manager()
            async with db_manager.get_session() as session:

                # Extract Cloudflare subject from JWT claims
                cloudflare_sub = authenticated_user.jwt_claims.get("sub", "unknown")

                # Try to find existing session
                stmt = select(UserSession).where(UserSession.email == authenticated_user.email)
                result = await session.execute(stmt)
                existing_session = result.scalar_one_or_none()

                if existing_session:
                    # Update existing session
                    update_stmt = (
                        update(UserSession)
                        .where(
                            UserSession.email == authenticated_user.email,
                        )
                        .values(
                            session_count=UserSession.session_count + 1,
                            last_seen=func.now(),
                            cloudflare_sub=cloudflare_sub,
                        )
                    )
                    await session.execute(update_stmt)
                else:
                    # Create new session
                    new_session = UserSession(
                        email=authenticated_user.email,
                        cloudflare_sub=cloudflare_sub,
                        session_count=1,
                        preferences={},
                        user_metadata={},
                    )
                    session.add(new_session)

                await session.commit()

        except DatabaseError as e:
            # Log database errors but don't fail authentication
            logger.warning(f"Database session update failed (graceful degradation): {e}")
        except Exception as e:
            # Log unexpected errors but don't fail authentication
            logger.warning(f"Unexpected error updating session (graceful degradation): {e}")

    def _get_client_ip(self, request: Request) -> str | None:
        """Extract client IP address from request headers.

        Args:
            request: HTTP request

        Returns:
            Client IP address or None if not available
        """
        # Check Cloudflare headers first
        cf_connecting_ip = request.headers.get("CF-Connecting-IP")
        if cf_connecting_ip:
            return cf_connecting_ip

        # Check standard forwarded headers
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            # Take the first IP in the chain
            return x_forwarded_for.split(",")[0].strip()

        x_real_ip = request.headers.get("X-Real-IP")
        if x_real_ip:
            return x_real_ip

        # Fallback to client host
        if hasattr(request, "client") and request.client:
            return request.client.host

        return None


def create_rate_limiter(config: AuthenticationConfig) -> Limiter:
    """Create rate limiter for authentication endpoints.

    Args:
        config: Authentication configuration

    Returns:
        Configured Limiter instance
    """

    def get_rate_limit_key(request: Request) -> str:
        """Get rate limiting key based on configuration."""
        if config.rate_limit_key_func == RATE_LIMIT_KEY_IP:
            return get_remote_address(request)
        if config.rate_limit_key_func == RATE_LIMIT_KEY_EMAIL:
            # Use authenticated user email if available
            if hasattr(request.state, "user_email"):
                return request.state.user_email
            # Fallback to IP if no authenticated user
            return get_remote_address(request)
        if config.rate_limit_key_func == RATE_LIMIT_KEY_USER:
            # Use authenticated user email if available
            if hasattr(request.state, "authenticated_user"):
                return request.state.authenticated_user.email
            # Fallback to IP if no authenticated user
            return get_remote_address(request)
        # Default to IP-based rate limiting
        return get_remote_address(request)

    return Limiter(
        key_func=get_rate_limit_key,
        default_limits=[f"{config.rate_limit_requests}/{config.rate_limit_window}seconds"],
    )


def setup_authentication(
    app: FastAPI,
    config: AuthenticationConfig,
    database_enabled: bool = True,
) -> tuple[AuthenticationMiddleware, Limiter]:
    """Setup authentication middleware and rate limiting for FastAPI app.

    Args:
        app: FastAPI application instance
        config: Authentication configuration
        database_enabled: Whether database integration is enabled

    Returns:
        Tuple of (AuthenticationMiddleware, Limiter) instances
    """
    # Create JWKS client
    jwks_client = JWKSClient(
        jwks_url=config.get_jwks_url(),
        cache_ttl=config.jwks_cache_ttl,
        max_cache_size=config.jwks_cache_max_size,
        timeout=config.jwks_timeout,
    )

    # Create JWT validator
    jwt_validator = JWTValidator(
        jwks_client=jwks_client,
        config=config,
        audience=config.cloudflare_audience,
        issuer=config.cloudflare_issuer,
        algorithm=config.jwt_algorithm,
    )

    # Create authentication middleware
    auth_middleware = AuthenticationMiddleware(
        app=app,
        config=config,
        jwt_validator=jwt_validator,
        database_enabled=database_enabled,
    )

    # Create rate limiter
    limiter = create_rate_limiter(config)

    # Add middleware to app
    app.add_middleware(auth_middleware)

    if config.rate_limiting_enabled:
        app.add_middleware(SlowAPIMiddleware)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("Authentication middleware and rate limiting configured")

    return auth_middleware, limiter


def get_current_user(request: Request) -> AuthenticatedUser | None:
    """Get current authenticated user from request state.

    Args:
        request: HTTP request

    Returns:
        AuthenticatedUser if authenticated, None otherwise
    """
    # Handle case where request.state doesn't exist
    if not hasattr(request, "state"):
        return None
    return getattr(request.state, "authenticated_user", None)


def require_authentication(request: Request) -> AuthenticatedUser:
    """Require authentication and return current user.

    Args:
        request: HTTP request

    Returns:
        AuthenticatedUser

    Raises:
        HTTPException: If user is not authenticated
    """
    user = get_current_user(request)
    if not user:
        raise AuthExceptionHandler.handle_authentication_error()
    return user


def require_role(request: Request, required_role: str) -> AuthenticatedUser:
    """Require specific role and return current user.

    Args:
        request: HTTP request
        required_role: Required user role

    Returns:
        AuthenticatedUser

    Raises:
        HTTPException: If user doesn't have required role
    """
    user = require_authentication(request)
    if user.role.value != required_role:
        raise AuthExceptionHandler.handle_permission_error(
            user_identifier=user.email,
            detail=f"Role '{required_role}' required",
        )
    return user
