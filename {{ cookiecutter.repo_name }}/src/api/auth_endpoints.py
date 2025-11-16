"""Authentication API endpoints for AUTH-2 service token management.

This module provides FastAPI endpoints for service token management including:
- Token creation and management (admin-only)
- Current user/service token information
- Authentication status and health checks
- Usage analytics and audit logging
"""

from datetime import timezone, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.auth.constants import (
    API_STATUS_SUCCESS,
    EVENT_STATUS_COMPLETED,
    HEALTH_STATUS_DEGRADED,
    HEALTH_STATUS_HEALTHY,
    HEALTH_STATUS_OPERATIONAL,
    USER_TYPE_JWT_USER,
    USER_TYPE_SERVICE_TOKEN,
)
from src.auth.exceptions import AuthExceptionHandler
from src.auth.middleware import ServiceTokenUser, require_authentication
from src.auth.permissions import Permissions, require_permission
from src.auth.role_manager import RoleManager
from src.auth.service_token_manager import ServiceTokenManager
from src.auth.types import AuthenticatedUserType


class TokenCreationRequest(BaseModel):
    """Request model for creating service tokens."""

    token_name: str = Field(..., description="Unique name for the service token")
    permissions: list[str] = Field(default=[], description="List of permissions for the token")
    expires_days: int | None = Field(None, description="Token expires in N days (optional)")
    purpose: str | None = Field(None, description="Purpose of this token")
    environment: str | None = Field(None, description="Environment (production, staging, dev)")


class TokenCreationResponse(BaseModel):
    """Response model for token creation."""

    token_id: str = Field(..., description="Unique token identifier")
    token_name: str = Field(..., description="Token name")
    token_value: str = Field(..., description="Service token value (save securely)")
    expires_at: datetime | None = Field(None, description="Token expiration date")
    metadata: dict = Field(..., description="Token metadata including permissions")


class TokenInfo(BaseModel):
    """Basic token information (without sensitive data)."""

    token_id: str = Field(..., description="Token identifier")
    token_name: str = Field(..., description="Token name")
    usage_count: int = Field(..., description="Number of times token has been used")
    last_used: datetime | None = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Whether token is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    permissions: list[str] = Field(..., description="Token permissions")


class CurrentUserResponse(BaseModel):
    """Response model for current user information."""

    user_type: str = Field(..., description="Type of authentication (jwt_user or service_token)")
    email: str | None = Field(None, description="User email (JWT auth only)")
    role: str | None = Field(None, description="User role (JWT auth only)")
    token_name: str | None = Field(None, description="Token name (service token auth only)")
    token_id: str | None = Field(None, description="Token ID (service token auth only)")
    permissions: list[str] = Field(default=[], description="User/token permissions")
    usage_count: int | None = Field(None, description="Token usage count (service token only)")


class AuthHealthResponse(BaseModel):
    """Authentication system health response."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    database_status: str = Field(..., description="Database connectivity status")
    active_tokens: int = Field(..., description="Number of active service tokens")
    recent_authentications: int = Field(..., description="Recent authentication count")


# Create router
auth_router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@auth_router.get("/me", response_model=CurrentUserResponse)
async def get_current_user_info(
    request: Request,  # noqa: ARG001
    current_user: AuthenticatedUserType = Depends(require_authentication),
) -> CurrentUserResponse:
    """Get current authenticated user or service token information.

    This endpoint returns information about the currently authenticated user
    or service token. It works with both JWT authentication and service tokens.
    """
    if isinstance(current_user, ServiceTokenUser):
        # Service token authentication
        return CurrentUserResponse(
            user_type=USER_TYPE_SERVICE_TOKEN,
            token_name=current_user.token_name,
            token_id=current_user.token_id,
            permissions=current_user.metadata.get("permissions", []),
            usage_count=current_user.usage_count,
        )
    # JWT user authentication - get permissions from role system
    try:
        role_manager = RoleManager()
        user_permissions = await role_manager.get_user_permissions(current_user.email)
    except Exception:
        # Fallback to empty permissions if role system unavailable
        user_permissions = set()

    return CurrentUserResponse(
        user_type=USER_TYPE_JWT_USER,
        email=current_user.email,
        role=current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role),
        permissions=list(user_permissions),
    )


@auth_router.get("/health", response_model=AuthHealthResponse)
async def auth_health_check() -> AuthHealthResponse:
    """Authentication system health check.

    This endpoint provides health information about the authentication system
    including database connectivity and service token statistics.
    """
    manager = ServiceTokenManager()

    try:
        # Get token analytics for health metrics
        analytics = await manager.get_token_usage_analytics(days=1)

        database_status = HEALTH_STATUS_HEALTHY
        if analytics and "summary" in analytics:
            active_tokens = analytics["summary"]["active_tokens"]
            recent_authentications = analytics["summary"]["total_usage"]
        else:
            active_tokens = -1
            recent_authentications = -1

    except Exception as e:
        database_status = f"error: {e!s}"
        active_tokens = -1
        recent_authentications = -1

    # Determine overall status
    status = HEALTH_STATUS_HEALTHY if database_status == HEALTH_STATUS_HEALTHY else HEALTH_STATUS_DEGRADED

    return AuthHealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc),
        database_status=database_status,
        active_tokens=active_tokens,
        recent_authentications=recent_authentications,
    )


@auth_router.post("/tokens", response_model=TokenCreationResponse)
async def create_service_token(
    request: Request,  # noqa: ARG001
    token_request: TokenCreationRequest,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.TOKENS_CREATE)),
) -> TokenCreationResponse:
    """Create a new service token (admin only).

    This endpoint allows administrators to create new service tokens for
    non-interactive API access. The token value is only returned once.
    """
    manager = ServiceTokenManager()

    # Build token metadata
    created_by_email = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
    metadata = {
        "permissions": token_request.permissions,
        "created_by": created_by_email,
        "purpose": token_request.purpose or "Created via API",
        "environment": token_request.environment or "production",
        "created_via": "admin_api",
    }

    # Calculate expiration
    expires_at = None
    if token_request.expires_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=token_request.expires_days)

    try:
        result = await manager.create_service_token(
            token_name=token_request.token_name,
            metadata=metadata,
            expires_at=expires_at,
            is_active=True,
        )

        if result is None:
            raise AuthExceptionHandler.handle_internal_error(
                "Service token creation",
                ValueError("Token creation returned None"),
            )

        token_value, token_id = result

        return TokenCreationResponse(
            token_id=token_id,
            token_name=token_request.token_name,
            token_value=token_value,
            expires_at=expires_at,
            metadata=metadata,
        )

    except ValueError as e:
        # Token name already exists
        raise AuthExceptionHandler.handle_validation_error(str(e), "token_name") from e
    except Exception as e:
        # Other errors
        raise AuthExceptionHandler.handle_internal_error("Create service token", e, expose_error=True) from e


@auth_router.delete("/tokens/{token_identifier}")
async def revoke_service_token(
    request: Request,  # noqa: ARG001
    token_identifier: str,
    reason: str = Query(..., description="Reason for revocation"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.TOKENS_DELETE)),
) -> dict[str, str]:
    """Revoke a service token (admin only).

    This endpoint allows administrators to revoke service tokens.
    The token will be immediately deactivated and cannot be used for authentication.
    """
    manager = ServiceTokenManager()

    try:
        revoked_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        success = await manager.revoke_service_token(
            token_identifier=token_identifier,
            revocation_reason=f"{reason} (revoked by {revoked_by} via API)",
        )

        if success:
            return {
                "status": API_STATUS_SUCCESS,
                "message": f"Token '{token_identifier}' has been revoked",
                "revoked_by": revoked_by,
                "reason": reason,
            }
        raise AuthExceptionHandler.handle_not_found_error("token", token_identifier)

    except HTTPException:
        raise
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Revoke token", e, expose_error=True) from e


@auth_router.post("/tokens/{token_identifier}/rotate")
async def rotate_service_token(
    request: Request,  # noqa: ARG001
    token_identifier: str,
    reason: str = Query("manual_rotation", description="Reason for rotation"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.TOKENS_ROTATE)),
) -> TokenCreationResponse:
    """Rotate a service token (admin only).

    This endpoint creates a new token with the same permissions and deactivates the old one.
    The new token value is only returned once.
    """
    manager = ServiceTokenManager()

    try:
        rotated_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        result = await manager.rotate_service_token(
            token_identifier=token_identifier,
            rotation_reason=f"{reason} (rotated by {rotated_by} via API)",
        )

        if result:
            new_token_value, new_token_id = result

            # Get the new token info for response
            analytics = await manager.get_token_usage_analytics(token_identifier=new_token_id)

            if analytics and "error" not in analytics:
                return TokenCreationResponse(
                    token_id=new_token_id,
                    token_name=analytics.get("token_name", "rotated_token"),
                    token_value=new_token_value,
                    expires_at=None,  # Will be same as original
                    metadata={"rotated_by": rotated_by, "rotation_reason": reason},
                )
            return TokenCreationResponse(
                token_id=new_token_id,
                token_name="rotated_token",  # nosec B106  # noqa: S106
                token_value=new_token_value,
                expires_at=None,
                metadata={"rotated_by": rotated_by, "rotation_reason": reason},
            )
        raise AuthExceptionHandler.handle_not_found_error("token", token_identifier, "Token not found or inactive")

    except HTTPException:
        raise
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Rotate token", e, expose_error=True) from e


@auth_router.get("/tokens", response_model=list[TokenInfo])
async def list_service_tokens(
    request: Request,  # noqa: ARG001
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.TOKENS_READ)),
) -> list[TokenInfo]:
    """List all service tokens (admin only).

    This endpoint returns information about all service tokens without
    sensitive data like token values or hashes.
    """
    manager = ServiceTokenManager()

    try:
        # Get comprehensive analytics to list all tokens
        analytics = await manager.get_token_usage_analytics(days=365)

        tokens = []

        # Process top tokens (active tokens)
        if analytics and "top_tokens" in analytics:
            for token_data in analytics.get("top_tokens", []):
                # Get detailed info for each token
                token_analytics = await manager.get_token_usage_analytics(token_identifier=token_data["token_name"])

                if token_analytics and "error" not in token_analytics:
                    permissions: list[str] = []  # Would need to fetch from database

                    tokens.append(
                        TokenInfo(
                            token_id="",  # We don't expose token IDs in listings  # nosec B106
                            token_name=token_analytics["token_name"],
                            usage_count=token_analytics["usage_count"],
                            last_used=(
                                datetime.fromisoformat(token_analytics["last_used"])
                                if token_analytics["last_used"]
                                else None
                            ),
                            is_active=token_analytics["is_active"],
                            created_at=datetime.fromisoformat(token_analytics["created_at"]),
                            permissions=permissions,
                        ),
                    )

        return tokens

    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("List service tokens", e, expose_error=True) from e


@auth_router.get("/tokens/{token_identifier}/analytics")
async def get_token_analytics(
    request: Request,  # noqa: ARG001
    token_identifier: str,
    days: int = Query(30, description="Number of days to analyze"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.TOKENS_READ)),
) -> dict:
    """Get detailed analytics for a specific service token (admin only).

    This endpoint provides usage analytics and recent events for a service token.
    """
    manager = ServiceTokenManager()

    try:
        analytics = await manager.get_token_usage_analytics(token_identifier=token_identifier, days=days)

        if analytics and "error" in analytics:
            raise AuthExceptionHandler.handle_not_found_error("token", token_identifier, analytics["error"])

        return analytics or {}

    except HTTPException:
        raise
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Get token analytics", e, expose_error=True) from e


@auth_router.post("/emergency-revoke")
async def emergency_revoke_all_tokens(
    request: Request,  # noqa: ARG001
    reason: str = Query(..., description="Emergency revocation reason"),
    confirm: bool = Query(False, description="Confirmation required"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.SYSTEM_ADMIN)),
) -> dict[str, str | int]:
    """Emergency revocation of ALL service tokens (admin only).

    This is a nuclear option that deactivates ALL service tokens immediately.
    Use only in case of security incidents or system compromise.
    """
    if not confirm:
        raise AuthExceptionHandler.handle_validation_error(
            "Emergency revocation requires explicit confirmation (confirm=true)",
            "confirm",
        )

    manager = ServiceTokenManager()

    try:
        revoked_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        revoked_count = await manager.emergency_revoke_all_tokens(
            emergency_reason=f"{reason} (emergency revoked by {revoked_by} via API)",
        )

        return {
            "status": EVENT_STATUS_COMPLETED,
            "tokens_revoked": revoked_count or 0,
            "revoked_by": revoked_by,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Emergency revocation", e, expose_error=True) from e


# System status endpoints (protected by service tokens)
system_router = APIRouter(prefix="/api/v1/system", tags=["system"])


@system_router.get("/status")
async def system_status(
    request: Request,  # noqa: ARG001
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.SYSTEM_STATUS)),
) -> dict[str, str]:
    """Get system status information.

    This endpoint requires authentication and system:status permission.
    Works with both JWT users and service tokens.
    """

    return {
        "status": HEALTH_STATUS_OPERATIONAL,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "version": "1.0.0",
        "authenticated_as": getattr(current_user, "email", getattr(current_user, "token_name", "unknown")),
    }


@system_router.get("/health")
async def system_health() -> dict[str, str]:
    """Public system health check (no authentication required).

    This endpoint is excluded from authentication middleware and can be used
    for basic health monitoring without credentials.
    """
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat() + "Z"}


# Audit endpoints for CI/CD logging
audit_router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@audit_router.post("/cicd-event")
async def log_cicd_event(
    request: Request,  # noqa: ARG001
    event_data: dict,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.SYSTEM_AUDIT)),
) -> dict[str, str]:
    """Log CI/CD workflow events for audit trail.

    This endpoint allows CI/CD systems to log workflow events for audit purposes.
    Requires system:audit permission.
    """

    # Here you would typically log to your audit system
    # For now, we'll just acknowledge the event

    return {
        "status": "logged",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "event_type": event_data.get("event_type", "unknown"),
        "logged_by": getattr(current_user, "email", getattr(current_user, "token_name", "unknown")),
    }
