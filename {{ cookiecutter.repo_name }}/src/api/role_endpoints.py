"""Role and Permission Management API endpoints for AUTH-3.

This module provides FastAPI endpoints for role and permission management including:
- CRUD operations for roles and permissions
- Role assignment and revocation for users
- Permission assignment to roles
- Role hierarchy management and validation
- User permission queries and analytics
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.auth.exceptions import AuthExceptionHandler
from src.auth.permissions import Permissions, require_permission
from src.auth.role_manager import (
    PermissionNotFoundError,
    RoleManager,
    RoleManagerError,
    RoleNotFoundError,
    UserNotFoundError,
)
from src.auth.types import AuthenticatedUserType

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class RoleCreateRequest(BaseModel):
    """Request model for creating roles."""

    name: str = Field(..., min_length=1, description="Unique role name (lowercase, underscore-separated)")
    description: str | None = Field(None, description="Human-readable description of the role")
    parent_role_name: str | None = Field(None, description="Parent role name for inheritance")


class RoleResponse(BaseModel):
    """Response model for role information."""

    id: int = Field(..., description="Role ID")
    name: str = Field(..., description="Role name")
    description: str | None = Field(None, description="Role description")
    parent_role_id: int | None = Field(None, description="Parent role ID")
    parent_role_name: str | None = Field(None, description="Parent role name")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    is_active: bool = Field(..., description="Whether role is active")


class PermissionAssignmentRequest(BaseModel):
    """Request model for assigning permissions to roles."""

    role_name: str = Field(..., description="Name of the role")
    permission_name: str = Field(..., description="Name of the permission to assign")


class UserRoleAssignmentRequest(BaseModel):
    """Request model for assigning roles to users."""

    user_email: str = Field(..., description="Email address of the user")
    role_name: str = Field(..., description="Name of the role to assign")


class UserRoleResponse(BaseModel):
    """Response model for user role assignments."""

    role_id: int = Field(..., description="Role ID")
    role_name: str = Field(..., description="Role name")
    role_description: str | None = Field(None, description="Role description")
    assigned_at: datetime = Field(..., description="Assignment timestamp")


class UserPermissionsResponse(BaseModel):
    """Response model for user permissions summary."""

    user_email: str = Field(..., description="User email")
    roles: list[UserRoleResponse] = Field(..., description="Assigned roles")
    permissions: list[str] = Field(..., description="All permissions (including inherited)")


class RolePermissionsResponse(BaseModel):
    """Response model for role permissions."""

    role_name: str = Field(..., description="Role name")
    permissions: list[str] = Field(..., description="All permissions (including inherited)")


# =============================================================================
# ROUTER SETUP
# =============================================================================

role_router = APIRouter(prefix="/api/v1/roles", tags=["role-management"])


# =============================================================================
# ROLE MANAGEMENT ENDPOINTS
# =============================================================================


@role_router.post("/", response_model=RoleResponse)
async def create_role(
    request: Request,  # noqa: ARG001
    role_request: RoleCreateRequest,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_CREATE)),
) -> RoleResponse:
    """Create a new role (admin only).

    This endpoint allows administrators to create new roles with optional
    parent role for hierarchical inheritance.
    """
    manager = RoleManager()

    try:
        # Validate role hierarchy if parent specified
        if role_request.parent_role_name:
            # Check if parent role exists by trying to get it
            parent_role = await manager.get_role(role_request.parent_role_name)
            if not parent_role:
                raise HTTPException(
                    status_code=400,
                    detail=f"Parent role '{role_request.parent_role_name}' does not exist",
                )

        role_data = await manager.create_role(
            name=role_request.name,
            description=role_request.description,
            parent_role_name=role_request.parent_role_name,
        )

        return RoleResponse(**role_data)

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Role creation", e, expose_error=True) from e


@role_router.get("/", response_model=list[RoleResponse])
async def list_roles(
    request: Request,  # noqa: ARG001
    include_inactive: bool = Query(False, description="Include inactive roles"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_READ)),
) -> list[RoleResponse]:
    """List all roles (admin only).

    This endpoint returns information about all roles in the system.
    """
    manager = RoleManager()

    try:
        roles_data = await manager.list_roles(include_inactive=include_inactive)
        return [RoleResponse(**role_data) for role_data in roles_data]

    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("List roles", e, expose_error=True) from e


# =============================================================================
# USER ROLE ASSIGNMENT ENDPOINTS (Must come before /{role_name} routes)
# =============================================================================


@role_router.post("/assignments")
async def assign_user_role(
    request: Request,  # noqa: ARG001
    assignment_request: UserRoleAssignmentRequest,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_ASSIGN)),
) -> dict[str, str]:
    """Assign a role to a user (admin only)."""
    manager = RoleManager()

    try:
        assigned_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        success = await manager.assign_user_role(
            user_email=assignment_request.user_email,
            role_name=assignment_request.role_name,
            assigned_by=assigned_by,
        )

        if success:
            return {
                "status": "success",
                "message": f"Role '{assignment_request.role_name}' assigned to user '{assignment_request.user_email}'",
                "assigned_by": assigned_by,
            }
        raise AuthExceptionHandler.handle_internal_error("Role assignment", ValueError("Assignment failed"))

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Assign role", e, expose_error=True) from e


@role_router.delete("/assignments")
async def revoke_user_role(
    request: Request,  # noqa: ARG001
    user_email: str = Query(..., description="User email address"),
    role_name: str = Query(..., description="Role name to revoke"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_ASSIGN)),
) -> dict[str, str]:
    """Revoke a role from a user (admin only)."""
    manager = RoleManager()

    try:
        revoked_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        success = await manager.revoke_user_role(user_email=user_email, role_name=role_name)

        if success:
            return {
                "status": "success",
                "message": f"Role '{role_name}' revoked from user '{user_email}'",
                "revoked_by": revoked_by,
            }
        return {
            "status": "no_change",
            "message": f"Role '{role_name}' was not assigned to user '{user_email}'",
            "revoked_by": revoked_by,
        }

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Revoke role", e, expose_error=True) from e


# =============================================================================
# ROLE HIERARCHY VALIDATION ENDPOINTS (Must come before /{role_name} routes)
# =============================================================================


@role_router.post("/validate-hierarchy")
async def validate_role_hierarchy(
    request: Request,  # noqa: ARG001
    role_name: str = Query(..., description="Role name to modify"),
    parent_role_name: str = Query(..., description="Proposed parent role name"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_READ)),
) -> dict[str, bool | str]:
    """Validate that adding a parent role won't create a circular dependency (admin only)."""
    manager = RoleManager()

    try:
        is_valid = await manager.validate_role_hierarchy(role_name, parent_role_name)

        return {
            "is_valid": is_valid,
            "message": (
                f"Setting '{parent_role_name}' as parent of '{role_name}' is valid"
                if is_valid
                else f"Setting '{parent_role_name}' as parent of '{role_name}' would create a circular dependency"
            ),
        }

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Validate hierarchy", e, expose_error=True) from e


# =============================================================================
# USER PERMISSION QUERY ENDPOINTS (Must come before /{role_name} routes)
# =============================================================================


@role_router.get("/users/{user_email}/roles", response_model=list[UserRoleResponse])
async def get_user_roles(
    request: Request,  # noqa: ARG001
    user_email: str,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.USERS_READ)),
) -> list[UserRoleResponse]:
    """Get all roles assigned to a user (admin only)."""
    manager = RoleManager()

    try:
        roles_data = await manager.get_user_roles(user_email)
        return [UserRoleResponse(**role_data) for role_data in roles_data]

    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Get user roles", e, expose_error=True) from e


@role_router.get("/users/{user_email}/permissions", response_model=UserPermissionsResponse)
async def get_user_permissions(
    request: Request,  # noqa: ARG001
    user_email: str,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.USERS_READ)),
) -> UserPermissionsResponse:
    """Get all permissions for a user through their assigned roles (admin only)."""
    manager = RoleManager()

    try:
        # Get user roles
        roles_data = await manager.get_user_roles(user_email)
        user_roles = [UserRoleResponse(**role_data) for role_data in roles_data]

        # Get user permissions
        permissions = await manager.get_user_permissions(user_email)

        return UserPermissionsResponse(
            user_email=user_email,
            roles=user_roles,
            permissions=list(permissions),
        )

    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Get user permissions", e, expose_error=True) from e


@role_router.get("/{role_name}", response_model=RoleResponse)
async def get_role(
    request: Request,  # noqa: ARG001
    role_name: str,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_READ)),
) -> RoleResponse:
    """Get role information by name (admin only)."""
    manager = RoleManager()

    try:
        role_data = await manager.get_role(role_name)
        if not role_data:
            raise AuthExceptionHandler.handle_not_found_error("role", role_name)

        return RoleResponse(**role_data)

    except HTTPException:
        # Re-raise HTTPException as-is (404, 403, etc.)
        raise
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Get role", e, expose_error=True) from e


@role_router.delete("/{role_name}")
async def delete_role(
    request: Request,  # noqa: ARG001
    role_name: str,
    force: bool = Query(False, description="Force deletion (removes assignments and child dependencies)"),
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_DELETE)),
) -> dict[str, str]:
    """Delete a role (admin only).

    This endpoint performs a soft delete by setting is_active=False.
    Use force=True to also remove user assignments and child role dependencies.
    """
    manager = RoleManager()

    try:
        deleted_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        success = await manager.delete_role(role_name, force=force)

        if success:
            return {
                "status": "success",
                "message": f"Role '{role_name}' has been deleted",
                "deleted_by": deleted_by,
                "force": str(force),
            }
        raise AuthExceptionHandler.handle_not_found_error("role", role_name)

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except HTTPException:
        # Re-raise HTTPException as-is (404, 403, etc.)
        raise
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Delete role", e, expose_error=True) from e


# =============================================================================
# ROLE PERMISSION MANAGEMENT ENDPOINTS
# =============================================================================


@role_router.get("/{role_name}/permissions", response_model=RolePermissionsResponse)
async def get_role_permissions(
    request: Request,  # noqa: ARG001
    role_name: str,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_READ)),
) -> RolePermissionsResponse:
    """Get all permissions for a role including inherited permissions (admin only)."""
    manager = RoleManager()

    try:
        permissions = await manager.get_role_permissions(role_name)
        return RolePermissionsResponse(
            role_name=role_name,
            permissions=list(permissions),
        )

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Get role permissions", e, expose_error=True) from e


@role_router.post("/{role_name}/permissions")
async def assign_permission_to_role(
    request: Request,  # noqa: ARG001
    role_name: str,
    permission_request: PermissionAssignmentRequest,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_UPDATE)),
) -> dict[str, str]:
    """Assign a permission to a role (admin only)."""
    manager = RoleManager()

    try:
        assigned_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        success = await manager.assign_permission_to_role(role_name, permission_request.permission_name)

        if success:
            return {
                "status": "success",
                "message": f"Permission '{permission_request.permission_name}' assigned to role '{role_name}'",
                "assigned_by": assigned_by,
            }
        raise AuthExceptionHandler.handle_internal_error("Permission assignment", ValueError("Assignment failed"))

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Assign permission", e, expose_error=True) from e


@role_router.delete("/{role_name}/permissions/{permission_name}")
async def revoke_permission_from_role(
    request: Request,  # noqa: ARG001
    role_name: str,
    permission_name: str,
    current_user: AuthenticatedUserType = Depends(require_permission(Permissions.ROLES_UPDATE)),
) -> dict[str, str]:
    """Revoke a permission from a role (admin only)."""
    manager = RoleManager()

    try:
        revoked_by = getattr(current_user, "email", None) or getattr(current_user, "token_name", "unknown")
        success = await manager.revoke_permission_from_role(role_name, permission_name)

        if success:
            return {
                "status": "success",
                "message": f"Permission '{permission_name}' revoked from role '{role_name}'",
                "revoked_by": revoked_by,
            }
        raise AuthExceptionHandler.handle_internal_error("Permission revocation", ValueError("Revocation failed"))

    except (RoleManagerError, RoleNotFoundError, PermissionNotFoundError, UserNotFoundError) as e:
        raise AuthExceptionHandler.handle_role_manager_error(e) from e
    except Exception as e:
        raise AuthExceptionHandler.handle_internal_error("Revoke permission", e, expose_error=True) from e


