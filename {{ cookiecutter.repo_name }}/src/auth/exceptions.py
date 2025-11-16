"""Authentication and authorization exception handling utilities.

This module provides standardized HTTP exception handling for AUTH-3 system
to eliminate duplicate exception handling patterns across API endpoints.

Key Features:
- Standardized exception mapping from domain errors to HTTP responses
- Consistent error message formatting and logging
- Security-aware error details (prevents information leakage)
- FastAPI integration with proper status codes and headers
- Audit logging integration for security events
"""

import logging

from fastapi import HTTPException, status

from src.auth.role_manager import (
    PermissionNotFoundError,
    RoleManagerError,
    RoleNotFoundError,
    UserNotFoundError,
)

logger = logging.getLogger(__name__)


class AuthExceptionHandler:
    """Standardized exception handler for authentication and authorization operations.

    Provides consistent HTTP exception handling across the AUTH-3 system,
    eliminating duplicate exception handling patterns in API endpoints.

    This class follows the fail-secure principle: when in doubt, deny access
    and provide minimal error information to prevent information disclosure.
    """

    @staticmethod
    def handle_authentication_error(
        detail: str = "Authentication required",
        log_message: str = "",
        user_identifier: str = "",
    ) -> HTTPException:
        """Handle authentication failures (401 Unauthorized).

        Args:
            detail: User-facing error message
            log_message: Internal log message (optional)
            user_identifier: User/token identifier for logging

        Returns:
            HTTPException with 401 status code
        """
        log_msg = log_message or detail
        if user_identifier:
            log_msg = f"{log_msg} - user: {user_identifier}"

        logger.warning(f"Authentication failed: {log_msg}")

        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @staticmethod
    def handle_permission_error(
        permission_name: str = "",
        user_identifier: str = "",
        detail: str = "",
    ) -> HTTPException:
        """Handle permission/authorization failures (403 Forbidden).

        Args:
            permission_name: Name of the required permission
            user_identifier: User/token identifier for logging
            detail: Custom error detail (optional)

        Returns:
            HTTPException with 403 status code
        """
        if not detail:
            if permission_name:
                detail = f"Insufficient permissions: {permission_name} required"
            else:
                detail = "Insufficient permissions"

        log_msg = f"Permission denied: {detail}"
        if user_identifier:
            log_msg = f"{log_msg} - user: {user_identifier}"

        logger.warning(log_msg)

        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )

    @staticmethod
    def handle_not_found_error(
        entity_type: str,
        entity_identifier: str = "",
        detail: str = "",
    ) -> HTTPException:
        """Handle entity not found errors (404 Not Found).

        Args:
            entity_type: Type of entity (role, user, permission, etc.)
            entity_identifier: Identifier of the entity
            detail: Custom error detail (optional)

        Returns:
            HTTPException with 404 status code
        """
        if not detail:
            if entity_identifier:
                detail = f"{entity_type.title()} '{entity_identifier}' not found"
            else:
                detail = f"{entity_type.title()} not found"

        logger.info(f"Entity not found: {detail}")

        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )

    @staticmethod
    def handle_validation_error(
        detail: str,
        field_name: str = "",
        log_additional: str = "",
    ) -> HTTPException:
        """Handle input validation errors (400 Bad Request).

        Args:
            detail: User-facing validation error message
            field_name: Name of the field that failed validation
            log_additional: Additional information for logging

        Returns:
            HTTPException with 400 status code
        """
        log_msg = f"Validation error: {detail}"
        if field_name:
            log_msg = f"{log_msg} - field: {field_name}"
        if log_additional:
            log_msg = f"{log_msg} - {log_additional}"

        logger.info(log_msg)

        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

    @staticmethod
    def handle_conflict_error(
        detail: str,
        entity_type: str = "",
        entity_identifier: str = "",
    ) -> HTTPException:
        """Handle resource conflict errors (409 Conflict).

        Args:
            detail: User-facing conflict error message
            entity_type: Type of conflicting entity
            entity_identifier: Identifier of conflicting entity

        Returns:
            HTTPException with 409 status code
        """
        log_msg = f"Conflict error: {detail}"
        if entity_type and entity_identifier:
            log_msg = f"{log_msg} - {entity_type}: {entity_identifier}"

        logger.info(log_msg)

        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        )

    @staticmethod
    def handle_rate_limit_error(
        retry_after: int = 60,
        detail: str = "Too many requests",
        client_identifier: str = "",
    ) -> HTTPException:
        """Handle rate limiting errors (429 Too Many Requests).

        Args:
            retry_after: Seconds until client can retry
            detail: User-facing rate limit message
            client_identifier: Client identifier for logging

        Returns:
            HTTPException with 429 status code
        """
        log_msg = f"Rate limit exceeded: {detail}"
        if client_identifier:
            log_msg = f"{log_msg} - client: {client_identifier}"

        logger.warning(log_msg)

        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
        )

    @staticmethod
    def handle_internal_error(
        operation_name: str,
        error: Exception,
        detail: str = "Internal server error",
        expose_error: bool = False,
    ) -> HTTPException:
        """Handle internal server errors (500 Internal Server Error).

        Args:
            operation_name: Name of the operation that failed
            error: Original exception
            detail: User-facing error message
            expose_error: Whether to expose error details (dev only)

        Returns:
            HTTPException with 500 status code
        """
        logger.error(f"{operation_name} failed: {error}", exc_info=True)

        # In production, don't expose internal error details
        if expose_error:
            detail = f"{detail}: {error!s}"

        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )

    @classmethod
    def handle_role_manager_error(cls, error: Exception) -> HTTPException:
        """Handle role manager domain exceptions with appropriate HTTP mapping.

        Maps domain-specific exceptions to appropriate HTTP status codes:
        - RoleNotFoundError -> 404 Not Found
        - UserNotFoundError -> 404 Not Found
        - PermissionNotFoundError -> 404 Not Found
        - RoleManagerError (validation) -> 400 Bad Request
        - Other exceptions -> 500 Internal Server Error

        Args:
            error: Domain exception from role manager

        Returns:
            HTTPException with appropriate status code
        """
        if isinstance(error, RoleNotFoundError):
            return cls.handle_not_found_error("role", str(error).split("'")[1] if "'" in str(error) else "")

        if isinstance(error, UserNotFoundError):
            return cls.handle_not_found_error("user", str(error).split("'")[1] if "'" in str(error) else "")

        if isinstance(error, PermissionNotFoundError):
            return cls.handle_not_found_error("permission", str(error).split("'")[1] if "'" in str(error) else "")

        if isinstance(error, RoleManagerError):
            # Check for specific validation errors
            error_str = str(error).lower()
            if "already exists" in error_str:
                return cls.handle_conflict_error(str(error))
            if (
                "circular" in error_str
                or "hierarchy" in error_str
                or "dependencies" in error_str
                or "assigned to" in error_str
            ):
                return cls.handle_validation_error(str(error))
            return cls.handle_validation_error(str(error))

        return cls.handle_internal_error("Role management operation", error)

    @classmethod
    def handle_service_unavailable(
        cls,
        service_name: str,
        detail: str = "",
        retry_after: int = 60,
    ) -> HTTPException:
        """Handle service unavailable errors (503 Service Unavailable).

        Args:
            service_name: Name of the unavailable service
            detail: Custom error detail
            retry_after: Seconds until service might be available

        Returns:
            HTTPException with 503 status code
        """
        if not detail:
            detail = f"Service temporarily unavailable: {service_name}"

        logger.error(f"Service unavailable: {service_name}")

        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
        )


# Convenience functions for common exception patterns
def authentication_required(user_identifier: str = "") -> HTTPException:
    """Shorthand for authentication required error."""
    return AuthExceptionHandler.handle_authentication_error(
        detail="Authentication required",
        user_identifier=user_identifier,
    )


def permission_denied(permission_name: str, user_identifier: str = "") -> HTTPException:
    """Shorthand for permission denied error."""
    return AuthExceptionHandler.handle_permission_error(
        permission_name=permission_name,
        user_identifier=user_identifier,
    )


def role_not_found(role_name: str) -> HTTPException:
    """Shorthand for role not found error."""
    return AuthExceptionHandler.handle_not_found_error("role", role_name)


def user_not_found(user_email: str) -> HTTPException:
    """Shorthand for user not found error."""
    return AuthExceptionHandler.handle_not_found_error("user", user_email)


def permission_not_found(permission_name: str) -> HTTPException:
    """Shorthand for permission not found error."""
    return AuthExceptionHandler.handle_not_found_error("permission", permission_name)


def validation_failed(detail: str, field_name: str = "") -> HTTPException:
    """Shorthand for validation error."""
    return AuthExceptionHandler.handle_validation_error(detail, field_name)


def already_exists(entity_type: str, entity_name: str) -> HTTPException:
    """Shorthand for conflict/already exists error."""
    return AuthExceptionHandler.handle_conflict_error(
        f"{entity_type.title()} '{entity_name}' already exists",
        entity_type,
        entity_name,
    )


def operation_failed(operation_name: str, error: Exception, expose_details: bool = False) -> HTTPException:
    """Shorthand for operation failure error."""
    return AuthExceptionHandler.handle_internal_error(
        operation_name,
        error,
        expose_error=expose_details,
    )


# Export public interface
__all__ = [
    "AuthExceptionHandler",
    "already_exists",
    "authentication_required",
    "operation_failed",
    "permission_denied",
    "permission_not_found",
    "role_not_found",
    "user_not_found",
    "validation_failed",
]
