"""Authentication and authorization constants.

This module centralizes all permission, role, and authentication-related constants
to eliminate magic strings throughout the codebase and provide a single source
of truth for authorization configuration.
"""

from typing import Final

# User Roles
ROLE_ADMIN: Final[str] = "admin"
ROLE_USER: Final[str] = "user"

# Admin Role Prefixes (used for email-based role determination)
ADMIN_ROLE_PREFIXES: Final[list[str]] = ["admin", "administrator", "root", "superuser", "owner"]

# Permissions
PERMISSION_ADMIN: Final[str] = "admin"
PERMISSION_READ: Final[str] = "read"
PERMISSION_WRITE: Final[str] = "write"
PERMISSION_DELETE: Final[str] = "delete"
PERMISSION_CREATE: Final[str] = "create"
PERMISSION_UPDATE: Final[str] = "update"

# Rate Limiting Key Functions
RATE_LIMIT_KEY_IP: Final[str] = "ip"
RATE_LIMIT_KEY_EMAIL: Final[str] = "email"
RATE_LIMIT_KEY_USER: Final[str] = "user"

# Service Token Prefix
SERVICE_TOKEN_PREFIX: Final[str] = "sk_"

# Authentication Event Types
AUTH_EVENT_JWT: Final[str] = "jwt_auth"
AUTH_EVENT_SERVICE_TOKEN: Final[str] = "service_token_auth"
AUTH_EVENT_GENERAL: Final[str] = "auth"

# JWT Claim Names
JWT_CLAIM_EMAIL: Final[str] = "email"
JWT_CLAIM_SUB: Final[str] = "sub"
JWT_CLAIM_GROUPS: Final[str] = "groups"
JWT_CLAIM_EXP: Final[str] = "exp"
JWT_CLAIM_IAT: Final[str] = "iat"

# Permission Names for Authorization
PERMISSION_NAME_EMAIL_AUTHORIZATION: Final[str] = "email_authorization"
PERMISSION_NAME_ACCESS: Final[str] = "access"

# Error Codes
ERROR_CODE_TOKEN_NOT_FOUND: Final[str] = "token_not_found"
ERROR_CODE_TOKEN_INACTIVE: Final[str] = "token_inactive"
ERROR_CODE_TOKEN_EXPIRED: Final[str] = "token_expired"
ERROR_CODE_VALIDATION_EXCEPTION: Final[str] = "validation_exception"

# User Types
USER_TYPE_SERVICE_TOKEN: Final[str] = "service_token"
USER_TYPE_JWT: Final[str] = "jwt"
USER_TYPE_JWT_USER: Final[str] = "jwt_user"

# API Response Statuses
API_STATUS_SUCCESS: Final[str] = "success"
API_STATUS_NO_CHANGE: Final[str] = "no_change"

# System Health Statuses
HEALTH_STATUS_HEALTHY: Final[str] = "healthy"
HEALTH_STATUS_DEGRADED: Final[str] = "degraded"
HEALTH_STATUS_OPERATIONAL: Final[str] = "operational"

# Event/Log Statuses
EVENT_STATUS_LOGGED: Final[str] = "logged"
EVENT_STATUS_COMPLETED: Final[str] = "emergency_revocation_completed"

# Validation Response Fields
VALIDATION_FIELD_IS_VALID: Final[str] = "is_valid"
