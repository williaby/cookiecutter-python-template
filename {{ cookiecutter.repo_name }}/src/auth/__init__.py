"""Authentication module for PromptCraft.

This module provides JWT-based authentication using Cloudflare Access tokens.
It includes:
- JWT token validation against Cloudflare JWKS endpoint
- Email-based user identification and role mapping
- FastAPI middleware for authentication
- Rate limiting to prevent DOS attacks
- Secure caching of JWKS with TTL support
"""

from .config import AuthenticationConfig
from .exceptions import AuthExceptionHandler
from .jwks_client import JWKSClient
from .jwt_validator import JWTValidator
from .middleware import (
    AuthenticationMiddleware,
    get_current_user,
    require_authentication,
    require_role,
    setup_authentication,
)
from .models import AuthenticatedUser, AuthenticationError, JWKSError, JWTValidationError, UserRole

__all__ = [
    "AuthExceptionHandler",
    "AuthenticatedUser",
    "AuthenticationConfig",
    "AuthenticationError",
    "AuthenticationMiddleware",
    "JWKSClient",
    "JWKSError",
    "JWTValidationError",
    "JWTValidator",
    "UserRole",
    "get_current_user",
    "require_authentication",
    "require_role",
    "setup_authentication",
]
