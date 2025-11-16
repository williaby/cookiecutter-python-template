"""Authentication models for PromptCraft."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator

from .constants import ROLE_ADMIN, ROLE_USER


class UserRole(str, Enum):
    """User roles for role-based access control."""

    ADMIN = ROLE_ADMIN
    USER = ROLE_USER
    VIEWER = "viewer"  # Not centralized yet, keeping as is


class AuthenticatedUser(BaseModel):
    """Authenticated user model from JWT token."""

    email: str = Field(..., description="User email from JWT claims")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    jwt_claims: dict[str, Any] = Field(..., description="All JWT claims")

    @property
    def user_id(self) -> str | None:
        """Get user ID from JWT 'sub' claim."""
        return self.jwt_claims.get("sub")

    class Config:
        """Pydantic configuration."""

        # Allow arbitrary types for JWT claims
        arbitrary_types_allowed = True


class JWTValidationError(Exception):
    """Exception raised during JWT validation."""

    def __init__(self, message: str, error_type: str = "validation_error") -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class JWKSError(Exception):
    """Exception raised during JWKS operations."""

    def __init__(self, message: str, error_type: str = "jwks_error") -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class AuthenticationError(Exception):
    """Exception raised during authentication."""

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Service Token Pydantic Models


class ServiceTokenCreate(BaseModel):
    """Service token creation model."""

    token_name: str = Field(..., min_length=1, max_length=255, description="Human-readable token name")
    is_active: bool = Field(default=True, description="Whether token is active")
    expires_at: datetime | None = Field(default=None, description="Token expiration datetime")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional token metadata")

    @validator("token_name")
    def validate_token_name(cls, v: str) -> str:  # noqa: N805
        """Validate and clean token name."""
        return v.strip()


class ServiceTokenUpdate(BaseModel):
    """Service token update model."""

    token_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Human-readable token name",
    )
    is_active: bool | None = Field(default=None, description="Whether token is active")
    expires_at: datetime | None = Field(default=None, description="Token expiration datetime")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional token metadata")

    @validator("token_name")
    def validate_token_name(cls, v: str | None) -> str | None:  # noqa: N805
        """Validate and clean token name."""
        if v is not None:
            return v.strip()
        return v


class ServiceTokenResponse(BaseModel):
    """Service token response model."""

    id: uuid.UUID = Field(..., description="Token unique identifier")
    token_name: str = Field(..., description="Human-readable token name")
    created_at: datetime = Field(..., description="Token creation timestamp")
    last_used: datetime | None = Field(default=None, description="Last usage timestamp")
    usage_count: int = Field(..., description="Total usage count")
    expires_at: datetime | None = Field(default=None, description="Token expiration timestamp")
    is_active: bool = Field(..., description="Whether token is active")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional token metadata")
    is_expired: bool = Field(..., description="Whether token is expired")
    is_valid: bool = Field(..., description="Whether token is valid (active and not expired)")

    @classmethod
    def from_orm_model(cls, token: Any) -> "ServiceTokenResponse":
        """Create response from SQLAlchemy model."""
        return cls(
            id=token.id,
            token_name=token.token_name,
            created_at=token.created_at,
            last_used=token.last_used,
            usage_count=token.usage_count,
            expires_at=token.expires_at,
            is_active=token.is_active,
            metadata=token.token_metadata,  # Note: SQLAlchemy model uses token_metadata
            is_expired=token.is_expired,
            is_valid=getattr(token, "is_valid", token.is_active and not token.is_expired),
        )


class ServiceTokenListResponse(BaseModel):
    """Service token list response model."""

    tokens: list[ServiceTokenResponse] = Field(..., description="List of tokens")
    total: int = Field(..., ge=0, description="Total number of tokens")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class TokenValidationRequest(BaseModel):
    """Token validation request model."""

    token: str = Field(..., min_length=1, description="Token to validate")

    @validator("token")
    def validate_token(cls, v: str) -> str:  # noqa: N805
        """Validate and clean token."""
        return v.strip()


class TokenValidationResponse(BaseModel):
    """Token validation response model."""

    valid: bool = Field(..., description="Whether token is valid")
    token_id: uuid.UUID | None = Field(default=None, description="Token ID if valid")
    token_name: str | None = Field(default=None, description="Token name if valid")
    expires_at: datetime | None = Field(default=None, description="Token expiration if valid")
    metadata: dict[str, Any] | None = Field(default=None, description="Token metadata if valid")
    error: str | None = Field(default=None, description="Error message if invalid")
