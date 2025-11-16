"""SQLAlchemy models for PromptCraft authentication and service token management.

This module defines database models for:
- Service token management and tracking (AUTH-2)
- User session management (AUTH-1)
- Authentication event logging (AUTH-1)
- Role-based access control and permissions (AUTH-3)
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import TIMESTAMP, Boolean, Column, ForeignKey, Integer, String, Table, Text, func
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class ServiceToken(Base):
    """Service token model for API authentication and tracking."""

    __tablename__ = "service_tokens"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique service token identifier",
    )

    # Token identification
    token_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="Human-readable token name (e.g., 'ci-cd-pipeline')",
    )

    token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="SHA-256 hash of the token for secure storage",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Token creation timestamp",
    )

    last_used: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
        comment="Last time this token was used for authentication",
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
        comment="Token expiration time (NULL for no expiration)",
    )

    # Usage tracking
    usage_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times this token has been used",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether the token is active and can be used",
    )

    # Metadata
    token_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional metadata about the token (permissions, environment, etc.)",
    )

    def __repr__(self) -> str:
        """Return string representation of the service token."""
        status = "active" if self.is_active else "inactive"
        return f"<ServiceToken(name='{self.token_name}', {status}, uses={self.usage_count})>"

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        if self.expires_at is None:
            return False
        # Ensure both datetimes are timezone-aware and in UTC
        now_utc = datetime.now(timezone.utc)
        expires_at = self.expires_at
        if expires_at.tzinfo is None:
            # Assume naive expires_at is UTC
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return now_utc > expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the token is valid (active and not expired)."""
        return self.is_active and not self.is_expired


class UserSession(Base):
    """User session tracking model for authenticated users.

    Tracks user sessions, preferences, and metadata for enhanced
    authentication experience and user behavior analysis.
    """

    __tablename__ = "user_sessions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique session identifier",
    )

    # User identification
    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="User email address from Cloudflare Access JWT",
    )

    cloudflare_sub: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Cloudflare Access subject identifier",
    )

    # Session tracking
    first_seen: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="First time this user was seen",
    )

    last_seen: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last time this user was seen",
    )

    session_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Number of sessions for this user",
    )

    # User data
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment="User preferences and settings",
    )

    user_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment="Additional user metadata and context",
    )

    def __init__(self, **kwargs) -> None:
        """Initialize UserSession with proper defaults."""
        # Set defaults for fields that should have them
        if "session_count" not in kwargs:
            kwargs["session_count"] = 1
        if "preferences" not in kwargs:
            kwargs["preferences"] = {}
        if "user_metadata" not in kwargs:
            kwargs["user_metadata"] = {}

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Return string representation of the user session."""
        return f"<UserSession(id={self.id}, email='{self.email}', sessions={self.session_count})>"


class AuthenticationEvent(Base):
    """Authentication event model for audit logging and analytics.

    Tracks all authentication attempts, successes, and failures
    for security monitoring and performance analysis.
    """

    __tablename__ = "authentication_events"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique event identifier",
    )

    # Event identification
    user_email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="User email (from JWT or None for service token auth)",
    )

    service_token_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Service token name (for service token auth, None for JWT auth)",
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of authentication event (login, token_refresh, etc.)",
    )

    # Request context
    ip_address: Mapped[str | None] = mapped_column(
        INET,
        nullable=True,
        comment="Client IP address",
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Client user agent string",
    )

    cloudflare_ray_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Cloudflare Ray ID for request tracing",
    )

    # Event outcome
    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        index=True,
        comment="Whether the authentication was successful",
    )

    # Event details
    error_details: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Error details for failed authentication attempts",
    )

    performance_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Performance metrics (timing, etc.) for the authentication",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="Event timestamp",
    )

    def __init__(self, **kwargs) -> None:
        """Initialize AuthenticationEvent with proper defaults."""
        # Set defaults for fields that should have them
        if "success" not in kwargs:
            kwargs["success"] = True

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Return string representation of the authentication event."""
        status = "SUCCESS" if self.success else "FAILED"
        identifier = self.user_email or self.service_token_name or "unknown"
        return f"<AuthenticationEvent(id={self.id}, user='{identifier}', type='{self.event_type}', {status})>"


# Junction table for many-to-many relationship between roles and permissions
role_permissions_table = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    Column("permission_id", Integer, ForeignKey("permissions.id"), primary_key=True),
)

# Junction table for many-to-many relationship between users and roles
user_roles_table = Table(
    "user_roles",
    Base.metadata,
    Column("user_email", String(255), ForeignKey("user_sessions.email"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
)


class Role(Base):
    """Role model for hierarchical role-based access control (AUTH-3).

    Roles define sets of permissions and can inherit permissions from parent roles.
    This enables flexible permission management with role hierarchies.
    """

    __tablename__ = "roles"

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique role identifier",
    )

    # Role identification
    name: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique role name (e.g., 'admin', 'user', 'api_user')",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description of the role",
    )

    # Role hierarchy
    parent_role_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("roles.id"),
        nullable=True,
        comment="Parent role ID for inheritance (NULL for top-level roles)",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Role creation timestamp",
    )

    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Role last update timestamp",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether the role is active and can be assigned",
    )

    # Relationships
    permissions: Mapped[list["Permission"]] = relationship(
        "Permission",
        secondary=role_permissions_table,
        back_populates="roles",
        lazy="selectin",
    )

    parent_role: Mapped["Role | None"] = relationship(
        "Role",
        remote_side=[id],
        back_populates="child_roles",
    )

    child_roles: Mapped[list["Role"]] = relationship(
        "Role",
        back_populates="parent_role",
    )

    users: Mapped[list["UserSession"]] = relationship(
        "UserSession",
        secondary=user_roles_table,
        back_populates="roles",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """Return string representation of the role."""
        status = "active" if self.is_active else "inactive"
        parent = f", parent={self.parent_role.name}" if self.parent_role else ""
        return f"<Role(name='{self.name}', {status}, permissions={len(self.permissions)}{parent})>"

    def get_all_permissions(self) -> set[str]:
        """Get all permissions for this role including inherited permissions.

        Returns:
            Set of permission names including inherited permissions
        """
        permissions = {perm.name for perm in self.permissions}

        # Add permissions from parent role recursively
        if self.parent_role:
            permissions.update(self.parent_role.get_all_permissions())

        return permissions

    def has_permission(self, permission_name: str) -> bool:
        """Check if role has a specific permission (including inherited).

        Args:
            permission_name: Permission name to check

        Returns:
            True if role has permission, False otherwise
        """
        return permission_name in self.get_all_permissions()


class Permission(Base):
    """Permission model for fine-grained access control (AUTH-3).

    Permissions define specific actions that can be performed on resources.
    They follow the pattern: resource:action (e.g., 'tokens:create', 'users:read').
    """

    __tablename__ = "permissions"

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique permission identifier",
    )

    # Permission identification
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique permission name (e.g., 'tokens:create', 'users:read')",
    )

    resource: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Resource type this permission applies to (e.g., 'tokens', 'users')",
    )

    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Action this permission allows (e.g., 'create', 'read', 'update', 'delete')",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description of the permission",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Permission creation timestamp",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether the permission is active and can be assigned",
    )

    # Relationships
    roles: Mapped[list["Role"]] = relationship(
        "Role",
        secondary=role_permissions_table,
        back_populates="permissions",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """Return string representation of the permission."""
        status = "active" if self.is_active else "inactive"
        return f"<Permission(name='{self.name}', resource='{self.resource}', action='{self.action}', {status})>"


# Update UserSession to include role relationships
# Add this to the existing UserSession class by updating its relationships section
UserSession.roles = relationship(
    "Role",
    secondary=user_roles_table,
    back_populates="users",
    lazy="selectin",
)


# Export all models
__all__ = [
    "AuthenticationEvent",
    "Base",
    "Permission",
    "Role",
    "ServiceToken",
    "UserSession",
    "role_permissions_table",
    "user_roles_table",
]
