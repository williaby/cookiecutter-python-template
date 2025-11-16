"""Service Token Management for AUTH-2 implementation.

This module provides comprehensive service token management including:
- Token creation, rotation, and revocation
- Emergency revocation system
- Usage analytics and monitoring
- Bulk operations for administrative tasks
"""

import hashlib
import logging
import secrets
from datetime import timezone, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_db
from src.database.models import AuthenticationEvent, ServiceToken

logger = logging.getLogger(__name__)


class ServiceTokenManager:
    """Manages service token lifecycle and operations."""

    def __init__(self) -> None:
        """Initialize service token manager."""
        self.token_prefix = "sk_"  # nosec B105  # noqa: S105
        self.token_length = 32  # 32 bytes = 256 bits of entropy

    async def _get_session(self) -> AsyncSession:
        """Get database session from async generator.

        Raises:
            Exception: Database connection errors are allowed to propagate
        """
        # Handle both real usage and test mocking
        db_gen = get_db()
        # If it's mocked, it will have __aenter__ method
        if hasattr(db_gen, "__aenter__"):
            session: AsyncSession = await db_gen.__aenter__()
            return session
        # Real async generator usage
        async for session in db_gen:
            return session
        # This should never be reached in normal operation
        msg = "No database session available"
        raise RuntimeError(msg)

    def generate_token(self) -> str:
        """Generate a new cryptographically secure service token.

        Returns:
            Service token string with 'sk_' prefix
        """
        # Generate 32 bytes (256 bits) of cryptographically secure random data
        token_bytes = secrets.token_bytes(self.token_length)
        # Convert to hex and add prefix
        token_value = token_bytes.hex()
        return f"{self.token_prefix}{token_value}"

    def hash_token(self, token: str) -> str:
        """Hash a service token for secure database storage.

        Args:
            token: Raw service token string

        Returns:
            SHA-256 hash of the token
        """
        return hashlib.sha256(token.encode()).hexdigest()

    async def create_service_token(
        self,
        token_name: str,
        metadata: dict | None = None,
        expires_at: datetime | None = None,
        is_active: bool = True,
    ) -> tuple[str, str] | None:
        """Create a new service token.

        Args:
            token_name: Human-readable name for the token
            metadata: Token metadata (permissions, client info, etc.)
            expires_at: Optional expiration datetime
            is_active: Whether token should be active

        Returns:
            Tuple of (token_value, token_id) - token_value should be given to client

        Raises:
            ValueError: If token_name already exists
            Exception: For database errors
        """
        try:
            # Generate new token
            token_value = self.generate_token()
            token_hash = self.hash_token(token_value)

            session = await self._get_session()

            # Check if token name already exists
            result = await session.execute(
                text("SELECT COUNT(*) FROM service_tokens WHERE token_name = :token_name"),
                {"token_name": token_name},
            )

            count = result.scalar() or 0
            if count > 0:
                raise ValueError(f"Service token with name '{token_name}' already exists")

            # Create new token record
            new_token = ServiceToken(
                token_name=token_name,
                token_hash=token_hash,
                expires_at=expires_at,
                is_active=is_active,
                token_metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
            )

            session.add(new_token)
            await session.commit()
            await session.refresh(new_token)

            # Sanitize token_name for logging to prevent log injection
            safe_token_name = token_name.replace("\n", "").replace("\r", "")[:50]
            logger.info(f"Created service token: {safe_token_name}... (ID: {new_token.id})")

            return token_value, str(new_token.id)

        except ValueError as e:
            # Re-raise ValueError for duplicate names and other validation errors
            # Sanitize token_name for logging to prevent log injection
            safe_token_name = token_name.replace("\n", "").replace("\r", "")[:50]
            logger.error(f"Error creating service token '{safe_token_name}...': {e}")
            raise
        except Exception as e:
            # Sanitize token_name for logging to prevent log injection
            safe_token_name = token_name.replace("\n", "").replace("\r", "")[:50]

            # Database connection errors and other critical errors should propagate
            if "Database connection failed" in str(e):
                logger.error(f"Database connection failed for token '{safe_token_name}...': {e}")
                raise

            # Log and return None for other database operation errors
            logger.error(f"Error creating service token '{safe_token_name}...': {e}")
            return None

    async def revoke_service_token(
        self,
        token_identifier: str,
        revocation_reason: str = "manual_revocation",
    ) -> bool | None:
        """Revoke a service token (emergency or planned).

        Args:
            token_identifier: Token name, ID, or hash to revoke
            revocation_reason: Reason for revocation (for audit trail)

        Returns:
            True if token was revoked, False if not found
        """
        try:
            session = await self._get_session()

            # Try to find token by name, ID, or hash
            result = await session.execute(
                text(
                    """
                    SELECT id, token_name, is_active
                    FROM service_tokens
                    WHERE token_name = :identifier
                       OR id::text = :identifier
                       OR token_hash = :identifier
                """,
                ),
                {"identifier": token_identifier},
            )

            token_record = result.fetchone()

            if not token_record:
                # Sanitize token_identifier for logging to prevent log injection
                safe_identifier = token_identifier.replace("\n", "").replace("\r", "")[:50]
                logger.warning(f"Service token not found for revocation: {safe_identifier}...")
                return False

            # Deactivate the token
            await session.execute(
                text("UPDATE service_tokens SET is_active = FALSE WHERE id = :token_id"),
                {"token_id": token_record.id},
            )

            # Log revocation event
            revocation_event = AuthenticationEvent(
                service_token_name=token_record.token_name,
                event_type="service_token_revocation",
                success=True,
                error_details={"reason": revocation_reason},
                created_at=datetime.now(timezone.utc),
            )

            session.add(revocation_event)
            await session.commit()

            # Sanitize token name and reason for logging to prevent log injection
            safe_token_name = token_record.token_name.replace("\n", "").replace("\r", "")[:50]
            safe_reason = revocation_reason.replace("\n", "").replace("\r", "")[:30]
            logger.warning(f"REVOKED service token: {safe_token_name}... (reason: {safe_reason}...)")

            return True

        except Exception as e:
            # Sanitize token_identifier for logging to prevent log injection
            safe_identifier = token_identifier.replace("\n", "").replace("\r", "")[:50]

            # Database connection errors should propagate
            if "Database connection failed" in str(e):
                logger.error(f"Database connection failed for token revocation '{safe_identifier}...': {e}")
                raise

            # Log and return None for other errors
            logger.error(f"Error revoking service token '{safe_identifier}...': {e}")
            return None

    async def emergency_revoke_all_tokens(self, emergency_reason: str) -> int | None:
        """Emergency revocation of ALL service tokens.

        Args:
            emergency_reason: Reason for emergency revocation

        Returns:
            Number of tokens revoked
        """
        try:
            session = await self._get_session()

            # Get count of active tokens
            result = await session.execute(text("SELECT COUNT(*) FROM service_tokens WHERE is_active = TRUE"))
            active_count = result.scalar()

            if active_count == 0:
                logger.info("Emergency revocation: No active tokens to revoke")
                return 0

            # Deactivate all active tokens
            await session.execute(text("UPDATE service_tokens SET is_active = FALSE WHERE is_active = TRUE"))

            # Log emergency revocation event
            emergency_event = AuthenticationEvent(
                event_type="emergency_revocation_all",
                success=True,
                error_details={"reason": emergency_reason, "tokens_revoked": active_count},
                created_at=datetime.now(timezone.utc),
            )

            session.add(emergency_event)
            await session.commit()

            # Sanitize emergency_reason for logging to prevent log injection
            safe_reason = emergency_reason.replace("\n", "").replace("\r", "")[:100]
            logger.critical(f"EMERGENCY REVOCATION: Revoked {active_count} service tokens (reason: {safe_reason})")

            return active_count

        except Exception as e:
            logger.error(f"Error during emergency revocation: {e}")
            return None

    async def rotate_service_token(
        self,
        token_identifier: str,
        rotation_reason: str = "scheduled_rotation",
    ) -> tuple[str, str] | None:
        """Rotate a service token (create new, revoke old).

        Args:
            token_identifier: Token name, ID, or hash to rotate
            rotation_reason: Reason for rotation

        Returns:
            Tuple of (new_token_value, new_token_id) if successful, None if not found
        """
        try:
            session = await self._get_session()

            # Find existing token
            result = await session.execute(
                text(
                    """
                    SELECT id, token_name, token_metadata, expires_at
                    FROM service_tokens
                    WHERE (token_name = :identifier
                       OR id::text = :identifier
                       OR token_hash = :identifier)
                       AND is_active = TRUE
                """,
                ),
                {"identifier": token_identifier},
            )

            old_token = result.fetchone()

            if not old_token:
                # Sanitize token_identifier for logging to prevent log injection
                safe_identifier = token_identifier.replace("\n", "").replace("\r", "")[:50]
                logger.warning(f"Active service token not found for rotation: {safe_identifier}...")
                return None

            # Generate new token
            new_token_value = self.generate_token()
            new_token_hash = self.hash_token(new_token_value)

            # Create new token with same metadata
            new_token = ServiceToken(
                token_name=f"{old_token.token_name}_rotated_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                token_hash=new_token_hash,
                expires_at=old_token.expires_at,
                is_active=True,
                token_metadata=old_token.token_metadata,
                created_at=datetime.now(timezone.utc),
            )

            session.add(new_token)

            # Deactivate old token
            await session.execute(
                text("UPDATE service_tokens SET is_active = FALSE WHERE id = :token_id"),
                {"token_id": old_token.id},
            )

            # Log rotation event
            rotation_event = AuthenticationEvent(
                service_token_name=old_token.token_name,
                event_type="service_token_rotation",
                success=True,
                error_details={
                    "reason": rotation_reason,
                    "old_token_id": str(old_token.id),
                    "new_token_name": new_token.token_name,
                },
                created_at=datetime.now(timezone.utc),
            )

            session.add(rotation_event)
            await session.commit()
            await session.refresh(new_token)

            # Sanitize token names for logging to prevent log injection
            safe_old_name = old_token.token_name.replace("\n", "").replace("\r", "")[:50]
            safe_new_name = new_token.token_name.replace("\n", "").replace("\r", "")[:50]
            logger.info(f"Rotated service token: {safe_old_name}... -> {safe_new_name}...")

            return new_token_value, str(new_token.id)

        except Exception as e:
            # Sanitize token_identifier for logging to prevent log injection
            safe_identifier = token_identifier.replace("\n", "").replace("\r", "")[:50]

            # Database connection errors should propagate
            if "Database connection failed" in str(e):
                logger.error(f"Database connection failed for token rotation '{safe_identifier}...': {e}")
                raise

            # Log and return None for other errors
            logger.error(f"Error rotating service token '{safe_identifier}...': {e}")
            return None

    async def get_token_usage_analytics(
        self,
        token_identifier: str | None = None,
        days: int = 30,
    ) -> dict[str, Any] | None:
        """Get usage analytics for service tokens.

        Args:
            token_identifier: Specific token to analyze (None for all)
            days: Number of days to analyze

        Returns:
            Dictionary with usage analytics
        """
        try:
            session = await self._get_session()
            # Base analytics query
            if token_identifier:
                # Single token analytics
                result = await session.execute(
                    text(
                        """
                        SELECT
                            token_name,
                            usage_count,
                            last_used,
                            created_at,
                            is_active,
                            CASE
                                WHEN expires_at IS NULL THEN FALSE
                                WHEN expires_at > NOW() THEN FALSE
                                ELSE TRUE
                            END as is_expired
                        FROM service_tokens
                        WHERE token_name = :identifier
                           OR id::text = :identifier
                           OR token_hash = :identifier
                    """,
                    ),
                    {"identifier": token_identifier},
                )

                token_data = result.fetchone()
                if not token_data:
                    return {"error": "Token not found"}

                # Calculate cutoff date for security (avoid SQL injection)
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

                # Get recent authentication events
                auth_events = await session.execute(
                    text(
                        """
                        SELECT event_type, success, created_at
                        FROM authentication_events
                        WHERE service_token_name = :token_name
                          AND created_at >= :cutoff_date
                        ORDER BY created_at DESC
                        LIMIT 100
                    """,
                    ),
                    {"token_name": token_data.token_name, "cutoff_date": cutoff_date},
                )

                recent_events = [
                    {"event_type": row.event_type, "success": row.success, "timestamp": row.created_at.isoformat()}
                    for row in auth_events.fetchall()
                ]

                return {
                    "token_name": token_data.token_name,
                    "usage_count": token_data.usage_count,
                    "last_used": token_data.last_used.isoformat() if token_data.last_used else None,
                    "created_at": token_data.created_at.isoformat(),
                    "is_active": token_data.is_active,
                    "is_expired": token_data.is_expired,
                    "recent_events": recent_events,
                    "events_count": len(recent_events),
                }

            # All tokens summary
            summary_result = await session.execute(
                text(
                    """
                        SELECT
                            COUNT(*) as total_tokens,
                            COUNT(*) FILTER (WHERE is_active = TRUE) as active_tokens,
                            COUNT(*) FILTER (WHERE is_active = FALSE) as inactive_tokens,
                            COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at < NOW()) as expired_tokens,
                            SUM(usage_count) as total_usage,
                            AVG(usage_count) as avg_usage_per_token
                        FROM service_tokens
                    """,
                ),
            )

            summary = summary_result.fetchone()
            if not summary:
                return {"error": "Unable to fetch summary statistics"}

            # Get top used tokens
            top_tokens_result = await session.execute(
                text(
                    """
                        SELECT token_name, usage_count, last_used
                        FROM service_tokens
                        WHERE is_active = TRUE
                        ORDER BY usage_count DESC
                        LIMIT 10
                    """,
                ),
            )

            top_tokens = [
                {
                    "token_name": row.token_name,
                    "usage_count": row.usage_count,
                    "last_used": row.last_used.isoformat() if row.last_used else None,
                }
                for row in top_tokens_result.fetchall()
            ]

            return {
                "summary": {
                    "total_tokens": summary.total_tokens,
                    "active_tokens": summary.active_tokens,
                    "inactive_tokens": summary.inactive_tokens,
                    "expired_tokens": summary.expired_tokens,
                    "total_usage": summary.total_usage or 0,
                    "avg_usage_per_token": float(summary.avg_usage_per_token or 0),
                },
                "top_tokens": top_tokens,
                "analysis_period_days": days,
            }

        except Exception as e:
            logger.error(f"Error in analytics: {e}")
            return None

    async def cleanup_expired_tokens(self, deactivate_only: bool = True) -> dict[str, Any] | None:
        """Clean up expired service tokens.

        Args:
            deactivate_only: If True, deactivate expired tokens; if False, delete them

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            session = await self._get_session()
            # Find expired tokens
            expired_result = await session.execute(
                text(
                    """
                    SELECT id, token_name
                    FROM service_tokens
                    WHERE expires_at IS NOT NULL
                      AND expires_at < NOW()
                      AND is_active = TRUE
                """,
                ),
            )

            expired_tokens = expired_result.fetchall()
            expired_count = len(expired_tokens)

            if expired_count == 0:
                return {"expired_tokens_processed": 0, "action": "none_needed"}

            if deactivate_only:
                # Deactivate expired tokens
                await session.execute(
                    text(
                        """
                        UPDATE service_tokens
                        SET is_active = FALSE
                        WHERE expires_at IS NOT NULL
                          AND expires_at < NOW()
                          AND is_active = TRUE
                    """,
                    ),
                )
                action = "deactivated"
            else:
                # Delete expired tokens
                await session.execute(
                    text(
                        """
                        DELETE FROM service_tokens
                        WHERE expires_at IS NOT NULL
                          AND expires_at < NOW()
                    """,
                    ),
                )
                action = "deleted"

            # Log cleanup event
            cleanup_event = AuthenticationEvent(
                event_type="token_cleanup",
                success=True,
                error_details={
                    "expired_tokens_processed": expired_count,
                    "action": action,
                    "token_names": [token.token_name for token in expired_tokens],
                },
                created_at=datetime.now(timezone.utc),
            )

            session.add(cleanup_event)
            await session.commit()

            logger.info(f"Token cleanup: {action} {expired_count} expired tokens")

            return {
                "expired_tokens_processed": expired_count,
                "action": action,
                "token_names": [token.token_name for token in expired_tokens],
            }

        except Exception as e:
            logger.error(f"Error during token cleanup: {e}")
            return None
