"""PostgreSQL database connection management for PromptCraft authentication.

This module provides async database connection management with:
- Connection pooling for optimal performance
- Proper session lifecycle management
- Configuration-based connection parameters
- Health checks and graceful failover
- Configuration management from environment variables
- Performance monitoring and optimization
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# Note: Base class is now defined in models.py to avoid circular imports
class DatabaseError(Exception):
    """Base exception for database operations."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize database error.

        Args:
            message: Error message
            original_error: Original exception if available
        """
        super().__init__(message)
        self.original_error = original_error


class DatabaseConnectionError(DatabaseError):
    """Exception for database connection failures."""


class DatabaseManager:
    """Manages PostgreSQL database connections for authentication system.

    Provides async database operations with connection pooling,
    health monitoring, and graceful degradation capabilities.
    """

    def __init__(self) -> None:
        """Initialize database manager."""
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._settings = get_settings()
        self._connection_lock = asyncio.Lock()
        self._health_check_cache: dict[str, Any] = {}
        self._health_check_ttl = 30.0  # 30 seconds cache
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize database engine and connection pool.

        Raises:
            ConnectionError: If database connection cannot be established
        """
        async with self._connection_lock:
            if self._engine is not None:
                logger.debug("Database engine already initialized")
                return

            try:
                # Build connection string
                db_url = await self._build_database_url()
                logger.debug("Connecting to PostgreSQL at %s:%s", self._settings.db_host, self._settings.db_port)

                # Create async engine with environment-appropriate configuration
                engine_kwargs = {
                    "echo": getattr(self._settings, "db_echo", False),
                    "connect_args": {
                        "server_settings": {
                            "application_name": "promptcraft-auth",
                            "jit": "off",  # Disable JIT for faster connection times
                        },
                        "command_timeout": 5.0,  # 5 second query timeout
                    },
                }

                # Use NullPool for dev, connection pooling for production
                if getattr(self._settings, "environment", "prod") == "dev":
                    engine_kwargs["poolclass"] = NullPool
                else:
                    # Add pool settings only for production
                    engine_kwargs.update(
                        {
                            "pool_size": getattr(self._settings, "db_pool_size", 10),
                            "max_overflow": getattr(self._settings, "db_pool_max_overflow", 20),
                            "pool_timeout": getattr(self._settings, "db_pool_timeout", 30),
                            "pool_recycle": getattr(self._settings, "db_pool_recycle", 3600),
                            "pool_pre_ping": True,  # Validate connections before use
                        },
                    )

                self._engine = create_async_engine(db_url, **engine_kwargs)

                # Create session factory
                self._session_factory = async_sessionmaker(
                    bind=self._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=True,
                    autocommit=False,
                )

                # Test connection
                await self._test_connection()

                self._is_initialized = True
                logger.info(
                    "Database initialized successfully - Pool: %s, Max overflow: %s",
                    getattr(self._settings, "db_pool_size", 10),
                    getattr(self._settings, "db_pool_max_overflow", 20),
                )

            except Exception as e:
                logger.error("Failed to initialize database: %s", e)
                await self._cleanup()
                raise DatabaseConnectionError(f"Database initialization failed: {e}") from e

    async def _build_database_url(self) -> str:
        """Build database URL from configuration."""
        # Use database_url if provided
        if hasattr(self._settings, "database_url") and self._settings.database_url:
            return self._settings.database_url.get_secret_value()

        # Build URL from components
        host = getattr(self._settings, "db_host", "192.168.1.16")
        port = getattr(self._settings, "db_port", 5432)
        database = getattr(self._settings, "db_name", "promptcraft")
        username = getattr(self._settings, "db_user", "promptcraft_app")

        password = ""  # nosec B105
        if hasattr(self._settings, "database_password") and self._settings.database_password:
            password = self._settings.database_password.get_secret_value()
        elif hasattr(self._settings, "db_password") and self._settings.db_password:
            # Handle both SecretStr and plain string types
            if hasattr(self._settings.db_password, "get_secret_value"):
                password = self._settings.db_password.get_secret_value()
            else:
                password = str(self._settings.db_password)

        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from settings (alternative method).

        Returns:
            Database connection URL
        """
        # Use asyncpg driver for async operations - supporting AUTH-1 pattern
        password = getattr(self._settings, "db_password", "")
        return (
            f"postgresql+asyncpg://{self._settings.db_user}:{password}"
            f"@{self._settings.db_host}:{self._settings.db_port}/{self._settings.db_name}"
        )

    async def _test_connection(self) -> None:
        """Test database connection and basic functionality.

        Raises:
            ConnectionError: If connection test fails
        """
        if not self._engine:
            raise DatabaseConnectionError("Database engine not initialized")

        try:
            async with self._engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1 as test"))
                row = result.fetchone()

                if not row or row[0] != 1:
                    raise DatabaseConnectionError("Database connection test failed")

                # Test PostgreSQL version
                version_result = await conn.execute(text("SELECT version()"))
                version_row = version_result.fetchone()
                if version_row:
                    logger.debug("Connected to: %s", version_row[0])

        except Exception as e:
            logger.error("Database connection test failed: %s", e)
            raise DatabaseConnectionError(f"Connection test failed: {e}") from e

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive database health check.

        Returns:
            Health check results with status and metrics
        """
        # Check cache first
        now = time.time()
        if self._health_check_cache and now - self._health_check_cache.get("timestamp", 0) < self._health_check_ttl:
            return self._health_check_cache

        health_status = {
            "status": "unhealthy",
            "timestamp": now,
            "engine_initialized": self._engine is not None,
            "connection_test": False,
            "pool_status": {},
            "response_time_ms": 0,
        }

        if not self._engine:
            health_status["error"] = "Database engine not initialized"
            self._health_check_cache = health_status
            return health_status

        try:
            start_time = time.time()

            # Test basic connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                health_status["connection_test"] = True

            # Get pool status
            pool = self._engine.pool
            health_status["pool_status"] = {
                "size": pool.size(),  # type: ignore[attr-defined]
                "checked_in": pool.checkedin(),  # type: ignore[attr-defined]
                "checked_out": pool.checkedout(),  # type: ignore[attr-defined]
                "overflow": pool.overflow(),  # type: ignore[attr-defined]
                "total_connections": pool.size() + pool.overflow(),  # type: ignore[attr-defined]
            }

            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            health_status["response_time_ms"] = round(response_time, 2)

            # Mark as healthy if all checks pass
            if health_status["connection_test"]:
                health_status["status"] = "healthy"

        except Exception as e:
            logger.warning("Database health check failed: %s", e)
            health_status["error"] = str(e)

        # Cache result
        self._health_check_cache = health_status
        return health_status

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup.

        Yields:
            AsyncSession for database operations

        Raises:
            ConnectionError: If session cannot be created
        """
        if not self._session_factory:
            await self.initialize()

        if not self._session_factory:
            raise DatabaseConnectionError("Database session factory not available")
        async with self._session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error("Database session error, rolling back: %s", e)
                raise
            finally:
                await session.close()

    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Any:
        """Execute database operation with retry logic.

        Args:
            operation: Async callable to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Result of the operation

        Raises:
            DatabaseError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_error = e

                if attempt < max_retries:
                    logger.warning("Database operation failed (attempt %s), retrying: %s", attempt + 1, e)
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error("Database operation failed after %s attempts: %s", max_retries + 1, e)

        raise DatabaseError(f"Operation failed after {max_retries + 1} attempts") from last_error

    async def get_pool_status(self) -> dict:
        """Get connection pool status."""
        if not self._is_initialized:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "pool_size": getattr(self._engine.pool, "size", 0) if self._engine else 0,
            "checked_in": getattr(self._engine.pool, "checkedin", 0) if self._engine else 0,
            "checked_out": getattr(self._engine.pool, "checkedout", 0) if self._engine else 0,
        }

    # Alias for compatibility with tests
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Alias for get_session for test compatibility."""
        async with self.get_session() as session:
            yield session

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Internal cleanup method."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._is_initialized = False
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance (synchronous version)."""
    global _db_manager  # noqa: PLW0603
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_database_manager_async() -> DatabaseManager:
    """Get global database manager instance with initialization.

    Returns:
        DatabaseManager instance
    """
    global _db_manager  # noqa: PLW0603
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager


# Legacy compatibility functions
@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session (legacy compatibility)."""
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection.

    Yields:
        AsyncSession for database operations
    """
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session


async def database_health_check() -> dict[str, Any]:
    """Perform database health check (legacy compatibility)."""
    db_manager = get_database_manager()
    return await db_manager.health_check()


# Additional compatibility function for migration scripts
async def initialize_database() -> None:
    """Initialize database connection (used in validation scripts)."""
    db_manager = get_database_manager()
    await db_manager.initialize()


# Export public interface
__all__ = [
    "DatabaseConnectionError",
    "DatabaseError",
    "DatabaseManager",
    "database_health_check",
    "get_database_manager",
    "get_database_manager_async",
    "get_db",
    "get_db_session",
    "initialize_database",
]
