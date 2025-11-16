"""Database service base class for standardized session management.

This module provides a base class that eliminates duplicate database session
management patterns across the authentication system. It provides:
- Standardized session lifecycle management
- Consistent error handling and logging
- Transaction support with automatic rollback
- Connection management with proper resource cleanup
- Performance monitoring and optimization

All database service classes should inherit from DatabaseService to ensure
consistent patterns and eliminate code duplication.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import DatabaseError, get_database_manager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseService:
    """Base class for database services with standardized session management.

    Provides common database operations and patterns to eliminate duplicate
    session management code across the authentication system.

    All service classes that interact with the database should inherit from
    this class to ensure consistent error handling and session management.
    """

    def __init__(self) -> None:
        """Initialize database service."""
        self._db_manager = get_database_manager()

    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic resource management.

        This context manager provides automatic session cleanup and error handling.
        All database operations should use this method for session management.

        Yields:
            AsyncSession: Database session

        Raises:
            DatabaseError: If session creation fails
        """
        async with self._db_manager.get_session() as session:
            try:
                yield session
            except SQLAlchemyError as e:
                logger.error(f"Database operation failed: {e}")
                await session.rollback()
                raise DatabaseError(f"Database operation failed: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error during database operation: {e}")
                await session.rollback()
                raise

    async def execute_with_session(self, operation: callable, *args, **kwargs) -> Any:
        """Execute database operation with automatic session management.

        This method provides a standardized way to execute database operations
        with proper error handling and logging.

        Args:
            operation: Callable that takes session as first parameter
            *args: Additional positional arguments for operation
            **kwargs: Additional keyword arguments for operation

        Returns:
            Result of the operation

        Raises:
            DatabaseError: If operation fails
        """
        async with self.get_session() as session:
            return await operation(session, *args, **kwargs)

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        fetch_one: bool = False,
        fetch_scalar: bool = False,
    ) -> Any:
        """Execute raw SQL query with parameter binding.

        Args:
            query: SQL query string
            parameters: Query parameters for binding
            fetch_one: Whether to fetch only one result
            fetch_scalar: Whether to fetch scalar value

        Returns:
            Query results based on fetch parameters

        Raises:
            DatabaseError: If query execution fails
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), parameters or {})

            if fetch_scalar:
                return result.scalar()
            if fetch_one:
                return result.fetchone()
            return result.fetchall()

    async def handle_integrity_error(
        self,
        error: IntegrityError,
        operation_name: str,
        entity_name: str = "",
    ) -> None:
        """Handle database integrity constraint violations.

        Provides standardized handling for common integrity errors like
        unique constraint violations, foreign key violations, etc.

        Args:
            error: SQLAlchemy IntegrityError
            operation_name: Name of the operation that failed
            entity_name: Name of the entity being operated on

        Raises:
            DatabaseError: Standardized database error with helpful message
        """
        error_msg = str(error).lower()

        if "unique constraint" in error_msg or "duplicate key" in error_msg:
            message = f"{entity_name} already exists" if entity_name else "Duplicate entry"
            logger.error(f"{operation_name} failed - {message}: {error}")
            raise DatabaseError(f"{operation_name} failed: {message}") from error
        if "foreign key" in error_msg:
            message = "Referenced entity does not exist"
            logger.error(f"{operation_name} failed - {message}: {error}")
            raise DatabaseError(f"{operation_name} failed: {message}") from error
        logger.error(f"{operation_name} failed - integrity error: {error}")
        raise DatabaseError(f"{operation_name} failed: {error}") from error

    async def log_operation_success(
        self,
        operation_name: str,
        entity_id: Any = None,
        entity_name: str = "",
        additional_info: str = "",
    ) -> None:
        """Log successful database operation with consistent formatting.

        Args:
            operation_name: Name of the operation
            entity_id: ID of the entity operated on
            entity_name: Name of the entity operated on
            additional_info: Additional information to log
        """
        id_part = f" with ID {entity_id}" if entity_id else ""
        name_part = f" '{entity_name}'" if entity_name else ""
        info_part = f" - {additional_info}" if additional_info else ""

        logger.info(f"{operation_name}{name_part}{id_part}{info_part}")

    async def log_operation_error(
        self,
        operation_name: str,
        error: Exception,
        entity_name: str = "",
    ) -> None:
        """Log database operation error with consistent formatting.

        Args:
            operation_name: Name of the operation
            error: Exception that occurred
            entity_name: Name of the entity being operated on
        """
        entity_part = f" for {entity_name}" if entity_name else ""
        logger.error(f"{operation_name} failed{entity_part}: {error}")

    async def check_entity_exists(
        self,
        session: AsyncSession,
        model_class: type,
        filter_conditions: dict[str, Any],
        entity_name: str = "",
    ) -> bool:
        """Check if entity exists with given conditions.

        Args:
            session: Database session
            model_class: SQLAlchemy model class
            filter_conditions: Dictionary of field->value conditions
            entity_name: Name for error messages

        Returns:
            True if entity exists, False otherwise
        """
        from sqlalchemy import select

        query = select(model_class.id)
        for field, value in filter_conditions.items():
            query = query.where(getattr(model_class, field) == value)

        result = await session.execute(query)
        exists = result.scalar_one_or_none() is not None

        logger.debug(f"Entity existence check for {entity_name}: {exists}")
        return exists

    async def get_entity_by_conditions(
        self,
        session: AsyncSession,
        model_class: type,
        filter_conditions: dict[str, Any],
        entity_name: str = "",
    ) -> Any | None:
        """Get entity by filter conditions.

        Args:
            session: Database session
            model_class: SQLAlchemy model class
            filter_conditions: Dictionary of field->value conditions
            entity_name: Name for logging

        Returns:
            Entity if found, None otherwise
        """
        from sqlalchemy import select

        query = select(model_class)
        for field, value in filter_conditions.items():
            query = query.where(getattr(model_class, field) == value)

        result = await session.execute(query)
        entity = result.scalar_one_or_none()

        if entity:
            logger.debug(f"Found {entity_name} with conditions {filter_conditions}")
        else:
            logger.debug(f"No {entity_name} found with conditions {filter_conditions}")

        return entity


__all__ = ["DatabaseService"]
