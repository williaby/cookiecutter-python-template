"""Database package for PromptCraft authentication and service token management.

This package provides:
- Async PostgreSQL connection management
- SQLAlchemy models for authentication and service tokens
- Database utilities and migration support
- Session management, event logging, and user metadata storage
"""

from .base_service import DatabaseService
from .connection import (
    DatabaseConnectionError,
    DatabaseError,
    DatabaseManager,
    get_database_manager,
    get_db,
    get_db_session,
)
from .models import AuthenticationEvent, Base, ServiceToken, UserSession

__all__ = [
    "AuthenticationEvent",
    "Base",
    "DatabaseConnectionError",
    "DatabaseError",
    "DatabaseManager",
    "DatabaseService",
    "ServiceToken",
    "UserSession",
    "get_database_manager",
    "get_db",
    "get_db_session",
]
