"""Unit tests for database connection management."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from src.database.connection import (
    DatabaseConnectionError,
    DatabaseError,
    DatabaseManager,
    get_database_manager_async,
    get_db_session,
)


@pytest.mark.unit
@pytest.mark.fast
class TestDatabaseError:
    """Test database error classes."""

    def test_database_error_initialization(self):
        """Test DatabaseError initialization."""
        original_error = Exception("Original error")
        error = DatabaseError("Test error", original_error)

        assert str(error) == "Test error"
        assert error.original_error == original_error

    def test_database_error_without_original(self):
        """Test DatabaseError initialization without original error."""
        error = DatabaseError("Test error")

        assert str(error) == "Test error"
        assert error.original_error is None

    def test_connection_error_inheritance(self):
        """Test DatabaseConnectionError inherits from DatabaseError."""
        error = DatabaseConnectionError("Connection failed")

        assert isinstance(error, DatabaseError)
        assert str(error) == "Connection failed"


@pytest.mark.unit
@pytest.mark.fast
class TestDatabaseManager:
    """Test DatabaseManager class."""

    def test_initialization(self, mock_settings):
        """Test DatabaseManager initialization."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            assert manager._engine is None
            assert manager._session_factory is None
            assert manager._settings == mock_settings
            assert manager._health_check_ttl == 30.0
            assert isinstance(manager._health_check_cache, dict)

    def test_build_connection_string(self, mock_settings):
        """Test connection string building."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            connection_string = manager._build_connection_string()

            expected = (
                f"postgresql+asyncpg://{mock_settings.db_user}:{mock_settings.db_password}"
                f"@{mock_settings.db_host}:{mock_settings.db_port}/{mock_settings.db_name}"
            )
            assert connection_string == expected

    async def test_initialize_success(self, mock_settings, mock_async_engine, mock_session_factory):
        """Test successful database initialization."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", return_value=mock_async_engine),
            patch("src.database.connection.async_sessionmaker", return_value=mock_session_factory),
        ):

            manager = DatabaseManager()
            await manager.initialize()

            assert manager._engine == mock_async_engine
            assert manager._session_factory == mock_session_factory

            # Verify engine creation was called with correct parameters
            from src.database.connection import create_async_engine

            create_async_engine.assert_called_once()
            args, kwargs = create_async_engine.call_args

            assert kwargs["pool_size"] == mock_settings.db_pool_size
            assert kwargs["max_overflow"] == mock_settings.db_pool_max_overflow
            assert kwargs["pool_timeout"] == mock_settings.db_pool_timeout

    async def test_initialize_already_initialized(self, mock_settings, mock_async_engine):
        """Test initialization when already initialized."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", return_value=mock_async_engine),
        ):

            manager = DatabaseManager()
            manager._engine = mock_async_engine  # Already initialized

            with patch("src.database.connection.async_sessionmaker") as mock_session_maker:
                await manager.initialize()

                # Should not create new engine
                mock_session_maker.assert_not_called()

    async def test_initialize_connection_failure(self, mock_settings):
        """Test initialization with connection failure."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", side_effect=Exception("Connection failed")),
        ):

            manager = DatabaseManager()

            with pytest.raises(DatabaseConnectionError, match="Database initialization failed"):
                await manager.initialize()

    async def test_test_connection_success(self, mock_settings, mock_async_engine):
        """Test successful connection test."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            manager._engine = mock_async_engine

            # Should not raise exception
            await manager._test_connection()

            # Verify connection test was performed
            mock_async_engine.begin.assert_called_once()

    async def test_test_connection_failure(self, mock_settings):
        """Test connection test failure."""
        mock_engine = AsyncMock()
        mock_engine.begin.side_effect = Exception("Connection test failed")

        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            manager._engine = mock_engine

            with pytest.raises(DatabaseConnectionError, match="Connection test failed"):
                await manager._test_connection()

    async def test_test_connection_no_engine(self, mock_settings):
        """Test connection test with no engine."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            with pytest.raises(DatabaseConnectionError, match="Database engine not initialized"):
                await manager._test_connection()

    async def test_health_check_success(self, mock_settings, mock_async_engine):
        """Test successful health check."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            manager._engine = mock_async_engine

            health_status = await manager.health_check()

            assert health_status["status"] == "healthy"
            assert health_status["engine_initialized"] is True
            assert health_status["connection_test"] is True
            assert "response_time_ms" in health_status
            assert "pool_status" in health_status
            assert "timestamp" in health_status

    async def test_health_check_no_engine(self, mock_settings):
        """Test health check with no engine."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            health_status = await manager.health_check()

            assert health_status["status"] == "unhealthy"
            assert health_status["engine_initialized"] is False
            assert "error" in health_status

    async def test_health_check_connection_failure(self, mock_settings):
        """Test health check with connection failure."""
        mock_engine = AsyncMock()
        mock_engine.begin.side_effect = Exception("Connection failed")

        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            manager._engine = mock_engine

            health_status = await manager.health_check()

            assert health_status["status"] == "unhealthy"
            assert health_status["connection_test"] is False
            assert "error" in health_status

    async def test_health_check_caching(self, mock_settings, mock_async_engine):
        """Test health check caching."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            manager._engine = mock_async_engine
            manager._health_check_ttl = 1.0  # 1 second cache

            # First call
            health_status1 = await manager.health_check()
            timestamp1 = health_status1["timestamp"]

            # Second call immediately - should return cached result
            health_status2 = await manager.health_check()
            timestamp2 = health_status2["timestamp"]

            assert timestamp1 == timestamp2  # Same cached result

    async def test_get_session_success(self, mock_settings, mock_async_engine, mock_session_factory):
        """Test successful session retrieval."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            manager._session_factory = mock_session_factory

            async with manager.get_session() as session:
                assert session is not None

            # Verify session factory was called
            mock_session_factory.assert_called_once()

    async def test_get_session_auto_initialize(self, mock_settings, mock_async_engine, mock_session_factory):
        """Test session retrieval with auto-initialization."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", return_value=mock_async_engine),
            patch("src.database.connection.async_sessionmaker", return_value=mock_session_factory),
        ):

            manager = DatabaseManager()

            async with manager.get_session() as session:
                assert session is not None

            # Verify initialization occurred
            assert manager._session_factory == mock_session_factory

    async def test_get_session_initialization_failure(self, mock_settings):
        """Test session retrieval with initialization failure."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", side_effect=Exception("Init failed")),
        ):

            manager = DatabaseManager()

            with pytest.raises(DatabaseConnectionError):
                async with manager.get_session():
                    pass

    async def test_get_session_no_factory(self, mock_settings):
        """Test session retrieval with no factory after initialization."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            # Simulate failed initialization
            manager._session_factory = None

            # Mock initialize to not set factory
            with (
                patch.object(manager, "initialize", AsyncMock()),
                pytest.raises(DatabaseConnectionError, match="Database session factory not available"),
            ):
                async with manager.get_session():
                    pass

    async def test_execute_with_retry_success(self, mock_settings):
        """Test successful operation with retry mechanism."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            async def mock_operation():
                return "success"

            result = await manager.execute_with_retry(mock_operation)
            assert result == "success"

    async def test_execute_with_retry_failure_then_success(self, mock_settings):
        """Test retry mechanism with failure then success."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            call_count = 0

            async def mock_operation():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Temporary failure")
                return "success"

            result = await manager.execute_with_retry(mock_operation, max_retries=3)
            assert result == "success"
            assert call_count == 3

    async def test_execute_with_retry_max_retries_exceeded(self, mock_settings):
        """Test retry mechanism with max retries exceeded."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            async def mock_operation():
                raise Exception("Persistent failure")

            with pytest.raises(DatabaseError, match="Operation failed after 3 attempts"):
                await manager.execute_with_retry(mock_operation, max_retries=2)

    async def test_execute_with_retry_exponential_backoff(self, mock_settings):
        """Test retry mechanism uses exponential backoff."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            call_times = []

            async def mock_operation():
                call_times.append(time.time())
                raise Exception("Always fails")

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(DatabaseError):
                    await manager.execute_with_retry(mock_operation, max_retries=2, retry_delay=1.0)

                # Verify exponential backoff delays
                mock_sleep.assert_any_call(1.0)  # First retry: 1.0 * 1
                mock_sleep.assert_any_call(2.0)  # Second retry: 1.0 * 2

    async def test_close(self, mock_settings, mock_async_engine):
        """Test database connection cleanup."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()
            manager._engine = mock_async_engine

            await manager.close()

            assert manager._engine is None
            assert manager._session_factory is None
            mock_async_engine.dispose.assert_called_once()

    async def test_close_no_engine(self, mock_settings):
        """Test cleanup with no engine."""
        with patch("src.database.connection.get_settings", return_value=mock_settings):
            manager = DatabaseManager()

            # Should not raise exception
            await manager.close()


@pytest.mark.unit
@pytest.mark.fast
class TestGlobalDatabaseManager:
    """Test global database manager functions."""

    async def test_get_database_manager_singleton(self, mock_settings, mock_async_engine, mock_session_factory):
        """Test global database manager is singleton."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", return_value=mock_async_engine),
            patch("src.database.connection.async_sessionmaker", return_value=mock_session_factory),
        ):

            # Clear any existing global instance
            import src.database.connection

            src.database.connection._db_manager = None

            manager1 = await get_database_manager_async()
            manager2 = await get_database_manager_async()

            assert manager1 is manager2  # Same instance

    async def test_get_db_session(self, mock_settings, mock_async_engine, mock_session_factory):
        """Test get_db_session function."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", return_value=mock_async_engine),
            patch("src.database.connection.async_sessionmaker", return_value=mock_session_factory),
        ):

            # Clear any existing global instance
            import src.database.connection

            src.database.connection._db_manager = None

            async with get_db_session() as session:
                assert session is not None

    async def test_get_db_session_initialization_failure(self, mock_settings):
        """Test get_db_session with initialization failure."""
        with (
            patch("src.database.connection.get_settings", return_value=mock_settings),
            patch("src.database.connection.create_async_engine", side_effect=Exception("Init failed")),
        ):

            # Clear any existing global instance
            import src.database.connection

            src.database.connection._db_manager = None

            with pytest.raises(DatabaseConnectionError):
                async with get_db_session():
                    pass
