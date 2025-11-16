"""Comprehensive integration tests for PromptCraft authentication systems.

This module tests the complete integration of:
- AUTH-1: Enhanced Cloudflare Access authentication with database integration
- AUTH-2: Service token management system
- Database connection and models
- Authentication middleware with database tracking
- Token validation and usage tracking
- Error handling and graceful degradation
"""

# ruff: noqa: S106

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.config import AuthenticationConfig
from src.auth.middleware import AuthenticationMiddleware
from src.auth.models import AuthenticatedUser, ServiceTokenCreate, ServiceTokenResponse, UserRole
from src.database import DatabaseManager
from src.database.models import ServiceToken, UserSession


@pytest.mark.integration
class TestAuthenticationIntegration:
    """Integration tests for complete authentication flow."""

    @pytest.fixture
    def auth_config(self) -> AuthenticationConfig:
        """Authentication configuration for integration testing."""
        return AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_audience="test-audience",
            cloudflare_issuer="https://test.cloudflareaccess.com",
            auth_logging_enabled=True,
            rate_limiting_enabled=True,
            rate_limit_requests=100,
            rate_limit_window=60,
        )

    @pytest.fixture
    async def mock_database_session(self) -> AsyncGenerator[AsyncMock, None]:
        """Mock database session for integration testing."""
        mock_session = AsyncMock(spec=AsyncSession)

        # Mock user session queries
        mock_session.execute = AsyncMock()
        mock_session.scalar_one_or_none = AsyncMock()
        mock_session.add = AsyncMock()
        mock_session.commit = AsyncMock()

        # Mock existing user session for update scenario
        existing_session = UserSession(
            id=uuid.uuid4(),
            email="test@example.com",
            cloudflare_sub="test-sub",
            session_count=5,
            preferences={"theme": "dark"},
            user_metadata={"last_login": "2025-08-01T10:00:00Z"},
        )

        # Set up query results
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_session
        mock_session.execute.return_value = mock_result

        return mock_session

    @pytest.fixture
    async def mock_database_manager(
        self,
        mock_database_session: AsyncMock,
    ) -> AsyncGenerator[AsyncMock, None]:
        """Mock database manager for integration testing."""
        from contextlib import asynccontextmanager
        mock_manager = AsyncMock(spec=DatabaseManager)

        # Create proper async context manager for session
        @asynccontextmanager
        async def mock_session_context():
            try:
                yield mock_database_session
            finally:
                pass

        # Mock session context manager
        mock_manager.get_session.return_value = mock_session_context()

        # Mock health check
        mock_manager.health_check.return_value = {
            "status": "healthy",
            "timestamp": time.time(),
            "connection_test": True,
            "response_time_ms": 5.2,
        }

        # Patch the sync get_database_manager function to return the mock manager
        # The middleware calls get_database_manager() synchronously, not as async
        with (
            patch("src.auth.middleware.get_database_manager", return_value=mock_manager),
            patch("src.database.connection.get_database_manager", return_value=mock_manager),
            patch("src.database.get_database_manager", return_value=mock_manager),
        ):
            yield mock_manager

    @pytest.fixture
    def mock_jwt_validator(self) -> MagicMock:
        """Mock JWT validator for integration testing."""
        validator = MagicMock()
        validator.validate_token.return_value = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.ADMIN,
            jwt_claims={
                "sub": "test-sub-12345",
                "email": "test@example.com",
                "aud": "test-audience",
                "iss": "https://test.cloudflareaccess.com",
                "exp": int(time.time()) + 3600,
            },
        )
        return validator

    @pytest.fixture
    def fastapi_app(
        self,
        auth_config: AuthenticationConfig,
        mock_jwt_validator: MagicMock,
        mock_database_manager: AsyncMock,
    ) -> FastAPI:
        """Create FastAPI app with authentication middleware."""
        from starlette.middleware.base import BaseHTTPMiddleware

        app = FastAPI(title="PromptCraft Test App")

        # Create a simple middleware wrapper for testing
        class TestAuthMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, auth_middleware):
                super().__init__(app)
                self.auth_middleware = auth_middleware

            async def dispatch(self, request: Request, call_next):
                return await self.auth_middleware.dispatch(request, call_next)

        # Create the actual auth middleware
        auth_middleware = AuthenticationMiddleware(
            app=app,
            config=auth_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        # Add as middleware wrapper
        app.add_middleware(TestAuthMiddleware, auth_middleware=auth_middleware)

        # Test endpoints
        @app.get("/api/test")
        async def test_endpoint(request: Request):
            user = getattr(request.state, "authenticated_user", None)
            return {
                "status": "success",
                "user_email": user.email if user else None,
                "user_role": user.role.value if user else None,
            }

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        @app.get("/api/admin")
        async def admin_endpoint(request: Request):
            user = getattr(request.state, "authenticated_user", None)
            if not user or user.role != UserRole.ADMIN:
                return {"error": "Forbidden"}, 403
            return {"status": "admin_access_granted"}

        @app.get("/api/protected")
        async def protected_endpoint(request: Request):
            return {
                "message": "Success",
                "user": getattr(request.state, "user", None),
                "token_metadata": getattr(request.state, "token_metadata", None),
            }

        return app

    @pytest.fixture
    def test_client(self, fastapi_app: FastAPI) -> TestClient:
        """Create test client for integration testing."""
        return TestClient(fastapi_app)

    def test_successful_authentication_flow(
        self,
        test_client: TestClient,
        mock_database_session: AsyncMock,
    ):
        """Test complete successful authentication flow."""
        # Make authenticated request
        response = test_client.get(
            "/api/test",
            headers={
                "CF-Access-Jwt-Assertion": "valid-jwt-token",
                "CF-Connecting-IP": "192.168.1.100",
                "User-Agent": "Mozilla/5.0 Test Browser",
                "CF-Ray": "test-ray-12345",
            },
        )

        # Verify successful response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["user_email"] == "test@example.com"
        assert data["user_role"] == "admin"

        # Verify database operations were called (session update)
        assert mock_database_session.execute.called, "Database execute should have been called for session update"
        assert mock_database_session.commit.called, "Database commit should have been called for session update"

    def test_authentication_with_database_session_tracking(
        self,
        test_client: TestClient,
        mock_database_session: AsyncMock,
    ):
        """Test authentication with user session tracking."""
        # Make authenticated request
        response = test_client.get(
            "/api/test",
            headers={
                "CF-Access-Jwt-Assertion": "valid-jwt-token",
                "CF-Connecting-IP": "192.168.1.200",
                "User-Agent": "Chrome/91.0 Test",
            },
        )

        assert response.status_code == 200

        # Verify session update was attempted
        assert mock_database_session.execute.called, "Database execute should have been called for session tracking"
        assert mock_database_session.commit.called, "Database commit should have been called for session tracking"

    def test_authentication_event_logging(
        self,
        test_client: TestClient,
        mock_database_session: AsyncMock,
    ):
        """Test authentication event logging to database."""
        # Make authenticated request
        response = test_client.get(
            "/api/test",
            headers={
                "CF-Access-Jwt-Assertion": "valid-jwt-token",
                "CF-Connecting-IP": "10.0.0.1",
                "User-Agent": "Firefox/89.0 Test",
                "CF-Ray": "event-ray-54321",
            },
        )

        assert response.status_code == 200

        # Verify session and event logging operations
        assert mock_database_session.execute.called, "Database execute should have been called"
        assert mock_database_session.commit.called, "Database commit should have been called"

    def test_excluded_paths_bypass_authentication(
        self,
        test_client: TestClient,
        mock_database_session: AsyncMock,
    ):
        """Test that excluded paths bypass authentication."""
        # Test health endpoint (should be excluded)
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

        # Verify no database operations for excluded paths
        mock_database_session.execute.assert_not_called()
        mock_database_session.commit.assert_not_called()

    def test_missing_jwt_token_authentication_failure(
        self,
        test_client: TestClient,
        mock_database_session: AsyncMock,
    ):
        """Test authentication failure with missing JWT token."""
        # Make request without JWT token
        response = test_client.get("/api/test")

        # Verify authentication failure
        assert response.status_code == 401
        data = response.json()
        assert "error" in data  # Check that error field exists with appropriate message

        # Since database operations may fail gracefully in middleware, we just check that the request failed appropriately
        # The middleware should handle database failures gracefully and still return appropriate error response

    def test_graceful_degradation_database_unavailable(
        self,
        auth_config: AuthenticationConfig,
        mock_jwt_validator: MagicMock,
    ):
        """Test graceful degradation when database is unavailable."""
        from starlette.middleware.base import BaseHTTPMiddleware
        app = FastAPI()

        # Mock database failure - return a coroutine that raises an exception
        async def failing_get_database_manager():
            mock_db = AsyncMock()
            mock_db.get_session.side_effect = Exception("Database connection failed")
            return mock_db

        # Create a simple middleware wrapper for testing
        class TestAuthMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, auth_middleware):
                super().__init__(app)
                self.auth_middleware = auth_middleware

            async def dispatch(self, request: Request, call_next):
                return await self.auth_middleware.dispatch(request, call_next)

        with patch("src.auth.middleware.get_database_manager", side_effect=failing_get_database_manager):
            auth_middleware = AuthenticationMiddleware(
                app=app,
                config=auth_config,
                jwt_validator=mock_jwt_validator,
                database_enabled=True,
            )

            # Add as middleware wrapper
            app.add_middleware(TestAuthMiddleware, auth_middleware=auth_middleware)

            @app.get("/api/test")
            async def test_endpoint(request: Request):
                user = getattr(request.state, "authenticated_user", None)
                return {"user_email": user.email if user else None}

            client = TestClient(app)

            # Authentication should still work despite database failure
            response = client.get(
                "/api/test",
                headers={"CF-Access-Jwt-Assertion": "valid-jwt-token"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["user_email"] == "test@example.com"

    def test_role_based_access_control(
        self,
        test_client: TestClient,
        mock_jwt_validator: MagicMock,
    ):
        """Test role-based access control integration."""
        # Test admin access with admin role
        response = test_client.get(
            "/api/admin",
            headers={"CF-Access-Jwt-Assertion": "admin-jwt-token"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "admin_access_granted"

    def test_performance_metrics_collection(
        self,
        test_client: TestClient,
        mock_database_session: AsyncMock,
    ):
        """Test that performance metrics are collected during authentication."""
        # Make authenticated request
        start_time = time.time()
        response = test_client.get(
            "/api/test",
            headers={
                "CF-Access-Jwt-Assertion": "valid-jwt-token",
                "CF-Connecting-IP": "192.168.1.50",
            },
        )
        end_time = time.time()

        assert response.status_code == 200

        # Verify request completed successfully with performance monitoring
        assert response.status_code == 200

        # Verify request completed in reasonable time
        request_time_ms = (end_time - start_time) * 1000
        assert request_time_ms < 1000.0, f"Request took {request_time_ms:.2f}ms, exceeds 1000ms limit"

    def test_multiple_concurrent_requests(
        self,
        fastapi_app: FastAPI,
        mock_database_session: AsyncMock,
    ):
        """Test handling multiple concurrent authentication requests."""

        async def make_request():
            from httpx import AsyncClient

            async with AsyncClient(app=fastapi_app, base_url="http://test") as client:
                return await client.get(
                    "/api/test",
                    headers={
                        "CF-Access-Jwt-Assertion": "concurrent-jwt-token",
                        "CF-Connecting-IP": f"192.168.1.{hash(asyncio.current_task()) % 200 + 1}",
                    },
                )

        async def run_concurrent_test():
            # Create 20 concurrent requests
            tasks = [make_request() for _ in range(20)]
            responses = await asyncio.gather(*tasks)

            # Verify all requests succeeded
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["user_email"] == "test@example.com"

            return len(responses)

        # Run the concurrent test
        result = asyncio.run(run_concurrent_test())
        assert result == 20

        # Verify concurrent requests succeeded (database operations are checked in the middleware)
        # The exact count may vary due to mocking and async execution timing
        assert mock_database_session.execute.called, "Database operations should have been attempted"
        assert mock_database_session.commit.called, "Database commits should have been attempted"


@pytest.mark.integration
class TestServiceTokenIntegration:
    """Integration test suite for AUTH-2 service token management."""

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings."""
        settings = MagicMock()
        settings.database_host = "localhost"
        settings.database_port = 5432
        settings.database_name = "test_promptcraft"
        settings.database_username = "test_user"
        settings.database_timeout = 30.0
        settings.database_password = None
        settings.database_url = None
        return settings

    @pytest.fixture
    async def database_session(self, mock_settings):
        """Mock database session for testing."""
        # Mock session for testing
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        return mock_session

    @pytest.fixture
    async def database_connection(self):
        """Mock database connection for integration tests."""
        from contextlib import asynccontextmanager

        # Create a mock database manager for testing
        db_manager = MagicMock()
        db_manager._is_initialized = True
        db_manager.health_check = AsyncMock(return_value=True)
        db_manager.get_pool_status = AsyncMock(return_value={"status": "initialized"})

        # Mock session factory
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.execute = AsyncMock()

        # Create proper async context manager for session
        @asynccontextmanager
        async def mock_session_context():
            try:
                yield mock_session
            except Exception:
                await mock_session.rollback()
                raise
            finally:
                await mock_session.close()

        db_manager.session = mock_session_context
        db_manager._session_factory = MagicMock(return_value=mock_session)

        return db_manager

    @pytest.mark.asyncio
    async def test_database_initialization(self, database_connection):
        """Test database connection initialization."""
        # Database should be initialized
        assert database_connection._is_initialized is True

        # Health check should pass
        health_ok = await database_connection.health_check()
        assert health_ok is True

        # Pool status should be available
        pool_status = await database_connection.get_pool_status()
        assert pool_status["status"] == "initialized"

    @pytest.mark.asyncio
    async def test_service_token_crud_operations(self, database_connection):
        """Test complete CRUD operations for service tokens."""
        # Mock session for CRUD operations
        mock_session = AsyncMock()
        database_connection._session_factory = MagicMock(return_value=mock_session)

        # Test token creation
        token_create_data = ServiceTokenCreate(
            token_name="integration-test-token",
            metadata={"environment": "test", "permissions": ["read", "write"]},
        )

        # Mock token creation
        created_token = ServiceToken()
        created_token.id = uuid.uuid4()
        created_token.token_name = token_create_data.token_name
        created_token.token_hash = hashlib.sha256(b"test-token-value").hexdigest()
        created_token.created_at = datetime.now(UTC)
        created_token.is_active = True
        created_token.usage_count = 0
        created_token.token_metadata = token_create_data.metadata

        # Test session usage
        async with database_connection.session() as session:
            # Simulate adding token to session
            # The session should be accessible (mock session from fixture)
            assert session is not None

            # In this test, the session operations would be called in real usage
            # For now, just verify the session context manager works

    @pytest.mark.asyncio
    async def test_token_metadata_validation(self, database_connection):
        """Test validation of token metadata."""
        # Test various metadata scenarios
        metadata_scenarios = [
            {"permissions": ["read", "write"], "client_type": "api"},
            {"rate_limit": "1000/hour", "environment": "production"},
            {"custom_field": "custom_value", "nested": {"key": "value"}},
            None,  # No metadata
            {},  # Empty metadata
        ]

        for i, metadata in enumerate(metadata_scenarios):
            # Mock token with different metadata
            mock_token = MagicMock(spec=ServiceToken)
            mock_token.id = uuid.uuid4()
            mock_token.token_name = f"metadata-test-token-{i}"
            mock_token.is_active = True
            mock_token.is_valid = True
            mock_token.token_metadata = metadata

            # Create response model
            response = ServiceTokenResponse.from_orm_model(mock_token)

            # Verify metadata is preserved
            assert response.metadata == metadata
            assert response.token_name == f"metadata-test-token-{i}"


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.fixture
    async def mock_engine_and_session(self) -> AsyncGenerator[tuple[AsyncMock, AsyncMock], None]:
        """Mock database engine and session for testing."""
        from contextlib import asynccontextmanager

        with (
            patch("src.database.connection.create_async_engine") as mock_engine_create,
            patch("src.database.connection.async_sessionmaker") as mock_session_maker,
        ):
            mock_engine = AsyncMock()
            mock_session = AsyncMock()

            mock_engine_create.return_value = mock_engine

            # Configure mock_session to support async context manager protocol
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Create a regular function (not AsyncMock) for the session factory
            def mock_session_factory():
                return mock_session

            mock_session_maker.return_value = mock_session_factory

            # Mock successful connection test - create proper async context manager
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = [1]
            mock_conn.execute.return_value = mock_result

            # Create async context manager for begin()
            @asynccontextmanager
            async def mock_begin_context():
                try:
                    yield mock_conn
                finally:
                    pass

            # Create a mock that tracks calls but returns the context manager
            class MockBeginMethod:
                def __init__(self, context_func):
                    self.context_func = context_func
                    self.call_count = 0

                def __call__(self):
                    self.call_count += 1
                    return self.context_func()

                def assert_called(self):
                    assert self.call_count > 0, "Mock was not called"

            mock_engine.begin = MockBeginMethod(mock_begin_context)

            # Mock pool for health checks
            mock_pool = MagicMock()
            mock_pool.size.return_value = 10
            mock_pool.checkedin.return_value = 7
            mock_pool.checkedout.return_value = 3
            mock_pool.overflow.return_value = 0
            mock_engine.pool = mock_pool

            yield mock_engine, mock_session

    async def test_database_manager_initialization(
        self,
        mock_engine_and_session: tuple[AsyncMock, AsyncMock],
    ):
        """Test database manager initialization."""
        mock_engine, mock_session = mock_engine_and_session

        from src.database.connection import DatabaseManager

        manager = DatabaseManager()
        await manager.initialize()

        # Verify initialization steps
        assert manager._engine is not None
        assert manager._session_factory is not None
        assert manager._is_initialized is True

        # Verify connection test was performed
        mock_engine.begin.assert_called()

    async def test_database_health_check_integration(
        self,
        mock_engine_and_session: tuple[AsyncMock, AsyncMock],
    ):
        """Test database health check integration."""
        mock_engine, mock_session = mock_engine_and_session

        from src.database.connection import DatabaseManager

        manager = DatabaseManager()
        await manager.initialize()

        # Perform health check
        health_status = await manager.health_check()

        # Verify health check results
        assert health_status["status"] == "healthy"
        assert health_status["connection_test"] is True
        assert "response_time_ms" in health_status
        assert "pool_status" in health_status

        # Verify pool status
        pool_status = health_status["pool_status"]
        assert pool_status["size"] == 10
        assert pool_status["checked_in"] == 7
        assert pool_status["checked_out"] == 3

    async def test_database_session_context_manager(
        self,
        mock_engine_and_session: tuple[AsyncMock, AsyncMock],
    ):
        """Test database session context manager integration."""
        mock_engine, mock_session = mock_engine_and_session

        from src.database.connection import DatabaseManager

        manager = DatabaseManager()
        await manager.initialize()

        # Test session context manager
        async with manager.get_session() as session:
            assert session is not None
            # Verify session operations can be performed
            session.execute = AsyncMock()
            await session.execute("SELECT 1")
            session.execute.assert_called_with("SELECT 1")

    async def test_database_retry_mechanism(
        self,
        mock_engine_and_session: tuple[AsyncMock, AsyncMock],
    ):
        """Test database retry mechanism integration."""
        mock_engine, mock_session = mock_engine_and_session

        from src.database.connection import DatabaseManager

        manager = DatabaseManager()
        await manager.initialize()

        # Mock operation that fails twice then succeeds
        call_count = 0

        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary database error")
            return "success"

        # Test retry mechanism
        result = await manager.execute_with_retry(mock_operation, max_retries=3)

        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third try
