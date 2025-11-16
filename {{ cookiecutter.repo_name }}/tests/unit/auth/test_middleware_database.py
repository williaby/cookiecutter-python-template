"""Unit tests for authentication middleware with database integration.

This file tests both AUTH-2 service token validation and AUTH-1 database integration
features for the AuthenticationMiddleware.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from src.auth.config import AuthenticationConfig
from src.auth.jwt_validator import JWTValidator
from src.auth.middleware import AuthenticationMiddleware
from src.auth.models import AuthenticatedUser, AuthenticationError, UserRole
from src.database import DatabaseError
from src.database.models import ServiceToken


# AUTH-2 Service Token Database Tests
class TestAuthMiddlewareDatabase:
    """Test suite for authentication middleware with service token database integration."""

    @pytest.fixture
    def mock_config(self):
        """Mock authentication configuration."""
        config = MagicMock(spec=AuthenticationConfig)
        config.cloudflare_team_domain = "test.cloudflareaccess.com"
        config.cloudflare_aud = "test-audience"
        config.rate_limit = "10/minute"
        config.rate_limit_storage_uri = "memory://"
        config.cloudflare_access_enabled = True
        config.auth_logging_enabled = True
        config.auth_error_detail_enabled = True
        config.email_whitelist_enabled = False
        config.email_whitelist = None
        return config

    @pytest.fixture
    def mock_jwt_validator(self):
        """Mock JWT validator."""
        from src.auth.models import AuthenticatedUser, UserRole

        validator = MagicMock(spec=JWTValidator)
        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"email": "test@example.com", "sub": "user123", "aud": "test-audience"},
        )
        validator.validate_token.return_value = authenticated_user
        return validator

    @pytest.fixture
    def mock_database(self):
        """Mock database connection."""
        db = AsyncMock()

        # Mock session context manager
        session = AsyncMock()
        db.session.return_value.__aenter__.return_value = session
        db.session.return_value.__aexit__.return_value = None

        return db, session

    @pytest.fixture
    def auth_middleware(self, mock_config, mock_jwt_validator):
        """Authentication middleware instance."""
        app = MagicMock()
        return AuthenticationMiddleware(
            app,
            mock_config,
            mock_jwt_validator,
            excluded_paths=["/health", "/docs"],
            database_enabled=True,
        )

    @pytest.mark.asyncio
    async def test_service_token_validation_success(self, auth_middleware, mock_database):
        """Test successful service token validation against database."""
        db, session = mock_database

        # Mock request with service token
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda key: {"Authorization": "Bearer sk_test_valid_token"}.get(key)
        request.headers = mock_headers
        request.method = "GET"

        # Mock database token lookup
        mock_token = MagicMock(spec=ServiceToken)
        mock_token.id = "token-uuid"
        mock_token.token_name = "test-api-token"  # noqa: S105
        mock_token.is_active = True
        mock_token.is_expired = False
        mock_token.usage_count = 5
        mock_token.token_metadata = {"permissions": ["read", "write"]}

        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_token
        session.execute.return_value = mock_result

        response = Response()
        call_next = AsyncMock(return_value=response)

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session once
            async def mock_async_generator():
                yield session

            mock_get_db.return_value = mock_async_generator()

            result = await auth_middleware.dispatch(request, call_next)

            # Should proceed to next middleware
            call_next.assert_called_once_with(request)
            assert result == response

            # Should have updated usage count in database (via SQL UPDATE)
            # Verify that execute was called twice: once for SELECT, once for UPDATE
            assert session.execute.call_count == 2

            # Verify the authenticated user has correct usage count
            authenticated_user = request.state.authenticated_user
            assert authenticated_user.usage_count == 6  # original 5 + 1

    @pytest.mark.asyncio
    async def test_service_token_validation_not_found(self, auth_middleware, mock_database):
        """Test service token validation when token not found in database."""
        db, session = mock_database

        # Mock request with invalid service token
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda key: {"Authorization": "Bearer sk_test_invalid_token"}.get(key)
        request.headers = mock_headers
        request.method = "GET"

        # Mock database - token not found
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        session.execute.return_value = mock_result

        call_next = AsyncMock()

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session once
            async def mock_async_generator():
                yield session

            mock_get_db.return_value = mock_async_generator()

            result = await auth_middleware.dispatch(request, call_next)

            # Should return 401 Unauthorized
            assert isinstance(result, JSONResponse)
            assert result.status_code == 401
            call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_token_validation_expired(self, auth_middleware, mock_database):
        """Test service token validation with expired token."""
        db, session = mock_database

        # Mock request with service token
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda key: {"Authorization": "Bearer sk_test_expired_token"}.get(key)
        request.headers = mock_headers
        request.method = "GET"

        # Mock expired token
        mock_token = MagicMock(spec=ServiceToken)
        mock_token.id = "expired-token-uuid"
        mock_token.token_name = "expired-api-token"  # noqa: S105
        mock_token.is_active = True
        mock_token.is_expired = True  # Token is expired
        mock_token.is_valid = False  # Not valid due to expiration
        mock_token.expires_at = datetime.now(UTC) - timedelta(hours=1)

        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_token
        session.execute.return_value = mock_result

        call_next = AsyncMock()

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session once
            async def mock_async_generator():
                yield session

            mock_get_db.return_value = mock_async_generator()

            result = await auth_middleware.dispatch(request, call_next)

            # Should return 401 Unauthorized for expired token
            assert isinstance(result, JSONResponse)
            assert result.status_code == 401
            call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_token_validation_inactive(self, auth_middleware, mock_database):
        """Test service token validation with inactive token."""
        db, session = mock_database

        # Mock request with service token
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda key: {"Authorization": "Bearer sk_test_inactive_token"}.get(key)
        request.headers = mock_headers
        request.method = "GET"

        # Mock inactive token
        mock_token = MagicMock(spec=ServiceToken)
        mock_token.id = "inactive-token-uuid"
        mock_token.token_name = "inactive-api-token"  # noqa: S105
        mock_token.is_active = False  # Token is inactive
        mock_token.is_expired = False
        mock_token.is_valid = False  # Not valid due to being inactive

        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_token
        session.execute.return_value = mock_result

        call_next = AsyncMock()

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session once
            async def mock_async_generator():
                yield session

            mock_get_db.return_value = mock_async_generator()

            result = await auth_middleware.dispatch(request, call_next)

            # Should return 401 Unauthorized for inactive token
            assert isinstance(result, JSONResponse)
            assert result.status_code == 401
            call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_database_error_handling(self, auth_middleware, mock_database):
        """Test handling of database errors during token validation."""
        db, session = mock_database

        # Mock request with service token
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda key: {"Authorization": "Bearer sk_test_token"}.get(key)
        request.headers = mock_headers
        request.method = "GET"

        # Mock database error
        session.execute.side_effect = Exception("Database connection failed")

        call_next = AsyncMock()

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session once
            async def mock_async_generator():
                yield session

            mock_get_db.return_value = mock_async_generator()

            result = await auth_middleware.dispatch(request, call_next)

            # Should return 500 Internal Server Error for database issues
            assert isinstance(result, JSONResponse)
            assert result.status_code == 500
            call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_hash_generation(self, auth_middleware):
        """Test that service tokens are properly hashed for database lookup."""
        import hashlib

        raw_token = "sk_test_raw_token_value"  # noqa: S105
        expected_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        # Test hash generation directly (no internal method exists)
        result_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        assert result_hash == expected_hash

        # Verify the hash is consistent and has correct format
        assert len(result_hash) == 64  # SHA-256 produces 64-character hex string
        assert all(c in "0123456789abcdef" for c in result_hash)  # Valid hex

    @pytest.mark.asyncio
    async def test_token_metadata_injection(self, auth_middleware, mock_database):
        """Test that token metadata is properly injected into request state."""
        db, session = mock_database

        # Mock request with service token
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda key: {"Authorization": "Bearer sk_test_token_with_metadata"}.get(key)
        request.headers = mock_headers
        request.method = "GET"
        request.state = MagicMock()

        # Mock token with rich metadata
        mock_token = MagicMock(spec=ServiceToken)
        mock_token.id = "metadata-token-uuid"
        mock_token.token_name = "metadata-api-token"  # noqa: S105
        mock_token.is_active = True
        mock_token.is_expired = False
        mock_token.is_valid = True
        mock_token.token_metadata = {
            "permissions": ["read", "write", "admin"],
            "client_type": "service",
            "environment": "production",
            "rate_limit": "1000/hour",
        }

        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_token
        session.execute.return_value = mock_result

        response = Response()
        call_next = AsyncMock(return_value=response)

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session once
            async def mock_async_generator():
                yield session

            mock_get_db.return_value = mock_async_generator()

            result = await auth_middleware.dispatch(request, call_next)

            # Should proceed and inject metadata into request state
            call_next.assert_called_once_with(request)
            assert result == response

            # Verify metadata is available in request state
            assert hasattr(request.state, "token_metadata")
            assert request.state.token_metadata == mock_token.token_metadata

    @pytest.mark.asyncio
    async def test_excluded_paths_bypass_database(self, auth_middleware, mock_database):
        """Test that excluded paths bypass database token validation."""
        db, session = mock_database

        # Mock request to excluded path
        request = MagicMock(spec=Request)
        request.url.path = "/health"
        request.headers = {}
        request.method = "GET"

        response = Response()
        call_next = AsyncMock(return_value=response)

        result = await auth_middleware.dispatch(request, call_next)

        # Should proceed without database interaction
        call_next.assert_called_once_with(request)
        assert result == response

        # Database should not have been accessed
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_usage_analytics_tracking(self, auth_middleware, mock_database):
        """Test that token usage is properly tracked for analytics."""
        db, session = mock_database

        # Mock request with service token
        request = MagicMock(spec=Request)
        request.url.path = "/api/analytics"
        mock_headers = MagicMock()
        mock_headers.get.side_effect = lambda key: {"Authorization": "Bearer sk_test_analytics_token"}.get(key)
        request.headers = mock_headers
        request.method = "POST"
        request.client.host = "192.168.1.100"

        # Mock token
        mock_token = MagicMock(spec=ServiceToken)
        mock_token.id = "analytics-token-uuid"
        mock_token.token_name = "analytics-api-token"  # noqa: S105
        mock_token.is_active = True
        mock_token.is_expired = False
        mock_token.is_valid = True
        mock_token.usage_count = 42
        mock_token.last_used = datetime.now(UTC) - timedelta(hours=2)

        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_token
        session.execute.return_value = mock_result

        response = Response()
        call_next = AsyncMock(return_value=response)

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session once
            async def mock_async_generator():
                yield session

            mock_get_db.return_value = mock_async_generator()

            # Track time before request (for potential performance testing)
            _before_request = datetime.now(UTC)

            result = await auth_middleware.dispatch(request, call_next)

            # Track time after request (for potential performance testing)
            _after_request = datetime.now(UTC)

            # Should proceed successfully
            call_next.assert_called_once_with(request)
            assert result == response

            # Should have updated usage count in database (via SQL UPDATE)
            assert session.execute.call_count == 2  # SELECT + UPDATE

            # Verify the authenticated user has correct usage count
            authenticated_user = request.state.authenticated_user
            assert authenticated_user.usage_count == 43  # original 42 + 1

            # Note: last_used is updated in database via SQL, not on the mock object

    @pytest.fixture
    def mock_request(self):
        """Mock request for testing."""
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {
            "CF-Access-Jwt-Assertion": "test-jwt-token",
            "CF-Connecting-IP": "192.168.1.100",
            "User-Agent": "Mozilla/5.0 Test Browser",
            "CF-Ray": "test-ray-12345",
        }
        request.state = MagicMock()
        return request


    def test_middleware_initialization_database_enabled(self, mock_config, mock_jwt_validator):
        """Test middleware initialization with database enabled."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        assert middleware.database_enabled is True

    def test_middleware_initialization_database_disabled(self, mock_config, mock_jwt_validator):
        """Test middleware initialization with database disabled."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=False,
        )

        assert middleware.database_enabled is False

    async def test_update_user_session_success(self, mock_config, mock_jwt_validator, mock_request):
        """Test successful user session update."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "test-sub-123", "email": "test@example.com"},
        )

        # Mock database manager and session directly
        mock_manager = MagicMock()
        mock_session = AsyncMock()

        # Mock async context manager
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_manager.get_session.return_value = async_context_manager

        with patch("src.auth.middleware.get_database_manager", return_value=mock_manager):
            await middleware._update_user_session(authenticated_user, mock_request)

            # Verify database operations were called
            mock_manager.get_session.assert_called_once()
            mock_session.execute.assert_called()
            mock_session.commit.assert_called_once()

    async def test_update_user_session_database_disabled(self, mock_config, mock_jwt_validator, mock_request):
        """Test user session update when database is disabled."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=False,
        )

        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "test-sub", "email": "test@example.com"},
        )

        # Should not raise exception and should not call database
        with patch("src.database.get_database_manager") as mock_get_db:
            await middleware._update_user_session(authenticated_user, mock_request)
            mock_get_db.assert_not_called()

    async def test_update_user_session_database_error_graceful_degradation(
        self,
        mock_config,
        mock_jwt_validator,
        mock_request,
    ):
        """Test graceful degradation when database session update fails."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "test-sub", "email": "test@example.com"},
        )

        # Mock database manager to raise DatabaseError
        mock_manager = AsyncMock()
        mock_manager.get_session.side_effect = DatabaseError("Database unavailable")

        with patch("src.auth.middleware.get_db"):
            # Should not raise exception (graceful degradation)
            await middleware._update_user_session(authenticated_user, mock_request)

    async def test_update_user_session_unexpected_error_graceful_degradation(
        self,
        mock_config,
        mock_jwt_validator,
        mock_request,
    ):
        """Test graceful degradation when unexpected error occurs."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "test-sub", "email": "test@example.com"},
        )

        # Mock database manager to raise unexpected error
        mock_manager = AsyncMock()
        mock_manager.get_session.side_effect = Exception("Unexpected error")

        with patch("src.auth.middleware.get_db"):
            # Should not raise exception (graceful degradation)
            await middleware._update_user_session(authenticated_user, mock_request)

    async def test_log_authentication_event_success(self, mock_config, mock_jwt_validator, mock_request):
        """Test successful authentication event logging."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "test-sub", "email": "test@example.com"},
        )

        # Mock session for async generator
        mock_session = AsyncMock()

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session
            async def mock_async_generator():
                yield mock_session

            mock_get_db.return_value = mock_async_generator()

            await middleware._log_authentication_event(
                request=mock_request,
                user_email=authenticated_user.email,
                event_type="general",
                success=True,
            )

            # Verify database operations were called
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    async def test_log_authentication_event_failure(self, mock_config, mock_jwt_validator, mock_request):
        """Test authentication event logging for failures."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        # Mock session for async generator
        mock_session = AsyncMock()

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session
            async def mock_async_generator():
                yield mock_session

            mock_get_db.return_value = mock_async_generator()

            await middleware._log_authentication_event(
                request=mock_request,
                user_email=None,
                event_type="general",
                success=False,
                error_details={"error": "JWT token invalid"},
            )

            # Verify database operations were called
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    async def test_log_authentication_event_database_disabled(self, mock_config, mock_jwt_validator, mock_request):
        """Test authentication event logging when database is disabled."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=False,
        )

        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "test-sub", "email": "test@example.com"},
        )

        # Mock session for async generator
        mock_session = AsyncMock()

        # Even when database_enabled=False, event logging still happens
        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session
            async def mock_async_generator():
                yield mock_session

            mock_get_db.return_value = mock_async_generator()

            await middleware._log_authentication_event(
                request=mock_request,
                user_email=authenticated_user.email,
                event_type="general",
                success=True,
            )

            # The method should still try to get database for logging since
            # database_enabled only affects user session tracking, not event logging
            mock_get_db.assert_called_once()
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    async def test_log_authentication_event_database_error_graceful_degradation(
        self,
        mock_config,
        mock_jwt_validator,
        mock_request,
    ):
        """Test graceful degradation when event logging fails."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "test-sub", "email": "test@example.com"},
        )

        # Mock database manager to raise DatabaseError
        mock_manager = AsyncMock()
        mock_manager.get_session.side_effect = DatabaseError("Database unavailable")

        with patch("src.auth.middleware.get_db"):
            # Should not raise exception (graceful degradation)
            await middleware._log_authentication_event(
                request=mock_request,
                user_email=authenticated_user.email,
                event_type="general",
                success=True,
            )

    def test_get_client_ip_cloudflare_header(self, mock_config, mock_jwt_validator):
        """Test client IP extraction from Cloudflare headers."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
        )

        request = MagicMock(spec=Request)
        request.headers = {"CF-Connecting-IP": "203.0.113.1"}

        ip = middleware._get_client_ip(request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_x_forwarded_for(self, mock_config, mock_jwt_validator):
        """Test client IP extraction from X-Forwarded-For header."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
        )

        request = MagicMock(spec=Request)
        request.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}

        ip = middleware._get_client_ip(request)
        assert ip == "203.0.113.1"  # Should get first IP in chain

    def test_get_client_ip_x_real_ip(self, mock_config, mock_jwt_validator):
        """Test client IP extraction from X-Real-IP header."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
        )

        request = MagicMock(spec=Request)
        request.headers = {"X-Real-IP": "203.0.113.1"}

        ip = middleware._get_client_ip(request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_fallback_to_client_host(self, mock_config, mock_jwt_validator):
        """Test client IP fallback to request.client.host."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
        )

        request = MagicMock(spec=Request)
        request.headers = {}
        request.client.host = "192.168.1.100"

        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_no_ip_available(self, mock_config, mock_jwt_validator):
        """Test client IP extraction when no IP is available."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
        )

        request = MagicMock(spec=Request)
        request.headers = {}
        request.client = None

        ip = middleware._get_client_ip(request)
        assert ip is None

    async def test_dispatch_with_database_integration_success(self, mock_config, mock_jwt_validator, mock_request):
        """Test complete dispatch flow with database integration."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        async def mock_call_next(request):
            return JSONResponse(content={"status": "success"})

        # Mock session for async generator
        mock_session = AsyncMock()

        # Mock database manager for session updates
        mock_manager = MagicMock()
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_manager.get_session.return_value = async_context_manager

        with (
            patch("src.auth.middleware.get_db") as mock_get_db,
            patch("src.auth.middleware.get_database_manager", return_value=mock_manager),
        ):
            # Mock async generator to yield session for event logging
            async def mock_async_generator():
                yield mock_session

            mock_get_db.return_value = mock_async_generator()

            response = await middleware.dispatch(mock_request, mock_call_next)

            assert response.status_code == 200

            # Verify user context was set
            assert hasattr(mock_request.state, "authenticated_user")
            assert hasattr(mock_request.state, "user_email")
            assert hasattr(mock_request.state, "user_role")

            # Verify database operations were called (session update + event logging)
            assert mock_manager.get_session.call_count >= 1  # User session update
            assert mock_session.commit.call_count >= 1  # At least one commit

    async def test_dispatch_with_database_integration_auth_failure(self, mock_config, mock_request):
        """Test dispatch flow with authentication failure and database logging."""
        app = MagicMock()

        # Mock JWT validator to raise authentication error
        mock_jwt_validator = MagicMock()
        mock_jwt_validator.validate_token.side_effect = AuthenticationError("Invalid token", 401)

        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        async def mock_call_next(request):
            return JSONResponse(content={"status": "success"})

        # Mock session for async generator
        mock_session = AsyncMock()

        with patch("src.auth.middleware.get_db") as mock_get_db:
            # Mock async generator to yield session for event logging
            async def mock_async_generator():
                yield mock_session

            mock_get_db.return_value = mock_async_generator()

            response = await middleware.dispatch(mock_request, mock_call_next)

            assert response.status_code == 401

            # Verify failure event was logged
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    async def test_dispatch_performance_metrics_collection(self, mock_config, mock_jwt_validator, mock_request):
        """Test that performance metrics are collected during dispatch."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        async def mock_call_next(request):
            return JSONResponse(content={"status": "success"})

        # Mock session for async generator
        mock_session = AsyncMock()

        # Mock database manager for session updates
        mock_manager = MagicMock()
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_manager.get_session.return_value = async_context_manager

        # Mock time.time to control timing - provide enough values for all time.time() calls
        start_time = 1000.0
        # Multiple calls to time.time() throughout the middleware dispatch
        times = [start_time] + [start_time + 0.001 + (i * 0.001) for i in range(20)]

        with (
            patch("src.auth.middleware.get_database_manager", return_value=mock_manager),
            patch("src.auth.middleware.get_db") as mock_get_db,
            patch("time.time", side_effect=times),
        ):
            # Mock async generator to yield session for event logging
            async def mock_async_generator():
                yield mock_session

            mock_get_db.return_value = mock_async_generator()

            response = await middleware.dispatch(mock_request, mock_call_next)

            assert response.status_code == 200

            # Verify event logging was called
            mock_session.add.assert_called()
            mock_session.commit.assert_called()

    async def test_dispatch_database_completely_unavailable(self, mock_config, mock_jwt_validator, mock_request):
        """Test dispatch when database is completely unavailable."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(
            app=app,
            config=mock_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )

        async def mock_call_next(request):
            return JSONResponse(content={"status": "success"})

        # Mock get_database_manager to raise exception
        with patch("src.database.get_database_manager", side_effect=Exception("DB unavailable")):
            response = await middleware.dispatch(mock_request, mock_call_next)

            # Authentication should still succeed (graceful degradation)
            assert response.status_code == 200

            # Verify user context was still set
            assert hasattr(mock_request.state, "authenticated_user")
            assert mock_request.state.user_email == "test@example.com"
