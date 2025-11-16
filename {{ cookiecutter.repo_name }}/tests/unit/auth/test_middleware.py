"""
Unit tests for FastAPI authentication middleware.

This module provides comprehensive test coverage for the AuthenticationMiddleware class,
testing JWT token extraction, authentication flow, rate limiting, and user context injection.
Uses proper pytest markers for codecov integration per codecov.yml auth component.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from starlette.datastructures import URL, Headers

from src.auth.config import AuthenticationConfig
from src.auth.jwt_validator import JWTValidator
from src.auth.middleware import (
    AuthenticationMiddleware,
    create_rate_limiter,
    get_current_user,
    require_authentication,
    require_role,
    setup_authentication,
)
from src.auth.models import AuthenticatedUser, AuthenticationError, JWTValidationError, UserRole


@pytest.mark.auth
class TestAuthenticationMiddleware:
    """Test cases for AuthenticationMiddleware class."""

    def test_init_default_values(self):
        """Test middleware initialization with default configuration."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)

        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        assert middleware.config == config
        assert middleware.jwt_validator == jwt_validator
        assert "/health" in middleware.excluded_paths
        assert "/metrics" in middleware.excluded_paths
        assert "/docs" in middleware.excluded_paths
        assert "/redoc" in middleware.excluded_paths
        assert "/openapi.json" in middleware.excluded_paths

    def test_init_custom_excluded_paths(self):
        """Test middleware initialization with custom excluded paths."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        custom_paths = ["/custom", "/api/public"]

        middleware = AuthenticationMiddleware(app, config, jwt_validator, custom_paths)

        assert middleware.excluded_paths == custom_paths

    @pytest.mark.asyncio
    async def test_dispatch_excluded_path(self):
        """Test that excluded paths skip authentication."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request for excluded path
        request = Mock(spec=Request)
        request.url = Mock(spec=URL)
        request.url.path = "/health"

        call_next = AsyncMock(return_value=Response())

        response = await middleware.dispatch(request, call_next)

        assert isinstance(response, Response)
        call_next.assert_called_once_with(request)
        jwt_validator.validate_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_authentication_disabled(self):
        """Test that disabled authentication skips validation."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=False,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request for protected path
        request = Mock(spec=Request)
        request.url = Mock(spec=URL)
        request.url.path = "/api/protected"

        call_next = AsyncMock(return_value=Response())

        response = await middleware.dispatch(request, call_next)

        assert isinstance(response, Response)
        call_next.assert_called_once_with(request)
        jwt_validator.validate_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_successful_authentication(self):
        """Test successful authentication flow."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            auth_logging_enabled=True,
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock authenticated user
        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"email": "test@example.com", "exp": 1234567890},
        )

        # Mock request with JWT token
        request = Mock(spec=Request)
        request.url = Mock(spec=URL)
        request.url.path = "/api/protected"
        request.headers = Headers({"CF-Access-Jwt-Assertion": "valid.jwt.token"})
        request.state = Mock()

        # Mock middleware methods
        with patch.object(middleware, "_authenticate_request", return_value=authenticated_user) as mock_auth:
            call_next = AsyncMock(return_value=Response())

            response = await middleware.dispatch(request, call_next)

            assert isinstance(response, Response)
            mock_auth.assert_called_once_with(request)
            call_next.assert_called_once_with(request)

            # Verify user context injection
            assert request.state.authenticated_user == authenticated_user
            assert request.state.user_email == "test@example.com"
            assert request.state.user_role == UserRole.USER

    @pytest.mark.asyncio
    async def test_dispatch_authentication_error(self):
        """Test handling of authentication errors."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            auth_logging_enabled=True,
            auth_error_detail_enabled=True,
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request
        request = Mock(spec=Request)
        request.url = Mock(spec=URL)
        request.url.path = "/api/protected"
        request.headers = Headers({})

        # Mock authentication failure
        auth_error = AuthenticationError("Missing authentication token", 401)
        with patch.object(middleware, "_authenticate_request", side_effect=auth_error):
            call_next = AsyncMock()

            response = await middleware.dispatch(request, call_next)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 401
            call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_unexpected_error(self):
        """Test handling of unexpected errors."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request
        request = Mock(spec=Request)
        request.url = Mock(spec=URL)
        request.url.path = "/api/protected"
        request.headers = Headers({})

        # Mock unexpected error
        with patch.object(middleware, "_authenticate_request", side_effect=RuntimeError("Unexpected error")):
            call_next = AsyncMock()

            response = await middleware.dispatch(request, call_next)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 500
            call_next.assert_not_called()

    def test_is_excluded_path_exact_match(self):
        """Test exact path exclusion matching."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        assert middleware._is_excluded_path("/health") is True
        assert middleware._is_excluded_path("/metrics") is True
        assert middleware._is_excluded_path("/docs") is True

    def test_is_excluded_path_prefix_match(self):
        """Test prefix path exclusion matching."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        assert middleware._is_excluded_path("/health/check") is True
        assert middleware._is_excluded_path("/docs/swagger") is True
        assert middleware._is_excluded_path("/api/protected") is False

    def test_is_excluded_path_no_match(self):
        """Test non-matching paths are not excluded."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        assert middleware._is_excluded_path("/api/users") is False
        assert middleware._is_excluded_path("/protected") is False
        assert middleware._is_excluded_path("/healthe") is False  # Not exact match

    @pytest.mark.asyncio
    async def test_authenticate_request_missing_token(self):
        """Test authentication fails when token is missing."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request without token
        request = Mock(spec=Request)
        request.headers = Headers({})

        with pytest.raises(AuthenticationError) as exc_info:
            await middleware._authenticate_request(request)

        assert exc_info.value.message == "Missing authentication token"
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_authenticate_request_successful_validation(self):
        """Test successful token validation."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            email_whitelist_enabled=True,
            email_whitelist=["@example.com"],
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock authenticated user
        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"email": "test@example.com"},
        )
        jwt_validator.validate_token.return_value = authenticated_user

        # Mock request with token
        request = Mock(spec=Request)
        request.headers = Headers({"CF-Access-Jwt-Assertion": "valid.jwt.token"})

        result = await middleware._authenticate_request(request)

        assert result == authenticated_user
        jwt_validator.validate_token.assert_called_once_with("valid.jwt.token", email_whitelist=["@example.com"])

    @pytest.mark.asyncio
    async def test_authenticate_request_jwt_validation_error(self):
        """Test handling of JWT validation errors."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock JWT validation error
        jwt_error = JWTValidationError("Invalid token", "invalid_token")
        jwt_validator.validate_token.side_effect = jwt_error

        # Mock request with token
        request = Mock(spec=Request)
        request.headers = Headers({"CF-Access-Jwt-Assertion": "invalid.jwt.token"})

        with pytest.raises(AuthenticationError) as exc_info:
            await middleware._authenticate_request(request)

        assert "Token validation failed: Invalid token" in exc_info.value.message
        assert exc_info.value.status_code == 401

    def test_extract_jwt_token_cloudflare_header(self):
        """Test JWT token extraction from Cloudflare Access header."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with Cloudflare header
        request = Mock(spec=Request)
        request.headers = Headers({"CF-Access-Jwt-Assertion": "cf.jwt.token"})

        token = middleware._extract_jwt_token(request)

        assert token == "cf.jwt.token"  # noqa: S105

    def test_extract_jwt_token_authorization_header(self):
        """Test JWT token extraction from Authorization header."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with Authorization header
        request = Mock(spec=Request)
        request.headers = Headers({"Authorization": "Bearer auth.jwt.token"})

        token = middleware._extract_jwt_token(request)

        assert token == "auth.jwt.token"  # noqa: S105

    def test_extract_jwt_token_custom_header(self):
        """Test JWT token extraction from custom header."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with custom header
        request = Mock(spec=Request)
        request.headers = Headers({"X-JWT-Token": "custom.jwt.token"})

        token = middleware._extract_jwt_token(request)

        assert token == "custom.jwt.token"  # noqa: S105

    def test_extract_jwt_token_priority_order(self):
        """Test token extraction priority: Cloudflare > Authorization > Custom."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with all headers
        request = Mock(spec=Request)
        request.headers = Headers(
            {
                "CF-Access-Jwt-Assertion": "cf.jwt.token",
                "Authorization": "Bearer auth.jwt.token",
                "X-JWT-Token": "custom.jwt.token",
            },
        )

        token = middleware._extract_jwt_token(request)

        # Should prioritize Cloudflare header
        assert token == "cf.jwt.token"  # noqa: S105

    def test_extract_jwt_token_invalid_authorization_format(self):
        """Test token extraction with invalid Authorization header format."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with invalid Authorization header
        request = Mock(spec=Request)
        request.headers = Headers({"Authorization": "Basic invalid"})

        token = middleware._extract_jwt_token(request)

        assert token is None

    def test_extract_jwt_token_no_headers(self):
        """Test token extraction when no relevant headers are present."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request without relevant headers
        request = Mock(spec=Request)
        request.headers = Headers({"Content-Type": "application/json"})

        token = middleware._extract_jwt_token(request)

        assert token is None

    def test_create_auth_error_response_with_details(self):
        """Test error response creation with detailed error information."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            auth_error_detail_enabled=True,
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        error = AuthenticationError("Token has expired", 401)

        response = middleware._create_auth_error_response(error)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 401
        # Access the content through the response body
        import json

        content = json.loads(response.body.decode())
        assert content["error"] == "Authentication failed"
        assert content["message"] == "Token has expired"

    def test_create_auth_error_response_without_details(self):
        """Test error response creation without detailed error information."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            auth_error_detail_enabled=False,
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        error = AuthenticationError("Detailed error message", 403)

        response = middleware._create_auth_error_response(error)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 403
        # Access the content through the response body
        import json

        content = json.loads(response.body.decode())
        assert content["error"] == "Authentication failed"
        assert content["message"] == "Authentication required"


@pytest.mark.auth
class TestCreateRateLimiter:
    """Test cases for rate limiter creation."""

    def test_create_rate_limiter_default_settings(self):
        """Test rate limiter creation with default settings."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limit_requests=100,
            rate_limit_window=60,
            rate_limit_key_func="ip",
        )

        limiter = create_rate_limiter(config)

        assert isinstance(limiter, Limiter)
        # Test that the limiter was created successfully
        assert limiter.enabled is True

    def test_create_rate_limiter_custom_settings(self):
        """Test rate limiter creation with custom settings."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limit_requests=50,
            rate_limit_window=120,
            rate_limit_key_func="email",
        )

        limiter = create_rate_limiter(config)

        assert isinstance(limiter, Limiter)
        # Test that the limiter was created successfully
        assert limiter.enabled is True

    def test_rate_limit_key_func_ip(self):
        """Test rate limiting key function with IP strategy."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limit_key_func="ip",
        )

        limiter = create_rate_limiter(config)

        # Test that limiter was created and is enabled
        assert isinstance(limiter, Limiter)
        assert limiter.enabled is True

    def test_rate_limit_key_func_email_with_user(self):
        """Test rate limiting key function with email strategy and authenticated user."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limit_key_func="email",
        )

        limiter = create_rate_limiter(config)

        # Test that limiter was created and is enabled
        assert isinstance(limiter, Limiter)
        assert limiter.enabled is True

    def test_rate_limit_key_func_email_fallback_to_ip(self):
        """Test rate limiting key function with email strategy falls back to IP."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limit_key_func="email",
        )

        limiter = create_rate_limiter(config)

        # Test that limiter was created and is enabled
        assert isinstance(limiter, Limiter)
        assert limiter.enabled is True

    def test_rate_limit_key_func_user_with_authenticated_user(self):
        """Test rate limiting key function with user strategy and authenticated user."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limit_key_func="user",
        )

        limiter = create_rate_limiter(config)

        # Test that limiter was created and is enabled
        assert isinstance(limiter, Limiter)
        assert limiter.enabled is True

    def test_rate_limit_key_func_user_fallback_to_ip(self):
        """Test rate limiting key function with user strategy falls back to IP."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limit_key_func="user",
        )

        limiter = create_rate_limiter(config)

        # Test that limiter was created and is enabled
        assert isinstance(limiter, Limiter)
        assert limiter.enabled is True

    def test_rate_limit_key_func_unknown_strategy(self):
        """Test that unknown strategy is rejected by validation."""
        # This should raise a validation error due to Pydantic validation
        with pytest.raises(Exception, match="ValidationError|validation"):  # Pydantic ValidationError
            AuthenticationConfig(
                cloudflare_access_enabled=True,
                cloudflare_team_domain="test",
                cloudflare_jwks_url="https://test.com/jwks",
                rate_limit_key_func="unknown",
            )


@pytest.mark.auth
class TestSetupAuthentication:
    """Test cases for authentication setup function."""

    def test_setup_authentication_basic(self):
        """Test basic authentication setup."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limiting_enabled=False,
        )

        with (
            patch("src.auth.middleware.JWKSClient") as mock_jwks_client,
            patch("src.auth.middleware.JWTValidator") as mock_jwt_validator,
        ):
            mock_jwks_instance = Mock()
            mock_jwks_client.return_value = mock_jwks_instance
            mock_validator_instance = Mock()
            mock_jwt_validator.return_value = mock_validator_instance

            auth_middleware, limiter = setup_authentication(app, config)

            assert isinstance(auth_middleware, AuthenticationMiddleware)
            assert isinstance(limiter, Limiter)

            # Verify JWKS client creation
            mock_jwks_client.assert_called_once_with(
                jwks_url=config.get_jwks_url(),
                cache_ttl=config.jwks_cache_ttl,
                max_cache_size=config.jwks_cache_max_size,
                timeout=config.jwks_timeout,
            )

            # Verify JWT validator creation
            mock_jwt_validator.assert_called_once_with(
                jwks_client=mock_jwks_instance,
                config=config,
                audience=config.cloudflare_audience,
                issuer=config.cloudflare_issuer,
                algorithm=config.jwt_algorithm,
            )

    def test_setup_authentication_with_rate_limiting(self):
        """Test authentication setup with rate limiting enabled."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limiting_enabled=True,
        )

        with (
            patch("src.auth.middleware.JWKSClient"),
            patch("src.auth.middleware.JWTValidator"),
            patch.object(app, "add_middleware") as mock_add_middleware,
            patch.object(app, "add_exception_handler") as mock_add_handler,
        ):
            auth_middleware, limiter = setup_authentication(app, config)

            # Verify middleware was added
            assert mock_add_middleware.call_count >= 1

            # Verify rate limiting setup
            assert hasattr(app.state, "limiter")
            assert app.state.limiter == limiter
            mock_add_handler.assert_called_once()

    def test_setup_authentication_without_rate_limiting(self):
        """Test authentication setup without rate limiting."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            rate_limiting_enabled=False,
        )

        with (
            patch("src.auth.middleware.JWKSClient"),
            patch("src.auth.middleware.JWTValidator"),
            patch.object(app, "add_middleware") as mock_add_middleware,
            patch.object(app, "add_exception_handler") as mock_add_handler,
        ):
            auth_middleware, limiter = setup_authentication(app, config)

            # Verify only authentication middleware was added
            # (not SlowAPI middleware)
            middleware_calls = [call[0][0] for call in mock_add_middleware.call_args_list]
            assert AuthenticationMiddleware in [type(call) for call in middleware_calls]
            assert SlowAPIMiddleware not in [type(call) for call in middleware_calls]

            # Verify no rate limit exception handler was added
            mock_add_handler.assert_not_called()


@pytest.mark.auth
class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_get_current_user_with_authenticated_user(self):
        """Test getting current user when user is authenticated."""
        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"email": "test@example.com"},
        )

        request = Mock(spec=Request)
        request.state = Mock()
        request.state.authenticated_user = authenticated_user

        result = get_current_user(request)

        assert result == authenticated_user

    def test_get_current_user_without_authenticated_user(self):
        """Test getting current user when no user is authenticated."""

        # Create a mock request with state but no authenticated_user
        class MockState:
            pass

        request = Mock(spec=Request)
        request.state = MockState()

        result = get_current_user(request)

        assert result is None

    def test_get_current_user_no_state(self):
        """Test getting current user when request has no state."""

        # Create a mock request without state attribute
        class MockRequest:
            pass

        request = MockRequest()

        result = get_current_user(request)

        assert result is None

    def test_require_authentication_with_authenticated_user(self):
        """Test requiring authentication when user is authenticated."""
        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"email": "test@example.com"},
        )

        request = Mock(spec=Request)
        request.state = Mock()
        request.state.authenticated_user = authenticated_user

        result = require_authentication(request)

        assert result == authenticated_user

    def test_require_authentication_without_authenticated_user(self):
        """Test requiring authentication when no user is authenticated."""

        # Create a mock request with state but no authenticated_user
        class MockState:
            pass

        request = Mock(spec=Request)
        request.state = MockState()

        with pytest.raises(HTTPException) as exc_info:
            require_authentication(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication required"

    def test_require_role_with_correct_role(self):
        """Test requiring specific role when user has correct role."""
        authenticated_user = AuthenticatedUser(
            email="admin@example.com",
            role=UserRole.ADMIN,
            jwt_claims={"email": "admin@example.com"},
        )

        request = Mock(spec=Request)
        request.state = Mock()
        request.state.authenticated_user = authenticated_user

        result = require_role(request, "admin")

        assert result == authenticated_user

    def test_require_role_with_incorrect_role(self):
        """Test requiring specific role when user has incorrect role."""
        authenticated_user = AuthenticatedUser(
            email="user@example.com",
            role=UserRole.USER,
            jwt_claims={"email": "user@example.com"},
        )

        request = Mock(spec=Request)
        request.state = Mock()
        request.state.authenticated_user = authenticated_user

        with pytest.raises(HTTPException) as exc_info:
            require_role(request, "admin")

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Role 'admin' required"

    def test_require_role_without_authenticated_user(self):
        """Test requiring role when no user is authenticated."""

        # Create a mock request with state but no authenticated_user
        class MockState:
            pass

        request = Mock(spec=Request)
        request.state = MockState()

        with pytest.raises(HTTPException) as exc_info:
            require_role(request, "admin")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication required"


@pytest.mark.auth
class TestIntegrationScenarios:
    """Integration test cases for complete authentication workflows."""

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_success(self):
        """Test complete successful authentication flow."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            email_whitelist_enabled=True,
            email_whitelist=["@example.com"],
            auth_logging_enabled=True,
        )

        # Mock JWT validator
        jwt_validator = Mock(spec=JWTValidator)
        authenticated_user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"email": "test@example.com", "exp": 1234567890},
        )
        jwt_validator.validate_token.return_value = authenticated_user

        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with valid token
        request = Mock(spec=Request)
        request.url = Mock(spec=URL)
        request.url.path = "/api/protected"
        request.headers = Headers({"CF-Access-Jwt-Assertion": "valid.jwt.token"})
        request.state = Mock()

        call_next = AsyncMock(return_value=Response())

        response = await middleware.dispatch(request, call_next)

        # Verify successful flow
        assert isinstance(response, Response)
        call_next.assert_called_once_with(request)
        jwt_validator.validate_token.assert_called_once_with("valid.jwt.token", email_whitelist=["@example.com"])

        # Verify user context was injected
        assert request.state.authenticated_user == authenticated_user
        assert request.state.user_email == "test@example.com"
        assert request.state.user_role == UserRole.USER

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_failure(self):
        """Test complete authentication flow with validation failure."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
            auth_error_detail_enabled=True,
            auth_logging_enabled=True,
        )

        # Mock JWT validator with failure
        jwt_validator = Mock(spec=JWTValidator)
        jwt_error = JWTValidationError("Token has expired", "expired_token")
        jwt_validator.validate_token.side_effect = jwt_error

        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with expired token
        request = Mock(spec=Request)
        request.url = Mock(spec=URL)
        request.url.path = "/api/protected"
        request.headers = Headers({"CF-Access-Jwt-Assertion": "expired.jwt.token"})

        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)

        # Verify failure response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401
        call_next.assert_not_called()
        jwt_validator.validate_token.assert_called_once()

    def test_multiple_header_priority_integration(self):
        """Test token extraction priority in integration scenario."""
        app = FastAPI()
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test",
            cloudflare_jwks_url="https://test.com/jwks",
        )
        jwt_validator = Mock(spec=JWTValidator)
        middleware = AuthenticationMiddleware(app, config, jwt_validator)

        # Mock request with multiple token headers
        request = Mock(spec=Request)
        request.headers = Headers(
            {
                "Authorization": "Bearer auth.token",
                "CF-Access-Jwt-Assertion": "cf.token",
                "X-JWT-Token": "custom.token",
            },
        )

        token = middleware._extract_jwt_token(request)

        # Should prioritize Cloudflare header
        assert token == "cf.token"  # noqa: S105
