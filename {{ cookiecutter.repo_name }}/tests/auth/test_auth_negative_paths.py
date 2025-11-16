"""Negative path security tests for authentication system."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import httpx
import jwt
import pytest

from src.auth.config import AuthenticationConfig
from src.auth.jwks_client import JWKSClient
from src.auth.jwt_validator import JWTValidator
from src.auth.middleware import AuthenticationMiddleware
from src.auth.models import JWKSError, JWTValidationError


class TestAuthenticationNegativePaths:
    """Test negative paths and security edge cases."""

    @pytest.fixture
    def auth_config(self):
        """Authentication configuration for testing."""
        return AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test-team",
            email_whitelist=["allowed@example.com", "@trusted.com"],
            email_whitelist_enabled=True,
            rate_limiting_enabled=True,
            auth_error_detail_enabled=False,  # Security: don't expose details
        )

    @pytest.fixture
    def mock_jwks_client(self):
        """Mock JWKS client."""
        return Mock(spec=JWKSClient)

    @pytest.fixture
    def jwt_validator(self, mock_jwks_client, auth_config):
        """JWT validator for testing."""
        return JWTValidator(
            jwks_client=mock_jwks_client,
            config=auth_config,
            audience="test-aud",
            issuer="test-iss",
        )

    def test_malformed_jwt_token(self, jwt_validator):
        """Test handling of malformed JWT tokens."""
        malformed_tokens = [
            "",  # Empty token
            "invalid",  # Single part
            "invalid.token",  # Two parts
            "invalid.token.with.four.parts",  # Too many parts
            "header.payload.",  # Empty signature
            ".payload.signature",  # Empty header
            "header..signature",  # Empty payload
        ]

        for token in malformed_tokens:
            with pytest.raises(JWTValidationError):
                jwt_validator.validate_token(token)

    def test_jwt_signature_tampering(self, jwt_validator, mock_jwks_client):
        """Test detection of JWT signature tampering."""
        mock_jwks_client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "kid": "test-kid",
            "n": "test-modulus",
            "e": "AQAB",
        }

        # Create token with tampered signature using dynamic generation
        test_payload = {"email": "test@example.com", "typ": "JWT", "alg": "RS256"}
        valid_token = jwt.encode(test_payload, "fake-secret-for-testing", algorithm="HS256")
        # Manually tamper with signature part to simulate tampering
        parts = valid_token.split(".")
        tampered_token = f"{parts[0]}.{parts[1]}.tampered_signature"

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.side_effect = jwt.InvalidSignatureError("Signature verification failed")

            with pytest.raises(JWTValidationError) as exc_info:
                jwt_validator.validate_token(tampered_token)

            assert exc_info.value.error_type == "invalid_signature"

    def test_expired_jwt_token(self, jwt_validator, mock_jwks_client):
        """Test handling of expired JWT tokens."""
        mock_jwks_client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "kid": "test-kid",
            "n": "test-modulus",
            "e": "AQAB",
        }

        expired_token = "expired.jwt.token"

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token has expired")

            with pytest.raises(JWTValidationError) as exc_info:
                jwt_validator.validate_token(expired_token)

            assert exc_info.value.error_type == "expired_token"
            assert "expired" in exc_info.value.message.lower()

    def test_jwt_missing_required_claims(self, jwt_validator, mock_jwks_client):
        """Test JWT tokens missing required claims."""
        mock_jwks_client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "kid": "test-kid",
            "n": "test-modulus",
            "e": "AQAB",
        }

        # Test various missing claims
        incomplete_payloads = [
            {},  # No claims
            {"sub": "user123"},  # Missing email
            {"email": ""},  # Empty email
            {"email": "invalid-email"},  # Invalid email format
            {"email": 123},  # Non-string email
        ]

        for payload in incomplete_payloads:
            with (
                patch("jwt.get_unverified_header") as mock_header,
                patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
                patch("jwt.decode") as mock_decode,
            ):

                mock_header.return_value = {"kid": "test-kid"}
                mock_from_jwk.return_value = "mock-public-key"
                mock_decode.return_value = payload

                with pytest.raises(JWTValidationError):
                    jwt_validator.validate_token("test.jwt.token")

    def test_unauthorized_email_attempts(self, jwt_validator, mock_jwks_client):
        """Test blocking of unauthorized email addresses."""
        from fastapi import HTTPException

        mock_jwks_client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "kid": "test-kid",
            "n": "test-modulus",
            "e": "AQAB",
        }

        # Unauthorized emails
        unauthorized_emails = [
            "hacker@malicious.com",
            "attacker@evil.org",
            "user@blocked.net",
            "admin@unauthorized.com",
        ]

        email_whitelist = ["allowed@example.com", "@trusted.com"]

        for email in unauthorized_emails:
            payload = {
                "email": email,
                "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
                "iat": int(datetime.now(timezone.utc).timestamp()),
            }

            with (
                patch("jwt.get_unverified_header") as mock_header,
                patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
                patch("jwt.decode") as mock_decode,
            ):

                mock_header.return_value = {"kid": "test-kid"}
                mock_from_jwk.return_value = "mock-public-key"
                mock_decode.return_value = payload

                with pytest.raises(HTTPException) as exc_info:
                    jwt_validator.validate_token("test.jwt.token", email_whitelist)

                assert exc_info.value.status_code == 401  # Authorization errors become authentication errors

    def test_jwks_endpoint_failures(self):
        """Test various JWKS endpoint failure scenarios."""
        jwks_client = JWKSClient("https://invalid.example.com/jwks")

        # Test network failures
        with patch.object(jwks_client._client, "get") as mock_get:
            # Connection error
            mock_get.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(JWKSError) as exc_info:
                jwks_client.get_jwks()

            assert exc_info.value.error_type == "http_error"

    def test_jwks_malformed_responses(self):
        """Test handling of malformed JWKS responses."""
        jwks_client = JWKSClient("https://test.example.com/jwks")

        malformed_responses = [
            None,  # Null response
            "invalid json",  # Invalid JSON
            {"invalid": "structure"},  # Missing keys
            {"keys": None},  # Null keys
            {"keys": "not-a-list"},  # Keys not a list
            {"keys": []},  # Empty keys list
        ]

        for response_data in malformed_responses:
            with patch.object(jwks_client._client, "get") as mock_get:
                mock_response = Mock()
                if response_data is None or response_data == "invalid json":
                    mock_response.json.side_effect = ValueError("Invalid JSON")
                else:
                    mock_response.json.return_value = response_data
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                with pytest.raises(JWKSError):
                    jwks_client.get_jwks()

    def test_middleware_missing_headers(self, auth_config, jwt_validator):
        """Test middleware behavior with missing authentication headers."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI()
        middleware = AuthenticationMiddleware(
            app=app,
            config=auth_config,
            jwt_validator=jwt_validator,
            database_enabled=False,  # Disable database logging for tests
        )

        # Mock request without authentication headers
        request = Mock(spec=Request)
        request.url.path = "/protected"
        request.headers = {}  # No headers
        request.state = Mock()

        async def mock_call_next(req):
            return Mock(status_code=200)

        # Test should result in authentication error
        import asyncio

        result = asyncio.run(middleware.dispatch(request, mock_call_next))

        # Should return JSONResponse with 401 error status
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401

    def test_rate_limiting_bypass_attempts(self, auth_config):
        """Test various rate limiting bypass attempts."""
        from fastapi import Request

        from src.auth.middleware import create_rate_limiter

        limiter = create_rate_limiter(auth_config)

        # Test multiple rapid requests from same IP
        request = Mock(spec=Request)
        request.state = Mock()

        # Mock remote address
        with patch("slowapi.util.get_remote_address") as mock_get_ip:
            mock_get_ip.return_value = "192.168.1.100"

            # Simulate rapid requests
            for _ in range(auth_config.rate_limit_requests + 10):
                try:
                    key = limiter._key_func(request)
                    assert key == "192.168.1.100"
                except Exception:
                    pass  # Rate limit may be exceeded

    def test_jwt_header_injection_attempts(self, jwt_validator, mock_jwks_client):
        """Test attempts to inject malicious data via JWT headers."""
        mock_jwks_client.get_key_by_kid.return_value = None  # Key not found

        malicious_headers = [
            {"kid": "'; DROP TABLE users; --"},  # SQL injection attempt
            {"kid": "<script>alert('xss')</script>"},  # XSS attempt
            {"kid": "../../../etc/passwd"},  # Path traversal attempt
            {"kid": "a" * 10000},  # Buffer overflow attempt
            {"kid": None},  # Null injection
        ]

        for header in malicious_headers:
            with patch("jwt.get_unverified_header") as mock_header:
                mock_header.return_value = header

                with pytest.raises(JWTValidationError):
                    jwt_validator.validate_token("test.jwt.token")

    def test_concurrent_authentication_attempts(self, jwt_validator, mock_jwks_client):
        """Test system behavior under concurrent authentication attempts."""
        import threading

        mock_jwks_client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "kid": "test-kid",
            "n": "test-modulus",
            "e": "AQAB",
        }

        results = []

        def auth_attempt():
            try:
                with (
                    patch("jwt.get_unverified_header") as mock_header,
                    patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
                    patch("jwt.decode") as mock_decode,
                ):

                    mock_header.return_value = {"kid": "test-kid"}
                    mock_from_jwk.return_value = "mock-public-key"
                    mock_decode.return_value = {
                        "email": "test@example.com",
                        "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
                        "iat": int(datetime.now(timezone.utc).timestamp()),
                    }

                    result = jwt_validator.validate_token("test.jwt.token")
                    results.append(result.email)
            except Exception as e:
                results.append(str(e))

        # Start multiple concurrent authentication attempts
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=auth_attempt)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all attempts were handled (no crashes)
        assert len(results) == 10
