"""Tests for JWT validation functionality."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi import HTTPException

from src.auth.jwks_client import JWKSClient
from src.auth.jwt_validator import JWTValidator
from src.auth.models import JWTValidationError, UserRole


class TestJWTValidator:
    """Test cases for JWT validation."""

    @pytest.fixture
    def mock_jwks_client(self):
        """Mock JWKS client."""
        client = Mock(spec=JWKSClient)
        return client

    @pytest.fixture
    def auth_config(self):
        """Authentication configuration for testing."""
        from src.auth.config import AuthenticationConfig
        return AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="test-team",
            email_whitelist=["test@example.com", "@trusted.com"],
            email_whitelist_enabled=True,
        )

    @pytest.fixture
    def jwt_validator(self, mock_jwks_client, auth_config):
        """JWT validator instance."""
        return JWTValidator(
            jwks_client=mock_jwks_client,
            config=auth_config,
            audience="test-audience",
            issuer="test-issuer",
        )

    @pytest.fixture
    def sample_jwt_payload(self):
        """Sample JWT payload."""
        return {
            "email": "test@example.com",
            "aud": "test-audience",
            "iss": "test-issuer",
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "sub": "user123",
        }

    @pytest.fixture
    def sample_jwk(self):
        """Sample JWK for testing."""
        return {
            "kty": "RSA",
            "kid": "test-kid",
            "use": "sig",
            "n": "test-modulus",
            "e": "AQAB",
        }

    def test_validate_token_success(self, jwt_validator, mock_jwks_client, sample_jwt_payload, sample_jwk):
        """Test successful token validation."""
        # Mock JWKS response
        mock_jwks_client.get_key_by_kid.return_value = sample_jwk

        # Create a test token
        token = "test.jwt.token"

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.return_value = sample_jwt_payload

            # Validate token
            result = jwt_validator.validate_token(token)

            # Assertions
            assert result.email == "test@example.com"
            assert result.role == UserRole.USER
            assert result.jwt_claims == sample_jwt_payload

            # Verify JWKS client was called
            mock_jwks_client.get_key_by_kid.assert_called_once_with("test-kid")

    def test_validate_token_missing_kid(self, jwt_validator):
        """Test token validation with missing kid in header."""
        token = "test.jwt.token"

        with patch("jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {}  # No kid

            with pytest.raises(JWTValidationError) as exc_info:
                jwt_validator.validate_token(token)

            assert exc_info.value.error_type == "missing_kid"

    def test_validate_token_key_not_found(self, jwt_validator, mock_jwks_client):
        """Test token validation when key is not found in JWKS."""
        token = "test.jwt.token"
        mock_jwks_client.get_key_by_kid.return_value = None

        with patch("jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "unknown-kid"}

            with pytest.raises(JWTValidationError) as exc_info:
                jwt_validator.validate_token(token)

            assert exc_info.value.error_type == "key_not_found"
            assert "unknown-kid" in exc_info.value.message

    def test_validate_token_expired(self, jwt_validator, mock_jwks_client, sample_jwk):
        """Test validation of expired token."""
        token = "test.jwt.token"
        mock_jwks_client.get_key_by_kid.return_value = sample_jwk

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.side_effect = jwt.ExpiredSignatureError()

            with pytest.raises(JWTValidationError) as exc_info:
                jwt_validator.validate_token(token)

            assert exc_info.value.error_type == "expired_token"

    def test_validate_token_missing_email(self, jwt_validator, mock_jwks_client, sample_jwt_payload, sample_jwk):
        """Test token validation with missing email claim."""
        token = "test.jwt.token"
        mock_jwks_client.get_key_by_kid.return_value = sample_jwk

        # Remove email from payload
        payload_without_email = sample_jwt_payload.copy()
        del payload_without_email["email"]

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.return_value = payload_without_email

            with pytest.raises(JWTValidationError) as exc_info:
                jwt_validator.validate_token(token)

            assert exc_info.value.error_type == "missing_email"

    def test_validate_token_email_whitelist_allowed(
        self, jwt_validator, mock_jwks_client, sample_jwt_payload, sample_jwk
    ):
        """Test token validation with email whitelist - allowed email."""
        token = "test.jwt.token"
        mock_jwks_client.get_key_by_kid.return_value = sample_jwk
        email_whitelist = ["test@example.com", "@company.com"]

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.return_value = sample_jwt_payload

            result = jwt_validator.validate_token(token, email_whitelist)
            assert result.email == "test@example.com"

    def test_validate_token_email_whitelist_denied(
        self, jwt_validator, mock_jwks_client, sample_jwt_payload, sample_jwk
    ):
        """Test token validation with email whitelist - denied email."""
        token = "test.jwt.token"
        mock_jwks_client.get_key_by_kid.return_value = sample_jwk
        email_whitelist = ["allowed@example.com", "@company.com"]

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.return_value = sample_jwt_payload

            with pytest.raises(HTTPException) as exc_info:
                jwt_validator.validate_token(token, email_whitelist)

            assert exc_info.value.status_code == 401  # Authorization errors become authentication errors

    def test_determine_admin_role_by_email(self, jwt_validator, mock_jwks_client, sample_jwt_payload, sample_jwk):
        """Test admin role determination by email."""
        token = "test.jwt.token"
        mock_jwks_client.get_key_by_kid.return_value = sample_jwk

        # Use admin email
        admin_payload = sample_jwt_payload.copy()
        admin_payload["email"] = "admin@example.com"

        with (
            patch("jwt.get_unverified_header") as mock_header,
            patch("jwt.algorithms.RSAAlgorithm.from_jwk") as mock_from_jwk,
            patch("jwt.decode") as mock_decode,
        ):

            mock_header.return_value = {"kid": "test-kid"}
            mock_from_jwk.return_value = "mock-public-key"
            mock_decode.return_value = admin_payload

            result = jwt_validator.validate_token(token)
            assert result.role == UserRole.ADMIN

    def test_is_email_allowed_exact_match(self, jwt_validator):
        """Test email whitelist exact match."""
        email_whitelist = ["test@example.com", "admin@company.com"]

        assert jwt_validator.is_email_allowed("test@example.com", email_whitelist)
        assert jwt_validator.is_email_allowed("admin@company.com", email_whitelist)
        assert not jwt_validator.is_email_allowed("other@example.com", email_whitelist)

    def test_is_email_allowed_domain_match(self, jwt_validator):
        """Test email whitelist domain match."""
        email_whitelist = ["@company.com", "@trusted.org"]

        assert jwt_validator.is_email_allowed("user@company.com", email_whitelist)
        assert jwt_validator.is_email_allowed("admin@trusted.org", email_whitelist)
        assert not jwt_validator.is_email_allowed("user@external.com", email_whitelist)

    def test_validate_token_format_valid(self, jwt_validator):
        """Test token format validation for valid token."""
        # Create a properly formatted JWT (3 parts)
        token = "header.payload.signature"

        with patch("jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "test-kid"}

            assert jwt_validator.validate_token_format(token)

    def test_validate_token_format_invalid(self, jwt_validator):
        """Test token format validation for invalid token."""
        # Invalid format (not 3 parts)
        assert not jwt_validator.validate_token_format("invalid.token")
        assert not jwt_validator.validate_token_format("invalid")
        assert not jwt_validator.validate_token_format("")

        # Valid format but invalid header
        with patch("jwt.get_unverified_header") as mock_header:
            mock_header.side_effect = Exception("Invalid header")
            assert not jwt_validator.validate_token_format("header.payload.signature")
