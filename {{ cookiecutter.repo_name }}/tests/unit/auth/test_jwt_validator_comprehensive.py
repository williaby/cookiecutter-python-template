"""
Comprehensive unit tests for JWT validator covering critical security paths.

This module provides extensive test coverage for JWTValidator class focusing on:
- Token signature verification edge cases
- Payload validation and claim checking
- Email whitelist validation with domain patterns
- Role determination and permission mapping
- Error handling for malformed tokens
- Security boundary conditions

Aims to achieve 80%+ coverage for auth/jwt_validator.py
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi import HTTPException
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
)

from src.auth.jwks_client import JWKSClient
from src.auth.jwt_validator import JWTValidator
from src.auth.models import AuthenticatedUser, JWTValidationError, UserRole


@pytest.mark.unit
@pytest.mark.auth
class TestJWTValidatorTokenDecoding:
    """Test JWT token decoding and validation logic."""

    @pytest.fixture
    def mock_jwks_client(self):
        """Create mock JWKS client."""
        client = Mock(spec=JWKSClient)
        client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "use": "sig",
            "kid": "test-key-id",
            "n": "test-modulus",
            "e": "AQAB",
        }
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
    def validator(self, mock_jwks_client, auth_config):
        """Create JWT validator instance."""
        return JWTValidator(
            jwks_client=mock_jwks_client,
            config=auth_config,
            audience="https://test-app.com",
            issuer="https://test.cloudflareaccess.com",
        )

    @pytest.fixture
    def valid_token_payload(self):
        """Valid JWT payload for testing."""
        return {
            "iss": "https://test.cloudflareaccess.com",
            "aud": "https://test-app.com",
            "sub": "user123",
            "email": "test@example.com",
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "nbf": int(datetime.now(timezone.utc).timestamp()),
        }

    @pytest.fixture
    def valid_jwt_token(self):
        """Valid JWT token string for testing."""
        # Create a properly formatted JWT token string with valid header
        header = jwt.utils.base64url_encode(b'{"alg":"RS256","typ":"JWT","kid":"test-key-id"}')
        payload = jwt.utils.base64url_encode(b'{"email":"test@example.com","sub":"user123"}')
        signature = jwt.utils.base64url_encode(b"fake-signature")
        return f"{header.decode()}.{payload.decode()}.{signature.decode()}"

    def test_validate_token_format_invalid_structure(self, validator):
        """Test validation with malformed token structure."""
        invalid_tokens = [
            "",  # Empty string
            "invalid.token",  # Missing parts
            "header.payload",  # Missing signature
            "too.many.parts.here.invalid",  # Too many parts
            "header.payload.signature.extra",  # Extra parts
            "not-base64.payload.signature",  # Invalid base64
        ]

        for token in invalid_tokens:
            with pytest.raises(JWTValidationError) as exc_info:
                validator.validate_token(token)
            assert exc_info.value.error_type == "invalid_format"

    def test_validate_token_format_malformed_json(self, validator):
        """Test validation with malformed JSON in token parts."""
        # Create token with invalid JSON header
        header = jwt.utils.base64url_encode(b'{"alg":"RS256","typ":"JWT",invalid-json}')  # Malformed JSON
        payload = jwt.utils.base64url_encode(b'{"email":"test@example.com"}')
        signature = jwt.utils.base64url_encode(b"fake-signature")
        malformed_token = f"{header.decode()}.{payload.decode()}.{signature.decode()}"

        with pytest.raises(JWTValidationError) as exc_info:
            validator.validate_token(malformed_token)
        assert exc_info.value.error_type == "invalid_format"

    def test_validate_token_missing_kid_header(self, validator):
        """Test validation when token header is missing 'kid' claim."""
        # Create token with header missing 'kid'
        header = jwt.utils.base64url_encode(b'{"alg":"RS256","typ":"JWT"}')  # No 'kid'
        payload = jwt.utils.base64url_encode(b'{"email":"test@example.com"}')
        signature = jwt.utils.base64url_encode(b"fake-signature")
        token_without_kid = f"{header.decode()}.{payload.decode()}.{signature.decode()}"

        with pytest.raises(JWTValidationError) as exc_info:
            validator.validate_token(token_without_kid)
        assert exc_info.value.error_type == "missing_kid"

    def test_validate_token_jwks_key_not_found(self, validator):
        """Test validation when JWKS key is not found."""
        # Create token with valid header but key not found in JWKS
        header = jwt.utils.base64url_encode(b'{"alg":"RS256","typ":"JWT","kid":"unknown-key-id"}')
        payload = jwt.utils.base64url_encode(b'{"email":"test@example.com"}')
        signature = jwt.utils.base64url_encode(b"fake-signature")
        token_with_unknown_kid = f"{header.decode()}.{payload.decode()}.{signature.decode()}"

        validator.jwks_client.get_key_by_kid.return_value = None

        with pytest.raises(JWTValidationError, match="Key with kid .* not found in JWKS"):
            validator.validate_token(token_with_unknown_kid)

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_expired_signature(self, mock_decode, validator, valid_jwt_token):
        """Test validation with expired token."""
        mock_decode.side_effect = ExpiredSignatureError("Token has expired")

        with pytest.raises(JWTValidationError, match="Token has expired"):
            validator.validate_token(valid_jwt_token)

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_invalid_signature(self, mock_decode, validator, valid_jwt_token):
        """Test validation with invalid signature."""
        mock_decode.side_effect = InvalidSignatureError("Invalid signature")

        with pytest.raises(JWTValidationError, match="Token signature verification failed"):
            validator.validate_token(valid_jwt_token)

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_invalid_audience(self, mock_decode, validator, valid_jwt_token):
        """Test validation with mismatched audience."""
        mock_decode.side_effect = InvalidAudienceError("Invalid audience")

        with pytest.raises(JWTValidationError, match="Invalid token audience"):
            validator.validate_token(valid_jwt_token)

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_invalid_issuer(self, mock_decode, validator, valid_jwt_token):
        """Test validation with mismatched issuer."""
        mock_decode.side_effect = InvalidIssuerError("Invalid issuer")

        with pytest.raises(JWTValidationError, match="Invalid token issuer"):
            validator.validate_token(valid_jwt_token)

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_missing_email_claim(self, mock_decode, validator, valid_token_payload, valid_jwt_token):
        """Test validation when token is missing email claim."""
        payload_without_email = valid_token_payload.copy()
        del payload_without_email["email"]
        mock_decode.return_value = payload_without_email

        with pytest.raises(JWTValidationError, match="JWT payload missing required 'email' claim"):
            validator.validate_token(valid_jwt_token)

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_empty_email_claim(self, mock_decode, validator, valid_token_payload, valid_jwt_token):
        """Test validation when token has empty email claim."""
        payload_empty_email = valid_token_payload.copy()
        payload_empty_email["email"] = ""
        mock_decode.return_value = payload_empty_email

        with pytest.raises(JWTValidationError, match="JWT payload missing required 'email' claim"):
            validator.validate_token(valid_jwt_token)

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_none_email_claim(self, mock_decode, validator, valid_token_payload, valid_jwt_token):
        """Test validation when token has None email claim."""
        payload_none_email = valid_token_payload.copy()
        payload_none_email["email"] = None
        mock_decode.return_value = payload_none_email

        with pytest.raises(JWTValidationError, match="JWT payload missing required 'email' claim"):
            validator.validate_token(valid_jwt_token)


@pytest.mark.unit
@pytest.mark.auth
class TestJWTValidatorEmailWhitelist:
    """Test email whitelist validation functionality."""

    @pytest.fixture
    def mock_jwks_client(self):
        """Create mock JWKS client."""
        client = Mock(spec=JWKSClient)
        client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "use": "sig",
            "kid": "test-key-id",
            "n": "test-modulus",
            "e": "AQAB",
        }
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
    def validator(self, mock_jwks_client, auth_config):
        """Create JWT validator instance."""
        return JWTValidator(jwks_client=mock_jwks_client, config=auth_config)

    def test_is_email_allowed_exact_match(self, validator):
        """Test exact email address matching."""
        whitelist = ["user@example.com", "admin@company.org"]

        assert validator.is_email_allowed("user@example.com", whitelist) is True
        assert validator.is_email_allowed("admin@company.org", whitelist) is True
        assert validator.is_email_allowed("other@example.com", whitelist) is False
        assert validator.is_email_allowed("user@other.com", whitelist) is False

    def test_is_email_allowed_domain_wildcard(self, validator):
        """Test domain wildcard matching."""
        whitelist = ["@example.com", "@company.org"]

        assert validator.is_email_allowed("user@example.com", whitelist) is True
        assert validator.is_email_allowed("admin@example.com", whitelist) is True
        assert validator.is_email_allowed("test@company.org", whitelist) is True
        assert validator.is_email_allowed("user@other.com", whitelist) is False

    def test_is_email_allowed_mixed_whitelist(self, validator):
        """Test mixed exact and domain matching."""
        whitelist = ["specific@example.com", "@company.org", "admin@test.net"]

        # Exact matches
        assert validator.is_email_allowed("specific@example.com", whitelist) is True
        assert validator.is_email_allowed("admin@test.net", whitelist) is True

        # Domain matches
        assert validator.is_email_allowed("anyone@company.org", whitelist) is True
        assert validator.is_email_allowed("user@company.org", whitelist) is True

        # No matches
        assert validator.is_email_allowed("other@example.com", whitelist) is False
        assert validator.is_email_allowed("specific@other.com", whitelist) is False

    def test_is_email_allowed_case_sensitivity(self, validator):
        """Test case sensitivity in email matching."""
        whitelist = ["User@Example.Com", "@Company.ORG"]

        # Should be case insensitive
        assert validator.is_email_allowed("user@example.com", whitelist) is True
        assert validator.is_email_allowed("USER@EXAMPLE.COM", whitelist) is True
        assert validator.is_email_allowed("test@company.org", whitelist) is True
        assert validator.is_email_allowed("TEST@COMPANY.ORG", whitelist) is True

    def test_is_email_allowed_empty_whitelist(self, validator):
        """Test behavior with empty whitelist."""
        assert validator.is_email_allowed("user@example.com", []) is False
        assert validator.is_email_allowed("admin@company.org", []) is False

    def test_is_email_allowed_none_whitelist(self, validator):
        """Test behavior with None whitelist (should allow all)."""
        assert validator.is_email_allowed("user@example.com", None) is True
        assert validator.is_email_allowed("admin@company.org", None) is True

    def test_is_email_allowed_malformed_email(self, validator):
        """Test behavior with malformed email addresses."""
        whitelist = ["user@example.com", "@company.org"]

        # Malformed emails should not match
        assert validator.is_email_allowed("invalid-email", whitelist) is False
        assert validator.is_email_allowed("@domain.com", whitelist) is False
        assert validator.is_email_allowed("user@", whitelist) is False
        assert validator.is_email_allowed("", whitelist) is False

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_email_whitelist_allowed(self, mock_from_jwk, mock_decode, validator, valid_jwt_token):
        """Test successful validation with whitelisted email."""
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": "user@example.com", "sub": "user123"}

        whitelist = ["user@example.com", "@company.org"]
        result = validator.validate_token(valid_jwt_token, email_whitelist=whitelist)

        assert isinstance(result, AuthenticatedUser)
        assert result.email == "user@example.com"
        assert result.user_id == "user123"

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_email_whitelist_domain_allowed(
        self,
        mock_from_jwk,
        mock_decode,
        validator,
        valid_jwt_token,
    ):
        """Test successful validation with domain whitelisted email."""
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": "anyone@company.org", "sub": "user456"}

        whitelist = ["@company.org"]
        result = validator.validate_token(valid_jwt_token, email_whitelist=whitelist)

        assert isinstance(result, AuthenticatedUser)
        assert result.email == "anyone@company.org"
        assert result.user_id == "user456"

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_email_whitelist_denied(self, mock_from_jwk, mock_decode, validator, valid_jwt_token):
        """Test validation failure with non-whitelisted email."""
        from fastapi import HTTPException

        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": "denied@blocked.com", "sub": "user789"}

        whitelist = ["user@example.com", "@company.org"]

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_token(valid_jwt_token, email_whitelist=whitelist)

        assert exc_info.value.status_code == 401  # Authorization errors become authentication errors


@pytest.mark.unit
@pytest.mark.auth
class TestJWTValidatorRoleDetermination:
    """Test role determination logic."""

    @pytest.fixture
    def mock_jwks_client(self):
        """Create mock JWKS client."""
        return Mock(spec=JWKSClient)

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
    def validator(self, mock_jwks_client, auth_config):
        """Create JWT validator instance."""
        return JWTValidator(jwks_client=mock_jwks_client, config=auth_config)

    def test_determine_admin_role_admin_email(self, validator):
        """Test admin role determination for admin email."""
        result = validator.determine_admin_role("admin@company.com")
        assert result == UserRole.ADMIN

    def test_determine_admin_role_administrator_email(self, validator):
        """Test admin role determination for administrator email."""
        result = validator.determine_admin_role("administrator@company.com")
        assert result == UserRole.ADMIN

    def test_determine_admin_role_root_email(self, validator):
        """Test admin role determination for root email."""
        result = validator.determine_admin_role("root@company.com")
        assert result == UserRole.ADMIN

    def test_determine_admin_role_superuser_email(self, validator):
        """Test admin role determination for superuser email."""
        result = validator.determine_admin_role("superuser@company.com")
        assert result == UserRole.ADMIN

    def test_determine_admin_role_case_insensitive(self, validator):
        """Test admin role determination is case insensitive."""
        test_emails = [
            "ADMIN@company.com",
            "Admin@Company.COM",
            "ADMINISTRATOR@test.org",
            "Root@Example.com",
            "SUPERUSER@domain.net",
        ]

        for email in test_emails:
            result = validator.determine_admin_role(email)
            assert result == UserRole.ADMIN

    def test_determine_admin_role_partial_match_not_admin(self, validator):
        """Test that partial matches don't grant admin role."""
        non_admin_emails = [
            "adminuser@company.com",  # Contains 'admin' but not exact
            "user-admin@company.com",  # Contains 'admin' but not at start
            "administrat@company.com",  # Partial match
            "roots@company.com",  # Plural
            "super@company.com",  # Partial match
        ]

        for email in non_admin_emails:
            result = validator.determine_admin_role(email)
            assert result == UserRole.USER

    def test_determine_admin_role_regular_user(self, validator):
        """Test regular user role determination."""
        regular_emails = ["user@company.com", "test@example.org", "developer@startup.io", "manager@business.net"]

        for email in regular_emails:
            result = validator.determine_admin_role(email)
            assert result == UserRole.USER

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_admin_role_assignment(self, mock_from_jwk, mock_decode, validator, valid_jwt_token):
        """Test that admin roles are correctly assigned in validated user."""
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": "admin@company.com", "sub": "admin123"}

        result = validator.validate_token(valid_jwt_token)

        assert isinstance(result, AuthenticatedUser)
        assert result.role == UserRole.ADMIN
        assert result.email == "admin@company.com"

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_user_role_assignment(self, mock_from_jwk, mock_decode, validator, valid_jwt_token):
        """Test that user roles are correctly assigned in validated user."""
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": "user@company.com", "sub": "user123"}

        result = validator.validate_token(valid_jwt_token)

        assert isinstance(result, AuthenticatedUser)
        assert result.role == UserRole.USER
        assert result.email == "user@company.com"


@pytest.mark.unit
@pytest.mark.auth
class TestJWTValidatorEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_jwks_client(self):
        """Create mock JWKS client."""
        client = Mock(spec=JWKSClient)
        client.get_key_by_kid.return_value = {
            "kty": "RSA",
            "use": "sig",
            "kid": "test-key-id",
            "n": "test-modulus",
            "e": "AQAB",
        }
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
    def validator(self, mock_jwks_client, auth_config):
        """Create JWT validator instance."""
        return JWTValidator(jwks_client=mock_jwks_client, config=auth_config)

    def test_validate_token_extremely_long_token(self, validator):
        """Test validation with extremely long token."""
        # Create a very long token (simulating potential DoS)
        long_token = "a" * 10000 + ".b" * 10000 + ".c"

        with pytest.raises(JWTValidationError):
            validator.validate_token(long_token)

    def test_validate_token_unicode_characters(self, validator):
        """Test validation with unicode characters in token."""
        unicode_token = "émojis🚀.and.ünïcödé"  # noqa: S105

        with pytest.raises(JWTValidationError) as exc_info:
            validator.validate_token(unicode_token)
        assert exc_info.value.error_type == "invalid_format"

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_unexpected_error(self, mock_from_jwk, mock_decode, validator, valid_jwt_token):
        """Test handling of unexpected errors during validation."""
        from fastapi import HTTPException

        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = Exception("Unexpected error")

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_token(valid_jwt_token)

        assert exc_info.value.status_code == 401

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_none_payload(self, mock_decode, validator, valid_jwt_token):
        """Test handling when decode returns None."""
        mock_decode.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_token(valid_jwt_token)
        assert exc_info.value.status_code == 401

    @patch("src.auth.jwt_validator.jwt.decode")
    def test_validate_token_jwks_client_error(self, mock_decode, validator, valid_jwt_token):
        """Test handling of JWKS client errors."""
        from fastapi import HTTPException

        validator.jwks_client.get_key_by_kid.side_effect = Exception("JWKS error")
        mock_decode.side_effect = jwt.exceptions.InvalidTokenError("Key retrieval failed")

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_token(valid_jwt_token)

        assert exc_info.value.status_code == 401

    def test_validator_configuration_edge_cases(self):
        """Test validator with edge case configurations."""
        jwks_client = Mock(spec=JWKSClient)

        # Test with very long audience
        from src.auth.config import AuthenticationConfig
        config = AuthenticationConfig()
        long_audience = "https://" + "a" * 1000 + ".com"
        validator = JWTValidator(jwks_client, config=config, audience=long_audience)
        assert validator.audience == long_audience

        # Test with special characters in issuer
        special_issuer = "https://test-domain_123.example.com/auth"
        validator = JWTValidator(jwks_client, config=config, issuer=special_issuer)
        assert validator.issuer == special_issuer

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_payload_type_errors(self, mock_from_jwk, mock_decode, validator, valid_jwt_token):
        """Test handling of incorrect payload types."""
        mock_from_jwk.return_value = "mock_public_key"
        # Test with string payload instead of dict
        mock_decode.return_value = "not-a-dict"

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_token(valid_jwt_token)
        assert exc_info.value.status_code == 401

    @patch("src.auth.jwt_validator.jwt.decode")
    @patch("src.auth.jwt_validator.RSAAlgorithm.from_jwk")
    def test_validate_token_email_type_error(self, mock_from_jwk, mock_decode, validator, valid_jwt_token):
        """Test handling when email claim is not a string."""
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": 12345, "sub": "user123"}  # Integer instead of string

        with pytest.raises(JWTValidationError) as exc_info:
            validator.validate_token(valid_jwt_token)
        assert exc_info.value.error_type == "invalid_email"
