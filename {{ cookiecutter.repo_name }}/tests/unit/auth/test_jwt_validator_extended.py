"""
Extended comprehensive unit tests for JWT validator to achieve 80%+ coverage.

This module provides comprehensive test coverage for the JWTValidator class:
- Token validation with various scenarios
- Error handling and edge cases
- Email whitelist validation
- Role determination logic
- Token format validation
- Integration with JWKS client
"""

from unittest.mock import Mock, patch

import jwt
import pytest

from src.auth.config import AuthenticationConfig
from src.auth.jwks_client import JWKSClient
from src.auth.jwt_validator import JWTValidator
from src.auth.models import AuthenticatedUser, JWTValidationError, UserRole


class TestJWTValidatorInitialization:
    """Test JWT validator initialization and configuration."""

    def test_init_minimal_config(self):
        """Test initialization with minimal configuration."""
        jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        validator = JWTValidator(jwks_client=jwks_client, config=config)

        assert validator.jwks_client is jwks_client
        assert validator.audience is None
        assert validator.issuer is None
        assert validator.algorithm == "RS256"

    def test_init_full_config(self):
        """Test initialization with full configuration."""
        jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        validator = JWTValidator(
            jwks_client=jwks_client,
            config=config,
            audience="https://app.example.com",
            issuer="https://example.cloudflareaccess.com",
            algorithm="HS256",
        )

        assert validator.jwks_client is jwks_client
        assert validator.audience == "https://app.example.com"
        assert validator.issuer == "https://example.cloudflareaccess.com"
        assert validator.algorithm == "HS256"


class TestJWTValidatorTokenValidation:
    """Test core token validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        self.validator = JWTValidator(
            jwks_client=self.jwks_client,
            config=config,
            audience="https://app.example.com",
            issuer="https://example.cloudflareaccess.com",
        )

        # Mock a valid RSA key
        self.mock_key_dict = {"kty": "RSA", "kid": "test_key_id", "use": "sig", "n": "test_n_value", "e": "AQAB"}

    def test_validate_token_invalid_format_none(self):
        """Test validation with None token."""
        with pytest.raises(JWTValidationError, match="Invalid token format"):
            self.validator.validate_token(None)

    def test_validate_token_invalid_format_empty_string(self):
        """Test validation with empty string token."""
        with pytest.raises(JWTValidationError, match="Invalid token format"):
            self.validator.validate_token("")

    def test_validate_token_invalid_format_not_string(self):
        """Test validation with non-string token."""
        with pytest.raises(JWTValidationError, match="Invalid token format"):
            self.validator.validate_token(123)

    def test_validate_token_invalid_format_wrong_parts(self):
        """Test validation with wrong number of token parts."""
        with pytest.raises(JWTValidationError, match="Invalid token format"):
            self.validator.validate_token("only.two.parts.extra")

    def test_validate_token_invalid_format_one_part(self):
        """Test validation with only one part."""
        with pytest.raises(JWTValidationError, match="Invalid token format"):
            self.validator.validate_token("onlyonepart")

    @patch("jwt.get_unverified_header")
    def test_validate_token_decode_error(self, mock_get_header):
        """Test validation when JWT decode fails."""
        mock_get_header.side_effect = jwt.DecodeError("Invalid header")

        with pytest.raises(JWTValidationError, match="Invalid token format"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    def test_validate_token_missing_kid(self, mock_get_header):
        """Test validation when JWT header is missing kid."""
        mock_get_header.return_value = {"alg": "RS256"}  # No kid

        with pytest.raises(JWTValidationError, match="JWT token missing 'kid' in header"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    def test_validate_token_key_not_found(self, mock_get_header):
        """Test validation when JWKS key is not found."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "missing_key"}
        self.jwks_client.get_key_by_kid.return_value = None

        with pytest.raises(JWTValidationError, match="Key with kid 'missing_key' not found"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    def test_validate_token_invalid_jwk(self, mock_from_jwk, mock_get_header):
        """Test validation when JWK conversion fails."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.side_effect = ValueError("Invalid JWK format")

        with pytest.raises(JWTValidationError, match="Invalid JWK format"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_expired(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation with expired token."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = jwt.ExpiredSignatureError("Token has expired")

        with pytest.raises(JWTValidationError, match="Token has expired"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_invalid_signature(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation with invalid signature."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = jwt.InvalidSignatureError("Signature verification failed")

        with pytest.raises(JWTValidationError, match="Token signature verification failed"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_invalid_audience(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation with invalid audience."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = jwt.InvalidAudienceError("Invalid audience")

        with pytest.raises(JWTValidationError, match="Invalid token audience"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_invalid_issuer(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation with invalid issuer."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = jwt.InvalidIssuerError("Invalid issuer")

        with pytest.raises(JWTValidationError, match="Invalid token issuer"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_invalid_key(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation with invalid key."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = jwt.InvalidKeyError("Invalid key")

        with pytest.raises(JWTValidationError, match="Unable to verify token signature"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_missing_claim(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation with missing required claim."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = jwt.MissingRequiredClaimError("Missing required claim: email")

        with pytest.raises(JWTValidationError, match="Token missing required claim"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_invalid_token_generic(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation with generic invalid token error."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = self.mock_key_dict
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = jwt.InvalidTokenError("Generic token error")

        with pytest.raises(JWTValidationError, match="Invalid token"):
            self.validator.validate_token("header.payload.signature")


class TestJWTValidatorEmailValidation:
    """Test email validation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        self.validator = JWTValidator(jwks_client=self.jwks_client, config=config)

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_missing_email(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation when payload is missing email claim."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"sub": "user123"}  # No email

        with pytest.raises(JWTValidationError, match="JWT payload missing required 'email' claim"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_invalid_email_not_string(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation when email claim is not a string."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": 123}  # Not a string

        with pytest.raises(JWTValidationError, match="Invalid email format in JWT payload"):
            self.validator.validate_token("header.payload.signature")

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_invalid_email_no_at(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation when email claim doesn't contain @."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": "invalidemail"}  # No @

        with pytest.raises(JWTValidationError, match="Invalid email format in JWT payload"):
            self.validator.validate_token("header.payload.signature")


class TestJWTValidatorEmailWhitelist:
    """Test email whitelist validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        self.validator = JWTValidator(jwks_client=self.jwks_client, config=config)

    def testis_email_allowed_exact_match(self):
        """Test exact email match in whitelist."""
        email_whitelist = ["admin@example.com", "user@company.org"]

        assert self.validator.is_email_allowed("admin@example.com", email_whitelist) is True
        assert self.validator.is_email_allowed("user@company.org", email_whitelist) is True
        assert self.validator.is_email_allowed("other@example.com", email_whitelist) is False

    def testis_email_allowed_case_insensitive(self):
        """Test case insensitive email matching."""
        email_whitelist = ["Admin@Example.COM"]

        assert self.validator.is_email_allowed("admin@example.com", email_whitelist) is True
        assert self.validator.is_email_allowed("ADMIN@EXAMPLE.COM", email_whitelist) is True

    def testis_email_allowed_domain_match(self):
        """Test domain matching in whitelist."""
        email_whitelist = ["@example.com", "@company.org"]

        assert self.validator.is_email_allowed("user@example.com", email_whitelist) is True
        assert self.validator.is_email_allowed("admin@company.org", email_whitelist) is True
        assert self.validator.is_email_allowed("user@other.com", email_whitelist) is False

    def testis_email_allowed_domain_case_insensitive(self):
        """Test case insensitive domain matching."""
        email_whitelist = ["@Example.COM"]

        assert self.validator.is_email_allowed("user@example.com", email_whitelist) is True
        assert self.validator.is_email_allowed("user@EXAMPLE.COM", email_whitelist) is True

    def testis_email_allowed_mixed_whitelist(self):
        """Test whitelist with both exact emails and domains."""
        email_whitelist = ["specific@test.com", "@example.com"]

        assert self.validator.is_email_allowed("specific@test.com", email_whitelist) is True
        assert self.validator.is_email_allowed("anyone@example.com", email_whitelist) is True
        assert self.validator.is_email_allowed("other@test.com", email_whitelist) is False

    def testis_email_allowed_empty_whitelist(self):
        """Test behavior with empty whitelist."""
        email_whitelist = []

        assert self.validator.is_email_allowed("user@example.com", email_whitelist) is False

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_email_not_in_whitelist(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test validation when email is not in whitelist."""
        from fastapi import HTTPException

        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {"email": "blocked@example.com"}

        email_whitelist = ["allowed@example.com", "@company.org"]

        with pytest.raises(HTTPException) as exc_info:
            self.validator.validate_token("header.payload.signature", email_whitelist)

        assert exc_info.value.status_code == 401  # Authorization errors become authentication errors


class TestJWTValidatorRoleDetermination:
    """Test user role determination logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        self.validator = JWTValidator(jwks_client=self.jwks_client, config=config)

    def test_determine_user_role_admin_in_email(self):
        """Test admin role determination from email address."""
        payload = {"email": "admin@example.com"}

        role = self.validator._determine_user_role("admin@example.com", payload)
        assert role == UserRole.ADMIN

    def test_determine_user_role_owner_in_email(self):
        """Test admin role determination from owner email."""
        payload = {"email": "owner@example.com"}

        role = self.validator._determine_user_role("owner@example.com", payload)
        assert role == UserRole.ADMIN

    def test_determine_user_role_admin_in_groups(self):
        """Test admin role determination from JWT groups claim."""
        payload = {"email": "user@example.com", "groups": ["users", "admin", "editors"]}

        role = self.validator._determine_user_role("user@example.com", payload)
        assert role == UserRole.ADMIN

    def test_determine_user_role_admin_in_group_name(self):
        """Test admin role determination from group containing admin."""
        payload = {"email": "user@example.com", "groups": ["users", "system-admins", "editors"]}

        role = self.validator._determine_user_role("user@example.com", payload)
        assert role == UserRole.ADMIN

    def test_determine_user_role_default_user(self):
        """Test default user role determination."""
        payload = {"email": "user@example.com", "groups": ["users", "editors"]}

        role = self.validator._determine_user_role("user@example.com", payload)
        assert role == UserRole.USER

    def test_determine_user_role_no_groups(self):
        """Test role determination when no groups claim exists."""
        payload = {"email": "user@example.com"}

        role = self.validator._determine_user_role("user@example.com", payload)
        assert role == UserRole.USER

    def test_determine_user_role_non_list_groups(self):
        """Test role determination when groups claim is not a list."""
        payload = {"email": "user@example.com", "groups": "not-a-list"}

        role = self.validator._determine_user_role("user@example.com", payload)
        assert role == UserRole.USER

    def test_determine_user_role_non_string_groups(self):
        """Test role determination when groups contain non-string values."""
        payload = {"email": "user@example.com", "groups": ["users", 123, None, "editors"]}

        role = self.validator._determine_user_role("user@example.com", payload)
        assert role == UserRole.USER


class TestJWTValidatorTokenFormat:
    """Test token format validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        self.validator = JWTValidator(jwks_client=self.jwks_client, config=config)

    @patch("jwt.get_unverified_header")
    def test_validate_token_format_valid(self, mock_get_header):
        """Test valid token format validation."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}

        assert self.validator.validate_token_format("header.payload.signature") is True

    def test_validate_token_format_invalid_parts(self):
        """Test invalid token format - wrong number of parts."""
        assert self.validator.validate_token_format("only.two.parts.extra") is False
        assert self.validator.validate_token_format("onlyonepart") is False
        assert self.validator.validate_token_format("") is False

    @patch("jwt.get_unverified_header")
    def test_validate_token_format_decode_error(self, mock_get_header):
        """Test token format validation when header decode fails."""
        mock_get_header.side_effect = jwt.DecodeError("Invalid header")

        assert self.validator.validate_token_format("header.payload.signature") is False

    @patch("jwt.get_unverified_header")
    def test_validate_token_format_generic_exception(self, mock_get_header):
        """Test token format validation with generic exception."""
        mock_get_header.side_effect = ValueError("Generic error")

        assert self.validator.validate_token_format("header.payload.signature") is False


class TestJWTValidatorIntegration:
    """Test successful token validation integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwks_client = Mock(spec=JWKSClient)
        config = AuthenticationConfig()
        self.validator = JWTValidator(
            jwks_client=self.jwks_client,
            config=config,
            audience="https://app.example.com",
            issuer="https://example.cloudflareaccess.com",
        )

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_success_basic(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test successful token validation with basic user."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {
            "email": "user@example.com",
            "sub": "user123",
            "iat": 1234567890,
            "exp": 1234567890 + 3600,
        }

        result = self.validator.validate_token("header.payload.signature")

        assert isinstance(result, AuthenticatedUser)
        assert result.email == "user@example.com"
        assert result.role == UserRole.USER
        assert result.jwt_claims["email"] == "user@example.com"

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_success_admin(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test successful token validation with admin user."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {
            "email": "admin@example.com",
            "sub": "admin123",
            "groups": ["users", "admin"],
            "iat": 1234567890,
            "exp": 1234567890 + 3600,
        }

        result = self.validator.validate_token("header.payload.signature")

        assert isinstance(result, AuthenticatedUser)
        assert result.email == "admin@example.com"
        assert result.role == UserRole.ADMIN
        assert result.jwt_claims["groups"] == ["users", "admin"]

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_success_with_whitelist(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test successful token validation with email whitelist."""
        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.return_value = {
            "email": "allowed@example.com",
            "sub": "user123",
            "iat": 1234567890,
            "exp": 1234567890 + 3600,
        }

        email_whitelist = ["allowed@example.com", "@company.org"]
        result = self.validator.validate_token("header.payload.signature", email_whitelist)

        assert isinstance(result, AuthenticatedUser)
        assert result.email == "allowed@example.com"
        assert result.role == UserRole.USER

    @patch("jwt.get_unverified_header")
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk")
    @patch("jwt.decode")
    def test_validate_token_unexpected_error(self, mock_decode, mock_from_jwk, mock_get_header):
        """Test unexpected error handling during validation."""
        from fastapi import HTTPException

        mock_get_header.return_value = {"alg": "RS256", "kid": "test_key"}
        self.jwks_client.get_key_by_kid.return_value = {"kty": "RSA"}
        mock_from_jwk.return_value = "mock_public_key"
        mock_decode.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(HTTPException) as exc_info:
            self.validator.validate_token("header.payload.signature")

        assert exc_info.value.status_code == 401
