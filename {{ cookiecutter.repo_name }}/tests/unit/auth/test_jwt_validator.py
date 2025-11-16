"""
Unit tests for JWT token validation.

This module provides comprehensive test coverage for the JWTValidator class,
testing JWT signature verification, payload validation, email whitelist checking,
and role determination. Uses proper pytest markers for codecov integration.
"""

from unittest.mock import Mock, patch

import jwt
import pytest
from jwt.algorithms import RSAAlgorithm
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from src.auth.jwks_client import JWKSClient
from src.auth.jwt_validator import JWTValidator
from src.auth.models import AuthenticatedUser, JWTValidationError, UserRole


@pytest.mark.auth
class TestJWTValidatorInitialization:
    """Test cases for JWTValidator initialization."""

    def test_init_default_values(self):
        """Test initialization with default configuration values."""
        jwks_client = Mock(spec=JWKSClient)
        config = Mock()

        validator = JWTValidator(jwks_client, config)

        assert validator.jwks_client == jwks_client
        assert validator.audience is None
        assert validator.issuer is None
        assert validator.algorithm == "RS256"

    def test_init_custom_values(self):
        """Test initialization with custom configuration values."""
        jwks_client = Mock(spec=JWKSClient)
        config = Mock()
        audience = "my-app"
        issuer = "https://myteam.cloudflareaccess.com"
        algorithm = "RS512"

        validator = JWTValidator(jwks_client=jwks_client, config=config, audience=audience, issuer=issuer, algorithm=algorithm)

        assert validator.jwks_client == jwks_client
        assert validator.audience == audience
        assert validator.issuer == issuer
        assert validator.algorithm == algorithm

    def test_init_with_none_values(self):
        """Test initialization with explicitly None values."""
        jwks_client = Mock(spec=JWKSClient)
        config = Mock()

        validator = JWTValidator(jwks_client=jwks_client, config=config, audience=None, issuer=None)

        assert validator.audience is None
        assert validator.issuer is None


@pytest.mark.auth
class TestJWTValidatorValidateToken:
    """Test cases for JWT token validation."""

    def test_validate_token_success_minimal(self):
        """Test successful token validation with minimal claims."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        # Mock token and key retrieval
        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT decoding
        mock_payload = {
            "email": "test@example.com",
            "exp": 1234567890,
            "iat": 1234567800,
        }

        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
        ):
            jwks_client.get_key_by_kid.return_value = mock_key_dict
            result = validator.validate_token(token)

            assert isinstance(result, AuthenticatedUser)
            assert result.email == "test@example.com"
            assert result.role == UserRole.USER
            assert result.jwt_claims == mock_payload

    def test_validate_token_success_with_audience_issuer(self):
        """Test successful token validation with audience and issuer verification."""
        jwks_client = Mock(spec=JWKSClient)
        audience = "my-app"
        issuer = "https://test.cloudflareaccess.com"
        validator = JWTValidator(jwks_client, Mock(), audience=audience, issuer=issuer)

        # Mock token and key retrieval
        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT decoding with audience and issuer
        mock_payload = {
            "email": "test@example.com",
            "aud": audience,
            "iss": issuer,
            "exp": 1234567890,
            "iat": 1234567800,
        }

        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload) as mock_decode,
        ):
            jwks_client.get_key_by_kid.return_value = mock_key_dict
            result = validator.validate_token(token)

            assert isinstance(result, AuthenticatedUser)
            assert result.email == "test@example.com"

            # Verify jwt.decode was called with correct parameters
            mock_decode.assert_called_once_with(
                token,
                mock_public_key,
                algorithms=["RS256"],
                audience=audience,
                issuer=issuer,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "email"],
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

    def test_validate_token_success_with_email_whitelist(self):
        """Test successful token validation with email whitelist."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        # Mock token and key retrieval
        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT decoding
        mock_payload = {
            "email": "allowed@example.com",
            "exp": 1234567890,
            "iat": 1234567800,
        }

        email_whitelist = ["allowed@example.com", "@company.com"]

        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
        ):
            jwks_client.get_key_by_kid.return_value = mock_key_dict
            result = validator.validate_token(token, email_whitelist=email_whitelist)

            assert isinstance(result, AuthenticatedUser)
            assert result.email == "allowed@example.com"

    def test_validate_token_missing_kid_in_header(self):
        """Test token validation fails when 'kid' is missing from header."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105

        with patch("jwt.get_unverified_header", return_value={}):  # No 'kid'
            with pytest.raises(JWTValidationError) as exc_info:
                validator.validate_token(token)

            assert "JWT token missing 'kid' in header" in str(exc_info.value)
            assert exc_info.value.error_type == "missing_kid"

    def test_validate_token_key_not_found_in_jwks(self):
        """Test token validation fails when key ID is not found in JWKS."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "nonexistent-key"

        with patch("jwt.get_unverified_header", return_value={"kid": kid}):
            jwks_client.get_key_by_kid.return_value = None  # Key not found

            with pytest.raises(JWTValidationError) as exc_info:
                validator.validate_token(token)

            assert f"Key with kid '{kid}' not found in JWKS" in str(exc_info.value)
            assert exc_info.value.error_type == "key_not_found"

    def test_validate_token_invalid_jwk_format(self):
        """Test token validation fails when JWK format is invalid."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"invalid": "format"}

        jwks_client.get_key_by_kid.return_value = mock_key_dict
        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", side_effect=ValueError("Invalid JWK")),
            pytest.raises(JWTValidationError) as exc_info,
        ):
            validator.validate_token(token)

        assert "Invalid JWK format" in str(exc_info.value)
        assert exc_info.value.error_type == "invalid_jwk"

    def test_validate_token_expired_signature(self):
        """Test token validation fails when token is expired."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        jwks_client.get_key_by_kid.return_value = mock_key_dict
        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", side_effect=ExpiredSignatureError("Token expired")),
            pytest.raises(JWTValidationError) as exc_info,
        ):
            validator.validate_token(token)

        assert "Token has expired" in str(exc_info.value)
        assert exc_info.value.error_type == "expired_token"

    def test_validate_token_invalid_token(self):
        """Test token validation fails with invalid token."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        jwks_client.get_key_by_kid.return_value = mock_key_dict
        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", side_effect=InvalidTokenError("Invalid signature")),
            pytest.raises(JWTValidationError) as exc_info,
        ):
            validator.validate_token(token)

        assert "Invalid token" in str(exc_info.value)
        assert exc_info.value.error_type == "invalid_token"

    def test_validate_token_missing_email_claim(self):
        """Test token validation fails when email claim is missing."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT payload without email
        mock_payload = {
            "exp": 1234567890,
            "iat": 1234567800,
        }

        jwks_client.get_key_by_kid.return_value = mock_key_dict
        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
            pytest.raises(JWTValidationError) as exc_info,
        ):
            validator.validate_token(token)

        assert "JWT payload missing required 'email' claim" in str(exc_info.value)
        assert exc_info.value.error_type == "missing_email"

    def test_validate_token_invalid_email_format(self):
        """Test token validation fails with invalid email format."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT payload with invalid email
        mock_payload = {
            "email": "invalid-email-format",  # Missing @
            "exp": 1234567890,
            "iat": 1234567800,
        }

        jwks_client.get_key_by_kid.return_value = mock_key_dict
        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
            pytest.raises(JWTValidationError) as exc_info,
        ):
            validator.validate_token(token)

        assert "Invalid email format in JWT payload" in str(exc_info.value)
        assert exc_info.value.error_type == "invalid_email"

    def test_validate_token_email_not_authorized(self):
        """Test token validation fails when email is not in whitelist."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT payload with unauthorized email
        mock_payload = {
            "email": "unauthorized@example.com",
            "exp": 1234567890,
            "iat": 1234567800,
        }

        email_whitelist = ["@company.com", "allowed@example.com"]

        jwks_client.get_key_by_kid.return_value = mock_key_dict
        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
        ):
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                validator.validate_token(token, email_whitelist=email_whitelist)

            # The AuthExceptionHandler converts email authorization to HTTPException
            assert exc_info.value.status_code == 401  # Authentication error wraps permission error
            assert "Authentication failed" in str(exc_info.value.detail)

    def test_validate_token_non_string_email(self):
        """Test token validation fails when email is not a string."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT payload with non-string email
        mock_payload = {
            "email": 12345,  # Non-string email
            "exp": 1234567890,
            "iat": 1234567800,
        }

        jwks_client.get_key_by_kid.return_value = mock_key_dict
        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
            pytest.raises(JWTValidationError) as exc_info,
        ):
            validator.validate_token(token)

        assert "Invalid email format in JWT payload" in str(exc_info.value)
        assert exc_info.value.error_type == "invalid_email"

    def test_validate_token_unexpected_error(self):
        """Test token validation handles unexpected errors."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105

        with patch("jwt.get_unverified_header", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(JWTValidationError) as exc_info:
                validator.validate_token(token)

            assert "Invalid token format" in str(exc_info.value)
            assert exc_info.value.error_type == "invalid_format"

    def test_validate_token_admin_role_from_email(self):
        """Test admin role determination from email."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT payload with admin email
        mock_payload = {
            "email": "admin@example.com",
            "exp": 1234567890,
            "iat": 1234567800,
        }

        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
        ):
            jwks_client.get_key_by_kid.return_value = mock_key_dict
            result = validator.validate_token(token)

            assert result.role == UserRole.ADMIN

    def test_validate_token_admin_role_from_groups(self):
        """Test admin role determination from groups claim."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "test-key-id"
        mock_key_dict = {"kid": kid, "kty": "RSA", "n": "test", "e": "AQAB"}
        mock_public_key = Mock()

        # Mock JWT payload with admin groups
        mock_payload = {
            "email": "user@example.com",
            "groups": ["users", "admin-group"],
            "exp": 1234567890,
            "iat": 1234567800,
        }

        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
        ):
            jwks_client.get_key_by_kid.return_value = mock_key_dict
            result = validator.validate_token(token)

            assert result.role == UserRole.ADMIN


@pytest.mark.auth
class TestJWTValidatorIsEmailAllowed:
    """Test cases for email whitelist validation."""

    def test_is_email_allowed_exact_match(self):
        """Test email whitelist with exact email match."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "test@example.com"
        whitelist = ["test@example.com", "@company.com"]

        result = validator._is_email_allowed(email, whitelist)

        assert result is True

    def test_is_email_allowed_domain_match(self):
        """Test email whitelist with domain match."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@company.com"
        whitelist = ["@company.com", "specific@example.com"]

        result = validator._is_email_allowed(email, whitelist)

        assert result is True

    def test_is_email_allowed_case_insensitive_exact(self):
        """Test email whitelist is case insensitive for exact matches."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "Test@Example.COM"
        whitelist = ["test@example.com"]

        result = validator._is_email_allowed(email, whitelist)

        assert result is True

    def test_is_email_allowed_case_insensitive_domain(self):
        """Test email whitelist is case insensitive for domain matches."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "User@Company.COM"
        whitelist = ["@company.com"]

        result = validator._is_email_allowed(email, whitelist)

        assert result is True

    def test_is_email_allowed_not_in_whitelist(self):
        """Test email not in whitelist is rejected."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "unauthorized@badcompany.com"
        whitelist = ["@company.com", "allowed@example.com"]

        result = validator._is_email_allowed(email, whitelist)

        assert result is False

    def test_is_email_allowed_empty_whitelist(self):
        """Test email validation with empty whitelist."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "test@example.com"
        whitelist = []

        result = validator._is_email_allowed(email, whitelist)

        assert result is False

    def test_is_email_allowed_multiple_domain_matches(self):
        """Test email validation with multiple domain matches."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@company.com"
        whitelist = ["@example.com", "@company.com", "@another.com"]

        result = validator._is_email_allowed(email, whitelist)

        assert result is True

    def test_is_email_allowed_partial_domain_no_match(self):
        """Test that partial domain matches don't work without @ prefix."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@company.com"
        whitelist = ["company.com"]  # Missing @ prefix

        result = validator._is_email_allowed(email, whitelist)

        assert result is False


@pytest.mark.auth
class TestJWTValidatorDetermineUserRole:
    """Test cases for user role determination."""

    def test_determine_user_role_admin_from_email_admin(self):
        """Test admin role determination from 'admin' in email."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "admin@example.com"
        payload = {"email": email}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.ADMIN

    def test_determine_user_role_admin_from_email_owner(self):
        """Test admin role determination from 'owner' in email."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "owner@example.com"
        payload = {"email": email}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.ADMIN

    def test_determine_user_role_admin_from_email_case_insensitive(self):
        """Test admin role determination is case insensitive."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "ADMIN@EXAMPLE.COM"
        payload = {"email": email}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.ADMIN

    def test_determine_user_role_admin_from_groups_list(self):
        """Test admin role determination from groups claim."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@example.com"
        payload = {"email": email, "groups": ["users", "admin-users", "developers"]}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.ADMIN

    def test_determine_user_role_admin_from_groups_case_insensitive(self):
        """Test admin role determination from groups is case insensitive."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@example.com"
        payload = {"email": email, "groups": ["users", "ADMIN-GROUP"]}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.ADMIN

    def test_determine_user_role_user_default(self):
        """Test default user role assignment."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "regular@example.com"
        payload = {"email": email}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.USER

    def test_determine_user_role_user_with_non_admin_groups(self):
        """Test user role with non-admin groups."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@example.com"
        payload = {"email": email, "groups": ["users", "developers", "testers"]}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.USER

    def test_determine_user_role_groups_not_list(self):
        """Test role determination when groups is not a list."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@example.com"
        payload = {"email": email, "groups": "not-a-list"}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.USER

    def test_determine_user_role_groups_missing(self):
        """Test role determination when groups claim is missing."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@example.com"
        payload = {"email": email}

        result = validator._determine_user_role(email, payload)

        assert result == UserRole.USER

    def test_determine_user_role_groups_with_non_string_elements(self):
        """Test role determination when groups contains non-string elements."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        email = "user@example.com"
        payload = {"email": email, "groups": ["users", 123, {"not": "string"}, "admin-group"]}

        result = validator._determine_user_role(email, payload)

        # Should still find admin-group even with mixed types
        assert result == UserRole.ADMIN


@pytest.mark.auth
class TestJWTValidatorValidateTokenFormat:
    """Test cases for token format validation."""

    def test_validate_token_format_valid_token(self):
        """Test format validation with valid JWT token."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105

        with patch("jwt.get_unverified_header", return_value={"alg": "RS256"}):
            result = validator.validate_token_format(token)
            assert result is True

    def test_validate_token_format_invalid_parts_count(self):
        """Test format validation with wrong number of parts."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        # Token with only 2 parts
        token = "header.payload"  # noqa: S105

        result = validator.validate_token_format(token)
        assert result is False

    def test_validate_token_format_too_many_parts(self):
        """Test format validation with too many parts."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        # Token with 4 parts
        token = "header.payload.signature.extra"  # noqa: S105

        result = validator.validate_token_format(token)
        assert result is False

    def test_validate_token_format_empty_token(self):
        """Test format validation with empty token."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = ""

        result = validator.validate_token_format(token)
        assert result is False

    def test_validate_token_format_invalid_header(self):
        """Test format validation with invalid header."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "invalid.payload.signature"  # noqa: S105

        with patch("jwt.get_unverified_header", side_effect=jwt.DecodeError("Invalid header")):
            result = validator.validate_token_format(token)
            assert result is False

    def test_validate_token_format_exception_handling(self):
        """Test format validation handles any exception."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105

        with patch("jwt.get_unverified_header", side_effect=RuntimeError("Unexpected error")):
            result = validator.validate_token_format(token)
            assert result is False


@pytest.mark.auth
class TestJWTValidatorIntegration:
    """Integration test cases for complete JWT validation workflows."""

    def test_complete_validation_workflow_success(self):
        """Test complete successful JWT validation workflow."""
        # Mock JWKS client
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock(), audience="my-app", issuer="https://test.cloudflareaccess.com")

        # Mock complete workflow
        token = "test-header.test-payload.test-signature"  # noqa: S105
        kid = "test-key"

        # Mock JWKS response
        mock_key_dict = {"kid": kid, "kty": "RSA", "alg": "RS256", "use": "sig", "n": "example_modulus", "e": "AQAB"}

        # Mock RSA public key
        mock_public_key = Mock()

        # Mock JWT payload
        mock_payload = {
            "email": "test@example.com",
            "aud": "my-app",
            "iss": "https://test.cloudflareaccess.com",
            "exp": 1234567890,
            "iat": 1234567800,
            "groups": ["users", "testers"],
        }

        email_whitelist = ["@example.com", "allowed@test.com"]

        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid, "alg": "RS256"}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
        ):
            jwks_client.get_key_by_kid.return_value = mock_key_dict
            result = validator.validate_token(token, email_whitelist=email_whitelist)

            # Verify successful validation
            assert isinstance(result, AuthenticatedUser)
            assert result.email == "test@example.com"
            assert result.role == UserRole.USER
            assert result.jwt_claims == mock_payload

            # Verify all methods were called correctly
            jwks_client.get_key_by_kid.assert_called_once_with(kid)

    def test_complete_validation_workflow_admin_user(self):
        """Test complete JWT validation workflow for admin user."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        token = "header.payload.signature"  # noqa: S105
        kid = "admin-key"

        # Mock JWKS response
        mock_key_dict = {
            "kid": kid,
            "kty": "RSA",
            "alg": "RS256",
        }

        mock_public_key = Mock()

        # Mock JWT payload for admin user
        mock_payload = {
            "email": "admin@company.com",
            "exp": 1234567890,
            "iat": 1234567800,
            "groups": ["users", "admin-staff", "developers"],
        }

        with (
            patch("jwt.get_unverified_header", return_value={"kid": kid}),
            patch.object(RSAAlgorithm, "from_jwk", return_value=mock_public_key),
            patch("jwt.decode", return_value=mock_payload),
        ):
            jwks_client.get_key_by_kid.return_value = mock_key_dict
            result = validator.validate_token(token)

            # Should be admin due to both email and groups
            assert result.role == UserRole.ADMIN
            assert result.email == "admin@company.com"

    def test_complete_validation_workflow_multiple_failure_modes(self):
        """Test validation workflow handles multiple types of failures."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        # Test 1: Missing kid
        token1 = "header.payload.signature"
        with patch("jwt.get_unverified_header", return_value={}):
            with pytest.raises(JWTValidationError) as exc_info:
                validator.validate_token(token1)
            assert exc_info.value.error_type == "missing_kid"

        # Test 2: Key not found
        token2 = "header.payload.signature"
        with patch("jwt.get_unverified_header", return_value={"kid": "missing"}):
            jwks_client.get_key_by_kid.return_value = None
            with pytest.raises(JWTValidationError) as exc_info:
                validator.validate_token(token2)
            assert exc_info.value.error_type == "key_not_found"

        # Test 3: Invalid JWK
        token3 = "header.payload.signature"
        jwks_client.get_key_by_kid.return_value = {"invalid": "jwk"}
        with (
            patch("jwt.get_unverified_header", return_value={"kid": "test"}),
            patch.object(RSAAlgorithm, "from_jwk", side_effect=ValueError("Bad JWK")),
            pytest.raises(JWTValidationError) as exc_info,
        ):
            validator.validate_token(token3)
        assert exc_info.value.error_type == "invalid_jwk"

    def test_format_validation_integration(self):
        """Test format validation integration with different token formats."""
        jwks_client = Mock(spec=JWKSClient)
        validator = JWTValidator(jwks_client, Mock())

        # Valid format
        valid_token = "header.payload.signature"  # noqa: S105
        with patch("jwt.get_unverified_header", return_value={"alg": "RS256"}):
            assert validator.validate_token_format(valid_token) is True

        # Invalid formats
        invalid_tokens = [
            "header.payload",  # Missing signature
            "header",  # Missing payload and signature
            "",  # Empty
            "header.payload.signature.extra",  # Too many parts
        ]

        for token in invalid_tokens:
            assert validator.validate_token_format(token) is False
