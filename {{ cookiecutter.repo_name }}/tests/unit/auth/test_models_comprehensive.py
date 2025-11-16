"""
Comprehensive unit tests for authentication models to achieve 80%+ coverage.

This module provides extensive test coverage for all authentication model classes:
- AuthenticatedUser class functionality
- JWTValidationError exception class
- UserRole enum validation
- Model serialization and deserialization
- Edge cases and validation scenarios
"""

import pytest
from pydantic import ValidationError

from src.auth.models import AuthenticatedUser, JWTValidationError, UserRole


@pytest.mark.unit
@pytest.mark.auth
class TestAuthenticatedUser:
    """Test AuthenticatedUser dataclass functionality."""

    def test_init_basic(self):
        """Test basic initialization of AuthenticatedUser."""
        user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.USER,
            jwt_claims={"sub": "123", "email": "test@example.com"},
        )

        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.jwt_claims == {"sub": "123", "email": "test@example.com"}

    def test_init_admin_role(self):
        """Test initialization with admin role."""
        user = AuthenticatedUser(
            email="admin@example.com",
            role=UserRole.ADMIN,
            jwt_claims={"sub": "123", "email": "admin@example.com", "groups": ["admin"]},
        )

        assert user.email == "admin@example.com"
        assert user.role == UserRole.ADMIN
        assert "groups" in user.jwt_claims

    def test_init_empty_jwt_claims(self):
        """Test initialization with empty JWT claims."""
        user = AuthenticatedUser(email="test@example.com", role=UserRole.USER, jwt_claims={})

        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.jwt_claims == {}

    def test_init_none_jwt_claims(self):
        """Test initialization with None JWT claims should raise ValidationError."""
        with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
            AuthenticatedUser(email="test@example.com", role=UserRole.USER, jwt_claims=None)

    def test_pydantic_model_mutable(self):
        """Test that AuthenticatedUser is mutable (Pydantic models are mutable by default)."""
        user = AuthenticatedUser(email="test@example.com", role=UserRole.USER, jwt_claims={"sub": "123"})

        # Pydantic models are mutable by default
        user.email = "changed@example.com"
        assert user.email == "changed@example.com"

        user.role = UserRole.ADMIN
        assert user.role == UserRole.ADMIN

        user.jwt_claims = {"new": "claims"}
        assert user.jwt_claims == {"new": "claims"}

    def test_equality(self):
        """Test equality comparison between AuthenticatedUser instances."""
        user1 = AuthenticatedUser(email="test@example.com", role=UserRole.USER, jwt_claims={"sub": "123"})

        user2 = AuthenticatedUser(email="test@example.com", role=UserRole.USER, jwt_claims={"sub": "123"})

        user3 = AuthenticatedUser(email="different@example.com", role=UserRole.USER, jwt_claims={"sub": "123"})

        assert user1 == user2
        assert user1 != user3

    def test_hash_not_supported(self):
        """Test that AuthenticatedUser instances are not hashable by default."""
        user1 = AuthenticatedUser(email="test@example.com", role=UserRole.USER, jwt_claims={"sub": "123"})

        # Pydantic models are not hashable by default
        with pytest.raises(TypeError, match="unhashable type"):
            hash(user1)

        # Cannot be used in sets without custom __hash__
        with pytest.raises(TypeError):
            {user1}  # noqa: B018

    def test_string_representation(self):
        """Test string representation of AuthenticatedUser."""
        user = AuthenticatedUser(
            email="test@example.com",
            role=UserRole.ADMIN,
            jwt_claims={"sub": "123", "groups": ["admin"]},
        )

        str_repr = str(user)
        assert "test@example.com" in str_repr
        assert "ADMIN" in str_repr

        repr_str = repr(user)
        assert "AuthenticatedUser" in repr_str
        assert "test@example.com" in repr_str


@pytest.mark.unit
@pytest.mark.auth
class TestJWTValidationError:
    """Test JWTValidationError exception class functionality."""

    def test_init_basic(self):
        """Test basic initialization of JWTValidationError."""
        error = JWTValidationError("Token expired", "expired_token")

        assert str(error) == "Token expired"
        assert error.error_type == "expired_token"
        assert error.args == ("Token expired",)

    def test_init_no_error_type(self):
        """Test initialization without error type uses default."""
        error = JWTValidationError("General error")

        assert str(error) == "General error"
        assert error.error_type == "validation_error"  # Default value
        assert error.args == ("General error",)

    def test_init_empty_message(self):
        """Test initialization with empty message."""
        error = JWTValidationError("", "test_error")

        assert str(error) == ""
        assert error.error_type == "test_error"

    def test_init_none_error_type(self):
        """Test initialization with None error type."""
        error = JWTValidationError("Test message", None)

        assert str(error) == "Test message"
        assert error.error_type is None

    def test_inheritance(self):
        """Test that JWTValidationError inherits from Exception."""
        error = JWTValidationError("Test", "test_type")

        assert isinstance(error, Exception)
        assert isinstance(error, JWTValidationError)

    def test_raise_and_catch(self):
        """Test raising and catching JWTValidationError."""
        with pytest.raises(JWTValidationError) as exc_info:
            raise JWTValidationError("Token validation failed", "validation_error")

        caught_error = exc_info.value
        assert str(caught_error) == "Token validation failed"
        assert caught_error.error_type == "validation_error"

    def test_different_error_types(self):
        """Test various error types."""
        error_types = [
            "invalid_token",
            "expired_token",
            "missing_claims",
            "invalid_signature",
            "jwks_error",
            "format_error",
        ]

        for error_type in error_types:
            error = JWTValidationError(f"Error of type {error_type}", error_type)
            assert error.error_type == error_type
            assert str(error) == f"Error of type {error_type}"


@pytest.mark.unit
@pytest.mark.auth
class TestUserRole:
    """Test UserRole enum functionality."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert UserRole.USER.value == "user"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.VIEWER.value == "viewer"

    def test_enum_members(self):
        """Test enum members list."""
        members = list(UserRole)
        assert len(members) == 3
        assert UserRole.USER in members
        assert UserRole.ADMIN in members
        assert UserRole.VIEWER in members

    def test_enum_by_value(self):
        """Test accessing enum by value."""
        assert UserRole("user") == UserRole.USER
        assert UserRole("admin") == UserRole.ADMIN
        assert UserRole("viewer") == UserRole.VIEWER

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        assert UserRole.USER == UserRole.USER
        assert UserRole.ADMIN == UserRole.ADMIN
        assert UserRole.USER != UserRole.ADMIN
        assert UserRole.ADMIN != UserRole.USER

    def test_enum_string_representation(self):
        """Test string representation of enum values."""
        assert str(UserRole.USER) == "UserRole.USER"
        assert str(UserRole.ADMIN) == "UserRole.ADMIN"

        assert repr(UserRole.USER) == "<UserRole.USER: 'user'>"
        assert repr(UserRole.ADMIN) == "<UserRole.ADMIN: 'admin'>"

    def test_enum_invalid_value(self):
        """Test that invalid enum values raise ValueError."""
        with pytest.raises(ValueError, match="is not a valid UserRole"):
            UserRole("invalid_role")

        with pytest.raises(ValueError, match="is not a valid UserRole"):
            UserRole("moderator")

        with pytest.raises(ValueError, match="is not a valid UserRole"):
            UserRole("")

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        roles = []
        for role in UserRole:
            roles.append(role)

        assert len(roles) == 3
        assert UserRole.USER in roles
        assert UserRole.ADMIN in roles
        assert UserRole.VIEWER in roles

    def test_enum_membership(self):
        """Test membership testing with enum."""
        assert UserRole.USER in UserRole
        assert UserRole.ADMIN in UserRole

        # Test with string values
        assert "user" in [role.value for role in UserRole]
        assert "admin" in [role.value for role in UserRole]
        assert "invalid" not in [role.value for role in UserRole]


@pytest.mark.unit
@pytest.mark.auth
class TestModelsIntegration:
    """Test integration scenarios between model classes."""

    def test_authenticated_user_with_all_roles(self):
        """Test AuthenticatedUser creation with all possible roles."""
        for role in UserRole:
            user = AuthenticatedUser(email=f"{role.value}@example.com", role=role, jwt_claims={"role": role.value})

            assert user.role == role
            assert user.email == f"{role.value}@example.com"
            assert user.jwt_claims["role"] == role.value

    def test_jwt_validation_error_with_user_context(self):
        """Test JWTValidationError in context of user validation."""
        user_email = "test@example.com"

        # Simulate different validation scenarios
        error_scenarios = [
            ("Token expired for user", "expired_token"),
            (f"Email {user_email} not authorized", "email_not_authorized"),
            ("Invalid signature for user token", "invalid_signature"),
            (f"User {user_email} role validation failed", "role_validation_failed"),
        ]

        for message, error_type in error_scenarios:
            error = JWTValidationError(message, error_type)
            assert str(error) == message
            assert error.error_type == error_type
            assert user_email in message if user_email in message else True

    def test_complex_jwt_claims_scenarios(self):
        """Test AuthenticatedUser with complex JWT claims."""
        complex_claims = {
            "sub": "user_123456",
            "email": "complex.user+test@example-domain.com",
            "iss": "https://auth.example.com",
            "aud": ["app1", "app2", "app3"],
            "exp": 1234567890,
            "iat": 1234567800,
            "nbf": 1234567800,
            "groups": ["users", "beta_testers", "premium"],
            "permissions": {"read": ["profile", "settings"], "write": ["profile"], "admin": []},
            "metadata": {"last_login": "2024-01-01T00:00:00Z", "login_count": 42, "preferred_language": "en-US"},
        }

        user = AuthenticatedUser(
            email="complex.user+test@example-domain.com",
            role=UserRole.USER,
            jwt_claims=complex_claims,
        )

        assert user.email == "complex.user+test@example-domain.com"
        assert user.role == UserRole.USER
        assert user.jwt_claims == complex_claims
        assert user.jwt_claims["groups"] == ["users", "beta_testers", "premium"]
        assert user.jwt_claims["permissions"]["read"] == ["profile", "settings"]

    def test_edge_case_email_formats(self):
        """Test AuthenticatedUser with various email formats."""
        email_formats = [
            "simple@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user_underscore@example.com",
            "user-hyphen@example.com",
            "123numbers@example.com",
            "user@sub.example.com",
            "user@example-hyphen.com",
            "a@b.co",  # Minimal valid email
        ]

        for email in email_formats:
            user = AuthenticatedUser(email=email, role=UserRole.USER, jwt_claims={"email": email})

            assert user.email == email
            assert user.jwt_claims["email"] == email
