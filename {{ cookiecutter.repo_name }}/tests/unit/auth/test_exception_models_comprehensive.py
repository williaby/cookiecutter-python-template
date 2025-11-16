"""
Comprehensive unit tests for authentication exception models to achieve 100% coverage.

This module provides complete test coverage for all authentication exception classes:
- JWKSError exception class
- AuthenticationError exception class
- Exception inheritance and behavior
- Error message handling and attributes
"""

import pytest

from src.auth.models import AuthenticationError, JWKSError


@pytest.mark.unit
@pytest.mark.auth
class TestJWKSError:
    """Test JWKSError exception class functionality."""

    def test_init_basic(self):
        """Test basic initialization of JWKSError."""
        error = JWKSError("JWKS fetch failed", "fetch_error")

        assert str(error) == "JWKS fetch failed"
        assert error.error_type == "fetch_error"
        assert error.message == "JWKS fetch failed"
        assert error.args == ("JWKS fetch failed",)

    def test_init_no_error_type(self):
        """Test initialization without error type uses default."""
        error = JWKSError("General JWKS error")

        assert str(error) == "General JWKS error"
        assert error.error_type == "jwks_error"  # Default value
        assert error.message == "General JWKS error"
        assert error.args == ("General JWKS error",)

    def test_init_empty_message(self):
        """Test initialization with empty message."""
        error = JWKSError("", "test_error")

        assert str(error) == ""
        assert error.error_type == "test_error"
        assert error.message == ""

    def test_init_none_error_type(self):
        """Test initialization with None error type."""
        error = JWKSError("Test message", None)

        assert str(error) == "Test message"
        assert error.error_type is None
        assert error.message == "Test message"

    def test_inheritance(self):
        """Test that JWKSError inherits from Exception."""
        error = JWKSError("Test", "test_type")

        assert isinstance(error, Exception)
        assert isinstance(error, JWKSError)

    def test_raise_and_catch(self):
        """Test raising and catching JWKSError."""
        with pytest.raises(JWKSError) as exc_info:
            raise JWKSError("JWKS operation failed", "operation_error")

        caught_error = exc_info.value
        assert str(caught_error) == "JWKS operation failed"
        assert caught_error.error_type == "operation_error"
        assert caught_error.message == "JWKS operation failed"

    def test_different_error_types(self):
        """Test various JWKS error types."""
        error_types = [
            "fetch_error",
            "parse_error",
            "cache_error",
            "timeout_error",
            "connection_error",
            "invalid_response",
        ]

        for error_type in error_types:
            error = JWKSError(f"Error of type {error_type}", error_type)
            assert error.error_type == error_type
            assert str(error) == f"Error of type {error_type}"
            assert error.message == f"Error of type {error_type}"

    def test_message_attribute_consistency(self):
        """Test that message attribute is consistent with str representation."""
        test_cases = [
            ("Simple message", "error_type"),
            ("", "empty_message"),
            ("Multi\nLine\nMessage", "multiline"),
            ("Message with special chars: !@#$%^&*()", "special_chars"),
        ]

        for message, error_type in test_cases:
            error = JWKSError(message, error_type)
            assert error.message == message
            assert str(error) == message
            assert error.error_type == error_type


@pytest.mark.unit
@pytest.mark.auth
class TestAuthenticationError:
    """Test AuthenticationError exception class functionality."""

    def test_init_basic(self):
        """Test basic initialization of AuthenticationError."""
        error = AuthenticationError("Authentication failed", 401)

        assert str(error) == "Authentication failed"
        assert error.message == "Authentication failed"
        assert error.status_code == 401
        assert error.args == ("Authentication failed",)

    def test_init_no_status_code(self):
        """Test initialization without status code uses default."""
        error = AuthenticationError("General auth error")

        assert str(error) == "General auth error"
        assert error.message == "General auth error"
        assert error.status_code == 401  # Default value
        assert error.args == ("General auth error",)

    def test_init_custom_status_code(self):
        """Test initialization with custom status code."""
        error = AuthenticationError("Forbidden access", 403)

        assert str(error) == "Forbidden access"
        assert error.message == "Forbidden access"
        assert error.status_code == 403

    def test_init_empty_message(self):
        """Test initialization with empty message."""
        error = AuthenticationError("", 500)

        assert str(error) == ""
        assert error.message == ""
        assert error.status_code == 500

    def test_inheritance(self):
        """Test that AuthenticationError inherits from Exception."""
        error = AuthenticationError("Test", 401)

        assert isinstance(error, Exception)
        assert isinstance(error, AuthenticationError)

    def test_raise_and_catch(self):
        """Test raising and catching AuthenticationError."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise AuthenticationError("Access denied", 403)

        caught_error = exc_info.value
        assert str(caught_error) == "Access denied"
        assert caught_error.message == "Access denied"
        assert caught_error.status_code == 403

    def test_different_status_codes(self):
        """Test various HTTP status codes."""
        status_codes = [
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
        ]

        for status_code, description in status_codes:
            error = AuthenticationError(f"Error: {description}", status_code)
            assert error.status_code == status_code
            assert str(error) == f"Error: {description}"
            assert error.message == f"Error: {description}"

    def test_message_attribute_consistency(self):
        """Test that message attribute is consistent with str representation."""
        test_cases = [
            ("User not found", 404),
            ("", 401),
            ("Session expired", 401),
            ("Rate limit exceeded", 429),
            ("Server error occurred", 500),
        ]

        for message, status_code in test_cases:
            error = AuthenticationError(message, status_code)
            assert error.message == message
            assert str(error) == message
            assert error.status_code == status_code


@pytest.mark.unit
@pytest.mark.auth
class TestExceptionModelsIntegration:
    """Test integration scenarios between exception classes."""

    def test_exception_chaining_jwks_error(self):
        """Test exception chaining with JWKSError."""
        try:
            # Simulate a chain of errors
            try:
                raise ValueError("Original connection error")
            except ValueError as e:
                raise JWKSError("Failed to fetch JWKS", "fetch_error") from e
        except JWKSError as jwks_error:
            assert str(jwks_error) == "Failed to fetch JWKS"  # noqa: PT017
            assert jwks_error.error_type == "fetch_error"  # noqa: PT017
            assert jwks_error.__cause__ is not None  # noqa: PT017
            assert isinstance(jwks_error.__cause__, ValueError)  # noqa: PT017

    def test_exception_chaining_auth_error(self):
        """Test exception chaining with AuthenticationError."""
        try:
            # Simulate a chain of errors
            try:
                raise JWKSError("JWKS fetch failed", "fetch_error")
            except JWKSError as e:
                raise AuthenticationError("Authentication failed due to JWKS error", 500) from e
        except AuthenticationError as auth_error:
            assert str(auth_error) == "Authentication failed due to JWKS error"  # noqa: PT017
            assert auth_error.status_code == 500  # noqa: PT017
            assert auth_error.__cause__ is not None  # noqa: PT017
            assert isinstance(auth_error.__cause__, JWKSError)  # noqa: PT017

    def test_multiple_exception_types_in_context(self):
        """Test using multiple exception types in various contexts."""
        # Test creating and comparing different exception types
        jwks_error = JWKSError("JWKS error", "fetch_error")
        auth_error = AuthenticationError("Auth error", 401)

        # They should be different types
        assert not isinstance(jwks_error, type(auth_error))
        assert not isinstance(jwks_error, AuthenticationError)
        assert not isinstance(auth_error, JWKSError)

        # But both should be Exceptions
        assert isinstance(jwks_error, Exception)
        assert isinstance(auth_error, Exception)

    def test_exception_in_error_handling_scenarios(self):
        """Test exceptions in realistic error handling scenarios."""

        def simulate_jwks_operation():
            """Simulate a JWKS operation that can fail."""
            raise JWKSError("Connection timeout", "timeout_error")

        def simulate_auth_operation():
            """Simulate an auth operation that can fail."""
            try:
                simulate_jwks_operation()
            except JWKSError:
                raise AuthenticationError("Authentication service unavailable", 503) from None

        # Test the error propagation
        with pytest.raises(AuthenticationError) as exc_info:
            simulate_auth_operation()

        error = exc_info.value
        assert error.status_code == 503
        assert "unavailable" in str(error)

    def test_error_attributes_immutability(self):
        """Test that error attributes can be modified after creation."""
        jwks_error = JWKSError("Original message", "original_type")
        auth_error = AuthenticationError("Original message", 401)

        # Exception attributes are mutable in Python
        jwks_error.message = "Modified message"
        jwks_error.error_type = "modified_type"

        auth_error.message = "Modified message"
        auth_error.status_code = 500

        assert jwks_error.message == "Modified message"
        assert jwks_error.error_type == "modified_type"
        assert auth_error.message == "Modified message"
        assert auth_error.status_code == 500

    def test_exception_string_representations(self):
        """Test string representations of all exception types."""
        jwks_error = JWKSError("JWKS operation failed", "operation_error")
        auth_error = AuthenticationError("Authentication failed", 401)

        # Test str() representations
        assert str(jwks_error) == "JWKS operation failed"
        assert str(auth_error) == "Authentication failed"

        # Test repr() representations
        jwks_repr = repr(jwks_error)
        auth_repr = repr(auth_error)

        assert "JWKSError" in jwks_repr
        assert "JWKS operation failed" in jwks_repr

        assert "AuthenticationError" in auth_repr
        assert "Authentication failed" in auth_repr
