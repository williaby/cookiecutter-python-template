"""Shared fixtures for authentication tests."""

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import Mock

import pytest

from src.auth.jwks_client import JWKSClient
from src.auth.jwt_validator import JWTValidator


@pytest.fixture
def mock_jwks_client():
    """Create mock JWKS client with valid key data."""
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
def jwt_validator(mock_jwks_client):
    """Create JWT validator instance with mock client."""
    return JWTValidator(
        jwks_client=mock_jwks_client,
        audience="https://test-app.com",
        issuer="https://test.cloudflareaccess.com",
    )


@pytest.fixture
def valid_jwt_payload():
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


def create_jwt_token(payload: dict[str, Any], header: dict[str, Any] | None = None) -> str:
    """Create a properly formatted JWT token for testing.

    Args:
        payload: JWT payload claims
        header: JWT header (defaults to standard RS256 header)

    Returns:
        Formatted JWT token string
    """
    if header is None:
        header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-id"}

    # Encode header and payload
    header_encoded = (
        base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")).decode("utf-8").rstrip("=")
    )

    payload_encoded = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("utf-8").rstrip("=")
    )

    # Create fake signature
    signature = base64.urlsafe_b64encode(b"fake-signature").decode("utf-8").rstrip("=")

    return f"{header_encoded}.{payload_encoded}.{signature}"


@pytest.fixture
def create_test_jwt():
    """Factory fixture for creating test JWT tokens."""
    return create_jwt_token


@pytest.fixture
def valid_jwt_token(valid_jwt_payload, create_test_jwt):
    """Create a valid JWT token for testing."""
    return create_test_jwt(valid_jwt_payload)


@pytest.fixture
def jwt_token_without_kid(valid_jwt_payload, create_test_jwt):
    """Create JWT token without 'kid' in header."""
    header = {"alg": "RS256", "typ": "JWT"}  # Missing 'kid'
    return create_test_jwt(valid_jwt_payload, header)


@pytest.fixture
def jwt_token_missing_email(valid_jwt_payload, create_test_jwt):
    """Create JWT token without email claim."""
    payload = valid_jwt_payload.copy()
    del payload["email"]
    return create_test_jwt(payload)


@pytest.fixture
def jwt_token_empty_email(valid_jwt_payload, create_test_jwt):
    """Create JWT token with empty email claim."""
    payload = valid_jwt_payload.copy()
    payload["email"] = ""
    return create_test_jwt(payload)


@pytest.fixture
def jwt_token_none_email(valid_jwt_payload, create_test_jwt):
    """Create JWT token with None email claim."""
    payload = valid_jwt_payload.copy()
    payload["email"] = None
    return create_test_jwt(payload)
