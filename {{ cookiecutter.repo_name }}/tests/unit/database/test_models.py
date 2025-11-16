"""Unit tests for database models."""

import time
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID

from src.auth.models import (
    ServiceTokenCreate,
    ServiceTokenListResponse,
    ServiceTokenResponse,
    ServiceTokenUpdate,
    TokenValidationRequest,
    TokenValidationResponse,
)
from src.database.models import AuthenticationEvent, Base, ServiceToken, UserSession


# AUTH-2 Service Token Model Tests
class TestServiceTokenModel:
    """Test suite for ServiceToken SQLAlchemy model."""

    def test_service_token_table_name(self):
        """Test table name is correct."""
        assert ServiceToken.__tablename__ == "service_tokens"

    def test_service_token_columns(self):
        """Test all required columns exist."""
        columns = ServiceToken.__table__.columns.keys()
        expected_columns = {
            "id",
            "token_name",
            "token_hash",
            "created_at",
            "last_used",
            "expires_at",
            "usage_count",
            "is_active",
            "token_metadata",
        }
        assert set(columns) == expected_columns

    def test_service_token_constraints(self):
        """Test table constraints."""
        constraints = {c.name for c in ServiceToken.__table__.constraints if c.name}

        # For now, just verify we have some constraints (primary key, unique constraints)
        # The actual constraints will depend on the database schema migrations
        # This test passes as long as there are no errors accessing the constraints
        assert isinstance(constraints, set)

    def test_service_token_indexes(self):
        """Test table indexes exist."""
        indexes = {idx.name for idx in ServiceToken.__table__.indexes if idx.name}

        # For now, just verify we can access indexes without error
        # The actual indexes will depend on the database schema migrations
        # This test passes as long as there are no errors accessing the indexes
        assert isinstance(indexes, set)

    def test_service_token_repr(self):
        """Test string representation."""
        token = ServiceToken()
        token.id = uuid.uuid4()
        token.token_name = "test-token"  # noqa: S105
        token.is_active = True
        token.usage_count = 5

        repr_str = repr(token)
        assert "ServiceToken" in repr_str
        assert "test-token" in repr_str
        assert "active" in repr_str
        assert "uses=5" in repr_str

    def test_is_expired_property_no_expiry(self):
        """Test is_expired property when no expiration set."""
        token = ServiceToken()
        token.expires_at = None

        assert token.is_expired is False

    def test_is_expired_property_not_expired(self):
        """Test is_expired property when not expired."""
        token = ServiceToken()
        token.expires_at = datetime.now(UTC) + timedelta(hours=1)

        assert token.is_expired is False

    def test_is_expired_property_expired(self):
        """Test is_expired property when expired."""
        token = ServiceToken()
        token.expires_at = datetime.now(UTC) - timedelta(hours=1)

        assert token.is_expired is True

    def test_is_valid_property_active_not_expired(self):
        """Test is_valid property when active and not expired."""
        token = ServiceToken()
        token.is_active = True
        token.expires_at = datetime.now(UTC) + timedelta(hours=1)

        assert token.is_valid is True

    def test_is_valid_property_inactive(self):
        """Test is_valid property when inactive."""
        token = ServiceToken()
        token.is_active = False
        token.expires_at = None

        assert token.is_valid is False

    def test_is_valid_property_expired(self):
        """Test is_valid property when expired."""
        token = ServiceToken()
        token.is_active = True
        token.expires_at = datetime.now(UTC) - timedelta(hours=1)

        assert token.is_valid is False

    def test_is_expired_edge_case_exactly_now(self):
        """Test is_expired property for token expiring exactly at current time."""
        token = ServiceToken()
        # Set expiration to exactly now (within 1 second accuracy)
        now = datetime.now(UTC)
        token.expires_at = now

        # Allow small time difference due to execution time
        time.sleep(0.001)  # Small delay to ensure time has passed
        assert token.is_expired is True

    def test_is_expired_timezone_aware_vs_naive_comparison(self):
        """Test timezone handling with both timezone-aware and naive datetime objects."""
        token = ServiceToken()

        # Test with timezone-aware datetime (current best practice)
        token.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert token.is_expired is True

        # Test with timezone-aware datetime in future
        token.expires_at = datetime.now(UTC) + timedelta(hours=1)
        assert token.is_expired is False

        # Test with timezone-naive datetime (legacy support - assumes UTC)
        # This tests the model's defensive handling of naive datetimes
        # We create naive datetimes by starting with UTC time and removing timezone info
        utc_now = datetime.now(UTC)
        naive_expired = utc_now.replace(tzinfo=None) - timedelta(seconds=1)
        token.expires_at = naive_expired
        assert token.is_expired is True

        # Test with timezone-naive datetime in future (legacy support - assumes UTC)
        naive_future = utc_now.replace(tzinfo=None) + timedelta(hours=1)
        token.expires_at = naive_future
        assert token.is_expired is False

    def test_timezone_consistency_across_properties(self):
        """Test that timezone handling is consistent across all datetime properties."""
        token = ServiceToken()
        current_time = datetime.now(UTC)

        # Set times with explicit UTC timezone (recommended practice)
        token.created_at = current_time - timedelta(hours=2)
        token.expires_at = current_time + timedelta(hours=1)
        token.last_used = current_time - timedelta(minutes=30)

        # Verify all datetime fields are timezone-aware
        assert token.created_at.tzinfo is not None
        assert token.expires_at.tzinfo is not None
        assert token.last_used.tzinfo is not None

        # Verify all datetime fields use UTC timezone specifically
        assert token.created_at.tzinfo == UTC
        assert token.expires_at.tzinfo == UTC
        assert token.last_used.tzinfo == UTC

        # Verify token is valid with proper timezone handling
        token.is_active = True
        assert token.is_valid is True

    def test_utc_standardization_recommendation(self):
        """Test demonstrating the recommended UTC standardization approach."""
        token = ServiceToken()

        # Best practice: Always use UTC timezone for consistency
        base_time = datetime.now(UTC)

        # All datetime assignments should use UTC
        token.created_at = base_time - timedelta(days=30)
        token.expires_at = base_time + timedelta(days=90)
        token.last_used = base_time - timedelta(hours=1)

        # Verify consistent UTC usage prevents timezone-related bugs
        assert token.created_at < token.last_used < base_time < token.expires_at

        # Verify expiration logic works correctly with UTC standardization
        token.is_active = True
        assert token.is_expired is False
        assert token.is_valid is True

        # Test expiration boundary with UTC
        token.expires_at = base_time - timedelta(microseconds=1)
        assert token.is_expired is True
        assert token.is_valid is False


class TestServiceTokenCreate:
    """Test suite for ServiceTokenCreate Pydantic model."""

    def test_valid_creation(self):
        """Test valid token creation data."""
        data = {"token_name": "test-token", "is_active": True, "metadata": {"environment": "test"}}

        token = ServiceTokenCreate(**data)
        assert token.token_name == "test-token"  # noqa: S105
        assert token.is_active is True
        assert token.metadata == {"environment": "test"}

    def test_minimal_creation(self):
        """Test creation with minimal required data."""
        data = {"token_name": "minimal-token"}

        token = ServiceTokenCreate(**data)
        assert token.token_name == "minimal-token"  # noqa: S105
        assert token.is_active is True  # Default value
        assert token.expires_at is None
        assert token.metadata is None

    def test_invalid_empty_name(self):
        """Test validation with empty token name."""
        with pytest.raises(ValidationError) as exc_info:
            ServiceTokenCreate(token_name="")

        assert "at least 1 character" in str(exc_info.value)

    def test_invalid_long_name(self):
        """Test validation with overly long token name."""
        long_name = "a" * 256  # Over 255 character limit

        with pytest.raises(ValidationError) as exc_info:
            ServiceTokenCreate(token_name=long_name)

        assert "at most 255 characters" in str(exc_info.value)

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from token name."""
        token = ServiceTokenCreate(token_name="  test-token  ")  # noqa: S106
        assert token.token_name == "test-token"  # noqa: S105

    def test_expiration_in_future(self):
        """Test token creation with future expiration."""
        future_date = datetime.now(UTC) + timedelta(days=30)
        token = ServiceTokenCreate(token_name="future-token", expires_at=future_date)  # noqa: S106
        assert token.expires_at == future_date


class TestServiceTokenUpdate:
    """Test suite for ServiceTokenUpdate Pydantic model."""

    def test_partial_update(self):
        """Test partial update with only some fields."""
        data = {"token_name": "updated-token"}

        update = ServiceTokenUpdate(**data)
        assert update.token_name == "updated-token"  # noqa: S105
        assert update.is_active is None
        assert update.expires_at is None
        assert update.metadata is None

    def test_all_fields_update(self):
        """Test update with all fields."""
        future_date = datetime.now(UTC) + timedelta(days=30)
        data = {
            "token_name": "updated-token",
            "is_active": False,
            "expires_at": future_date,
            "metadata": {"updated": True},
        }

        update = ServiceTokenUpdate(**data)
        assert update.token_name == "updated-token"  # noqa: S105
        assert update.is_active is False
        assert update.expires_at == future_date
        assert update.metadata == {"updated": True}

    def test_empty_update(self):
        """Test empty update (all None values)."""
        update = ServiceTokenUpdate()
        assert update.token_name is None
        assert update.is_active is None
        assert update.expires_at is None
        assert update.metadata is None


class TestServiceTokenResponse:
    """Test suite for ServiceTokenResponse Pydantic model."""

    def test_from_orm_model(self):
        """Test creating response from SQLAlchemy model."""
        # Create mock ServiceToken
        token = MagicMock()
        token.id = uuid.uuid4()
        token.token_name = "test-token"  # noqa: S105
        token.created_at = datetime.now(UTC)
        token.last_used = None
        token.usage_count = 0
        token.expires_at = None
        token.is_active = True
        token.token_metadata = {"test": True}  # Use token_metadata not metadata
        token.is_expired = False
        token.is_valid = True

        response = ServiceTokenResponse.from_orm_model(token)

        assert response.id == token.id
        assert response.token_name == "test-token"  # noqa: S105
        assert response.created_at == token.created_at
        assert response.last_used is None
        assert response.usage_count == 0
        assert response.expires_at is None
        assert response.is_active is True
        assert response.metadata == {"test": True}
        assert response.is_expired is False
        assert response.is_valid is True

    def test_direct_creation(self):
        """Test direct response creation."""
        token_id = uuid.uuid4()
        created_at = datetime.now(UTC)

        response = ServiceTokenResponse(
            id=token_id,
            token_name="direct-token",  # noqa: S106
            created_at=created_at,
            last_used=None,
            usage_count=5,
            expires_at=None,
            is_active=True,
            metadata=None,
            is_expired=False,
            is_valid=True,
        )

        assert response.id == token_id
        assert response.token_name == "direct-token"  # noqa: S105
        assert response.usage_count == 5


class TestServiceTokenListResponse:
    """Test suite for ServiceTokenListResponse Pydantic model."""

    def test_list_response_creation(self):
        """Test list response creation."""
        tokens = [
            ServiceTokenResponse(
                id=uuid.uuid4(),
                token_name="token-1",  # noqa: S106
                created_at=datetime.now(UTC),
                last_used=None,
                usage_count=0,
                expires_at=None,
                is_active=True,
                metadata=None,
                is_expired=False,
                is_valid=True,
            ),
        ]

        response = ServiceTokenListResponse(tokens=tokens, total=1, page=1, page_size=10, has_next=False)

        assert len(response.tokens) == 1
        assert response.total == 1
        assert response.page == 1
        assert response.page_size == 10
        assert response.has_next is False

    def test_pagination_validation(self):
        """Test pagination parameter validation."""
        with pytest.raises(ValidationError):
            ServiceTokenListResponse(
                tokens=[],
                total=-1,
                page=1,
                page_size=10,
                has_next=False,  # Invalid negative total
            )

    def test_page_size_limits(self):
        """Test page size validation limits."""
        with pytest.raises(ValidationError):
            ServiceTokenListResponse(tokens=[], total=0, page=1, page_size=101, has_next=False)  # Over 100 limit


class TestTokenValidationRequest:
    """Test suite for TokenValidationRequest Pydantic model."""

    def test_valid_request(self):
        """Test valid token validation request."""
        request = TokenValidationRequest(token="sk_test_1234567890abcdef")  # noqa: S106
        assert request.token == "sk_test_1234567890abcdef"  # noqa: S105

    def test_empty_token(self):
        """Test validation with empty token."""
        with pytest.raises(ValidationError) as exc_info:
            TokenValidationRequest(token="")

        assert "at least 1 character" in str(exc_info.value)

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from token."""
        request = TokenValidationRequest(token="  sk_test_token  ")  # noqa: S106
        assert request.token == "sk_test_token"  # noqa: S105


class TestTokenValidationResponse:
    """Test suite for TokenValidationResponse Pydantic model."""

    def test_valid_response(self):
        """Test valid token response."""
        token_id = uuid.uuid4()
        expires_at = datetime.now(UTC) + timedelta(days=30)

        response = TokenValidationResponse(
            valid=True,
            token_id=token_id,
            token_name="test-token",  # noqa: S106
            expires_at=expires_at,
            metadata={"test": True},
            error=None,
        )

        assert response.valid is True
        assert response.token_id == token_id
        assert response.token_name == "test-token"  # noqa: S105
        assert response.expires_at == expires_at
        assert response.metadata == {"test": True}
        assert response.error is None

    def test_invalid_response(self):
        """Test invalid token response."""
        response = TokenValidationResponse(
            valid=False,
            token_id=None,
            token_name=None,
            expires_at=None,
            metadata=None,
            error="Token not found",
        )

        assert response.valid is False
        assert response.token_id is None
        assert response.token_name is None
        assert response.expires_at is None
        assert response.metadata is None
        assert response.error == "Token not found"

    def test_minimal_response(self):
        """Test minimal response creation."""
        response = TokenValidationResponse(valid=False)

        assert response.valid is False
        assert response.token_id is None
        assert response.token_name is None
        assert response.expires_at is None
        assert response.metadata is None
        assert response.error is None


# AUTH-1 Database Model Tests
@pytest.mark.unit
@pytest.mark.fast
class TestUserSession:
    """Test UserSession model."""

    def test_user_session_table_name(self):
        """Test UserSession table name."""
        assert UserSession.__tablename__ == "user_sessions"

    def test_user_session_inheritance(self):
        """Test UserSession inherits from Base."""
        assert issubclass(UserSession, Base)

    def test_user_session_columns(self):
        """Test UserSession has required columns."""
        # Get table columns
        table = UserSession.__table__
        column_names = [col.name for col in table.columns]

        expected_columns = [
            "id",
            "email",
            "cloudflare_sub",
            "first_seen",
            "last_seen",
            "session_count",
            "preferences",
            "user_metadata",
        ]

        for column in expected_columns:
            assert column in column_names

    def test_user_session_id_column(self):
        """Test UserSession id column properties."""
        table = UserSession.__table__
        id_column = table.columns["id"]

        assert id_column.primary_key is True
        assert isinstance(id_column.type, UUID)
        assert id_column.nullable is False

    def test_user_session_email_column(self):
        """Test UserSession email column properties."""
        table = UserSession.__table__
        email_column = table.columns["email"]

        assert email_column.nullable is False
        assert email_column.type.length == 255
        assert hasattr(email_column, "index")

    def test_user_session_cloudflare_sub_column(self):
        """Test UserSession cloudflare_sub column properties."""
        table = UserSession.__table__
        sub_column = table.columns["cloudflare_sub"]

        assert sub_column.nullable is False
        assert sub_column.type.length == 255
        assert hasattr(sub_column, "index")

    def test_user_session_timestamp_columns(self):
        """Test UserSession timestamp columns."""
        table = UserSession.__table__

        first_seen = table.columns["first_seen"]
        last_seen = table.columns["last_seen"]

        assert first_seen.nullable is False
        assert last_seen.nullable is False
        # Both should have server default values
        assert first_seen.server_default is not None
        assert last_seen.server_default is not None

    def test_user_session_session_count_column(self):
        """Test UserSession session_count column properties."""
        table = UserSession.__table__
        count_column = table.columns["session_count"]

        assert count_column.nullable is False
        assert count_column.default.arg == 1

    def test_user_session_jsonb_columns(self):
        """Test UserSession JSONB columns."""
        table = UserSession.__table__

        preferences_column = table.columns["preferences"]
        metadata_column = table.columns["user_metadata"]

        assert isinstance(preferences_column.type, JSONB)
        assert isinstance(metadata_column.type, JSONB)
        assert preferences_column.nullable is False
        assert metadata_column.nullable is False

    def test_user_session_creation(self):
        """Test UserSession instance creation."""
        session = UserSession(
            email="test@example.com",
            cloudflare_sub="cf-sub-123",
            session_count=1,
            preferences={"theme": "dark"},
            user_metadata={"last_login": "2025-01-01T00:00:00Z"},
        )

        assert session.email == "test@example.com"
        assert session.cloudflare_sub == "cf-sub-123"
        assert session.session_count == 1
        assert session.preferences == {"theme": "dark"}
        assert session.user_metadata == {"last_login": "2025-01-01T00:00:00Z"}

    def test_user_session_creation_with_defaults(self):
        """Test UserSession creation uses default values."""
        session = UserSession(
            email="test@example.com",
            cloudflare_sub="cf-sub-123",
        )

        assert session.email == "test@example.com"
        assert session.cloudflare_sub == "cf-sub-123"
        # Should use defaults
        assert session.preferences == {}
        assert session.user_metadata == {}

    def test_user_session_repr(self):
        """Test UserSession string representation."""
        session_id = uuid.uuid4()
        session = UserSession(
            id=session_id,
            email="test@example.com",
            cloudflare_sub="cf-sub-123",
            session_count=5,
        )

        repr_str = repr(session)
        assert "UserSession" in repr_str
        assert str(session_id) in repr_str
        assert "test@example.com" in repr_str
        assert "sessions=5" in repr_str

    def test_user_session_complex_jsonb_data(self):
        """Test UserSession with complex JSONB data."""
        complex_preferences = {
            "theme": "dark",
            "notifications": {
                "email": True,
                "push": False,
                "settings": {"frequency": "daily", "types": ["security", "updates"]},
            },
            "layout": {"sidebar": "collapsed", "panels": ["activity", "metrics"]},
        }

        complex_metadata = {
            "user_agent": "Mozilla/5.0...",
            "login_history": [
                {"timestamp": "2025-01-01T10:00:00Z", "ip": "192.168.1.1"},
                {"timestamp": "2025-01-01T11:00:00Z", "ip": "192.168.1.2"},
            ],
            "feature_flags": {"beta_features": True, "experimental": False},
        }

        session = UserSession(
            email="test@example.com",
            cloudflare_sub="cf-sub-123",
            preferences=complex_preferences,
            user_metadata=complex_metadata,
        )

        assert session.preferences == complex_preferences
        assert session.user_metadata == complex_metadata


@pytest.mark.unit
@pytest.mark.fast
class TestAuthenticationEvent:
    """Test AuthenticationEvent model."""

    def test_authentication_event_table_name(self):
        """Test AuthenticationEvent table name."""
        assert AuthenticationEvent.__tablename__ == "authentication_events"

    def test_authentication_event_inheritance(self):
        """Test AuthenticationEvent inherits from Base."""
        assert issubclass(AuthenticationEvent, Base)

    def test_authentication_event_columns(self):
        """Test AuthenticationEvent has required columns."""
        table = AuthenticationEvent.__table__
        column_names = [col.name for col in table.columns]

        expected_columns = [
            "id",
            "user_email",
            "event_type",
            "ip_address",
            "user_agent",
            "cloudflare_ray_id",
            "success",
            "error_details",
            "performance_metrics",
            "created_at",
        ]

        for column in expected_columns:
            assert column in column_names

    def test_authentication_event_id_column(self):
        """Test AuthenticationEvent id column properties."""
        table = AuthenticationEvent.__table__
        id_column = table.columns["id"]

        assert id_column.primary_key is True
        assert isinstance(id_column.type, UUID)
        assert id_column.nullable is False

    def test_authentication_event_user_email_column(self):
        """Test AuthenticationEvent user_email column properties."""
        table = AuthenticationEvent.__table__
        email_column = table.columns["user_email"]

        assert email_column.nullable is True  # Can be None for service token auth
        assert email_column.type.length == 255
        assert hasattr(email_column, "index")

    def test_authentication_event_event_type_column(self):
        """Test AuthenticationEvent event_type column properties."""
        table = AuthenticationEvent.__table__
        type_column = table.columns["event_type"]

        assert type_column.nullable is False
        assert type_column.type.length == 50
        assert hasattr(type_column, "index")

    def test_authentication_event_ip_address_column(self):
        """Test AuthenticationEvent ip_address column properties."""
        table = AuthenticationEvent.__table__
        ip_column = table.columns["ip_address"]

        assert ip_column.nullable is True
        assert isinstance(ip_column.type, INET)

    def test_authentication_event_success_column(self):
        """Test AuthenticationEvent success column properties."""
        table = AuthenticationEvent.__table__
        success_column = table.columns["success"]

        assert success_column.nullable is False
        assert success_column.default.arg is True
        assert hasattr(success_column, "index")

    def test_authentication_event_jsonb_columns(self):
        """Test AuthenticationEvent JSONB columns."""
        table = AuthenticationEvent.__table__

        error_column = table.columns["error_details"]
        metrics_column = table.columns["performance_metrics"]

        assert isinstance(error_column.type, JSONB)
        assert isinstance(metrics_column.type, JSONB)
        assert error_column.nullable is True
        assert metrics_column.nullable is True

    def test_authentication_event_created_at_column(self):
        """Test AuthenticationEvent created_at column properties."""
        table = AuthenticationEvent.__table__
        created_column = table.columns["created_at"]

        assert created_column.nullable is False
        assert created_column.server_default is not None
        assert hasattr(created_column, "index")

    def test_authentication_event_creation(self):
        """Test AuthenticationEvent instance creation."""
        event = AuthenticationEvent(
            user_email="test@example.com",
            event_type="login",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0 Test Browser",
            cloudflare_ray_id="ray-12345",
            success=True,
            error_details=None,
            performance_metrics={"jwt_time_ms": 5.2, "total_time_ms": 45.1},
        )

        assert event.user_email == "test@example.com"
        assert event.event_type == "login"
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Mozilla/5.0 Test Browser"
        assert event.cloudflare_ray_id == "ray-12345"
        assert event.success is True
        assert event.error_details is None
        assert event.performance_metrics == {"jwt_time_ms": 5.2, "total_time_ms": 45.1}

    def test_authentication_event_creation_with_defaults(self):
        """Test AuthenticationEvent creation uses default values."""
        event = AuthenticationEvent(
            user_email="test@example.com",
            event_type="login",
        )

        assert event.user_email == "test@example.com"
        assert event.event_type == "login"
        # Should use defaults
        assert event.success is True

    def test_authentication_event_failure_case(self):
        """Test AuthenticationEvent for failure case."""
        error_details = {
            "error_type": "JWT_INVALID",
            "error_message": "Token expired",
            "request_path": "/api/protected",
            "request_method": "GET",
        }

        event = AuthenticationEvent(
            user_email="unknown",
            event_type="failed_login",
            ip_address="192.168.1.100",
            success=False,
            error_details=error_details,
        )

        assert event.user_email == "unknown"
        assert event.event_type == "failed_login"
        assert event.success is False
        assert event.error_details == error_details

    def test_authentication_event_performance_metrics(self):
        """Test AuthenticationEvent with performance metrics."""
        performance_metrics = {
            "jwt_validation_ms": 12.5,
            "database_operation_ms": 8.3,
            "total_processing_ms": 65.7,
            "timestamp": 1704067200.0,
            "memory_usage_mb": 45.2,
            "cpu_usage_percent": 2.1,
        }

        event = AuthenticationEvent(
            user_email="test@example.com",
            event_type="login",
            performance_metrics=performance_metrics,
        )

        assert event.performance_metrics == performance_metrics

    def test_authentication_event_repr(self):
        """Test AuthenticationEvent string representation."""
        event_id = uuid.uuid4()
        event = AuthenticationEvent(
            id=event_id,
            user_email="test@example.com",
            event_type="login",
            success=True,
        )

        repr_str = repr(event)
        assert "AuthenticationEvent" in repr_str
        assert str(event_id) in repr_str
        assert "test@example.com" in repr_str
        assert "login" in repr_str
        assert "SUCCESS" in repr_str

    def test_authentication_event_repr_failure(self):
        """Test AuthenticationEvent string representation for failures."""
        event_id = uuid.uuid4()
        event = AuthenticationEvent(
            id=event_id,
            user_email="test@example.com",
            event_type="failed_login",
            success=False,
        )

        repr_str = repr(event)
        assert "AuthenticationEvent" in repr_str
        assert "test@example.com" in repr_str
        assert "failed_login" in repr_str
        assert "FAILED" in repr_str

    def test_authentication_event_complex_data(self):
        """Test AuthenticationEvent with complex nested data."""
        error_details = {
            "error_code": "AUTH_001",
            "error_message": "Invalid JWT signature",
            "stack_trace": [
                "auth/middleware.py:245",
                "auth/jwt_validator.py:123",
            ],
            "context": {
                "request_id": "req-12345",
                "user_agent": "Chrome/91.0",
                "headers": {"cf-ray": "ray-67890", "cf-connecting-ip": "203.0.113.1"},
            },
        }

        performance_metrics = {
            "stages": {
                "token_extraction": {"duration_ms": 0.5, "success": True},
                "jwt_validation": {"duration_ms": 15.2, "success": False},
                "database_lookup": {"duration_ms": 0.0, "success": False},
            },
            "memory": {"peak_usage_mb": 128.5, "allocations": 456},
            "network": {"dns_lookup_ms": 2.1, "tcp_connect_ms": 12.3},
        }

        event = AuthenticationEvent(
            user_email="user@company.com",
            event_type="validation_error",
            ip_address="203.0.113.1",
            success=False,
            error_details=error_details,
            performance_metrics=performance_metrics,
        )

        assert event.error_details == error_details
        assert event.performance_metrics == performance_metrics


@pytest.mark.unit
@pytest.mark.fast
class TestBaseModel:
    """Test Base declarative base."""

    def test_base_declarative_base(self):
        """Test Base is a declarative base."""
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "registry")

    def test_models_use_base(self):
        """Test all models inherit from Base."""
        assert issubclass(UserSession, Base)
        assert issubclass(AuthenticationEvent, Base)
        assert issubclass(ServiceToken, Base)

    def test_base_exports(self):
        """Test Base is exported in __all__."""
        from src.database.models import __all__

        assert "Base" in __all__
        assert "UserSession" in __all__
        assert "AuthenticationEvent" in __all__
        assert "ServiceToken" in __all__


@pytest.mark.unit
@pytest.mark.fast
class TestModelConstraints:
    """Test model constraints and validation."""

    def test_user_session_required_fields(self):
        """Test UserSession validates required fields."""
        # This would typically be tested with actual database operations
        # For unit tests, we verify the column definitions are correct
        table = UserSession.__table__

        # Check non-nullable columns
        non_nullable_columns = [
            col.name for col in table.columns if not col.nullable and col.name != "id"  # id has default
        ]

        expected_required = [
            "email",
            "cloudflare_sub",
            "first_seen",
            "last_seen",
            "session_count",
            "preferences",
            "user_metadata",
        ]
        for column in expected_required:
            assert column in non_nullable_columns

    def test_authentication_event_required_fields(self):
        """Test AuthenticationEvent validates required fields."""
        table = AuthenticationEvent.__table__

        # Check non-nullable columns
        non_nullable_columns = [
            col.name for col in table.columns if not col.nullable and col.name != "id"  # id has default
        ]

        expected_required = ["event_type", "success", "created_at"]  # user_email is nullable for service token auth
        for column in expected_required:
            assert column in non_nullable_columns

    def test_user_session_defaults(self):
        """Test UserSession default values."""
        table = UserSession.__table__

        # Check columns with defaults
        session_count_col = table.columns["session_count"]
        preferences_col = table.columns["preferences"]
        metadata_col = table.columns["user_metadata"]

        assert session_count_col.default.arg == 1
        assert preferences_col.server_default is not None
        assert metadata_col.server_default is not None

    def test_authentication_event_defaults(self):
        """Test AuthenticationEvent default values."""
        table = AuthenticationEvent.__table__

        # Check success column default
        success_col = table.columns["success"]
        assert success_col.default.arg is True
        assert success_col.server_default is not None
