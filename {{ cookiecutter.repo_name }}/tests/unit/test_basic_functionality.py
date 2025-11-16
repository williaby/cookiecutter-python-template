"""Basic functionality tests to improve coverage."""

from src.config.health import ConfigurationStatusModel, get_configuration_health_summary, get_configuration_status
from src.config.settings import ApplicationSettings, get_settings
from src.config.validation import validate_configuration_on_startup
from src.security import audit_logging, error_handlers, input_validation, middleware, rate_limiting
from src.security.audit_logging import AuditEventSeverity, AuditEventType
from src.security.input_validation import SecureEmailField, SecureStringField
from src.security.rate_limiting import RateLimits


class TestBasicConfiguration:
    """Test basic configuration functionality."""

    def test_get_settings(self):
        """Test basic settings retrieval."""
        settings = get_settings()
        assert isinstance(settings, ApplicationSettings)
        assert hasattr(settings, "environment")

    def test_configuration_status(self):
        """Test configuration status check."""
        settings = get_settings()
        status = get_configuration_status(settings)
        assert isinstance(status, ConfigurationStatusModel)
        assert hasattr(status, "config_healthy")

    def test_configuration_health_summary(self):
        """Test configuration health summary."""
        summary = get_configuration_health_summary()
        assert isinstance(summary, dict)
        assert "healthy" in summary

    def test_validate_configuration_on_startup(self):
        """Test configuration validation."""
        settings = get_settings()
        # This should not raise an exception in normal circumstances
        validate_configuration_on_startup(settings)


class TestSecurityModules:
    """Test security module basic functionality."""

    def test_security_imports(self):
        """Test that security modules can be imported."""
        assert audit_logging is not None
        assert error_handlers is not None
        assert input_validation is not None
        assert middleware is not None
        assert rate_limiting is not None

    def test_audit_event_types(self):
        """Test audit event types are defined."""
        # Test that enum values exist
        assert hasattr(AuditEventType, "AUTH_LOGIN_SUCCESS")
        assert hasattr(AuditEventSeverity, "LOW")

    def test_rate_limits_constants(self):
        """Test rate limit constants."""
        assert hasattr(RateLimits, "API_DEFAULT")
        assert hasattr(RateLimits, "HEALTH_CHECK")

    def test_input_validation_fields(self):
        """Test input validation field types."""
        # Test valid inputs
        result = SecureStringField.validate("Hello World")
        assert result == "Hello World"

        email_result = SecureEmailField.validate("test@example.com")
        assert email_result == "test@example.com"
