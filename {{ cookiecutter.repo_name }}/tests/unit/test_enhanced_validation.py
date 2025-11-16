"""Tests for enhanced validation and error handling in configuration settings.

This module tests the Phase 4 enhancements including detailed validation,
custom error messages, startup validation, and comprehensive logging.
"""

import logging
from unittest.mock import patch

import pytest

from src.config.settings import (
    ApplicationSettings,
    ConfigurationValidationError,
    reload_settings,
    validate_configuration_on_startup,
    validate_field_requirements_by_environment,
)
from src.utils.setup_validator import (
    validate_system_requirements,
)


class TestEnhancedValidation:
    """Test enhanced validation with detailed error messages."""

    def test_port_validation_detailed_errors(self):
        """Test enhanced port validation with detailed error messages."""
        # Test port too low
        with pytest.raises(ValueError, match="Port 0 is outside valid range") as exc_info:
            ApplicationSettings(api_port=0)

        error_msg = str(exc_info.value)
        assert "Port 0 is outside valid range" in error_msg
        assert "Ports must be between 1-65535" in error_msg
        assert "Common choices: 8000" in error_msg

        # Test port too high
        with pytest.raises(ValueError, match="Port 70000 is outside valid range") as exc_info:
            ApplicationSettings(api_port=70000)

        error_msg = str(exc_info.value)
        assert "Port 70000 is outside valid range" in error_msg
        assert "8080 (alternative HTTP)" in error_msg

    def test_host_validation_detailed_errors(self):
        """Test enhanced host validation with detailed error messages."""
        # Test empty host
        with pytest.raises(ValueError, match="API host cannot be empty") as exc_info:
            ApplicationSettings(api_host="")

        error_msg = str(exc_info.value)
        assert "API host cannot be empty" in error_msg
        assert "Common values: '0.0.0.0'" in error_msg
        assert "'localhost'" in error_msg

        # Test invalid host format
        with pytest.raises(ValueError, match="Invalid API host format") as exc_info:
            ApplicationSettings(api_host="invalid@host!")

        error_msg = str(exc_info.value)
        assert "Invalid API host format" in error_msg
        assert "valid IP address" in error_msg
        assert "hostname" in error_msg

    def test_version_validation_semver(self):
        """Test semantic version validation."""
        # Valid versions should pass
        valid_versions = [
            "1.0.0",
            "0.1.0",
            "2.3.4",
            "1.0",
            "1.0.0-alpha",
            "1.0.0+build.1",
        ]
        for version in valid_versions:
            settings = ApplicationSettings(version=version)
            assert settings.version == version

        # Invalid versions should fail with helpful messages
        with pytest.raises(ValueError, match="Invalid version format") as exc_info:
            ApplicationSettings(version="invalid-version")

        error_msg = str(exc_info.value)
        assert "Invalid version format" in error_msg
        assert "semantic versioning" in error_msg
        assert "MAJOR.MINOR.PATCH" in error_msg

    def test_app_name_validation_enhanced(self):
        """Test enhanced application name validation."""
        # Test empty name
        with pytest.raises(ValueError, match="Application name cannot be empty") as exc_info:
            ApplicationSettings(app_name="")

        error_msg = str(exc_info.value)
        assert "Application name cannot be empty" in error_msg
        assert "descriptive name" in error_msg

        # Test name too long
        long_name = "x" * 101
        with pytest.raises(ValueError, match="too long") as exc_info:
            ApplicationSettings(app_name=long_name)

        error_msg = str(exc_info.value)
        assert "too long (101 characters)" in error_msg
        assert "Maximum length is 100" in error_msg

        # Test invalid characters
        with pytest.raises(ValueError, match="Invalid application name") as exc_info:
            ApplicationSettings(app_name="app@name!")

        error_msg = str(exc_info.value)
        assert "Invalid application name" in error_msg
        assert "letters, numbers, spaces, hyphens" in error_msg

    def test_secret_field_validation_detailed(self):
        """Test enhanced secret field validation with field-specific guidance."""
        # Test database password
        with pytest.raises(ValueError, match="Secret values cannot be empty") as exc_info:
            ApplicationSettings(database_password="")

        error_msg = str(exc_info.value)
        assert "database_password" in error_msg
        assert "Secret values cannot be empty strings" in error_msg

        # Test API key
        with pytest.raises(ValueError, match="Secret values cannot be empty") as exc_info:
            ApplicationSettings(api_key="")

        error_msg = str(exc_info.value)
        assert "api_key" in error_msg
        assert "Secret values cannot be empty strings" in error_msg

        # Test secret key validation
        with pytest.raises(ValueError, match="Secret values cannot be empty") as exc_info:
            ApplicationSettings(secret_key="")

        error_msg = str(exc_info.value)
        assert "secret_key" in error_msg
        assert "Secret values cannot be empty strings" in error_msg

    def test_environment_specific_validation(self):
        """Test environment-specific validation requirements."""
        # Production requirements
        prod_required = validate_field_requirements_by_environment("prod")
        assert "secret_key" in prod_required
        assert "jwt_secret_key" in prod_required

        # Staging requirements
        staging_required = validate_field_requirements_by_environment("staging")
        assert "secret_key" in staging_required
        assert "jwt_secret_key" not in staging_required

        # Development requirements
        dev_required = validate_field_requirements_by_environment("dev")
        assert "secret_key" not in dev_required
        assert "jwt_secret_key" not in dev_required


class TestConfigurationValidationError:
    """Test the custom ConfigurationValidationError class."""

    def test_basic_error_creation(self):
        """Test basic error creation and formatting."""
        error = ConfigurationValidationError("Main error message")
        assert "Main error message" in str(error)

    def test_error_with_field_errors(self):
        """Test error with field-specific errors."""
        field_errors = ["Port must be > 0", "Host cannot be empty"]
        error = ConfigurationValidationError("Config failed", field_errors=field_errors)

        error_str = str(error)
        assert "Config failed" in error_str
        assert "Field Validation Errors:" in error_str
        assert "Port must be > 0" in error_str
        assert "Host cannot be empty" in error_str

    def test_error_with_suggestions(self):
        """Test error with suggestions."""
        suggestions = ["Set PROMPTCRAFT_PORT=8000", "Use valid hostname"]
        error = ConfigurationValidationError("Config failed", suggestions=suggestions)

        error_str = str(error)
        assert "Suggestions:" in error_str
        assert "Set PROMPTCRAFT_PORT=8000" in error_str
        assert "Use valid hostname" in error_str

    def test_error_with_all_components(self):
        """Test error with all components."""
        field_errors = ["Invalid port"]
        suggestions = ["Use port 8000"]
        error = ConfigurationValidationError(
            "Main error",
            field_errors=field_errors,
            suggestions=suggestions,
        )

        error_str = str(error)
        assert "Main error" in error_str
        assert "Field Validation Errors:" in error_str
        assert "Invalid port" in error_str
        assert "Suggestions:" in error_str
        assert "Use port 8000" in error_str


class TestStartupValidation:
    """Test startup validation functionality."""

    @patch("src.config.settings.validate_environment_keys")
    @patch("src.config.settings._detect_environment")
    def test_production_validation_errors(self, mock_detect_env, mock_validate_keys):
        """Test that production validation catches configuration issues."""
        mock_detect_env.return_value = "prod"
        mock_validate_keys.return_value = None

        # Create production settings with issues
        settings = ApplicationSettings(
            environment="prod",
            debug=True,  # Should be False in prod
            api_host="localhost",  # Should not be localhost in prod
        )

        with pytest.raises(ConfigurationValidationError) as exc_info:
            validate_configuration_on_startup(settings)

        error = exc_info.value
        assert "prod environment" in str(error).lower()
        assert "debug mode should be disabled" in str(error).lower()

    @patch("src.config.settings.validate_environment_keys")
    def test_staging_validation(self, mock_validate_keys):
        """Test staging environment validation."""
        mock_validate_keys.return_value = None

        settings = ApplicationSettings(
            environment="staging",
            secret_key="test-secret-key",  # noqa: S106
        )

        # Should not raise an error
        validate_configuration_on_startup(settings)

    @patch("src.config.settings.validate_environment_keys")
    def test_development_validation_passes(self, mock_validate_keys):
        """Test that development validation is more lenient."""
        mock_validate_keys.return_value = None

        settings = ApplicationSettings(
            environment="dev",
            debug=True,
            api_host="localhost",
        )

        # Should not raise an error
        validate_configuration_on_startup(settings)

    def test_system_requirements_validation(self):
        """Test system requirements validation."""
        success, errors = validate_system_requirements()

        # Should pass in normal test environment
        if not success:
            # Log errors for debugging
            for error in errors:
                print(f"System validation error: {error}")

        # At minimum, Python version should be OK since tests are running
        assert success or any("Python" in error for error in errors)


class TestLoggingIntegration:
    """Test logging integration in configuration system."""

    @patch("src.config.settings.validate_environment_keys")
    @patch("src.config.settings._detect_environment")
    def test_configuration_logging(self, mock_detect_env, mock_validate_keys, caplog):
        """Test that configuration loading is properly logged."""
        mock_detect_env.return_value = "dev"
        mock_validate_keys.return_value = None

        with caplog.at_level(logging.INFO):
            _ = reload_settings()

        # Check that important information is logged
        log_messages = [record.message for record in caplog.records]

        # Should log environment detection
        assert any("environment" in msg.lower() for msg in log_messages)

        # Should log configuration status
        assert any("configuration" in msg.lower() for msg in log_messages)

    @patch("src.config.settings.validate_environment_keys")
    def test_secret_masking_in_logs(self, mock_validate_keys, caplog):
        """Test that secret values are properly masked in logs."""
        mock_validate_keys.return_value = None

        with caplog.at_level(logging.INFO):
            settings = ApplicationSettings(
                api_key="secret-api-key-12345",
                secret_key="super-secret-key",  # noqa: S106
            )
            validate_configuration_on_startup(settings)

        # Check that secret values are not exposed in logs
        log_text = " ".join(record.message for record in caplog.records)
        assert "secret-api-key-12345" not in log_text
        assert "super-secret-key" not in log_text

        # But should mention that secrets are configured
        assert any("secret" in record.message.lower() for record in caplog.records)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_ip_address_validation(self):
        """Test IP address validation edge cases."""
        # Valid IP addresses
        valid_ips = ["192.168.1.1", "10.0.0.1", "172.16.0.1", "255.255.255.255"]
        for ip in valid_ips:
            settings = ApplicationSettings(api_host=ip)
            assert settings.api_host == ip

        # Invalid formats that should fail (completely invalid hostnames/IPs)
        invalid_formats = ["", "   ", "...", "256.256.256.256.", "-invalid-", "host..name"]
        for invalid_format in invalid_formats:
            with pytest.raises(ValueError, match="Invalid|cannot be empty"):
                ApplicationSettings(api_host=invalid_format)

        # Note: "192.168.01.1" may be valid depending on regex interpretation
        # Testing only clearly invalid cases

    def test_hostname_validation(self):
        """Test hostname validation edge cases."""
        # Valid hostnames
        valid_hostnames = ["example.com", "api.example.com", "test-server", "server1"]
        for hostname in valid_hostnames:
            settings = ApplicationSettings(api_host=hostname)
            assert settings.api_host == hostname

        # Invalid hostnames (too long subdomain)
        with pytest.raises(ValueError, match="Invalid"):
            ApplicationSettings(api_host="a" * 64 + ".example.com")

    @patch("src.config.settings.validate_environment_keys")
    def test_privileged_port_warning(self, mock_validate_keys, caplog):
        """Test warning for privileged ports."""
        mock_validate_keys.return_value = None

        with caplog.at_level(logging.WARNING):
            ApplicationSettings(api_port=80)

        # Should warn about privileged port
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("privileged port" in msg.lower() for msg in warning_messages)

    def test_cross_field_validation_warning(self, caplog):
        """Test cross-field validation warnings."""
        with caplog.at_level(logging.WARNING):
            ApplicationSettings(
                database_url="postgresql://user:pass@host/db",
                database_password="separate-password",  # noqa: S106
            )
            validate_configuration_on_startup(
                ApplicationSettings(
                    database_url="postgresql://user:pass@host/db",
                    database_password="separate-password",  # noqa: S106
                ),
            )

        # Should warn about both database_url and database_password being set
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("database_url and database_password" in msg for msg in warning_messages)
