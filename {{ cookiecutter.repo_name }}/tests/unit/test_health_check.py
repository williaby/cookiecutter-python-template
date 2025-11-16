"""Tests for health check functionality and configuration status monitoring.

This module tests Phase 5 of the Core Configuration System: Health Check Integration.
It verifies that health check endpoints expose only safe operational information
and never leak sensitive configuration data.
"""

import json
import os
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.config.health import (
    ConfigurationStatusModel,
    _count_configured_secrets,
    _determine_config_source,
    _sanitize_validation_errors,
    get_configuration_health_summary,
    get_configuration_status,
)
from src.config.settings import ApplicationSettings, ConfigurationValidationError
from src.main import app


class TestConfigurationStatusModel:
    """Test the ConfigurationStatusModel Pydantic model."""

    def test_model_creation_with_valid_data(self) -> None:
        """Test creating model with valid configuration data."""
        status = ConfigurationStatusModel(
            environment="dev",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=False,
            config_source="defaults",
            validation_status="passed",
            validation_errors=[],
            secrets_configured=2,
            api_host="localhost",
            api_port=8000,
        )

        assert status.environment == "dev"
        assert status.version == "1.0.0"
        assert status.debug is True
        assert status.config_loaded is True
        assert status.encryption_enabled is False
        assert status.config_source == "defaults"
        assert status.validation_status == "passed"
        assert status.validation_errors == []
        assert status.secrets_configured == 2
        assert status.api_host == "localhost"
        assert status.api_port == 8000
        assert isinstance(status.timestamp, datetime)

    def test_config_healthy_property_when_healthy(self) -> None:
        """Test config_healthy computed property when configuration is healthy."""
        status = ConfigurationStatusModel(
            environment="prod",
            version="1.0.0",
            debug=False,
            config_loaded=True,
            encryption_enabled=True,
            config_source="env_vars",
            validation_status="passed",
            validation_errors=[],
            secrets_configured=5,
            api_host="0.0.0.0",  # noqa: S104
            api_port=80,
        )

        assert status.config_healthy is True

    def test_config_healthy_property_with_warnings(self) -> None:
        """Test config_healthy computed property with validation warnings."""
        status = ConfigurationStatusModel(
            environment="staging",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=True,
            config_source="env_files",
            validation_status="warning",
            validation_errors=["Non-critical warning"],
            secrets_configured=3,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is True  # Warnings still count as healthy

    def test_config_healthy_property_when_unhealthy(self) -> None:
        """Test config_healthy computed property when configuration is unhealthy."""
        status = ConfigurationStatusModel(
            environment="prod",
            version="1.0.0",
            debug=True,  # This would be a validation error in prod
            config_loaded=True,
            encryption_enabled=False,
            config_source="defaults",
            validation_status="failed",
            validation_errors=["Debug mode enabled in production"],
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is False

    def test_config_healthy_when_not_loaded(self) -> None:
        """Test config_healthy when configuration failed to load."""
        status = ConfigurationStatusModel(
            environment="dev",
            version="unknown",
            debug=False,
            config_loaded=False,
            encryption_enabled=False,
            config_source="defaults",
            validation_status="failed",
            validation_errors=["Configuration loading failed"],
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is False

    def test_json_serialization(self) -> None:
        """Test that the model can be serialized to JSON properly."""
        status = ConfigurationStatusModel(
            environment="dev",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=False,
            config_source="defaults",
            validation_status="passed",
            validation_errors=[],
            secrets_configured=2,
            api_host="localhost",
            api_port=8000,
        )

        # Should not raise exception
        json_data = status.model_dump_json()
        parsed = json.loads(json_data)

        assert parsed["environment"] == "dev"
        assert parsed["config_healthy"] is True
        assert "timestamp" in parsed
        assert parsed["timestamp"].endswith("Z")  # ISO format with Z suffix


class TestConfigurationStatusHelpers:
    """Test helper functions for configuration status."""

    def test_count_configured_secrets_with_all_secrets(self) -> None:
        """Test counting secrets when all secret fields are configured."""
        settings = ApplicationSettings(
            database_password=SecretStr("secret_db_pass"),
            database_url=SecretStr("postgresql://user:pass@host:5432/db"),
            api_key=SecretStr("api_key_value"),
            secret_key=SecretStr("app_secret_key"),
            azure_openai_api_key=SecretStr("azure_key"),
            jwt_secret_key=SecretStr("jwt_secret"),
            qdrant_api_key=SecretStr("qdrant_key"),
            encryption_key=SecretStr("encryption_key"),
        )

        count = _count_configured_secrets(settings)
        assert count == 8

    def test_count_configured_secrets_with_some_secrets(self) -> None:
        """Test counting secrets when only some secret fields are configured."""
        # Clear environment to avoid picking up unexpected values
        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings(
                api_key=SecretStr("api_key_value"),
                secret_key=SecretStr("app_secret_key"),
                jwt_secret_key=SecretStr("jwt_secret"),
                # Other secrets left as None
            )

        count = _count_configured_secrets(settings)
        assert count == 3

    def test_count_configured_secrets_with_empty_strings(self) -> None:
        """Test counting secrets ignores empty string values."""
        # Clear environment to avoid picking up unexpected values
        with patch.dict(os.environ, {}, clear=True):
            # Create settings first with valid values
            settings = ApplicationSettings(
                api_key=SecretStr("valid_key"),
            )

            # Manually set empty values to bypass validation (for testing purposes)
            settings.secret_key = SecretStr("")
            settings.jwt_secret_key = SecretStr("   ")

            count = _count_configured_secrets(settings)
            assert count == 1  # Only api_key counts

    def test_count_configured_secrets_with_no_secrets(self) -> None:
        """Test counting secrets when no secret fields are configured."""
        # Clear environment to avoid picking up unexpected values
        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

            count = _count_configured_secrets(settings)
            assert count == 0

    @patch.dict(
        os.environ,
        {
            "PROMPTCRAFT_ENVIRONMENT": "prod",
            "PROMPTCRAFT_API_HOST": "api.example.com",
            "PROMPTCRAFT_API_PORT": "443",
            "PROMPTCRAFT_DEBUG": "false",
        },
    )
    def test_determine_config_source_env_vars(self) -> None:
        """Test determining config source when environment variables are set."""
        settings = ApplicationSettings()
        source = _determine_config_source(settings)
        assert source == "env_vars"

    @patch.dict(os.environ, {}, clear=True)
    def test_determine_config_source_files(self) -> None:
        """Test determining config source when .env files exist."""
        settings = ApplicationSettings(environment="staging")

        # Mock file existence
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            source = _determine_config_source(settings)
            assert source == "env_files"

    @patch.dict(os.environ, {}, clear=True)
    def test_determine_config_source_defaults(self) -> None:
        """Test determining config source when using defaults."""
        settings = ApplicationSettings()

        # Mock no files exist
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            source = _determine_config_source(settings)
            assert source == "defaults"

    def test_sanitize_validation_errors_removes_sensitive_info(self) -> None:
        """Test that validation error sanitization removes sensitive information."""
        errors = [
            "Password field is required but missing",
            "Secret key must be at least 32 characters",
            "API key validation failed for external service",
            "Configuration file not found at /home/user/.env.prod",
            "Invalid database URL format",
            "General validation error without secrets",
        ]

        sanitized = _sanitize_validation_errors(errors)

        assert "Password configuration issue (details hidden)" in sanitized
        assert "Secret key configuration issue (details hidden)" in sanitized
        assert "API key configuration issue (details hidden)" in sanitized
        assert "Configuration file path issue (path hidden)" in sanitized
        assert "Invalid database URL format" in sanitized  # No secrets, preserved
        assert "General validation error without secrets" in sanitized  # No secrets, preserved

    def test_sanitize_validation_errors_with_empty_list(self) -> None:
        """Test sanitizing empty error list."""
        sanitized = _sanitize_validation_errors([])
        assert sanitized == []


class TestConfigurationStatusGeneration:
    """Test the get_configuration_status function."""

    @patch("src.config.health.validate_encryption_available")
    @patch("src.config.settings.validate_configuration_on_startup")
    def test_get_configuration_status_healthy(
        self,
        mock_validate_startup,
        mock_validate_encryption,
    ) -> None:
        """Test generating configuration status for healthy configuration."""
        mock_validate_encryption.return_value = True
        mock_validate_startup.return_value = None  # No exception = validation passed

        # Clear environment to avoid picking up unexpected values
        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings(
                app_name="Test App",
                version="1.2.3",
                environment="staging",
                debug=False,
                api_host="api.staging.com",
                api_port=8080,
                api_key=SecretStr("test_api_key"),
                secret_key=SecretStr("test_secret_key"),
            )

            status = get_configuration_status(settings)

            assert status.environment == "staging"
            assert status.version == "1.2.3"
            assert status.debug is False
            assert status.config_loaded is True
            assert status.encryption_enabled is True
            assert status.validation_status == "passed"
            assert status.validation_errors == []
            assert status.secrets_configured == 2  # api_key and secret_key
            assert status.api_host == "api.staging.com"
            assert status.api_port == 8080
            assert status.config_healthy is True

    @patch("src.config.health.validate_encryption_available")
    @patch("src.config.settings.validate_configuration_on_startup")
    def test_get_configuration_status_validation_failed(
        self,
        mock_validate_startup,
        mock_validate_encryption,
    ) -> None:
        """Test generating configuration status when validation fails."""
        mock_validate_encryption.return_value = False
        mock_validate_startup.side_effect = ConfigurationValidationError(
            "Validation failed",
            field_errors=[
                "Debug mode enabled in production",
                "Missing required secret key",
            ],
            suggestions=[
                "Set DEBUG=false",
                "Configure SECRET_KEY environment variable",
            ],
        )

        settings = ApplicationSettings(
            environment="prod",
            debug=True,  # Would cause validation error
        )

        status = get_configuration_status(settings)

        assert status.environment == "prod"
        assert status.config_loaded is True
        assert status.encryption_enabled is False
        assert status.validation_status == "failed"
        assert len(status.validation_errors) == 2
        assert status.config_healthy is False

    @patch("src.config.health.validate_encryption_available")
    @patch("src.config.health.validate_configuration_on_startup")
    def test_get_configuration_status_unexpected_error(
        self,
        mock_validate_startup,
        mock_validate_encryption,
    ) -> None:
        """Test generating configuration status when unexpected error occurs."""
        mock_validate_encryption.return_value = True
        mock_validate_startup.side_effect = RuntimeError("Unexpected error")

        settings = ApplicationSettings()

        # Now the function should raise the exception instead of handling it
        with pytest.raises(RuntimeError):
            get_configuration_status(settings)


class TestConfigurationHealthSummary:
    """Test the get_configuration_health_summary function."""

    @patch("src.config.settings.get_settings")
    @patch("src.config.health.get_configuration_status")
    def test_get_health_summary_success(
        self,
        mock_get_status,
        mock_get_settings,
    ) -> None:
        """Test getting health summary when everything works."""
        mock_settings = ApplicationSettings(environment="dev", version="1.0.0")
        mock_get_settings.return_value = mock_settings

        mock_status = ConfigurationStatusModel(
            environment="dev",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=True,
            config_source="defaults",
            validation_status="passed",
            validation_errors=[],
            secrets_configured=2,
            api_host="localhost",
            api_port=8000,
        )
        mock_get_status.return_value = mock_status

        summary = get_configuration_health_summary()

        assert summary["healthy"] is True
        assert summary["environment"] == "dev"
        assert summary["version"] == "1.0.0"
        assert summary["config_loaded"] is True
        assert summary["encryption_available"] is True
        assert "timestamp" in summary

    @patch("src.config.health.get_settings")
    def test_get_health_summary_failure(self, mock_get_settings) -> None:
        """Test getting health summary when configuration fails."""
        mock_get_settings.side_effect = ConfigurationValidationError("Config failed")

        summary = get_configuration_health_summary()

        assert summary["healthy"] is False
        assert summary["error"] == "Configuration health check failed"
        assert "timestamp" in summary


class TestHealthCheckEndpoints:
    """Test the FastAPI health check endpoints."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = TestClient(app)

    @patch("src.main.get_configuration_health_summary")
    def test_health_endpoint_healthy(self, mock_health_summary) -> None:
        """Test /health endpoint when application is healthy."""
        mock_health_summary.return_value = {
            "healthy": True,
            "environment": "dev",
            "version": "1.0.0",
            "config_loaded": True,
            "encryption_available": False,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "promptcraft-hybrid"
        assert data["healthy"] is True
        assert data["environment"] == "dev"

    @patch("src.main.get_configuration_health_summary")
    def test_health_endpoint_unhealthy(self, mock_health_summary) -> None:
        """Test /health endpoint when application is unhealthy."""
        health_summary = {
            "healthy": False,
            "error": "Configuration validation failed",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        mock_health_summary.return_value = health_summary

        response = self.client.get("/health")

        assert response.status_code == 503
        data = response.json()
        # AuthExceptionHandler.handle_service_unavailable returns detailed error message
        expected_error = f"Health check failed: configuration unhealthy - {health_summary}"
        assert data["error"] == expected_error
        assert data["status_code"] == 503

    @patch("src.main.get_configuration_health_summary")
    def test_health_endpoint_exception(self, mock_health_summary) -> None:
        """Test /health endpoint when health check raises exception."""
        mock_health_summary.side_effect = RuntimeError("Health check failed")

        response = self.client.get("/health")

        assert response.status_code == 500
        data = response.json()
        # AuthExceptionHandler.handle_internal_error with expose_error=True returns detailed error
        expected_error = "Health check endpoint failed: Health check failed"
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    @patch("src.main.get_settings")
    @patch("src.main.get_configuration_status")
    def test_config_health_endpoint_success(
        self,
        mock_get_status,
        mock_get_settings,
    ) -> None:
        """Test /health/config endpoint returns detailed configuration status."""
        mock_settings = ApplicationSettings()
        mock_get_settings.return_value = mock_settings

        mock_status = ConfigurationStatusModel(
            environment="dev",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=False,
            config_source="defaults",
            validation_status="passed",
            validation_errors=[],
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )
        mock_get_status.return_value = mock_status

        response = self.client.get("/health/config")

        assert response.status_code == 200
        data = response.json()
        assert data["environment"] == "dev"
        assert data["version"] == "1.0.0"
        assert data["config_loaded"] is True
        assert data["validation_status"] == "passed"
        assert data["config_healthy"] is True

    @patch("src.main.get_settings")
    def test_config_health_endpoint_validation_error(self, mock_get_settings) -> None:
        """Test /health/config endpoint when configuration validation fails."""
        # First call in try block raises ConfigurationValidationError
        # Second call in except block for debug check should succeed with debug=False
        mock_get_settings.side_effect = [
            ConfigurationValidationError(
                "Validation failed",
                field_errors=[
                    "Error 1",
                    "Error 2",
                    "Error 3",
                    "Error 4",
                    "Error 5",
                    "Error 6",
                ],
                suggestions=[
                    "Suggestion 1",
                    "Suggestion 2",
                    "Suggestion 3",
                    "Suggestion 4",
                ],
            ),
            ApplicationSettings(debug=False),  # Second call succeeds with debug=False
        ]

        response = self.client.get("/health/config")

        assert response.status_code == 500
        data = response.json()
        # In production mode (debug=False), the error message should contain minimal detail
        expected_detail = {
            "error": "Configuration validation failed",
            "details": "Contact system administrator",
        }
        expected_error = str(expected_detail)
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    @patch("src.main.get_settings")
    def test_config_health_endpoint_validation_error_debug_mode(self, mock_get_settings) -> None:
        """Test /health/config endpoint when configuration validation fails in debug mode."""
        # First call in try block raises ConfigurationValidationError
        # Second call in except block for debug check should succeed with debug=True
        field_errors = [
            "Error 1",
            "Error 2",
            "Error 3",
            "Error 4",
            "Error 5",
            "Error 6",
        ]
        suggestions = [
            "Suggestion 1",
            "Suggestion 2",
            "Suggestion 3",
            "Suggestion 4",
        ]
        validation_error = ConfigurationValidationError(
            "Validation failed",
            field_errors=field_errors,
            suggestions=suggestions,
        )
        mock_get_settings.side_effect = [
            validation_error,
            ApplicationSettings(debug=True),  # Second call succeeds with debug=True
        ]

        response = self.client.get("/health/config")

        assert response.status_code == 500
        data = response.json()
        # In debug mode (debug=True), the error message should contain detailed errors with limits applied
        # HEALTH_CHECK_ERROR_LIMIT=5, HEALTH_CHECK_SUGGESTION_LIMIT=3
        expected_detail = {
            "error": "Configuration validation failed",
            "field_errors": field_errors[:5],  # First 5 errors (HEALTH_CHECK_ERROR_LIMIT)
            "suggestions": suggestions[:3],    # First 3 suggestions (HEALTH_CHECK_SUGGESTION_LIMIT)
        }
        # With expose_error=True in debug mode, the error message format is: "detail: exception_message"
        # The exception_message is the full string representation of ConfigurationValidationError
        full_exception_str = str(validation_error)
        expected_error = f"{expected_detail}: {full_exception_str}"
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    def test_root_endpoint(self) -> None:
        """Test root endpoint returns application information."""
        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_ping_endpoint(self) -> None:
        """Test ping endpoint for load balancer checks."""
        response = self.client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "pong"


class TestSecurityRequirements:
    """Test that health checks never expose sensitive information."""

    def test_configuration_status_model_no_secret_fields(self) -> None:
        """Test that ConfigurationStatusModel doesn't include any secret fields."""
        status = ConfigurationStatusModel(
            environment="prod",
            version="1.0.0",
            debug=False,
            config_loaded=True,
            encryption_enabled=True,
            config_source="env_vars",
            validation_status="passed",
            validation_errors=[],
            secrets_configured=5,
            api_host="0.0.0.0",  # noqa: S104
            api_port=443,
        )

        # Get all field values as dict
        status_dict = status.model_dump()

        # Verify no secret-like fields are present
        forbidden_fields = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "database_password",
            "database_url",
            "api_key",
            "secret_key",
            "azure_openai_api_key",
            "jwt_secret_key",
            "qdrant_api_key",
            "encryption_key",
        ]

        for field_name in status_dict:
            for forbidden in forbidden_fields:
                if forbidden in field_name.lower():
                    # Only allowed field is secrets_configured (count, not value)
                    assert field_name == "secrets_configured"

    def test_health_summary_no_sensitive_data(self) -> None:
        """Test that health summary doesn't expose sensitive configuration."""
        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = ApplicationSettings(
                database_password=SecretStr("super_secret_password"),
                api_key=SecretStr("secret_api_key"),
                secret_key=SecretStr("secret_application_key"),
            )
            mock_get_settings.return_value = mock_settings

            summary = get_configuration_health_summary()

            # Convert to JSON string to check for secret leakage
            summary_json = json.dumps(summary)

            # Verify no secret values are in the JSON
            assert "super_secret_password" not in summary_json
            assert "secret_api_key" not in summary_json
            assert "secret_application_key" not in summary_json

            # Should not contain any SecretStr representations
            assert "SecretStr" not in summary_json

    def test_validation_error_sanitization(self) -> None:
        """Test that validation errors are properly sanitized."""
        sensitive_errors = [
            "Database password 'my_secret_password' is too short",
            "API key 'sk-1234567890abcdef' is invalid",
            "Secret key must not contain 'supersecret123'",
            "JWT secret 'jwt_secret_value' failed validation",
            "Configuration file /home/user/.env contains errors",
        ]

        sanitized = _sanitize_validation_errors(sensitive_errors)

        # Check that actual secret values are removed
        sanitized_text = " ".join(sanitized)
        assert "my_secret_password" not in sanitized_text
        assert "sk-1234567890abcdef" not in sanitized_text
        assert "supersecret123" not in sanitized_text
        assert "jwt_secret_value" not in sanitized_text
        assert "/home/user/.env" not in sanitized_text

        # But sanitized messages should still be present
        assert len(sanitized) == len(sensitive_errors)

        # Check that sensitive patterns are replaced
        assert any("Password configuration issue (details hidden)" in error for error in sanitized)
        assert any("API key configuration issue (details hidden)" in error for error in sanitized)
        assert any("Secret key configuration issue (details hidden)" in error for error in sanitized)
        assert any("JWT secret configuration issue (details hidden)" in error for error in sanitized)
        assert any("Configuration file path issue (path hidden)" in error for error in sanitized)
