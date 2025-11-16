"""
Comprehensive unit tests for Configuration Management system.

This module provides comprehensive unit test coverage for the configuration
management system including settings, validation, health checks, and performance
configuration.
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pydantic import BaseModel, Field, SecretStr, ValidationError

import src.config.settings
from src.config.constants import (
    CORS_ORIGINS_BY_ENVIRONMENT,
    FILE_PATH_PATTERNS,
    HEALTH_CHECK_ERROR_LIMIT,
    HEALTH_CHECK_SUGGESTION_LIMIT,
    QDRANT_DEFAULT_HOST,
    QDRANT_DEFAULT_PORT,
    QDRANT_DEFAULT_TIMEOUT,
    SECRET_FIELD_NAMES,
    SENSITIVE_ERROR_PATTERNS,
)
from src.config.health import (
    ConfigurationStatusModel,
    _count_configured_secrets,
    _determine_config_source,
    _sanitize_validation_errors,
    get_configuration_health_summary,
    get_configuration_status,
    get_mcp_configuration_health,
)
from src.config.performance_config import (
    OPERATION_THRESHOLDS,
    PerformanceConfig,
    get_async_config,
    get_cache_config,
    get_connection_pool_config,
    get_monitoring_config,
    get_optimization_recommendations,
    get_performance_config,
    validate_performance_requirements,
)
from src.config.settings import (
    ApplicationSettings,
    ConfigurationValidationError,
    _detect_environment,
    _env_file_settings,
    _get_project_root,
    _load_encrypted_env_file,
    _load_env_file,
    _log_configuration_status,
    _log_encryption_status,
    _mask_secret_value,
    _process_validation_errors,
    _validate_general_security,
    _validate_production_requirements,
    _validate_staging_requirements,
    get_settings,
    reload_settings,
    validate_configuration_on_startup,
    validate_encryption_available,
    validate_field_requirements_by_environment,
)
from src.utils.encryption import EncryptionError, GPGError


class TestApplicationSettings:
    """Test suite for ApplicationSettings class."""

    def test_application_settings_initialization(self):
        """Test ApplicationSettings initialization with defaults."""
        settings = ApplicationSettings()

        assert settings.app_name == "PromptCraft-Hybrid"
        assert settings.version == "0.1.0"
        assert settings.environment == "dev"
        assert settings.debug is True
        assert settings.api_host == "0.0.0.0"  # noqa: S104
        assert settings.api_port == 8000
        assert settings.mcp_server_url == "http://localhost:3000"
        assert settings.mcp_timeout == 30.0
        assert settings.mcp_max_retries == 3
        assert settings.mcp_enabled is True
        # Allow for environment override in CI (localhost) or production (192.168.1.16)
        assert settings.qdrant_host in ["192.168.1.16", "localhost"]
        assert settings.qdrant_port == 6333
        assert settings.qdrant_timeout == 30.0
        assert settings.qdrant_enabled is True
        assert settings.vector_store_type == "auto"
        assert settings.vector_dimensions == 384

    def test_application_settings_with_env_vars(self):
        """Test ApplicationSettings with environment variables."""
        env_vars = {
            "PROMPTCRAFT_APP_NAME": "Test-App",
            "PROMPTCRAFT_VERSION": "1.2.3",
            "PROMPTCRAFT_ENVIRONMENT": "staging",
            "PROMPTCRAFT_DEBUG": "false",
            "PROMPTCRAFT_API_HOST": "localhost",
            "PROMPTCRAFT_API_PORT": "3000",
            "PROMPTCRAFT_MCP_SERVER_URL": "http://localhost:4000",
            "PROMPTCRAFT_QDRANT_HOST": "127.0.0.1",
            "PROMPTCRAFT_QDRANT_PORT": "6334",
        }

        with patch.dict(os.environ, env_vars):
            settings = ApplicationSettings()

            assert settings.app_name == "Test-App"
            assert settings.version == "1.2.3"
            assert settings.environment == "staging"
            assert settings.debug is False
            assert settings.api_host == "localhost"
            assert settings.api_port == 3000
            assert settings.mcp_server_url == "http://localhost:4000"
            assert settings.qdrant_host == "127.0.0.1"
            assert settings.qdrant_port == 6334

    def test_application_settings_validation_errors(self):
        """Test ApplicationSettings validation errors."""
        # Test invalid port
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(api_port=70000)

        error = exc_info.value
        assert "Port 70000 is outside valid range" in str(error)

        # Test invalid environment
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(environment="invalid")

        error = exc_info.value
        assert "Input should be 'dev', 'staging' or 'prod'" in str(error)

        # Test invalid version
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(version="invalid-version")

        error = exc_info.value
        assert "Invalid version format" in str(error)

        # Test invalid app name
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(app_name="")

        error = exc_info.value
        assert "Application name cannot be empty" in str(error)

        # Test invalid host
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(api_host="invalid..host")

        error = exc_info.value
        assert "Invalid API host format" in str(error)

    def test_secret_field_validation(self):
        """Test secret field validation."""
        # Test valid secret
        settings = ApplicationSettings(
            database_password=SecretStr("valid_password"),
            api_key=SecretStr("valid_api_key"),
        )

        assert settings.database_password.get_secret_value() == "valid_password"
        assert settings.api_key.get_secret_value() == "valid_api_key"

        # Test empty secret validation
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(database_password=SecretStr(""))

        error = exc_info.value
        assert "Secret values cannot be empty strings" in str(error)

    def test_secret_field_names_consistency(self):
        """Test that SECRET_FIELD_NAMES matches actual secret fields."""
        settings = ApplicationSettings()

        # Check that all secret field names exist in the model
        for field_name in SECRET_FIELD_NAMES:
            assert hasattr(settings, field_name), f"Field {field_name} not found in ApplicationSettings"
            field_value = getattr(settings, field_name)
            # Should be None or SecretStr
            assert field_value is None or hasattr(field_value, "get_secret_value")


class TestConfigurationValidation:
    """Test suite for configuration validation functions."""

    def test_validate_field_requirements_by_environment(self):
        """Test field requirements validation by environment."""
        dev_fields = validate_field_requirements_by_environment("dev")
        staging_fields = validate_field_requirements_by_environment("staging")
        prod_fields = validate_field_requirements_by_environment("prod")

        # Base fields should be in all environments
        base_fields = {"app_name", "version", "environment", "api_host", "api_port"}
        assert base_fields.issubset(dev_fields)
        assert base_fields.issubset(staging_fields)
        assert base_fields.issubset(prod_fields)

        # Staging should require secret_key
        assert "secret_key" in staging_fields

        # Production should require both secret_key and jwt_secret_key
        assert "secret_key" in prod_fields
        assert "jwt_secret_key" in prod_fields

    def test_validate_production_requirements(self):
        """Test production requirements validation."""

        # Test production settings without required secrets
        settings = ApplicationSettings(
            environment="prod",
            debug=True,  # Should be False in production
            api_host="localhost",  # Should not be localhost in production
        )

        errors, suggestions = _validate_production_requirements(settings)

        assert len(errors) > 0
        assert "Debug mode should be disabled in production" in errors
        assert "Production API host should not be localhost/127.0.0.1" in errors
        assert "Required secret 'secret_key' is missing in production" in errors
        assert "Required secret 'jwt_secret_key' is missing in production" in errors

        assert len(suggestions) > 0
        assert any("PROMPTCRAFT_DEBUG=false" in s for s in suggestions)

    def test_validate_staging_requirements(self):
        """Test staging requirements validation."""
        settings = ApplicationSettings(environment="staging")

        errors, suggestions = _validate_staging_requirements(settings)

        assert len(errors) > 0
        assert "Secret key should be configured in staging environment" in errors
        assert len(suggestions) > 0

    def test_validate_general_security(self):
        """Test general security validation."""
        settings = ApplicationSettings(environment="dev", api_host="0.0.0.0", api_port=80)  # noqa: S104

        errors, suggestions = _validate_general_security(settings)

        assert len(errors) > 0
        assert any("Using standard web port 80 in dev environment" in e for e in errors)
        assert len(suggestions) > 0

    def test_validate_encryption_available(self):
        """Test encryption availability validation."""
        # Mock encryption validation - success case
        with patch("src.config.settings.validate_environment_keys") as mock_validate:
            mock_validate.return_value = True
            assert validate_encryption_available() is True

        # Mock encryption validation - failure case
        with patch("src.config.settings.validate_environment_keys") as mock_validate:
            mock_validate.side_effect = EncryptionError("Encryption not available")
            assert validate_encryption_available() is False

    def test_validate_configuration_on_startup(self):
        """Test configuration validation on startup."""
        # Test valid configuration
        settings = ApplicationSettings(environment="dev", debug=True, secret_key=SecretStr("dev_secret"))

        # Should not raise exception for valid dev config
        validate_configuration_on_startup(settings)

        # Test invalid production configuration
        prod_settings = ApplicationSettings(
            environment="prod",
            debug=True,
            api_host="localhost",  # Invalid for production  # Invalid for production
        )

        with pytest.raises(ConfigurationValidationError) as exc_info:
            validate_configuration_on_startup(prod_settings)

        error = exc_info.value
        assert "Configuration validation failed for prod environment" in str(error)
        assert len(error.field_errors) > 0
        assert len(error.suggestions) > 0

    def test_configuration_validation_error(self):
        """Test ConfigurationValidationError functionality."""
        field_errors = ["Field 1 error", "Field 2 error"]
        suggestions = ["Suggestion 1", "Suggestion 2"]

        error = ConfigurationValidationError("Test error", field_errors=field_errors, suggestions=suggestions)

        assert error.field_errors == field_errors
        assert error.suggestions == suggestions

        error_str = str(error)
        assert "Test error" in error_str
        assert "Field Validation Errors:" in error_str
        assert "Field 1 error" in error_str
        assert "Suggestions:" in error_str
        assert "Suggestion 1" in error_str

    def test_process_validation_errors(self):
        """Test processing of Pydantic validation errors."""
        # Mock Pydantic validation errors
        mock_errors = [
            {"loc": ("api_port",), "msg": "Port must be between 1-65535"},
            {"loc": ("environment",), "msg": "Invalid environment value"},
            {"loc": ("api_host",), "msg": "Invalid host format"},
        ]

        field_errors, suggestions = _process_validation_errors(mock_errors)

        assert len(field_errors) == 3
        assert "api_port: Port must be between 1-65535" in field_errors
        assert "environment: Invalid environment value" in field_errors
        assert "api_host: Invalid host format" in field_errors

        assert len(suggestions) == 3
        assert any("port between 1-65535" in s for s in suggestions)
        assert any("PROMPTCRAFT_ENVIRONMENT" in s for s in suggestions)
        assert any("0.0.0.0" in s for s in suggestions)  # noqa: S104


class TestEnvironmentDetection:
    """Test suite for environment detection functions."""

    def test_detect_environment_from_env_var(self):
        """Test environment detection from environment variable."""
        with patch.dict(os.environ, {"PROMPTCRAFT_ENVIRONMENT": "staging"}):
            env = _detect_environment()
            assert env == "staging"

        with patch.dict(os.environ, {"PROMPTCRAFT_ENVIRONMENT": "prod"}):
            env = _detect_environment()
            assert env == "prod"

    def test_detect_environment_from_file(self):
        """Test environment detection from .env file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("PROMPTCRAFT_ENVIRONMENT=staging\nOTHER_VAR=value")

            with (
                patch("src.config.settings._get_project_root", return_value=Path(tmp_dir)),
                patch.dict(os.environ, {}, clear=True),
            ):
                env = _detect_environment()
                assert env == "staging"

    def test_detect_environment_default(self):
        """Test environment detection default fallback."""
        with patch.dict(os.environ, {}, clear=True), patch("src.config.settings._get_project_root") as mock_root:
            mock_root.return_value = Path("/nonexistent")
            env = _detect_environment()
            assert env == "dev"

    def test_get_project_root(self):
        """Test project root detection."""
        root = _get_project_root()
        assert isinstance(root, Path)
        assert root.is_absolute()
        # Should be three levels up from src/config/settings.py
        assert root.name != "config"


class TestFileLoading:
    """Test suite for file loading functions."""

    def test_load_env_file(self):
        """Test loading environment variables from .env file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_content = """
# Comment line
PROMPTCRAFT_APP_NAME=Test App
PROMPTCRAFT_VERSION=1.0.0
PROMPTCRAFT_DEBUG=true
OTHER_VAR=ignored
PROMPTCRAFT_EMPTY=
"""
            env_file.write_text(env_content)

            env_vars = _load_env_file(env_file)

            assert env_vars["app_name"] == "Test App"
            assert env_vars["version"] == "1.0.0"
            assert env_vars["debug"] == "true"
            assert "other_var" not in env_vars  # Should be ignored
            assert "empty" in env_vars  # Should be included even if empty

    def test_load_env_file_nonexistent(self):
        """Test loading from non-existent file."""
        nonexistent_file = Path("/nonexistent/file.env")
        env_vars = _load_env_file(nonexistent_file)
        assert env_vars == {}

    def test_load_encrypted_env_file(self):
        """Test loading encrypted environment file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            encrypted_file = Path(tmp_dir) / ".env.gpg"
            encrypted_file.write_text("encrypted_content")

            # Mock the encryption utility
            with patch("src.config.settings.load_encrypted_env") as mock_load:
                mock_load.return_value = {
                    "PROMPTCRAFT_APP_NAME": "Encrypted App",
                    "PROMPTCRAFT_VERSION": "2.0.0",
                    "OTHER_VAR": "ignored",
                }

                env_vars = _load_encrypted_env_file(encrypted_file)

                assert env_vars["app_name"] == "Encrypted App"
                assert env_vars["version"] == "2.0.0"
                assert "other_var" not in env_vars

    def test_load_encrypted_env_file_error(self):
        """Test loading encrypted file with error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            encrypted_file = Path(tmp_dir) / ".env.gpg"
            encrypted_file.write_text("encrypted_content")

            with patch("src.config.settings.load_encrypted_env") as mock_load:
                mock_load.side_effect = GPGError("Decryption failed")

                env_vars = _load_encrypted_env_file(encrypted_file)
                assert env_vars == {}

    def test_env_file_settings(self):
        """Test environment file settings loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create base .env file
            base_env = Path(tmp_dir) / ".env"
            base_env.write_text("PROMPTCRAFT_APP_NAME=Base App\nPROMPTCRAFT_VERSION=1.0.0")

            # Create environment-specific file
            env_specific = Path(tmp_dir) / ".env.dev"
            env_specific.write_text("PROMPTCRAFT_APP_NAME=Dev App\nPROMPTCRAFT_DEBUG=true")

            with (
                patch("src.config.settings._get_project_root", return_value=Path(tmp_dir)),
                patch("src.config.settings._detect_environment", return_value="dev"),
                patch("src.config.settings.load_encrypted_env", side_effect=Exception("No encryption")),
            ):
                env_vars = _env_file_settings()

                # Environment-specific should override base
                assert env_vars["app_name"] == "Dev App"
                assert env_vars["version"] == "1.0.0"  # From base
                assert env_vars["debug"] == "true"  # From env-specific


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_mask_secret_value(self):
        """Test secret value masking."""
        assert _mask_secret_value("password123") == "*******d123"  # 11 chars: 7 asterisks + last 4
        assert _mask_secret_value("short") == "*hort"  # 5 chars: 1 asterisk + last 4
        assert (
            _mask_secret_value("verylongpassword", show_chars=6) == "**********ssword"
        )  # 16 chars: 10 asterisks + last 6
        assert _mask_secret_value("abc") == "***"  # 3 chars: all asterisks (≤ 4)
        assert _mask_secret_value("1234") == "****"  # 4 chars: all asterisks (≤ 4)

    def test_log_configuration_status(self):
        """Test configuration status logging."""
        settings = ApplicationSettings(
            environment="dev",
            app_name="Test App",
            version="1.0.0",
            api_host="localhost",
            api_port=8000,
            debug=True,
            secret_key=SecretStr("test_secret"),
            database_password=SecretStr("db_password"),
        )

        # Test that it doesn't raise an exception
        with patch("src.config.settings.logging.getLogger") as mock_logger:
            mock_logger.return_value.info = Mock()
            mock_logger.return_value.debug = Mock()

            _log_configuration_status(settings)

            # Should have called logger methods
            assert mock_logger.return_value.info.called

    def test_log_encryption_status(self):
        """Test encryption status logging."""
        with patch("src.config.settings.logging.getLogger") as mock_logger:
            mock_logger.return_value.warning = Mock()
            mock_logger.return_value.info = Mock()

            # Test production without encryption
            _log_encryption_status("prod", False, mock_logger.return_value)
            mock_logger.return_value.warning.assert_called()

            # Test development without encryption
            _log_encryption_status("dev", False, mock_logger.return_value)
            mock_logger.return_value.info.assert_called()

            # Test with encryption available
            _log_encryption_status("prod", True, mock_logger.return_value)
            mock_logger.return_value.info.assert_called()


class TestSettingsFactory:
    """Test suite for settings factory functions."""

    def setUp(self):
        """Set up test environment."""
        # Clear global settings
        src.config.settings._settings = None

    def tearDown(self):
        """Clean up test environment."""
        # Clear global settings
        src.config.settings._settings = None

    def test_get_settings_singleton(self):
        """Test settings singleton behavior."""
        self.setUp()

        with patch("src.config.settings.validate_encryption_available", return_value=True):
            settings1 = get_settings(validate_on_startup=False)
            settings2 = get_settings(validate_on_startup=False)

            # Should be the same instance
            assert settings1 is settings2

        self.tearDown()

    def test_get_settings_validation_error(self):
        """Test settings loading with validation error."""
        self.setUp()

        with (
            patch("src.config.settings.validate_encryption_available", return_value=False),
            patch("src.config.settings.ApplicationSettings") as mock_settings,
        ):
            # Create a mock ValidationError by creating a real validation error
            # with invalid input to trigger the same code path
            try:

                class MockModel(BaseModel):
                    api_port: int = Field(gt=0, le=65535)

                # This will create a real ValidationError
                MockModel(api_port=-1)  # Invalid port to trigger error
            except ValidationError as e:
                mock_settings.side_effect = e

            with pytest.raises(ConfigurationValidationError) as exc_info:
                get_settings()

            error = exc_info.value
            assert "Configuration field validation failed" in str(error)
            assert len(error.field_errors) > 0

        self.tearDown()

    def test_reload_settings(self):
        """Test settings reloading."""
        self.setUp()

        with patch("src.config.settings.validate_encryption_available", return_value=True):
            settings1 = get_settings(validate_on_startup=False)
            settings2 = reload_settings(validate_on_startup=False)

            # Should be different instances after reload
            assert settings1 is not settings2
            assert settings2.app_name == settings1.app_name  # Same content

        self.tearDown()


class TestConfigurationHealthChecks:
    """Test suite for configuration health check functions."""

    def test_configuration_status_model(self):
        """Test ConfigurationStatusModel creation and validation."""
        status = ConfigurationStatusModel(
            environment="dev",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=True,
            config_source="env_vars",
            validation_status="passed",
            validation_errors=[],
            secrets_configured=3,
            api_host="localhost",
            api_port=8000,
        )

        assert status.environment == "dev"
        assert status.version == "1.0.0"
        assert status.debug is True
        assert status.config_loaded is True
        assert status.encryption_enabled is True
        assert status.config_source == "env_vars"
        assert status.validation_status == "passed"
        assert status.validation_errors == []
        assert status.secrets_configured == 3
        assert status.api_host == "localhost"
        assert status.api_port == 8000
        assert status.config_healthy is True
        assert isinstance(status.timestamp, datetime)

    def test_configuration_status_unhealthy(self):
        """Test ConfigurationStatusModel unhealthy state."""
        status = ConfigurationStatusModel(
            environment="prod",
            version="1.0.0",
            debug=False,
            config_loaded=False,
            encryption_enabled=False,
            config_source="defaults",
            validation_status="failed",
            validation_errors=["Test error"],
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is False

    def test_count_configured_secrets(self):
        """Test counting configured secrets."""
        # Create settings with specific secrets, isolating from environment
        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings(
                secret_key=SecretStr("secret1"),
                database_password=SecretStr("secret2"),
                api_key=SecretStr("secret3"),
            )

        count = _count_configured_secrets(settings)
        assert count == 3

        # Test with empty secrets (isolate from environment)
        with patch.dict(os.environ, {}, clear=True):
            settings_empty = ApplicationSettings(database_password=None)

        count_empty = _count_configured_secrets(settings_empty)
        assert count_empty == 0

    def test_determine_config_source(self):
        """Test determining configuration source."""
        settings = ApplicationSettings()

        # Test with environment variables
        with patch.dict(os.environ, {"PROMPTCRAFT_ENVIRONMENT": "dev"}):
            source = _determine_config_source(settings)
            assert source == "env_vars"

        # Test with files
        with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.exists", return_value=True):
            source = _determine_config_source(settings)
            assert source == "env_files"

        # Test defaults
        with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.exists", return_value=False):
            source = _determine_config_source(settings)
            assert source == "defaults"

    def test_sanitize_validation_errors(self):
        """Test validation error sanitization."""
        errors = [
            "Password configuration failed",
            "API key is invalid",
            "Secret key missing",
            "File path /home/user/config.env not found",
            "Invalid value 'test_value'",
            "Normal error message",
        ]

        sanitized = _sanitize_validation_errors(errors)

        assert len(sanitized) == len(errors)
        assert "Password configuration issue (details hidden)" in sanitized
        assert "API key configuration issue (details hidden)" in sanitized
        assert "Secret key configuration issue (details hidden)" in sanitized
        assert "Configuration file path issue (path hidden)" in sanitized
        assert "Invalid value '***'" in sanitized
        assert "Normal error message" in sanitized

    def test_get_configuration_status(self):
        """Test getting configuration status."""
        # Create settings with specific secrets, isolating from environment
        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings(
                environment="dev",
                version="1.0.0",
                debug=True,
                secret_key=SecretStr("test_secret"),
                api_key=SecretStr("test_api_key"),
            )

            with (
                patch("src.config.health.validate_encryption_available", return_value=True),
                patch("src.config.health.validate_configuration_on_startup"),
            ):
                status = get_configuration_status(settings)

                assert isinstance(status, ConfigurationStatusModel)
                assert status.environment == "dev"
                assert status.version == "1.0.0"
                assert status.debug is True
                assert status.config_loaded is True
                assert status.encryption_enabled is True
                assert status.validation_status == "passed"
                assert status.secrets_configured == 2
                assert status.config_healthy is True

    def test_get_configuration_status_with_errors(self):
        """Test getting configuration status with validation errors."""
        settings = ApplicationSettings(environment="prod", debug=True)

        with (
            patch("src.config.health.validate_encryption_available", return_value=False),
            patch("src.config.health.validate_configuration_on_startup") as mock_validate,
        ):
            mock_validate.side_effect = ConfigurationValidationError(
                "Test error",
                field_errors=["Debug mode enabled in production"],
                suggestions=["Set debug=false"],
            )

            status = get_configuration_status(settings)

            assert status.validation_status == "failed"
            assert len(status.validation_errors) > 0
            assert status.config_healthy is False

    def test_get_configuration_health_summary(self):
        """Test getting configuration health summary."""
        with (
            patch("src.config.health.get_settings") as mock_get_settings,
            patch("src.config.health.get_configuration_status") as mock_get_status,
        ):
            mock_settings = ApplicationSettings(environment="dev", version="1.0.0")
            mock_get_settings.return_value = mock_settings

            mock_status = ConfigurationStatusModel(
                environment="dev",
                version="1.0.0",
                debug=True,
                config_loaded=True,
                encryption_enabled=True,
                config_source="env_vars",
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

    def test_get_configuration_health_summary_error(self):
        """Test configuration health summary with error."""
        with patch("src.config.health.get_settings") as mock_get_settings:
            mock_get_settings.side_effect = ValueError("Configuration error")

            summary = get_configuration_health_summary()

            assert summary["healthy"] is False
            assert summary["error"] == "Configuration health check failed"
            assert "timestamp" in summary

    @pytest.mark.asyncio
    async def test_get_mcp_configuration_health_no_mcp(self):
        """Test MCP configuration health when MCP is not available."""
        with patch("src.config.health.MCPClient", None):
            health = await get_mcp_configuration_health()

            assert health["healthy"] is False
            assert "MCP integration not available" in health["error"]
            assert "timestamp" in health

    @pytest.mark.asyncio
    async def test_get_mcp_configuration_health_success(self):
        """Test successful MCP configuration health check."""
        # Mock MCP components
        mock_config_manager = Mock()
        mock_config_manager.get_health_status.return_value = {"configuration_valid": True}

        mock_mcp_client = Mock()
        mock_mcp_client.health_check = AsyncMock(return_value={"overall_status": "healthy"})

        mock_parallel_executor = Mock()
        mock_parallel_executor.health_check = AsyncMock(return_value={"status": "healthy"})

        with (
            patch("src.config.health.MCPConfigurationManager", return_value=mock_config_manager),
            patch("src.config.health.MCPClient", return_value=mock_mcp_client),
            patch("src.config.health.ParallelSubagentExecutor", return_value=mock_parallel_executor),
        ):
            health = await get_mcp_configuration_health()

            assert health["healthy"] is True
            assert "mcp_configuration" in health
            assert "mcp_client" in health
            assert "parallel_executor" in health
            assert "timestamp" in health

    @pytest.mark.asyncio
    async def test_get_mcp_configuration_health_failure(self):
        """Test MCP configuration health check failure."""
        # Mock all MCP components as available (not None)
        with (
            patch("src.config.health.MCPConfigurationManager", Mock()) as mock_manager,
            patch("src.config.health.MCPClient", Mock()),
            patch("src.config.health.ParallelSubagentExecutor", Mock()),
        ):
            mock_manager.side_effect = Exception("MCP initialization failed")

            health = await get_mcp_configuration_health()

            assert health["healthy"] is False
            assert "MCP health check failed" in health["error"]
            assert "timestamp" in health


class TestPerformanceConfiguration:
    """Test suite for performance configuration."""

    def test_performance_config_initialization(self):
        """Test PerformanceConfig initialization."""
        config = PerformanceConfig()

        assert config.max_response_time_ms == 2000
        assert config.target_response_time_ms == 1000
        assert config.cache_response_time_ms == 100
        assert config.query_cache_size == 500
        assert config.query_cache_ttl_seconds == 300
        assert config.hyde_cache_size == 200
        assert config.hyde_cache_ttl_seconds == 600
        assert config.vector_cache_size == 1000
        assert config.vector_cache_ttl_seconds == 180
        assert config.max_vector_connections == 20
        assert config.max_mcp_connections == 10
        assert config.connection_timeout_seconds == 30
        assert config.batch_size == 25
        assert config.max_batch_wait_time_ms == 50
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_reset_timeout_seconds == 60
        assert config.performance_monitoring_enabled is True
        assert config.slow_query_threshold_ms == 1500
        assert config.memory_usage_threshold_mb == 512
        assert config.max_concurrent_queries == 100
        assert config.semaphore_limit == 50
        assert config.vector_search_timeout_seconds == 10
        assert config.vector_batch_size == 100
        assert config.vector_max_retries == 3
        assert config.mcp_request_timeout_seconds == 15
        assert config.mcp_max_retries == 3
        assert config.mcp_backoff_multiplier == 1.5

    def test_get_performance_config_dev(self):
        """Test performance configuration for development."""
        with patch("src.config.performance_config.get_settings") as mock_get_settings:
            mock_settings = ApplicationSettings(environment="dev")
            mock_get_settings.return_value = mock_settings

            config = get_performance_config()

            assert config.max_response_time_ms == 3000  # Relaxed for dev
            assert config.query_cache_size == 100  # Smaller for dev
            assert config.performance_monitoring_enabled is True
            assert config.max_concurrent_queries == 50

    def test_get_performance_config_staging(self):
        """Test performance configuration for staging."""
        with patch("src.config.performance_config.get_settings") as mock_get_settings:
            mock_settings = ApplicationSettings(environment="staging")
            mock_get_settings.return_value = mock_settings

            config = get_performance_config()

            assert config.max_response_time_ms == 2000
            assert config.query_cache_size == 300
            assert config.performance_monitoring_enabled is True
            assert config.max_concurrent_queries == 75

    def test_get_performance_config_prod(self):
        """Test performance configuration for production."""
        with patch("src.config.performance_config.get_settings") as mock_get_settings:
            mock_settings = ApplicationSettings(environment="prod")
            mock_get_settings.return_value = mock_settings

            config = get_performance_config()

            assert config.max_response_time_ms == 2000
            assert config.target_response_time_ms == 800  # Stricter target
            assert config.query_cache_size == 500
            assert config.hyde_cache_size == 300
            assert config.vector_cache_size == 1500
            assert config.max_concurrent_queries == 100
            assert config.performance_monitoring_enabled is True
            assert config.slow_query_threshold_ms == 1000  # Stricter threshold

    def test_get_cache_config(self):
        """Test cache configuration."""
        with patch("src.config.performance_config.get_performance_config") as mock_get_config:
            mock_config = PerformanceConfig()
            mock_get_config.return_value = mock_config

            cache_config = get_cache_config()

            assert "query_cache" in cache_config
            assert "hyde_cache" in cache_config
            assert "vector_cache" in cache_config

            assert cache_config["query_cache"]["max_size"] == 500
            assert cache_config["query_cache"]["ttl_seconds"] == 300
            assert cache_config["hyde_cache"]["max_size"] == 200
            assert cache_config["hyde_cache"]["ttl_seconds"] == 600
            assert cache_config["vector_cache"]["max_size"] == 1000
            assert cache_config["vector_cache"]["ttl_seconds"] == 180

    def test_get_connection_pool_config(self):
        """Test connection pool configuration."""
        with patch("src.config.performance_config.get_performance_config") as mock_get_config:
            mock_config = PerformanceConfig()
            mock_get_config.return_value = mock_config

            pool_config = get_connection_pool_config()

            assert "vector_store" in pool_config
            assert "mcp_client" in pool_config

            assert pool_config["vector_store"]["max_connections"] == 20
            assert pool_config["vector_store"]["timeout_seconds"] == 30
            assert pool_config["mcp_client"]["max_connections"] == 10
            assert pool_config["mcp_client"]["timeout_seconds"] == 15

    def test_get_async_config(self):
        """Test async configuration."""
        with patch("src.config.performance_config.get_performance_config") as mock_get_config:
            mock_config = PerformanceConfig()
            mock_get_config.return_value = mock_config

            async_config = get_async_config()

            assert async_config["max_concurrent_queries"] == 100
            assert async_config["semaphore_limit"] == 50
            assert async_config["batch_size"] == 25
            assert async_config["max_batch_wait_time_ms"] == 50

    def test_get_monitoring_config(self):
        """Test monitoring configuration."""
        with patch("src.config.performance_config.get_performance_config") as mock_get_config:
            mock_config = PerformanceConfig()
            mock_get_config.return_value = mock_config

            monitoring_config = get_monitoring_config()

            assert monitoring_config["enabled"] is True
            assert monitoring_config["slow_query_threshold_ms"] == 1500
            assert monitoring_config["memory_usage_threshold_mb"] == 512
            assert monitoring_config["max_response_time_ms"] == 2000
            assert monitoring_config["target_response_time_ms"] == 1000

    def test_validate_performance_requirements(self):
        """Test performance requirements validation."""
        with patch("src.config.performance_config.get_performance_config") as mock_get_config:
            # Test valid configuration
            valid_config = PerformanceConfig()
            mock_get_config.return_value = valid_config

            assert validate_performance_requirements() is True

            # Test invalid configuration
            invalid_config = PerformanceConfig()
            invalid_config.target_response_time_ms = 3000  # > max_response_time_ms
            invalid_config.query_cache_ttl_seconds = 30  # < 60
            invalid_config.max_vector_connections = 2  # < 5
            mock_get_config.return_value = invalid_config

            assert validate_performance_requirements() is False

    def test_get_optimization_recommendations(self):
        """Test performance optimization recommendations."""
        with (
            patch("src.config.performance_config.get_performance_config") as mock_get_config,
            patch("src.config.performance_config.get_settings") as mock_get_settings,
        ):
            mock_config = PerformanceConfig()
            mock_get_config.return_value = mock_config

            # Test development recommendations
            mock_settings = ApplicationSettings(environment="dev")
            mock_get_settings.return_value = mock_settings

            recommendations = get_optimization_recommendations()

            assert "caching" in recommendations
            assert "monitoring" in recommendations
            assert "connections" in recommendations
            assert "response_time" in recommendations
            assert "batching" in recommendations
            assert "concurrency" in recommendations

            assert "Enable all caches for realistic testing" in recommendations["caching"]
            assert "Use detailed performance monitoring" in recommendations["monitoring"]
            assert "Use smaller connection pools" in recommendations["connections"]

            # Test production recommendations
            mock_settings = ApplicationSettings(environment="prod")
            mock_get_settings.return_value = mock_settings

            recommendations = get_optimization_recommendations()

            assert "Maximize cache sizes and TTL" in recommendations["caching"]
            assert "Enable alerts for performance degradation" in recommendations["monitoring"]
            assert "Use maximum connection pooling" in recommendations["connections"]
            assert "optimization" in recommendations

    def test_operation_thresholds(self):
        """Test operation thresholds constants."""
        assert "query_analysis" in OPERATION_THRESHOLDS
        assert "hyde_processing" in OPERATION_THRESHOLDS
        assert "vector_search" in OPERATION_THRESHOLDS
        assert "agent_orchestration" in OPERATION_THRESHOLDS
        assert "response_synthesis" in OPERATION_THRESHOLDS
        assert "end_to_end" in OPERATION_THRESHOLDS

        assert OPERATION_THRESHOLDS["query_analysis"] == 500
        assert OPERATION_THRESHOLDS["hyde_processing"] == 800
        assert OPERATION_THRESHOLDS["vector_search"] == 400
        assert OPERATION_THRESHOLDS["agent_orchestration"] == 1000
        assert OPERATION_THRESHOLDS["response_synthesis"] == 300
        assert OPERATION_THRESHOLDS["end_to_end"] == 2000


class TestConfigurationConstants:
    """Test suite for configuration constants."""

    def test_secret_field_names(self):
        """Test secret field names constant."""
        assert isinstance(SECRET_FIELD_NAMES, list)
        assert len(SECRET_FIELD_NAMES) > 0
        assert "database_password" in SECRET_FIELD_NAMES
        assert "api_key" in SECRET_FIELD_NAMES
        assert "secret_key" in SECRET_FIELD_NAMES
        assert "jwt_secret_key" in SECRET_FIELD_NAMES
        assert "qdrant_api_key" in SECRET_FIELD_NAMES
        assert "mcp_api_key" in SECRET_FIELD_NAMES

    def test_qdrant_defaults(self):
        """Test Qdrant default constants."""
        assert QDRANT_DEFAULT_HOST == "192.168.1.16"
        assert QDRANT_DEFAULT_PORT == 6333
        assert QDRANT_DEFAULT_TIMEOUT == 30.0

    def test_sensitive_error_patterns(self):
        """Test sensitive error patterns."""
        assert isinstance(SENSITIVE_ERROR_PATTERNS, list)
        assert len(SENSITIVE_ERROR_PATTERNS) > 0

        # Each pattern should be a tuple of (pattern, replacement)
        for pattern, replacement in SENSITIVE_ERROR_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(replacement, str)
            assert len(pattern) > 0
            assert len(replacement) > 0

    def test_file_path_patterns(self):
        """Test file path patterns."""
        assert isinstance(FILE_PATH_PATTERNS, list)
        assert len(FILE_PATH_PATTERNS) > 0
        assert "/home/" in FILE_PATH_PATTERNS
        assert "C:\\\\" in FILE_PATH_PATTERNS
        assert "/Users/" in FILE_PATH_PATTERNS

    def test_health_check_constants(self):
        """Test health check constants."""
        assert isinstance(HEALTH_CHECK_ERROR_LIMIT, int)
        assert HEALTH_CHECK_ERROR_LIMIT > 0
        assert isinstance(HEALTH_CHECK_SUGGESTION_LIMIT, int)
        assert HEALTH_CHECK_SUGGESTION_LIMIT > 0

    def test_cors_origins(self):
        """Test CORS origins configuration."""
        assert isinstance(CORS_ORIGINS_BY_ENVIRONMENT, dict)
        assert "dev" in CORS_ORIGINS_BY_ENVIRONMENT
        assert "staging" in CORS_ORIGINS_BY_ENVIRONMENT
        assert "prod" in CORS_ORIGINS_BY_ENVIRONMENT

        # Each environment should have a list of origins
        for _env, origins in CORS_ORIGINS_BY_ENVIRONMENT.items():
            assert isinstance(origins, list)
            assert len(origins) > 0
            for origin in origins:
                assert isinstance(origin, str)
                assert origin.startswith("http")


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_end_to_end_configuration_loading(self):
        """Test end-to-end configuration loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test .env file
            env_file = Path(tmp_dir) / ".env"
            env_content = """
PROMPTCRAFT_APP_NAME=Integration Test App
PROMPTCRAFT_VERSION=1.0.0
PROMPTCRAFT_ENVIRONMENT=dev
PROMPTCRAFT_DEBUG=true
PROMPTCRAFT_API_HOST=localhost
PROMPTCRAFT_API_PORT=8000
PROMPTCRAFT_SECRET_KEY=test_secret_key
"""
            env_file.write_text(env_content)

            # Isolate from environment variables
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("src.config.settings._get_project_root", return_value=Path(tmp_dir)),
                patch("src.config.settings.validate_environment_keys"),
            ):
                # Clear global settings completely
                # Reset all cached settings
                src.config.settings._settings = None
                if hasattr(src.config.settings, "_cached_settings"):
                    src.config.settings._cached_settings = None

                # Force reload from the temporary directory
                settings = src.config.settings.ApplicationSettings(
                    _env_file=env_file,
                    _env_file_encoding="utf-8",
                )

                assert settings.app_name == "Integration Test App"
                assert settings.version == "1.0.0"
                assert settings.environment == "dev"
                assert settings.debug is True
                assert settings.api_host == "localhost"
                assert settings.api_port == 8000
                assert settings.secret_key.get_secret_value() == "test_secret_key"

                # Test health check
                status = get_configuration_status(settings)
                assert status.config_healthy is True
                assert status.environment == "dev"
                assert status.version == "1.0.0"

                # Test performance config
                perf_config = get_performance_config()
                assert perf_config.max_response_time_ms == 3000  # Dev setting

                # Clean up
                src.config.settings._settings = None

    def test_configuration_with_validation_errors(self):
        """Test configuration with validation errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test .env file with invalid values
            env_file = Path(tmp_dir) / ".env"
            env_content = """
PROMPTCRAFT_APP_NAME=
PROMPTCRAFT_VERSION=invalid-version
PROMPTCRAFT_ENVIRONMENT=prod
PROMPTCRAFT_DEBUG=true
PROMPTCRAFT_API_HOST=invalid..host
PROMPTCRAFT_API_PORT=70000
"""
            env_file.write_text(env_content)

            # Isolate from environment variables
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("src.config.settings._get_project_root", return_value=Path(tmp_dir)),
                patch("src.config.settings.validate_environment_keys"),
            ):
                # Clear global settings
                src.config.settings._settings = None

                with pytest.raises((ConfigurationValidationError, ValidationError)) as exc_info:
                    # Create settings directly from the env file to force validation
                    validate_configuration_on_startup(
                        src.config.settings.ApplicationSettings(
                            _env_file=env_file,
                            _env_file_encoding="utf-8",
                        ),
                    )

                error = exc_info.value
                if isinstance(error, ConfigurationValidationError):
                    assert len(error.field_errors) > 0
                    assert len(error.suggestions) > 0
                else:
                    # Pydantic ValidationError
                    assert len(error.errors()) > 0

                # Clean up
                src.config.settings._settings = None

    def test_configuration_health_check_integration(self):
        """Test configuration health check integration."""
        settings = ApplicationSettings(
            environment="staging",
            version="1.0.0",
            debug=False,
            secret_key=SecretStr("staging_secret"),
            api_host="0.0.0.0",  # noqa: S104
            api_port=8000,
        )

        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health.validate_configuration_on_startup"),
        ):
            status = get_configuration_status(settings)

            assert status.config_healthy is True
            assert status.environment == "staging"
            assert status.secrets_configured >= 1
            assert status.encryption_enabled is True
            assert status.validation_status == "passed"

            # Test health summary
            with patch("src.config.health.get_settings", return_value=settings):
                summary = get_configuration_health_summary()

                assert summary["healthy"] is True
                assert summary["environment"] == "staging"
                assert summary["version"] == "1.0.0"
                assert summary["config_loaded"] is True
                assert summary["encryption_available"] is True

    def test_performance_config_integration(self):
        """Test performance configuration integration."""
        with patch("src.config.performance_config.get_settings") as mock_get_settings:
            # Test with different environments
            environments = ["dev", "staging", "prod"]

            for env in environments:
                mock_settings = ApplicationSettings(environment=env)
                mock_get_settings.return_value = mock_settings

                # Test all config functions
                perf_config = get_performance_config()
                cache_config = get_cache_config()
                pool_config = get_connection_pool_config()
                async_config = get_async_config()
                monitoring_config = get_monitoring_config()
                recommendations = get_optimization_recommendations()

                # Verify all configs are valid
                assert isinstance(perf_config, PerformanceConfig)
                assert isinstance(cache_config, dict)
                assert isinstance(pool_config, dict)
                assert isinstance(async_config, dict)
                assert isinstance(monitoring_config, dict)
                assert isinstance(recommendations, dict)

                # Verify environment-specific settings
                if env == "dev":
                    assert perf_config.max_response_time_ms == 3000
                    assert perf_config.query_cache_size == 100
                elif env == "staging":
                    assert perf_config.max_response_time_ms == 2000
                    assert perf_config.query_cache_size == 300
                elif env == "prod":
                    assert perf_config.max_response_time_ms == 2000
                    assert perf_config.target_response_time_ms == 800
                    assert perf_config.query_cache_size == 500

                # Verify performance requirements
                assert validate_performance_requirements() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
