"""
Unit tests for health check configuration models and utilities.

This module provides comprehensive test coverage for the health check functionality
that monitors configuration status and system health. Tests include health checker,
configuration status models, validation error sanitization, and MCP integration.
Uses proper pytest markers for codecov integration per codecov.yml config component.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config.health import (
    ConfigurationStatusModel,
    HealthChecker,
    _count_configured_secrets,
    _determine_config_source,
    _sanitize_validation_errors,
    get_configuration_health_summary,
    get_configuration_status,
    get_mcp_configuration_health,
)
from src.config.settings import ApplicationSettings, ConfigurationValidationError


@pytest.mark.unit
class TestConfigurationStatusModel:
    """Test cases for ConfigurationStatusModel."""

    def test_model_initialization_minimal(self):
        """Test model initialization with minimal required fields."""
        status = ConfigurationStatusModel(
            environment="test",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=False,
            config_source="defaults",
            validation_status="passed",
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.environment == "test"
        assert status.version == "1.0.0"
        assert status.debug is True
        assert status.config_loaded is True
        assert status.encryption_enabled is False
        assert status.config_source == "defaults"
        assert status.validation_status == "passed"
        assert status.validation_errors == []
        assert status.secrets_configured == 0
        assert status.api_host == "localhost"
        assert status.api_port == 8000
        assert isinstance(status.timestamp, datetime)

    def test_model_initialization_complete(self):
        """Test model initialization with all fields."""
        timestamp = datetime.now(UTC)
        validation_errors = ["Error 1", "Error 2"]

        status = ConfigurationStatusModel(
            environment="production",
            version="2.1.0",
            debug=False,
            config_loaded=True,
            encryption_enabled=True,
            config_source="env_vars",
            validation_status="warning",
            validation_errors=validation_errors,
            secrets_configured=5,
            api_host="api.example.com",
            api_port=443,
            timestamp=timestamp,
        )

        assert status.environment == "production"
        assert status.version == "2.1.0"
        assert status.debug is False
        assert status.encryption_enabled is True
        assert status.config_source == "env_vars"
        assert status.validation_status == "warning"
        assert status.validation_errors == validation_errors
        assert status.secrets_configured == 5
        assert status.api_host == "api.example.com"
        assert status.api_port == 443
        assert status.timestamp == timestamp

    def test_config_healthy_passed_status(self):
        """Test config_healthy computed field with passed validation."""
        status = ConfigurationStatusModel(
            environment="test",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=True,
            config_source="defaults",
            validation_status="passed",
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is True

    def test_config_healthy_warning_status(self):
        """Test config_healthy computed field with warning validation."""
        status = ConfigurationStatusModel(
            environment="test",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=True,
            config_source="defaults",
            validation_status="warning",
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is True

    def test_config_healthy_failed_status(self):
        """Test config_healthy computed field with failed validation."""
        status = ConfigurationStatusModel(
            environment="test",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=True,
            config_source="defaults",
            validation_status="failed",
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is False

    def test_config_healthy_config_not_loaded(self):
        """Test config_healthy computed field when config not loaded."""
        status = ConfigurationStatusModel(
            environment="test",
            version="1.0.0",
            debug=True,
            config_loaded=False,
            encryption_enabled=True,
            config_source="defaults",
            validation_status="passed",
            secrets_configured=0,
            api_host="localhost",
            api_port=8000,
        )

        assert status.config_healthy is False

    def test_model_serialization(self):
        """Test model serialization to dictionary."""
        status = ConfigurationStatusModel(
            environment="test",
            version="1.0.0",
            debug=True,
            config_loaded=True,
            encryption_enabled=True,
            config_source="env_files",
            validation_status="passed",
            validation_errors=["Warning message"],
            secrets_configured=3,
            api_host="localhost",
            api_port=8000,
        )

        data = status.model_dump()

        assert data["environment"] == "test"
        assert data["version"] == "1.0.0"
        assert data["debug"] is True
        assert data["config_loaded"] is True
        assert data["encryption_enabled"] is True
        assert data["config_source"] == "env_files"
        assert data["validation_status"] == "passed"
        assert data["validation_errors"] == ["Warning message"]
        assert data["secrets_configured"] == 3
        assert data["api_host"] == "localhost"
        assert data["api_port"] == 8000
        assert data["config_healthy"] is True
        assert "timestamp" in data


@pytest.mark.unit
class TestHealthChecker:
    """Test cases for HealthChecker class."""

    def test_init(self):
        """Test HealthChecker initialization."""
        settings = Mock(spec=ApplicationSettings)
        checker = HealthChecker(settings)

        assert checker.settings == settings
        assert hasattr(checker, "logger")

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """Test successful health check."""
        settings = Mock(spec=ApplicationSettings)
        checker = HealthChecker(settings)

        # Mock configuration status
        mock_status = Mock()
        mock_status.config_healthy = True
        mock_status.model_dump.return_value = {"environment": "test", "config_healthy": True}

        # Mock MCP health
        mock_mcp_health = {"healthy": True, "status": "operational"}

        with (
            patch("src.config.health.get_configuration_status", return_value=mock_status),
            patch("src.config.health.get_mcp_configuration_health", return_value=mock_mcp_health),
        ):
            result = await checker.check_health()

            assert result["healthy"] is True
            assert "configuration" in result
            assert "mcp" in result
            assert "timestamp" in result
            assert result["configuration"]["config_healthy"] is True
            assert result["mcp"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_check_health_config_unhealthy(self):
        """Test health check with unhealthy configuration."""
        settings = Mock(spec=ApplicationSettings)
        checker = HealthChecker(settings)

        # Mock unhealthy configuration status
        mock_status = Mock()
        mock_status.config_healthy = False
        mock_status.model_dump.return_value = {"environment": "test", "config_healthy": False}

        # Mock healthy MCP
        mock_mcp_health = {"healthy": True, "status": "operational"}

        with (
            patch("src.config.health.get_configuration_status", return_value=mock_status),
            patch("src.config.health.get_mcp_configuration_health", return_value=mock_mcp_health),
        ):
            result = await checker.check_health()

            assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_check_health_mcp_unhealthy(self):
        """Test health check with unhealthy MCP."""
        settings = Mock(spec=ApplicationSettings)
        checker = HealthChecker(settings)

        # Mock healthy configuration status
        mock_status = Mock()
        mock_status.config_healthy = True
        mock_status.model_dump.return_value = {"environment": "test", "config_healthy": True}

        # Mock unhealthy MCP
        mock_mcp_health = {"healthy": False, "error": "MCP service unavailable"}

        with (
            patch("src.config.health.get_configuration_status", return_value=mock_status),
            patch("src.config.health.get_mcp_configuration_health", return_value=mock_mcp_health),
        ):
            result = await checker.check_health()

            assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_check_health_exception(self):
        """Test health check with exception handling."""
        settings = Mock(spec=ApplicationSettings)
        checker = HealthChecker(settings)

        with patch("src.config.health.get_configuration_status", side_effect=RuntimeError("Test error")):
            result = await checker.check_health()

            assert result["healthy"] is False
            assert "error" in result
            assert "Health check failed: Test error" in result["error"]
            assert "timestamp" in result


@pytest.mark.unit
class TestCountConfiguredSecrets:
    """Test cases for _count_configured_secrets function."""

    def test_count_secrets_none_configured(self):
        """Test counting secrets when none are configured."""
        settings = Mock(spec=ApplicationSettings)

        # Mock all secret fields as None or empty
        with patch("src.config.health.SECRET_FIELD_NAMES", ["secret1", "secret2", "secret3"]):
            settings.secret1 = None
            settings.secret2 = Mock()
            settings.secret2.get_secret_value.return_value = ""
            settings.secret3 = Mock()
            settings.secret3.get_secret_value.return_value = "   "  # Whitespace only

            count = _count_configured_secrets(settings)

            assert count == 0

    def test_count_secrets_some_configured(self):
        """Test counting secrets when some are configured."""
        settings = Mock(spec=ApplicationSettings)

        with patch("src.config.health.SECRET_FIELD_NAMES", ["secret1", "secret2", "secret3"]):
            settings.secret1 = Mock()
            settings.secret1.get_secret_value.return_value = "configured_secret"
            settings.secret2 = None
            settings.secret3 = Mock()
            settings.secret3.get_secret_value.return_value = "another_secret"

            count = _count_configured_secrets(settings)

            assert count == 2

    def test_count_secrets_all_configured(self):
        """Test counting secrets when all are configured."""
        settings = Mock(spec=ApplicationSettings)

        with patch("src.config.health.SECRET_FIELD_NAMES", ["secret1", "secret2"]):
            settings.secret1 = Mock()
            settings.secret1.get_secret_value.return_value = "secret_value_1"
            settings.secret2 = Mock()
            settings.secret2.get_secret_value.return_value = "secret_value_2"

            count = _count_configured_secrets(settings)

            assert count == 2

    def test_count_secrets_missing_attribute(self):
        """Test counting secrets when attribute is missing."""
        settings = Mock(spec=ApplicationSettings)

        with patch("src.config.health.SECRET_FIELD_NAMES", ["secret1", "missing_secret"]):
            settings.secret1 = Mock()
            settings.secret1.get_secret_value.return_value = "configured_secret"
            # missing_secret attribute doesn't exist

            count = _count_configured_secrets(settings)

            # Should handle missing attribute gracefully and count only existing one
            assert count == 1


@pytest.mark.unit
class TestDetermineConfigSource:
    """Test cases for _determine_config_source function."""

    def test_determine_source_env_vars(self):
        """Test config source determination with environment variables."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"

        with patch.dict(os.environ, {"PROMPTCRAFT_ENVIRONMENT": "test", "PROMPTCRAFT_API_HOST": "localhost"}):
            source = _determine_config_source(settings)

            assert source == "env_vars"

    def test_determine_source_env_files(self):
        """Test config source determination with env files."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"

        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True

            source = _determine_config_source(settings)

            assert source == "env_files"

    def test_determine_source_defaults(self):
        """Test config source determination with defaults."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"

        # Clear environment variables and no env files
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "exists", return_value=False):
            source = _determine_config_source(settings)

            assert source == "defaults"

    def test_determine_source_partial_env_vars(self):
        """Test config source determination with partial env vars."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"

        # Only set one environment variable
        with (
            patch.dict(os.environ, {"PROMPTCRAFT_DEBUG": "true"}, clear=True),
            patch.object(Path, "exists", return_value=False),
        ):
            source = _determine_config_source(settings)

            assert source == "env_vars"

    def test_determine_source_specific_env_file_patterns(self):
        """Test config source determination checks specific env file patterns."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "production"

        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "exists") as mock_exists:

            def exists_side_effect():
                # Return True for .env.production.gpg file
                return True

            mock_exists.side_effect = exists_side_effect

            source = _determine_config_source(settings)

            assert source == "env_files"


@pytest.mark.unit
class TestSanitizeValidationErrors:
    """Test cases for _sanitize_validation_errors function."""

    def test_sanitize_no_sensitive_data(self):
        """Test sanitization with no sensitive data."""
        errors = ["Configuration field 'port' must be positive", "Invalid timeout value", "Missing required field"]

        result = _sanitize_validation_errors(errors)

        assert result == errors

    def test_sanitize_quoted_values(self):
        """Test sanitization of quoted values."""
        errors = [
            "Invalid value 'secret123' for field password",
            'Database connection "postgres://user:pass@host/db" failed',
            "Field 'api_key' has invalid format",
        ]

        result = _sanitize_validation_errors(errors)

        assert "secret123" not in result[0]
        assert "'***'" in result[0]
        assert "postgres://user:pass@host/db" not in result[1]
        assert '"***"' in result[1]
        assert "'api_key'" in result[2]  # Field names should remain

    def test_sanitize_with_compiled_patterns(self):
        """Test sanitization using compiled sensitive patterns."""
        errors = ["Test error message"]

        # Mock compiled patterns
        mock_pattern = Mock()
        mock_pattern.search.return_value = True  # Pattern matches

        with patch("src.config.health._COMPILED_SENSITIVE_PATTERNS", [(mock_pattern, "SENSITIVE_DATA_REDACTED")]):
            result = _sanitize_validation_errors(errors)

            assert result == ["SENSITIVE_DATA_REDACTED"]

    def test_sanitize_with_file_path_patterns(self):
        """Test sanitization of file path patterns."""
        errors = ["Test error with file path"]

        # Mock compiled patterns - no sensitive pattern match but file path match
        mock_sensitive_pattern = Mock()
        mock_sensitive_pattern.search.return_value = False

        mock_file_pattern = Mock()
        mock_file_pattern.search.return_value = True

        with (
            patch("src.config.health._COMPILED_SENSITIVE_PATTERNS", [(mock_sensitive_pattern, "SENSITIVE")]),
            patch("src.config.health._COMPILED_FILE_PATH_PATTERNS", [mock_file_pattern]),
        ):
            result = _sanitize_validation_errors(errors)

            assert result == ["Configuration file path issue (path hidden)"]

    def test_sanitize_multiple_errors(self):
        """Test sanitization of multiple errors with different patterns."""
        errors = [
            "Normal error message",
            "Invalid password 'secret123'",
            'Database URL "postgres://user:pass@host" invalid',
        ]

        result = _sanitize_validation_errors(errors)

        assert result[0] == "Normal error message"
        assert "'***'" in result[1]
        assert "secret123" not in result[1]
        assert '"***"' in result[2]
        assert "postgres://user:pass@host" not in result[2]

    def test_sanitize_empty_errors(self):
        """Test sanitization with empty error list."""
        errors = []

        result = _sanitize_validation_errors(errors)

        assert result == []


@pytest.mark.unit
class TestGetConfigurationStatus:
    """Test cases for get_configuration_status function."""

    def test_get_status_success_all_healthy(self):
        """Test getting configuration status when everything is healthy."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"
        settings.version = "1.0.0"
        settings.debug = False
        settings.api_host = "localhost"
        settings.api_port = 8000

        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health._count_configured_secrets", return_value=3),
            patch("src.config.health._determine_config_source", return_value="env_vars"),
            patch("src.config.health.validate_configuration_on_startup"),
        ):
            status = get_configuration_status(settings)

            assert status.environment == "test"
            assert status.version == "1.0.0"
            assert status.debug is False
            assert status.config_loaded is True
            assert status.encryption_enabled is True
            assert status.config_source == "env_vars"
            assert status.validation_status == "passed"
            assert status.validation_errors == []
            assert status.secrets_configured == 3
            assert status.api_host == "localhost"
            assert status.api_port == 8000
            assert status.config_healthy is True

    def test_get_status_validation_error(self):
        """Test getting configuration status with validation errors."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"
        settings.version = "1.0.0"
        settings.debug = False
        settings.api_host = "localhost"
        settings.api_port = 8000

        # Mock validation error
        validation_error = ConfigurationValidationError(
            "Validation failed",
            ["Field 'password' invalid", "Missing required field"],
        )

        with (
            patch("src.config.health.validate_encryption_available", return_value=False),
            patch("src.config.health._count_configured_secrets", return_value=0),
            patch("src.config.health._determine_config_source", return_value="defaults"),
            patch("src.config.health.validate_configuration_on_startup", side_effect=validation_error),
            patch("src.config.health._sanitize_validation_errors", return_value=["Sanitized error"]),
        ):
            status = get_configuration_status(settings)

            assert status.validation_status == "failed"
            assert status.validation_errors == ["Sanitized error"]
            assert status.config_healthy is False

    def test_get_status_value_error(self):
        """Test getting configuration status with ValueError."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"
        settings.version = "1.0.0"
        settings.debug = False
        settings.api_host = "localhost"
        settings.api_port = 8000

        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health._count_configured_secrets", return_value=0),
            patch("src.config.health._determine_config_source", return_value="defaults"),
            patch("src.config.health.validate_configuration_on_startup", side_effect=ValueError("Format error")),
        ):
            status = get_configuration_status(settings)

            assert status.validation_status == "failed"
            assert status.validation_errors == ["Configuration format error"]
            assert status.config_healthy is False

    def test_get_status_type_error(self):
        """Test getting configuration status with TypeError."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"
        settings.version = "1.0.0"
        settings.debug = False
        settings.api_host = "localhost"
        settings.api_port = 8000

        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health._count_configured_secrets", return_value=0),
            patch("src.config.health._determine_config_source", return_value="defaults"),
            patch("src.config.health.validate_configuration_on_startup", side_effect=TypeError("Type error")),
        ):
            status = get_configuration_status(settings)

            assert status.validation_status == "failed"
            assert status.validation_errors == ["Configuration format error"]

    def test_get_status_attribute_error(self):
        """Test getting configuration status with AttributeError."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"
        settings.version = "1.0.0"
        settings.debug = False
        settings.api_host = "localhost"
        settings.api_port = 8000

        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health._count_configured_secrets", return_value=0),
            patch("src.config.health._determine_config_source", return_value="defaults"),
            patch("src.config.health.validate_configuration_on_startup", side_effect=AttributeError("Attribute error")),
        ):
            status = get_configuration_status(settings)

            assert status.validation_status == "failed"
            assert status.validation_errors == ["Configuration format error"]

    def test_get_status_unexpected_exception(self):
        """Test getting configuration status with unexpected exception."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "test"
        settings.version = "1.0.0"
        settings.debug = False
        settings.api_host = "localhost"
        settings.api_port = 8000

        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health._count_configured_secrets", return_value=0),
            patch("src.config.health._determine_config_source", return_value="defaults"),
            patch("src.config.health.validate_configuration_on_startup", side_effect=RuntimeError("Unexpected error")),
            pytest.raises(RuntimeError),
        ):
            get_configuration_status(settings)


@pytest.mark.unit
class TestGetConfigurationHealthSummary:
    """Test cases for get_configuration_health_summary function."""

    def test_get_health_summary_success(self):
        """Test getting health summary successfully."""
        mock_settings = Mock(spec=ApplicationSettings)
        mock_status = Mock()
        mock_status.config_healthy = True
        mock_status.environment = "test"
        mock_status.version = "1.0.0"
        mock_status.config_loaded = True
        mock_status.encryption_enabled = True
        mock_status.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        with (
            patch("src.config.health.get_settings", return_value=mock_settings),
            patch("src.config.health.get_configuration_status", return_value=mock_status),
        ):
            summary = get_configuration_health_summary()

            assert summary["healthy"] is True
            assert summary["environment"] == "test"
            assert summary["version"] == "1.0.0"
            assert summary["config_loaded"] is True
            assert summary["encryption_available"] is True
            assert "timestamp" in summary
            assert "2024-01-01T12:00:00" in summary["timestamp"]

    def test_get_health_summary_unhealthy(self):
        """Test getting health summary when unhealthy."""
        mock_settings = Mock(spec=ApplicationSettings)
        mock_status = Mock()
        mock_status.config_healthy = False
        mock_status.environment = "test"
        mock_status.version = "1.0.0"
        mock_status.config_loaded = False
        mock_status.encryption_enabled = False
        mock_status.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        with (
            patch("src.config.health.get_settings", return_value=mock_settings),
            patch("src.config.health.get_configuration_status", return_value=mock_status),
        ):
            summary = get_configuration_health_summary()

            assert summary["healthy"] is False
            assert summary["config_loaded"] is False
            assert summary["encryption_available"] is False

    def test_get_health_summary_value_error(self):
        """Test getting health summary with ValueError."""
        with patch("src.config.health.get_settings", side_effect=ValueError("Settings error")):
            summary = get_configuration_health_summary()

            assert summary["healthy"] is False
            assert summary["error"] == "Configuration health check failed"
            assert "timestamp" in summary

    def test_get_health_summary_type_error(self):
        """Test getting health summary with TypeError."""
        with patch("src.config.health.get_settings", side_effect=TypeError("Type error")):
            summary = get_configuration_health_summary()

            assert summary["healthy"] is False
            assert summary["error"] == "Configuration health check failed"

    def test_get_health_summary_attribute_error(self):
        """Test getting health summary with AttributeError."""
        with patch("src.config.health.get_settings", side_effect=AttributeError("Attribute error")):
            summary = get_configuration_health_summary()

            assert summary["healthy"] is False
            assert summary["error"] == "Configuration health check failed"

    def test_get_health_summary_unexpected_exception(self):
        """Test getting health summary with unexpected exception."""
        with patch("src.config.health.get_settings", side_effect=RuntimeError("Unexpected error")):
            summary = get_configuration_health_summary()

            assert summary["healthy"] is False
            assert summary["error"] == "Configuration health check failed"


@pytest.mark.unit
class TestGetMCPConfigurationHealth:
    """Test cases for get_mcp_configuration_health function."""

    @pytest.mark.asyncio
    async def test_mcp_health_success(self):
        """Test successful MCP health check."""
        # Mock MCP components
        mock_config_manager = Mock()
        mock_mcp_client = Mock()
        mock_parallel_executor = Mock()

        # Mock health responses
        config_health = {"configuration_valid": True, "status": "ok"}
        client_health = {"overall_status": "healthy", "connections": 3}
        executor_health = {"status": "healthy", "active_tasks": 2}

        mock_config_manager.get_health_status.return_value = config_health
        mock_mcp_client.health_check = AsyncMock(return_value=client_health)
        mock_parallel_executor.health_check = AsyncMock(return_value=executor_health)

        with (
            patch("src.config.health.MCPConfigurationManager", return_value=mock_config_manager),
            patch("src.config.health.MCPClient", return_value=mock_mcp_client),
            patch("src.config.health.ParallelSubagentExecutor", return_value=mock_parallel_executor),
        ):
            result = await get_mcp_configuration_health()

            assert result["healthy"] is True
            assert result["mcp_configuration"] == config_health
            assert result["mcp_client"] == client_health
            assert result["parallel_executor"] == executor_health
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_mcp_health_config_invalid(self):
        """Test MCP health check with invalid configuration."""
        mock_config_manager = Mock()
        mock_mcp_client = Mock()
        mock_parallel_executor = Mock()

        config_health = {"configuration_valid": False, "status": "error"}
        client_health = {"overall_status": "healthy", "connections": 3}
        executor_health = {"status": "healthy", "active_tasks": 2}

        mock_config_manager.get_health_status.return_value = config_health
        mock_mcp_client.health_check = AsyncMock(return_value=client_health)
        mock_parallel_executor.health_check = AsyncMock(return_value=executor_health)

        with (
            patch("src.config.health.MCPConfigurationManager", return_value=mock_config_manager),
            patch("src.config.health.MCPClient", return_value=mock_mcp_client),
            patch("src.config.health.ParallelSubagentExecutor", return_value=mock_parallel_executor),
        ):
            result = await get_mcp_configuration_health()

            assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_mcp_health_client_unhealthy(self):
        """Test MCP health check with unhealthy client."""
        mock_config_manager = Mock()
        mock_mcp_client = Mock()
        mock_parallel_executor = Mock()

        config_health = {"configuration_valid": True, "status": "ok"}
        client_health = {"overall_status": "error", "error": "Connection failed"}
        executor_health = {"status": "healthy", "active_tasks": 2}

        mock_config_manager.get_health_status.return_value = config_health
        mock_mcp_client.health_check = AsyncMock(return_value=client_health)
        mock_parallel_executor.health_check = AsyncMock(return_value=executor_health)

        with (
            patch("src.config.health.MCPConfigurationManager", return_value=mock_config_manager),
            patch("src.config.health.MCPClient", return_value=mock_mcp_client),
            patch("src.config.health.ParallelSubagentExecutor", return_value=mock_parallel_executor),
        ):
            result = await get_mcp_configuration_health()

            assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_mcp_health_executor_unhealthy(self):
        """Test MCP health check with unhealthy executor."""
        mock_config_manager = Mock()
        mock_mcp_client = Mock()
        mock_parallel_executor = Mock()

        config_health = {"configuration_valid": True, "status": "ok"}
        client_health = {"overall_status": "healthy", "connections": 3}
        executor_health = {"status": "error", "error": "Task execution failed"}

        mock_config_manager.get_health_status.return_value = config_health
        mock_mcp_client.health_check = AsyncMock(return_value=client_health)
        mock_parallel_executor.health_check = AsyncMock(return_value=executor_health)

        with (
            patch("src.config.health.MCPConfigurationManager", return_value=mock_config_manager),
            patch("src.config.health.MCPClient", return_value=mock_mcp_client),
            patch("src.config.health.ParallelSubagentExecutor", return_value=mock_parallel_executor),
        ):
            result = await get_mcp_configuration_health()

            assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_mcp_health_import_error(self):
        """Test MCP health check with import error (components not available)."""
        with (
            patch("src.config.health.MCPClient", None),
            patch("src.config.health.MCPConfigurationManager", None),
            patch("src.config.health.ParallelSubagentExecutor", None),
        ):
            result = await get_mcp_configuration_health()

            assert result["healthy"] is False
            assert "MCP integration not available" in result["error"]
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_mcp_health_runtime_error(self):
        """Test MCP health check with runtime error."""
        # Create mock classes to ensure they're not None
        Mock()
        mock_mcp_client = Mock()
        mock_parallel_executor = Mock()

        # First ensure the imports are available (not None)
        with (
            patch("src.config.health.MCPClient", mock_mcp_client),
            patch("src.config.health.ParallelSubagentExecutor", mock_parallel_executor),
            patch(
                "src.config.health.MCPConfigurationManager",
                side_effect=RuntimeError("MCP initialization failed"),
            ),
        ):
            result = await get_mcp_configuration_health()

            assert result["healthy"] is False
            assert "MCP health check failed" in result["error"]
            assert "timestamp" in result


@pytest.mark.unit
class TestIntegrationScenarios:
    """Integration test cases for complete health check workflows."""

    @pytest.mark.asyncio
    async def test_complete_health_check_workflow_healthy(self):
        """Test complete health check workflow when all systems are healthy."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "production"
        settings.version = "2.0.0"
        settings.debug = False
        settings.api_host = "api.example.com"
        settings.api_port = 443

        checker = HealthChecker(settings)

        # Mock all dependencies as healthy
        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health._count_configured_secrets", return_value=5),
            patch("src.config.health._determine_config_source", return_value="env_vars"),
            patch("src.config.health.validate_configuration_on_startup"),
            patch("src.config.health.get_mcp_configuration_health", return_value={"healthy": True}),
        ):
            result = await checker.check_health()

            assert result["healthy"] is True
            assert result["configuration"]["environment"] == "production"
            assert result["configuration"]["config_healthy"] is True
            assert result["mcp"]["healthy"] is True
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_complete_health_check_workflow_unhealthy(self):
        """Test complete health check workflow with system issues."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "production"
        settings.version = "2.0.0"
        settings.debug = False
        settings.api_host = "api.example.com"
        settings.api_port = 443

        checker = HealthChecker(settings)

        # Mock configuration as unhealthy
        validation_error = ConfigurationValidationError(
            "Critical validation failed",
            ["Database connection failed", "Missing API key"],
        )

        with (
            patch("src.config.health.validate_encryption_available", return_value=False),
            patch("src.config.health._count_configured_secrets", return_value=0),
            patch("src.config.health._determine_config_source", return_value="defaults"),
            patch("src.config.health.validate_configuration_on_startup", side_effect=validation_error),
            patch("src.config.health._sanitize_validation_errors", return_value=["Sanitized errors"]),
            patch(
                "src.config.health.get_mcp_configuration_health",
                return_value={"healthy": False},
            ),
        ):
            result = await checker.check_health()

            assert result["healthy"] is False
            assert result["configuration"]["config_healthy"] is False
            assert result["configuration"]["validation_status"] == "failed"
            assert result["mcp"]["healthy"] is False

    def test_configuration_status_model_integration(self):
        """Test integration between configuration status and model serialization."""
        settings = Mock(spec=ApplicationSettings)
        settings.environment = "staging"
        settings.version = "1.5.0"
        settings.debug = True
        settings.api_host = "staging.example.com"
        settings.api_port = 8080

        with (
            patch("src.config.health.validate_encryption_available", return_value=True),
            patch("src.config.health._count_configured_secrets", return_value=2),
            patch("src.config.health._determine_config_source", return_value="env_files"),
            patch("src.config.health.validate_configuration_on_startup"),
        ):
            status = get_configuration_status(settings)

            # Test model properties
            assert status.config_healthy is True
            assert status.environment == "staging"
            assert status.encryption_enabled is True
            assert status.secrets_configured == 2

            # Test serialization
            data = status.model_dump()
            assert data["config_healthy"] is True
            assert "timestamp" in data

            # Test health summary integration
            with (
                patch("src.config.health.get_settings", return_value=settings),
                patch("src.config.health.get_configuration_status", return_value=status),
            ):
                summary = get_configuration_health_summary()

                assert summary["healthy"] is True
                assert summary["environment"] == "staging"
                assert summary["version"] == "1.5.0"
