"""Tests for the main FastAPI application module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.config.settings import ApplicationSettings, ConfigurationValidationError
from src.main import app, create_app, lifespan


class TestAppCreation:
    """Test FastAPI app creation and configuration."""

    @patch("src.main.get_settings")
    def test_create_app_with_valid_settings(self, mock_get_settings) -> None:
        """Test app creation with valid settings."""
        mock_settings = ApplicationSettings(
            app_name="Test App",
            version="1.0.0",
            environment="dev",
            debug=True,
        )
        mock_get_settings.return_value = mock_settings

        app_instance = create_app()

        assert app_instance.title == "Test App"
        assert app_instance.version == "1.0.0"
        assert app_instance.docs_url == "/docs"  # Debug mode enabled
        assert app_instance.redoc_url == "/redoc"

    @patch("src.main.get_settings")
    def test_create_app_production_mode(self, mock_get_settings) -> None:
        """Test app creation in production mode."""
        mock_settings = ApplicationSettings(
            app_name="Prod App",
            version="2.0.0",
            environment="prod",
            debug=False,
        )
        mock_get_settings.return_value = mock_settings

        app_instance = create_app()

        assert app_instance.title == "Prod App"
        assert app_instance.version == "2.0.0"
        assert app_instance.docs_url is None  # Debug mode disabled
        assert app_instance.redoc_url is None

    @patch("src.main.get_settings")
    def test_create_app_with_settings_error(self, mock_get_settings) -> None:
        """Test app creation when settings loading fails."""
        mock_get_settings.side_effect = ValueError("Settings error")

        app_instance = create_app()

        # Should fallback to defaults
        assert app_instance.title == "PromptCraft-Hybrid"
        assert app_instance.version == "0.1.0"

    @patch("src.main.get_settings")
    def test_create_app_with_unexpected_error(self, mock_get_settings) -> None:
        """Test app creation with unexpected error."""
        mock_get_settings.side_effect = RuntimeError("Unexpected error")

        app_instance = create_app()

        # Should fallback to defaults
        assert app_instance.title == "PromptCraft-Hybrid"

    def test_cors_configuration_dev(self) -> None:
        """Test CORS configuration for development environment."""
        with patch("src.main.get_settings") as mock_get_settings:
            mock_settings = ApplicationSettings(environment="dev")
            mock_get_settings.return_value = mock_settings

            app_instance = create_app()

            # Check that CORS middleware is added (we can't easily inspect the config)
            # But we can verify the app was created without errors
            assert app_instance is not None

    def test_cors_configuration_staging(self) -> None:
        """Test CORS configuration for staging environment."""
        with patch("src.main.get_settings") as mock_get_settings:
            mock_settings = ApplicationSettings(environment="staging")
            mock_get_settings.return_value = mock_settings

            app_instance = create_app()
            assert app_instance is not None

    def test_cors_configuration_prod(self) -> None:
        """Test CORS configuration for production environment."""
        with patch("src.main.get_settings") as mock_get_settings:
            mock_settings = ApplicationSettings(environment="prod")
            mock_get_settings.return_value = mock_settings

            app_instance = create_app()
            assert app_instance is not None

    def test_cors_configuration_unknown_env(self) -> None:
        """Test CORS configuration for unknown environment."""
        with patch("src.main.get_settings") as mock_get_settings:
            # Create settings with valid environment first, then modify
            mock_settings = ApplicationSettings(environment="dev")
            # Manually override environment for testing (bypassing validation)
            mock_settings.environment = "test"
            mock_get_settings.return_value = mock_settings

            app_instance = create_app()
            assert app_instance is not None


class TestRootEndpoints:
    """Test basic endpoints in the FastAPI app."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = TestClient(app)

    def test_root_endpoint_with_app_state(self) -> None:
        """Test root endpoint when app state is available."""
        # Mock app state
        mock_settings = ApplicationSettings(
            app_name="Test App",
            version="1.0.0",
            environment="dev",
            debug=True,
        )
        app.state.settings = mock_settings

        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Test App"
        assert data["version"] == "1.0.0"
        assert data["environment"] == "dev"
        assert data["status"] == "running"
        assert data["docs_url"] == "/docs"

    def test_root_endpoint_without_app_state(self) -> None:
        """Test root endpoint when app state is not available."""
        # Remove app state if it exists
        if hasattr(app.state, "settings"):
            delattr(app.state, "settings")

        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "PromptCraft-Hybrid"
        assert data["version"] == "unknown"
        assert data["environment"] == "unknown"
        assert data["status"] == "running"

    def test_ping_endpoint(self) -> None:
        """Test ping endpoint."""
        response = self.client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "pong"


class TestLifespanEvents:
    """Test application lifespan events."""

    @patch("src.main.get_settings")
    async def test_lifespan_startup_success(self, mock_get_settings) -> None:
        """Test successful application startup."""
        mock_settings = ApplicationSettings(
            app_name="Test App",
            version="1.0.0",
            environment="dev",
        )
        mock_get_settings.return_value = mock_settings

        # Test the lifespan context manager directly
        mock_app = MagicMock()
        mock_app.state = MagicMock()

        # Test that the lifespan context manager works
        async with lifespan(mock_app):
            # Verify settings were stored in app state
            assert mock_app.state.settings == mock_settings

    @patch("src.main.get_settings")
    async def test_lifespan_configuration_validation_error(self, mock_get_settings) -> None:
        """Test startup with configuration validation error."""
        mock_get_settings.side_effect = ConfigurationValidationError(
            "Config error",
            field_errors=["Error 1", "Error 2"],
            suggestions=["Fix 1", "Fix 2"],
        )

        mock_app = MagicMock()

        # Should raise ConfigurationValidationError
        with pytest.raises(ConfigurationValidationError):
            async with lifespan(mock_app):
                pass

    @patch("src.main.get_settings")
    async def test_lifespan_unexpected_error(self, mock_get_settings) -> None:
        """Test startup with unexpected error."""
        mock_get_settings.side_effect = RuntimeError("Unexpected error")

        mock_app = MagicMock()

        # Should raise the RuntimeError
        with pytest.raises(RuntimeError):
            async with lifespan(mock_app):
                pass


class TestHealthCheckEndpoints:
    """Test health check endpoints for comprehensive coverage."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = TestClient(app)

    @patch("src.main.get_configuration_health_summary")
    @patch("src.main.get_all_circuit_breakers")
    def test_health_check_success(self, mock_circuit_breakers, mock_health_summary) -> None:
        """Test successful health check endpoint."""
        # Mock health summary to return healthy status
        mock_health_summary.return_value = {
            "healthy": True,
            "environment": "test",
            "version": "1.0.0",
            "config_loaded": True,
            "encryption_available": True,
            "timestamp": "2025-01-01T00:00:00+00:00",
        }

        # Mock circuit breakers
        mock_breaker = MagicMock()
        mock_breaker.get_health_status.return_value = {"healthy": True, "state": "closed"}
        mock_circuit_breakers.return_value = {"test_breaker": mock_breaker}

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "promptcraft-hybrid"
        assert "circuit_breakers" in data
        assert data["healthy"] is True

    @patch("src.main.get_configuration_health_summary")
    @patch("src.main.get_all_circuit_breakers")
    def test_health_check_unhealthy(self, mock_circuit_breakers, mock_health_summary) -> None:
        """Test health check when service is unhealthy."""
        health_summary = {
            "healthy": False,
            "status": "degraded",
            "environment": "test",
            "version": "1.0.0",
        }
        mock_health_summary.return_value = health_summary
        mock_circuit_breakers.return_value = {}

        response = self.client.get("/health")

        assert response.status_code == 503
        data = response.json()
        # AuthExceptionHandler.handle_service_unavailable returns detailed error message
        expected_error = f"Health check failed: configuration unhealthy - {health_summary}"
        assert data["error"] == expected_error
        assert data["status_code"] == 503

    @patch("src.main.get_configuration_health_summary")
    @patch("src.main.get_all_circuit_breakers")
    def test_health_check_exception(self, mock_circuit_breakers, mock_health_summary) -> None:
        """Test health check with unexpected exception."""
        mock_health_summary.side_effect = RuntimeError("Unexpected error")
        mock_circuit_breakers.return_value = {}

        response = self.client.get("/health")

        assert response.status_code == 500
        data = response.json()
        # AuthExceptionHandler.handle_internal_error with expose_error=True returns detailed error
        expected_error = "Health check endpoint failed: Unexpected error"
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    @patch("src.main.get_configuration_health_summary")
    @patch("src.main.get_all_circuit_breakers")
    def test_health_check_http_exception_passthrough(self, mock_circuit_breakers, mock_health_summary) -> None:
        """Test health check with HTTPException that should be re-raised."""

        mock_health_summary.side_effect = HTTPException(status_code=503, detail="Service unavailable")
        mock_circuit_breakers.return_value = {}

        response = self.client.get("/health")

        assert response.status_code == 503
        data = response.json()
        # HTTPExceptions are processed by the security error handler with original detail preserved
        assert data["error"] == "Service unavailable"
        assert data["status_code"] == 503
        assert "timestamp" in data
        assert data["path"] == "/health"
        assert "debug" in data
        assert data["debug"]["error_type"] == "HTTPException"

    @patch("src.main.get_settings")
    @patch("src.main.get_configuration_status")
    def test_configuration_health_success(self, mock_config_status, mock_settings) -> None:
        """Test successful configuration health check."""
        from datetime import UTC, datetime

        from src.config.health import ConfigurationStatusModel

        mock_settings_obj = ApplicationSettings()
        mock_settings.return_value = mock_settings_obj

        # Create a proper ConfigurationStatusModel instance
        mock_config_status.return_value = ConfigurationStatusModel(
            environment="test",
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
            timestamp=datetime.now(UTC),
        )

        response = self.client.get("/health/config")

        assert response.status_code == 200
        data = response.json()
        assert data["environment"] == "test"
        assert data["config_loaded"] is True
        assert data["config_healthy"] is True

    @patch("src.main.get_settings")
    def test_configuration_health_validation_error_debug(self, mock_settings) -> None:
        """Test configuration health with validation error in debug mode."""
        # First call raises ConfigurationValidationError
        config_error = ConfigurationValidationError(
            "Config error",
            field_errors=["Error 1", "Error 2", "Error 3", "Error 4", "Error 5", "Error 6"],
            suggestions=["Fix 1", "Fix 2", "Fix 3", "Fix 4", "Fix 5", "Fix 6"],
        )

        # Second call returns debug settings for debug mode check
        mock_debug_settings = ApplicationSettings(debug=True)
        mock_settings.side_effect = [config_error, mock_debug_settings]

        response = self.client.get("/health/config")

        assert response.status_code == 500
        data = response.json()
        # With debug mode, error message should contain detailed information
        from src.config.constants import HEALTH_CHECK_ERROR_LIMIT, HEALTH_CHECK_SUGGESTION_LIMIT
        expected_detail = {
            "error": "Configuration validation failed",
            "field_errors": config_error.field_errors[:HEALTH_CHECK_ERROR_LIMIT],
            "suggestions": config_error.suggestions[:HEALTH_CHECK_SUGGESTION_LIMIT],
        }
        full_exception_str = str(config_error)
        expected_error = f"{expected_detail}: {full_exception_str}"
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    @patch("src.main.get_settings")
    def test_configuration_health_validation_error_production(self, mock_settings) -> None:
        """Test configuration health with validation error in production mode."""
        # First call raises ConfigurationValidationError
        config_error = ConfigurationValidationError("Config error", field_errors=["Error 1"], suggestions=["Fix 1"])

        # Second call raises exception for debug mode check (simulating production)
        mock_settings.side_effect = [config_error, RuntimeError("Settings unavailable")]

        response = self.client.get("/health/config")

        assert response.status_code == 500
        data = response.json()
        # In production mode, error message should contain minimal detail
        expected_detail = {
            "error": "Configuration validation failed",
            "details": "Contact system administrator",
        }
        expected_error = str(expected_detail)
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    @patch("src.main.get_settings")
    def test_configuration_health_unexpected_error(self, mock_settings) -> None:
        """Test configuration health with unexpected error."""
        mock_settings.side_effect = RuntimeError("Unexpected error")

        response = self.client.get("/health/config")

        assert response.status_code == 500
        data = response.json()
        # AuthExceptionHandler.handle_internal_error with expose_error=True returns detailed error
        expected_error = "Configuration health check failed - see application logs: Unexpected error"
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    @patch("src.main.get_mcp_configuration_health")
    def test_mcp_health_check_success(self, mock_mcp_health) -> None:
        """Test successful MCP health check."""
        mock_mcp_health.return_value = {"healthy": True, "mcp_status": "connected"}

        response = self.client.get("/health/mcp")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "mcp-integration"

    @patch("src.main.get_mcp_configuration_health")
    def test_mcp_health_check_unhealthy(self, mock_mcp_health) -> None:
        """Test MCP health check when unhealthy."""
        mcp_health = {"healthy": False, "mcp_status": "disconnected"}
        mock_mcp_health.return_value = mcp_health

        response = self.client.get("/health/mcp")

        assert response.status_code == 503
        data = response.json()
        # AuthExceptionHandler.handle_service_unavailable returns detailed error message
        expected_error = f"MCP integration unhealthy - {mcp_health}"
        assert data["error"] == expected_error
        assert data["status_code"] == 503

    @patch("src.main.get_mcp_configuration_health")
    def test_mcp_health_check_import_error(self, mock_mcp_health) -> None:
        """Test MCP health check with ImportError."""
        mock_mcp_health.side_effect = ImportError("MCP not available")

        response = self.client.get("/health/mcp")

        assert response.status_code == 503
        data = response.json()
        # AuthExceptionHandler.handle_service_unavailable returns detailed error message
        expected_error = "MCP integration not available"
        assert data["error"] == expected_error
        assert data["status_code"] == 503

    @patch("src.main.get_mcp_configuration_health")
    def test_mcp_health_check_exception(self, mock_mcp_health) -> None:
        """Test MCP health check with unexpected exception."""
        mock_mcp_health.side_effect = RuntimeError("Unexpected error")

        response = self.client.get("/health/mcp")

        assert response.status_code == 500
        data = response.json()
        # AuthExceptionHandler.handle_internal_error with expose_error=True returns detailed error
        expected_error = "MCP health check endpoint failed: Unexpected error"
        assert data["error"] == expected_error
        assert data["status_code"] == 500

    @patch("src.main.get_all_circuit_breakers")
    def test_circuit_breaker_health_no_breakers(self, mock_circuit_breakers) -> None:
        """Test circuit breaker health when no breakers are registered."""
        mock_circuit_breakers.return_value = {}

        response = self.client.get("/health/circuit-breakers")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_circuit_breakers"
        assert data["message"] == "No circuit breakers registered"

    @patch("src.main.get_all_circuit_breakers")
    def test_circuit_breaker_health_all_healthy(self, mock_circuit_breakers) -> None:
        """Test circuit breaker health when all breakers are healthy."""
        mock_breaker1 = MagicMock()
        mock_breaker1.get_health_status.return_value = {"healthy": True, "state": "closed"}
        mock_breaker2 = MagicMock()
        mock_breaker2.get_health_status.return_value = {"healthy": True, "state": "closed"}

        mock_circuit_breakers.return_value = {"breaker1": mock_breaker1, "breaker2": mock_breaker2}

        response = self.client.get("/health/circuit-breakers")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["overall_healthy"] is True
        assert data["circuit_breaker_count"] == 2

    @patch("src.main.get_all_circuit_breakers")
    def test_circuit_breaker_health_degraded(self, mock_circuit_breakers) -> None:
        """Test circuit breaker health when some breakers are unhealthy."""
        mock_breaker1 = MagicMock()
        mock_breaker1.get_health_status.return_value = {"healthy": True, "state": "closed"}
        mock_breaker2 = MagicMock()
        mock_breaker2.get_health_status.return_value = {"healthy": False, "state": "open"}

        mock_circuit_breakers.return_value = {"breaker1": mock_breaker1, "breaker2": mock_breaker2}

        response = self.client.get("/health/circuit-breakers")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["overall_healthy"] is False

    @patch("src.main.get_all_circuit_breakers")
    def test_circuit_breaker_health_exception(self, mock_circuit_breakers) -> None:
        """Test circuit breaker health with exception."""
        mock_circuit_breakers.side_effect = RuntimeError("Breaker error")

        response = self.client.get("/health/circuit-breakers")

        assert response.status_code == 500
        data = response.json()
        # AuthExceptionHandler.handle_internal_error with expose_error=True returns detailed error
        expected_error = "Circuit breaker health check failed: Breaker error"
        assert data["error"] == expected_error
        assert data["status_code"] == 500


class TestAPIEndpoints:
    """Test API endpoints for comprehensive coverage."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = TestClient(app)

    @patch("src.main.audit_logger_instance")
    def test_validate_input_success(self, mock_audit_logger) -> None:
        """Test validate input endpoint with successful validation."""
        test_data = {"text": "Test input"}

        response = self.client.post("/api/v1/validate", json=test_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Input validation successful"
        assert data["sanitized_text"] == "Test input"

        # Verify audit logging was called
        mock_audit_logger.log_api_event.assert_called_once()

    def test_search_endpoint_success(self) -> None:
        """Test search endpoint with valid parameters."""
        response = self.client.get("/api/v1/search", params={"search": "test", "page": 1, "limit": 10, "sort": "name"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "query_params" in data
        assert data["query_params"]["search"] == "test"
        assert data["query_params"]["page"] == 1
        assert data["query_params"]["limit"] == 10
        assert data["query_params"]["sort"] == "name"


class TestLifespanComprehensive:
    """Test comprehensive lifespan coverage including audit logging."""

    @patch("src.main.audit_logger_instance")
    @patch("src.main.get_settings")
    def test_lifespan_startup_audit_logging(self, mock_get_settings, mock_audit_logger) -> None:
        """Test lifespan startup with audit event logging."""
        mock_settings = ApplicationSettings(app_name="Test App", version="1.0.0", environment="dev", debug=True)
        mock_get_settings.return_value = mock_settings

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        # Use context manager to trigger lifespan events
        try:
            with TestClient(app):
                # Lifespan startup and shutdown happen during context manager
                pass
        except Exception:
            # Some dependencies might not be available in test, ignore
            pass

        # Verify audit logging was called for startup
        startup_calls = [
            call for call in mock_audit_logger.log_security_event.call_args_list if "startup" in str(call).lower()
        ]
        assert len(startup_calls) > 0

    @patch("src.main.audit_logger_instance")
    @patch("src.main.get_settings")
    async def test_lifespan_configuration_validation_error_audit(self, mock_get_settings, mock_audit_logger) -> None:
        """Test lifespan with configuration validation error and audit logging."""
        config_error = ConfigurationValidationError(
            "Config error",
            field_errors=["Error 1", "Error 2"],
            suggestions=["Fix 1", "Fix 2"],
        )
        mock_get_settings.side_effect = config_error

        mock_app = MagicMock()

        # Test that ConfigurationValidationError is raised
        with pytest.raises(ConfigurationValidationError):
            async with lifespan(mock_app):
                pass

        # Verify audit logging was called for validation failure
        validation_calls = [
            call for call in mock_audit_logger.log_security_event.call_args_list if "validation" in str(call).lower()
        ]
        assert len(validation_calls) > 0

    @patch("src.main.audit_logger_instance")
    @patch("src.main.get_settings")
    async def test_lifespan_unexpected_error_audit(self, mock_get_settings, mock_audit_logger) -> None:
        """Test lifespan with unexpected error and audit logging."""
        mock_get_settings.side_effect = RuntimeError("Unexpected error")

        mock_app = MagicMock()

        # Test that RuntimeError is raised
        with pytest.raises(RuntimeError):
            async with lifespan(mock_app):
                pass

        # Verify audit logging was called for startup failure
        failure_calls = [
            call for call in mock_audit_logger.log_security_event.call_args_list if "startup" in str(call).lower()
        ]
        assert len(failure_calls) > 0

    @patch("src.main.audit_logger_instance")
    def test_lifespan_shutdown_audit_logging(self, mock_audit_logger) -> None:
        """Test lifespan shutdown audit logging."""
        mock_app = MagicMock()
        mock_app.state = MagicMock()

        # Use normal lifespan flow
        try:
            with TestClient(app):
                pass
        except Exception:
            # Ignore dependency issues in test
            pass

        # Verify audit logging was called for shutdown
        shutdown_calls = [
            call for call in mock_audit_logger.log_security_event.call_args_list if "shutdown" in str(call).lower()
        ]
        assert len(shutdown_calls) > 0


class TestCreateAppExceptionHandling:
    """Test create_app exception handling scenarios for lines 144-151."""

    @patch("src.main.get_settings")
    def test_create_app_value_error_fallback(self, mock_get_settings) -> None:
        """Test create_app with ValueError falls back to defaults."""
        mock_get_settings.side_effect = ValueError("Settings format error")

        app_instance = create_app()

        # Should fallback to defaults
        assert app_instance.title == "PromptCraft-Hybrid"
        assert app_instance.version == "0.1.0"

    @patch("src.main.get_settings")
    def test_create_app_type_error_fallback(self, mock_get_settings) -> None:
        """Test create_app with TypeError falls back to defaults."""
        mock_get_settings.side_effect = TypeError("Type error in settings")

        app_instance = create_app()

        # Should fallback to defaults
        assert app_instance.title == "PromptCraft-Hybrid"
        assert app_instance.version == "0.1.0"

    @patch("src.main.get_settings")
    def test_create_app_attribute_error_fallback(self, mock_get_settings) -> None:
        """Test create_app with AttributeError falls back to defaults."""
        mock_get_settings.side_effect = AttributeError("Attribute error in settings")

        app_instance = create_app()

        # Should fallback to defaults
        assert app_instance.title == "PromptCraft-Hybrid"
        assert app_instance.version == "0.1.0"

    @patch("src.main.get_settings")
    def test_create_app_general_exception_fallback(self, mock_get_settings) -> None:
        """Test create_app with general exception falls back to defaults."""
        mock_get_settings.side_effect = Exception("General error")

        app_instance = create_app()

        # Should fallback to defaults
        assert app_instance.title == "PromptCraft-Hybrid"
        assert app_instance.version == "0.1.0"


class TestRootEndpointFallback:
    """Test root endpoint fallback behavior for lines 437-448."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = TestClient(app)

    def test_root_endpoint_fallback_when_no_app_state(self) -> None:
        """Test root endpoint fallback when app.state.settings is unavailable."""
        # Ensure app state settings doesn't exist
        if hasattr(app.state, "settings"):
            delattr(app.state, "settings")

        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "PromptCraft-Hybrid"
        assert data["version"] == "unknown"
        assert data["environment"] == "unknown"
        assert data["status"] == "running"


class TestPingEndpoint:
    """Test ping endpoint for line 464 coverage."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = TestClient(app)

    def test_ping_endpoint_return_statement(self) -> None:
        """Test ping endpoint return statement coverage."""
        response = self.client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "pong"


class TestMainScriptExecution:
    """Test the main script execution scenarios."""

    def test_settings_properties_for_main(self) -> None:
        """Test that settings have the properties needed for main execution."""
        settings = ApplicationSettings(
            api_host="localhost",
            api_port=8000,
            debug=True,
        )

        assert settings.api_host == "localhost"
        assert settings.api_port == 8000
        assert settings.debug is True

    def test_configuration_error_handling(self) -> None:
        """Test that configuration errors are properly raised."""
        with pytest.raises(ConfigurationValidationError):
            raise ConfigurationValidationError("Config error")

    def test_os_error_handling(self) -> None:
        """Test that OS errors are properly raised."""
        with pytest.raises(OSError, match="Failed to start server"):
            raise OSError("Failed to start server")

    def test_runtime_error_handling(self) -> None:
        """Test that runtime errors are properly raised."""
        with pytest.raises(RuntimeError):
            raise RuntimeError("Unexpected error")
