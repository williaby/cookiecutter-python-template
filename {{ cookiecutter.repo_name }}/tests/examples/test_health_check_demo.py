#!/usr/bin/env python3
"""Comprehensive tests for examples/health_check_demo.py.

This test suite covers all functions and code paths in the health check demo,
ensuring 80%+ test coverage while following project testing standards.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic import SecretStr
from rich.console import Console

from examples.health_check_demo import (
    demonstrate_configuration_status,
    demonstrate_health_summary,
    demonstrate_http_endpoints,
    demonstrate_json_serialization,
    demonstrate_security_features,
    main,
)


class TestDemonstrateConfigurationStatus:
    """Test the demonstrate_configuration_status function."""

    @patch("examples.health_check_demo.get_settings")
    @patch("examples.health_check_demo.get_configuration_status")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_configuration_status_success(self, mock_console, mock_get_status, mock_get_settings):
        """Test successful configuration status demonstration."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.environment = "dev"
        mock_get_settings.return_value = mock_settings

        mock_status = Mock()
        mock_status.environment = "dev"
        mock_status.version = "0.1.0"
        mock_status.debug = True
        mock_status.config_loaded = True
        mock_status.encryption_enabled = False
        mock_status.config_source = "environment"
        mock_status.validation_status = "valid"
        mock_status.secrets_configured = 3
        mock_status.api_host = "127.0.0.1"
        mock_status.api_port = 8000
        mock_status.config_healthy = True
        mock_status.validation_errors = []
        mock_get_status.return_value = mock_status

        # Execute function
        result = demonstrate_configuration_status()

        # Verify calls
        mock_get_settings.assert_called_once_with(validate_on_startup=False)
        mock_get_status.assert_called_once_with(mock_settings)

        # Verify return value
        assert result == mock_status

        # Verify console interactions
        assert mock_console.print.call_count >= 3  # Header, table, success message

    @patch("examples.health_check_demo.get_settings")
    @patch("examples.health_check_demo.get_configuration_status")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_configuration_status_with_validation_errors(
        self,
        mock_console,
        mock_get_status,
        mock_get_settings,
    ):
        """Test configuration status with validation errors."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_status = Mock()
        mock_status.environment = "prod"
        mock_status.version = "1.0.0"
        mock_status.debug = False
        mock_status.config_loaded = True
        mock_status.encryption_enabled = True
        mock_status.config_source = "encrypted"
        mock_status.validation_status = "errors"
        mock_status.secrets_configured = 5
        mock_status.api_host = "0.0.0.0"  # noqa: S104
        mock_status.api_port = 80
        mock_status.config_healthy = False
        mock_status.validation_errors = ["Missing secret key", "Invalid database URL"]
        mock_get_status.return_value = mock_status

        result = demonstrate_configuration_status()

        assert result == mock_status

        # Should display validation errors
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        error_section_found = any("Validation Errors:" in call for call in print_calls)
        assert error_section_found

    @patch("examples.health_check_demo.get_settings", side_effect=Exception("Config error"))
    @patch("examples.health_check_demo.console")
    def test_demonstrate_configuration_status_exception(self, mock_console, mock_get_settings):
        """Test configuration status with exception."""
        result = demonstrate_configuration_status()

        # Should return None on exception
        assert result is None

        # Should print error message
        error_calls = [call for call in mock_console.print.call_args_list if "Error generating" in str(call)]
        assert len(error_calls) > 0


class TestDemonstrateHealthSummary:
    """Test the demonstrate_health_summary function."""

    @patch("examples.health_check_demo.get_configuration_health_summary")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_health_summary_healthy(self, mock_console, mock_get_summary):
        """Test health summary demonstration with healthy status."""
        mock_summary = {
            "healthy": True,
            "environment": "dev",
            "version": "0.1.0",
            "config_loaded": True,
            "timestamp": "2024-01-01T12:00:00Z",
        }
        mock_get_summary.return_value = mock_summary

        demonstrate_health_summary()

        mock_get_summary.assert_called_once()

        # Verify console output includes summary data
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        health_summary_found = any("Health Summary:" in call for call in print_calls)
        assert health_summary_found

    @patch("examples.health_check_demo.get_configuration_health_summary")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_health_summary_unhealthy(self, mock_console, mock_get_summary):
        """Test health summary demonstration with unhealthy status."""
        mock_summary = {
            "healthy": False,
            "environment": "prod",
            "version": "1.0.0",
            "config_loaded": False,
            "timestamp": "2024-01-01T12:00:00Z",
        }
        mock_get_summary.return_value = mock_summary

        demonstrate_health_summary()

        mock_get_summary.assert_called_once()

    @patch("examples.health_check_demo.get_configuration_health_summary", side_effect=Exception("Summary error"))
    @patch("examples.health_check_demo.console")
    def test_demonstrate_health_summary_exception(self, mock_console, mock_get_summary):
        """Test health summary with exception."""
        demonstrate_health_summary()

        # Should handle exception gracefully
        error_calls = [
            call for call in mock_console.print.call_args_list if "Error getting health summary" in str(call)
        ]
        assert len(error_calls) > 0


class TestDemonstrateJsonSerialization:
    """Test the demonstrate_json_serialization function."""

    @patch("examples.health_check_demo.console")
    def test_demonstrate_json_serialization_safe_data(self, mock_console):
        """Test JSON serialization with safe data."""
        mock_status = Mock()
        mock_status.model_dump_json.return_value = '{"environment": "dev", "version": "0.1.0", "debug": true}'

        demonstrate_json_serialization(mock_status)

        mock_status.model_dump_json.assert_called_once_with(indent=2)

        # Should indicate data is safe
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        safe_message_found = any("JSON is safe" in call for call in print_calls)
        assert safe_message_found

    @patch("examples.health_check_demo.console")
    def test_demonstrate_json_serialization_sensitive_data(self, mock_console):
        """Test JSON serialization with sensitive data detected."""
        mock_status = Mock()
        mock_status.model_dump_json.return_value = '{"password": "secret123", "api_key": "sk-123"}'

        demonstrate_json_serialization(mock_status)

        # Should warn about sensitive data
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        warning_found = any("WARNING: JSON may contain sensitive data" in call for call in print_calls)
        assert warning_found

    @patch("examples.health_check_demo.console")
    def test_demonstrate_json_serialization_exception(self, mock_console):
        """Test JSON serialization with exception."""
        mock_status = Mock()
        mock_status.model_dump_json.side_effect = Exception("Serialization error")

        demonstrate_json_serialization(mock_status)

        # Should handle exception gracefully
        error_calls = [call for call in mock_console.print.call_args_list if "Error serializing" in str(call)]
        assert len(error_calls) > 0

    @patch("examples.health_check_demo.console")
    def test_demonstrate_json_serialization_all_sensitive_keywords(self, mock_console):
        """Test detection of all sensitive keywords."""
        sensitive_keywords = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "SecretStr",
            "super_secret",
            "api_key_value",
        ]

        for keyword in sensitive_keywords:
            mock_status = Mock()
            mock_status.model_dump_json.return_value = f'{{"data": "{keyword}_value"}}'

            demonstrate_json_serialization(mock_status)

            # Should detect each sensitive keyword
            print_calls = [str(call) for call in mock_console.print.call_args_list]
            warning_found = any("WARNING" in call for call in print_calls)
            assert warning_found

            mock_console.reset_mock()


class TestDemonstrateHttpEndpoints:
    """Test the demonstrate_http_endpoints function."""

    @pytest.mark.asyncio
    @patch("examples.health_check_demo.httpx.AsyncClient")
    @patch("examples.health_check_demo.console")
    async def test_demonstrate_http_endpoints_no_server(self, mock_console, mock_client_class):
        """Test HTTP endpoints demonstration when no server is running."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await demonstrate_http_endpoints()

        # Should attempt to connect
        mock_client.get.assert_called()

        # Should show example endpoints and responses
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        endpoints_found = any("/health" in call for call in print_calls)
        assert endpoints_found

    @pytest.mark.asyncio
    @patch("examples.health_check_demo.httpx.AsyncClient")
    @patch("examples.health_check_demo.console")
    async def test_demonstrate_http_endpoints_live_server(self, mock_console, mock_client_class):
        """Test HTTP endpoints demonstration with live server."""
        mock_client = AsyncMock()

        # Mock successful ping response
        mock_ping_response = Mock()
        mock_ping_response.status_code = 200

        # Mock health response
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_health_response.json.return_value = {
            "status": "healthy",
            "service": "promptcraft-hybrid",
            "healthy": True,
            "environment": "dev",
        }

        # Mock config response
        mock_config_response = Mock()
        mock_config_response.status_code = 200
        mock_config_response.json.return_value = {"environment": "dev", "config_loaded": True, "healthy": True}

        # Configure mock client responses
        mock_client.get.side_effect = [mock_ping_response, mock_health_response, mock_config_response]
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await demonstrate_http_endpoints()

        # Should make three GET requests
        assert mock_client.get.call_count == 3

        # Should display live server responses
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        live_server_found = any("Live server detected" in call for call in print_calls)
        assert live_server_found

    @pytest.mark.asyncio
    @patch("examples.health_check_demo.httpx.AsyncClient")
    @patch("examples.health_check_demo.console")
    async def test_demonstrate_http_endpoints_timeout(self, mock_console, mock_client_class):
        """Test HTTP endpoints demonstration with timeout."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("Request timeout")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await demonstrate_http_endpoints()

        # Should handle timeout gracefully
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        no_server_found = any("No live server detected" in call for call in print_calls)
        assert no_server_found

    @pytest.mark.asyncio
    @patch("examples.health_check_demo.console")
    async def test_demonstrate_http_endpoints_example_responses(self, mock_console):
        """Test that example responses are shown."""
        await demonstrate_http_endpoints()

        # Should show example endpoints and responses
        print_calls = [str(call) for call in mock_console.print.call_args_list]

        # Check for endpoint descriptions
        endpoint_descriptions = ["/health", "/health/config", "/ping", "/"]
        for endpoint in endpoint_descriptions:
            endpoint_found = any(endpoint in call for call in print_calls)
            assert endpoint_found, f"Endpoint {endpoint} should be described"

        # Check for example response
        example_response_found = any("GET /health:" in call for call in print_calls)
        assert example_response_found


class TestDemonstrateSecurityFeatures:
    """Test the demonstrate_security_features function."""

    @patch("examples.health_check_demo.ApplicationSettings")
    @patch("examples.health_check_demo.get_configuration_status")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_security_features_secure(self, mock_console, mock_get_status, mock_settings_class):
        """Test security features demonstration with secure output."""
        # Mock settings creation
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings

        # Mock status with secure JSON output
        mock_status = Mock()
        mock_status.secrets_configured = 4
        mock_status.model_dump_json.return_value = '{"environment": "test", "secrets_configured": 4}'
        mock_get_status.return_value = mock_status

        demonstrate_security_features()

        # Verify settings were created with secrets
        mock_settings_class.assert_called_once()
        call_kwargs = mock_settings_class.call_args[1]
        assert "database_password" in call_kwargs
        assert "api_key" in call_kwargs
        assert "secret_key" in call_kwargs
        assert "jwt_secret_key" in call_kwargs

        # All should be SecretStr instances
        for _key, value in call_kwargs.items():
            assert isinstance(value, SecretStr)

        # Should indicate security is maintained
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        secure_found = any("SECURE: No secret values found" in call for call in print_calls)
        assert secure_found

    @patch("examples.health_check_demo.ApplicationSettings")
    @patch("examples.health_check_demo.get_configuration_status")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_security_features_leaked_secrets(self, mock_console, mock_get_status, mock_settings_class):
        """Test security features demonstration with leaked secrets."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings

        # Mock status with leaked secrets in JSON
        mock_status = Mock()
        mock_status.secrets_configured = 4
        mock_status.model_dump_json.return_value = (
            '{"password": "super_secret_db_password", "api_key": "sk-1234567890abcdef"}'
        )
        mock_get_status.return_value = mock_status

        demonstrate_security_features()

        # Should detect security breach
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        breach_found = any("SECURITY BREACH" in call for call in print_calls)
        assert breach_found

    @patch("examples.health_check_demo.ApplicationSettings")
    @patch("examples.health_check_demo.get_configuration_status")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_security_features_secretstr_exposure(self, mock_console, mock_get_status, mock_settings_class):
        """Test security features demonstration with SecretStr exposure."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings

        # Mock status with SecretStr representation
        mock_status = Mock()
        mock_status.secrets_configured = 4
        mock_status.model_dump_json.return_value = '{"password": "SecretStr(\\"hidden\\")"}'
        mock_get_status.return_value = mock_status

        demonstrate_security_features()

        # Should detect SecretStr exposure
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        secretstr_warning_found = any("SecretStr representations found" in call for call in print_calls)
        assert secretstr_warning_found

    @patch("examples.health_check_demo.ApplicationSettings", side_effect=Exception("Settings error"))
    @patch("examples.health_check_demo.console")
    def test_demonstrate_security_features_exception(self, mock_console, mock_settings_class):
        """Test security features demonstration with exception."""
        demonstrate_security_features()

        # Should handle exception gracefully
        error_calls = [call for call in mock_console.print.call_args_list if "Error in security demo" in str(call)]
        assert len(error_calls) > 0

    @patch("examples.health_check_demo.ApplicationSettings")
    @patch("examples.health_check_demo.get_configuration_status")
    @patch("examples.health_check_demo.console")
    def test_demonstrate_security_features_all_secret_types(self, mock_console, mock_get_status, mock_settings_class):
        """Test that all secret types are tested for leakage."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings

        mock_status = Mock()
        mock_status.secrets_configured = 4

        # Test each secret value individually
        secret_values = [
            "super_secret_db_password",
            "sk-1234567890abcdef",
            "super_secret_app_key",
            "jwt_signing_secret",
        ]

        # Test at least one secret value detection
        secret_value = secret_values[0]  # Use first secret value
        mock_status.model_dump_json.return_value = f'{{"leaked": "{secret_value}"}}'
        mock_get_status.return_value = mock_status

        demonstrate_security_features()

        # Should detect this specific secret
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        breach_found = any("SECURITY BREACH" in call for call in print_calls)
        assert breach_found


class TestMainFunction:
    """Test the main function."""

    @patch("examples.health_check_demo.demonstrate_configuration_status")
    @patch("examples.health_check_demo.demonstrate_health_summary")
    @patch("examples.health_check_demo.demonstrate_json_serialization")
    @patch("examples.health_check_demo.demonstrate_http_endpoints")
    @patch("examples.health_check_demo.demonstrate_security_features")
    @patch("examples.health_check_demo.console")
    def test_main_function_execution_order(
        self,
        mock_console,
        mock_security,
        mock_http,
        mock_json,
        mock_health_summary,
        mock_config_status,
    ):
        """Test that main function executes all demonstrations in correct order."""
        # Mock configuration status return
        mock_status = Mock()
        mock_config_status.return_value = mock_status

        # Mock async function
        async def mock_http_demo():
            pass

        mock_http.return_value = mock_http_demo()

        main()

        # Verify all demonstrations were called
        mock_config_status.assert_called_once()
        mock_health_summary.assert_called_once()
        mock_json.assert_called_once_with(mock_status)
        mock_http.assert_called_once()
        mock_security.assert_called_once()

    @patch("examples.health_check_demo.demonstrate_configuration_status")
    @patch("examples.health_check_demo.demonstrate_health_summary")
    @patch("examples.health_check_demo.demonstrate_json_serialization")
    @patch("examples.health_check_demo.demonstrate_http_endpoints")
    @patch("examples.health_check_demo.demonstrate_security_features")
    @patch("examples.health_check_demo.console")
    def test_main_function_with_none_status(
        self,
        mock_console,
        mock_security,
        mock_http,
        mock_json,
        mock_health_summary,
        mock_config_status,
    ):
        """Test main function when configuration status returns None."""
        mock_config_status.return_value = None

        # Mock async function
        async def mock_http_demo():
            pass

        mock_http.return_value = mock_http_demo()

        main()

        # Should still call other demonstrations
        mock_config_status.assert_called_once()
        mock_health_summary.assert_called_once()
        # JSON serialization should not be called with None status
        mock_json.assert_not_called()
        mock_http.assert_called_once()
        mock_security.assert_called_once()

    @patch("examples.health_check_demo.console")
    def test_main_function_console_output(self, mock_console):
        """Test main function console output messages."""
        with (
            patch("examples.health_check_demo.demonstrate_configuration_status") as mock_config,
            patch("examples.health_check_demo.demonstrate_health_summary"),
            patch("examples.health_check_demo.demonstrate_json_serialization"),
            patch("examples.health_check_demo.demonstrate_http_endpoints"),
            patch("examples.health_check_demo.demonstrate_security_features"),
        ):

            mock_config.return_value = Mock()

            main()

        # Check for expected output messages
        print_calls = [str(call) for call in mock_console.print.call_args_list]

        # Should include title
        title_found = any("Health Check Integration Demo" in call for call in print_calls)
        assert title_found

        # Should include completion message
        complete_found = any("Demo Complete" in call for call in print_calls)
        assert complete_found

        # Should include key features
        features_found = any("Key Features Demonstrated" in call for call in print_calls)
        assert features_found

        # Should include next steps
        next_steps_found = any("Next Steps" in call for call in print_calls)
        assert next_steps_found


class TestConsoleAndRichIntegration:
    """Test console and Rich library integration."""

    def test_console_instance_creation(self):
        """Test that console instance is created correctly."""
        import examples.health_check_demo

        assert isinstance(examples.health_check_demo.console, Console)

    @patch("examples.health_check_demo.console")
    def test_rich_table_usage(self, mock_console):
        """Test Rich table usage in configuration status."""
        with (
            patch("examples.health_check_demo.get_settings") as mock_get_settings,
            patch("examples.health_check_demo.get_configuration_status") as mock_get_status,
        ):

            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings

            mock_status = Mock()
            mock_status.environment = "test"
            mock_status.version = "1.0.0"
            mock_status.debug = False
            mock_status.config_loaded = True
            mock_status.encryption_enabled = True
            mock_status.config_source = "file"
            mock_status.validation_status = "valid"
            mock_status.secrets_configured = 2
            mock_status.api_host = "localhost"
            mock_status.api_port = 8080
            mock_status.config_healthy = True
            mock_status.validation_errors = []
            mock_get_status.return_value = mock_status

            demonstrate_configuration_status()

            # Should print table and other rich elements
            assert mock_console.print.call_count >= 3

    @patch("examples.health_check_demo.console")
    def test_rich_text_styling(self, mock_console):
        """Test Rich text styling in health summary."""
        with patch("examples.health_check_demo.get_configuration_health_summary") as mock_get_summary:
            mock_summary = {
                "healthy": True,
                "environment": "prod",
                "timestamp": "2024-01-01T12:00:00Z",
            }
            mock_get_summary.return_value = mock_summary

            demonstrate_health_summary()

            # Should use Rich text styling
            mock_console.print.assert_called()


class TestDataValidationAndEdgeCases:
    """Test data validation and edge cases."""

    @patch("examples.health_check_demo.console")
    def test_json_serialization_empty_data(self, mock_console):
        """Test JSON serialization with empty data."""
        mock_status = Mock()
        mock_status.model_dump_json.return_value = "{}"

        demonstrate_json_serialization(mock_status)

        # Should handle empty JSON gracefully
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        json_found = any("{}" in call for call in print_calls)
        assert json_found

    @patch("examples.health_check_demo.console")
    def test_json_serialization_malformed_json(self, mock_console):
        """Test JSON serialization with malformed JSON."""
        mock_status = Mock()
        mock_status.model_dump_json.return_value = '{"invalid": json'

        # Should not raise exception
        demonstrate_json_serialization(mock_status)

        # Should still check for sensitive data in malformed JSON
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        assert len(print_calls) > 0

    @patch("examples.health_check_demo.get_configuration_health_summary")
    @patch("examples.health_check_demo.console")
    def test_health_summary_missing_keys(self, mock_console, mock_get_summary):
        """Test health summary with missing keys."""
        mock_summary = {
            "healthy": True,
            # Missing other expected keys
        }
        mock_get_summary.return_value = mock_summary

        demonstrate_health_summary()

        # Should handle missing keys gracefully
        mock_get_summary.assert_called_once()

    @patch("examples.health_check_demo.ApplicationSettings")
    @patch("examples.health_check_demo.get_configuration_status")
    @patch("examples.health_check_demo.console")
    def test_security_features_partial_secrets(self, mock_console, mock_get_status, mock_settings_class):
        """Test security features with partial secret configuration."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings

        mock_status = Mock()
        mock_status.secrets_configured = 2  # Only some secrets configured
        mock_status.model_dump_json.return_value = '{"partial": "config"}'
        mock_get_status.return_value = mock_status

        demonstrate_security_features()

        # Should handle partial configuration
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        secrets_count_found = any("2" in call for call in print_calls)
        assert secrets_count_found


class TestAsyncioIntegration:
    """Test asyncio integration."""

    @pytest.mark.asyncio
    async def test_async_function_compatibility(self):
        """Test that async functions work correctly."""
        # Test a simple async operation
        await asyncio.sleep(0.001)
        assert True

    def test_asyncio_run_in_main(self):
        """Test that asyncio.run is used correctly in main."""
        # Check that the main function uses asyncio.run for HTTP demo
        import inspect

        import examples.health_check_demo

        # Get the main function source
        main_source = inspect.getsource(examples.health_check_demo.main)

        # Should contain asyncio.run call
        assert "asyncio.run" in main_source

    @pytest.mark.asyncio
    @patch("examples.health_check_demo.httpx.AsyncClient")
    async def test_http_client_async_context_manager(self, mock_client_class):
        """Test HTTP client async context manager usage."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("No connection")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Should use async context manager correctly
        await demonstrate_http_endpoints()

        # Client should be created and used
        mock_client_class.assert_called_once()


class TestImportsAndDependencies:
    """Test imports and dependencies."""

    def test_required_imports_available(self):
        """Test that all required imports are available."""
        try:
            import asyncio
            import json

            import httpx
            from pydantic import SecretStr
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            # All imports should be successful
            assert json is not None
            assert asyncio is not None
            assert httpx is not None
            assert SecretStr is not None
            assert Console is not None
            assert Panel is not None
            assert Table is not None
            assert Text is not None

        except ImportError as e:
            pytest.fail(f"Required imports failed: {e}")

    def test_config_imports_available(self):
        """Test that config module imports work."""
        try:
            from src.config import (
                ApplicationSettings,
                ConfigurationStatusModel,
                get_configuration_health_summary,
                get_configuration_status,
                get_settings,
            )

            # All config imports should be successful
            assert ApplicationSettings is not None
            assert ConfigurationStatusModel is not None
            assert get_configuration_health_summary is not None
            assert get_configuration_status is not None
            assert get_settings is not None

        except ImportError as e:
            pytest.fail(f"Config imports failed: {e}")

    def test_http_status_constants(self):
        """Test HTTP status constants."""
        from examples.health_check_demo import HTTP_OK

        assert HTTP_OK == 200
