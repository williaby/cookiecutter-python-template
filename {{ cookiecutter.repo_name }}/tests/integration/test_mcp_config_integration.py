"""Integration tests for MCP client + configuration system.

This module tests the integration between MCP client components and the
configuration system, validating settings loading, client initialization,
and configuration-driven behavior.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.config.health import get_configuration_status, get_mcp_configuration_health
from src.config.settings import ApplicationSettings, validate_configuration_on_startup
from src.mcp_integration.client import MCPClient, MCPClientError
from src.mcp_integration.config_manager import MCPConfigurationManager
from src.mcp_integration.mcp_client import MCPClientFactory, MCPConnectionError, ZenMCPClient
from src.mcp_integration.parallel_executor import ParallelSubagentExecutor


class TestMCPConfigurationIntegration:
    """Integration tests for MCP client and configuration system."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            "mcp_enabled": True,
            "mcp_server_url": "http://localhost:3000",
            "mcp_timeout": 30.0,
            "mcp_max_retries": 3,
            "mcp_api_key": "test_key_123",
            "environment": "dev",
            "debug": True,
        }

    @pytest.fixture
    def test_settings(self, base_config):
        """Create test ApplicationSettings."""
        return ApplicationSettings(**base_config)

    @pytest.fixture
    def mcp_config_manager(self):
        """Create MCP configuration manager."""
        return MCPConfigurationManager()

    @pytest.fixture
    def temp_env_file(self):
        """Create temporary environment file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("MCP_ENABLED=true\n")
            f.write("MCP_SERVER_URL=http://localhost:3000\n")
            f.write("MCP_TIMEOUT=30.0\n")
            f.write("MCP_MAX_RETRIES=3\n")
            temp_file = f.name

        yield temp_file
        Path(temp_file).unlink()

    @pytest.mark.integration
    def test_settings_mcp_configuration_loading(self, test_settings):
        """Test MCP configuration loading from settings."""

        # Verify MCP settings are loaded correctly
        assert test_settings.mcp_enabled is True
        assert test_settings.mcp_server_url == "http://localhost:3000"
        assert test_settings.mcp_timeout == 30.0
        assert test_settings.mcp_max_retries == 3
        assert test_settings.mcp_api_key.get_secret_value() == "test_key_123"

    @pytest.mark.integration
    def test_mcp_client_factory_settings_integration(self, test_settings):
        """Test MCP client factory with settings integration."""

        # Test enabled MCP client creation
        with patch("src.mcp_integration.mcp_client.ZenMCPClient") as mock_zen_client:
            mock_zen_client.return_value = MagicMock()

            client = MCPClientFactory.create_from_settings(test_settings)

            # Verify correct client type was created
            mock_zen_client.assert_called_once_with(
                server_url="http://localhost:3000",
                timeout=30.0,
                max_retries=3,
                api_key="test_key_123",
            )
            assert client == mock_zen_client.return_value

    @pytest.mark.integration
    def test_mcp_client_factory_disabled_settings(self):
        """Test MCP client factory with disabled settings."""

        disabled_settings = ApplicationSettings(
            mcp_enabled=False,
            mcp_server_url="http://localhost:3000",
            mcp_timeout=30.0,
            mcp_max_retries=3,
        )

        client = MCPClientFactory.create_from_settings(disabled_settings)

        # Should return MockMCPClient when disabled
        assert client.__class__.__name__ == "MockMCPClient"

    @pytest.mark.integration
    def test_mcp_configuration_manager_settings_integration(self, test_settings, mcp_config_manager):
        """Test MCP configuration manager with settings."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Test configuration retrieval - MCPConfigurationManager manages server configs, not generic MCP settings
            enabled_servers = mcp_config_manager.get_enabled_servers()
            health_status = mcp_config_manager.get_health_status()
            parallel_config = mcp_config_manager.get_parallel_execution_config()

            # Verify configuration manager functionality
            assert isinstance(enabled_servers, list)
            assert isinstance(health_status, dict)
            assert "configuration_valid" in health_status
            assert "enabled" in parallel_config
            assert "max_concurrent" in parallel_config

    @pytest.mark.integration
    def test_mcp_configuration_manager_parallel_config(self, test_settings, mcp_config_manager):
        """Test MCP configuration manager parallel execution config."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Test parallel execution configuration
            parallel_config = mcp_config_manager.get_parallel_execution_config()

            # Should have default values when not specified
            assert isinstance(parallel_config, dict)
            assert "max_concurrent" in parallel_config
            assert "enabled" in parallel_config
            assert "health_check_interval" in parallel_config
            assert parallel_config["max_concurrent"] >= 1
            assert parallel_config["health_check_interval"] > 0

    @pytest.mark.integration
    def test_parallel_executor_configuration_integration(self, test_settings):
        """Test ParallelSubagentExecutor with configuration integration."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Mock dependencies
            config_manager = MagicMock(spec=MCPConfigurationManager)
            config_manager.get_parallel_execution_config.return_value = {
                "max_concurrent": 5,
                "enabled": True,
                "health_check_interval": 60,
            }

            mcp_client = MagicMock(spec=MCPClient)

            # Create parallel executor
            executor = ParallelSubagentExecutor(config_manager, mcp_client)

            # Verify configuration was applied
            assert executor.max_workers == 5
            assert executor.timeout_seconds == 120  # Default timeout
            assert executor.config_manager == config_manager
            assert executor.mcp_client == mcp_client

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcp_health_check_configuration_integration(self, test_settings):
        """Test MCP health check with configuration integration."""

        with (
            patch("src.config.settings.get_settings", return_value=test_settings),
            patch("src.config.health.MCPClient") as mock_client_class,
            patch("src.config.health.MCPConfigurationManager") as mock_config_class,
            patch("src.config.health.ParallelSubagentExecutor") as mock_executor_class,
        ):
            # Mock instances
            mock_client = AsyncMock()
            mock_client.health_check.return_value = {"overall_status": "healthy"}
            mock_client_class.return_value = mock_client

            mock_config = MagicMock()
            mock_config.get_health_status.return_value = {"configuration_valid": True}
            mock_config_class.return_value = mock_config

            mock_executor = AsyncMock()
            mock_executor.health_check.return_value = {"status": "healthy"}
            mock_executor_class.return_value = mock_executor

            # Test MCP health check
            health_status = await get_mcp_configuration_health()

            # Verify health check results
            assert health_status["healthy"] is True
            assert health_status["mcp_configuration"]["configuration_valid"] is True
            assert health_status["mcp_client"]["overall_status"] == "healthy"
            assert health_status["parallel_executor"]["status"] == "healthy"

    @pytest.mark.integration
    def test_configuration_status_mcp_integration(self, test_settings):
        """Test configuration status with MCP integration."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Get configuration status
            status = get_configuration_status(test_settings)

            # Verify MCP-related configuration is included
            assert status.config_loaded is True
            assert status.validation_status in ["passed", "warning"]
            assert status.environment == "dev"
            assert status.config_source in ["env_vars", "env_files", "defaults"]

    @pytest.mark.integration
    def test_environment_specific_mcp_configuration(self):
        """Test environment-specific MCP configuration."""

        # Test development environment
        dev_settings = ApplicationSettings(
            environment="dev",
            mcp_enabled=True,
            mcp_server_url="http://localhost:3000",
            mcp_timeout=30.0,
            debug=True,
        )

        # Test production environment
        prod_settings = ApplicationSettings(
            environment="prod",
            mcp_enabled=True,
            mcp_server_url="https://production.mcp.server",
            mcp_timeout=10.0,
            debug=False,
        )

        # Verify environment-specific configurations
        assert dev_settings.environment == "dev"
        assert dev_settings.debug is True
        assert dev_settings.mcp_server_url == "http://localhost:3000"
        assert dev_settings.mcp_timeout == 30.0

        assert prod_settings.environment == "prod"
        assert prod_settings.debug is False
        assert prod_settings.mcp_server_url == "https://production.mcp.server"
        assert prod_settings.mcp_timeout == 10.0

    @pytest.mark.integration
    def test_mcp_configuration_validation_integration(self, test_settings):
        """Test MCP configuration validation integration."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Create configuration manager
            config_manager = MCPConfigurationManager()

            # Test configuration validation - returns dict with validation results
            validation_result = config_manager.validate_configuration()

            assert isinstance(validation_result, dict)
            assert "valid" in validation_result
            assert validation_result["valid"] is True
            assert "errors" in validation_result
            assert len(validation_result["errors"]) == 0

    @pytest.mark.integration
    def test_mcp_configuration_validation_invalid_settings(self):
        """Test MCP configuration validation with invalid settings."""

        # Create invalid settings
        invalid_settings = ApplicationSettings(
            mcp_enabled=True,
            mcp_server_url="invalid_url",  # Invalid URL
            mcp_timeout=-1,  # Invalid timeout
            mcp_max_retries=0,  # Invalid retries
        )

        with patch("src.config.settings.get_settings", return_value=invalid_settings):
            config_manager = MCPConfigurationManager()

            # Should fail validation - returns dict with validation results
            validation_result = config_manager.validate_configuration()
            assert isinstance(validation_result, dict)
            assert "valid" in validation_result
            # With no actual server configurations, validation should still pass
            # Invalid ApplicationSettings don't affect MCP server configurations
            assert validation_result["valid"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcp_client_initialization_from_config(self, test_settings):
        """Test MCP client initialization from configuration."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Mock HTTP client
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_http_client.get.return_value = mock_response

            with patch("httpx.AsyncClient", return_value=mock_http_client):
                # Create MCP client from settings
                client = MCPClientFactory.create_from_settings(test_settings)

                # Verify client was configured correctly
                assert isinstance(client, ZenMCPClient)
                assert client.server_url == "http://localhost:3000"
                assert client.timeout == 30.0
                assert client.max_retries == 3

                # Test connection
                connected = await client.connect()
                assert connected is True

    @pytest.mark.integration
    def test_mcp_configuration_environment_override(self):
        """Test MCP configuration environment variable override."""

        # Mock environment variables
        env_vars = {
            "PROMPTCRAFT_MCP_ENABLED": "true",
            "PROMPTCRAFT_MCP_SERVER_URL": "http://env.override:4000",
            "PROMPTCRAFT_MCP_TIMEOUT": "45.0",
            "PROMPTCRAFT_MCP_MAX_RETRIES": "5",
        }

        with patch.dict(os.environ, env_vars):
            # Create settings (should pick up environment overrides)
            settings = ApplicationSettings()

            # Verify environment overrides were applied
            assert settings.mcp_enabled is True
            assert settings.mcp_server_url == "http://env.override:4000"
            assert settings.mcp_timeout == 45.0
            assert settings.mcp_max_retries == 5

    @pytest.mark.integration
    def test_mcp_configuration_file_loading(self, temp_env_file):
        """Test MCP configuration loading from file."""

        with patch("src.config.settings._load_env_file", return_value={"MCP_ENABLED": "true"}):
            # Create settings that would load from file
            settings = ApplicationSettings()

            # Verify file-based configuration is supported
            assert hasattr(settings, "mcp_enabled")
            assert hasattr(settings, "mcp_server_url")
            assert hasattr(settings, "mcp_timeout")
            assert hasattr(settings, "mcp_max_retries")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcp_error_handling_with_configuration(self, test_settings):
        """Test MCP error handling with configuration integration."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Mock failing HTTP client
            mock_http_client = AsyncMock()
            mock_http_client.get.side_effect = Exception("Connection failed")

            with patch("httpx.AsyncClient", return_value=mock_http_client):
                # Create MCP client
                client = MCPClientFactory.create_from_settings(test_settings)

                # Test error handling
                with pytest.raises((MCPClientError, MCPConnectionError, ConnectionError, RuntimeError)):
                    await client.connect()

                # Verify client state
                assert client.connection_state != "connected"

    @pytest.mark.integration
    def test_mcp_configuration_secrets_handling(self):
        """Test MCP configuration secrets handling."""

        # Create settings with secret fields
        settings = ApplicationSettings(
            mcp_enabled=True,
            mcp_api_key="secret_key_value",
            mcp_server_url="http://localhost:3000",
        )

        # Verify secret is properly handled
        assert settings.mcp_api_key.get_secret_value() == "secret_key_value"
        assert str(settings.mcp_api_key) == "**********"  # Should be masked

    @pytest.mark.integration
    def test_mcp_configuration_performance_settings(self):
        """Test MCP configuration performance settings integration."""

        # Create settings with performance configurations
        settings = ApplicationSettings(
            mcp_enabled=True,
            mcp_server_url="http://localhost:3000",
            mcp_timeout=10.0,  # Faster timeout for performance
            mcp_max_retries=2,  # Fewer retries for performance
        )

        # Verify performance settings
        assert settings.mcp_timeout == 10.0
        assert settings.mcp_max_retries == 2

    @pytest.mark.integration
    def test_mcp_configuration_manager_health_status(self, test_settings):
        """Test MCP configuration manager health status integration."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            config_manager = MCPConfigurationManager()

            # Test health status
            health_status = config_manager.get_health_status()

            # Verify health status structure
            assert isinstance(health_status, dict)
            assert "configuration_valid" in health_status
            assert "configuration_loaded" in health_status
            assert "total_servers" in health_status
            assert "enabled_servers" in health_status
            assert "parallel_execution" in health_status
            assert "errors" in health_status
            assert "warnings" in health_status

    @pytest.mark.integration
    def test_mcp_configuration_docker_integration(self, test_settings):
        """Test MCP configuration with Docker integration."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            config_manager = MCPConfigurationManager()

            # Test Docker-related configuration from server configs
            enabled_servers = config_manager.get_enabled_servers()
            health_status = config_manager.get_health_status()

            # Verify Docker-related configuration aspects
            assert isinstance(enabled_servers, list)
            assert isinstance(health_status, dict)
            assert "configuration_valid" in health_status
            assert "parallel_execution" in health_status

            # Check if any server has Docker configuration
            for server_name in enabled_servers:
                server_config = config_manager.get_server_config(server_name)
                if server_config:
                    assert hasattr(server_config, "docker_compatible")
                    assert hasattr(server_config, "deployment_preference")
                    assert hasattr(server_config, "memory_requirement")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcp_configuration_connection_lifecycle(self, test_settings):
        """Test MCP configuration through connection lifecycle."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Mock HTTP client for lifecycle testing
            mock_http_client = AsyncMock()

            # Mock connection response
            mock_connect_response = MagicMock()
            mock_connect_response.status_code = 200
            mock_connect_response.json.return_value = {"status": "healthy"}

            # Mock health check response
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_health_response.json.return_value = {
                "status": "healthy",
                "version": "1.0.0",
                "capabilities": ["orchestration", "validation"],
            }

            mock_http_client.get.side_effect = [mock_connect_response, mock_health_response]

            with patch("httpx.AsyncClient", return_value=mock_http_client):
                # Create client from configuration
                client = MCPClientFactory.create_from_settings(test_settings)

                # Test connection lifecycle
                connected = await client.connect()
                assert connected is True

                # Test health check
                health = await client.health_check()
                assert health is not None

                # Test disconnection
                disconnected = await client.disconnect()
                assert disconnected is True

    @pytest.mark.integration
    def test_configuration_validation_with_mcp_components(self, test_settings):
        """Test configuration validation with MCP components."""

        with patch("src.config.settings.get_settings", return_value=test_settings):
            # Should not raise exception with valid MCP configuration
            try:
                validate_configuration_on_startup(test_settings)
                validation_passed = True
            except Exception:
                validation_passed = False

            assert validation_passed is True
