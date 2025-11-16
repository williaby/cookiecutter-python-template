"""MCP Integration Coverage Tests for client.py and config_manager.py.

This test suite specifically targets improving coverage for:
- src/mcp_integration/client.py
- src/mcp_integration/config_manager.py

Goal: Achieve 80%+ coverage for both files.
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from src.mcp_integration.client import MCPClient, MCPClientError
from src.mcp_integration.config_manager import (
    MCPConfigurationBundle,
    MCPConfigurationManager,
    MCPServerConfig,
)


class TestMCPClient:
    """Tests for MCPClient class to improve coverage."""

    def test_init_default_config_path(self):
        """Test MCPClient initialization with default config path."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            assert client.config_path == Path(".mcp.json")

    def test_init_custom_config_path(self):
        """Test MCPClient initialization with custom config path."""
        custom_path = Path("/custom/path/config.json")
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient(config_path=custom_path)
            assert client.config_path == custom_path

    def test_load_configuration_success(self):
        """Test successful configuration loading."""
        config_data = '{"mcpServers": {"test": {"command": "test-cmd"}}}'
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", mock_open(read_data=config_data)),
        ):
            client = MCPClient()
            assert "test" in client.servers
            assert client.servers["test"]["command"] == "test-cmd"

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_configuration_file_not_found(self, mock_exists):
        """Test configuration loading when file doesn't exist."""
        client = MCPClient()
        assert client.servers == {}

    def test_load_configuration_invalid_json(self):
        """Test configuration loading with invalid JSON."""
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", mock_open(read_data="invalid json")),
            pytest.raises(MCPClientError),
        ):
            MCPClient()

    @pytest.mark.asyncio
    async def test_connect_server_success(self):
        """Test successful server connection."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.servers = {"test-server": {"command": "test-cmd"}}

            result = await client.connect_server("test-server")
            assert result is True
            assert "test-server" in client.connections

    @pytest.mark.asyncio
    async def test_connect_server_not_configured(self):
        """Test connecting to unconfigured server."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.servers = {}

            result = await client.connect_server("unknown-server")
            assert result is False

    def test_disconnect_server_success(self):
        """Test successful server disconnection."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.connections = {"test-server": {"status": "connected"}}

            result = client.disconnect_server("test-server")
            assert result is True
            assert "test-server" not in client.connections

    def test_disconnect_server_not_connected(self):
        """Test disconnecting from unconnected server."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.connections = {}

            result = client.disconnect_server("unknown-server")
            assert result is False

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful message sending."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.connections = {"test-server": {"status": "connected"}}

            message = {"type": "test", "data": "hello"}
            response = await client.send_message("test-server", message)

            assert response["status"] == "success"
            assert response["server"] == "test-server"
            assert response["echo"] == message

    @pytest.mark.asyncio
    async def test_send_message_server_not_connected(self):
        """Test sending message to unconnected server."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.connections = {}

            message = {"type": "test", "data": "hello"}

            with pytest.raises(MCPClientError, match="not connected"):
                await client.send_message("unknown-server", message)

    def test_get_connected_servers(self):
        """Test getting list of connected servers."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.connections = {"server1": {"status": "connected"}, "server2": {"status": "connected"}}

            servers = client.get_connected_servers()
            assert set(servers) == {"server1", "server2"}

    def test_get_server_status_not_configured(self):
        """Test getting status for unconfigured server."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.servers = {}

            status = client.get_server_status("unknown-server")
            assert status == {"status": "not_configured"}

    def test_get_server_status_disconnected(self):
        """Test getting status for configured but disconnected server."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.servers = {"test-server": {"command": "test-cmd"}}
            client.connections = {}

            status = client.get_server_status("test-server")
            assert status == {"status": "disconnected", "configured": True}

    def test_get_server_status_connected(self):
        """Test getting status for connected server."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.servers = {"test-server": {"command": "test-cmd"}}
            connection_info = {"status": "connected", "server": "test-server"}
            client.connections = {"test-server": connection_info}

            status = client.get_server_status("test-server")
            assert status["status"] == "connected"
            assert status["configured"] is True
            assert status["connection"] == connection_info

    @pytest.mark.asyncio
    async def test_health_check_all_connected(self):
        """Test health check when all servers are connected."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.servers = {"server1": {"command": "cmd1"}, "server2": {"command": "cmd2"}}
            client.connections = {"server1": {"status": "connected"}, "server2": {"status": "connected"}}

            health = await client.health_check()

            assert health["overall_status"] == "healthy"
            assert health["total_configured"] == 2
            assert health["total_connected"] == 2

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check when some servers are disconnected."""
        with patch.object(MCPClient, "_load_configuration"):
            client = MCPClient()
            client.servers = {"server1": {"command": "cmd1"}, "server2": {"command": "cmd2"}}
            client.connections = {"server1": {"status": "connected"}}

            health = await client.health_check()

            assert health["overall_status"] == "degraded"
            assert health["total_configured"] == 2
            assert health["total_connected"] == 1


class TestMCPServerConfig:
    """Tests for MCPServerConfig model."""

    def test_valid_config_with_command(self):
        """Test valid configuration with command."""
        config = MCPServerConfig(
            command="test-command",
            args=["--arg1", "--arg2"],
            env={"VAR1": "value1"},
            enabled=True,
            priority=50,
        )
        assert config.command == "test-command"
        assert config.args == ["--arg1", "--arg2"]
        assert config.env == {"VAR1": "value1"}
        assert config.enabled is True
        assert config.priority == 50

    def test_valid_config_with_transport(self):
        """Test valid configuration with transport."""
        config = MCPServerConfig(transport={"type": "http", "url": "http://localhost:8080"}, enabled=True)
        assert config.command is None
        assert config.transport == {"type": "http", "url": "http://localhost:8080"}

    def test_invalid_config_no_command_or_transport(self):
        """Test invalid configuration without command or transport."""
        with pytest.raises(ValidationError):
            MCPServerConfig()

    def test_docker_integration_fields(self):
        """Test Docker MCP Toolkit integration fields."""
        config = MCPServerConfig(
            command="test-command",
            docker_compatible=True,
            memory_requirement="2GB",
            deployment_preference="docker",
            docker_features=["feature1", "feature2"],
            self_hosted_features=["feature3"],
        )
        assert config.docker_compatible is True
        assert config.memory_requirement == "2GB"
        assert config.deployment_preference == "docker"
        assert config.docker_features == ["feature1", "feature2"]
        assert config.self_hosted_features == ["feature3"]

    def test_default_values(self):
        """Test default values for optional fields."""
        config = MCPServerConfig(command="test-command")
        assert config.args == []
        assert config.env == {}
        assert config.enabled is True
        assert config.priority == 100
        assert config.timeout == 30
        assert config.retry_attempts == 3
        assert config.docker_compatible is False
        assert config.memory_requirement == "unknown"
        assert config.deployment_preference == "auto"
        assert config.docker_features == []
        assert config.self_hosted_features == []

    def test_priority_validation_valid(self):
        """Test valid priority values."""
        config = MCPServerConfig(command="test", priority=500)
        assert config.priority == 500

    def test_priority_validation_invalid_low(self):
        """Test invalid priority (too low)."""
        with pytest.raises(ValidationError):
            MCPServerConfig(command="test", priority=0)

    def test_priority_validation_invalid_high(self):
        """Test invalid priority (too high)."""
        with pytest.raises(ValidationError):
            MCPServerConfig(command="test", priority=1001)


class TestMCPConfigurationBundle:
    """Tests for MCPConfigurationBundle model."""

    def test_valid_bundle(self):
        """Test valid configuration bundle."""
        bundle = MCPConfigurationBundle(
            mcpServers={"server1": MCPServerConfig(command="cmd1"), "server2": MCPServerConfig(command="cmd2")},
            parallel_execution=True,
            max_concurrent_servers=10,
        )
        assert len(bundle.mcp_servers) == 2
        assert bundle.parallel_execution is True
        assert bundle.max_concurrent_servers == 10

    def test_alias_support(self):
        """Test field alias support."""
        raw_data = {"mcpServers": {"server1": {"command": "cmd1"}}}
        bundle = MCPConfigurationBundle(**raw_data)
        assert len(bundle.mcp_servers) == 1

    def test_default_values(self):
        """Test default values."""
        bundle = MCPConfigurationBundle(mcpServers={})
        assert bundle.version == "1.0"
        assert bundle.global_settings == {}
        assert bundle.parallel_execution is True
        assert bundle.max_concurrent_servers == 5
        assert bundle.health_check_interval == 60


class TestMCPConfigurationManager:
    """Tests for MCPConfigurationManager."""

    def test_init_default_path(self):
        """Test initialization with default config path."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            assert manager.config_path == Path(".mcp.json")
            assert manager.backup_path == Path(".mcp.json.backup")

    def test_init_custom_path(self):
        """Test initialization with custom config path."""
        custom_path = Path("custom_config.json")
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager(config_path=custom_path)
            assert manager.config_path == custom_path

    def test_load_configuration_success(self):
        """Test successful configuration loading."""
        config_data = '{"mcpServers": {"test": {"command": "test-cmd"}}}'
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", mock_open(read_data=config_data)),
        ):
            manager = MCPConfigurationManager()
            assert manager.configuration is not None
            assert len(manager.configuration.mcp_servers) == 1
            assert "test" in manager.configuration.mcp_servers

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_configuration_file_not_found(self, mock_exists):
        """Test configuration loading when file doesn't exist."""
        manager = MCPConfigurationManager()
        assert manager.configuration is not None
        assert len(manager.configuration.mcp_servers) == 0

    def test_save_configuration_success(self):
        """Test successful configuration saving."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(mcpServers={})

            with patch("builtins.open", mock_open()), patch("pathlib.Path.exists", return_value=False):
                result = manager.save_configuration(backup_current=False)
                assert result is True

    def test_save_configuration_no_config(self):
        """Test saving when no configuration is loaded."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = None

            result = manager.save_configuration()
            assert result is False

    def test_get_server_config_success(self):
        """Test getting server configuration."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            server_config = MCPServerConfig(command="test-cmd")
            manager.configuration = MCPConfigurationBundle(mcpServers={"test-server": server_config})

            result = manager.get_server_config("test-server")
            assert result == server_config

    def test_get_server_config_not_found(self):
        """Test getting server configuration that doesn't exist."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(mcpServers={})

            result = manager.get_server_config("nonexistent")
            assert result is None

    def test_add_server_config_success(self):
        """Test adding server configuration."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(mcpServers={})

            server_config = MCPServerConfig(command="new-cmd")
            result = manager.add_server_config("new-server", server_config)

            assert result is True
            assert "new-server" in manager.configuration.mcp_servers

    def test_add_server_config_no_configuration(self):
        """Test adding server configuration when no configuration exists."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = None

            server_config = MCPServerConfig(command="new-cmd")
            result = manager.add_server_config("new-server", server_config)

            assert result is True
            assert manager.configuration is not None

    def test_remove_server_config_success(self):
        """Test removing server configuration."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            server_config = MCPServerConfig(command="test-cmd")
            manager.configuration = MCPConfigurationBundle(mcpServers={"test-server": server_config})

            result = manager.remove_server_config("test-server")
            assert result is True
            assert "test-server" not in manager.configuration.mcp_servers

    def test_remove_server_config_not_found(self):
        """Test removing server configuration that doesn't exist."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(mcpServers={})

            result = manager.remove_server_config("nonexistent")
            assert result is False

    def test_get_enabled_servers_success(self):
        """Test getting enabled servers sorted by priority."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(
                mcpServers={
                    "server1": MCPServerConfig(command="cmd1", enabled=True, priority=200),
                    "server2": MCPServerConfig(command="cmd2", enabled=True, priority=100),
                    "server3": MCPServerConfig(command="cmd3", enabled=False, priority=50),
                },
            )

            enabled = manager.get_enabled_servers()
            assert enabled == ["server2", "server1"]  # Sorted by priority

    def test_get_enabled_servers_no_configuration(self):
        """Test getting enabled servers when no configuration exists."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = None

            enabled = manager.get_enabled_servers()
            assert enabled == []

    def test_get_parallel_execution_config_success(self):
        """Test getting parallel execution configuration."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(
                mcpServers={},
                parallel_execution=True,
                max_concurrent_servers=10,
                health_check_interval=120,
            )

            config = manager.get_parallel_execution_config()
            assert config["enabled"] is True
            assert config["max_concurrent"] == 10
            assert config["health_check_interval"] == 120

    def test_get_parallel_execution_config_no_configuration(self):
        """Test getting parallel execution config when no configuration exists."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = None

            config = manager.get_parallel_execution_config()
            assert config == {"enabled": False, "max_concurrent": 1}

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(
                mcpServers={
                    "server1": MCPServerConfig(command="cmd1", enabled=True),
                    "server2": MCPServerConfig(command="cmd2", enabled=False),
                },
            )

            result = manager.validate_configuration()
            assert result["valid"] is True
            assert result["server_count"] == 2
            assert result["enabled_count"] == 1

    def test_validate_configuration_no_configuration(self):
        """Test validation when no configuration exists."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = None

            result = manager.validate_configuration()
            assert result["valid"] is False
            assert "No configuration loaded" in result["errors"]

    def test_get_health_status_success(self):
        """Test getting health status."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = MCPConfigurationBundle(
                mcpServers={"server1": MCPServerConfig(command="cmd1", enabled=True)},
                parallel_execution=True,
            )

            health = manager.get_health_status()
            assert health["configuration_loaded"] is True
            assert health["configuration_valid"] is True
            assert health["total_servers"] == 1
            assert health["enabled_servers"] == 1
            assert health["parallel_execution"] is True

    def test_get_health_status_no_configuration(self):
        """Test getting health status when no configuration exists."""
        with patch.object(MCPConfigurationManager, "_load_configuration"):
            manager = MCPConfigurationManager()
            manager.configuration = None

            health = manager.get_health_status()
            assert health["configuration_loaded"] is False
            assert health["configuration_valid"] is False
            assert health["parallel_execution"] is False
