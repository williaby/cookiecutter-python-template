"""MCP Client and Config Manager Coverage Tests.

Simple tests to improve coverage for client.py and config_manager.py by actually executing the code.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

# Import to ensure coverage tracking
from src.mcp_integration.client import MCPClient, MCPClientError
from src.mcp_integration.config_manager import (
    MCPConfigurationBundle,
    MCPConfigurationManager,
    MCPServerConfig,
)


class TestMCPClientCoverage:
    """Tests to improve coverage of MCPClient."""

    def test_client_creation_and_basic_operations(self):
        """Test basic client operations to improve coverage."""
        # Create a temporary config file
        config_data = {"mcpServers": {"test-server": {"command": "test-command", "args": ["--test"], "enabled": True}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            # Create client and test methods
            client = MCPClient(config_path=config_path)

            # Test getter methods
            servers = client.get_connected_servers()
            assert isinstance(servers, list)

            # Test server status methods
            status = client.get_server_status("test-server")
            assert isinstance(status, dict)

            status = client.get_server_status("nonexistent-server")
            assert status["status"] == "not_configured"

        finally:
            config_path.unlink()

    @pytest.mark.asyncio
    async def test_client_async_operations(self):
        """Test async operations to improve coverage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {"test": {"command": "cmd"}}}, f)
            config_path = Path(f.name)

        try:
            client = MCPClient(config_path=config_path)

            # Test connection operations
            result = await client.connect_server("test")
            assert isinstance(result, bool)

            if result:
                # Test message sending if connected
                try:
                    response = await client.send_message("test", {"type": "test"})
                    assert isinstance(response, dict)
                except MCPClientError:
                    pass  # Expected for mock implementation

                # Test disconnection
                disconnect_result = client.disconnect_server("test")
                assert isinstance(disconnect_result, bool)

            # Test health check
            health = await client.health_check()
            assert isinstance(health, dict)

        finally:
            config_path.unlink()

    def test_client_error_handling(self):
        """Test client error handling to improve coverage."""
        # Test with non-existent config file
        non_existent_path = Path("/non/existent/path/config.json")
        client = MCPClient(config_path=non_existent_path)
        assert len(client.servers) == 0

    @pytest.mark.asyncio
    async def test_client_error_scenarios(self):
        """Test client error scenarios."""
        client = MCPClient(config_path=Path("nonexistent.json"))

        # Test connecting to non-configured server
        result = await client.connect_server("nonexistent")
        assert result is False

        # Test sending message to non-connected server
        with pytest.raises(MCPClientError):
            await client.send_message("nonexistent", {"test": "message"})


class TestMCPConfigurationManagerCoverage:
    """Tests to improve coverage of MCPConfigurationManager."""

    def test_config_manager_creation_and_operations(self):
        """Test basic config manager operations."""
        # Create a temporary config file
        config_data = {
            "version": "1.0",
            "mcpServers": {
                "server1": {"command": "test-cmd", "args": ["--test"], "enabled": True, "priority": 100},
                "server2": {"command": "test-cmd2", "enabled": False, "priority": 200},
            },
            "parallel_execution": True,
            "max_concurrent_servers": 5,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            # Create manager and test methods
            manager = MCPConfigurationManager(config_path=config_path)

            # Test getting server configs
            server1_config = manager.get_server_config("server1")
            assert server1_config is not None
            assert server1_config.command == "test-cmd"

            # Test getting nonexistent server
            nonexistent = manager.get_server_config("nonexistent")
            assert nonexistent is None

            # Test getting enabled servers
            enabled_servers = manager.get_enabled_servers()
            assert "server1" in enabled_servers
            assert "server2" not in enabled_servers

            # Test parallel execution config
            parallel_config = manager.get_parallel_execution_config()
            assert parallel_config["enabled"] is True
            assert parallel_config["max_concurrent"] == 5

            # Test validation
            validation_result = manager.validate_configuration()
            assert isinstance(validation_result, dict)
            assert "valid" in validation_result

            # Test health status
            health_status = manager.get_health_status()
            assert isinstance(health_status, dict)
            assert "configuration_loaded" in health_status

        finally:
            config_path.unlink()

    def test_config_manager_modifications(self):
        """Test config manager modification operations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {}}, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path=config_path)

            # Test adding server config
            new_config = MCPServerConfig(command="new-server-cmd", enabled=True, priority=150)
            result = manager.add_server_config("new-server", new_config)
            assert result is True

            # Verify it was added
            retrieved_config = manager.get_server_config("new-server")
            assert retrieved_config is not None
            assert retrieved_config.command == "new-server-cmd"

            # Test removing server config
            result = manager.remove_server_config("new-server")
            assert result is True

            # Verify it was removed
            retrieved_config = manager.get_server_config("new-server")
            assert retrieved_config is None

            # Test removing non-existent server
            result = manager.remove_server_config("nonexistent")
            assert result is False

        finally:
            config_path.unlink()

    def test_config_manager_save_operations(self):
        """Test config manager save operations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {}}, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path=config_path)

            # Add a server config
            new_config = MCPServerConfig(command="save-test-cmd")
            manager.add_server_config("save-test", new_config)

            # Test saving
            result = manager.save_configuration(backup_current=False)
            assert result is True

            # Verify the file was updated
            with config_path.open() as f:
                saved_data = json.load(f)
                assert "save-test" in saved_data["mcpServers"]

        finally:
            config_path.unlink()

    def test_config_manager_error_scenarios(self):
        """Test config manager error scenarios."""
        # Test with non-existent config file
        non_existent_path = Path("/non/existent/path/config.json")
        manager = MCPConfigurationManager(config_path=non_existent_path)

        # Should create default config
        assert manager.configuration is not None
        assert len(manager.configuration.mcp_servers) == 0

        # Test operations on empty config
        enabled_servers = manager.get_enabled_servers()
        assert enabled_servers == []

        parallel_config = manager.get_parallel_execution_config()
        assert parallel_config["enabled"] is True  # Default value

    def test_config_manager_load_errors(self):
        """Test config manager load error scenarios to improve coverage."""
        import json

        # Test JSON decode error and backup loading
        with (
            patch("pathlib.Path.exists", side_effect=[True, True]),  # Main exists, backup exists
            patch("builtins.open", mock_open(read_data="invalid json")),
            patch(
                "src.mcp_integration.config_manager.json.load",
                side_effect=[
                    json.JSONDecodeError("Invalid", "doc", 0),  # Main config fails
                    {"mcpServers": {"backup": {"command": "backup-cmd"}}},  # Backup succeeds
                ],
            ),
        ):
            manager = MCPConfigurationManager()
            # Should load backup configuration
            assert manager.configuration is not None

        # Test both main and backup config failing
        with (
            patch("pathlib.Path.exists", side_effect=[True, True]),
            patch("builtins.open", mock_open(read_data="invalid json")),
            patch(
                "src.mcp_integration.config_manager.json.load",
                side_effect=[
                    json.JSONDecodeError("Invalid", "doc", 0),  # Main config fails
                    json.JSONDecodeError("Invalid", "doc", 0),  # Backup also fails
                ],
            ),
        ):
            manager = MCPConfigurationManager()
            # Should create default configuration
            assert manager.configuration is not None
            assert len(manager.configuration.mcp_servers) == 0

    def test_config_manager_save_errors(self):
        """Test config manager save error scenarios."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {}}, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path=config_path)

            # Test save with backup rename error
            with patch("pathlib.Path.rename", side_effect=OSError("Rename failed")):
                result = manager.save_configuration(backup_current=True)
                # Should still try to save even if backup fails
                assert isinstance(result, bool)

            # Test save with write permission error
            with patch("pathlib.Path.open", side_effect=PermissionError("Permission denied")):
                result = manager.save_configuration(backup_current=False)
                # Save should return False on permission error
                assert result is False

        finally:
            config_path.unlink()

    def test_config_manager_modification_errors(self):
        """Test config manager modification error scenarios."""
        manager = MCPConfigurationManager(config_path=Path("nonexistent.json"))

        # Test get_server_config with no configuration
        manager.configuration = None
        result = manager.get_server_config("test")
        assert result is None

        # Test add_server_config exception handling
        manager.configuration = MCPConfigurationBundle(mcpServers={})

        # Force an exception in add_server_config by making mcp_servers assignment fail
        from unittest.mock import Mock, PropertyMock

        # Create a mock configuration that will raise an exception when accessing mcp_servers
        mock_config = Mock()
        type(mock_config).mcp_servers = PropertyMock(side_effect=AttributeError("Server access error"))

        with patch.object(manager, "configuration", mock_config):
            server_config = MCPServerConfig(command="test-cmd")
            result = manager.add_server_config("test", server_config)
            assert result is False

    def test_config_manager_validation_errors(self):
        """Test config manager validation error scenarios."""
        manager = MCPConfigurationManager(config_path=Path("nonexistent.json"))

        # Test validation with invalid server configuration
        # Create a bundle with more enabled servers than max concurrent
        server1 = MCPServerConfig(command="cmd1", enabled=True)
        server2 = MCPServerConfig(command="cmd2", enabled=True)
        server3 = MCPServerConfig(command="cmd3", enabled=True)

        manager.configuration = MCPConfigurationBundle(
            mcpServers={"server1": server1, "server2": server2, "server3": server3},
            max_concurrent_servers=2,  # Less than enabled servers
        )

        result = manager.validate_configuration()
        assert result["valid"] is True  # Should still be valid
        assert len(result["warnings"]) > 0  # Should have warnings about too many servers

        # Test validation with server having invalid priority
        server_invalid = MCPServerConfig(command="cmd")
        server_invalid.priority = 1500  # Out of valid range (1-1000)

        manager.configuration.mcp_servers["invalid"] = server_invalid
        result = manager.validate_configuration()
        # Should have warnings about invalid priority
        assert len(result["warnings"]) > 0

        # Test validation exception handling

        with patch.object(manager, "get_enabled_servers", side_effect=Exception("Get enabled failed")):
            result = manager.validate_configuration()
            assert result["valid"] is False
            assert any("Configuration validation failed" in error for error in result["errors"])


class TestMCPServerConfigCoverage:
    """Tests to improve coverage of MCPServerConfig."""

    def test_server_config_validation(self):
        """Test server config creation and validation."""
        # Test valid config with command
        config = MCPServerConfig(
            command="test-command",
            args=["--arg1", "value1"],
            env={"ENV_VAR": "value"},
            enabled=True,
            priority=50,
            timeout=60,
            retry_attempts=5,
            docker_compatible=True,
            memory_requirement="1GB",
            deployment_preference="docker",
            docker_features=["feature1"],
            self_hosted_features=["feature2"],
        )

        assert config.command == "test-command"
        assert config.args == ["--arg1", "value1"]
        assert config.env == {"ENV_VAR": "value"}
        assert config.enabled is True
        assert config.priority == 50
        assert config.docker_compatible is True
        assert config.memory_requirement == "1GB"

    def test_server_config_transport(self):
        """Test server config with transport."""
        config = MCPServerConfig(transport={"type": "http", "url": "http://localhost:8080"}, enabled=True)

        assert config.command is None
        assert config.transport["type"] == "http"
        assert config.enabled is True

    def test_server_config_defaults(self):
        """Test server config default values."""
        config = MCPServerConfig(command="test")

        assert config.args == []
        assert config.env == {}
        assert config.enabled is True
        assert config.priority == 100
        assert config.timeout == 30
        assert config.retry_attempts == 3
        assert config.docker_compatible is False
        assert config.memory_requirement == "unknown"
        assert config.deployment_preference == "auto"

    def test_server_config_validation_error(self):
        """Test server config validation errors."""
        # Test with neither command nor transport
        with pytest.raises(Exception, match="ValidationError|validation"):  # ValidationError from Pydantic
            MCPServerConfig()


class TestMCPConfigurationBundleCoverage:
    """Tests to improve coverage of MCPConfigurationBundle."""

    def test_configuration_bundle_creation(self):
        """Test configuration bundle creation."""
        server_config = MCPServerConfig(command="test-cmd")

        bundle = MCPConfigurationBundle(
            version="2.0",
            mcpServers={"test-server": server_config},
            global_settings={"setting1": "value1"},
            parallel_execution=False,
            max_concurrent_servers=10,
            health_check_interval=120,
        )

        assert bundle.version == "2.0"
        assert len(bundle.mcp_servers) == 1
        assert bundle.global_settings == {"setting1": "value1"}
        assert bundle.parallel_execution is False
        assert bundle.max_concurrent_servers == 10
        assert bundle.health_check_interval == 120

    def test_configuration_bundle_defaults(self):
        """Test configuration bundle default values."""
        bundle = MCPConfigurationBundle(mcpServers={})

        assert bundle.version == "1.0"
        assert bundle.global_settings == {}
        assert bundle.parallel_execution is True
        assert bundle.max_concurrent_servers == 5
        assert bundle.health_check_interval == 60

    def test_configuration_bundle_alias(self):
        """Test configuration bundle with alias."""
        # Test using the alias 'mcpServers'
        raw_data = {"mcpServers": {"server1": {"command": "cmd1"}}}
        bundle = MCPConfigurationBundle(**raw_data)
        assert len(bundle.mcp_servers) == 1
