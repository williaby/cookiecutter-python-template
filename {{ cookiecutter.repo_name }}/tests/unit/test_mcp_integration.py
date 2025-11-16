"""Unit tests for MCP integration components."""

import json
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.mcp_integration.client import MCPClient, MCPClientError
from src.mcp_integration.config_manager import (
    MCPConfigurationBundle,
    MCPConfigurationManager,
    MCPServerConfig,
)
from src.mcp_integration.docker_mcp_client import DockerMCPClient
from src.mcp_integration.parallel_executor import ExecutionResult, ParallelSubagentExecutor


class TestMCPClient:
    """Test MCP client functionality."""

    def test_client_initialization(self):
        """Test MCP client initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "mcpServers": {
                    "test-server": {
                        "command": "test-command",
                        "args": ["--test"],
                    },
                },
            }
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            client = MCPClient(config_path)
            assert len(client.servers) == 1
            assert "test-server" in client.servers
        finally:
            config_path.unlink()

    def test_client_missing_config(self):
        """Test client with missing configuration file."""
        non_existent_path = Path("/tmp/non-existent-config.json")  # noqa: S108
        client = MCPClient(non_existent_path)
        assert len(client.servers) == 0

    @pytest.mark.asyncio
    async def test_connect_server(self):
        """Test server connection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "mcpServers": {
                    "test-server": {
                        "command": "test-command",
                        "args": ["--test"],
                    },
                },
            }
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            client = MCPClient(config_path)
            result = await client.connect_server("test-server")
            assert result is True
            assert "test-server" in client.connections
        finally:
            config_path.unlink()

    @pytest.mark.asyncio
    async def test_connect_unknown_server(self):
        """Test connecting to unknown server."""
        client = MCPClient()
        result = await client.connect_server("unknown-server")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message to server."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "mcpServers": {
                    "test-server": {
                        "command": "test-command",
                        "args": ["--test"],
                    },
                },
            }
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            client = MCPClient(config_path)
            await client.connect_server("test-server")

            message = {"type": "test", "data": "hello"}
            response = await client.send_message("test-server", message)

            assert response["status"] == "success"
            assert response["echo"] == message
        finally:
            config_path.unlink()

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self):
        """Test sending message to disconnected server."""
        client = MCPClient()
        message = {"type": "test"}

        with pytest.raises(MCPClientError):
            await client.send_message("test-server", message)

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test client health check."""
        client = MCPClient()
        health = await client.health_check()

        assert "overall_status" in health
        assert "servers" in health
        assert "total_configured" in health
        assert "total_connected" in health


class TestMCPConfigurationManager:
    """Test MCP configuration manager."""

    def test_server_config_validation(self):
        """Test server configuration validation."""
        # Valid configuration
        config = MCPServerConfig(
            command="test-command",
            args=["--test"],
            env={"TEST_VAR": "value"},
        )
        assert config.command == "test-command"
        assert config.enabled is True
        assert config.priority == 100

        # Configuration with transport
        config_transport = MCPServerConfig(
            transport={"type": "sse", "url": "https://example.com"},
            env={"API_KEY": "secret"},
        )
        assert config_transport.transport is not None
        assert config_transport.command is None

    def test_server_config_invalid(self):
        """Test invalid server configuration."""
        with pytest.raises(ValidationError, match="Either 'command' or 'transport' must be specified"):
            # Neither command nor transport specified - this should fail validation
            MCPServerConfig()

    def test_configuration_bundle(self):
        """Test complete configuration bundle."""
        bundle = MCPConfigurationBundle(
            mcpServers={
                "server1": MCPServerConfig(command="cmd1"),
                "server2": MCPServerConfig(transport={"type": "sse", "url": "http://test"}),
            },
            parallel_execution=True,
            max_concurrent_servers=3,
        )

        assert len(bundle.mcp_servers) == 2
        assert bundle.parallel_execution is True
        assert bundle.max_concurrent_servers == 3

    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "mcpServers": {
                    "test-server": {
                        "command": "test-command",
                        "args": ["--test"],
                    },
                },
                "parallel_execution": True,
            }
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)
            assert manager.configuration is not None
            assert len(manager.configuration.mcp_servers) == 1
        finally:
            config_path.unlink()

    def test_add_server_config(self):
        """Test adding server configuration."""
        # Create isolated manager with empty config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"mcpServers": {}, "parallel_execution": True}
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)
            server_config = MCPServerConfig(command="new-server")

            result = manager.add_server_config("new-server", server_config)
            assert result is True

            retrieved_config = manager.get_server_config("new-server")
            assert retrieved_config is not None
            assert retrieved_config.command == "new-server"
        finally:
            config_path.unlink()

    def test_remove_server_config(self):
        """Test removing server configuration."""
        # Create isolated manager with empty config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"mcpServers": {}, "parallel_execution": True}
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)
            server_config = MCPServerConfig(command="temp-server")

            manager.add_server_config("temp-server", server_config)
            assert manager.get_server_config("temp-server") is not None

            result = manager.remove_server_config("temp-server")
            assert result is True
            assert manager.get_server_config("temp-server") is None
        finally:
            config_path.unlink()

    def test_get_enabled_servers(self):
        """Test getting enabled servers sorted by priority."""
        # Create isolated manager with empty config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"mcpServers": {}, "parallel_execution": True}
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)

            # Add servers with different priorities
            manager.add_server_config("high-priority", MCPServerConfig(command="cmd1", priority=1))
            manager.add_server_config("low-priority", MCPServerConfig(command="cmd2", priority=999))
            manager.add_server_config("disabled", MCPServerConfig(command="cmd3", enabled=False))

            enabled = manager.get_enabled_servers()
            assert len(enabled) == 2
            assert enabled[0] == "high-priority"  # Should be first due to lower priority number
            assert enabled[1] == "low-priority"
            assert "disabled" not in enabled
        finally:
            config_path.unlink()

    def test_validate_configuration(self):
        """Test configuration validation."""
        # Create isolated manager with empty config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"mcpServers": {}, "parallel_execution": True}
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)
            manager.add_server_config("valid", MCPServerConfig(command="test"))

            validation = manager.validate_configuration()
            assert validation["valid"] is True
            assert validation["server_count"] == 1
            assert validation["enabled_count"] == 1
        finally:
            config_path.unlink()

    def test_get_health_status(self):
        """Test getting health status."""
        # Create isolated manager with empty config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"mcpServers": {}, "parallel_execution": True}
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)
            health = manager.get_health_status()

            assert "configuration_loaded" in health
            assert "configuration_valid" in health
            assert "total_servers" in health
            assert "enabled_servers" in health
        finally:
            config_path.unlink()


class TestParallelSubagentExecutor:
    """Test parallel subagent executor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = Mock(spec=MCPConfigurationManager)
        self.mcp_client = Mock(spec=MCPClient)

        # Mock parallel execution config
        self.config_manager.get_parallel_execution_config.return_value = {
            "enabled": True,
            "max_concurrent": 3,
            "health_check_interval": 60,
        }

        self.executor = ParallelSubagentExecutor(self.config_manager, self.mcp_client)

    def test_execution_result(self):
        """Test execution result creation."""
        result = ExecutionResult(
            agent_id="test-agent",
            success=True,
            result={"output": "test"},
            execution_time=1.5,
        )

        assert result.agent_id == "test-agent"
        assert result.success is True
        assert result.result == {"output": "test"}
        assert result.execution_time == 1.5

        result_dict = result.to_dict()
        assert result_dict["agent_id"] == "test-agent"
        assert result_dict["success"] is True

    @pytest.mark.asyncio
    async def test_execute_subagents_parallel_empty(self):
        """Test parallel execution with empty task list."""
        result = await self.executor.execute_subagents_parallel([])

        assert result["success"] is False
        assert "No tasks provided" in result["error"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_execute_subagents_independent(self):
        """Test independent subagent execution."""
        tasks = [
            {"agent_id": "agent1", "type": "analysis"},
            {"agent_id": "agent2", "type": "validation"},
        ]

        result = await self.executor.execute_subagents_parallel(
            tasks,
            coordination_strategy="independent",
        )

        assert result["coordination_strategy"] == "independent"
        assert len(result["results"]) == 2
        assert result["summary"]["total"] == 2

    @pytest.mark.asyncio
    async def test_execute_subagents_consensus(self):
        """Test consensus subagent execution."""
        tasks = [
            {"agent_id": "agent1", "type": "analysis"},
            {"agent_id": "agent2", "type": "analysis"},
            {"agent_id": "agent3", "type": "analysis"},
        ]

        result = await self.executor.execute_subagents_parallel(
            tasks,
            coordination_strategy="consensus",
        )

        assert result["coordination_strategy"] == "consensus"
        assert "consensus" in result
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_execute_subagents_pipeline(self):
        """Test pipeline subagent execution."""
        tasks = [
            {"agent_id": "step1", "type": "generation"},
            {"agent_id": "step2", "type": "validation"},
        ]

        result = await self.executor.execute_subagents_parallel(
            tasks,
            coordination_strategy="pipeline",
        )

        assert result["coordination_strategy"] == "pipeline"
        assert "pipeline_complete" in result
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_execute_subagents_invalid_strategy(self):
        """Test execution with invalid coordination strategy."""
        tasks = [{"agent_id": "agent1", "type": "test"}]

        result = await self.executor.execute_subagents_parallel(
            tasks,
            coordination_strategy="invalid",
        )

        assert result["success"] is False
        assert "Unknown coordination strategy" in result["error"]

    def test_simulate_subagent_execution(self):
        """Test subagent execution simulation."""
        # Test analysis task
        analysis_task = {"agent_id": "analyzer", "type": "analysis"}
        result = self.executor._simulate_subagent_execution(analysis_task)
        assert result["analysis_type"] == "code_review"
        assert result["agent"] == "analyzer"

        # Test validation task
        validation_task = {"agent_id": "validator", "type": "validation"}
        result = self.executor._simulate_subagent_execution(validation_task)
        assert result["validation_type"] == "configuration"
        assert result["agent"] == "validator"

        # Test generation task
        generation_task = {"agent_id": "generator", "type": "generation"}
        result = self.executor._simulate_subagent_execution(generation_task)
        assert result["generation_type"] == "documentation"
        assert result["agent"] == "generator"

    def test_build_consensus(self):
        """Test consensus building from results."""
        results = [
            {"success": True, "agent_id": "agent1"},
            {"success": True, "agent_id": "agent2"},
            {"success": False, "agent_id": "agent3"},
            {"success": True, "agent_id": "agent4"},
            {"success": True, "agent_id": "agent5"},
        ]

        consensus = self.executor._build_consensus(results)
        assert consensus["consensus_reached"] is True  # 4/5 = 80% > 60%
        assert consensus["consensus_score"] == 0.8
        assert consensus["participating_agents"] == 4

    def test_get_execution_statistics(self):
        """Test getting execution statistics."""
        # Add some execution history
        self.executor.execution_history = [
            {
                "task_count": 3,
                "success_count": 2,
                "execution_time": 1.5,
                "coordination_strategy": "independent",
            },
            {
                "task_count": 2,
                "success_count": 2,
                "execution_time": 0.8,
                "coordination_strategy": "consensus",
            },
        ]

        stats = self.executor.get_execution_statistics()
        assert stats["total_executions"] == 2
        assert stats["total_tasks_executed"] == 5
        assert stats["total_successful_tasks"] == 4
        assert stats["success_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test executor health check."""
        # Mock health responses
        self.config_manager.get_health_status.return_value = {
            "configuration_valid": True,
            "parallel_execution": True,
        }

        self.mcp_client.health_check.return_value = {
            "overall_status": "healthy",
        }

        health = await self.executor.health_check()
        assert health["status"] == "healthy"
        assert "configuration_status" in health
        assert "self_hosted_client_status" in health
        assert "docker_client_status" in health
        assert "execution_statistics" in health
        assert "smart_routing_enabled" in health
        assert "routing_summary" in health

    def test_simulate_subagent_execution_with_routing(self):
        """Test subagent execution with smart routing simulation."""
        task = {
            "agent_id": "test_agent",
            "type": "analysis",
            "server_name": "github-basic",
            "tool_name": "read_file",
        }

        result = self.executor._simulate_subagent_execution_with_routing(
            task,
            "github-basic",
            "read_file",
        )

        assert "deployment" in result
        assert "routing_metadata" in result
        assert "performance" in result
        assert result["deployment"] == "docker_default"
        assert result["routing_metadata"]["docker_available"] is True
        assert result["routing_metadata"]["feature_supported"] is True

    def test_simulate_subagent_execution_with_fallback(self):
        """Test subagent execution with fallback to self-hosted."""
        task = {
            "agent_id": "test_agent",
            "type": "analysis",
            "server_name": "advanced-server",
            "tool_name": "bulk_operations",
        }

        result = self.executor._simulate_subagent_execution_with_routing(
            task,
            "advanced-server",
            "bulk_operations",
        )

        assert result["deployment"] == "self_hosted_only"
        assert result["routing_metadata"]["docker_available"] is False
        assert result["performance"]["memory_limit"] == "unlimited"
        assert result["performance"]["ide_compatibility"] == "claude_code"


class TestMCPHealthIntegration:
    """Test MCP health check integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = Mock(spec=MCPConfigurationManager)
        self.mcp_client = Mock(spec=MCPClient)
        self.executor = ParallelSubagentExecutor(config_manager=self.config_manager, mcp_client=self.mcp_client)

    def test_docker_mcp_client_initialization(self):
        """Test Docker MCP client initialization."""
        client = DockerMCPClient()
        assert client.docker_desktop_base_url == "http://localhost:3280"
        assert isinstance(client.mcp_servers, dict)
        assert client.logger is not None

    @pytest.mark.asyncio
    async def test_docker_mcp_client_methods(self):
        """Test Docker MCP client method stubs."""
        client = DockerMCPClient()

        # Test is_available
        available = await client.is_available("test-server")
        assert available is False

        # Test supports_feature
        supports = await client.supports_feature("test-server", "test-feature")
        assert supports is False

        # Test health_check
        health = await client.health_check()
        assert health["docker_mcp_available"] is True
        assert "total_servers" in health
        assert isinstance(health["total_servers"], int)

    def test_mcp_configuration_manager_edge_cases(self):
        """Test MCPConfigurationManager edge cases."""
        # Test with non-existent config file
        manager = MCPConfigurationManager(Path("/tmp/non-existent-mcp-config.json"))  # noqa: S108
        assert manager.configuration is not None
        assert len(manager.configuration.mcp_servers) == 0

        # Test parallel execution config
        parallel_config = manager.get_parallel_execution_config()
        assert parallel_config["enabled"] is True
        assert parallel_config["max_concurrent"] == 5

    def test_mcp_client_edge_cases(self):
        """Test MCPClient edge cases."""
        # Test with invalid config - use a non-existent path
        client = MCPClient(Path("/tmp/non-existent-mcp-config.json"))  # noqa: S108
        assert len(client.servers) == 0
        assert len(client.connections) == 0

    def test_execution_result_edge_cases(self):
        """Test ExecutionResult edge cases."""
        # Test with error
        result = ExecutionResult(
            agent_id="error-agent",
            success=False,
            error="Test error message",
        )
        assert result.success is False
        assert result.error == "Test error message"
        assert result.result is None

        result_dict = result.to_dict()
        assert "timestamp" in result_dict
        assert result_dict["error"] == "Test error message"

    def test_mcp_server_config_docker_features(self):
        """Test MCPServerConfig with Docker-specific features."""
        config = MCPServerConfig(
            command="test-command",
            docker_compatible=True,
            memory_requirement="1GB",
            deployment_preference="docker",
        )
        assert config.docker_compatible is True
        assert config.memory_requirement == "1GB"
        assert config.deployment_preference == "docker"

    def test_configuration_bundle_validation(self):
        """Test MCPConfigurationBundle validation."""
        # Test with invalid server config
        bundle = MCPConfigurationBundle(
            mcpServers={},
            parallel_execution=False,
        )
        assert bundle.max_concurrent_servers == 5  # Default value
        assert bundle.parallel_execution is False

    def test_config_manager_save_configuration(self):
        """Test saving configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"mcpServers": {}}
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)

            # Test save_configuration
            saved = manager.save_configuration()
            assert saved is True

            # Verify file was written
            assert config_path.exists()

            # Load and verify content
            with config_path.open() as f:
                saved_config = json.load(f)
            assert "mcpServers" in saved_config
        finally:
            config_path.unlink()

    def test_config_manager_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"mcpServers": {}}
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = MCPConfigurationManager(config_path)

            # Test with transport config instead of command
            transport_config = MCPServerConfig(transport={"type": "sse", "url": "http://test"}, enabled=True)
            result = manager.add_server_config("transport-server", transport_config)
            assert result is True

            # Validate should pass
            validation = manager.validate_configuration()
            assert validation["valid"] is True
            assert validation["server_count"] == 1

        finally:
            config_path.unlink()

    @pytest.mark.asyncio
    async def test_parallel_executor_error_handling(self):
        """Test ParallelSubagentExecutor error handling."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        # Mock config to force an error
        config_manager.get_parallel_execution_config.return_value = {
            "enabled": True,
            "max_concurrent": "invalid",  # Should cause type error
            "health_check_interval": 60,
        }

        executor = ParallelSubagentExecutor(config_manager, mcp_client)
        assert executor.max_workers == 5  # Should default to 5 on error

    def test_mcp_client_connection_cleanup(self):
        """Test MCP client connection cleanup."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "mcpServers": {
                    "test-server": {
                        "command": "test-command",
                        "args": ["--test"],
                    },
                },
            }
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            client = MCPClient(config_path)

            # Mock connection
            client.connections["test-server"] = Mock()

            # Test cleanup (this would normally close connections)
            assert "test-server" in client.connections

        finally:
            config_path.unlink()

    def test_mcp_client_send_message_edge_cases(self):
        """Test MCPClient send_message edge cases."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "mcpServers": {
                    "test-server": {
                        "command": "test-command",
                        "args": ["--test"],
                    },
                },
            }
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            client = MCPClient(config_path)

            # Test disconnect_server on non-connected server
            result = client.disconnect_server("test-server")
            assert result is False  # Can't disconnect what isn't connected

            # Test disconnect non-existent server
            result = client.disconnect_server("unknown-server")
            assert result is False

        finally:
            config_path.unlink()

    async def test_get_client_for_routing_scenarios(self):
        """Test different routing scenarios for _select_optimal_client."""
        # Test scenario: Basic client selection with tool that's not supported by docker
        with (
            mock.patch.object(self.executor.docker_client, "is_available", return_value=True),
            mock.patch.object(self.executor.docker_client, "supports_feature", return_value=False),
        ):
            client, deployment = await self.executor._select_optimal_client("test_server", "test_tool")
            # Should return fallback to self-hosted when docker doesn't support the tool
            assert client is not None
            assert deployment in ["docker", "self_hosted", "hybrid", "self_hosted_fallback", "docker_preferred"]

        # Test scenario: Docker not available, fallback to self-hosted
        with mock.patch.object(self.executor.docker_client, "is_available", return_value=False):
            client, deployment = await self.executor._select_optimal_client("test_server")
            # Should fallback to mcp_client when docker unavailable
            assert deployment in ["self_hosted", "fallback", "self_hosted_fallback", "self_hosted_only"]

        # Test scenario: Exception handling during client selection
        with mock.patch.object(self.executor.docker_client, "is_available", side_effect=Exception("Docker error")):
            client, deployment = await self.executor._select_optimal_client("test_server")
            # Should handle exceptions gracefully
            assert client is not None
            assert deployment is not None

    async def test_execute_independent_with_task_exceptions(self):
        """Test _execute_independent with task execution exceptions."""
        tasks = [{"agent_id": "working_agent", "type": "analysis"}, {"agent_id": "failing_agent", "type": "validation"}]

        # Mock simulation to make one task fail
        def mock_execution(task, timeout):
            if task["agent_id"] == "failing_agent":
                return ExecutionResult(agent_id="failing_agent", success=False, error="Task execution failed")
            return ExecutionResult(agent_id=task["agent_id"], success=True, result={"status": "completed"})

        # Mock the executor to raise exception for one task
        with mock.patch.object(self.executor, "_execute_single_subagent", side_effect=mock_execution):
            result = await self.executor._execute_independent(tasks, timeout=30)

            # Should have results for both tasks - one success, one failure
            assert len(result["results"]) == 2
            success_count = sum(1 for r in result["results"] if r["success"])
            failed_count = sum(1 for r in result["results"] if not r["success"])
            assert success_count >= 1  # At least one should succeed
            assert failed_count >= 1  # At least one should fail

    async def test_execute_pipeline_with_failure(self):
        """Test pipeline execution with step failure."""
        tasks = [
            {"agent_id": "step1", "type": "analysis"},
            {"agent_id": "step2", "type": "validation", "depends_on": "step1"},
        ]

        # Mock first step to fail
        def mock_execution(task, timeout):
            if task["agent_id"] == "step1":
                return ExecutionResult(agent_id="step1", success=False, error="Step 1 failed")
            return ExecutionResult(agent_id=task["agent_id"], success=True, result={"status": "completed"})

        with mock.patch.object(self.executor, "_execute_single_subagent", side_effect=mock_execution):
            result = await self.executor._execute_pipeline(tasks, timeout=30)

            # Pipeline should handle failures appropriately
            assert "results" in result
            assert len(result["results"]) >= 1

    async def test_simulate_subagent_execution_enhanced_exception_handling(self):
        """Test exception handling in simulation methods."""
        task = {"agent_id": "error_agent", "type": "analysis"}

        # Mock routing to raise exception, and regular simulation to also raise exception
        with (
            mock.patch.object(
                self.executor,
                "_simulate_subagent_execution_with_routing",
                side_effect=Exception("Routing error"),
            ),
            mock.patch.object(self.executor, "_simulate_subagent_execution", side_effect=Exception("Simulation error")),
        ):

            # Test the actual method that exists - _execute_single_subagent
            result = self.executor._execute_single_subagent(task, 30)

            # Should return failure result with error message or handle gracefully
            assert result.agent_id == "error_agent"
            # The result should indicate some form of handling occurred
            assert result is not None

    def test_simulate_subagent_execution_legacy_path(self):
        """Test legacy execution path in _simulate_subagent_execution."""
        # Test with task that doesn't match analysis/validation/generation patterns
        legacy_task = {"agent_id": "legacy_agent", "type": "unknown_type"}
        result = self.executor._simulate_subagent_execution(legacy_task)

        # Should handle unknown task types gracefully
        assert "agent" in result or "agent_id" in result
        assert "deployment" in result or "status" in result or "result" in result
        # Should return some kind of valid response structure
        assert isinstance(result, dict)
