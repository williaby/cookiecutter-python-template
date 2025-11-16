"""Comprehensive unit tests for ParallelSubagentExecutor.

This module provides comprehensive unit test coverage for the ParallelSubagentExecutor
class that coordinates parallel execution of subagents via MCP servers.
"""

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.mcp_integration.client import MCPClient
from src.mcp_integration.config_manager import MCPConfigurationManager
from src.mcp_integration.docker_mcp_client import DockerMCPClient
from src.mcp_integration.parallel_executor import (
    ExecutionResult,
    ParallelSubagentExecutor,
)


class TestExecutionResult:
    """Test suite for ExecutionResult class."""

    def test_execution_result_initialization(self):
        """Test ExecutionResult initialization."""
        start_time = time.time()
        result = ExecutionResult(
            agent_id="test_agent",
            success=True,
            result={"data": "test"},
            error=None,
            execution_time=1.5,
        )

        assert result.agent_id == "test_agent"
        assert result.success is True
        assert result.result == {"data": "test"}
        assert result.error is None
        assert result.execution_time == 1.5
        assert result.timestamp >= start_time

    def test_execution_result_initialization_defaults(self):
        """Test ExecutionResult initialization with defaults."""
        result = ExecutionResult(agent_id="test_agent", success=False)

        assert result.agent_id == "test_agent"
        assert result.success is False
        assert result.result is None
        assert result.error is None
        assert result.execution_time == 0.0
        assert result.timestamp > 0

    def test_execution_result_initialization_with_error(self):
        """Test ExecutionResult initialization with error."""
        result = ExecutionResult(agent_id="test_agent", success=False, error="Test error message", execution_time=0.5)

        assert result.agent_id == "test_agent"
        assert result.success is False
        assert result.result is None
        assert result.error == "Test error message"
        assert result.execution_time == 0.5

    def test_execution_result_to_dict(self):
        """Test ExecutionResult to_dict method."""
        result = ExecutionResult(
            agent_id="test_agent",
            success=True,
            result={"data": "test"},
            error=None,
            execution_time=1.5,
        )

        result_dict = result.to_dict()

        assert result_dict["agent_id"] == "test_agent"
        assert result_dict["success"] is True
        assert result_dict["result"] == {"data": "test"}
        assert result_dict["error"] is None
        assert result_dict["execution_time"] == 1.5
        assert result_dict["timestamp"] == result.timestamp

    def test_execution_result_to_dict_with_error(self):
        """Test ExecutionResult to_dict method with error."""
        result = ExecutionResult(agent_id="test_agent", success=False, error="Test error", execution_time=0.3)

        result_dict = result.to_dict()

        assert result_dict["agent_id"] == "test_agent"
        assert result_dict["success"] is False
        assert result_dict["result"] is None
        assert result_dict["error"] == "Test error"
        assert result_dict["execution_time"] == 0.3


class TestParallelSubagentExecutor:
    """Test suite for ParallelSubagentExecutor class."""

    def test_parallel_executor_initialization(self):
        """Test ParallelSubagentExecutor initialization."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        # Mock configuration
        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5, "timeout_seconds": 120}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert executor.config_manager == config_manager
        assert executor.mcp_client == mcp_client
        assert isinstance(executor.docker_client, DockerMCPClient)
        assert executor.max_workers == 5
        assert executor.timeout_seconds == 120
        assert executor.execution_history == []

    def test_parallel_executor_initialization_with_docker_client(self):
        """Test ParallelSubagentExecutor initialization creates DockerMCPClient."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 3}

        with patch("src.mcp_integration.parallel_executor.DockerMCPClient") as mock_docker_client:
            executor = ParallelSubagentExecutor(config_manager, mcp_client)

            mock_docker_client.assert_called_once()
            assert executor.docker_client == mock_docker_client.return_value

    def test_get_max_workers_valid_config(self):
        """Test _get_max_workers with valid configuration."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 8}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert executor.max_workers == 8

    def test_get_max_workers_exceeds_cap(self):
        """Test _get_max_workers caps at maximum limit."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 20}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert executor.max_workers == 10  # Capped at 10

    def test_get_max_workers_invalid_config(self):
        """Test _get_max_workers with invalid configuration."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": "invalid"}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert executor.max_workers == 5  # Default value

    def test_get_max_workers_missing_config(self):
        """Test _get_max_workers with missing configuration."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert executor.max_workers == 5  # Default value

    @pytest.mark.asyncio
    async def test_select_optimal_client_docker_available(self):
        """Test _select_optimal_client selects Docker when available."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=True)

        # Mock config manager
        config_manager.get_server_config.return_value = Mock(deployment_preference="docker")

        client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

        assert client == executor.docker_client
        assert deployment_type == "docker_preferred"

    @pytest.mark.asyncio
    async def test_select_optimal_client_feature_not_supported(self):
        """Test _select_optimal_client falls back when feature not supported."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability but feature not supported
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=False)

        client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

        assert client == executor.mcp_client
        assert deployment_type == "self_hosted_fallback"

    @pytest.mark.asyncio
    async def test_select_optimal_client_docker_not_available(self):
        """Test _select_optimal_client uses self-hosted when Docker not available."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client not available
        executor.docker_client.is_available = AsyncMock(return_value=False)

        client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

        assert client == executor.mcp_client
        assert deployment_type == "self_hosted_only"

    @pytest.mark.asyncio
    async def test_select_optimal_client_no_tool_name(self):
        """Test _select_optimal_client without specific tool name."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=True)

        # Mock config manager
        config_manager.get_server_config.return_value = Mock(deployment_preference="docker")

        client, deployment_type = await executor._select_optimal_client("test_server")

        assert client == executor.docker_client
        assert deployment_type == "docker_preferred"
        # Should not call supports_feature when no tool name provided
        executor.docker_client.supports_feature.assert_not_called()

    @pytest.mark.asyncio
    async def test_select_optimal_client_self_hosted_preference(self):
        """Test _select_optimal_client respects self-hosted preference."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=True)

        # Mock config manager with self-hosted preference
        config_manager.get_server_config.return_value = Mock(deployment_preference="self-hosted")

        client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

        assert client == executor.mcp_client
        assert deployment_type == "self_hosted_configured"

    @pytest.mark.asyncio
    async def test_select_optimal_client_no_server_config(self):
        """Test _select_optimal_client with no server configuration."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=True)

        # Mock config manager with no server config
        config_manager.get_server_config.return_value = None

        client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

        assert client == executor.docker_client
        assert deployment_type == "docker_default"

    @pytest.mark.asyncio
    async def test_select_optimal_client_logging(self):
        """Test _select_optimal_client logs decisions appropriately."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability but feature not supported
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=False)

        with patch.object(executor, "logger") as mock_logger:
            await executor._select_optimal_client("test_server", "test_tool")

            mock_logger.info.assert_called_once()
            assert "Docker MCP doesn't support" in mock_logger.info.call_args[0][0]

    def test_execution_history_initialization(self):
        """Test execution history is properly initialized."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert executor.execution_history == []
        assert isinstance(executor.execution_history, list)

    def test_timeout_configuration(self):
        """Test timeout configuration is properly set."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert executor.timeout_seconds == 120  # Default timeout

    def test_logger_mixin_inheritance(self):
        """Test ParallelSubagentExecutor inherits from LoggerMixin."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Should have logger attribute from LoggerMixin
        assert hasattr(executor, "logger")
        assert executor.logger is not None

    @pytest.mark.asyncio
    async def test_select_optimal_client_exception_handling(self):
        """Test _select_optimal_client handles exceptions gracefully."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client to raise exception
        executor.docker_client.is_available = AsyncMock(side_effect=Exception("Docker error"))

        # Should fall back to self-hosted client
        client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

        assert client == executor.mcp_client
        assert deployment_type == "self_hosted"

    def test_parallel_executor_attributes_types(self):
        """Test ParallelSubagentExecutor attribute types are correct."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        assert isinstance(executor.max_workers, int)
        assert isinstance(executor.timeout_seconds, int)
        assert isinstance(executor.execution_history, list)
        assert hasattr(executor.config_manager, "get_parallel_execution_config")
        assert hasattr(executor.mcp_client, "__class__")
        assert hasattr(executor.docker_client, "__class__")

    @pytest.mark.asyncio
    async def test_select_optimal_client_with_different_preferences(self):
        """Test _select_optimal_client with different deployment preferences."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=True)

        # Test different preferences
        preferences = ["docker", "self-hosted", "auto", "unknown"]
        expected_clients = [
            executor.docker_client,
            executor.mcp_client,
            executor.docker_client,  # auto defaults to docker when available
            executor.docker_client,  # unknown defaults to docker when available
        ]

        for preference, expected_client in zip(preferences, expected_clients, strict=False):
            config_manager.get_server_config.return_value = Mock(deployment_preference=preference)

            client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

            assert client == expected_client

    @pytest.mark.asyncio
    async def test_select_optimal_client_memory_considerations(self):
        """Test _select_optimal_client with auto preference defaults to docker."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 5}

        executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Mock Docker client availability
        executor.docker_client.is_available = AsyncMock(return_value=True)
        executor.docker_client.supports_feature = AsyncMock(return_value=True)

        # Mock server config with auto preference (no memory consideration in current impl)
        server_config = Mock(deployment_preference="auto")
        server_config.memory_mb = 3000  # This is ignored in current implementation
        config_manager.get_server_config.return_value = server_config

        client, deployment_type = await executor._select_optimal_client("test_server", "test_tool")

        # Should default to docker when available (auto preference)
        assert client == executor.docker_client
        assert deployment_type == "docker_default"

    def test_max_workers_boundary_conditions(self):
        """Test max_workers handles boundary conditions correctly."""
        config_manager = Mock(spec=MCPConfigurationManager)
        mcp_client = Mock(spec=MCPClient)

        # Test minimum boundary
        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 0}
        executor = ParallelSubagentExecutor(config_manager, mcp_client)
        assert executor.max_workers == 5  # Should default to 5

        # Test maximum boundary
        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 50}
        executor = ParallelSubagentExecutor(config_manager, mcp_client)
        assert executor.max_workers == 10  # Should cap at 10

        # Test exact boundary
        config_manager.get_parallel_execution_config.return_value = {"max_concurrent": 10}
        executor = ParallelSubagentExecutor(config_manager, mcp_client)
        assert executor.max_workers == 10  # Should allow exactly 10

    def test_execution_result_timestamp_accuracy(self):
        """Test ExecutionResult timestamp is accurately set."""
        start_time = time.time()

        result = ExecutionResult(agent_id="test_agent", success=True)

        end_time = time.time()

        # Timestamp should be between start and end time
        assert start_time <= result.timestamp <= end_time

        # Timestamp should be a float
        assert isinstance(result.timestamp, float)

    def test_execution_result_immutability_after_creation(self):
        """Test ExecutionResult attributes can be modified after creation."""
        result = ExecutionResult(agent_id="test_agent", success=True, result={"initial": "value"})

        # Attributes should be modifiable (not frozen)
        result.success = False
        result.error = "New error"

        assert result.success is False
        assert result.error == "New error"

    def test_execution_result_with_complex_data_types(self):
        """Test ExecutionResult handles complex data types."""
        complex_result = {
            "nested": {"dict": {"with": "values"}},
            "list": [1, 2, 3],
            "tuple": (4, 5, 6),
            "bool": True,
            "none": None,
        }

        result = ExecutionResult(agent_id="test_agent", success=True, result=complex_result)

        assert result.result == complex_result
        result_dict = result.to_dict()
        assert result_dict["result"] == complex_result
