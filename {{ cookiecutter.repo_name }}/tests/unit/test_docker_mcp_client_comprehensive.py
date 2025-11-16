"""Comprehensive unit tests for Docker MCP Client.

This test suite provides comprehensive coverage for the DockerMCPClient class,
ensuring >80% test coverage by testing all methods, error conditions, and edge cases.
"""

import logging
from unittest.mock import Mock, patch

import pytest

from src.mcp_integration.docker_mcp_client import DockerMCPClient, DockerMCPError


class TestDockerMCPClientComprehensive:
    """Comprehensive tests for DockerMCPClient to achieve >80% coverage."""

    @pytest.fixture
    def docker_client(self):
        """Create DockerMCPClient instance for testing."""
        # Create a fresh instance for each test
        return DockerMCPClient()

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger for testing."""
        return Mock(spec=logging.Logger)

    def test_initialization_comprehensive(self, docker_client):
        """Test comprehensive initialization of DockerMCPClient."""
        # Test all initialization attributes
        assert docker_client.docker_desktop_base_url == "http://localhost:3280"
        assert isinstance(docker_client.mcp_servers, dict)
        assert isinstance(docker_client.docker_servers, dict)
        assert isinstance(docker_client.authenticated_services, dict)
        assert docker_client.logger is not None

        # Test that _discover_docker_servers was called during init
        assert len(docker_client.docker_servers) == 3
        expected_servers = ["github-basic", "filesystem", "sequential-thinking"]
        for server in expected_servers:
            assert server in docker_client.docker_servers

    def test_discover_docker_servers_comprehensive(self, docker_client):
        """Test comprehensive _discover_docker_servers method."""
        # Clear existing servers to test discovery
        docker_client.docker_servers = {}

        # Test discovery
        docker_client._discover_docker_servers()

        # Verify all expected servers are discovered
        assert len(docker_client.docker_servers) == 3

        # Test github-basic server configuration
        github_server = docker_client.docker_servers["github-basic"]
        assert github_server["status"] == "available"
        assert github_server["memory_limit"] == "500MB"
        assert github_server["oauth_required"] is True
        assert "read_file" in github_server["features"]
        assert "list_files" in github_server["features"]
        assert "basic_search" in github_server["features"]

        # Test filesystem server configuration
        filesystem_server = docker_client.docker_servers["filesystem"]
        assert filesystem_server["status"] == "available"
        assert filesystem_server["memory_limit"] == "100MB"
        assert filesystem_server["oauth_required"] is False
        assert "read" in filesystem_server["features"]
        assert "write" in filesystem_server["features"]
        assert "basic_scaffolding" in filesystem_server["features"]

        # Test sequential-thinking server configuration
        thinking_server = docker_client.docker_servers["sequential-thinking"]
        assert thinking_server["status"] == "available"
        assert thinking_server["memory_limit"] == "800MB"
        assert thinking_server["oauth_required"] is False
        assert "step_by_step_reasoning" in thinking_server["features"]
        assert "basic_analysis" in thinking_server["features"]

    @pytest.mark.asyncio
    async def test_is_available_comprehensive(self, docker_client):
        """Test comprehensive is_available method."""
        # Test available servers
        assert await docker_client.is_available("github-basic") is True
        assert await docker_client.is_available("filesystem") is True
        assert await docker_client.is_available("sequential-thinking") is True

        # Test unavailable servers
        assert await docker_client.is_available("non-existent-server") is False
        assert await docker_client.is_available("") is False
        assert await docker_client.is_available("invalid-server") is False

    @pytest.mark.asyncio
    async def test_supports_feature_comprehensive(self, docker_client):
        """Test comprehensive supports_feature method."""
        # Test supported features for github-basic
        assert await docker_client.supports_feature("github-basic", "read_file") is True
        assert await docker_client.supports_feature("github-basic", "list_files") is True
        assert await docker_client.supports_feature("github-basic", "basic_search") is True

        # Test unsupported features for github-basic
        assert await docker_client.supports_feature("github-basic", "write_file") is False
        assert await docker_client.supports_feature("github-basic", "bulk_operations") is False

        # Test supported features for filesystem
        assert await docker_client.supports_feature("filesystem", "read") is True
        assert await docker_client.supports_feature("filesystem", "write") is True
        assert await docker_client.supports_feature("filesystem", "basic_scaffolding") is True

        # Test unsupported features for filesystem
        assert await docker_client.supports_feature("filesystem", "oauth") is False

        # Test supported features for sequential-thinking
        assert await docker_client.supports_feature("sequential-thinking", "step_by_step_reasoning") is True
        assert await docker_client.supports_feature("sequential-thinking", "basic_analysis") is True

        # Test unsupported features for sequential-thinking
        assert await docker_client.supports_feature("sequential-thinking", "write_file") is False

        # Test non-existent server
        assert await docker_client.supports_feature("non-existent-server", "any_feature") is False

        # Test empty server features
        docker_client.docker_servers["empty-server"] = {"status": "available"}
        assert await docker_client.supports_feature("empty-server", "any_feature") is False

    @pytest.mark.asyncio
    async def test_authenticate_service_comprehensive(self, docker_client):
        """Test comprehensive authenticate_service method."""
        # Test authentication for OAuth-required service
        result = await docker_client.authenticate_service("github-basic")
        assert result is True
        assert docker_client.authenticated_services["github-basic"] is True

        # Test authentication for non-OAuth service
        result = await docker_client.authenticate_service("filesystem")
        assert result is True
        assert docker_client.authenticated_services["filesystem"] is True

        # Test authentication for sequential-thinking (no OAuth)
        result = await docker_client.authenticate_service("sequential-thinking")
        assert result is True
        assert docker_client.authenticated_services["sequential-thinking"] is True

        # Test authentication for non-existent server
        with pytest.raises(DockerMCPError, match="Server 'non-existent-server' not found in Docker toolkit"):
            await docker_client.authenticate_service("non-existent-server")

        # Test authentication for empty string
        with pytest.raises(DockerMCPError, match="Server '' not found in Docker toolkit"):
            await docker_client.authenticate_service("")

    @pytest.mark.asyncio
    async def test_call_tool_comprehensive(self, docker_client):
        """Test comprehensive call_tool method."""
        # Test successful call on OAuth-required service (pre-authenticated)
        await docker_client.authenticate_service("github-basic")

        params = {"file_path": "/test/file.py", "lines": 10}
        result = await docker_client.call_tool("github-basic", "read_file", params)

        assert result["success"] is True
        assert result["server"] == "github-basic"
        assert result["tool"] == "read_file"
        assert result["result"] == "Docker MCP result for read_file"
        assert result["deployment"] == "docker_toolkit"
        assert result["memory_limit"] == "500MB"

        # Test successful call on non-OAuth service
        result = await docker_client.call_tool("filesystem", "read", {"path": "/test"})

        assert result["success"] is True
        assert result["server"] == "filesystem"
        assert result["tool"] == "read"
        assert result["deployment"] == "docker_toolkit"
        assert result["memory_limit"] == "100MB"

        # Test call on unavailable server
        with pytest.raises(DockerMCPError, match="Server 'non-existent-server' not available in Docker toolkit"):
            await docker_client.call_tool("non-existent-server", "test_tool", {})

        # Test call on OAuth-required service without authentication
        docker_client.authenticated_services.clear()
        result = await docker_client.call_tool("github-basic", "read_file", {})
        assert result["success"] is True  # Should auto-authenticate
        assert docker_client.authenticated_services["github-basic"] is True

        # Test call with complex parameters
        complex_params = {"nested": {"data": "value"}, "array": [1, 2, 3], "boolean": True, "number": 42.5}
        result = await docker_client.call_tool("sequential-thinking", "basic_analysis", complex_params)
        assert result["success"] is True
        assert result["memory_limit"] == "800MB"

    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, docker_client):
        """Test call_tool error handling."""
        # Mock an exception during tool call
        with patch.object(docker_client, "logger") as mock_logger:
            # Force an exception by making server info access fail
            original_servers = docker_client.docker_servers
            docker_client.docker_servers = {"test-server": {"status": "available"}}  # Missing required fields

            with pytest.raises(DockerMCPError, match="Docker MCP call failed"):
                await docker_client.call_tool("test-server", "test_tool", {})

            # Verify error was logged
            mock_logger.error.assert_called()

            # Restore original servers
            docker_client.docker_servers = original_servers

    def test_get_server_capabilities_comprehensive(self, docker_client):
        """Test comprehensive get_server_capabilities method."""
        # Test capabilities for github-basic
        capabilities = docker_client.get_server_capabilities("github-basic")
        assert capabilities["available"] is True
        assert capabilities["deployment"] == "docker_toolkit"
        assert capabilities["memory_limit"] == "500MB"
        assert capabilities["oauth_required"] is True
        assert capabilities["authenticated"] is False  # Initially not authenticated
        assert "read_file" in capabilities["features"]
        assert "list_files" in capabilities["features"]
        assert "basic_search" in capabilities["features"]

        # Test capabilities for filesystem
        capabilities = docker_client.get_server_capabilities("filesystem")
        assert capabilities["available"] is True
        assert capabilities["deployment"] == "docker_toolkit"
        assert capabilities["memory_limit"] == "100MB"
        assert capabilities["oauth_required"] is False
        assert capabilities["authenticated"] is False
        assert "read" in capabilities["features"]
        assert "write" in capabilities["features"]
        assert "basic_scaffolding" in capabilities["features"]

        # Test capabilities for sequential-thinking
        capabilities = docker_client.get_server_capabilities("sequential-thinking")
        assert capabilities["available"] is True
        assert capabilities["deployment"] == "docker_toolkit"
        assert capabilities["memory_limit"] == "800MB"
        assert capabilities["oauth_required"] is False
        assert capabilities["authenticated"] is False
        assert "step_by_step_reasoning" in capabilities["features"]
        assert "basic_analysis" in capabilities["features"]

        # Test capabilities for non-existent server
        capabilities = docker_client.get_server_capabilities("non-existent-server")
        assert capabilities["available"] is False
        assert len(capabilities) == 1  # Only "available" key

        # Test capabilities with authentication
        docker_client.authenticated_services["github-basic"] = True
        capabilities = docker_client.get_server_capabilities("github-basic")
        assert capabilities["authenticated"] is True

    @pytest.mark.asyncio
    async def test_health_check_comprehensive(self, docker_client):
        """Test comprehensive health_check method."""
        # Ensure clean state for this test - fresh instance should be clean
        assert (
            len(docker_client.authenticated_services) == 0
        ), f"Expected clean state, but got: {docker_client.authenticated_services}"

        # Test health check without authentication
        health = await docker_client.health_check()

        assert health["docker_mcp_available"] is True
        assert health["total_servers"] == 3
        assert health["authenticated_servers"] == 0
        assert "servers" in health
        assert len(health["servers"]) == 3

        # Verify server details in health check
        assert "github-basic" in health["servers"]
        assert "filesystem" in health["servers"]
        assert "sequential-thinking" in health["servers"]

        # Test each server's capabilities in health check
        for _server_name, server_caps in health["servers"].items():
            assert server_caps["available"] is True
            assert server_caps["deployment"] == "docker_toolkit"
            assert server_caps["authenticated"] is False

        # Test health check with some authentication
        # Clear state first then authenticate specific services
        docker_client.authenticated_services.clear()
        await docker_client.authenticate_service("github-basic")
        await docker_client.authenticate_service("filesystem")

        health = await docker_client.health_check()
        assert health["authenticated_servers"] == 2
        assert health["servers"]["github-basic"]["authenticated"] is True
        assert health["servers"]["filesystem"]["authenticated"] is True
        assert health["servers"]["sequential-thinking"]["authenticated"] is False

        # Test health check with all services authenticated
        await docker_client.authenticate_service("sequential-thinking")
        health = await docker_client.health_check()
        assert health["authenticated_servers"] == 3

        # Test health check with no servers
        original_servers = docker_client.docker_servers
        docker_client.docker_servers = {}

        health = await docker_client.health_check()
        assert health["docker_mcp_available"] is False
        assert health["total_servers"] == 0
        assert health["authenticated_servers"] == 0
        assert health["servers"] == {}

        # Restore original servers
        docker_client.docker_servers = original_servers

    def test_docker_mcp_error_comprehensive(self):
        """Test comprehensive DockerMCPError functionality."""
        # Test basic error creation
        error = DockerMCPError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

        # Test error with complex message
        complex_message = "Docker MCP operation failed: server 'test-server' unavailable"
        error = DockerMCPError(complex_message)
        assert str(error) == complex_message

        # Test error inheritance
        assert issubclass(DockerMCPError, Exception)

        # Test error raising
        with pytest.raises(DockerMCPError, match="Test error"):
            raise DockerMCPError("Test error")

    @pytest.mark.asyncio
    async def test_workflow_integration(self, docker_client):
        """Test comprehensive workflow integration."""
        # Test complete workflow: check availability -> authenticate -> call tool -> get capabilities

        # Step 1: Check server availability
        assert await docker_client.is_available("github-basic") is True

        # Step 2: Check feature support
        assert await docker_client.supports_feature("github-basic", "read_file") is True

        # Step 3: Authenticate service
        auth_result = await docker_client.authenticate_service("github-basic")
        assert auth_result is True

        # Step 4: Call tool
        tool_result = await docker_client.call_tool("github-basic", "read_file", {"path": "/test"})
        assert tool_result["success"] is True

        # Step 5: Get server capabilities
        capabilities = docker_client.get_server_capabilities("github-basic")
        assert capabilities["authenticated"] is True

        # Step 6: Health check
        health = await docker_client.health_check()
        assert health["authenticated_servers"] >= 1

    def test_edge_cases_comprehensive(self, docker_client):
        """Test comprehensive edge cases."""
        # Test with empty string
        capabilities = docker_client.get_server_capabilities("")
        assert capabilities["available"] is False

        # Test server with missing features
        docker_client.docker_servers["incomplete-server"] = {
            "status": "available",
            "memory_limit": "1GB",
            "features": [],  # Empty features list
        }
        capabilities = docker_client.get_server_capabilities("incomplete-server")
        assert capabilities["available"] is True
        assert capabilities["features"] == []  # Should handle missing features gracefully

    @pytest.mark.asyncio
    async def test_performance_characteristics(self, docker_client):
        """Test performance characteristics and resource usage."""
        # Test multiple rapid calls
        import time

        start_time = time.time()
        tasks = []
        for _i in range(10):
            task = docker_client.is_available("github-basic")
            tasks.append(task)

        # All calls should complete quickly
        results = []
        for task in tasks:
            results.append(await task)

        end_time = time.time()

        # All results should be True
        assert all(results)

        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0

        # Test memory usage characteristics
        initial_servers = len(docker_client.docker_servers)

        # Simulate adding many servers
        for i in range(100):
            docker_client.docker_servers[f"test-server-{i}"] = {
                "status": "available",
                "memory_limit": "1MB",
                "features": ["test"],
                "oauth_required": False,
            }

        # Health check should still work efficiently
        health = await docker_client.health_check()
        assert health["total_servers"] == initial_servers + 100
        assert health["docker_mcp_available"] is True

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, docker_client):
        """Test concurrent operations handling."""
        import asyncio

        # Test concurrent is_available calls
        tasks = [
            docker_client.is_available("github-basic"),
            docker_client.is_available("filesystem"),
            docker_client.is_available("sequential-thinking"),
            docker_client.is_available("non-existent-server"),
        ]

        results = await asyncio.gather(*tasks)
        assert results == [True, True, True, False]

        # Test concurrent authentication
        auth_tasks = [
            docker_client.authenticate_service("github-basic"),
            docker_client.authenticate_service("filesystem"),
            docker_client.authenticate_service("sequential-thinking"),
        ]

        auth_results = await asyncio.gather(*auth_tasks)
        assert all(auth_results)

        # Test concurrent tool calls
        tool_tasks = [
            docker_client.call_tool("github-basic", "read_file", {"path": "/test1"}),
            docker_client.call_tool("filesystem", "read", {"path": "/test2"}),
            docker_client.call_tool("sequential-thinking", "basic_analysis", {"query": "test"}),
        ]

        tool_results = await asyncio.gather(*tool_tasks)
        assert all(result["success"] for result in tool_results)

    def test_logging_integration(self, docker_client):
        """Test logging integration and coverage."""
        # Test that logger is properly configured
        assert docker_client.logger is not None
        assert hasattr(docker_client.logger, "name")

        # Test logging during server discovery
        with patch.object(docker_client.logger, "info") as mock_info:
            docker_client._discover_docker_servers()

            # Verify logging calls
            mock_info.assert_called()
            log_calls = mock_info.call_args_list
            assert any("Discovering Docker MCP servers" in str(call) for call in log_calls)
            assert any("Discovered 3 Docker MCP servers" in str(call) for call in log_calls)

    @pytest.mark.asyncio
    async def test_authentication_logging(self, docker_client):
        """Test authentication logging coverage."""
        with patch.object(docker_client.logger, "info") as mock_info:
            await docker_client.authenticate_service("github-basic")

            # Verify authentication logging
            mock_info.assert_called()
            log_calls = mock_info.call_args_list
            assert any("Initiating OAuth for github-basic through Docker Desktop" in str(call) for call in log_calls)

    @pytest.mark.asyncio
    async def test_tool_call_logging(self, docker_client):
        """Test tool call logging coverage."""
        with patch.object(docker_client.logger, "debug") as mock_debug:
            await docker_client.call_tool("filesystem", "read", {"path": "/test"})

            # Verify tool call logging
            mock_debug.assert_called()
            log_calls = mock_debug.call_args_list
            assert any("Docker MCP call successful: filesystem.read" in str(call) for call in log_calls)

    @pytest.mark.asyncio
    async def test_error_logging_coverage(self, docker_client):
        """Test error logging coverage."""
        with patch.object(docker_client.logger, "error") as mock_error:
            # Force an error by corrupting server data
            original_servers = docker_client.docker_servers
            docker_client.docker_servers = {"broken-server": {"broken": "data"}}

            try:
                with pytest.raises(DockerMCPError):
                    await docker_client.call_tool("broken-server", "test_tool", {})

                # Verify error logging
                mock_error.assert_called()
                log_calls = mock_error.call_args_list
                assert any("Docker MCP call failed for broken-server.test_tool" in str(call) for call in log_calls)

            finally:
                # Restore original servers
                docker_client.docker_servers = original_servers

    def test_server_discovery_comprehensive_coverage(self, docker_client):
        """Test comprehensive server discovery coverage."""
        # Test that all server properties are set correctly
        servers = docker_client.docker_servers

        # Test github-basic server
        github = servers["github-basic"]
        assert github["status"] == "available"
        assert github["memory_limit"] == "500MB"
        assert github["oauth_required"] is True
        assert len(github["features"]) == 3

        # Test filesystem server
        filesystem = servers["filesystem"]
        assert filesystem["status"] == "available"
        assert filesystem["memory_limit"] == "100MB"
        assert filesystem["oauth_required"] is False
        assert len(filesystem["features"]) == 3

        # Test sequential-thinking server
        thinking = servers["sequential-thinking"]
        assert thinking["status"] == "available"
        assert thinking["memory_limit"] == "800MB"
        assert thinking["oauth_required"] is False
        assert len(thinking["features"]) == 2

        # Test server count
        assert len(servers) == 3

        # Test that all servers have required fields
        for _server_name, server_info in servers.items():
            assert "status" in server_info
            assert "memory_limit" in server_info
            assert "features" in server_info
            assert "oauth_required" in server_info
            assert isinstance(server_info["features"], list)
            assert isinstance(server_info["oauth_required"], bool)

    def test_initialization_attribute_coverage(self, docker_client):
        """Test initialization attribute coverage."""
        # Test all initialized attributes
        assert hasattr(docker_client, "docker_desktop_base_url")
        assert hasattr(docker_client, "mcp_servers")
        assert hasattr(docker_client, "docker_servers")
        assert hasattr(docker_client, "authenticated_services")
        assert hasattr(docker_client, "logger")

        # Test attribute types
        assert isinstance(docker_client.docker_desktop_base_url, str)
        assert isinstance(docker_client.mcp_servers, dict)
        assert isinstance(docker_client.docker_servers, dict)
        assert isinstance(docker_client.authenticated_services, dict)

        # Test initial state
        assert docker_client.docker_desktop_base_url == "http://localhost:3280"
        assert len(docker_client.mcp_servers) == 0  # Initially empty
        assert len(docker_client.docker_servers) == 3  # Populated by discovery
        assert len(docker_client.authenticated_services) == 0  # Initially empty
