"""MCP Client for external service communication."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.utils.logging_mixin import LoggerMixin

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Base exception for MCP client operations."""


class MCPClient(LoggerMixin):
    """Client for communicating with MCP servers.

    Provides connection management, message serialization, and error handling
    for MCP protocol communication with external services like Zen MCP Server.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize MCP client with configuration.

        Args:
            config_path: Path to MCP configuration file (defaults to .mcp.json)
        """
        super().__init__()
        self.config_path = config_path or Path(".mcp.json")
        self.servers: dict[str, dict[str, Any]] = {}
        self.connections: dict[str, Any] = {}
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load MCP server configuration from file."""
        try:
            if self.config_path.exists():
                with self.config_path.open() as f:
                    config = json.load(f)
                    self.servers = config.get("mcpServers", {})
                    self.logger.info(f"Loaded {len(self.servers)} MCP server configurations")
            else:
                self.logger.warning(f"MCP configuration file not found: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load MCP configuration: {e}")
            raise MCPClientError(f"Configuration load failed: {e}") from e

    async def connect_server(self, server_name: str) -> bool:
        """Connect to a specific MCP server.

        Args:
            server_name: Name of the server to connect to

        Returns:
            True if connection successful, False otherwise
        """
        if server_name not in self.servers:
            self.logger.error(f"Server '{server_name}' not found in configuration")
            return False

        try:
            # Placeholder for actual MCP connection logic
            # In real implementation, this would establish MCP protocol connection
            self.connections[server_name] = {"status": "connected", "server": server_name}
            self.logger.info(f"Connected to MCP server: {server_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server {server_name}: {e}")
            return False

    def disconnect_server(self, server_name: str) -> bool:
        """Disconnect from a specific MCP server.

        Args:
            server_name: Name of the server to disconnect from

        Returns:
            True if disconnection successful, False otherwise
        """
        if server_name not in self.connections:
            self.logger.warning(f"Server '{server_name}' not connected")
            return False

        try:
            # Placeholder for actual MCP disconnection logic
            del self.connections[server_name]
            self.logger.info(f"Disconnected from MCP server: {server_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect from server {server_name}: {e}")
            return False

    async def send_message(self, server_name: str, message: dict[str, Any]) -> dict[str, Any]:
        """Send message to MCP server and get response.

        Args:
            server_name: Name of the target server
            message: Message payload to send

        Returns:
            Response from the server

        Raises:
            MCPClientError: If server not connected or communication fails
        """
        if server_name not in self.connections:
            raise MCPClientError(f"Server '{server_name}' not connected")

        try:
            # Placeholder for actual MCP message sending
            # In real implementation, this would use MCP protocol
            response = {
                "status": "success",
                "server": server_name,
                "echo": message,
                "timestamp": asyncio.get_event_loop().time(),
            }
            self.logger.debug(f"Sent message to {server_name}: {message}")
            return response
        except Exception as e:
            self.logger.error(f"Failed to send message to {server_name}: {e}")
            raise MCPClientError(f"Message send failed: {e}") from e

    def get_connected_servers(self) -> list[str]:
        """Get list of currently connected servers.

        Returns:
            List of connected server names
        """
        return list(self.connections.keys())

    def get_server_status(self, server_name: str) -> dict[str, Any]:
        """Get status information for a specific server.

        Args:
            server_name: Name of the server

        Returns:
            Status information dictionary
        """
        if server_name not in self.servers:
            return {"status": "not_configured"}

        if server_name not in self.connections:
            return {"status": "disconnected", "configured": True}

        return {"status": "connected", "configured": True, "connection": self.connections[server_name]}

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all configured servers.

        Returns:
            Health status for all servers
        """
        health_status: dict[str, Any] = {
            "overall_status": "healthy",
            "servers": {},
            "total_configured": len(self.servers),
            "total_connected": len(self.connections),
        }

        for server_name in self.servers:
            status = self.get_server_status(server_name)
            health_status["servers"][server_name] = status

            if status["status"] != "connected":
                health_status["overall_status"] = "degraded"

        return health_status
