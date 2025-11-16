"""MCP Configuration Manager for server configuration and validation."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.utils.logging_mixin import LoggerMixin

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    transport: dict[str, Any] | None = None
    enabled: bool = True
    priority: int = Field(default=100, ge=1, le=1000)
    timeout: int = Field(default=30, ge=1, le=300)
    retry_attempts: int = Field(default=3, ge=0, le=10)

    # NEW: Docker MCP Toolkit integration
    docker_compatible: bool = False
    memory_requirement: str = "unknown"  # e.g., "500MB", "2GB", "8GB"
    deployment_preference: str = "auto"  # "docker", "self-hosted", "auto"
    docker_features: list[str] = Field(default_factory=list)
    self_hosted_features: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_connection_method(self) -> "MCPServerConfig":
        """Ensure either command or transport is specified."""
        if not self.command and not self.transport:
            raise ValueError("Either 'command' or 'transport' must be specified")

        return self


class MCPConfigurationBundle(BaseModel):
    """Complete MCP configuration bundle."""

    version: str = "1.0"
    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")
    global_settings: dict[str, Any] = Field(default_factory=dict)
    parallel_execution: bool = True
    max_concurrent_servers: int = Field(default=5, ge=1, le=20)
    health_check_interval: int = Field(default=60, ge=10, le=3600)

    class Config:
        allow_population_by_field_name = True


class MCPConfigurationManager(LoggerMixin):
    """Manager for MCP server configuration and validation.

    Handles loading, validation, and management of MCP server configurations,
    including support for parallel execution and health monitoring.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to MCP configuration file
        """
        super().__init__()
        self.config_path = config_path or Path(".mcp.json")
        self.backup_path = self.config_path.with_suffix(".json.backup")
        self.configuration: MCPConfigurationBundle | None = None
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load and validate MCP configuration."""
        try:
            if self.config_path.exists():
                with self.config_path.open() as f:
                    raw_config = json.load(f)

                self.configuration = MCPConfigurationBundle(**raw_config)
                self.logger.info(f"Loaded MCP configuration with {len(self.configuration.mcp_servers)} servers")
            else:
                self.logger.warning(f"MCP configuration file not found: {self.config_path}")
                self.configuration = MCPConfigurationBundle(mcpServers={})
        except Exception as e:
            self.logger.error(f"Failed to load MCP configuration: {e}")
            self._load_backup_configuration()

    def _load_backup_configuration(self) -> None:
        """Load backup configuration if main config fails."""
        try:
            if self.backup_path.exists():
                with self.backup_path.open() as f:
                    raw_config = json.load(f)

                self.configuration = MCPConfigurationBundle(**raw_config)
                self.logger.info(f"Loaded backup MCP configuration with {len(self.configuration.mcp_servers)} servers")
            else:
                self.logger.warning("No backup configuration available, using defaults")
                self.configuration = MCPConfigurationBundle(mcpServers={})
        except Exception as e:
            self.logger.error(f"Failed to load backup configuration: {e}")
            self.configuration = MCPConfigurationBundle(mcpServers={})

    def save_configuration(self, backup_current: bool = True) -> bool:
        """Save current configuration to file.

        Args:
            backup_current: Whether to backup current config before saving

        Returns:
            True if save successful, False otherwise
        """
        if not self.configuration:
            self.logger.error("No configuration to save")
            return False

        try:
            # Backup current configuration if requested
            if backup_current and self.config_path.exists():
                self.config_path.rename(self.backup_path)
                self.logger.info("Backed up current configuration")

            # Save new configuration
            with self.config_path.open("w") as f:
                config_dict = self.configuration.dict(by_alias=True)
                json.dump(config_dict, f, indent=2)

            self.logger.info("Saved MCP configuration")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def get_server_config(self, server_name: str) -> MCPServerConfig | None:
        """Get configuration for a specific server.

        Args:
            server_name: Name of the server

        Returns:
            Server configuration or None if not found
        """
        if not self.configuration:
            return None

        return self.configuration.mcp_servers.get(server_name)

    def add_server_config(self, server_name: str, config: MCPServerConfig) -> bool:
        """Add or update server configuration.

        Args:
            server_name: Name of the server
            config: Server configuration

        Returns:
            True if configuration added/updated successfully
        """
        if not self.configuration:
            self.configuration = MCPConfigurationBundle(mcpServers={})

        try:
            self.configuration.mcp_servers[server_name] = config
            self.logger.info(f"Added/updated configuration for server: {server_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add server configuration: {e}")
            return False

    def remove_server_config(self, server_name: str) -> bool:
        """Remove server configuration.

        Args:
            server_name: Name of the server to remove

        Returns:
            True if server removed successfully
        """
        if not self.configuration or server_name not in self.configuration.mcp_servers:
            self.logger.warning(f"Server '{server_name}' not found in configuration")
            return False

        try:
            del self.configuration.mcp_servers[server_name]
            self.logger.info(f"Removed configuration for server: {server_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove server configuration: {e}")
            return False

    def get_enabled_servers(self) -> list[str]:
        """Get list of enabled server names.

        Returns:
            List of enabled server names sorted by priority
        """
        if not self.configuration:
            return []

        enabled_servers = [
            (name, config.priority) for name, config in self.configuration.mcp_servers.items() if config.enabled
        ]

        # Sort by priority (lower number = higher priority)
        enabled_servers.sort(key=lambda x: x[1])
        return [name for name, _ in enabled_servers]

    def get_parallel_execution_config(self) -> dict[str, Any]:
        """Get parallel execution configuration.

        Returns:
            Parallel execution settings
        """
        if not self.configuration:
            return {"enabled": False, "max_concurrent": 1}

        return {
            "enabled": self.configuration.parallel_execution,
            "max_concurrent": self.configuration.max_concurrent_servers,
            "health_check_interval": self.configuration.health_check_interval,
        }

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration.

        Returns:
            Validation results with errors and warnings
        """
        validation_result: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "server_count": 0,
            "enabled_count": 0,
        }

        if not self.configuration:
            validation_result["valid"] = False
            validation_result["errors"].append("No configuration loaded")
            return validation_result

        try:
            # Count servers
            validation_result["server_count"] = len(self.configuration.mcp_servers)
            validation_result["enabled_count"] = len(self.get_enabled_servers())

            # Validate individual servers
            for server_name, config in self.configuration.mcp_servers.items():
                try:
                    # Pydantic validation is already done, check additional rules
                    if not config.command and not config.transport:
                        validation_result["errors"].append(
                            f"Server '{server_name}': either command or transport required",
                        )
                        validation_result["valid"] = False

                    max_priority = 1000
                    if config.priority < 1 or config.priority > max_priority:
                        validation_result["warnings"].append(
                            f"Server '{server_name}': priority should be between 1-1000",
                        )

                except Exception as e:
                    validation_result["errors"].append(f"Server '{server_name}' validation failed: {e}")
                    validation_result["valid"] = False

            # Check for conflicts
            if validation_result["enabled_count"] > self.configuration.max_concurrent_servers:
                validation_result["warnings"].append(
                    f"More enabled servers ({validation_result['enabled_count']}) than max concurrent "
                    f"({self.configuration.max_concurrent_servers})",
                )

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Configuration validation failed: {e}")

        return validation_result

    def get_health_status(self) -> dict[str, Any]:
        """Get configuration health status.

        Returns:
            Health status information
        """
        validation = self.validate_configuration()

        return {
            "configuration_loaded": self.configuration is not None,
            "configuration_valid": validation["valid"],
            "total_servers": validation["server_count"],
            "enabled_servers": validation["enabled_count"],
            "parallel_execution": self.configuration.parallel_execution if self.configuration else False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }
