"""
Agent Registry System for PromptCraft-Hybrid.

This module provides the centralized registry for discovering and managing agent classes
throughout the PromptCraft system. It implements a decorator-based registration pattern
that enables automatic agent discovery and type-safe agent instantiation.

The registry system supports:
- Decorator-based agent registration
- Type-safe agent class retrieval
- Agent discovery and listing
- Validation of agent interface compliance
- Capability matching for agent selection

Architecture:
    The registry uses a singleton pattern with a global registry instance that
    is populated at import time through decorator usage. All agents must use
    the @agent_registry.register("agent_id") decorator to be discoverable.

Example:
    ```python
    from src.agents.registry import agent_registry
    from src.agents.base_agent import BaseAgent

    @agent_registry.register("my_agent")
    class MyAgent(BaseAgent):
        def __init__(self, config):
            super().__init__(config)

    # Later, retrieve the agent class
    agent_class = agent_registry.get_agent_class("my_agent")
    ```

Dependencies:
    - logging: For registration event logging
    - typing: For type annotations and generic types
    - inspect: For agent class validation
    - src.agents.base_agent: For BaseAgent interface validation

Called by:
    - src/agents/__init__.py: During module initialization
    - src/main.py: For agent instantiation and dependency injection
    - Agent implementations: Through decorator registration
    - FastAPI endpoints: For agent discovery and execution

Complexity: O(1) for registration and retrieval operations
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

from .exceptions import AgentRegistrationError


class AgentRegistry:
    """
    Centralized registry for agent classes and instances.

    This class manages the registration, discovery, and instantiation of agents
    throughout the PromptCraft system. It provides a decorator-based registration
    pattern and supports capability matching for agent selection.

    Attributes:
        _registry (Dict[str, Type]): Registered agent classes by ID
        _instances (Dict[str, Any]): Cached agent instances by ID
        _capabilities (Dict[str, Dict[str, Any]]): Agent capabilities by ID
        _config (Dict[str, Dict[str, Any]]): Agent configurations by ID
        logger (logging.Logger): Logger instance

    Example:
        ```python
        registry = AgentRegistry()

        @registry.register("my_agent")
        class MyAgent(BaseAgent):
            pass

        agent = registry.get_agent("my_agent", {"agent_id": "my_agent"})
        ```
    """

    def __init__(self) -> None:
        """Initialize the agent registry."""
        self._registry: dict[str, type] = {}
        self._instances: dict[str, Any] = {}
        self._capabilities: dict[str, dict[str, Any]] = {}
        self._config: dict[str, dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def register(self, agent_id: str) -> Callable:
        """
        Decorator to register an agent class.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Callable: Decorator function

        Raises:
            AgentRegistrationError: If agent ID already exists or class is invalid

        Example:
            ```python
            @agent_registry.register("my_agent")
            class MyAgent(BaseAgent):
                pass
            ```
        """

        def decorator(agent_class: type) -> type:
            self._register_agent_class(agent_id, agent_class)
            return agent_class

        return decorator

    def _register_agent_class(self, agent_id: str, agent_class: type) -> None:
        """
        Register an agent class with validation.

        Args:
            agent_id: Unique identifier for the agent
            agent_class: Agent class to register

        Raises:
            AgentRegistrationError: If registration fails
        """
        # Validate agent ID
        if not agent_id or not isinstance(agent_id, str):
            raise AgentRegistrationError(
                message="Agent ID must be a non-empty string",
                error_code="INVALID_AGENT_ID",
                context={"agent_id": agent_id},
            )

        # Validate agent ID format (snake_case)
        if not agent_id.replace("_", "").isalnum():
            raise AgentRegistrationError(
                message=f"Agent ID '{agent_id}' must contain only alphanumeric characters and underscores",
                error_code="INVALID_AGENT_ID",
                context={"agent_id": agent_id, "format": "snake_case"},
            )

        # Check for duplicate registration
        if agent_id in self._registry:
            existing_class = self._registry[agent_id]
            raise AgentRegistrationError(
                message=f"Agent ID '{agent_id}' already exists",
                error_code="DUPLICATE_AGENT_ID",
                context={
                    "agent_id": agent_id,
                    "existing_class": existing_class.__name__,
                    "new_class": agent_class.__name__,
                },
                agent_id=agent_id,
            )

        # Validate agent class
        self._validate_agent_class(agent_id, agent_class)

        # Register the agent class
        self._registry[agent_id] = agent_class

        # Log successful registration
        self.logger.info(
            "Registered agent with class",
            extra={"agent_id": agent_id, "agent_class": agent_class.__name__},
        )

    def _validate_agent_class(self, agent_id: str, agent_class: type) -> None:
        """
        Validate that an agent class implements the BaseAgent interface.

        Args:
            agent_id: Agent identifier
            agent_class: Agent class to validate

        Raises:
            AgentRegistrationError: If agent class is invalid
        """
        # Import here to avoid circular imports
        from .base_agent import BaseAgent  # noqa: PLC0415

        # Check if class inherits from BaseAgent
        if not issubclass(agent_class, BaseAgent):
            raise AgentRegistrationError(
                message=f"Agent class '{agent_class.__name__}' must inherit from BaseAgent",
                error_code="INVALID_AGENT_CLASS",
                context={"agent_class": agent_class.__name__, "required_base": "BaseAgent"},
                agent_id=agent_id,
            )

        # Check if class has required methods
        required_methods = ["execute"]
        for method_name in required_methods:
            if not hasattr(agent_class, method_name):
                raise AgentRegistrationError(
                    message=f"Agent class '{agent_class.__name__}' missing required method '{method_name}'",
                    error_code="MISSING_REQUIRED_METHOD",
                    context={"agent_class": agent_class.__name__, "missing_method": method_name},
                    agent_id=agent_id,
                )

            # Check if method is abstract (not implemented)
            method = getattr(agent_class, method_name)
            if getattr(method, "__isabstractmethod__", False):
                raise AgentRegistrationError(
                    message=f"Agent class '{agent_class.__name__}' missing required method '{method_name}'",
                    error_code="MISSING_REQUIRED_METHOD",
                    context={"agent_class": agent_class.__name__, "missing_method": method_name},
                    agent_id=agent_id,
                )

        # Check if execute method is async
        execute_method = agent_class.execute
        if not inspect.iscoroutinefunction(execute_method):
            raise AgentRegistrationError(
                message=f"Agent class '{agent_class.__name__}' execute method must be async",
                error_code="INVALID_METHOD_SIGNATURE",
                context={"agent_class": agent_class.__name__, "method": "execute", "required": "async"},
                agent_id=agent_id,
            )

    def get_agent_class(self, agent_id: str) -> type:
        """
        Get a registered agent class by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Type: Agent class

        Raises:
            AgentRegistrationError: If agent not found
        """
        if agent_id not in self._registry:
            raise AgentRegistrationError(
                message=f"Agent '{agent_id}' not found in registry",
                error_code="AGENT_NOT_FOUND",
                context={"agent_id": agent_id, "available_agents": list(self._registry.keys())},
                agent_id=agent_id,
            )

        return self._registry[agent_id]

    def get_agent(self, agent_id: str, config: dict[str, Any]) -> Any:
        """
        Get or create an agent instance.

        Args:
            agent_id: Agent identifier
            config: Agent configuration

        Returns:
            Any: Agent instance

        Raises:
            AgentRegistrationError: If agent not found or instantiation fails
        """
        # Get agent class
        agent_class = self.get_agent_class(agent_id)

        # Ensure agent_id is in config
        if "agent_id" not in config:
            config = config.copy()
            config["agent_id"] = agent_id

        try:
            # Create instance
            instance = agent_class(config)

            # Store capabilities if instance supports it
            if hasattr(instance, "get_capabilities"):
                self._capabilities[agent_id] = instance.get_capabilities()

            return instance

        except Exception as e:
            raise AgentRegistrationError(
                message=f"Failed to instantiate agent '{agent_id}': {e!s}",
                error_code="INSTANTIATION_FAILED",
                context={"agent_id": agent_id, "agent_class": agent_class.__name__, "config": config, "error": str(e)},
                agent_id=agent_id,
            ) from e

    def get_cached_agent(self, agent_id: str, config: dict[str, Any]) -> Any:
        """
        Get a cached agent instance or create a new one.

        Args:
            agent_id: Agent identifier
            config: Agent configuration

        Returns:
            Any: Agent instance
        """
        # Check if instance exists and config hasn't changed
        if agent_id in self._instances:
            stored_config = self._config.get(agent_id, {})
            if stored_config == config:
                return self._instances[agent_id]

        # Create new instance
        instance = self.get_agent(agent_id, config)

        # Cache instance and config
        self._instances[agent_id] = instance
        self._config[agent_id] = config.copy()

        return instance

    def list_agents(self) -> list[str]:
        """
        List all registered agent IDs.

        Returns:
            List[str]: List of agent IDs
        """
        return list(self._registry.keys())

    def list_agent_classes(self) -> dict[str, type]:
        """
        List all registered agent classes.

        Returns:
            Dict[str, Type]: Dictionary of agent ID to class mappings
        """
        return self._registry.copy()

    def get_agent_info(self, agent_id: str) -> dict[str, Any]:
        """
        Get information about a registered agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict[str, Any]: Agent information

        Raises:
            AgentRegistrationError: If agent not found
        """
        if agent_id not in self._registry:
            raise AgentRegistrationError(
                message=f"Agent '{agent_id}' not found in registry",
                error_code="AGENT_NOT_FOUND",
                context={"agent_id": agent_id},
                agent_id=agent_id,
            )

        agent_class = self._registry[agent_id]
        capabilities = self._capabilities.get(agent_id, {})
        config = self._config.get(agent_id, {})

        return {
            "agent_id": agent_id,
            "agent_class": agent_class.__name__,
            "module": agent_class.__module__,
            "capabilities": capabilities,
            "config": config,
            "is_cached": agent_id in self._instances,
        }

    def find_agents_by_capability(self, capability: str, value: Any = None) -> list[str]:
        """
        Find agents that have a specific capability.

        Args:
            capability: Capability name to search for
            value: Optional specific value to match

        Returns:
            List[str]: List of agent IDs that match the capability
        """
        matching_agents = []

        for agent_id, capabilities in self._capabilities.items():
            if capability in capabilities and (value is None or capabilities[capability] == value):
                matching_agents.append(agent_id)

        return matching_agents

    def find_agents_by_type(self, input_type: str, output_type: str | None = None) -> list[str]:
        """
        Find agents by input/output type capability.

        Args:
            input_type: Required input type
            output_type: Optional required output type

        Returns:
            List[str]: List of matching agent IDs
        """
        matching_agents = []

        for agent_id, capabilities in self._capabilities.items():
            input_types = capabilities.get("input_types", [])
            output_types = capabilities.get("output_types", [])

            if input_type in input_types and (output_type is None or output_type in output_types):
                matching_agents.append(agent_id)

        return matching_agents

    def unregister(self, agent_id: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentRegistrationError: If agent not found
        """
        if agent_id not in self._registry:
            raise AgentRegistrationError(
                message=f"Agent '{agent_id}' not found in registry",
                error_code="AGENT_NOT_FOUND",
                context={"agent_id": agent_id},
                agent_id=agent_id,
            )

        # Remove from all registries
        del self._registry[agent_id]

        if agent_id in self._instances:
            del self._instances[agent_id]

        if agent_id in self._capabilities:
            del self._capabilities[agent_id]

        if agent_id in self._config:
            del self._config[agent_id]

        self.logger.info("Unregistered agent", extra={"agent_id": agent_id})

    def clear(self) -> None:
        """Clear all registrations."""
        self._registry.clear()
        self._instances.clear()
        self._capabilities.clear()
        self._config.clear()
        self.logger.info("Cleared all agent registrations")

    def get_registry_status(self) -> dict[str, Any]:
        """
        Get the current status of the registry.

        Returns:
            Dict[str, Any]: Registry status information
        """
        return {
            "total_agents": len(self._registry),
            "cached_instances": len(self._instances),
            "agents_with_capabilities": len(self._capabilities),
            "registered_agents": list(self._registry.keys()),
        }

    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self._registry

    def __len__(self) -> int:
        """Get the number of registered agents."""
        return len(self._registry)

    def __iter__(self) -> Any:
        """Iterate over registered agent IDs."""
        return iter(self._registry.keys())


# Global registry instance
agent_registry = AgentRegistry()

# Export registry and related functions
__all__ = ["AgentRegistry", "agent_registry"]
