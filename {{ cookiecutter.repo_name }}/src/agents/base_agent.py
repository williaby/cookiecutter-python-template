"""
Base Agent Framework for PromptCraft-Hybrid.

This module provides the foundational abstract base class and data models for all agents
in the PromptCraft system. It defines the standardized interface contracts that ensure
consistent behavior across all agent implementations.

The module implements:
- BaseAgent: Abstract base class defining the core agent contract
- Agent lifecycle management (instantiation and execution)
- Configuration and dependency injection support
- Error handling and logging integration

Architecture:
    All agents must inherit from BaseAgent and implement the execute() method.
    The system uses dependency injection for configuration and supports
    runtime configuration overrides through AgentInput.

Example:
    ```python
    from src.agents.base_agent import BaseAgent
    from src.agents.models import AgentInput, AgentOutput
    from src.agents.registry import agent_registry

    @agent_registry.register("my_agent")
    class MyAgent(BaseAgent):
        def __init__(self, config: dict[str, Any]):
            super().__init__(config)

        async def execute(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(
                content="Hello World",
                metadata={"greeting": True},
                confidence=1.0,
                processing_time=0.1,
                agent_id=self.agent_id
            )
    ```

Dependencies:
    - abc: For abstract base class functionality
    - typing: For type annotations
    - asyncio: For async execution support
    - logging: For operation logging
    - time: For performance tracking

Called by:
    - src/agents/registry.py: Agent registration and management
    - src/agents/create_agent.py: CreateAgent implementation
    - src/main.py: FastAPI integration and dependency injection
    - All agent implementations throughout the system

Complexity: O(1) - Abstract interface with no algorithmic complexity
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from src.utils.observability import (
    create_structured_logger,
    get_metrics_collector,
    log_agent_event,
    trace_agent_operation,
)

from .exceptions import AgentConfigurationError, AgentExecutionError, AgentTimeoutError, handle_agent_error
from .models import AgentInput, AgentOutput


class BaseAgent(ABC):
    """
    Abstract base class for all PromptCraft agents.

    This class defines the standardized interface that all agents must implement.
    It provides common functionality for configuration, logging, error handling,
    and lifecycle management while requiring concrete implementations to define
    the core execute() method.

    Attributes:
        agent_id (str): Unique identifier for the agent
        config (Dict[str, Any]): Agent configuration parameters
        logger (logging.Logger): Logger instance for the agent
        _initialized (bool): Whether the agent has been initialized

    Example:
        ```python
        class MyAgent(BaseAgent):
            def __init__(self, config: Dict[str, Any]):
                super().__init__(config)

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Implementation logic here
                return AgentOutput(
                    content="Processed input",
                    metadata={"processed": True},
                    confidence=0.95,
                    processing_time=1.0,
                    agent_id=self.agent_id
                )
        ```
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the BaseAgent.

        Args:
            config: Configuration dictionary containing agent parameters

        Raises:
            AgentConfigurationError: If required configuration is missing
        """
        # Validate config is a dictionary
        if not isinstance(config, dict):
            raise AgentConfigurationError(
                message=f"Configuration must be a dictionary, got {type(config).__name__}",
                error_code="INVALID_CONFIG_TYPE",
            )

        self.config = config.copy()
        self.agent_id = self._validate_agent_id(config.get("agent_id"))
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        self.structured_logger = create_structured_logger(f"agent.{self.agent_id}")
        self.metrics = get_metrics_collector()
        self._initialized = False

        # Initialize the agent
        self._initialize()

    def _validate_agent_id(self, agent_id: str | None) -> str:
        """
        Validate the agent ID from configuration.

        Args:
            agent_id: Agent ID from configuration

        Returns:
            str: Validated agent ID

        Raises:
            AgentConfigurationError: If agent ID is invalid
        """
        if not agent_id:
            raise AgentConfigurationError(
                message="Agent ID is required in configuration",
                error_code="MISSING_REQUIRED_CONFIG",
                context={"required_field": "agent_id"},
            )

        # Validate agent ID format (snake_case)
        if not agent_id.replace("_", "").isalnum():
            raise AgentConfigurationError(
                message=f"Agent ID '{agent_id}' must contain only alphanumeric characters and underscores",
                error_code="INVALID_CONFIG_VALUE",
                context={"agent_id": agent_id, "format": "snake_case"},
            )

        return agent_id

    def _initialize(self) -> None:
        """
        Initialize the agent instance.

        This method is called during construction and can be overridden by
        subclasses to perform additional initialization logic.
        """
        try:
            # Perform common initialization
            self.structured_logger.info(
                "Initializing agent",
                agent_id=self.agent_id,
                event_type="agent_initialization_start",
            )

            # Validate configuration
            self._validate_configuration()

            # Set initialized flag
            self._initialized = True

            # Log successful initialization
            log_agent_event(
                event_type="agent_initialization_success",
                agent_id=self.agent_id,
                message=f"Agent '{self.agent_id}' initialized successfully",
                config_keys=list(self.config.keys()),
            )

        except Exception as e:
            error = handle_agent_error(e, agent_id=self.agent_id)

            # Log initialization failure
            log_agent_event(
                event_type="agent_initialization_failed",
                agent_id=self.agent_id,
                message=f"Failed to initialize agent '{self.agent_id}': {error}",
                level="error",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            raise error from e

    def _validate_configuration(self) -> None:
        """
        Validate the agent configuration.

        This method can be overridden by subclasses to implement specific
        configuration validation logic.

        Raises:
            AgentConfigurationError: If configuration is invalid
        """
        # Base validation - can be extended by subclasses
        if not isinstance(self.config, dict):
            raise AgentConfigurationError(
                message="Configuration must be a dictionary",
                error_code="INVALID_CONFIG_VALUE",
                context={"config_type": type(self.config).__name__},
                agent_id=self.agent_id,
            )

    def _merge_config(self, config_overrides: dict[str, Any] | None) -> dict[str, Any]:
        """
        Merge configuration overrides with base configuration.

        Args:
            config_overrides: Runtime configuration overrides

        Returns:
            Dict[str, Any]: Merged configuration
        """
        if not config_overrides:
            return self.config

        # Create a copy and merge overrides
        merged_config = self.config.copy()
        merged_config.update(config_overrides)

        return merged_config

    def _create_output(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
        processing_time: float = 0.0,
        request_id: str | None = None,
    ) -> AgentOutput:
        """
        Create an AgentOutput instance with standard fields.

        Args:
            content: The response content
            metadata: Additional metadata
            confidence: Confidence score (0.0 to 1.0)
            processing_time: Processing time in seconds
            request_id: Request ID from the input

        Returns:
            AgentOutput: Standardized output object
        """
        return AgentOutput(
            content=content,
            metadata=metadata or {},
            confidence=confidence,
            processing_time=processing_time,
            agent_id=self.agent_id,
            request_id=request_id,
        )

    async def _execute_with_timeout(self, agent_input: AgentInput, timeout: float | None = None) -> AgentOutput:
        """
        Execute the agent with timeout handling.

        Args:
            agent_input: Input for the agent
            timeout: Timeout in seconds (optional)

        Returns:
            AgentOutput: Agent response

        Raises:
            AgentTimeoutError: If execution times out
            AgentExecutionError: If execution fails
        """
        if timeout is None:
            timeout = self.config.get("timeout", 30.0)

        try:
            # Execute with timeout
            return await asyncio.wait_for(self.execute(agent_input), timeout=timeout)

        except TimeoutError as timeout_err:
            raise AgentTimeoutError(
                message=f"Agent execution timed out after {timeout} seconds",
                timeout=timeout,
                agent_id=self.agent_id,
                request_id=agent_input.request_id,
            ) from timeout_err
        except Exception as e:
            error = handle_agent_error(e, agent_id=self.agent_id, request_id=agent_input.request_id)
            raise error from e

    @trace_agent_operation("agent_process")
    async def process(self, agent_input: AgentInput) -> AgentOutput:
        """
        Process an agent input and return the response.

        This method provides the main entry point for agent execution,
        handling configuration merging, error handling, and performance tracking.

        Args:
            agent_input: Input data for the agent

        Returns:
            AgentOutput: Agent response

        Raises:
            AgentError: If processing fails
        """
        if not self._initialized:
            raise AgentExecutionError(
                message="Agent not initialized",
                error_code="AGENT_NOT_INITIALIZED",
                agent_id=self.agent_id,
                request_id=agent_input.request_id,
            )

        start_time = time.time()

        try:
            # Log processing start with structured data
            self.structured_logger.info(
                "Processing request",
                request_id=agent_input.request_id,
                agent_id=self.agent_id,
                event_type="agent_processing_start",
                content_length=len(agent_input.content),
                has_context=bool(agent_input.context),
                has_config_overrides=bool(agent_input.config_overrides),
            )

            # Merge configuration overrides
            if agent_input.config_overrides:
                original_config = self.config.copy()
                self.config = self._merge_config(agent_input.config_overrides)

                # Log config override
                self.structured_logger.debug(
                    "Applied configuration overrides",
                    request_id=agent_input.request_id,
                    agent_id=self.agent_id,
                    config_overrides=agent_input.config_overrides,
                )

                try:
                    # Process with merged config
                    result = await self._execute_with_timeout(agent_input)
                finally:
                    # Restore original config
                    self.config = original_config
            else:
                # Process with original config
                result = await self._execute_with_timeout(agent_input)

            # Update processing time
            processing_time = time.time() - start_time
            # Ensure processing time is non-negative to avoid validation errors
            processing_time = max(0.0, processing_time)
            result.processing_time = processing_time

            # Record success metrics
            self.metrics.increment_counter("agent_executions_success")
            self.metrics.record_duration("agent_execution_duration_seconds", processing_time)

            # Log successful processing
            log_agent_event(
                event_type="agent_processing_success",
                agent_id=self.agent_id,
                message=f"Request {agent_input.request_id} processed successfully in {processing_time:.2f}s",
                request_id=agent_input.request_id,
                processing_time=processing_time,
                confidence=result.confidence,
                content_length=len(result.content),
                metadata_keys=list(result.metadata.keys()),
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            # Ensure processing time is non-negative to avoid validation errors
            processing_time = max(0.0, processing_time)
            error = handle_agent_error(e, agent_id=self.agent_id, request_id=agent_input.request_id)

            # Record failure metrics
            self.metrics.increment_counter("agent_executions_failed")
            self.metrics.record_duration("agent_execution_duration_seconds", processing_time)

            # Log processing failure
            log_agent_event(
                event_type="agent_processing_failed",
                agent_id=self.agent_id,
                message=f"Request {agent_input.request_id} failed after {processing_time:.2f}s: {error}",
                level="error",
                request_id=agent_input.request_id,
                processing_time=processing_time,
                error_code=error.error_code,
                error_type=type(e).__name__,
                error_message=str(e),
            )

            raise error from e

    @abstractmethod
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """
        Execute the agent's core functionality.

        This is the main method that must be implemented by all concrete agent
        classes. It defines the agent's specific behavior and processing logic.

        Args:
            agent_input: Input data for the agent

        Returns:
            AgentOutput: Agent response

        Raises:
            AgentError: If execution fails

        Example:
            ```python
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Process the input
                result = self.process_content(agent_input.content)

                # Return response
                return self._create_output(
                    content=result,
                    metadata={"processed": True},
                    confidence=0.95,
                    request_id=agent_input.request_id
                )
            ```
        """

    def get_capabilities(self) -> dict[str, Any]:
        """
        Get the agent's capabilities.

        This method returns information about what the agent can do,
        which is used by the registry for capability matching.

        Returns:
            Dict[str, Any]: Capabilities dictionary

        Example:
            ```python
            def get_capabilities(self) -> Dict[str, Any]:
                return {
                    "input_types": ["text", "code"],
                    "output_types": ["text", "analysis"],
                    "max_input_length": 10000,
                    "languages": ["python", "javascript"],
                    "frameworks": ["fastapi", "react"]
                }
            ```
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "input_types": ["text"],
            "output_types": ["text"],
            "async_execution": True,
            "timeout_support": True,
            "config_overrides": True,
        }

    def get_status(self) -> dict[str, Any]:
        """
        Get the agent's current status.

        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "initialized": self._initialized,
            "config": self.config,
        }

    def __str__(self) -> str:
        """Return string representation of the agent."""
        return f"{self.__class__.__name__}(agent_id='{self.agent_id}')"

    def __repr__(self) -> str:
        """Return detailed string representation of the agent."""
        return f"{self.__class__.__name__}(agent_id='{self.agent_id}', initialized={self._initialized})"


# Type alias for better code readability
BaseAgentType = BaseAgent

# Export the base agent class
__all__ = ["BaseAgent", "BaseAgentType"]
