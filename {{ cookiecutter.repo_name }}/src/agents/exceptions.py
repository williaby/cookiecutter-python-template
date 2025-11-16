"""
Agent Exception Classes for PromptCraft-Hybrid.

This module defines custom exception classes for the agent system, providing
structured error handling and logging for all agent operations. The exceptions
are designed to be informative and actionable for both developers and users.

The module implements:
- AgentError: Base exception class for all agent errors
- AgentConfigurationError: Configuration-related errors
- AgentExecutionError: Runtime execution errors
- AgentRegistrationError: Agent registration errors
- AgentValidationError: Input/output validation errors

Architecture:
    All exceptions inherit from AgentError, which provides common functionality
    like error codes, context information, and logging integration. Each exception
    type is designed for specific error scenarios in the agent lifecycle.

Example:
    ```python
    from src.agents.exceptions import AgentExecutionError

    try:
        result = await agent.execute(agent_input)
    except AgentExecutionError as e:
        logger.error(f"Agent execution failed: {e}")
        # Handle specific error based on error_code
        if e.error_code == "TIMEOUT":
            # Handle timeout scenario
            pass
    ```

Dependencies:
    - typing: For type annotations
    - logging: For error logging integration

Called by:
    - src/agents/base_agent.py: Agent execution errors
    - src/agents/registry.py: Registration errors
    - src/agents/models.py: Validation errors
    - All agent implementations throughout the system

Complexity: O(1) - Exception classes have constant time complexity
"""

import logging
from typing import Any


class AgentError(Exception):
    """
    Base exception class for all agent-related errors.

    This class provides common functionality for all agent exceptions,
    including error codes, context information, and logging integration.
    All other agent exceptions should inherit from this class.

    Attributes:
        message (str): Human-readable error message
        error_code (str): Machine-readable error code
        context (Dict[str, Any]): Additional context information
        agent_id (Optional[str]): ID of the agent that caused the error
        request_id (Optional[str]): ID of the request that caused the error

    Example:
        ```python
        raise AgentError(
            message="Agent failed to process request",
            error_code="PROCESSING_FAILED",
            context={"input_length": 1000, "timeout": 30.0},
            agent_id="security_agent",
            request_id="req-123"
        )
        ```
    """

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """
        Initialize the AgentError.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            context: Additional context information
            agent_id: ID of the agent that caused the error
            request_id: ID of the request that caused the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.agent_id = agent_id
        self.request_id = request_id

        # Log the error for debugging
        logger = logging.getLogger(__name__)
        logger.error(
            "AgentError occurred",
            extra={
                "error_code": error_code,
                "agent_id": agent_id,
                "request_id": request_id,
                "context": context,
                "error_message": message,
            },
        )

    def __str__(self) -> str:
        """Return string representation of the error."""
        parts = [f"[{self.error_code}] {self.message}"]

        if self.agent_id:
            parts.append(f"Agent: {self.agent_id}")

        if self.request_id:
            parts.append(f"Request: {self.request_id}")

        if self.context:
            parts.append(f"Context: {self.context}")

        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "agent_id": self.agent_id,
            "request_id": self.request_id,
        }


class AgentConfigurationError(AgentError):
    """
    Exception raised for agent configuration-related errors.

    This exception is raised when there are issues with agent configuration,
    such as missing required parameters, invalid values, or configuration
    conflicts. It provides specific error codes for different configuration
    scenarios.

    Common error codes:
        - MISSING_REQUIRED_CONFIG: Required configuration parameter is missing
        - INVALID_CONFIG_VALUE: Configuration value is invalid
        - CONFIG_CONFLICT: Configuration parameters conflict with each other
        - CONFIG_VALIDATION_FAILED: Configuration validation failed

    Example:
        ```python
        raise AgentConfigurationError(
            message="Required configuration parameter 'api_key' is missing",
            error_code="MISSING_REQUIRED_CONFIG",
            context={"required_params": ["api_key", "model"]},
            agent_id="openai_agent"
        )
        ```
    """

    def __init__(
        self,
        message: str,
        error_code: str = "CONFIGURATION_ERROR",
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Initialize the AgentConfigurationError."""
        super().__init__(message, error_code, context, agent_id, request_id)


class AgentExecutionError(AgentError):
    """
    Exception raised for agent execution-related errors.

    This exception is raised when there are issues during agent execution,
    such as timeouts, resource exhaustion, or unexpected failures. It provides
    specific error codes for different execution scenarios.

    Common error codes:
        - EXECUTION_TIMEOUT: Agent execution timed out
        - EXECUTION_FAILED: Agent execution failed unexpectedly
        - RESOURCE_EXHAUSTED: Agent ran out of resources
        - INVALID_INPUT: Agent received invalid input
        - OUTPUT_GENERATION_FAILED: Agent failed to generate output

    Example:
        ```python
        raise AgentExecutionError(
            message="Agent execution timed out after 30 seconds",
            error_code="EXECUTION_TIMEOUT",
            context={"timeout": 30.0, "processing_time": 30.1},
            agent_id="slow_agent",
            request_id="req-456"
        )
        ```
    """

    def __init__(
        self,
        message: str,
        error_code: str = "EXECUTION_ERROR",
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Initialize the AgentExecutionError."""
        super().__init__(message, error_code, context, agent_id, request_id)


class AgentRegistrationError(AgentError):
    """
    Exception raised for agent registration-related errors.

    This exception is raised when there are issues with agent registration,
    such as duplicate agent IDs, invalid agent classes, or registration
    validation failures. It provides specific error codes for different
    registration scenarios.

    Common error codes:
        - DUPLICATE_AGENT_ID: Agent ID already exists in registry
        - INVALID_AGENT_CLASS: Agent class does not inherit from BaseAgent
        - REGISTRATION_VALIDATION_FAILED: Agent registration validation failed
        - AGENT_NOT_FOUND: Agent not found in registry

    Example:
        ```python
        raise AgentRegistrationError(
            message="Agent ID 'my_agent' already exists in registry",
            error_code="DUPLICATE_AGENT_ID",
            context={"existing_agent": "MyAgent", "new_agent": "AnotherAgent"},
            agent_id="my_agent"
        )
        ```
    """

    def __init__(
        self,
        message: str,
        error_code: str = "REGISTRATION_ERROR",
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Initialize the AgentRegistrationError."""
        super().__init__(message, error_code, context, agent_id, request_id)


class AgentValidationError(AgentError):
    """
    Exception raised for agent input/output validation errors.

    This exception is raised when there are issues with agent input or output
    validation, such as invalid data types, constraint violations, or schema
    validation failures. It provides specific error codes for different
    validation scenarios.

    Common error codes:
        - INVALID_INPUT_FORMAT: Input data format is invalid
        - INPUT_VALIDATION_FAILED: Input validation failed
        - OUTPUT_VALIDATION_FAILED: Output validation failed
        - SCHEMA_VALIDATION_ERROR: Schema validation error
        - CONSTRAINT_VIOLATION: Data constraint violation

    Example:
        ```python
        raise AgentValidationError(
            message="Input content exceeds maximum length of 10000 characters",
            error_code="CONSTRAINT_VIOLATION",
            context={"max_length": 10000, "actual_length": 15000},
            agent_id="text_agent",
            request_id="req-789"
        )
        ```
    """

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Initialize the AgentValidationError."""
        super().__init__(message, error_code, context, agent_id, request_id)


class AgentTimeoutError(AgentExecutionError):
    """
    Exception raised when agent execution times out.

    This is a specialized version of AgentExecutionError for timeout scenarios.
    It automatically sets the error code to "EXECUTION_TIMEOUT" and provides
    timeout-specific context information.

    Example:
        ```python
        raise AgentTimeoutError(
            message="Agent execution timed out",
            timeout=30.0,
            processing_time=30.1,
            agent_id="slow_agent",
            request_id="req-timeout"
        )
        ```
    """

    def __init__(
        self,
        message: str = "Agent execution timed out",
        timeout: float | None = None,
        processing_time: float | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """
        Initialize the AgentTimeoutError.

        Args:
            message: Human-readable error message
            timeout: Timeout value in seconds
            processing_time: Actual processing time in seconds
            agent_id: ID of the agent that timed out
            request_id: ID of the request that timed out
        """
        context = {}
        if timeout is not None:
            context["timeout"] = timeout
        if processing_time is not None:
            context["processing_time"] = processing_time

        super().__init__(
            message=message,
            error_code="EXECUTION_TIMEOUT",
            context=context,
            agent_id=agent_id,
            request_id=request_id,
        )

    @property
    def timeout(self) -> float | None:
        """Get the timeout value from context."""
        return self.context.get("timeout")

    @property
    def processing_time(self) -> float | None:
        """Get the processing time from context."""
        return self.context.get("processing_time")


# Utility functions for error handling
def create_agent_error(
    error_type: str,
    message: str,
    **kwargs: Any,
) -> "AgentError":
    """
    Factory function to create agent errors based on error type.

    Args:
        error_type: Type of error (configuration, execution, registration, validation)
        message: Human-readable error message
        **kwargs: Additional parameters including:
            - error_code: Machine-readable error code (default: "UNKNOWN_ERROR")
            - context: Additional context information
            - agent_id: ID of the agent that caused the error
            - request_id: ID of the request that caused the error

    Returns:
        AgentError: Appropriate exception instance

    Example:
        ```python
        error = create_agent_error(
            error_type="execution",
            message="Agent failed to process request",
            error_code="PROCESSING_FAILED",
            agent_id="my_agent"
        )
        raise error
        ```
    """
    error_classes = {
        "configuration": AgentConfigurationError,
        "execution": AgentExecutionError,
        "registration": AgentRegistrationError,
        "validation": AgentValidationError,
        "timeout": AgentTimeoutError,
    }

    error_class = error_classes.get(error_type, AgentError)

    # Extract kwargs with defaults
    error_code = kwargs.get("error_code", "UNKNOWN_ERROR")
    context = kwargs.get("context")
    agent_id = kwargs.get("agent_id")
    request_id = kwargs.get("request_id")

    # Handle timeout error special case (different signature)
    if error_type == "timeout":
        return error_class(  # type: ignore[no-any-return]
            message=message,
            agent_id=agent_id,
            request_id=request_id,
        )

    return error_class(  # type: ignore[no-any-return]
        message=message,
        error_code=error_code,
        context=context,
        agent_id=agent_id,
        request_id=request_id,
    )


def handle_agent_error(
    error: Exception,
    agent_id: str | None = None,
    request_id: str | None = None,
) -> "AgentError":
    """
    Convert generic exceptions to AgentError instances.

    This function wraps generic exceptions in AgentError instances to provide
    consistent error handling throughout the agent system.

    Args:
        error: The original exception
        agent_id: ID of the agent that caused the error
        request_id: ID of the request that caused the error

    Returns:
        AgentError: Wrapped exception

    Example:
        ```python
        try:
            # Some operation that might fail
            result = risky_operation()
        except Exception as e:
            agent_error = handle_agent_error(e, agent_id="my_agent")
            raise agent_error
        ```
    """
    if isinstance(error, AgentError):
        return error

    # Convert common exceptions to appropriate AgentError types
    if isinstance(error, ValueError):
        return AgentExecutionError(
            message=str(error),
            error_code="EXECUTION_ERROR",
            context={"original_error": type(error).__name__},
            agent_id=agent_id,
            request_id=request_id,
        )
    if isinstance(error, TimeoutError):
        return AgentTimeoutError(message=str(error), agent_id=agent_id, request_id=request_id)
    if isinstance(error, KeyError):
        return AgentConfigurationError(
            message=str(error),
            error_code="MISSING_REQUIRED_CONFIG",
            context={"original_error": type(error).__name__},
            agent_id=agent_id,
            request_id=request_id,
        )
    return AgentError(
        message=str(error),
        error_code="UNKNOWN_ERROR",
        context={"original_error": type(error).__name__},
        agent_id=agent_id,
        request_id=request_id,
    )


# Export all exception classes
__all__ = [
    "AgentConfigurationError",
    "AgentError",
    "AgentExecutionError",
    "AgentRegistrationError",
    "AgentTimeoutError",
    "AgentValidationError",
    "create_agent_error",
    "handle_agent_error",
]
