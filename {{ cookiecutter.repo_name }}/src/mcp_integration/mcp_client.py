"""
Abstract MCP Client Interface for PromptCraft-Hybrid.

This module provides abstract interfaces and implementations for MCP (Model Context Protocol)
integration, supporting both mock development environments and real Zen MCP Server integration.
It implements comprehensive error handling, graceful fallback strategies, and health monitoring
for robust query orchestration.

Key Features:
    - Abstract base class for MCP client implementations
    - Mock implementation for development and testing
    - Real Zen MCP Server integration interface
    - Comprehensive error handling with typed exceptions
    - Health check and connection management
    - Retry logic with exponential backoff
    - Graceful fallback for service unavailability
    - Performance monitoring and metrics

Architecture:
    The MCP client interface follows the Abstract Factory pattern to provide seamless
    switching between development mock implementations and production Zen MCP Server
    integration. This enables parallel development while external dependencies are
    being established.

Error Handling Strategy:
    - Typed exceptions for specific error scenarios
    - Circuit breaker pattern for cascading failure prevention
    - Graceful degradation with fallback responses
    - Comprehensive logging and monitoring
    - User-friendly error messages with actionable guidance

Dependencies:
    - src.core.query_counselor: For WorkflowStep and Response models
    - tenacity: For retry logic with exponential backoff
    - asyncio: For asynchronous operation support
    - logging: For comprehensive operation logging
    - typing: For type safety and interface contracts

Called by:
    - src.core.query_counselor: For agent orchestration workflows
    - src.main: For service initialization and health checks
    - Integration tests: For end-to-end workflow validation

Time Complexity: O(n) for n agent orchestration requests
Space Complexity: O(k) where k is the number of concurrent connections
"""

import asyncio
import contextlib
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import get_settings
from src.utils.secure_random import secure_random

logger = logging.getLogger(__name__)

# Constants for mock client behavior
DEGRADED_ERROR_THRESHOLD = 3
MAX_QUERY_LENGTH = 10000
HTTP_OK_STATUS = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503


class MCPConnectionState(str, Enum):
    """MCP connection states for health monitoring."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    FAILED = "failed"


class MCPErrorType(str, Enum):
    """Types of MCP errors for structured error handling."""

    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"
    VALIDATION_ERROR = "validation_error"
    SERVICE_ERROR = "service_error"
    INVALID_RESPONSE = "invalid_response"


class MCPError(Exception):
    """Base exception for MCP client operations."""

    def __init__(
        self,
        message: str,
        error_type: MCPErrorType = MCPErrorType.UNKNOWN_ERROR,
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
    ) -> None:
        """Initialize MCP error with structured information.

        Args:
            message: Human-readable error message
            error_type: Categorized error type for handling logic
            details: Additional error context and metadata
            retry_after: Suggested retry delay in seconds
        """
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.retry_after = retry_after


class MCPConnectionError(MCPError):
    """Connection-related MCP errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, MCPErrorType.CONNECTION_ERROR, details)


class MCPTimeoutError(MCPError):
    """Timeout-related MCP errors."""

    def __init__(self, message: str, timeout_seconds: float, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, MCPErrorType.TIMEOUT_ERROR, details)
        self.timeout_seconds = timeout_seconds


class MCPServiceUnavailableError(MCPError):
    """Service unavailability errors."""

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, MCPErrorType.SERVICE_UNAVAILABLE, details, retry_after)


class MCPRateLimitError(MCPError):
    """Rate limiting errors."""

    def __init__(self, message: str, retry_after: int, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, MCPErrorType.RATE_LIMIT_ERROR, details, retry_after)


class MCPAuthenticationError(MCPError):
    """Authentication-related MCP errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "AUTHENTICATION_FAILED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, MCPErrorType.AUTHENTICATION_ERROR, details)
        self.error_code = error_code


class MCPValidationError(MCPError):
    """Validation-related MCP errors."""

    def __init__(
        self,
        message: str,
        validation_errors: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, MCPErrorType.INVALID_REQUEST, details)
        self.validation_errors = validation_errors or {}


class MCPHealthStatus(BaseModel):
    """Health status information for MCP connections."""

    connection_state: MCPConnectionState
    last_successful_request: float | None = Field(None, description="Timestamp of last successful request")
    error_count: int = Field(0, description="Number of consecutive errors")
    response_time_ms: float | None = Field(None, description="Average response time in milliseconds")
    capabilities: list[str] = Field(default_factory=list, description="Available MCP server capabilities")
    server_version: str | None = Field(None, description="MCP server version information")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional health metadata")


class WorkflowStep(BaseModel):
    """Workflow step for agent orchestration (imported from query_counselor)."""

    step_id: str
    agent_id: str
    input_data: dict[str, Any]
    dependencies: list[str] = Field(default_factory=list)
    timeout_seconds: int = 30


class Response(BaseModel):
    """Response from agent execution (imported from query_counselor)."""

    agent_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    success: bool = True
    error_message: str | None = None


class MCPClientInterface(ABC):
    """
    Abstract interface for MCP client implementations.

    Defines the contract for all MCP client implementations, ensuring consistent
    behavior between mock development clients and production Zen MCP integrations.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to MCP server.

        Returns:
            bool: True if connection successful, False otherwise

        Raises:
            MCPConnectionError: If connection cannot be established
        """

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from MCP server and cleanup resources.

        Returns:
            bool: True if disconnection successful, False otherwise
        """

    @abstractmethod
    async def health_check(self) -> MCPHealthStatus:
        """
        Perform comprehensive health check on MCP connection.

        Returns:
            MCPHealthStatus: Current health status and metrics

        Raises:
            MCPError: If health check fails
        """

    @abstractmethod
    async def validate_query(self, query: str) -> dict[str, Any]:
        """
        Validate and sanitize user query for security.

        Args:
            query: Raw user query string

        Returns:
            Dict containing validation results with keys:
                - is_valid: bool
                - sanitized_query: str
                - potential_issues: List[str]

        Raises:
            MCPError: If validation service fails
        """

    @abstractmethod
    async def orchestrate_agents(self, workflow_steps: list[WorkflowStep]) -> list[Response]:
        """
        Orchestrate multi-agent workflow execution.

        Args:
            workflow_steps: List of workflow steps to execute

        Returns:
            List[Response]: Responses from all agents

        Raises:
            MCPError: If orchestration fails
            MCPTimeoutError: If execution exceeds timeout
        """

    @abstractmethod
    async def get_capabilities(self) -> list[str]:
        """
        Get list of available MCP server capabilities.

        Returns:
            List[str]: Available capability names

        Raises:
            MCPError: If capability query fails
        """


class MockMCPClient(MCPClientInterface):
    """
    Mock MCP client implementation for development and testing.

    Provides realistic simulation of MCP server behavior without requiring
    external dependencies. Supports error injection for testing error handling
    and fallback mechanisms.
    """

    def __init__(
        self,
        simulate_failures: bool = False,
        failure_rate: float = 0.1,
        response_delay: float = 0.1,
        max_agents: int = 10,
    ) -> None:
        """
        Initialize mock MCP client with configurable behavior.

        Args:
            simulate_failures: Whether to randomly simulate failures
            failure_rate: Probability of simulated failures (0.0-1.0)
            response_delay: Simulated processing delay in seconds
            max_agents: Maximum number of agents to support
        """
        self.simulate_failures = simulate_failures
        self.failure_rate = failure_rate
        self.response_delay = response_delay
        self.max_agents = max_agents
        self.connection_state = MCPConnectionState.DISCONNECTED
        self.error_count = 0
        self.last_successful_request: float | None = None
        self.request_count = 0

        # Mock capabilities
        self.capabilities = [
            "agent_orchestration",
            "query_validation",
            "workflow_execution",
            "health_monitoring",
            "error_recovery",
        ]

    async def connect(self) -> bool:
        """Simulate MCP server connection."""
        try:
            self.connection_state = MCPConnectionState.CONNECTING
            await asyncio.sleep(0.05)  # Simulate connection time

            if self.simulate_failures and self._should_fail():
                self.connection_state = MCPConnectionState.FAILED
                self.error_count += 1
                raise MCPConnectionError("Mock connection failure", {"mock": True, "attempt": self.error_count})

            self.connection_state = MCPConnectionState.CONNECTED
            self.error_count = 0
            logger.info("Mock MCP client connected successfully")
            return True

        except Exception as e:
            self.connection_state = MCPConnectionState.FAILED
            logger.error(f"Mock MCP connection failed: {e}")
            raise

    async def disconnect(self) -> bool:
        """Simulate MCP server disconnection."""
        try:
            await asyncio.sleep(0.01)  # Simulate disconnection time
            self.connection_state = MCPConnectionState.DISCONNECTED
            logger.info("Mock MCP client disconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Mock MCP disconnection failed: {e}")
            return False

    async def health_check(self) -> MCPHealthStatus:
        """Simulate health check with realistic metrics."""
        start_time = time.time()

        # Simulate health check processing
        await asyncio.sleep(0.02)

        # Calculate mock response time
        response_time_ms = (time.time() - start_time) * 1000

        # Determine connection state
        if self.simulate_failures and self._should_fail():
            self.error_count += 1
            state = (
                MCPConnectionState.DEGRADED
                if self.error_count < DEGRADED_ERROR_THRESHOLD
                else MCPConnectionState.FAILED
            )
        else:
            state = self.connection_state
            self.error_count = max(0, self.error_count - 1)

        return MCPHealthStatus(
            connection_state=state,
            last_successful_request=self.last_successful_request,
            error_count=self.error_count,
            response_time_ms=response_time_ms,
            capabilities=self.capabilities,
            server_version="MockMCP-1.0.0",
            metadata={
                "mock_client": True,
                "request_count": self.request_count,
                "failure_simulation": self.simulate_failures,
                "failure_rate": self.failure_rate,
            },
        )

    async def validate_query(self, query: str) -> dict[str, Any]:
        """Simulate query validation with security checks."""
        start_time = time.time()

        if not query or not query.strip():
            return {"is_valid": False, "sanitized_query": "", "potential_issues": ["Empty query"]}

        # Simulate processing delay
        await asyncio.sleep(self.response_delay * 0.1)

        # Mock validation logic
        sanitized_query = query.strip()
        potential_issues = []

        # Check for potential security issues (mock)
        if len(query) > MAX_QUERY_LENGTH:
            potential_issues.append("Query exceeds maximum length")

        if any(keyword in query.lower() for keyword in ["<script>", "javascript:", "eval("]):
            potential_issues.append("Potential XSS attempt detected")
            sanitized_query = sanitized_query.replace("<script>", "").replace("javascript:", "").replace("eval(", "")

        # Simulate failure if configured
        if self.simulate_failures and self._should_fail():
            self.error_count += 1
            raise MCPServiceUnavailableError("Mock validation service unavailable", 5, {"mock": True})

        self.last_successful_request = time.time()
        self.request_count += 1

        logger.debug(f"Mock query validation completed in {(time.time() - start_time) * 1000:.2f}ms")

        return {
            "is_valid": len(potential_issues) == 0,
            "sanitized_query": sanitized_query,
            "potential_issues": potential_issues,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    async def orchestrate_agents(self, workflow_steps: list[WorkflowStep]) -> list[Response]:
        """Simulate multi-agent workflow orchestration."""
        start_time = time.time()

        if len(workflow_steps) > self.max_agents:
            raise MCPError(
                f"Too many agents requested: {len(workflow_steps)} > {self.max_agents}",
                MCPErrorType.INVALID_REQUEST,
                {"max_agents": self.max_agents, "requested": len(workflow_steps)},
            )

        # Simulate failure if configured
        if self.simulate_failures and self._should_fail():
            self.error_count += 1
            raise MCPServiceUnavailableError("Mock orchestration service degraded", 10, {"mock": True})

        responses = []

        for step in workflow_steps:
            step_start_time = time.time()

            # Simulate processing time
            await asyncio.sleep(self.response_delay)

            # Generate mock response
            response = Response(
                agent_id=step.agent_id,
                content=f"Mock response from {step.agent_id} for step {step.step_id}",
                confidence=0.85 + (hash(step.agent_id) % 15) / 100,  # Deterministic but varied confidence
                processing_time=time.time() - step_start_time,
                metadata={
                    "mock": True,
                    "step_id": step.step_id,
                    "agent_type": step.input_data.get("agent_type", "unknown"),
                    "capabilities": step.input_data.get("capabilities", []),
                    "simulated_delay": self.response_delay,
                },
            )
            responses.append(response)

        self.last_successful_request = time.time()
        self.request_count += 1

        total_time = time.time() - start_time
        logger.info(f"Mock orchestration completed {len(workflow_steps)} steps in {total_time:.3f}s")

        return responses

    async def get_capabilities(self) -> list[str]:
        """Get mock MCP server capabilities."""
        await asyncio.sleep(0.01)  # Simulate small delay

        if self.simulate_failures and self._should_fail():
            raise MCPConnectionError("Mock capabilities service unavailable", {"mock": True})

        return self.capabilities.copy()

    def _should_fail(self) -> bool:
        """Determine if mock operation should fail based on failure rate."""
        return secure_random.random() < self.failure_rate


class ZenMCPClient(MCPClientInterface):
    """
    Real Zen MCP Server client implementation.

    Provides integration with the actual Zen MCP Server for production use.
    Implements comprehensive error handling, retry logic, and health monitoring
    for robust multi-agent orchestration.
    """

    def __init__(
        self,
        server_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        """
        Initialize Zen MCP client with connection parameters.

        Args:
            server_url: Zen MCP Server endpoint URL
            api_key: Authentication API key (if required)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.connection_state = MCPConnectionState.DISCONNECTED
        self.session: Any | None = None  # HTTP session for connection pooling
        self.error_count = 0
        self.last_successful_request: float | None = None

    async def connect(self) -> bool:
        """
        Establish connection to Zen MCP Server.

        Implements real HTTP client connection with authentication and health validation.
        """
        try:
            if httpx is None:
                raise ImportError("httpx is not installed")

            self.connection_state = MCPConnectionState.CONNECTING
            logger.info(f"Connecting to Zen MCP Server at {self.server_url}")

            # Set up headers
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Initialize HTTP client session with connection pooling
            self.session = httpx.AsyncClient(
                base_url=self.server_url,
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

            # Perform server health check and handshake
            try:
                response = await self.session.get("/health")
                if response.status_code == HTTP_UNAUTHORIZED:  # Unauthorized
                    error_data = {}
                    try:
                        if response.headers.get("content-type", "").startswith("application/json"):
                            error_data = response.json()
                    except Exception as e:
                        logger.debug("Failed to parse JSON error response: %s", e)
                    raise MCPAuthenticationError(
                        "Authentication failed",
                        error_code="INVALID_API_KEY",
                        details={"status_code": response.status_code, "response": response.text[:200], **error_data},
                    )
                if response.status_code != HTTP_OK_STATUS:
                    raise MCPConnectionError(
                        f"Server health check failed: HTTP {response.status_code}",
                        {"status_code": response.status_code, "response": response.text[:200]},
                    )

                # Validate server response
                health_data = (
                    response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                )
                if health_data.get("status") != "healthy":
                    logger.warning(f"Server health check returned non-healthy status: {health_data}")

            except httpx.ConnectError as e:
                raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e
            except httpx.TimeoutException as e:
                raise MCPTimeoutError(f"Connection timeout to MCP server: {e}", self.timeout) from e
            except (MCPAuthenticationError, MCPValidationError):
                # Re-raise MCP-specific errors without wrapping
                raise
            except Exception as e:
                raise MCPConnectionError(f"Unexpected connection error: {e}") from e

            self.connection_state = MCPConnectionState.CONNECTED
            self.error_count = 0
            logger.info("Successfully connected to Zen MCP Server")
            return True

        except (MCPAuthenticationError, MCPValidationError) as e:
            # Re-raise MCP-specific errors without wrapping
            self.connection_state = MCPConnectionState.FAILED
            self.error_count += 1
            logger.error(f"Failed to connect to Zen MCP Server: {e}")

            # Cleanup session on connection failure
            if hasattr(self, "session") and self.session:
                await self.session.aclose()
                self.session = None

            raise
        except Exception as e:
            self.connection_state = MCPConnectionState.FAILED
            self.error_count += 1
            logger.error(f"Failed to connect to Zen MCP Server: {e}")

            # Cleanup session on connection failure
            if hasattr(self, "session") and self.session:
                await self.session.aclose()
                self.session = None

            raise MCPConnectionError(f"Connection failed: {e}") from e

    async def disconnect(self) -> bool:
        """Disconnect from Zen MCP Server and cleanup resources."""
        try:
            if self.session:
                # Close HTTP session and cleanup resources
                await self.session.aclose()
                self.session = None

            self.connection_state = MCPConnectionState.DISCONNECTED
            logger.info("Disconnected from Zen MCP Server")
            return True

        except Exception as e:
            logger.error(f"Error during Zen MCP disconnection: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((MCPConnectionError, MCPTimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def health_check(self) -> MCPHealthStatus:
        """Perform health check with retry logic."""
        start_time = time.time()

        try:
            if httpx is None:
                raise ImportError("httpx is not installed")

            if self.connection_state == MCPConnectionState.DISCONNECTED or not self.session:
                raise MCPConnectionError("Not connected to server")

            # Perform real health check
            try:
                response = await self.session.get("/health", timeout=5.0)
                response.raise_for_status()

                # Parse health response
                health_data = {}
                if response.headers.get("content-type", "").startswith("application/json"):
                    health_data = response.json()

                response_time_ms = (time.time() - start_time) * 1000
                self.last_successful_request = time.time()

                # Extract server information from response
                server_capabilities = health_data.get(
                    "capabilities",
                    ["zen_orchestration", "multi_agent", "consensus", "validation"],
                )
                server_version = health_data.get("version", "ZenMCP-1.0.0")
                server_status = health_data.get("status", "healthy")

                # Determine connection state based on server status
                if server_status == "degraded":
                    current_state = MCPConnectionState.DEGRADED
                elif server_status == "healthy":
                    current_state = MCPConnectionState.CONNECTED
                else:
                    current_state = MCPConnectionState.DEGRADED

                return MCPHealthStatus(
                    connection_state=current_state,
                    last_successful_request=self.last_successful_request,
                    error_count=self.error_count,
                    response_time_ms=response_time_ms,
                    capabilities=server_capabilities,
                    server_version=server_version,
                    metadata={
                        "server_url": self.server_url,
                        "has_api_key": bool(self.api_key),
                        "server_status": server_status,
                        "http_status": response.status_code,
                    },
                )

            except httpx.HTTPStatusError as e:
                self.error_count += 1
                raise MCPConnectionError(f"Health check HTTP error: {e.response.status_code}") from e
            except httpx.TimeoutException as e:
                self.error_count += 1
                raise MCPTimeoutError(f"Health check timeout: {e}", 5.0) from e
            except httpx.ConnectError as e:
                self.error_count += 1
                raise MCPConnectionError(f"Health check connection error: {e}") from e

        except Exception as e:
            self.error_count += 1
            if isinstance(e, MCPError):
                raise
            raise MCPConnectionError(f"Health check failed: {e}") from e

    async def validate_query(self, query: str) -> dict[str, Any]:
        """Validate query using Zen MCP Server security services."""
        if not query or not query.strip():
            return {"is_valid": False, "sanitized_query": "", "potential_issues": ["Empty query"]}

        start_time = time.time()

        try:
            if httpx is None:
                raise ImportError("httpx is not installed")

            if not self.session:
                raise MCPConnectionError("Not connected to server")

            # Prepare validation payload
            payload = {"query": query, "validation_level": "standard", "sanitize": True}

            # Send validation request to MCP server
            try:
                response = await self.session.post("/validate", json=payload, timeout=self.timeout)
                response.raise_for_status()

                validation_result = response.json()
                processing_time_ms = (time.time() - start_time) * 1000

                # Add processing time to result
                if isinstance(validation_result, dict):
                    # Ensure we return the expected dict structure
                    typed_result: dict[str, Any] = {
                        "is_valid": validation_result.get("is_valid", True),
                        "sanitized_query": validation_result.get("sanitized_query", query),
                        "potential_issues": validation_result.get("potential_issues", []),
                        "processing_time_ms": processing_time_ms,
                    }
                    # Add any additional fields from the response
                    for key, value in validation_result.items():
                        if key not in typed_result:
                            typed_result[key] = value

                    self.last_successful_request = time.time()
                    return typed_result
                # Fallback for non-dict responses
                return {
                    "is_valid": False,
                    "sanitized_query": query,
                    "potential_issues": ["Invalid response format"],
                    "processing_time_ms": processing_time_ms,
                }

            except httpx.HTTPStatusError as e:
                if e.response.status_code == HTTP_BAD_REQUEST:
                    # Bad request - raise validation error
                    error_data = (
                        e.response.json()
                        if e.response.headers.get("content-type", "").startswith("application/json")
                        else {}
                    )
                    raise MCPValidationError(
                        error_data.get("error", "Validation failed"),
                        validation_errors=error_data.get("details", {}),
                        details={
                            "status_code": e.response.status_code,
                            "response": e.response.text[:200],
                            **error_data,
                        },
                    ) from e
                if e.response.status_code == HTTP_SERVICE_UNAVAILABLE:  # Service Unavailable
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    raise MCPServiceUnavailableError("Validation service unavailable", retry_after) from e
                raise MCPServiceUnavailableError(f"Validation service error: HTTP {e.response.status_code}") from e

            except httpx.TimeoutException as e:
                raise MCPTimeoutError(f"Query validation timeout: {e}", self.timeout) from e
            except httpx.ConnectError as e:
                raise MCPConnectionError(f"Validation service connection error: {e}") from e

        except Exception as e:
            if isinstance(e, MCPError):
                raise
            logger.error(f"Query validation failed: {e}")
            raise MCPServiceUnavailableError(f"Validation service unavailable: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((MCPServiceUnavailableError, MCPTimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def orchestrate_agents(self, workflow_steps: list[WorkflowStep]) -> list[Response]:
        """Orchestrate workflow using Zen MCP Server."""
        start_time = time.time()

        try:
            if httpx is None:
                raise ImportError("httpx is not installed")

            if not self.session:
                raise MCPConnectionError("Not connected to server")

            # Prepare orchestration payload
            payload = {
                "workflow_steps": [
                    {
                        "step_id": step.step_id,
                        "agent_id": step.agent_id,
                        "input_data": step.input_data,
                        "dependencies": step.dependencies,
                        "timeout_seconds": step.timeout_seconds,
                    }
                    for step in workflow_steps
                ],
                "execution_mode": "parallel",
                "timeout": self.timeout,
                "max_concurrent": 5,  # Limit concurrent executions
            }

            # Send orchestration request to MCP server
            try:
                response = await self.session.post("/orchestrate", json=payload, timeout=self.timeout)
                response.raise_for_status()

                result = response.json()

                # Convert response to Response objects
                responses = []
                for resp_data in result.get("responses", []):
                    response_obj = Response(
                        agent_id=resp_data["agent_id"],
                        content=resp_data["content"],
                        metadata=resp_data.get("metadata", {}),
                        confidence=resp_data.get("confidence", 0.9),
                        processing_time=resp_data.get("processing_time", 0.0),
                        success=resp_data.get("success", True),
                        error_message=resp_data.get("error_message"),
                    )
                    responses.append(response_obj)

                self.last_successful_request = time.time()
                total_time = time.time() - start_time
                logger.info(f"Zen MCP orchestration completed {len(workflow_steps)} steps in {total_time:.3f}s")

                return responses

            except httpx.HTTPStatusError as e:
                self.error_count += 1
                if e.response.status_code == HTTP_UNAUTHORIZED:  # Unauthorized
                    error_data = {}
                    try:
                        if e.response.headers.get("content-type", "").startswith("application/json"):
                            error_data = e.response.json()
                    except Exception as parse_error:
                        logger.debug("Failed to parse JSON error response: %s", parse_error)
                    raise MCPAuthenticationError(
                        "Authentication failed",
                        error_code="INVALID_API_KEY",
                        details={
                            "status_code": e.response.status_code,
                            "response": e.response.text[:200],
                            **error_data,
                        },
                    ) from e
                if e.response.status_code == HTTP_BAD_REQUEST:
                    error_data = (
                        e.response.json()
                        if e.response.headers.get("content-type", "").startswith("application/json")
                        else {}
                    )
                    raise MCPError(
                        f"Invalid orchestration request: {error_data.get('message', 'Bad request')}",
                        MCPErrorType.INVALID_REQUEST,
                        error_data,
                    ) from e
                if e.response.status_code == HTTP_TOO_MANY_REQUESTS:
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    raise MCPRateLimitError("Rate limit exceeded", retry_after) from e
                if e.response.status_code == HTTP_SERVICE_UNAVAILABLE:  # Service Unavailable
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    raise MCPServiceUnavailableError("Service unavailable", retry_after) from e
                raise MCPServiceUnavailableError(
                    f"Orchestration service error: HTTP {e.response.status_code}",
                ) from e

            except httpx.TimeoutException as e:
                self.error_count += 1
                raise MCPTimeoutError(f"Orchestration timeout: {e}", self.timeout) from e
            except httpx.ConnectError as e:
                self.error_count += 1
                raise MCPConnectionError(f"Orchestration service connection error: {e}") from e

        except Exception as e:
            self.error_count += 1
            if isinstance(e, MCPError):
                raise
            logger.error(f"Zen MCP orchestration failed: {e}")
            raise MCPServiceUnavailableError(f"Orchestration service unavailable: {e}") from e

    async def get_capabilities(self) -> list[str]:
        """Get Zen MCP Server capabilities."""
        try:
            if httpx is None:
                raise ImportError("httpx is not installed")

            if not self.session:
                raise MCPConnectionError("Not connected to server")

            # Query server capabilities
            try:
                response = await self.session.get("/capabilities", timeout=5.0)
                response.raise_for_status()

                capabilities_data = response.json()
                # Ensure we return a list of strings
                capabilities = capabilities_data.get(
                    "capabilities",
                    ["zen_orchestration", "multi_agent", "consensus", "validation"],
                )
                # Type-check the capabilities
                if isinstance(capabilities, list):
                    return [str(cap) for cap in capabilities]
                return ["zen_orchestration", "multi_agent", "consensus", "validation"]

            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to get capabilities: HTTP {e.response.status_code}")
                # Fall back to default capabilities on server error
                return ["zen_orchestration", "multi_agent", "consensus", "validation"]
            except httpx.TimeoutException as e:
                raise MCPTimeoutError(f"Capabilities query timeout: {e}", 5.0) from e
            except httpx.ConnectError as e:
                raise MCPConnectionError(f"Capabilities query connection error: {e}") from e

        except Exception as e:
            if isinstance(e, MCPError):
                raise
            logger.error(f"Failed to get Zen MCP capabilities: {e}")
            raise MCPConnectionError(f"Capabilities query failed: {e}") from e


class MCPClientFactory:
    """
    Factory for creating appropriate MCP client instances.

    Provides centralized creation logic for MCP clients, enabling easy switching
    between mock development clients and production Zen MCP implementations.
    """

    @staticmethod
    def create_client(
        client_type: str = "mock",
        server_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> MCPClientInterface:
        """
        Create MCP client instance based on configuration.

        Args:
            client_type: Type of client ("mock" or "zen")
            server_url: Server URL for real clients
            api_key: API key for authentication
            **kwargs: Additional client-specific parameters

        Returns:
            MCPClientInterface: Configured client instance

        Raises:
            ValueError: If client_type is not supported
        """
        if client_type.lower() == "mock":
            return MockMCPClient(**kwargs)

        if client_type.lower() == "zen":
            if not server_url:
                raise ValueError("server_url required for Zen MCP client")
            return ZenMCPClient(server_url=server_url, api_key=api_key, **kwargs)

        raise ValueError(f"Unsupported client type: {client_type}")

    @staticmethod
    def create_from_settings(settings: Any | None = None) -> MCPClientInterface:
        """
        Create MCP client instance from application settings.

        Args:
            settings: ApplicationSettings instance (optional, will load if not provided)

        Returns:
            MCPClientInterface: Configured client instance based on settings
        """
        if settings is None:
            settings = get_settings()

        # Determine client type based on settings
        if not settings.mcp_enabled:
            logger.info("MCP disabled in configuration, using mock client")
            return MockMCPClient()

        # Extract API key if available
        api_key = None
        if settings.mcp_api_key:
            api_key = settings.mcp_api_key.get_secret_value()

        # Create Zen client with settings
        return ZenMCPClient(
            server_url=settings.mcp_server_url,
            api_key=api_key,
            timeout=settings.mcp_timeout,
            max_retries=settings.mcp_max_retries,
        )


class MCPConnectionManager:
    """
    Connection manager for MCP client lifecycle management.

    Provides centralized connection management, health monitoring, and automatic
    reconnection for MCP clients. Implements circuit breaker pattern to prevent
    cascading failures.
    """

    def __init__(
        self,
        client: MCPClientInterface,
        health_check_interval: float = 30.0,
        max_consecutive_failures: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ) -> None:
        """
        Initialize connection manager.

        Args:
            client: MCP client instance to manage
            health_check_interval: Health check frequency in seconds
            max_consecutive_failures: Failures before circuit breaker opens
            circuit_breaker_timeout: Circuit breaker reset timeout
        """
        self.client = client
        self.health_check_interval = health_check_interval
        self.max_consecutive_failures = max_consecutive_failures
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.consecutive_failures = 0
        self.circuit_breaker_open_time: float | None = None
        self.is_circuit_breaker_open = False
        self._health_check_task: asyncio.Task | None = None

    async def start(self) -> bool:
        """Start connection manager and health monitoring."""
        try:
            # Initial connection
            await self.client.connect()

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            logger.info("MCP connection manager started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start MCP connection manager: {e}")
            return False

    async def stop(self) -> None:
        """Stop connection manager and cleanup resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        await self.client.disconnect()
        logger.info("MCP connection manager stopped")

    async def execute_with_fallback(self, operation: str, *args: Any, **kwargs: Any) -> Any | dict[str, Any]:
        """
        Execute MCP operation with circuit breaker and fallback.

        Args:
            operation: Name of the operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result or fallback response

        Raises:
            MCPError: If operation fails and no fallback available
        """
        # Check circuit breaker
        if self.is_circuit_breaker_open:
            if self._should_attempt_reset():
                logger.info("Attempting to reset circuit breaker")
                self.is_circuit_breaker_open = False
                self.circuit_breaker_open_time = None
            else:
                return self._get_fallback_response(operation)

        try:
            # Execute operation
            method = getattr(self.client, operation)
            result = await method(*args, **kwargs)

            # Reset failure count on success
            self.consecutive_failures = 0
            return result

        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"MCP operation '{operation}' failed (attempt {self.consecutive_failures}): {e}")

            # Open circuit breaker if too many failures
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.is_circuit_breaker_open = True
                self.circuit_breaker_open_time = time.time()
                logger.warning(f"Circuit breaker opened after {self.consecutive_failures} consecutive failures")

            # Return fallback response
            return self._get_fallback_response(operation, error=e)

    async def _health_monitor(self) -> None:
        """Continuous health monitoring task."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                health_status = await self.client.health_check()

                if health_status.connection_state == MCPConnectionState.FAILED:
                    logger.warning("Health check indicates failed state, attempting reconnection")
                    await self.client.disconnect()
                    await self.client.connect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.circuit_breaker_open_time:
            return False
        return time.time() - self.circuit_breaker_open_time > self.circuit_breaker_timeout

    def _get_fallback_response(self, operation: str, error: Exception | None = None) -> dict[str, Any]:
        """Generate fallback response for failed operations."""
        base_response = {
            "fallback": True,
            "operation": operation,
            "error": str(error) if error else "Circuit breaker open",
            "timestamp": time.time(),
        }

        if operation == "orchestrate_agents":
            # Return dict with empty responses for agent orchestration
            return {
                "responses": [],
                "total_agents": 0,
                **base_response,
            }

        if operation == "validate_query":
            # Return permissive validation for query validation
            return {
                "is_valid": True,
                "sanitized_query": "",
                "potential_issues": ["MCP validation service unavailable"],
                **base_response,
            }

        if operation == "get_capabilities":
            # Return dict with minimal capabilities
            return {
                "capabilities": ["basic_processing"],
                **base_response,
            }

        return base_response


# Alias for backward compatibility with tests
MCPClient = ZenMCPClient
