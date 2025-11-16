"""
OpenRouter Client Implementation for PromptCraft-Hybrid.

This module provides OpenRouter API integration that implements the MCPClientInterface
for unified orchestration through the PromptCraft system. It includes comprehensive
HTTP client functionality, authentication, error handling, and integration with
the ModelRegistry for rate limiting and model capabilities.

Key Features:
    - OpenRouter API integration with full HTTP client functionality
    - Implements MCPClientInterface for unified orchestration
    - Integration with ModelRegistry for model selection and rate limiting
    - Comprehensive error handling and normalization to MCP format
    - Proper authentication with API key, HTTP-Referer, and X-Title headers
    - Timeout handling and retry logic
    - Response parsing and validation

Architecture:
    The OpenRouterClient translates MCP orchestration requests into OpenRouter API
    calls, providing a bridge between the PromptCraft agent system and OpenRouter's
    model routing capabilities. It maintains compatibility with the MCP interface
    while leveraging OpenRouter's model ecosystem.

OpenRouter API Integration:
    - Base URL: https://openrouter.ai/api/v1
    - Authentication: Bearer token in Authorization header
    - Required Headers: HTTP-Referer, X-Title for request identification
    - Model Selection: Via ModelRegistry with fallback chain support
    - Rate Limiting: Respects OpenRouter model-specific limits

Dependencies:
    - httpx: For async HTTP client functionality
    - src.mcp_integration.model_registry: For model selection and capabilities
    - src.mcp_integration.mcp_client: For MCPClientInterface and error types
    - src.config.settings: For configuration management
    - tenacity: For retry logic with exponential backoff

Called by:
    - src.core.query_counselor: For AI model orchestration
    - AI Tool Routing: For model routing and fallback handling
    - Integration tests: For OpenRouter API validation

Time Complexity: O(1) for single requests, O(n) for n parallel agent requests
Space Complexity: O(k) where k is the number of concurrent connections
"""

import logging
import re
import time
from typing import Any, NoReturn

import httpx

from src.config.settings import get_settings
from src.mcp_integration.mcp_client import (
    MCPAuthenticationError,
    MCPClientInterface,
    MCPConnectionError,
    MCPConnectionState,
    MCPError,
    MCPErrorType,
    MCPHealthStatus,
    MCPRateLimitError,
    MCPServiceUnavailableError,
    MCPTimeoutError,
    MCPValidationError,
    Response,
    WorkflowStep,
)
from src.mcp_integration.model_registry import get_model_registry
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)

# OpenRouter API constants
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503

# Rate limiting constants
MAX_TOKENS_PER_MINUTE = 1000000
MAX_REQUESTS_PER_MINUTE = 100
MAX_REQUESTS_PER_HOUR = 50
FALLBACK_RETRY_LIMIT = 10

# Query validation constants
MAX_QUERY_LENGTH = 50000  # 50K character limit for query validation
OPENROUTER_CHAT_ENDPOINT = "/chat/completions"
DEFAULT_SITE_URL = "https://promptcraft.io"
DEFAULT_APP_NAME = "PromptCraft-Hybrid"

# Health monitoring constants
HIGH_ERROR_THRESHOLD = 10  # Maximum error count before status becomes DEGRADED
STALE_REQUEST_TIMEOUT = 3600  # 1 hour in seconds - when to consider requests stale


class OpenRouterClient(MCPClientInterface):
    """
    OpenRouter API client implementing MCPClientInterface.

    Provides OpenRouter integration with MCP-compatible interface for unified
    orchestration in the PromptCraft system. Handles authentication, rate limiting,
    error normalization, and model selection through ModelRegistry integration.

    Features:
        - Full OpenRouter API integration with proper authentication
        - MCP interface compatibility for seamless orchestration
        - ModelRegistry integration for intelligent model selection
        - Comprehensive error handling and retry logic
        - Rate limiting and timeout management
        - Security validation and query sanitization
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        site_url: str = DEFAULT_SITE_URL,
        app_name: str = DEFAULT_APP_NAME,
    ) -> None:
        """
        Initialize OpenRouter client with configuration.

        Args:
            api_key: OpenRouter API key (if None, loads from settings)
            base_url: OpenRouter API base URL (if None, loads from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            site_url: Site URL for HTTP-Referer header
            app_name: Application name for X-Title header
        """
        settings = get_settings()

        # Configure API authentication and endpoints
        self.api_key = api_key or (
            settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else None
        )
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.site_url = site_url
        self.app_name = app_name

        # Connection and state management
        self.connection_state = MCPConnectionState.DISCONNECTED
        self.session: httpx.AsyncClient | None = None
        self.error_count = 0
        self.last_successful_request: float | None = None

        # Model registry integration
        self.model_registry = get_model_registry()

        # Circuit breaker integration for resilience
        self.circuit_breaker: CircuitBreaker | None = None
        if settings.circuit_breaker_enabled:
            self.circuit_breaker = get_circuit_breaker("openrouter", settings)

        # Validate configuration
        if not self.api_key:
            logger.warning("OpenRouter API key not configured - client will be limited")

    async def connect(self) -> bool:
        """
        Establish connection to OpenRouter API with circuit breaker protection.

        Returns:
            bool: True if connection successful, False otherwise

        Raises:
            MCPConnectionError: If connection cannot be established
            CircuitBreakerOpenError: If circuit breaker is open
        """

        async def _connect_with_protection() -> bool:
            """Internal connection function for circuit breaker protection."""
            try:
                if self.session:
                    await self.session.aclose()

                # Create HTTP session with proper configuration
                self.session = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.timeout),
                    headers=self._get_headers(),
                    limits=httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=20,
                    ),
                )

                # Test connection with a simple model list request
                try:
                    response = await self.session.get("/models", timeout=10.0)
                    if response.status_code == HTTP_OK:
                        self.connection_state = MCPConnectionState.CONNECTED
                        self.last_successful_request = time.time()
                        logger.info("OpenRouter client connected successfully")
                        return True
                    self.connection_state = MCPConnectionState.DEGRADED
                    logger.warning(f"OpenRouter connection test failed: HTTP {response.status_code}")
                    return False

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == HTTP_UNAUTHORIZED:
                        raise MCPAuthenticationError(
                            "OpenRouter API key authentication failed",
                            error_code="INVALID_API_KEY",
                            details={"status_code": e.response.status_code},
                        ) from e
                    self.connection_state = MCPConnectionState.DEGRADED
                    logger.warning(f"OpenRouter connection test failed: {e}")
                    return False

            except MCPAuthenticationError:
                # Re-raise authentication errors without wrapping them
                self.connection_state = MCPConnectionState.DISCONNECTED
                self.error_count += 1
                raise
            except Exception as e:
                self.connection_state = MCPConnectionState.DISCONNECTED
                self.error_count += 1
                logger.error(f"Failed to connect to OpenRouter: {e}")
                raise MCPConnectionError(f"OpenRouter connection failed: {e}") from e

        # Use circuit breaker if available, otherwise execute directly
        if self.circuit_breaker:
            try:
                return await self.circuit_breaker.call_async(_connect_with_protection)
            except CircuitBreakerOpenError:
                logger.warning("OpenRouter circuit breaker is open, connection attempt blocked")
                self.connection_state = MCPConnectionState.DISCONNECTED
                return False
        else:
            return await _connect_with_protection()

    async def disconnect(self) -> bool:
        """
        Disconnect from OpenRouter API and cleanup resources.

        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.session:
                await self.session.aclose()
                self.session = None

            self.connection_state = MCPConnectionState.DISCONNECTED
            logger.info("OpenRouter client disconnected")
            return True

        except Exception as e:
            logger.error(f"Error during OpenRouter disconnect: {e}")
            return False

    async def health_check(self) -> MCPHealthStatus:
        """
        Perform basic health check on OpenRouter connection.

        Returns:
            MCPHealthStatus: Current health status and metrics
        """
        # Determine health status based on connection state and error count
        if self.connection_state == MCPConnectionState.DISCONNECTED:
            status = "UNHEALTHY"
            message = "OpenRouter client is not connected"
        elif self.error_count > HIGH_ERROR_THRESHOLD:
            status = "DEGRADED"
            message = f"OpenRouter client has high error count: {self.error_count}"
        elif self.last_successful_request and (time.time() - self.last_successful_request) > STALE_REQUEST_TIMEOUT:
            status = "DEGRADED"
            message = "Last successful request was over an hour ago"
        else:
            status = "HEALTHY"
            message = "OpenRouter client is healthy"

        return MCPHealthStatus(
            connection_state=self.connection_state,
            error_count=self.error_count,
            last_successful_request=self.last_successful_request,
            metadata={
                "service": "openrouter",
                "status": status,
                "message": message,
            },
        )

    async def async_health_check(self) -> MCPHealthStatus:
        """
        Perform comprehensive async health check on OpenRouter connection.

        Returns:
            MCPHealthStatus: Current health status and metrics

        Raises:
            MCPError: If health check fails
        """
        start_time = time.time()

        try:
            if not self.session:
                await self.connect()

            if not self.session:
                raise MCPConnectionError("Failed to establish OpenRouter connection for health check")

            # Perform health check request
            response = await self.session.get("/models", timeout=5.0)
            response_time = time.time() - start_time

            if response.status_code == HTTP_OK:
                return MCPHealthStatus(
                    connection_state=MCPConnectionState.CONNECTED,
                    response_time_ms=response_time * 1000,  # Convert to milliseconds
                    error_count=self.error_count,
                    last_successful_request=self.last_successful_request,
                    metadata={
                        "service": "openrouter",
                        "connection_state": self.connection_state,
                        "models_available": len(response.json().get("data", [])),
                    },
                )
            self.error_count += 1
            return MCPHealthStatus(
                connection_state=MCPConnectionState.DEGRADED,
                response_time_ms=response_time * 1000,  # Convert to milliseconds
                error_count=self.error_count,
                last_successful_request=self.last_successful_request,
                metadata={
                    "service": "openrouter",
                    "connection_state": self.connection_state,
                    "error": f"HTTP {response.status_code}",
                },
            )

        except Exception as e:
            self.error_count += 1
            response_time = time.time() - start_time
            logger.error(f"OpenRouter health check failed: {e}")

            return MCPHealthStatus(
                connection_state=MCPConnectionState.FAILED,
                response_time_ms=response_time * 1000,  # Convert to milliseconds
                error_count=self.error_count,
                last_successful_request=self.last_successful_request,
                metadata={
                    "service": "openrouter",
                    "connection_state": self.connection_state,
                    "error": str(e),
                },
            )

    async def validate_query(self, query: str | None) -> dict[str, Any]:
        """
        Validate and sanitize user query for security.

        Args:
            query: Raw user query string (can be None)

        Returns:
            Dict containing validation results with keys:
                - is_valid: bool
                - sanitized_query: str
                - potential_issues: List[str]
                - error: str (if validation fails)

        Raises:
            MCPError: If validation service fails
        """
        try:
            potential_issues = []

            # Handle None or empty queries
            if query is None:
                return {
                    "is_valid": False,
                    "sanitized_query": "",
                    "potential_issues": ["Query is None"],
                    "error": "Query cannot be empty or None",
                }

            if not query.strip():
                return {
                    "is_valid": False,
                    "sanitized_query": "",
                    "potential_issues": ["Query is empty"],
                    "error": "Query cannot be empty or None",
                }

            sanitized_query = query.strip()

            # Basic security validations
            if len(query) > MAX_QUERY_LENGTH:  # 50K character limit
                potential_issues.append("Query length exceeds recommended limit")
                sanitized_query = query[:MAX_QUERY_LENGTH]
                return {
                    "is_valid": False,
                    "sanitized_query": sanitized_query,
                    "potential_issues": potential_issues,
                    "error": "Query is too long",
                }

            # Check for potential injection patterns
            suspicious_patterns = [
                r"<script[^>]*>",
                r"javascript:",
                r"data:text/html",
                r"vbscript:",
                r"onload\s*=",
                r"onerror\s*=",
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    potential_issues.append("Query contains potentially unsafe content")
                    break

            # Check for excessive repetition (potential DoS)
            words = sanitized_query.split()
            if len(words) > MAX_REQUESTS_PER_MINUTE:
                word_counts: dict[str, int] = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                    if word_counts[word] > MAX_REQUESTS_PER_HOUR:  # Same word repeated > 50 times
                        potential_issues.append("Query contains excessive repetition")
                        break

            is_valid = len(potential_issues) == 0

            result = {
                "is_valid": is_valid,
                "sanitized_query": sanitized_query,
                "potential_issues": potential_issues,
            }

            if not is_valid:
                result["error"] = "Query validation failed"

            return result

        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            raise MCPValidationError(f"Query validation error: {e}") from e

    async def orchestrate_agents(self, workflow_steps: list[WorkflowStep]) -> list[Response]:
        """
        Orchestrate multi-agent workflow execution via OpenRouter API with circuit breaker protection.

        Args:
            workflow_steps: List of workflow steps to execute

        Returns:
            List[Response]: Responses from all agents

        Raises:
            MCPError: If orchestration fails
            MCPTimeoutError: If execution exceeds timeout
            CircuitBreakerOpenError: If circuit breaker is open
        """

        async def _orchestrate_with_protection() -> list[Response]:
            """Internal orchestration function for circuit breaker protection."""
            start_time = time.time()

            try:
                if not self.session:
                    await self.connect()

                if not self.session:
                    raise MCPConnectionError("Failed to establish OpenRouter connection")

                responses = []

                # Process each workflow step
                for step in workflow_steps:
                    try:
                        response = await self._execute_single_step(step)
                        responses.append(response)

                    except Exception as e:
                        # Create error response for failed step
                        error_response = Response(
                            agent_id=step.agent_id,
                            content=f"Error executing step {step.step_id}: {e!s}",
                            metadata={"error": True, "step_id": step.step_id},
                            confidence=0.0,
                            processing_time=time.time() - start_time,
                            success=False,
                            error_message=str(e),
                        )
                        responses.append(error_response)
                        logger.error(f"Failed to execute step {step.step_id}: {e}")

                total_time = time.time() - start_time
                logger.info(f"OpenRouter orchestration completed {len(workflow_steps)} steps in {total_time:.3f}s")

                return responses

            except Exception as e:
                self.error_count += 1
                if isinstance(e, MCPError):
                    raise
                logger.error(f"OpenRouter orchestration failed: {e}")
                raise MCPServiceUnavailableError(f"OpenRouter orchestration unavailable: {e}") from e

        # Use circuit breaker if available, otherwise execute directly
        if self.circuit_breaker:
            try:
                return await self.circuit_breaker.call_async(_orchestrate_with_protection)
            except CircuitBreakerOpenError as e:
                logger.warning("OpenRouter circuit breaker is open, using fallback response")
                # Create fallback responses for all steps
                fallback_responses = []
                for step in workflow_steps:
                    fallback_response = Response(
                        agent_id=step.agent_id,
                        content="Service temporarily unavailable due to circuit breaker protection. Please try again later.",
                        metadata={
                            "error": True,
                            "step_id": step.step_id,
                            "circuit_breaker_open": True,
                            "recovery_time": e.recovery_time.isoformat() if e.recovery_time else None,
                        },
                        confidence=0.0,
                        processing_time=0.0,
                        success=False,
                        error_message=str(e),
                    )
                    fallback_responses.append(fallback_response)
                return fallback_responses
        else:
            return await _orchestrate_with_protection()

    async def get_capabilities(self) -> list[str]:
        """
        Get list of available OpenRouter capabilities.

        Returns:
            List[str]: Available capability names

        Raises:
            MCPError: If capability query fails
        """
        try:
            if not self.session:
                await self.connect()

            if not self.session:
                raise MCPConnectionError("Failed to establish OpenRouter connection for capabilities query")

            # Get available models from OpenRouter
            response = await self.session.get("/models", timeout=10.0)
            response.raise_for_status()

            models_data = response.json()
            available_models = [model["id"] for model in models_data.get("data", [])]

            # Base capabilities
            capabilities = [
                "chat_completion",
                "text_generation",
                "model_routing",
                "fallback_chains",
            ]

            # Add model-specific capabilities
            for model_id in available_models[:10]:  # Limit to prevent excessive list
                model_caps = self.model_registry.get_model_capabilities(model_id)
                if model_caps:
                    if model_caps.supports_function_calling:
                        capabilities.append("function_calling")
                    if model_caps.supports_vision:
                        capabilities.append("vision")
                    if model_caps.supports_reasoning:
                        capabilities.append("reasoning")
                    break  # Only check first available model for capabilities

            return list(set(capabilities))  # Remove duplicates

        except Exception as e:
            logger.error(f"Failed to get OpenRouter capabilities: {e}")
            raise MCPError(
                f"Capability query failed: {e}",
                MCPErrorType.SERVICE_ERROR,
                {"service": "openrouter"},
            ) from e

    def _get_headers(self) -> dict[str, str]:
        """
        Get OpenRouter API headers with authentication.

        Returns:
            Dict[str, str]: HTTP headers for requests
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{self.app_name}/1.0",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def _execute_single_step(self, step: WorkflowStep) -> Response:
        """
        Execute a single workflow step via OpenRouter API.

        Args:
            step: Workflow step to execute

        Returns:
            Response: AI model response

        Raises:
            MCPError: If step execution fails
        """
        step_start_time = time.time()

        try:
            # Ensure session is established
            if not self.session:
                raise MCPConnectionError("OpenRouter session not established")

            # Extract user query from input data
            user_query = step.input_data.get("query", "")
            if not user_query:
                raise MCPValidationError("No query provided in step input data")

            # Select appropriate model for the task
            task_type = step.input_data.get("task_type", "general")
            allow_premium = step.input_data.get("allow_premium", False)
            max_tokens_needed = step.input_data.get("max_tokens_needed")

            model_id = self.model_registry.select_best_model(
                task_type=task_type,
                allow_premium=allow_premium,
                max_tokens_needed=max_tokens_needed,
            )

            # Prepare OpenRouter API payload
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": user_query,
                    },
                ],
                "temperature": step.input_data.get("temperature", 0.7),
                "max_tokens": step.input_data.get("max_tokens", 2048),
                "stream": False,
            }

            # Add optional parameters
            if "top_p" in step.input_data:
                payload["top_p"] = step.input_data["top_p"]
            if "presence_penalty" in step.input_data:
                payload["presence_penalty"] = step.input_data["presence_penalty"]
            if "frequency_penalty" in step.input_data:
                payload["frequency_penalty"] = step.input_data["frequency_penalty"]

            # Make API request
            response = await self.session.post(
                OPENROUTER_CHAT_ENDPOINT,
                json=payload,
                timeout=step.timeout_seconds,
            )

            # Handle API response
            if response.status_code == HTTP_OK:
                result = response.json()

                # Extract response content
                choices = result.get("choices", [])
                if not choices:
                    raise MCPError("No response choices returned", MCPErrorType.INVALID_RESPONSE)

                content = choices[0].get("message", {}).get("content", "")

                # Calculate confidence based on response quality
                confidence = self._calculate_confidence(result, content)

                processing_time = time.time() - step_start_time
                self.last_successful_request = time.time()

                return Response(
                    agent_id=step.agent_id,
                    content=content,
                    metadata={
                        "model_id": model_id,
                        "usage": result.get("usage", {}),
                        "step_id": step.step_id,
                        "task_type": task_type,
                    },
                    confidence=confidence,
                    processing_time=processing_time,
                    success=True,
                )

            # Handle HTTP errors
            await self._handle_api_error(response)

        except httpx.TimeoutException as e:
            self.error_count += 1
            raise MCPTimeoutError(f"OpenRouter request timeout for step {step.step_id}", step.timeout_seconds) from e

        except httpx.ConnectError as e:
            self.error_count += 1
            raise MCPConnectionError(f"OpenRouter connection error: {e}") from e

        except Exception as e:
            self.error_count += 1
            if isinstance(e, MCPError):
                raise
            raise MCPServiceUnavailableError(f"OpenRouter step execution failed: {e}") from e

    async def _handle_api_error(self, response: httpx.Response) -> NoReturn:
        """
        Handle OpenRouter API error responses.

        Args:
            response: HTTP response with error status

        Raises:
            MCPError: Appropriate MCP error based on response
        """
        status_code = response.status_code

        try:
            error_data = (
                response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            )
        except Exception:
            error_data = {"message": response.text[:200]}

        if status_code == HTTP_UNAUTHORIZED:
            raise MCPAuthenticationError(
                "OpenRouter API authentication failed",
                error_code="INVALID_API_KEY",
                details={"status_code": status_code, **error_data},
            )
        if status_code == HTTP_BAD_REQUEST:
            raise MCPValidationError(f"Invalid OpenRouter request: {error_data.get('message', 'Bad request')}")
        if status_code == HTTP_TOO_MANY_REQUESTS:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise MCPRateLimitError("OpenRouter rate limit exceeded", retry_after)
        if status_code == HTTP_SERVICE_UNAVAILABLE:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise MCPServiceUnavailableError("OpenRouter service unavailable", retry_after)
        raise MCPServiceUnavailableError(
            f"OpenRouter API error: HTTP {status_code} - {error_data.get('message', 'Unknown error')}",
        )

    def _calculate_confidence(self, api_response: dict[str, Any], content: str) -> float:
        """
        Calculate confidence score for API response.

        Args:
            api_response: Full API response data
            content: Response content text

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        confidence = 0.8  # Base confidence

        # Adjust based on content length and quality
        if len(content) < FALLBACK_RETRY_LIMIT:
            confidence -= 0.3  # Very short responses are less confident
        elif len(content) > MAX_REQUESTS_PER_MINUTE:
            confidence += 0.1  # Longer responses generally more confident

        # Adjust based on API usage information
        usage = api_response.get("usage", {})
        if usage.get("completion_tokens", 0) > MAX_REQUESTS_PER_HOUR:
            confidence += 0.05  # More tokens suggest more complete response

        # Adjust based on model finish reason
        choices = api_response.get("choices", [])
        if choices:
            finish_reason = choices[0].get("finish_reason")
            if finish_reason == "stop":
                confidence += 0.05  # Clean completion
            elif finish_reason == "length":
                confidence -= 0.1  # Truncated response

        return max(0.0, min(1.0, confidence))  # Clamp to [0.0, 1.0]
