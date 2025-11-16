"""
Hybrid Router for OpenRouter and MCP Integration.

Provides intelligent routing between OpenRouter and MCP services with gradual rollout,
circuit breaker protection, and comprehensive monitoring. Implements the MCPClientInterface
for seamless integration into the PromptCraft system.

Key Features:
    - Deterministic hash-based routing for gradual rollout
    - OpenRouter primary, MCP fallback strategy
    - Circuit breaker integration for failure detection
    - Performance monitoring and metrics collection
    - Load balancing across available models
    - Configurable traffic percentage for safe deployment

Architecture:
    The HybridRouter acts as a intelligent proxy between the application and
    multiple AI service providers, making routing decisions based on:
    - Request characteristics and requirements
    - Service health and availability
    - Configured rollout percentages
    - Circuit breaker state
    - Performance metrics and SLA requirements

Time Complexity: O(1) for routing decisions, O(n) for workflow orchestration
Space Complexity: O(k) where k is the number of active connections
"""

import logging
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config.settings import get_settings
from src.mcp_integration.mcp_client import (
    MCPClientInterface,
    MCPConnectionError,
    MCPConnectionState,
    MCPError,
    MCPErrorType,
    MCPHealthStatus,
    MCPServiceUnavailableError,
    Response,
    WorkflowStep,
    ZenMCPClient,
)
from src.mcp_integration.openrouter_client import OpenRouterClient
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    get_circuit_breaker,
)
from src.utils.logging_mixin import LoggerMixin

# Routing constants
MAX_TRAFFIC_PERCENTAGE = 100  # Maximum traffic percentage for rollout
FALLBACK_THRESHOLD = 0.6  # Threshold for fallback routing
CIRCUIT_BREAKER_MAX_PERCENTAGE = 100  # Maximum percentage for circuit breaker

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy options for hybrid routing."""

    OPENROUTER_PRIMARY = "openrouter_primary"  # OpenRouter first, MCP fallback
    MCP_PRIMARY = "mcp_primary"  # MCP first, OpenRouter fallback
    ROUND_ROBIN = "round_robin"  # Alternate between services
    LOAD_BALANCED = "load_balanced"  # Route based on current load
    CAPABILITY_BASED = "capability_based"  # Route based on required capabilities


@dataclass
class RoutingDecision:
    """Represents a routing decision made by the HybridRouter."""

    service: str  # "openrouter" or "mcp"
    reason: str  # Human-readable explanation for the routing decision
    confidence: float  # Confidence in the decision (0.0 to 1.0)
    fallback_available: bool  # Whether fallback is available if this choice fails
    request_id: str  # Unique identifier for the request
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert routing decision to dictionary for logging/metrics."""
        return {
            "service": self.service,
            "reason": self.reason,
            "confidence": self.confidence,
            "fallback_available": self.fallback_available,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }


@dataclass
class RoutingMetrics:
    """Tracks routing performance and decision metrics."""

    total_requests: int = 0
    openrouter_requests: int = 0
    mcp_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    fallback_uses: int = 0
    average_response_time: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_routes / self.total_requests) * 100.0

    @property
    def openrouter_percentage(self) -> float:
        """Calculate percentage of requests routed to OpenRouter."""
        if self.total_requests == 0:
            return 0.0
        return (self.openrouter_requests / self.total_requests) * 100.0

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback usage rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.fallback_uses / self.total_requests) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for monitoring/logging."""
        return {
            "total_requests": self.total_requests,
            "openrouter_requests": self.openrouter_requests,
            "mcp_requests": self.mcp_requests,
            "successful_routes": self.successful_routes,
            "failed_routes": self.failed_routes,
            "fallback_uses": self.fallback_uses,
            "success_rate": self.success_rate,
            "openrouter_percentage": self.openrouter_percentage,
            "fallback_rate": self.fallback_rate,
            "average_response_time": self.average_response_time,
            "last_updated": self.last_updated,
        }


class HybridRouter(MCPClientInterface, LoggerMixin):
    """
    Hybrid Router for intelligent routing between OpenRouter and MCP services.

    Implements MCPClientInterface for seamless integration into PromptCraft's
    agent orchestration system. Provides intelligent routing decisions based on
    gradual rollout configuration, service health, and performance metrics.

    Features:
        - Deterministic hash-based routing for consistent request handling
        - Configurable gradual rollout with OPENROUTER_TRAFFIC_PERCENTAGE
        - Circuit breaker integration for OpenRouter failure detection
        - Comprehensive metrics collection and monitoring
        - Seamless fallback between OpenRouter and MCP services
        - Load balancing and capability-based routing
    """

    def __init__(
        self,
        openrouter_client: OpenRouterClient | None = None,
        mcp_client: MCPClientInterface | None = None,
        strategy: RoutingStrategy = RoutingStrategy.OPENROUTER_PRIMARY,
        enable_gradual_rollout: bool = True,
    ) -> None:
        """
        Initialize HybridRouter with service clients and routing configuration.

        Args:
            openrouter_client: OpenRouter client (if None, creates default)
            mcp_client: MCP client (if None, creates ZenMCPClient)
            strategy: Routing strategy to use
            enable_gradual_rollout: Whether to use gradual rollout configuration
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing HybridRouter with hybrid routing capabilities")

        settings = get_settings()

        # Initialize service clients
        self.openrouter_client = openrouter_client or OpenRouterClient()
        self.mcp_client = mcp_client or ZenMCPClient(server_url=settings.mcp_server_url or "http://localhost:3000")

        # Routing configuration
        self.strategy = strategy
        self.enable_gradual_rollout = enable_gradual_rollout

        # Gradual rollout configuration (0-100, default 0)
        self.openrouter_traffic_percentage = int(settings.openrouter_traffic_percentage or 0)
        if not (0 <= self.openrouter_traffic_percentage <= MAX_TRAFFIC_PERCENTAGE):
            self.logger.warning(
                f"Invalid OPENROUTER_TRAFFIC_PERCENTAGE: {self.openrouter_traffic_percentage}, using default 0",
            )
            self.openrouter_traffic_percentage = 0

        # Circuit breaker for OpenRouter
        self.circuit_breaker: CircuitBreaker | None = None
        if settings.circuit_breaker_enabled:
            self.circuit_breaker = get_circuit_breaker("openrouter", settings)

        # Metrics and monitoring
        self.metrics = RoutingMetrics()
        self.connection_state = MCPConnectionState.DISCONNECTED
        self.error_count = 0
        self.last_successful_request: float | None = None

        # Request tracking for round-robin and unique request IDs
        self._request_counter = 0

        self.logger.info(
            f"HybridRouter initialized: strategy={self.strategy.value}, "
            f"openrouter_traffic={self.openrouter_traffic_percentage}%, "
            f"circuit_breaker_enabled={self.circuit_breaker is not None}",
        )

    async def connect(self) -> bool:
        """
        Establish connections to both OpenRouter and MCP services.

        Returns:
            bool: True if at least one service connects successfully

        Raises:
            MCPConnectionError: If both services fail to connect
        """
        self.logger.info("Connecting HybridRouter to OpenRouter and MCP services")

        openrouter_connected = False
        mcp_connected = False

        # Attempt to connect to OpenRouter
        try:
            openrouter_connected = await self.openrouter_client.connect()
            if openrouter_connected:
                self.logger.info("OpenRouter client connected successfully")
            else:
                self.logger.warning("OpenRouter client connection failed")
        except Exception as e:
            self.logger.warning(f"OpenRouter connection error: {e}")

        # Attempt to connect to MCP
        try:
            mcp_connected = await self.mcp_client.connect()
            if mcp_connected:
                self.logger.info("MCP client connected successfully")
            else:
                self.logger.warning("MCP client connection failed")
        except Exception as e:
            self.logger.warning(f"MCP connection error: {e}")

        # Determine overall connection state
        if openrouter_connected and mcp_connected:
            self.connection_state = MCPConnectionState.CONNECTED
            self.logger.info("HybridRouter: Both services connected")
            return True
        if openrouter_connected or mcp_connected:
            self.connection_state = MCPConnectionState.DEGRADED
            service = "OpenRouter" if openrouter_connected else "MCP"
            self.logger.warning(f"HybridRouter: Only {service} connected (degraded mode)")
            return True
        self.connection_state = MCPConnectionState.DISCONNECTED
        self.logger.error("HybridRouter: Both services failed to connect")
        raise MCPConnectionError("Failed to connect to any service (OpenRouter or MCP)")

    async def disconnect(self) -> bool:
        """
        Disconnect from both OpenRouter and MCP services.

        Returns:
            bool: True if all disconnections successful
        """
        self.logger.info("Disconnecting HybridRouter from all services")

        openrouter_disconnected = True
        mcp_disconnected = True

        # Disconnect from OpenRouter
        try:
            openrouter_disconnected = await self.openrouter_client.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting from OpenRouter: {e}")
            openrouter_disconnected = False

        # Disconnect from MCP
        try:
            mcp_disconnected = await self.mcp_client.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting from MCP: {e}")
            mcp_disconnected = False

        self.connection_state = MCPConnectionState.DISCONNECTED

        success = openrouter_disconnected and mcp_disconnected
        if success:
            self.logger.info("HybridRouter disconnected successfully")
        else:
            self.logger.warning("HybridRouter disconnection completed with errors")

        return success

    async def health_check(self) -> MCPHealthStatus:
        """
        Perform comprehensive health check on both services.

        Returns:
            MCPHealthStatus: Aggregated health status from both services

        Raises:
            MCPError: If health check fails
        """
        start_time = time.time()

        try:
            # Perform health checks on both services
            openrouter_health = None
            mcp_health = None

            try:
                openrouter_health = await self.openrouter_client.health_check()
            except Exception as e:
                self.logger.warning(f"OpenRouter health check failed: {e}")

            try:
                mcp_health = await self.mcp_client.health_check()
            except Exception as e:
                self.logger.warning(f"MCP health check failed: {e}")

            # Determine overall health status
            response_time = time.time() - start_time

            # Check if services are healthy based on connection state
            openrouter_healthy = (
                openrouter_health and openrouter_health.connection_state == MCPConnectionState.CONNECTED
            )
            mcp_healthy = mcp_health and mcp_health.connection_state == MCPConnectionState.CONNECTED

            # Check if services are degraded
            openrouter_degraded = (
                openrouter_health and openrouter_health.connection_state == MCPConnectionState.DEGRADED
            )
            mcp_degraded = mcp_health and mcp_health.connection_state == MCPConnectionState.DEGRADED

            if openrouter_healthy and mcp_healthy:
                # Both services healthy
                status = "healthy"
            elif (openrouter_healthy or mcp_healthy) or (openrouter_degraded or mcp_degraded):
                # One service healthy or degraded
                status = "degraded"
            else:
                # Both services unhealthy
                status = "unhealthy"
                self.error_count += 1

            # Aggregate metadata
            metadata = {
                "service": "hybrid_router",
                "strategy": self.strategy.value,
                "openrouter_traffic_percentage": self.openrouter_traffic_percentage,
                "routing_metrics": self.metrics.to_dict(),
                "openrouter_health": {
                    "status": "healthy" if openrouter_healthy else "degraded" if openrouter_degraded else "unhealthy",
                    **(openrouter_health.metadata if openrouter_health else {}),
                },
                "mcp_health": {
                    "status": "healthy" if mcp_healthy else "degraded" if mcp_degraded else "unhealthy",
                    **(mcp_health.metadata if mcp_health else {}),
                },
                "circuit_breaker_enabled": self.circuit_breaker is not None,
            }

            return MCPHealthStatus(
                connection_state=(
                    MCPConnectionState.CONNECTED
                    if status == "healthy"
                    else MCPConnectionState.DEGRADED if status == "degraded" else MCPConnectionState.FAILED
                ),
                response_time_ms=response_time * 1000,  # Convert to milliseconds
                error_count=self.error_count,
                last_successful_request=self.last_successful_request,
                metadata=metadata,
            )

        except Exception as e:
            self.error_count += 1
            response_time = time.time() - start_time
            self.logger.error(f"HybridRouter health check failed: {e}")

            return MCPHealthStatus(
                connection_state=MCPConnectionState.FAILED,
                response_time_ms=response_time * 1000,  # Convert to milliseconds
                error_count=self.error_count,
                last_successful_request=self.last_successful_request,
                metadata={
                    "service": "hybrid_router",
                    "error": str(e),
                },
            )

    async def validate_query(self, query: str) -> dict[str, Any]:
        """
        Validate query using the best available service.

        Args:
            query: Raw user query string

        Returns:
            Dict containing validation results

        Raises:
            MCPError: If validation fails on all services
        """
        # Generate unique request ID using both timestamp and counter for rapid requests
        self._request_counter += 1
        request_id = f"validate_{int(time.time() * 1000)}_{self._request_counter}"

        # Make routing decision for validation
        routing_decision = self._make_routing_decision(request_id, "validation")

        try:
            if routing_decision.service == "openrouter":
                return await self.openrouter_client.validate_query(query)
            return await self.mcp_client.validate_query(query)

        except Exception as e:
            # Try fallback if available
            if routing_decision.fallback_available:
                self.logger.warning(f"Query validation failed on {routing_decision.service}, trying fallback: {e}")
                self.metrics.fallback_uses += 1

                try:
                    if routing_decision.service == "openrouter":
                        return await self.mcp_client.validate_query(query)
                    return await self.openrouter_client.validate_query(query)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback validation also failed: {fallback_error}")

            # All validation attempts failed
            self.error_count += 1
            self.metrics.failed_routes += 1
            raise MCPError(
                f"Query validation failed on all services: {e}",
                MCPErrorType.VALIDATION_ERROR,
                {"routing_decision": routing_decision.to_dict()},
            ) from e

    async def orchestrate_agents(self, workflow_steps: list[WorkflowStep]) -> list[Response]:
        """
        Orchestrate multi-agent workflow using intelligent routing.

        Args:
            workflow_steps: List of workflow steps to execute

        Returns:
            List[Response]: Responses from orchestrated agents

        Raises:
            MCPError: If orchestration fails on all services
        """
        start_time = time.time()
        # Generate unique request ID using both timestamp and counter for rapid requests
        self._request_counter += 1
        request_id = f"orchestrate_{int(time.time() * 1000)}_{self._request_counter}"

        # Update metrics
        self.metrics.total_requests += 1

        # Make routing decision
        routing_decision = self._make_routing_decision(request_id, "orchestration", workflow_steps)

        self.logger.info(f"Routing orchestration to {routing_decision.service}: {routing_decision.reason}")

        try:
            # Execute on primary service
            if routing_decision.service == "openrouter":
                self.metrics.openrouter_requests += 1
                responses = await self.openrouter_client.orchestrate_agents(workflow_steps)
            else:
                self.metrics.mcp_requests += 1
                responses = await self.mcp_client.orchestrate_agents(workflow_steps)

            # Update success metrics
            self.metrics.successful_routes += 1
            self.last_successful_request = time.time()

            # Update average response time
            response_time = time.time() - start_time
            self._update_average_response_time(response_time)

            return responses

        except Exception as e:
            self.logger.warning(f"Orchestration failed on {routing_decision.service}: {e}")

            # Try fallback if available
            if routing_decision.fallback_available:
                self.logger.info("Attempting fallback orchestration")
                self.metrics.fallback_uses += 1

                try:
                    if routing_decision.service == "openrouter":
                        self.metrics.mcp_requests += 1
                        responses = await self.mcp_client.orchestrate_agents(workflow_steps)
                    else:
                        self.metrics.openrouter_requests += 1
                        responses = await self.openrouter_client.orchestrate_agents(workflow_steps)

                    # Update success metrics for fallback
                    self.metrics.successful_routes += 1
                    self.last_successful_request = time.time()

                    response_time = time.time() - start_time
                    self._update_average_response_time(response_time)

                    self.logger.info("Fallback orchestration succeeded")
                    return responses

                except Exception as fallback_error:
                    self.logger.error(f"Fallback orchestration also failed: {fallback_error}")

            # All orchestration attempts failed
            self.error_count += 1
            self.metrics.failed_routes += 1

            if isinstance(e, MCPError):
                raise

            raise MCPServiceUnavailableError(
                f"Orchestration failed on all services: {e}",
                details={"routing_decision": routing_decision.to_dict()},
            ) from e

    async def get_capabilities(self) -> list[str]:
        """
        Get aggregated capabilities from both services.

        Returns:
            List[str]: Combined capability names from both services

        Raises:
            MCPError: If capability query fails on all services
        """
        capabilities = set()

        # Try to get capabilities from both services
        try:
            openrouter_caps = await self.openrouter_client.get_capabilities()
            capabilities.update(openrouter_caps)
        except Exception as e:
            self.logger.warning(f"Failed to get OpenRouter capabilities: {e}")

        try:
            mcp_caps = await self.mcp_client.get_capabilities()
            capabilities.update(mcp_caps)
        except Exception as e:
            self.logger.warning(f"Failed to get MCP capabilities: {e}")

        if not capabilities:
            raise MCPError(
                "Failed to get capabilities from any service",
                MCPErrorType.SERVICE_ERROR,
                {"services_tried": ["openrouter", "mcp"]},
            )

        # Add hybrid-specific capabilities
        hybrid_capabilities = [
            "hybrid_routing",
            "gradual_rollout",
            "fallback_orchestration",
            "circuit_breaker_protection",
            "load_balancing",
        ]
        capabilities.update(hybrid_capabilities)

        return sorted(capabilities)

    def _make_routing_decision(  # noqa: PLR0911
        self,
        request_id: str,
        operation: str,
        workflow_steps: list[WorkflowStep] | None = None,
    ) -> RoutingDecision:
        """
        Make intelligent routing decision based on strategy and conditions.

        Args:
            request_id: Unique identifier for the request
            operation: Type of operation (validation, orchestration, etc.)
            workflow_steps: Optional workflow steps for analysis

        Returns:
            RoutingDecision: Decision with service choice and reasoning
        """
        # Check if gradual rollout applies
        if self.enable_gradual_rollout and self.openrouter_traffic_percentage > 0:
            # Use deterministic hash-based routing for consistency
            hash_value = zlib.crc32(request_id.encode()) % 100

            if hash_value < self.openrouter_traffic_percentage:
                # Route to OpenRouter based on percentage
                if self._is_openrouter_available():
                    return RoutingDecision(
                        service="openrouter",
                        reason=f"Gradual rollout: {hash_value} < {self.openrouter_traffic_percentage}%",
                        confidence=0.9,
                        fallback_available=True,
                        request_id=request_id,
                    )
                # OpenRouter unavailable, fallback to MCP for this gradual rollout request
                return RoutingDecision(
                    service="mcp",
                    reason=f"Gradual rollout: {hash_value} < {self.openrouter_traffic_percentage}% but OpenRouter unavailable, using MCP fallback",
                    confidence=0.7,
                    fallback_available=False,
                    request_id=request_id,
                )
            # Route to MCP for remaining traffic
            return RoutingDecision(
                service="mcp",
                reason=f"Gradual rollout: {hash_value} >= {self.openrouter_traffic_percentage}%",
                confidence=0.9,
                fallback_available=self._is_openrouter_available(),
                request_id=request_id,
            )

        # Apply routing strategy
        if self.strategy == RoutingStrategy.OPENROUTER_PRIMARY:
            if self._is_openrouter_available():
                return RoutingDecision(
                    service="openrouter",
                    reason="OpenRouter primary strategy and service available",
                    confidence=0.9,
                    fallback_available=True,
                    request_id=request_id,
                )
            return RoutingDecision(
                service="mcp",
                reason="OpenRouter primary but unavailable, using MCP fallback",
                confidence=0.7,
                fallback_available=False,
                request_id=request_id,
            )

        if self.strategy == RoutingStrategy.MCP_PRIMARY:
            return RoutingDecision(
                service="mcp",
                reason="MCP primary strategy",
                confidence=0.9,
                fallback_available=self._is_openrouter_available(),
                request_id=request_id,
            )

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            self._request_counter += 1
            service = "openrouter" if self._request_counter % 2 == 0 else "mcp"

            # Check availability of chosen service
            if service == "openrouter" and not self._is_openrouter_available():
                service = "mcp"
            elif service == "mcp" and not self._is_mcp_available():
                service = "openrouter"

            return RoutingDecision(
                service=service,
                reason=f"Round-robin routing (request #{self._request_counter})",
                confidence=0.8,
                fallback_available=True,
                request_id=request_id,
            )

        if self.strategy == RoutingStrategy.LOAD_BALANCED:
            # Simple load balancing based on current success rates
            if self.metrics.total_requests > 0:
                openrouter_success_rate = (
                    (self.metrics.openrouter_requests / self.metrics.total_requests)
                    if self.metrics.openrouter_requests > 0
                    else 0.0
                )

                # Route to less-used service
                service = "mcp" if openrouter_success_rate > FALLBACK_THRESHOLD else "openrouter"
            else:
                service = "openrouter"  # Default for first request

            # Check availability
            if service == "openrouter" and not self._is_openrouter_available():
                service = "mcp"
            elif service == "mcp" and not self._is_mcp_available():
                service = "openrouter"

            return RoutingDecision(
                service=service,
                reason="Load balancing based on request distribution",
                confidence=0.8,
                fallback_available=True,
                request_id=request_id,
            )

        if self.strategy == RoutingStrategy.CAPABILITY_BASED:
            # Analyze workflow requirements (simplified implementation)
            if workflow_steps:
                # Check if any step requires specific capabilities
                for step in workflow_steps:
                    task_type = step.input_data.get("task_type", "general")
                    if task_type in ["reasoning", "complex_analysis"]:
                        # Route complex tasks to MCP
                        return RoutingDecision(
                            service="mcp",
                            reason=f"Complex task type '{task_type}' routed to MCP",
                            confidence=0.9,
                            fallback_available=self._is_openrouter_available(),
                            request_id=request_id,
                        )

            # Default to primary strategy for non-specific tasks
            service = "openrouter" if self._is_openrouter_available() else "mcp"
            return RoutingDecision(
                service=service,
                reason="Capability-based routing: no specific requirements",
                confidence=0.7,
                fallback_available=True,
                request_id=request_id,
            )

        # Fallback for unknown strategies (defensive programming for future enum additions)
        return RoutingDecision(  # type: ignore[unreachable]
            service="mcp",
            reason=f"Unknown strategy {self.strategy}, defaulting to MCP",
            confidence=0.5,
            fallback_available=self._is_openrouter_available(),
            request_id=request_id,
        )

    def _is_openrouter_available(self) -> bool:
        """Check if OpenRouter is available (circuit breaker and connection)."""
        # Check circuit breaker state
        if self.circuit_breaker:
            try:
                return self.circuit_breaker.state != CircuitBreakerState.OPEN
            except Exception:
                return False

        # Check connection state
        return (
            hasattr(self.openrouter_client, "connection_state")
            and self.openrouter_client.connection_state != MCPConnectionState.DISCONNECTED
        )

    def _is_mcp_available(self) -> bool:
        """Check if MCP is available."""
        return (
            hasattr(self.mcp_client, "connection_state")
            and self.mcp_client.connection_state != MCPConnectionState.DISCONNECTED
        )

    def _update_average_response_time(self, response_time: float) -> None:
        """Update running average response time."""
        if self.metrics.total_requests == 1:
            self.metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1  # Smoothing factor
            self.metrics.average_response_time = (
                alpha * response_time + (1 - alpha) * self.metrics.average_response_time
            )

        self.metrics.last_updated = time.time()

    def get_routing_metrics(self) -> dict[str, Any]:
        """Get current routing metrics for monitoring."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset routing metrics (useful for testing)."""
        self.metrics = RoutingMetrics()
        self.logger.info("Routing metrics reset")

    def set_traffic_percentage(self, percentage: int) -> None:
        """
        Update OpenRouter traffic percentage for gradual rollout.

        Args:
            percentage: New percentage (0-100)
        """
        if not (0 <= percentage <= CIRCUIT_BREAKER_MAX_PERCENTAGE):
            raise ValueError(f"Traffic percentage must be 0-{CIRCUIT_BREAKER_MAX_PERCENTAGE}, got {percentage}")

        old_percentage = self.openrouter_traffic_percentage
        self.openrouter_traffic_percentage = percentage

        self.logger.info(f"Updated OpenRouter traffic percentage: {old_percentage}% -> {percentage}%")
