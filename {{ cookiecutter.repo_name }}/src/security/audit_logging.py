"""Audit logging system for security events.

This module provides comprehensive audit logging capabilities for tracking
security-relevant events throughout the application lifecycle.
It implements structured logging with JSON formatting, automatic severity
mapping, and comprehensive event classification for security monitoring.

The module provides:
- Structured audit event data models with automatic metadata extraction
- Configurable severity levels for proper alerting and response
- Specialized logging methods for common security events
- JSON-formatted output for log aggregation systems
- Client IP extraction with proxy header support
- Thread-safe logging operations

Architecture:
    The audit logging system uses structlog for structured JSON logging
    with automatic timestamp formatting and exception handling. Events are
    classified by type and severity for efficient monitoring and analysis.

Key Components:
    - AuditEvent: Structured event data model with metadata
    - AuditLogger: Centralized logging system with specialized methods
    - AuditEventType: Enumeration of all trackable event types
    - AuditEventSeverity: Severity levels for alerting and response
    - Convenience functions: Simplified interface for common events

Dependencies:
    - structlog: For structured JSON logging and metadata handling
    - fastapi: For Request object processing and context extraction
    - src.config.settings: For environment-specific configuration
    - datetime: For ISO timestamp generation

Called by:
    - src.security.middleware: For request/response logging
    - src.security.rate_limiting: For rate limit violation logging
    - src.security.error_handlers: For error event logging
    - Authentication systems: For login/logout event tracking
    - API endpoints: For access control and data operation logging

Complexity: O(1) for event creation and logging, O(n) for header processing where n is header count
"""

from datetime import timezone, datetime
from enum import Enum
from typing import Any, cast

import structlog
from fastapi import Request, status

from src.config.settings import get_settings

# Configure structured logging for comprehensive audit trails
# This configuration creates JSON-formatted logs with ISO timestamps,
# stack traces, and structured metadata for security monitoring
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,  # Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structlog.stdlib.add_logger_name,  # Add logger name to each log entry
        structlog.stdlib.add_log_level,  # Add log level to each entry
        structlog.stdlib.PositionalArgumentsFormatter(),  # Format positional arguments
        structlog.processors.TimeStamper(fmt="iso"),  # Add ISO format timestamps
        structlog.processors.StackInfoRenderer(),  # Include stack trace information
        structlog.processors.format_exc_info,  # Format exception information
        structlog.processors.JSONRenderer(),  # Output in JSON format for parsing
    ],
    context_class=dict,  # Use dict for context storage
    logger_factory=structlog.stdlib.LoggerFactory(),  # Use stdlib logger factory
    wrapper_class=structlog.stdlib.BoundLogger,  # Use bound logger for context
    cache_logger_on_first_use=True,  # Cache loggers for performance
)

# Get structured logger for audit events
# This logger is specifically configured for security audit events
# and will output JSON-formatted logs with structured metadata
audit_logger = structlog.get_logger("audit")


class AuditEventType(str, Enum):
    """Types of audit events to track.

    This enumeration defines all possible audit event types that can be logged
    by the security system. Events are categorized by functional area:

    - Authentication: Login, logout, token operations
    - Authorization: Access control decisions
    - Data: CRUD operations on sensitive data
    - Security: Rate limiting, validation failures, suspicious activity
    - Administrative: System configuration changes
    - API: Request/response tracking

    Each event type uses a hierarchical naming convention (category.subcategory.action)
    to enable efficient filtering and analysis in log aggregation systems.
    """

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_CREATED = "auth.token.created"  # noqa: S105  # nosec B105  # Not a password - audit event type
    AUTH_TOKEN_REVOKED = "auth.token.revoked"  # noqa: S105  # nosec B105  # Not a password - audit event type

    # Authorization events
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_PERMISSION_ESCALATION = "authz.permission.escalation"

    # Data access events
    DATA_READ = "data.read"
    DATA_CREATE = "data.create"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"

    # Security events
    SECURITY_RATE_LIMIT_EXCEEDED = "security.rate_limit.exceeded"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious.activity"
    SECURITY_ERROR_HANDLER_TRIGGERED = "security.error_handler.triggered"
    SECURITY_VALIDATION_FAILURE = "security.validation.failure"

    # Administrative events
    ADMIN_CONFIG_CHANGE = "admin.config.change"
    ADMIN_USER_CREATE = "admin.user.create"
    ADMIN_USER_DELETE = "admin.user.delete"
    ADMIN_SYSTEM_SHUTDOWN = "admin.system.shutdown"
    ADMIN_SYSTEM_STARTUP = "admin.system.startup"

    # API events
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"


class AuditEventSeverity(str, Enum):
    """Severity levels for audit events.

    Defines the severity classification for audit events to enable proper
    alerting and response prioritization:

    - LOW: Informational events (normal operations)
    - MEDIUM: Warning events (potential issues, validation failures)
    - HIGH: Error events (failed authentication, access denied)
    - CRITICAL: Security incidents (suspicious activity, system compromise)

    Severity levels map to standard syslog levels for integration with
    monitoring and alerting systems.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEvent:
    """Structured audit event for security logging.

    Encapsulates all information needed for comprehensive security audit logging.
    Each event contains structured metadata including timestamps, request context,
    user information, and additional contextual data.

    The event supports automatic IP extraction from various header sources
    (X-Forwarded-For, X-Real-IP) to handle reverse proxy environments.

    Attributes:
        event_type: The type of audit event (from AuditEventType enum)
        severity: The severity level (from AuditEventSeverity enum)
        message: Human-readable description of the event
        timestamp: ISO format timestamp of when the event occurred
        request: FastAPI request object for context (optional)
        user_id: Identifier of the user involved (optional)
        resource: Resource being accessed/modified (optional)
        action: Action being performed (optional)
        outcome: Result of the action (success/failure/etc.)
        additional_data: Extra contextual information (optional)

    Example:
        >>> event = AuditEvent(
        ...     event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
        ...     severity=AuditEventSeverity.MEDIUM,
        ...     message="User logged in successfully",
        ...     user_id="user123",
        ...     outcome="success"
        ... )
        >>> logger.log_event(event)
    """

    def __init__(
        self,
        event_type: AuditEventType,
        severity: AuditEventSeverity,
        message: str,
        request: Request | None = None,
        user_id: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        outcome: str | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize audit event.

        Creates a structured audit event with automatic timestamp generation
        and metadata organization for security monitoring systems.

        Args:
            event_type: Type of audit event
            severity: Event severity level
            message: Human-readable event description
            request: FastAPI request object (if applicable)
            user_id: User identifier (if authenticated)
            resource: Resource being accessed/modified
            action: Action being performed
            outcome: Outcome of the action (success/failure/etc.)
            additional_data: Additional context data

        Time Complexity: O(1) - Simple attribute assignment and timestamp generation
        Space Complexity: O(1) - Fixed memory allocation for event attributes

        Called by:
            - AuditLogger.log_authentication_event(): For authentication events
            - AuditLogger.log_security_event(): For security events
            - AuditLogger.log_api_event(): For API request events
            - Convenience functions: For common event types

        Calls:
            - datetime.now(): For ISO timestamp generation
            - UTC timezone handling for consistent timestamps
        """
        self.event_type = event_type
        self.severity = severity
        self.message = message
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.request = request
        self.user_id = user_id
        self.resource = resource
        self.action = action
        self.outcome = outcome
        self.additional_data = additional_data or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert audit event to dictionary for logging.

        Transforms the audit event into a structured dictionary format suitable
        for JSON serialization and log ingestion. Automatically extracts client
        IP from request headers and includes all relevant context.

        The dictionary includes:
        - Core event metadata (type, severity, timestamp)
        - Request information (method, path, query params, client IP, headers)
        - User and resource context
        - Additional contextual data

        Returns:
            Dictionary representation of the audit event with all structured
            metadata ready for JSON serialization

        Note:
            Sensitive headers are not included in the request information
            to prevent credential leakage in logs.
        """
        event_data = {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "outcome": self.outcome,
        }

        # Add request information if available
        if self.request:
            # Add request information - cast to Any to avoid type issues
            request_data = {
                "method": self.request.method,
                "path": self.request.url.path,
                "query_params": dict(self.request.query_params) if self.request.query_params else {},
                "client_ip": self._get_client_ip(self.request),
                "user_agent": self.request.headers.get("user-agent", "unknown"),
                "referer": self.request.headers.get("referer"),
            }
            event_data["request"] = cast(Any, request_data)

        # Add user information
        if self.user_id:
            event_data["user_id"] = self.user_id

        # Add resource and action
        if self.resource:
            event_data["resource"] = self.resource

        if self.action:
            event_data["action"] = self.action

        # Add additional context data
        if self.additional_data:
            event_data["additional_data"] = dict(self.additional_data)  # type: ignore[assignment]

        return event_data

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request.

        Attempts to determine the true client IP address by checking various
        headers in order of preference. This handles reverse proxy scenarios
        where the direct client IP may be the proxy server.

        Extraction order:
        1. X-Forwarded-For header (takes first IP in chain)
        2. X-Real-IP header
        3. Direct client IP from request object

        Args:
            request: FastAPI request object containing headers and client info

        Returns:
            Client IP address as string, or "unknown" if cannot be determined

        Note:
            For X-Forwarded-For, only the first IP is used as this represents
            the original client in a proxy chain.
        """
        # Check forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"


class AuditLogger:
    """Centralized audit logging system.

    Provides a unified interface for logging security audit events throughout
    the application. Routes events to appropriate log levels based on severity
    and provides specialized methods for common event types.

    The logger automatically integrates with the application settings to
    ensure consistent behavior across environments.

    Attributes:
        settings: Application settings instance
        logger: Structured logger instance for audit events

    Thread Safety:
        This class is thread-safe as it uses structlog's thread-safe logger
        implementation.
    """

    def __init__(self) -> None:
        """Initialize audit logger."""
        self.settings = get_settings(validate_on_startup=False)
        self.logger = audit_logger

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.

        Routes the event to the appropriate log level based on its severity:
        - CRITICAL -> logger.critical()
        - HIGH -> logger.error()
        - MEDIUM -> logger.warning()
        - LOW -> logger.info()

        The event is converted to a structured dictionary and logged with
        JSON formatting for easy parsing by log aggregation systems.

        Args:
            event: The audit event to log, containing all necessary metadata

        Complexity:
            O(1) - Constant time operation for event serialization and logging
        """
        event_data = event.to_dict()

        # Log based on severity
        if event.severity == AuditEventSeverity.CRITICAL:
            self.logger.critical(event.message, **event_data)
        elif event.severity == AuditEventSeverity.HIGH:
            self.logger.error(event.message, **event_data)
        elif event.severity == AuditEventSeverity.MEDIUM:
            self.logger.warning(event.message, **event_data)
        else:
            self.logger.info(event.message, **event_data)

    def log_authentication_event(
        self,
        event_type: AuditEventType,
        request: Request,
        user_id: str | None = None,
        outcome: str = "success",
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Log authentication-related events.

        Specialized method for logging authentication events with automatic
        severity assignment based on outcome. Failed authentication attempts
        are logged at HIGH severity for security monitoring.

        Args:
            event_type: Authentication event type (e.g., AUTH_LOGIN_SUCCESS)
            request: FastAPI request object for context extraction
            user_id: User identifier involved in authentication (optional)
            outcome: Authentication outcome ("success" or "failure")
            additional_data: Additional context data (e.g., failure reason)

        Example:
            >>> logger.log_authentication_event(
            ...     AuditEventType.AUTH_LOGIN_FAILURE,
            ...     request,
            ...     outcome="failure",
            ...     additional_data={"failure_reason": "invalid_password"}
            ... )
        """
        severity = AuditEventSeverity.HIGH if outcome == "failure" else AuditEventSeverity.MEDIUM
        message = f"Authentication event: {event_type.value} - {outcome}"

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            request=request,
            user_id=user_id,
            action="authenticate",
            outcome=outcome,
            additional_data=additional_data,
        )

        self.log_event(event)

    def log_security_event(
        self,
        event_type: AuditEventType,
        message: str,
        request: Request | None = None,
        severity: AuditEventSeverity = AuditEventSeverity.HIGH,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Log security-related events.

        General-purpose method for logging security events such as rate limit
        violations, suspicious activity, validation failures, and error handler
        activations. Defaults to HIGH severity for security monitoring.

        Args:
            event_type: Security event type (e.g., SECURITY_RATE_LIMIT_EXCEEDED)
            message: Human-readable event description
            request: FastAPI request object for context (optional)
            severity: Event severity level (defaults to HIGH)
            additional_data: Additional context data (e.g., rate limit details)

        Example:
            >>> logger.log_security_event(
            ...     AuditEventType.SECURITY_RATE_LIMIT_EXCEEDED,
            ...     "Rate limit exceeded for API endpoint",
            ...     request,
            ...     additional_data={"rate_limit": "60/minute"}
            ... )
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            request=request,
            action="security_check",
            outcome="detected",
            additional_data=additional_data,
        )

        self.log_event(event)

    def log_api_event(
        self,
        request: Request,
        response_status: int,
        processing_time: float,
        user_id: str | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Log API request/response events.

        Logs all API requests with automatic severity assignment based on
        HTTP status codes:
        - 5xx status codes: HIGH severity (server errors)
        - 4xx status codes: MEDIUM severity (client errors)
        - 2xx/3xx status codes: LOW severity (successful)

        Args:
            request: FastAPI request object containing method, path, headers
            response_status: HTTP response status code for severity determination
            processing_time: Request processing time in seconds
            user_id: User identifier if authenticated (optional)
            additional_data: Additional context data to include in the log (optional)

        Side Effects:
            Automatically includes response status and processing time in
            additional_data for performance monitoring.

        Example:
            >>> logger.log_api_event(
            ...     request,
            ...     response_status=200,
            ...     processing_time=0.342,
            ...     user_id="user123"
            ... )
        """
        # HTTP status code thresholds
        server_error_threshold = status.HTTP_500_INTERNAL_SERVER_ERROR
        client_error_threshold = status.HTTP_400_BAD_REQUEST

        # Determine severity based on status code
        if response_status >= server_error_threshold:
            severity = AuditEventSeverity.HIGH
            outcome = "server_error"
        elif response_status >= client_error_threshold:
            severity = AuditEventSeverity.MEDIUM
            outcome = "client_error"
        else:
            severity = AuditEventSeverity.LOW
            outcome = "success"

        message = f"API request: {request.method} {request.url.path} -> {response_status}"

        # Merge default API data with any additional data provided
        api_data = {
            "response_status": response_status,
            "processing_time": processing_time,
        }
        if additional_data:
            api_data.update(additional_data)

        event = AuditEvent(
            event_type=AuditEventType.API_REQUEST,
            severity=severity,
            message=message,
            request=request,
            user_id=user_id,
            resource=request.url.path,
            action=request.method.lower(),
            outcome=outcome,
            additional_data=api_data,
        )

        self.log_event(event)


# Global audit logger instance
# Singleton instance for application-wide audit logging
# Thread-safe for concurrent access across the application
audit_logger_instance = AuditLogger()


# Convenience functions for common audit events
# These functions provide a simplified interface for the most common
# security events, using the global logger instance


def log_authentication_success(request: Request, user_id: str) -> None:
    """Log successful authentication.

    Convenience function for logging successful user authentication events.
    Automatically uses AUTH_LOGIN_SUCCESS event type and "success" outcome.

    Args:
        request: FastAPI request object for context
        user_id: Identifier of the successfully authenticated user

    Example:
        >>> log_authentication_success(request, "user123")
    """
    audit_logger_instance.log_authentication_event(
        AuditEventType.AUTH_LOGIN_SUCCESS,
        request,
        user_id,
        "success",
    )


def log_authentication_failure(request: Request, reason: str = "invalid_credentials") -> None:
    """Log failed authentication.

    Convenience function for logging failed authentication attempts.
    Automatically uses AUTH_LOGIN_FAILURE event type and HIGH severity.

    Args:
        request: FastAPI request object for context
        reason: Reason for authentication failure (defaults to "invalid_credentials")

    Example:
        >>> log_authentication_failure(request, "account_locked")
    """
    audit_logger_instance.log_authentication_event(
        AuditEventType.AUTH_LOGIN_FAILURE,
        request,
        None,
        "failure",
        {"failure_reason": reason},
    )


def log_rate_limit_exceeded(request: Request, limit: str) -> None:
    """Log rate limit exceeded event.

    Convenience function for logging rate limit violations.
    Uses MEDIUM severity as rate limiting is a normal protective measure.

    Args:
        request: FastAPI request object for context
        limit: Rate limit that was exceeded (e.g., "60/minute")

    Example:
        >>> log_rate_limit_exceeded(request, "100/minute")
    """
    audit_logger_instance.log_security_event(
        AuditEventType.SECURITY_RATE_LIMIT_EXCEEDED,
        "Rate limit exceeded",
        request,
        AuditEventSeverity.MEDIUM,
        {"rate_limit": limit},
    )


def log_validation_failure(request: Request, validation_errors: list) -> None:
    """Log validation failure event.

    Convenience function for logging input validation failures.
    Uses MEDIUM severity as validation failures may indicate attack attempts.

    Args:
        request: FastAPI request object for context
        validation_errors: List of validation errors that occurred

    Example:
        >>> log_validation_failure(request, ["Invalid email format", "Password too short"])
    """
    audit_logger_instance.log_security_event(
        AuditEventType.SECURITY_VALIDATION_FAILURE,
        "Request validation failed",
        request,
        AuditEventSeverity.MEDIUM,
        {"validation_errors": validation_errors},
    )


def log_error_handler_triggered(request: Request, error_type: str, error_message: str) -> None:
    """Log error handler activation.

    Convenience function for logging when error handlers are triggered.
    Uses HIGH severity as error handlers indicate application issues.

    Args:
        request: FastAPI request object for context
        error_type: Type of error that occurred (e.g., "ValidationError")
        error_message: Error message details

    Example:
        >>> log_error_handler_triggered(request, "HTTPException", "Internal server error")
    """
    audit_logger_instance.log_security_event(
        AuditEventType.SECURITY_ERROR_HANDLER_TRIGGERED,
        f"Error handler triggered: {error_type}",
        request,
        AuditEventSeverity.HIGH,
        {"error_type": error_type, "error_message": error_message},
    )


def log_api_request(request: Request, response_status: int, processing_time: float) -> None:
    """Log API request/response.

    Convenience function for logging API requests and responses.
    Automatically determines severity based on HTTP status code.

    Args:
        request: FastAPI request object
        response_status: HTTP response status code
        processing_time: Request processing time in seconds

    Example:
        >>> log_api_request(request, 200, 0.245)
    """
    audit_logger_instance.log_api_event(request, response_status, processing_time)
