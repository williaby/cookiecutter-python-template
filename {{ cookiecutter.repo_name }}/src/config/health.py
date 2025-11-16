"""Health check models and utilities for configuration status monitoring.

This module implements Phase 5 of the Core Configuration System: Health Check Integration.
It provides configuration status models and utilities for exposing operational
information through health check endpoints without revealing sensitive data.
"""

import logging
import os
import re
import sys
from datetime import datetime

if sys.version_info >= (3, 11):
    from datetime import timezone
else:
    from datetime import timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from .constants import FILE_PATH_PATTERNS, SECRET_FIELD_NAMES, SENSITIVE_ERROR_PATTERNS
from .settings import (
    ApplicationSettings,
    ConfigurationValidationError,
    get_settings,
    validate_encryption_available,
)
from .validation import validate_configuration_on_startup

# Conditional import for MCP integration to avoid circular imports
try:
    from src.mcp_integration import MCPClient, MCPConfigurationManager, ParallelSubagentExecutor
except ImportError:
    # MCP integration may not be available during early bootstrap
    MCPClient = None  # type: ignore[misc,assignment]
    MCPConfigurationManager = None  # type: ignore[misc,assignment]
    ParallelSubagentExecutor = None  # type: ignore[misc,assignment]

# Compile regex patterns once for better performance
_COMPILED_SENSITIVE_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in SENSITIVE_ERROR_PATTERNS
]

_COMPILED_FILE_PATH_PATTERNS = [re.compile(pattern) for pattern in FILE_PATH_PATTERNS]

# Threshold for determining if a field name or value might be sensitive data
_SENSITIVE_DATA_LENGTH_THRESHOLD = 8

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for configuration and system components."""

    def __init__(self, settings: ApplicationSettings) -> None:
        """Initialize health checker with settings.

        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    async def check_health(self) -> dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            Dictionary with health status information
        """
        try:
            status = get_configuration_status(self.settings)
            mcp_health = await get_mcp_configuration_health()

            overall_healthy = bool(status.config_healthy) and bool(mcp_health.get("healthy", False))

            return {
                "healthy": overall_healthy,
                "configuration": status.model_dump(),
                "mcp": mcp_health,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self.logger.error("Health check failed: %s", e)
            return {
                "healthy": False,
                "error": f"Health check failed: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


class ConfigurationStatusModel(BaseModel):
    """Configuration status model for health check responses.

    This model represents the current state of the application configuration
    for operational monitoring purposes. It exposes only non-sensitive
    information that is useful for debugging and system health monitoring.

    Attributes:
        environment: Current deployment environment (dev/staging/prod)
        version: Application version string
        debug: Whether debug mode is enabled
        config_loaded: Whether configuration loaded successfully
        encryption_enabled: Whether encryption is available and working
        config_source: Primary source of configuration (env_vars, files, defaults)
        validation_status: Whether configuration validation passed
        validation_errors: Non-sensitive validation error summaries
        secrets_configured: Count of configured secret fields (not values)
        timestamp: When this status was generated
    """

    environment: str = Field(
        description="Current deployment environment (dev/staging/prod)",
    )

    version: str = Field(description="Application version string")

    debug: bool = Field(description="Whether debug mode is enabled")

    config_loaded: bool = Field(description="Whether configuration loaded successfully")

    encryption_enabled: bool = Field(
        description="Whether encryption is available and working",
    )

    config_source: str = Field(
        description="Primary source of configuration (env_vars, files, defaults)",
    )

    validation_status: str = Field(
        description="Configuration validation status (passed, failed, warning)",
    )

    validation_errors: list[str] = Field(
        default_factory=list,
        description="Non-sensitive validation error summaries",
    )

    secrets_configured: int = Field(
        description="Count of configured secret fields (not values)",
    )

    api_host: str = Field(
        description="API host address (safe to expose for operational monitoring)",
    )

    api_port: int = Field(description="API port number")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this status was generated (UTC)",
    )

    @computed_field
    def config_healthy(self) -> bool:
        """Computed field indicating overall configuration health.

        Returns:
            True if configuration is healthy (loaded and validated successfully)
        """
        return self.config_loaded and self.validation_status in ("passed", "warning")

    model_config = {"json_encoders": {datetime: lambda dt: dt.isoformat() + "Z"}}


def _count_configured_secrets(settings: ApplicationSettings) -> int:
    """Count the number of configured secret fields without exposing values.

    Args:
        settings: The settings instance to analyze

    Returns:
        Number of secret fields that have non-empty values
    """
    configured_count = 0
    for field_name in SECRET_FIELD_NAMES:
        field_value = getattr(settings, field_name, None)
        if field_value is not None and field_value.get_secret_value().strip():
            configured_count += 1

    return configured_count


def _determine_config_source(settings: ApplicationSettings) -> str:
    """Determine the primary source of configuration.

    This is a best-effort determination based on typical patterns.
    Since Pydantic doesn't track sources directly, we use heuristics.

    Args:
        settings: The settings instance to analyze

    Returns:
        Primary configuration source identifier
    """
    # Check if key environment variables are set
    env_vars_set = any(
        os.getenv(f"PROMPTCRAFT_{key.upper()}") for key in ["ENVIRONMENT", "API_HOST", "API_PORT", "DEBUG"]
    )

    if env_vars_set:
        return "env_vars"

    # Check if environment-specific .env files likely exist
    project_root = Path(__file__).parent.parent.parent
    env_files = [
        project_root / f".env.{settings.environment}",
        project_root / f".env.{settings.environment}.gpg",
        project_root / ".env",
        project_root / ".env.gpg",
    ]

    if any(env_file.exists() for env_file in env_files):
        return "env_files"

    return "defaults"


def _sanitize_validation_errors(errors: list[str]) -> list[str]:
    """Sanitize validation errors to remove sensitive information.

    Uses a two-stage hybrid approach recommended by security consensus:
    1. High-risk patterns get full message replacement (schema protection)
    2. Other messages get quote-level sanitization (context preservation)

    This balances maximum security for known sensitive patterns while
    preserving debugging context for general validation errors.

    Args:
        errors: List of validation error messages

    Returns:
        Sanitized error messages safe for health check exposure
    """
    sanitized = []

    # High-risk patterns that require full message replacement
    # These patterns reveal sensitive schema/validation information
    high_risk_patterns = [
        (
            re.compile(r"database\s+password.*(?:too\s+short|invalid|failed)", re.IGNORECASE),
            "Password configuration issue (details hidden)",
        ),
        (
            re.compile(r"api\s+key.*(?:invalid|failed|expired)", re.IGNORECASE),
            "API key configuration issue (details hidden)",
        ),
        (
            re.compile(r"secret\s+key.*(?:must\s+not\s+contain|invalid|failed)", re.IGNORECASE),
            "Secret key configuration issue (details hidden)",
        ),
        (
            re.compile(r"jwt\s+secret.*(?:failed|invalid)", re.IGNORECASE),
            "JWT secret configuration issue (details hidden)",
        ),
        (
            re.compile(r"configuration\s+file.*(?:contains\s+errors|invalid)", re.IGNORECASE),
            "Configuration file path issue (path hidden)",
        ),
    ]

    for error in errors:
        original_error = error
        sanitized_error = error
        high_risk_matched = False

        # Stage 1: Check for high-risk patterns first (full replacement)
        for pattern, replacement in high_risk_patterns:
            if pattern.search(error):
                sanitized_error = replacement
                high_risk_matched = True
                break

        # Stage 2: If no high-risk pattern matched, apply quote-level sanitization
        if not high_risk_matched:
            # First, sanitize quoted values that might contain sensitive data
            def is_sensitive_quoted_value(match: re.Match[str]) -> bool:
                value = match.group(1) if match.groups() else match.group(0)[1:-1]
                # Don't sanitize short field names that look like identifiers
                if (
                    len(value) <= _SENSITIVE_DATA_LENGTH_THRESHOLD
                    and "_" in value
                    and value.islower()
                    and value.isalnum() is False
                ):
                    return False
                # Check if the quoted value looks like sensitive data or is just any value being quoted
                sensitive_indicators = ["pass", "secret", "token", "auth", "credential"]
                return (
                    any(indicator in value.lower() for indicator in sensitive_indicators)
                    or len(value) > _SENSITIVE_DATA_LENGTH_THRESHOLD
                )

            # Replace quoted values that look sensitive
            sanitized_error = re.sub(
                r"'([^']*)'",
                lambda m: "'***'" if is_sensitive_quoted_value(m) else m.group(0),
                sanitized_error,
            )
            sanitized_error = re.sub(
                r'"([^"]*)"',
                lambda m: '"***"' if is_sensitive_quoted_value(m) else m.group(0),
                sanitized_error,
            )

            # Check if quote sanitization modified the message
            quote_sanitized = sanitized_error != original_error

            # Only apply legacy pattern replacement if quote sanitization didn't handle it
            if not quote_sanitized:
                # Check for sensitive patterns using pre-compiled regex patterns
                # Do pattern replacement for messages that contain sensitive terms
                pattern_matched = False
                # More selective pattern matching - avoid replacing whole messages for common field references
                contains_field_reference = re.search(r"field\s+['\"]*(?:password|api_key|secret)", error, re.IGNORECASE)
                if not contains_field_reference:
                    for compiled_pattern, replacement in _COMPILED_SENSITIVE_PATTERNS:
                        if compiled_pattern.search(error):
                            sanitized_error = replacement
                            pattern_matched = True
                            break

                # Check for file path patterns using pre-compiled patterns
                if not pattern_matched and sanitized_error == original_error:
                    for compiled_path_pattern in _COMPILED_FILE_PATH_PATTERNS:
                        if compiled_path_pattern.search(error):
                            sanitized_error = "Configuration file path issue (path hidden)"
                            break

        sanitized.append(sanitized_error)

    return sanitized


def get_configuration_status(settings: ApplicationSettings) -> ConfigurationStatusModel:
    """Generate current configuration status for health check endpoints.

    This function creates a comprehensive status model showing the current
    state of application configuration. It's designed to be safe for exposure
    in health check endpoints, containing only operational information and
    no sensitive data.

    Args:
        settings: The application settings instance to analyze

    Returns:
        ConfigurationStatusModel with current status information

    Example:
        >>> settings = get_settings()
        >>> status = get_configuration_status(settings)
        >>> print(f"Config healthy: {status.config_healthy}")
        >>> print(f"Secrets configured: {status.secrets_configured}")
    """
    logger.debug("Generating configuration status for health check")

    # Test encryption availability
    encryption_available = validate_encryption_available()

    # Count configured secrets
    secrets_count = _count_configured_secrets(settings)

    # Determine configuration source
    config_source = _determine_config_source(settings)

    # Test configuration validation
    validation_status = "passed"
    validation_errors: list[str] = []

    try:
        validate_configuration_on_startup(settings)
        logger.debug("Configuration validation passed")
    except ConfigurationValidationError as e:
        validation_status = "failed"
        validation_errors = _sanitize_validation_errors(e.field_errors)
        logger.warning(
            "Configuration validation failed: %d errors",
            len(validation_errors),
        )
    except (ValueError, TypeError, AttributeError) as e:
        validation_status = "failed"
        validation_errors = ["Configuration format error"]
        logger.error("Configuration format error: %s", type(e).__name__)
    except Exception:
        logger.exception("Unexpected validation error")
        raise

    # Create status model
    status = ConfigurationStatusModel(
        environment=settings.environment,
        version=settings.version,
        debug=settings.debug,
        config_loaded=True,  # If we got here, config loaded successfully
        encryption_enabled=encryption_available,
        config_source=config_source,
        validation_status=validation_status,
        validation_errors=validation_errors,
        secrets_configured=secrets_count,
        api_host=settings.api_host,
        api_port=settings.api_port,
    )

    logger.debug(
        "Configuration status generated: healthy=%s, secrets=%d, source=%s",
        status.config_healthy,
        secrets_count,
        config_source,
    )

    return status


def get_configuration_health_summary() -> dict[str, Any]:
    """Get a simplified configuration health summary for quick checks.

    This function provides a minimal health check response for use in
    simple monitoring systems that just need to know if configuration
    is working properly.

    Returns:
        Dictionary with basic health information

    Example:
        >>> summary = get_configuration_health_summary()
        >>> if summary["healthy"]:
        ...     print("Configuration is healthy")
    """
    try:
        settings = get_settings(validate_on_startup=False)
        status = get_configuration_status(settings)

        return {
            "healthy": status.config_healthy,
            "environment": status.environment,
            "version": status.version,
            "config_loaded": status.config_loaded,
            "encryption_available": status.encryption_enabled,
            "timestamp": status.timestamp.isoformat(),
        }
    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Configuration health summary format error: %s", type(e).__name__)
        return {
            "healthy": False,
            "error": "Configuration health check failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        logger.exception("Failed to generate configuration health summary")
        return {
            "healthy": False,
            "error": "Configuration health check failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def get_mcp_configuration_health() -> dict[str, Any]:
    """Get MCP configuration health status.

    Returns:
        Dictionary with MCP configuration health information
    """
    try:
        # Check if MCP components are available
        if MCPClient is None or MCPConfigurationManager is None or ParallelSubagentExecutor is None:
            raise ImportError("MCP integration components not available")

        # Initialize MCP components
        config_manager = MCPConfigurationManager()
        mcp_client = MCPClient()
        parallel_executor = ParallelSubagentExecutor(config_manager, mcp_client)

        # Get health status from all components
        config_health = config_manager.get_health_status()
        client_health = await mcp_client.health_check()
        executor_health = await parallel_executor.health_check()

        overall_healthy = (
            config_health.get("configuration_valid", False)
            and client_health.get("overall_status") == "healthy"
            and executor_health.get("status") == "healthy"
        )

        return {
            "healthy": overall_healthy,
            "mcp_configuration": config_health,
            "mcp_client": client_health,
            "parallel_executor": executor_health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ImportError as e:
        logger.warning("MCP integration not available: %s", e)
        return {
            "healthy": False,
            "error": "MCP integration not available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("MCP configuration health check failed: %s", e)
        return {
            "healthy": False,
            "error": f"MCP health check failed: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
