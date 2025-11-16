"""Core application settings using Pydantic BaseSettings.

This module defines the core configuration schema for the PromptCraft-Hybrid application.
It provides type-safe configuration with validation and environment-specific loading.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.encryption import (
    EncryptionError,
    GPGError,
    load_encrypted_env,
    validate_environment_keys,
)

# Import constants after other imports to avoid circular dependency
from .constants import SECRET_FIELD_NAMES

# Constants
MAX_HOSTNAME_LENGTH = 253
MAX_APP_NAME_LENGTH = 100
MAX_PORT_NUMBER = 65535
PRIVILEGED_PORT_THRESHOLD = 1024
HTTP_PORT = 80
HTTPS_PORT = 443


def _get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path object pointing to the project root directory.
    """
    current_file = Path(__file__).resolve()
    # Navigate up from src/config/settings.py to project root
    return current_file.parent.parent.parent


def _detect_environment() -> str:
    """Detect the current environment from environment variables or .env files.

    Returns:
        The detected environment string (dev, staging, or prod).
        Defaults to 'dev' if no environment is detected.
    """
    # First check environment variable
    env_from_var = os.getenv("PROMPTCRAFT_ENVIRONMENT")
    if env_from_var and env_from_var in ("dev", "staging", "prod"):
        return env_from_var

    # Then check .env file if it exists
    project_root = _get_project_root()
    env_file = project_root / ".env"

    if env_file.exists():
        try:
            with env_file.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line.startswith("PROMPTCRAFT_ENVIRONMENT="):
                        env_value = line.split("=", 1)[1].strip().strip("\"'")
                        if env_value in ("dev", "staging", "prod"):
                            return env_value
        except (OSError, UnicodeDecodeError):
            # If we can't read the file, fall back to default
            pass

    # Default to development environment
    return "dev"


def _load_env_file(file_path: Path) -> dict[str, Any]:
    """Load environment variables from a .env file.

    Args:
        file_path: Path to the .env file to load.

    Returns:
        Dictionary of environment variables from the file.
    """
    env_vars: dict[str, Any] = {}

    if not file_path.exists():
        return env_vars

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")  # Remove quotes

                    # Only load variables with PROMPTCRAFT_ prefix
                    if key.startswith("PROMPTCRAFT_"):
                        # Remove prefix for Pydantic
                        pydantic_key = key[len("PROMPTCRAFT_") :].lower()
                        env_vars[pydantic_key] = value

    except (OSError, UnicodeDecodeError) as e:
        # Log warning but don't fail - graceful degradation
        logger = logging.getLogger(__name__)
        logger.warning("Could not read %s: %s", file_path, e)

    return env_vars


def _load_encrypted_env_file(file_path: Path) -> dict[str, Any]:
    """Load environment variables from an encrypted .env file.

    Args:
        file_path: Path to the encrypted .env file to load.

    Returns:
        Dictionary of environment variables from the encrypted file.
    """
    env_vars: dict[str, Any] = {}

    if not file_path.exists():
        return env_vars

    try:
        # Use the encryption utility to load encrypted env file
        raw_env_vars = load_encrypted_env(str(file_path))

        # Process variables with PROMPTCRAFT_ prefix
        for key, value in raw_env_vars.items():
            if key.startswith("PROMPTCRAFT_"):
                # Remove prefix for Pydantic
                pydantic_key = key[len("PROMPTCRAFT_") :].lower()
                env_vars[pydantic_key] = value

    except (FileNotFoundError, GPGError) as e:
        # Log warning but don't fail - graceful degradation
        logger = logging.getLogger(__name__)
        logger.warning("Could not load encrypted file %s: %s", file_path, e)

    return env_vars


def _env_file_settings() -> dict[str, Any]:
    """Custom settings source for environment-specific .env file loading.

    This function implements the secure file loading hierarchy:
    1. .env.{environment}.gpg file (encrypted, highest priority)
    2. .env.{environment} file (fallback for development)
    3. .env.gpg file (encrypted base file)
    4. .env file (fallback base file)

    Returns:
        Dictionary of configuration values from .env files.
    """
    project_root = _get_project_root()
    env_vars: dict[str, Any] = {}

    # Detect current environment
    current_env = _detect_environment()

    # Load base .env file first (lowest priority)
    base_env_file = project_root / ".env"
    env_vars.update(_load_env_file(base_env_file))

    # Try to load encrypted base .env file (higher priority)
    base_encrypted_file = project_root / ".env.gpg"
    env_vars.update(_load_encrypted_env_file(base_encrypted_file))

    # Load environment-specific .env file (higher priority)
    env_specific_file = project_root / f".env.{current_env}"
    env_vars.update(_load_env_file(env_specific_file))

    # Try to load encrypted environment-specific file (highest priority)
    env_encrypted_file = project_root / f".env.{current_env}.gpg"
    env_vars.update(_load_encrypted_env_file(env_encrypted_file))

    return env_vars


class ConfigurationValidationError(Exception):
    """Raised when configuration validation fails with detailed error information."""

    def __init__(
        self,
        message: str,
        field_errors: list[str] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        """Initialize configuration validation error.

        Args:
            message: The main error message
            field_errors: List of specific field validation errors
            suggestions: List of suggested fixes or valid values
        """
        super().__init__(message)
        self.field_errors = field_errors or []
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        """Return formatted error message with all details."""
        parts = [super().__str__()]

        if self.field_errors:
            parts.append("\nField Validation Errors:")
            for error in self.field_errors:
                parts.append(f"  • {error}")

        if self.suggestions:
            parts.append("\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")

        return "\n".join(parts)


class ApplicationSettings(BaseSettings):
    """Core application configuration settings.

    This class defines the base configuration schema for the PromptCraft-Hybrid
    application. It uses Pydantic BaseSettings for environment variable loading
    and validation with environment-specific .env file support.

    Attributes:
        app_name: The application name for logging and identification.
        version: The application version string.
        environment: The deployment environment (dev, staging, prod).
        debug: Whether debug mode is enabled.
        api_host: The host address for the API server.
        api_port: The port number for the API server.
    """

    app_name: str = Field(
        default="PromptCraft-Hybrid",
        description="Application name for logging and identification",
    )

    version: str = Field(
        default="0.1.0",
        description="Application version string",
    )

    environment: Literal["dev", "staging", "prod"] = Field(
        default="dev",
        description="Deployment environment (dev, staging, prod)",
    )

    debug: bool = Field(
        default=True,
        description="Whether debug mode is enabled",
    )

    api_host: str = Field(
        default="0.0.0.0",  # nosec B104 # Intentional bind to all interfaces for development  # noqa: S104
        description="Host address for the API server",
    )

    api_port: int = Field(
        default=8000,
        description="Port number for the API server",
    )

    # Database Configuration
    database_host: str = Field(
        default="192.168.1.16",
        description="Database host address",
    )

    database_port: int = Field(
        default=5432,
        description="Database port number",
    )

    database_name: str = Field(
        default="promptcraft",
        description="Database name",
    )

    database_username: str = Field(
        default="promptcraft",
        description="Database username",
    )

    database_timeout: float = Field(
        default=30.0,
        description="Database connection timeout in seconds",
    )

    # Database Configuration (sensitive values)
    database_password: SecretStr | None = Field(
        default=None,
        description="Database password (sensitive - never logged)",
    )

    database_url: SecretStr | None = Field(
        default=None,
        description="Complete database connection URL (sensitive - never logged)",
    )

    # PostgreSQL Database Configuration for AUTH-1
    db_host: str = Field(
        default="192.168.1.16",
        description="PostgreSQL database host address",
    )

    db_port: int = Field(
        default=5435,
        description="PostgreSQL database port number",
    )

    db_name: str = Field(
        default="promptcraft_auth",
        description="PostgreSQL database name",
    )

    db_user: str = Field(
        default="promptcraft_user",
        description="PostgreSQL database user",
    )

    db_password: SecretStr | None = Field(
        default=None,
        description="PostgreSQL database password (sensitive - never logged)",
    )

    # Database Connection Pool Configuration
    db_pool_size: int = Field(
        default=5,
        description="Database connection pool size",
    )

    db_pool_max_overflow: int = Field(
        default=10,
        description="Maximum overflow connections beyond pool size",
    )

    db_pool_timeout: float = Field(
        default=30.0,
        description="Database connection pool timeout in seconds",
    )

    db_pool_recycle: int = Field(
        default=3600,
        description="Database connection recycle time in seconds",
    )

    db_echo: bool = Field(
        default=False,
        description="Whether to echo SQL queries for debugging",
    )

    # API Keys and Secrets (sensitive values)
    api_key: SecretStr | None = Field(
        default=None,
        description="Primary API key for external services (sensitive - never logged)",
    )

    secret_key: SecretStr | None = Field(
        default=None,
        description="Application secret key for encryption/signing (sensitive - never logged)",
    )

    azure_openai_api_key: SecretStr | None = Field(
        default=None,
        description="Azure OpenAI API key (sensitive - never logged)",
    )

    # JWT and Authentication Secrets
    jwt_secret_key: SecretStr | None = Field(
        default=None,
        description="JWT signing secret key (sensitive - never logged)",
    )

    # External Service Configuration
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant vector database API key (sensitive - never logged)",
    )

    encryption_key: SecretStr | None = Field(
        default=None,
        description="Encryption key for data at rest (sensitive - never logged)",
    )

    # MCP (Model Context Protocol) Configuration
    mcp_server_url: str = Field(
        default="http://localhost:3000",
        description="Zen MCP Server endpoint URL",
    )

    mcp_api_key: SecretStr | None = Field(
        default=None,
        description="MCP server API key for authentication (sensitive - never logged)",
    )

    mcp_timeout: float = Field(
        default=30.0,
        description="MCP request timeout in seconds",
    )

    mcp_max_retries: int = Field(
        default=3,
        description="Maximum number of MCP request retries",
    )

    mcp_enabled: bool = Field(
        default=True,
        description="Whether MCP integration is enabled",
    )

    mcp_health_check_interval: float = Field(
        default=60.0,
        description="MCP health check interval in seconds",
    )

    # OpenRouter Configuration
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key for AI model routing (sensitive - never logged)",
    )

    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    openrouter_timeout: float = Field(
        default=30.0,
        description="OpenRouter request timeout in seconds",
    )

    openrouter_max_retries: int = Field(
        default=3,
        description="Maximum number of OpenRouter request retries",
    )

    openrouter_enabled: bool = Field(
        default=True,
        description="Whether OpenRouter integration is enabled",
    )

    openrouter_traffic_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Percentage of traffic to route to OpenRouter (0-100) for gradual rollout",
    )

    # Qdrant Vector Database Configuration
    qdrant_host: str = Field(
        default="192.168.1.16",
        description="Qdrant vector database host address",
    )

    qdrant_port: int = Field(
        default=6333,
        description="Qdrant vector database port number",
    )

    qdrant_timeout: float = Field(
        default=30.0,
        description="Qdrant request timeout in seconds",
    )

    qdrant_enabled: bool = Field(
        default=True,
        description="Whether Qdrant vector database integration is enabled",
    )

    # Vector Store Configuration
    vector_store_type: str = Field(
        default="auto",
        description="Vector store type: 'auto', 'qdrant', or 'mock'",
    )

    vector_dimensions: int = Field(
        default=384,
        description="Default vector embedding dimensions",
    )

    # Network Security
    trusted_proxy_1: str | None = Field(
        default=None,
        description="IP address of the first trusted proxy server",
    )
    trusted_proxy_2: str | None = Field(
        default=None,
        description="IP address of the second trusted proxy server",
    )

    # Performance Configuration
    performance_monitoring_enabled: bool = Field(
        default=True,
        description="Whether performance monitoring is enabled",
    )

    max_concurrent_queries: int = Field(
        default=10,
        description="Maximum number of concurrent queries",
    )

    query_timeout: float = Field(
        default=30.0,
        description="Query timeout in seconds",
    )

    # File Upload Security Configuration
    max_files: int = Field(
        default=5,
        description="Maximum number of files that can be uploaded at once",
    )

    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file size in bytes (default: 10MB)",
    )

    supported_file_types: list[str] = Field(
        default=[
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".css",
            ".csv",
            ".tsv",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".sh",
            ".bat",
            ".ps1",
            ".sql",
            ".r",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".clj",
            ".hs",
            ".lua",
            ".pl",
            ".ts",
            ".jsx",
            ".tsx",
            ".vue",
            ".svelte",
            ".dart",
            ".f90",
            ".f95",
            ".m",
            ".mm",
            ".pas",
            ".asm",
            ".s",
            ".vb",
            ".cs",
            ".fs",
            ".ml",
            ".mli",
            ".ocaml",
            ".elm",
            ".purescript",
            ".nim",
            ".crystal",
            ".d",
            ".zig",
            ".odin",
            ".v",
            ".lean",
            ".agda",
            ".coq",
            ".idris",
            ".prolog",
            ".erlang",
            ".elixir",
            ".haskell",
            ".dockerfile",
            ".gitignore",
            ".gitattributes",
            ".license",
            ".readme",
            ".changelog",
            ".todo",
            ".cfg",
            ".ini",
            ".conf",
            ".config",
            ".properties",
            ".env",
            ".toml",
            ".lock",
            ".log",
        ],
        description="List of supported file extensions for upload",
    )

    # Health Check Configuration
    health_check_enabled: bool = Field(
        default=True,
        description="Whether health checks are enabled",
    )

    health_check_interval: float = Field(
        default=30.0,
        description="Health check interval in seconds",
    )

    # Error Recovery Configuration
    error_recovery_enabled: bool = Field(
        default=True,
        description="Whether error recovery is enabled",
    )

    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Whether circuit breaker is enabled",
    )

    retry_enabled: bool = Field(
        default=True,
        description="Whether retry logic is enabled",
    )

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Number of failures before opening circuit breaker",
    )
    circuit_breaker_success_threshold: int = Field(
        default=3,
        description="Number of successes to close circuit breaker from half-open",
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60,
        description="Seconds before attempting recovery from open state",
    )
    circuit_breaker_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts before circuit breaker action",
    )
    circuit_breaker_base_delay: float = Field(
        default=1.0,
        description="Base delay in seconds for exponential backoff",
    )
    circuit_breaker_max_delay: float = Field(
        default=60.0,
        description="Maximum delay in seconds between retries",
    )
    circuit_breaker_backoff_multiplier: float = Field(
        default=2.0,
        description="Multiplier for exponential backoff calculation",
    )
    circuit_breaker_jitter_enabled: bool = Field(
        default=True,
        description="Enable jitter to prevent thundering herd effect",
    )
    circuit_breaker_health_check_interval: int = Field(
        default=30,
        description="Seconds between automated health checks",
    )
    circuit_breaker_health_check_timeout: float = Field(
        default=5.0,
        description="Timeout in seconds for health check operations",
    )

    model_config = SettingsConfigDict(
        extra="forbid",  # Prevent unknown settings
        case_sensitive=False,
        env_prefix="PROMPTCRAFT_",
    )

    @field_validator("api_host")
    @classmethod
    def validate_api_host(cls, v: str) -> str:
        """Validate the API host address with detailed error messages.

        Args:
            v: The host address to validate.

        Returns:
            The validated host address.

        Raises:
            ValueError: If the host address is invalid with detailed guidance.
        """
        if not v.strip():
            raise ValueError(
                "API host cannot be empty. "
                "Common values: '0.0.0.0' (all interfaces), 'localhost' (local only), "
                "'127.0.0.1' (loopback), or a valid IP address/hostname.",
            )

        v = v.strip()

        # Check for valid IP address format
        ip_pattern = re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
        )

        # Check for valid hostname format
        hostname_pattern = re.compile(
            r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)$",
        )

        # Allow common special cases
        if v in (
            "0.0.0.0",  # nosec B104 # Valid development host binding  # noqa: S104
            "localhost",
            "127.0.0.1",
        ):
            return v

        # Validate IP address format
        if ip_pattern.match(v):
            return v

        # Validate hostname format
        if hostname_pattern.match(v) and len(v) <= MAX_HOSTNAME_LENGTH:
            return v

        # If none match, provide detailed error
        raise ValueError(
            f"Invalid API host format: '{v}'. "
            "Host must be a valid IP address (e.g., '192.168.1.100'), "
            "hostname (e.g., 'api.example.com'), or special value "
            "('0.0.0.0', 'localhost', '127.0.0.1').",
        )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate the version string with semantic version checking.

        Args:
            v: The version string to validate.

        Returns:
            The validated version string.

        Raises:
            ValueError: If the version string is invalid with guidance.
        """
        if not v.strip():
            raise ValueError(
                "Version cannot be empty. "
                "Use semantic versioning format: 'MAJOR.MINOR.PATCH' (e.g., '1.0.0', '0.1.0').",
            )

        v = v.strip()

        # Check for semantic version format (flexible)
        semver_pattern = re.compile(
            r"^\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.-]+)?(?:\+[a-zA-Z0-9.-]+)?$",
        )

        if not semver_pattern.match(v):
            raise ValueError(
                f"Invalid version format: '{v}'. "
                "Use semantic versioning: 'MAJOR.MINOR.PATCH' (e.g., '1.0.0'), "
                "'MAJOR.MINOR' (e.g., '1.0'), or include pre-release/build metadata "
                "(e.g., '1.0.0-alpha', '1.0.0+build.1').",
            )

        return v

    @field_validator("app_name")
    @classmethod
    def validate_app_name(cls, v: str) -> str:
        """Validate the application name with format requirements.

        Args:
            v: The application name to validate.

        Returns:
            The validated application name.

        Raises:
            ValueError: If the application name is invalid with guidance.
        """
        if not v.strip():
            raise ValueError(
                "Application name cannot be empty. "
                "Use a descriptive name like 'PromptCraft-Hybrid', 'MyApp', or 'API-Service'.",
            )

        v = v.strip()

        # Check length constraints
        if len(v) > MAX_APP_NAME_LENGTH:
            raise ValueError(
                f"Application name too long ({len(v)} characters). "
                "Maximum length is 100 characters for logging and identification purposes.",
            )

        # Check for reasonable characters (allow spaces, hyphens, underscores)
        if not re.match(r"^[a-zA-Z0-9._\s-]+$", v):
            raise ValueError(
                f"Invalid application name: '{v}'. "
                "Name should contain only letters, numbers, spaces, hyphens, "
                "underscores, and periods.",
            )

        return v

    @field_validator("api_port")
    @classmethod
    def validate_api_port_detailed(cls, v: int) -> int:
        """Enhanced port validation with detailed error messages.

        Args:
            v: The port number to validate.

        Returns:
            The validated port number.

        Raises:
            ValueError: If the port is invalid with suggested alternatives.
        """
        if not (1 <= v <= MAX_PORT_NUMBER):
            raise ValueError(
                f"Port {v} is outside valid range. "
                "Ports must be between 1-65535. "
                "Common choices: 8000 (development), 80 (HTTP), 443 (HTTPS), "
                "3000 (Node.js), 5000 (Flask), 8080 (alternative HTTP).",
            )

        # Warn about privileged ports in development
        if v < PRIVILEGED_PORT_THRESHOLD:
            # This is just informational, not an error
            logger = logging.getLogger(__name__)
            logger.warning(
                "Using privileged port %d (< 1024). "
                "This may require root privileges. "
                "Consider using ports 8000+ for development.",
                v,
            )

        return v

    @field_validator("environment")
    @classmethod
    def validate_environment_requirements(cls, v: str) -> str:
        """Validate environment with specific requirements per environment.

        Args:
            v: The environment value to validate.

        Returns:
            The validated environment value.

        Raises:
            ValueError: If environment-specific requirements are not met.
        """
        if v not in ("dev", "staging", "prod"):
            raise ValueError(
                f"Invalid environment: '{v}'. "
                "Valid environments: 'dev' (development), 'staging' (pre-production), "
                "'prod' (production).",
            )

        # Note: Cross-field validation (like checking debug mode based on environment)
        # will be handled in the startup validation function instead of here
        # since Pydantic v2 changed how cross-field validation works

        return v

    @field_validator("database_host")
    @classmethod
    def validate_database_host(cls, v: str) -> str:
        """Validate database host address.

        Args:
            v: The database host to validate.

        Returns:
            The validated database host.

        Raises:
            ValueError: If the database host is invalid.
        """
        if not v.strip():
            raise ValueError(
                "Database host cannot be empty. Use 'localhost', '127.0.0.1', or a valid IP address/hostname.",
            )

        v = v.strip()

        # Use the same validation logic as api_host
        ip_pattern = re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
        )

        hostname_pattern = re.compile(
            r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)$",
        )

        # Allow common special cases
        if v in ("localhost", "127.0.0.1"):
            return v

        # Validate IP address or hostname format
        if ip_pattern.match(v) or (hostname_pattern.match(v) and len(v) <= MAX_HOSTNAME_LENGTH):
            return v

        raise ValueError(
            f"Invalid database host format: '{v}'. "
            "Host must be a valid IP address, hostname, 'localhost', or '127.0.0.1'.",
        )

    @field_validator("database_port")
    @classmethod
    def validate_database_port(cls, v: int) -> int:
        """Validate database port number.

        Args:
            v: The database port to validate.

        Returns:
            The validated database port.

        Raises:
            ValueError: If the database port is invalid.
        """
        if not (1 <= v <= MAX_PORT_NUMBER):
            raise ValueError(
                f"Database port {v} is outside valid range (1-65535). "
                "Common PostgreSQL ports: 5432 (default), 5433, 5434.",
            )
        return v

    @field_validator("database_name", "database_username")
    @classmethod
    def validate_database_identifier(cls, v: str) -> str:
        """Validate database name and username identifiers.

        Args:
            v: The database identifier to validate.

        Returns:
            The validated database identifier.

        Raises:
            ValueError: If the database identifier is invalid.
        """
        if not v.strip():
            raise ValueError(
                "Database identifier cannot be empty. Use alphanumeric characters, underscores, and hyphens only.",
            )

        v = v.strip()

        # PostgreSQL identifier rules: start with letter/underscore, contain letters/digits/underscores
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(
                f"Invalid database identifier: '{v}'. "
                "Must start with letter or underscore, contain only letters, digits, and underscores.",
            )

        if len(v) > 63:  # PostgreSQL identifier limit
            raise ValueError(
                f"Database identifier too long ({len(v)} characters). "
                "PostgreSQL identifiers must be 63 characters or less.",
            )

        return v

    @field_validator("database_timeout")
    @classmethod
    def validate_database_timeout(cls, v: float) -> float:
        """Validate database timeout value.

        Args:
            v: The database timeout to validate.

        Returns:
            The validated database timeout.

        Raises:
            ValueError: If the database timeout is invalid.
        """
        if v <= 0:
            raise ValueError(
                "Database timeout must be positive. Common values: 30.0 (default), 60.0 (extended), 10.0 (fast).",
            )

        if v > 300:  # 5 minutes
            raise ValueError(
                "Database timeout too high (> 5 minutes). "
                "Long timeouts can cause request queuing and poor user experience.",
            )

        return v

    @field_validator(*SECRET_FIELD_NAMES)
    @classmethod
    def validate_secret_not_empty(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate that secret values are not empty strings.

        Args:
            v: The secret value to validate.

        Returns:
            The validated secret value.

        Raises:
            ValueError: If the secret is an empty string.
        """
        if v is not None and not v.get_secret_value().strip():
            raise ValueError(
                "Secret values cannot be empty strings. "
                "Provide a valid secret value or leave unset for optional secrets.",
            )

        return v


# Global settings instance for singleton pattern
_settings: ApplicationSettings | None = None

# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def validate_encryption_available() -> bool:
    """Check if encryption is available and properly configured.

    Returns:
        True if encryption is available, False otherwise.
    """
    try:
        validate_environment_keys()
        return True
    except EncryptionError:
        return False


def _log_encryption_status(
    current_env: str,
    encryption_available: bool,
    logger: Any,
) -> None:
    """Log encryption status with appropriate levels based on environment.

    Args:
        current_env: Current environment (dev, staging, prod)
        encryption_available: Whether encryption is available
        logger: Logger instance to use
    """
    if current_env == "prod" and not encryption_available:
        logger.warning(
            "Production environment detected but encryption not available. "
            "Some features may be limited. Ensure GPG and SSH keys are properly configured.",
        )
    elif current_env == "dev" and not encryption_available:
        logger.info(
            "Development environment - encryption not required but recommended for testing production features.",
        )
    elif encryption_available:
        logger.info("Encryption system is properly configured and available.")


def _process_validation_errors(
    errors: list[Any],  # Pydantic ErrorDetails type
) -> tuple[list[str], list[str]]:
    """Process Pydantic validation errors into field errors and suggestions.

    Args:
        errors: List of Pydantic validation errors (ErrorDetails objects)

    Returns:
        Tuple of (field_errors, suggestions)
    """
    field_errors = []
    suggestions = []

    for error in errors:
        field_name = ".".join(str(loc) for loc in error["loc"])
        error_msg = error["msg"]
        field_errors.append(f"{field_name}: {error_msg}")

        # Add field-specific suggestions
        if field_name == "api_port":
            suggestions.append(
                "Use a port between 1-65535, common choices: 8000, 3000, 5000",
            )
        elif field_name == "environment":
            suggestions.append(
                "Set PROMPTCRAFT_ENVIRONMENT to 'dev', 'staging', or 'prod'",
            )
        elif field_name == "api_host":
            suggestions.append(
                "Use '0.0.0.0', 'localhost', '127.0.0.1', or a valid IP/hostname",
            )

    return field_errors, suggestions


def _mask_secret_value(value: str, show_chars: int = 4) -> str:
    """Mask a secret value for safe logging.

    Args:
        value: The secret value to mask
        show_chars: Number of characters to show at the end

    Returns:
        Masked value safe for logging
    """
    if len(value) <= show_chars:
        return "*" * len(value)
    return "*" * (len(value) - show_chars) + value[-show_chars:]


def _log_configuration_status(settings: ApplicationSettings) -> None:
    """Log configuration loading status without exposing sensitive data.

    Args:
        settings: The loaded settings instance to log
    """
    logger = logging.getLogger(__name__)

    # Log basic configuration
    logger.info("Configuration loaded for environment: %s", settings.environment)
    logger.info("Application: %s v%s", settings.app_name, settings.version)
    logger.info("API server: %s:%s", settings.api_host, settings.api_port)
    logger.info("Debug mode: %s", settings.debug)

    # Log secret field status (without values) using centralized field names
    secret_fields = {field_name: getattr(settings, field_name) for field_name in SECRET_FIELD_NAMES}

    configured_secrets = []
    missing_secrets = []

    for field_name, value in secret_fields.items():
        if value is not None and value.get_secret_value().strip():
            configured_secrets.append(field_name)
        else:
            missing_secrets.append(field_name)

    if configured_secrets:
        logger.info("Configured secrets: %s", ", ".join(configured_secrets))

    if missing_secrets:
        if settings.environment == "prod":
            logger.warning(
                "Missing secrets in production: %s",
                ", ".join(missing_secrets),
            )
        else:
            logger.debug(
                "Optional secrets not configured: %s",
                ", ".join(missing_secrets),
            )


def _validate_production_requirements(
    settings: ApplicationSettings,
) -> tuple[list[str], list[str]]:
    """Validate production-specific requirements.

    Returns:
        Tuple of (validation_errors, suggestions)
    """
    validation_errors = []
    suggestions = []

    if settings.debug:
        validation_errors.append("Debug mode should be disabled in production")
        suggestions.append("Set PROMPTCRAFT_DEBUG=false for production deployment")

    # Check for required secrets in production
    required_secrets_prod = ["secret_key", "jwt_secret_key"]
    for secret_name in required_secrets_prod:
        secret_value = getattr(settings, secret_name, None)
        if not secret_value or not secret_value.get_secret_value().strip():
            validation_errors.append(
                f"Required secret '{secret_name}' is missing in production",
            )
            suggestions.append(
                f"Set PROMPTCRAFT_{secret_name.upper()} environment variable",
            )

    # Validate production-appropriate host binding
    if settings.api_host in ("127.0.0.1", "localhost"):
        validation_errors.append(
            "Production API host should not be localhost/127.0.0.1",
        )
        suggestions.append(
            "Use '0.0.0.0' to bind to all interfaces or specify production host",
        )

    return validation_errors, suggestions


def _validate_staging_requirements(
    settings: ApplicationSettings,
) -> tuple[list[str], list[str]]:
    """Validate staging-specific requirements.

    Returns:
        Tuple of (validation_errors, suggestions)
    """
    validation_errors = []
    suggestions = []

    if not settings.secret_key or not settings.secret_key.get_secret_value().strip():
        validation_errors.append(
            "Secret key should be configured in staging environment",
        )
        suggestions.append("Set PROMPTCRAFT_SECRET_KEY for staging testing")

    return validation_errors, suggestions


def _validate_general_security(
    settings: ApplicationSettings,
) -> tuple[list[str], list[str]]:
    """Validate general security requirements.

    Returns:
        Tuple of (validation_errors, suggestions)
    """
    validation_errors = []
    suggestions = []

    # General security validation
    if settings.api_port in {HTTP_PORT, HTTPS_PORT} and settings.environment != "prod":
        validation_errors.append(
            f"Using standard web port {settings.api_port} in {settings.environment} environment",
        )
        suggestions.append(
            "Consider using development ports like 8000, 3000, or 5000",
        )

    # Validate host/port combination makes sense
    if (
        settings.api_host == "0.0.0.0"  # nosec B104 # Intentional bind to all interfaces  # noqa: S104
        and settings.environment == "dev"
        and settings.api_port < PRIVILEGED_PORT_THRESHOLD
    ):
        validation_errors.append(
            "Binding to all interfaces with privileged port in development",
        )
        suggestions.append("Use 'localhost' host or port >= 1024 for development")

    # Encryption availability check
    encryption_available = validate_encryption_available()
    if settings.environment == "prod" and not encryption_available:
        validation_errors.append("Encryption not available in production environment")
        suggestions.extend(
            [
                "Ensure GPG keys are properly configured",
                "Verify SSH keys are loaded for signed commits",
                "Run: poetry run python src/utils/encryption.py",
            ],
        )

    return validation_errors, suggestions


def validate_configuration_on_startup(settings: ApplicationSettings) -> None:
    """Perform comprehensive configuration validation on application startup.

    This function validates the entire configuration and provides detailed
    error reporting with actionable suggestions for fixing issues.

    Args:
        settings: The settings instance to validate

    Raises:
        ConfigurationValidationError: If validation fails with detailed errors
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting configuration validation...")

    validation_errors = []
    suggestions = []

    # Environment-specific validation
    if settings.environment == "prod":
        prod_errors, prod_suggestions = _validate_production_requirements(settings)
        validation_errors.extend(prod_errors)
        suggestions.extend(prod_suggestions)
    elif settings.environment == "staging":
        staging_errors, staging_suggestions = _validate_staging_requirements(settings)
        validation_errors.extend(staging_errors)
        suggestions.extend(staging_suggestions)

    # General security validation
    security_errors, security_suggestions = _validate_general_security(settings)
    validation_errors.extend(security_errors)
    suggestions.extend(security_suggestions)

    # Cross-field validation
    if settings.database_url and settings.database_password:
        logger.warning(
            "Both database_url and database_password are set. URL typically includes password.",
        )

    # Log validation results
    if validation_errors:
        logger.error(
            "Configuration validation failed with %d error(s)",
            len(validation_errors),
        )
        raise ConfigurationValidationError(
            f"Configuration validation failed for {settings.environment} environment",
            field_errors=validation_errors,
            suggestions=suggestions,
        )
    logger.info("Configuration validation completed successfully")

    # Log configuration status for operational awareness
    _log_configuration_status(settings)


def validate_field_requirements_by_environment(environment: str) -> set[str]:
    """Get required fields for a specific environment.

    Args:
        environment: The environment to get requirements for

    Returns:
        Set of field names that are required for the environment
    """
    base_required = {"app_name", "version", "environment", "api_host", "api_port"}

    if environment == "prod":
        return base_required | {"secret_key", "jwt_secret_key"}
    if environment == "staging":
        return base_required | {"secret_key"}
    # dev
    return base_required


def get_settings(validate_on_startup: bool = True) -> ApplicationSettings:
    """Get settings instance based on current environment with comprehensive validation.

    This factory function provides a singleton pattern for application settings.
    It detects the current environment and loads configuration with the proper
    precedence hierarchy:

    1. Environment variables (highest priority)
    2. .env.{environment}.gpg file (encrypted environment-specific)
    3. .env.{environment} file (plain environment-specific)
    4. .env.gpg file (encrypted base)
    5. .env file (plain base)
    6. Pydantic field defaults (lowest priority)

    The function ensures that settings are loaded only once per application
    lifecycle and provides graceful fallback when encryption is unavailable
    during development. It also performs comprehensive validation and logging.

    Args:
        validate_on_startup: Whether to perform full startup validation (default: True)

    Returns:
        ApplicationSettings instance configured for the current environment.

    Raises:
        ConfigurationValidationError: If configuration validation fails
        ValidationError: If Pydantic field validation fails

    Example:
        >>> settings = get_settings()
        >>> print(f"Running in {settings.environment} mode on {settings.api_host}:{settings.api_port}")
        >>> # Secret values are protected
        >>> if settings.database_password:
        ...     print("Database password is configured (value hidden)")
    """
    global _settings  # Singleton pattern requires global state  # noqa: PLW0603
    logger = logging.getLogger(__name__)

    if _settings is None:
        logger.info("Initializing application configuration...")

        # Check encryption availability
        encryption_available = validate_encryption_available()
        current_env = _detect_environment()

        logger.info("Detected environment: %s", current_env)
        logger.info("Encryption available: %s", encryption_available)

        # Log encryption status with appropriate levels
        _log_encryption_status(current_env, encryption_available, logger)

        try:
            # Load settings with enhanced error handling
            logger.debug("Loading configuration from environment and files...")
            _settings = ApplicationSettings()

            # Perform startup validation if requested
            if validate_on_startup:
                validate_configuration_on_startup(_settings)
            else:
                # Still log basic configuration info
                _log_configuration_status(_settings)

        except ValidationError as e:
            # Convert Pydantic validation errors to our enhanced format
            field_errors, suggestions = _process_validation_errors(e.errors())

            raise ConfigurationValidationError(
                f"Configuration field validation failed in {current_env} environment",
                field_errors=field_errors,
                suggestions=suggestions,
            ) from e

        except (OSError, ValueError, TypeError) as e:
            logger.error("Unexpected error during configuration loading: %s", e)
            raise ConfigurationValidationError(
                "Unexpected configuration loading error",
                field_errors=[str(e)],
                suggestions=[
                    "Check environment variables and .env files for syntax errors",
                ],
            ) from e

        logger.info("Configuration initialization completed successfully")

    return _settings


def reload_settings(validate_on_startup: bool = True) -> ApplicationSettings:
    """Reload settings from environment and files with validation.

    This function forces a reload of settings, useful for testing or when
    environment configuration has changed during runtime.

    Args:
        validate_on_startup: Whether to perform full startup validation (default: True)

    Returns:
        Fresh ApplicationSettings instance with current configuration.

    Raises:
        ConfigurationValidationError: If configuration validation fails
    """
    global _settings  # Singleton pattern requires global state  # noqa: PLW0603
    logger = logging.getLogger(__name__)
    logger.info("Reloading configuration...")
    _settings = None
    return get_settings(validate_on_startup=validate_on_startup)
