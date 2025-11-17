"""Configuration settings for {{ cookiecutter.project_name }}.

Settings can be overridden via environment variables with prefix {{ cookiecutter.project_slug|upper }}_
"""

import os
from typing import Literal

from {{ cookiecutter.project_slug }}.utils.logging import get_logger

logger = get_logger(__name__)


class Settings:
    """Configuration settings for {{ cookiecutter.project_name }}.

    Settings can be overridden via environment variables with prefix
    {{ cookiecutter.project_slug|upper }}_.

    Environment variables take precedence over defaults but are overridden
    by keyword arguments passed to __init__.

    Example:
        >>> settings = Settings()
        >>> settings.log_level
        'INFO'

        With environment variable:
        >>> os.environ["{{ cookiecutter.project_slug|upper }}_LOG_LEVEL"] = "DEBUG"
        >>> settings = Settings()
        >>> settings.log_level
        'DEBUG'
    """

    def __init__(
        self,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
        json_logs: bool | None = None,
        include_timestamp: bool | None = None,
    ) -> None:
        """Initialize settings from environment variables or keyword arguments.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                Defaults to INFO. Can be overridden by {{ cookiecutter.project_slug|upper }}_LOG_LEVEL
                environment variable.
            json_logs: If True, output JSON logs (production). If False, use rich console.
                Defaults to False. Can be overridden by {{ cookiecutter.project_slug|upper }}_JSON_LOGS
                environment variable.
            include_timestamp: Whether to include timestamps in logs.
                Defaults to True. Can be overridden by {{ cookiecutter.project_slug|upper }}_INCLUDE_TIMESTAMP
                environment variable.
        """
        # Logging configuration
        self.log_level: str = (
            log_level
            if log_level is not None
            else self._get_str_env(
                "{{ cookiecutter.project_slug|upper }}_LOG_LEVEL", "INFO"
            )
        )

        self.json_logs: bool = (
            json_logs
            if json_logs is not None
            else self._get_bool_env(
                "{{ cookiecutter.project_slug|upper }}_JSON_LOGS", default=False
            )
        )

        self.include_timestamp: bool = (
            include_timestamp
            if include_timestamp is not None
            else self._get_bool_env(
                "{{ cookiecutter.project_slug|upper }}_INCLUDE_TIMESTAMP", default=True
            )
        )

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable.

        Args:
            key: Environment variable name.
            default: Default value if environment variable is not set.

        Returns:
            Boolean value from environment or default. Interprets "true", "1",
            "yes", and "on" (case-insensitive) as True.
        """
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer value from environment variable.

        Args:
            key: Environment variable name.
            default: Default value if environment variable is not set or invalid.

        Returns:
            Integer value from environment or default.
        """
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}: {value}. Using default: {default}"
            )
            return default

    def _get_str_env(self, key: str, default: str) -> str:
        """Get string value from environment variable.

        Args:
            key: Environment variable name.
            default: Default value if environment variable is not set.

        Returns:
            String value from environment or default.
        """
        return os.getenv(key, default)
