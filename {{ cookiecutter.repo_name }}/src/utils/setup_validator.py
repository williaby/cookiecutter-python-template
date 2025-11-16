"""
Startup validation utilities for PromptCraft-Hybrid.

This module provides comprehensive validation functions to ensure the application
is properly configured and ready to run in any environment.
"""

import logging
import os
import sys

try:
    import gnupg
except ImportError:
    gnupg = None

try:
    import pydantic
except ImportError:
    pydantic = None  # type: ignore[assignment]

from src.config.settings import (
    ApplicationSettings,
    ConfigurationValidationError,
    get_settings,
    validate_configuration_on_startup,
)

from .encryption import EncryptionError, validate_environment_keys

logger = logging.getLogger(__name__)


def validate_system_requirements() -> tuple[bool, list[str]]:
    """Validate system-level requirements for the application.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Python version check
    if sys.version_info < (3, 11):  # noqa: UP036
        errors.append(f"Python 3.11+ required, but found Python {sys.version_info[0]}.{sys.version_info[1]}")

    if gnupg is None:
        errors.append("python-gnupg package not available - required for encryption")

    return len(errors) == 0, errors


def validate_environment_setup() -> tuple[bool, list[str], list[str]]:
    """Validate environment setup including keys and secrets.

    Returns:
        Tuple of (success: bool, errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []

    # Check encryption setup
    try:
        validate_environment_keys()
        logger.info("✓ Encryption keys validated successfully")
    except EncryptionError as e:
        errors.append(f"Encryption setup issue: {e}")
        warnings.append("Some features requiring encryption may not be available")

    # Check if we're running with appropriate privileges
    try:
        if os.getuid() == 0:  # Running as root
            warnings.append(
                "Running as root user - consider using a non-privileged user",
            )
    except AttributeError:
        # Windows doesn't have getuid
        pass

    return len(errors) == 0, errors, warnings


def validate_startup_configuration(
    settings: ApplicationSettings | None = None,
) -> bool:
    """Perform complete startup validation with detailed reporting.

    Args:
        settings: Optional settings instance (will load if not provided)

    Returns:
        True if validation passes, False otherwise (errors are logged)
    """
    logger.info("Starting comprehensive application validation...")

    validation_success = True

    # 1. System requirements
    logger.info("Validating system requirements...")
    sys_ok, sys_errors = validate_system_requirements()
    if not sys_ok:
        logger.error("System requirements validation failed:")
        for error in sys_errors:
            logger.error("  ✗ %s", error)
        validation_success = False
    else:
        logger.info("✓ System requirements validation passed")

    # 2. Environment setup
    logger.info("Validating environment setup...")
    env_ok, env_errors, env_warnings = validate_environment_setup()
    if not env_ok:
        logger.error("Environment setup validation failed:")
        for error in env_errors:
            logger.error("  ✗ %s", error)
        validation_success = False
    else:
        logger.info("✓ Environment setup validation passed")

    # Log warnings
    for warning in env_warnings:
        logger.warning("  ⚠ %s", warning)

    # 3. Configuration validation
    logger.info("Validating application configuration...")
    try:
        if settings is None:
            # Load settings with validation
            settings = get_settings(validate_on_startup=True)
        else:
            # Validate provided settings
            validate_configuration_on_startup(settings)

        logger.info("✓ Configuration validation passed")

    except ConfigurationValidationError as e:
        logger.error("Configuration validation failed:")
        logger.error("  ✗ %s", e)
        validation_success = False

    except RuntimeError as e:
        logger.error("Unexpected error during configuration validation: %s", e)
        validation_success = False

    # Final status
    if validation_success:
        logger.info("🎉 All startup validation checks passed!")
        env_name = settings.environment if settings else "unknown"
        logger.info("Application ready to run in %s environment", env_name)
    else:
        logger.error("❌ Startup validation failed - please fix the above issues")

    return validation_success


def run_startup_checks() -> None:
    """Run all startup checks and exit if validation fails.

    This function is designed to be called during application startup
    to ensure everything is properly configured before continuing.
    """
    if not validate_startup_configuration():
        logger.error("Startup validation failed - exiting")
        sys.exit(1)


if __name__ == "__main__":
    # Command-line validation tool
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Running PromptCraft-Hybrid startup validation...")
    success = validate_startup_configuration()
    sys.exit(0 if success else 1)
