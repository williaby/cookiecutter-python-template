"""
Utility modules for PromptCraft-Hybrid.

This package provides essential utility modules that support the core functionality
of the PromptCraft system. These utilities are designed to be reusable across
different components of the application.

Modules:
    encryption: GPG-based encryption utilities for secure configuration management
    resilience: Circuit breaker, retry, and other error handling patterns
    secure_random: Cryptographically secure random number generation
    logging_mixin: Standardized logging mixins for consistent logging across components
    setup_validator: Startup validation utilities to ensure proper configuration

Architecture:
    The utilities are organized into focused modules that provide specific capabilities:

    - Security utilities (encryption, secure_random) provide cryptographic functions
    - Resilience utilities provide fault tolerance and error handling patterns
    - Logging utilities provide standardized logging across all components
    - Validation utilities ensure system readiness and proper configuration

Usage:
    Import specific utilities as needed:

    >>> from src.utils.encryption import validate_environment_keys
    >>> from src.utils.resilience import CompositeResilienceHandler
    >>> from src.utils.secure_random import secure_jitter
    >>> from src.utils.logging_mixin import LoggerMixin
    >>> from src.utils.setup_validator import validate_startup_configuration

Dependencies:
    These utilities have minimal dependencies and are designed to be lightweight
    and reusable. They form the foundation for higher-level components in the
    PromptCraft system.

Time Complexity: N/A (package initialization)
Space Complexity: O(1) - minimal memory for package setup
"""
