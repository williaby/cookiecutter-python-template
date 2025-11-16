"""Centralized configuration constants to eliminate code duplication.

This module defines shared constants used across the configuration system
to maintain consistency and reduce maintenance burden.
"""

# Secret field names used throughout the application
# These must match the field names in ApplicationSettings
SECRET_FIELD_NAMES = [
    "database_password",
    "database_url",
    "db_password",
    "api_key",
    "secret_key",
    "azure_openai_api_key",
    "jwt_secret_key",
    "qdrant_api_key",
    "encryption_key",
    "mcp_api_key",
    "openrouter_api_key",
]

# Database configuration defaults
DATABASE_DEFAULT_HOST = "192.168.1.16"
DATABASE_DEFAULT_PORT = 5432
DATABASE_DEFAULT_NAME = "promptcraft"
DATABASE_DEFAULT_USERNAME = "promptcraft"
DATABASE_DEFAULT_TIMEOUT = 30.0

# Qdrant configuration defaults
QDRANT_DEFAULT_HOST = "192.168.1.16"
QDRANT_DEFAULT_PORT = 6333
QDRANT_DEFAULT_TIMEOUT = 30.0
QDRANT_DEFAULT_COLLECTION = "default"

# Vector store configuration
VECTOR_STORE_AUTO_DETECT = "auto"
VECTOR_STORE_QDRANT = "qdrant"
VECTOR_STORE_MOCK = "mock"
DEFAULT_VECTOR_DIMENSIONS = 384

# Sensitive error patterns for validation error sanitization
# Each tuple contains (pattern, replacement_message)
SENSITIVE_ERROR_PATTERNS = [
    (r"password", "Password configuration issue (details hidden)"),
    (r"api[\s_]key", "API key configuration issue (details hidden)"),
    (r"secret.*key", "Secret key configuration issue (details hidden)"),
    (r"key.*secret", "Secret key configuration issue (details hidden)"),
    (r"jwt.*secret", "JWT secret configuration issue (details hidden)"),
]

# File path patterns for sanitization
FILE_PATH_PATTERNS = [
    r"/home/",
    r"C:\\",
    r"/Users/",
]

# Health check response limits
HEALTH_CHECK_ERROR_LIMIT = 5
HEALTH_CHECK_SUGGESTION_LIMIT = 3

# CORS origins configuration by environment
CORS_ORIGINS_BY_ENVIRONMENT = {
    "dev": ["http://localhost:3000", "http://localhost:5173", "http://localhost:7860"],
    "staging": ["https://staging.promptcraft.io"],
    "prod": ["https://promptcraft.io"],
}
