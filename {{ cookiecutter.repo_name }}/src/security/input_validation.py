"""Enhanced input validation and sanitization for FastAPI.

This module provides comprehensive input validation and sanitization
to prevent XSS, injection attacks, and other malicious input.
"""

import html
import re
import urllib.parse
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SecureStringField(str):
    """String field with automatic HTML escaping and validation.

    Custom string field that automatically sanitizes input to prevent XSS attacks
    and other malicious content. Provides multiple layers of validation:

    1. Length validation (10KB limit)
    2. Null byte detection and rejection
    3. HTML entity escaping for XSS prevention
    4. Suspicious pattern detection (script tags, JavaScript protocols)

    **Security Features:**
    - Automatic HTML escaping with quote=True
    - Null byte injection prevention
    - Malicious pattern detection (script tags, event handlers)
    - Configurable length limits

    Example:
        >>> field = SecureStringField.validate("<script>alert('xss')</script>")
        >>> print(field)  # "&lt;script&gt;alert('xss')&lt;/script&gt;"

    Note:
        This class extends str to maintain string compatibility while adding
        security validation.
    """

    @classmethod
    def __get_validators__(cls) -> Any:
        """Pydantic v1 compatibility for custom validation.

        Provides backward compatibility with Pydantic v1 validation system.
        Yields the validate method for use in Pydantic model validation.

        Returns:
            Generator yielding the validate method

        Note:
            This method maintains compatibility with older Pydantic versions
            while allowing the security validation to work seamlessly.
        """
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> str:
        """Validate and sanitize string input.

        Multi-layer security validation that sanitizes input to prevent various
        attack vectors including XSS, null byte injection, and malicious patterns.

        **Validation Steps:**
        1. Type coercion to string
        2. Length validation (10KB limit)
        3. Null byte detection
        4. HTML entity escaping
        5. Suspicious pattern detection

        **Protected Against:**
        - XSS attacks (HTML escaping)
        - Null byte injection
        - Script tag injection
        - JavaScript protocol injection
        - Event handler injection

        Args:
            value: Input value to validate (any type, converted to string)

        Returns:
            Sanitized string value with HTML entities escaped

        Raises:
            ValueError: If input contains dangerous content or exceeds limits

        Complexity:
            O(n) where n is the length of the input string

        Example:
            >>> SecureStringField.validate("<script>alert('xss')</script>")
            "&lt;script&gt;alert('xss')&lt;/script&gt;"
        """
        if not isinstance(value, str):
            value = str(value)

        # Basic length check to prevent DoS attacks
        max_input_length = 10000  # 10KB limit for reasonable input size
        if len(value) > max_input_length:
            raise ValueError(f"Input too long (maximum {max_input_length} characters)")

        # Check for null bytes (potential injection vector)
        if "\x00" in value:
            raise ValueError("Null bytes not allowed in input")

        # HTML escape to prevent XSS attacks
        # quote=True escapes both single and double quotes
        sanitized = html.escape(value, quote=True)

        # Additional checks for suspicious patterns
        # These patterns catch common XSS and injection attempts
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",  # Script tags (complete)
            r"javascript:",  # JavaScript protocol
            r"vbscript:",  # VBScript protocol
            r"onload\s*=",  # Event handlers
            r"onerror\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
        ]

        # Check each pattern against the original value (before escaping)
        for pattern in suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE | re.DOTALL):
                raise ValueError("Potentially dangerous content detected")

        return str(sanitized)


class SecurePathField(str):
    r"""Path field with directory traversal protection.

    Specialized string field for validating file paths with security measures
    to prevent directory traversal attacks and other path-based exploits.

    **Security Features:**
    - Directory traversal prevention ("../" sequences)
    - Absolute path rejection
    - Dangerous character filtering
    - URL decoding before validation

    **Protected Against:**
    - Directory traversal attacks (../../../etc/passwd)
    - Absolute path access (/etc/passwd, C:\Windows\System32)
    - Null byte injection
    - Command injection characters (|, &, ;, $, `)

    Example:
        >>> SecurePathField.validate("../../../etc/passwd")
        ValueError: Directory traversal not allowed

    Note:
        Validates the URL-decoded path but returns the original value
        to preserve encoding if needed.
    """

    @classmethod
    def __get_validators__(cls) -> Any:
        """Pydantic v1 compatibility for custom validation.

        Provides backward compatibility with Pydantic v1 validation system.
        Yields the validate method for use in Pydantic model validation.

        Returns:
            Generator yielding the validate method
        """
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> str:
        """Validate path input to prevent directory traversal.

        Multi-step validation process that prevents various path-based attacks
        by checking the URL-decoded path against dangerous patterns.

        **Validation Process:**
        1. URL decode the path to handle encoded traversal attempts
        2. Check for directory traversal sequences ("..")
        3. Reject absolute paths
        4. Filter dangerous characters
        5. Return original value if validation passes

        Args:
            value: Path value to validate (any type, converted to string)

        Returns:
            Original path value if validation passes

        Raises:
            ValueError: If path contains dangerous patterns or characters

        Example:
            >>> SecurePathField.validate("docs/file.txt")  # Valid
            "docs/file.txt"
            >>> SecurePathField.validate("../../../etc/passwd")  # Invalid
            ValueError: Directory traversal not allowed
        """
        if not isinstance(value, str):
            value = str(value)

        # 1. Decode the value first to get the intended path
        # This prevents encoded traversal attempts like %2e%2e%2f
        decoded_value = urllib.parse.unquote(value)

        # 2. Run all validations on the decoded path
        # Check for directory traversal sequences
        if ".." in decoded_value:
            raise ValueError("Directory traversal not allowed")

        # Check for absolute paths (both Unix and Windows styles)
        if decoded_value.startswith("/") or (len(decoded_value) > 1 and decoded_value[1] == ":"):
            raise ValueError("Absolute paths not allowed")

        # Check for suspicious characters in decoded path
        # These characters could be used for command injection
        dangerous_chars = ["\x00", "\r", "\n", "|", "&", ";", "$", "`"]
        for char in dangerous_chars:
            if char in decoded_value:
                raise ValueError(f"Dangerous character '{char}' not allowed")

        # 3. Return the original validated value, preserving encoding if needed
        # This allows the application to handle URL-encoded paths appropriately
        return str(value)


class SecureEmailField(str):
    """Email field with enhanced validation."""

    @classmethod
    def __get_validators__(cls) -> Any:
        """Pydantic v1 compatibility for custom validation."""
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> str:
        """Validate email input.

        Args:
            value: Email value to validate

        Returns:
            Validated email value

        Raises:
            ValueError: If email format is invalid
        """
        if not isinstance(value, str):
            value = str(value)

        # Basic email validation
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, value):
            raise ValueError("Invalid email format")

        # Length checks
        max_email_length = 320  # RFC 5321 limit
        max_local_length = 64  # RFC 5321 limit
        if len(value) > max_email_length:
            raise ValueError("Email too long")

        local, domain = value.rsplit("@", 1)
        if len(local) > max_local_length:
            raise ValueError("Email local part too long")

        return str(value).lower()  # Normalize to lowercase


class BaseSecureModel(BaseModel):
    """Base model with security enhancements."""

    class Config:
        """Pydantic configuration for security."""

        # Prevent unknown fields
        extra = "forbid"
        # Validate assignments
        validate_assignment = True
        # Use enum values
        use_enum_values = True
        # Strict validation
        str_strip_whitespace = True
        anystr_strip_whitespace = True


class SecureTextInput(BaseSecureModel):
    """Secure text input validation model."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text content with XSS protection",
    )

    @field_validator("text")
    @classmethod
    def validate_text_content(cls, value: str) -> str:
        """Validate and sanitize text content.

        Args:
            value: Text value to validate

        Returns:
            Sanitized text value
        """
        return SecureStringField.validate(value)


class SecureFileUpload(BaseSecureModel):
    """Secure file upload validation model."""

    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Filename with path traversal protection",
    )
    content_type: str = Field(
        ...,
        description="MIME content type",
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Validate filename for security.

        Args:
            value: Filename to validate

        Returns:
            Validated filename
        """
        # Use path validation
        validated = SecurePathField.validate(value)

        # Additional filename checks
        if not re.match(r"^[a-zA-Z0-9._-]+$", validated):
            raise ValueError("Filename contains invalid characters")

        # Check for executable extensions
        dangerous_extensions = [
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".scr",
            ".js",
            ".vbs",
            ".php",
            ".asp",
            ".jsp",
            ".sh",
            ".py",
            ".pl",
        ]

        for ext in dangerous_extensions:
            if validated.lower().endswith(ext):
                raise ValueError(f"File type '{ext}' not allowed")

        return validated

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str) -> str:
        """Validate MIME content type.

        Args:
            value: Content type to validate

        Returns:
            Validated content type
        """
        # Basic MIME type validation
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^]*\/[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^]*$", value):
            raise ValueError("Invalid content type format")

        # Whitelist safe content types
        safe_types = [
            "text/plain",
            "text/csv",
            "text/markdown",
            "application/json",
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]

        if value not in safe_types:
            raise ValueError(f"Content type '{value}' not allowed")

        return value


class SecureQueryParams(BaseSecureModel):
    """Secure query parameter validation."""

    search: str | None = Field(
        None,
        max_length=1000,
        description="Search query with XSS protection",
    )
    page: int = Field(
        1,
        ge=1,
        le=10000,
        description="Page number",
    )
    limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Items per page",
    )
    sort: str | None = Field(
        None,
        max_length=50,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(:asc|:desc)?$",
        description="Sort field and direction",
    )

    @field_validator("search")
    @classmethod
    def validate_search(cls, value: str | None) -> str | None:
        """Validate search query.

        Args:
            value: Search query to validate

        Returns:
            Sanitized search query
        """
        if value is None:
            return None

        return SecureStringField.validate(value)


def create_input_sanitizer() -> dict[str, Any]:
    """Create input sanitization configuration.

    Returns:
        Dictionary of sanitization functions
    """
    return {
        "string": SecureStringField.validate,
        "path": SecurePathField.validate,
        "email": SecureEmailField.validate,
    }


def sanitize_dict_values(data: dict[str, Any], sanitizer_type: str = "string") -> dict[str, Any]:
    """Sanitize all string values in a dictionary.

    Args:
        data: Dictionary to sanitize
        sanitizer_type: Type of sanitization to apply

    Returns:
        Dictionary with sanitized values
    """
    sanitizers = create_input_sanitizer()
    sanitizer = sanitizers.get(sanitizer_type, sanitizers["string"])

    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = sanitizer(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict_values(value, sanitizer_type)
        elif isinstance(value, list):
            sanitized[key] = [sanitizer(item) if isinstance(item, str) else item for item in value]
        else:
            sanitized[key] = value

    return sanitized
