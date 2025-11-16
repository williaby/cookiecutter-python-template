"""
Time utilities for PromptCraft-Hybrid.

This module provides standardized time utilities that work consistently across
Python versions and avoid timezone import compatibility issues.

Key design principles:
- Use timezone.utc for maximum compatibility (Python 3.2+)
- Provide consistent API for common datetime operations
- Eliminate UTC import issues between Python 3.11 and 3.12
- Work seamlessly with linting tools (ruff, black, mypy)
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Get current UTC datetime - standardized across project.

    Returns:
        datetime: Current UTC datetime with timezone info

    Example:
        >>> now = utc_now()
        >>> now.tzinfo == timezone.utc
        True
    """
    return datetime.now(tz=timezone.utc)


def utc_timestamp() -> str:
    """Get current UTC timestamp as ISO string.

    Returns:
        str: Current UTC timestamp in ISO format

    Example:
        >>> timestamp = utc_timestamp()
        >>> timestamp.endswith('+00:00')
        True
    """
    return utc_now().isoformat()


def to_utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    """Create UTC datetime from components.

    Args:
        year: Year component
        month: Month component (1-12)
        day: Day component (1-31)
        hour: Hour component (0-23), defaults to 0
        minute: Minute component (0-59), defaults to 0
        second: Second component (0-59), defaults to 0
        microsecond: Microsecond component (0-999999), defaults to 0

    Returns:
        datetime: UTC datetime with timezone info

    Example:
        >>> dt = to_utc_datetime(2024, 1, 1, 12, 0, 0)
        >>> dt.tzinfo == timezone.utc
        True
    """
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)


def from_timestamp(timestamp: float) -> datetime:
    """Create UTC datetime from Unix timestamp.

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        datetime: UTC datetime with timezone info

    Example:
        >>> dt = from_timestamp(1640995200)  # 2022-01-01 00:00:00 UTC
        >>> dt.year == 2022
        True
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp.

    Args:
        dt: Datetime object (timezone-aware or naive)

    Returns:
        float: Unix timestamp (seconds since epoch)

    Note:
        If datetime is naive, it's assumed to be UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """Format datetime for display.

    Args:
        dt: Datetime object
        fmt: Format string, defaults to "%Y-%m-%d %H:%M:%S UTC"

    Returns:
        str: Formatted datetime string

    Example:
        >>> dt = to_utc_datetime(2024, 1, 1, 12, 0, 0)
        >>> format_datetime(dt)
        '2024-01-01 12:00:00 UTC'
    """
    # Convert to UTC if needed
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime(fmt)


def parse_iso_datetime(iso_string: str) -> datetime:
    """Parse ISO format datetime string to UTC datetime.

    Args:
        iso_string: ISO format datetime string

    Returns:
        datetime: UTC datetime with timezone info

    Example:
        >>> dt = parse_iso_datetime("2024-01-01T12:00:00+00:00")
        >>> dt.tzinfo == timezone.utc
        True
    """
    dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


# Commonly used timezone constant
UTC = timezone.utc  # Safe alias for compatibility


# Export public API
__all__ = [
    "UTC",
    "format_datetime",
    "from_timestamp",
    "parse_iso_datetime",
    "to_timestamp",
    "to_utc_datetime",
    "utc_now",
    "utc_timestamp",
]
