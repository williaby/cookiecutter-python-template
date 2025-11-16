"""
Cross-version datetime compatibility layer for Python 3.10-3.13+

Handles UTC constant, deprecated methods, and timezone operations to ensure
consistent behavior across Python versions.

Phase 1: Emergency fix for Python 3.10 compatibility
Phase 2: Full compatibility layer (expanded incrementally)
"""
import sys
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional, Union
from functools import lru_cache

# Version detection
PY_VERSION = sys.version_info[:2]
PY_311_PLUS = PY_VERSION >= (3, 11)
PY_312_PLUS = PY_VERSION >= (3, 12)

# UTC constant compatibility - works across all Python 3.10+
if PY_311_PLUS:
    from datetime import UTC
else:
    UTC = timezone.utc

# Export the UTC constant for consistent imports
__all__ = [
    'UTC', 
    'utc_now', 
    'utc_from_timestamp', 
    'ensure_aware', 
    'is_aware', 
    'is_naive',
    'aware_to_naive',
    'naive_to_aware',
    'safe_compare'
]


def utc_now() -> datetime:
    """
    Get current UTC time as timezone-aware datetime.
    Replaces deprecated datetime.utcnow() with proper timezone handling.
    
    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(UTC)


def utc_from_timestamp(timestamp: float) -> datetime:
    """
    Create UTC datetime from timestamp.
    Replaces deprecated datetime.utcfromtimestamp().
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.fromtimestamp(timestamp, UTC)


def is_aware(dt: datetime) -> bool:
    """
    Check if datetime is timezone-aware.
    
    Args:
        dt: Datetime object to check
        
    Returns:
        True if datetime has timezone information
    """
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def is_naive(dt: datetime) -> bool:
    """
    Check if datetime is naive (no timezone).
    
    Args:
        dt: Datetime object to check
        
    Returns:
        True if datetime has no timezone information
    """
    return not is_aware(dt)


def ensure_aware(dt: datetime, tz: Optional[timezone] = None) -> datetime:
    """
    Ensure datetime is timezone-aware.
    If naive, assumes UTC unless another timezone specified.
    
    Args:
        dt: Datetime object
        tz: Timezone to apply if datetime is naive (defaults to UTC)
        
    Returns:
        Timezone-aware datetime
    """
    if is_aware(dt):
        return dt
    return dt.replace(tzinfo=tz or UTC)


def naive_to_aware(dt: datetime, tz: Optional[timezone] = None) -> datetime:
    """
    Convert naive datetime to aware.
    
    Args:
        dt: Naive datetime object
        tz: Target timezone (defaults to UTC)
        
    Returns:
        Timezone-aware datetime
        
    Raises:
        ValueError: If datetime is already timezone-aware
    """
    if is_aware(dt):
        raise ValueError("datetime is already timezone-aware")
    
    target_tz = tz or UTC
    return dt.replace(tzinfo=target_tz)


def aware_to_naive(dt: datetime, preserve_utc: bool = True) -> datetime:
    """
    Convert aware datetime to naive.
    
    Args:
        dt: Timezone-aware datetime
        preserve_utc: If True, converts to UTC first (recommended for storage)
        
    Returns:
        Naive datetime
        
    Raises:
        ValueError: If datetime is already naive
    """
    if is_naive(dt):
        raise ValueError("datetime is already naive")
    
    if preserve_utc:
        # Convert to UTC then strip timezone - safe for database storage
        utc_dt = dt.astimezone(UTC)
        return utc_dt.replace(tzinfo=None)
    else:
        # Just strip timezone (keeps local time) - use with caution
        return dt.replace(tzinfo=None)


@lru_cache(maxsize=128)
def _normalize_for_comparison(dt: datetime) -> datetime:
    """
    Normalize datetime for comparison (cached for performance).
    
    Args:
        dt: Datetime to normalize
        
    Returns:
        Timezone-aware datetime in UTC
    """
    if is_aware(dt):
        return dt.astimezone(UTC)
    return ensure_aware(dt, UTC)


def safe_compare(dt1: datetime, dt2: datetime) -> int:
    """
    Safely compare datetimes regardless of timezone awareness.
    
    Args:
        dt1: First datetime
        dt2: Second datetime
        
    Returns:
        -1 if dt1 < dt2, 0 if equal, 1 if dt1 > dt2
    """
    norm1 = _normalize_for_comparison(dt1)
    norm2 = _normalize_for_comparison(dt2)
    
    if norm1 < norm2:
        return -1
    elif norm1 > norm2:
        return 1
    return 0


# Migration helpers - temporary functions to catch issues during migration
def assert_datetime_aware(dt: datetime, context: str = "") -> datetime:
    """
    Temporary helper to catch naive datetime bugs during migration.
    
    Args:
        dt: Datetime to check
        context: Context string for debugging
        
    Returns:
        The same datetime object
        
    Warns:
        If datetime is naive (during migration period)
    """
    if is_naive(dt):
        warnings.warn(
            f"Naive datetime detected in {context}. "
            f"Consider using ensure_aware() or utc_now()",
            DeprecationWarning,
            stacklevel=2
        )
    return dt


# Legacy compatibility aliases (can be removed after migration)
def utcnow_compat() -> datetime:
    """Legacy compatibility for datetime.utcnow() - use utc_now() instead."""
    warnings.warn(
        "utcnow_compat() is deprecated, use utc_now() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return utc_now()


def utcfromtimestamp_compat(timestamp: float) -> datetime:
    """Legacy compatibility for datetime.utcfromtimestamp() - use utc_from_timestamp() instead."""
    warnings.warn(
        "utcfromtimestamp_compat() is deprecated, use utc_from_timestamp() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return utc_from_timestamp(timestamp)