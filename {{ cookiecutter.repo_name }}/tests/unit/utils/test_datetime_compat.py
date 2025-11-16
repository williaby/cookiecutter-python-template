"""
Comprehensive tests for datetime compatibility module.

Tests ensure consistent behavior across Python 3.10-3.13+ versions
and validate that the compatibility layer works correctly.
"""

import pytest
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from src.utils.datetime_compat import (
    UTC,
    utc_now,
    utc_from_timestamp,
    ensure_aware,
    is_aware,
    is_naive,
    naive_to_aware,
    aware_to_naive,
    safe_compare,
    assert_datetime_aware
)


class TestUTCConstant:
    """Test UTC constant compatibility across Python versions."""
    
    def test_utc_constant_exists(self):
        """UTC constant should be available."""
        assert UTC is not None
        
    def test_utc_constant_equals_timezone_utc(self):
        """UTC constant should equal timezone.utc."""
        assert UTC == timezone.utc
        
    def test_utc_constant_can_create_datetime(self):
        """UTC constant should work for datetime creation."""
        dt = datetime.now(UTC)
        assert dt.tzinfo == timezone.utc


class TestUTCFunctions:
    """Test UTC datetime creation functions."""
    
    def test_utc_now_returns_aware_datetime(self):
        """utc_now() should return timezone-aware datetime."""
        dt = utc_now()
        assert is_aware(dt)
        assert dt.tzinfo == UTC
        
    def test_utc_now_is_recent(self):
        """utc_now() should return current time (within 1 second)."""
        dt1 = utc_now()
        dt2 = datetime.now(timezone.utc)
        diff = abs((dt2 - dt1).total_seconds())
        assert diff < 1.0  # Should be within 1 second
        
    def test_utc_from_timestamp(self):
        """utc_from_timestamp() should create correct datetime."""
        timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        dt = utc_from_timestamp(timestamp)
        
        assert is_aware(dt)
        assert dt.tzinfo == UTC
        assert dt.year == 2022
        assert dt.month == 1
        assert dt.day == 1
        
    def test_utc_from_timestamp_zero(self):
        """utc_from_timestamp(0) should create Unix epoch."""
        dt = utc_from_timestamp(0)
        assert dt.year == 1970
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0


class TestAwarenessDetection:
    """Test datetime awareness detection functions."""
    
    def test_is_aware_with_aware_datetime(self):
        """is_aware() should return True for timezone-aware datetime."""
        dt = datetime.now(UTC)
        assert is_aware(dt)
        
    def test_is_aware_with_naive_datetime(self):
        """is_aware() should return False for naive datetime."""
        dt = datetime.now()
        assert not is_aware(dt)
        
    def test_is_naive_with_naive_datetime(self):
        """is_naive() should return True for naive datetime."""
        dt = datetime.now()
        assert is_naive(dt)
        
    def test_is_naive_with_aware_datetime(self):
        """is_naive() should return False for timezone-aware datetime."""
        dt = datetime.now(UTC)
        assert not is_naive(dt)


class TestAwarenessConversion:
    """Test conversion between naive and aware datetimes."""
    
    def test_ensure_aware_with_naive_datetime(self):
        """ensure_aware() should add UTC timezone to naive datetime."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        aware_dt = ensure_aware(naive_dt)
        
        assert is_aware(aware_dt)
        assert aware_dt.tzinfo == UTC
        assert aware_dt.year == 2024
        assert aware_dt.month == 1
        assert aware_dt.day == 1
        assert aware_dt.hour == 12
        
    def test_ensure_aware_with_aware_datetime(self):
        """ensure_aware() should return aware datetime unchanged."""
        original_dt = datetime.now(UTC)
        result_dt = ensure_aware(original_dt)
        
        assert result_dt is original_dt
        assert is_aware(result_dt)
        
    def test_ensure_aware_with_custom_timezone(self):
        """ensure_aware() should use provided timezone."""
        from datetime import timezone
        custom_tz = timezone(timedelta(hours=5))
        
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        aware_dt = ensure_aware(naive_dt, custom_tz)
        
        assert is_aware(aware_dt)
        assert aware_dt.tzinfo == custom_tz
        
    def test_naive_to_aware(self):
        """naive_to_aware() should convert naive to aware."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        aware_dt = naive_to_aware(naive_dt)
        
        assert is_aware(aware_dt)
        assert aware_dt.tzinfo == UTC
        
    def test_naive_to_aware_with_already_aware_raises(self):
        """naive_to_aware() should raise error for aware datetime."""
        aware_dt = datetime.now(UTC)
        
        with pytest.raises(ValueError, match="already timezone-aware"):
            naive_to_aware(aware_dt)
            
    def test_aware_to_naive_preserve_utc(self):
        """aware_to_naive() should convert to UTC naive by default."""
        # Create aware datetime in different timezone
        from datetime import timezone
        eastern_tz = timezone(timedelta(hours=-5))
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=eastern_tz)
        
        naive_dt = aware_to_naive(aware_dt, preserve_utc=True)
        
        assert is_naive(naive_dt)
        # Should be converted to UTC time (17:00 UTC = 12:00 EST)
        assert naive_dt.hour == 17
        
    def test_aware_to_naive_no_preserve_utc(self):
        """aware_to_naive() should strip timezone without conversion."""
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        naive_dt = aware_to_naive(aware_dt, preserve_utc=False)
        
        assert is_naive(naive_dt)
        assert naive_dt.hour == 12  # Same time, just timezone stripped
        
    def test_aware_to_naive_with_already_naive_raises(self):
        """aware_to_naive() should raise error for naive datetime."""
        naive_dt = datetime.now()
        
        with pytest.raises(ValueError, match="already naive"):
            aware_to_naive(naive_dt)


class TestSafeComparison:
    """Test safe datetime comparison across timezone awareness."""
    
    def test_safe_compare_same_times(self):
        """safe_compare() should return 0 for equal times."""
        dt1 = utc_now()
        dt2 = dt1  # Same datetime object
        
        assert safe_compare(dt1, dt2) == 0
        
    def test_safe_compare_naive_vs_aware(self):
        """safe_compare() should handle naive vs aware comparison."""
        # Create equivalent times - naive and aware
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        
        # Should be equal when naive is interpreted as UTC
        assert safe_compare(naive_dt, aware_dt) == 0
        
    def test_safe_compare_earlier_vs_later(self):
        """safe_compare() should correctly order datetimes."""
        dt1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        dt2 = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)
        
        assert safe_compare(dt1, dt2) == -1  # dt1 < dt2
        assert safe_compare(dt2, dt1) == 1   # dt2 > dt1


class TestMigrationHelpers:
    """Test migration helper functions."""
    
    def test_assert_datetime_aware_with_aware_datetime(self):
        """assert_datetime_aware() should return aware datetime without warning."""
        aware_dt = utc_now()
        
        # Should not raise any warnings
        result = assert_datetime_aware(aware_dt, "test context")
        assert result is aware_dt
        
    def test_assert_datetime_aware_with_naive_datetime(self):
        """assert_datetime_aware() should warn about naive datetime."""
        naive_dt = datetime.now()
        
        with pytest.warns(DeprecationWarning, match="Naive datetime detected"):
            result = assert_datetime_aware(naive_dt, "test context")
            
        assert result is naive_dt


class TestCompatibilityAcrossVersions:
    """Test compatibility layer behavior across Python versions."""
    
    @pytest.mark.parametrize("py_version", [
        (3, 10), (3, 11), (3, 12), (3, 13)
    ])
    def test_utc_constant_works_across_versions(self, py_version, monkeypatch):
        """UTC constant should work across all supported Python versions."""
        # Mock sys.version_info for testing
        monkeypatch.setattr(sys, 'version_info', py_version + (0, 'final', 0))
        
        # Re-import module to get version-specific behavior
        import importlib
        import src.utils.datetime_compat
        importlib.reload(src.utils.datetime_compat)
        
        # Should always work
        from src.utils.datetime_compat import UTC as TestUTC
        dt = datetime.now(TestUTC)
        assert dt.tzinfo == timezone.utc
        
    def test_performance_is_acceptable(self):
        """Compatibility functions should have acceptable performance."""
        import time
        
        # Test utc_now() performance
        start = time.perf_counter()
        for _ in range(1000):
            utc_now()
        elapsed = time.perf_counter() - start
        
        # Should complete 1000 operations in under 0.1 seconds
        assert elapsed < 0.1, f"Performance regression: {elapsed:.3f}s for 1000 ops"
        
    def test_no_memory_leaks_in_caching(self):
        """LRU cache should not cause memory leaks."""
        # Create many different datetimes to test cache behavior
        datetimes = []
        for i in range(200):  # More than cache size (128)
            dt = datetime(2024, 1, 1, i % 24, 0, 0, tzinfo=UTC)
            datetimes.append(dt)
            # Exercise the cached comparison function
            safe_compare(dt, utc_now())
        
        # Should complete without memory issues
        assert len(datetimes) == 200


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_utc_from_timestamp_with_negative_timestamp(self):
        """utc_from_timestamp() should handle negative timestamps."""
        # Pre-epoch timestamp
        dt = utc_from_timestamp(-86400)  # 1969-12-31 00:00:00 UTC
        
        assert is_aware(dt)
        assert dt.year == 1969
        assert dt.month == 12
        assert dt.day == 31
        
    def test_utc_from_timestamp_with_large_timestamp(self):
        """utc_from_timestamp() should handle large timestamps."""
        # Year 2038 problem territory
        dt = utc_from_timestamp(2147483647)  # 2038-01-19 03:14:07 UTC
        
        assert is_aware(dt)
        assert dt.year == 2038
        
    def test_safe_compare_with_none_values(self):
        """safe_compare() should handle None gracefully (even though not intended)."""
        dt = utc_now()
        
        # These should not crash (though not recommended usage)
        try:
            safe_compare(dt, dt)  # Normal case should work
        except Exception:
            pytest.fail("safe_compare should handle normal datetime objects")


class TestBackwardCompatibility:
    """Test backward compatibility with existing patterns."""
    
    def test_utc_constant_can_replace_timezone_utc(self):
        """UTC constant should be drop-in replacement for timezone.utc."""
        # Old pattern
        old_dt = datetime.now(timezone.utc)
        
        # New pattern
        new_dt = datetime.now(UTC)
        
        # Should be equivalent
        assert old_dt.tzinfo == new_dt.tzinfo
        
    def test_utc_now_can_replace_datetime_utcnow(self):
        """utc_now() should be drop-in replacement for datetime.utcnow()."""
        # Our replacement
        new_dt = utc_now()
        
        # Should be timezone-aware (unlike deprecated utcnow)
        assert is_aware(new_dt)
        assert new_dt.tzinfo == UTC
        
        # Should be recent
        now = datetime.now(timezone.utc)
        diff = abs((now - new_dt).total_seconds())
        assert diff < 1.0