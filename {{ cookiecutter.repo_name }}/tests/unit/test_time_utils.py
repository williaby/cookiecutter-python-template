"""Tests for time_utils module."""

from datetime import UTC, datetime, timedelta, timezone

from src.utils.time_utils import (
    __all__,
    format_datetime,
    from_timestamp,
    parse_iso_datetime,
    to_timestamp,
    to_utc_datetime,
    utc_now,
    utc_timestamp,
)


class TestTimeUtils:
    """Test cases for time utilities."""

    def test_utc_now(self):
        """Test utc_now function."""
        now = utc_now()
        assert now.tzinfo == UTC
        assert isinstance(now, datetime)

    def test_utc_timestamp(self):
        """Test utc_timestamp function."""
        timestamp = utc_timestamp()
        assert isinstance(timestamp, str)
        assert timestamp.endswith("+00:00")

    def test_to_utc_datetime(self):
        """Test to_utc_datetime function."""
        dt = to_utc_datetime(2024, 1, 1, 12, 0, 0)
        assert dt.tzinfo == UTC
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12
        assert dt.minute == 0
        assert dt.second == 0

    def test_to_utc_datetime_with_defaults(self):
        """Test to_utc_datetime with default parameters."""
        dt = to_utc_datetime(2024, 1, 1)
        assert dt.tzinfo == UTC
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.microsecond == 0

    def test_from_timestamp(self):
        """Test from_timestamp function."""
        # 2022-01-01 00:00:00 UTC
        timestamp = 1640995200
        dt = from_timestamp(timestamp)
        assert dt.tzinfo == UTC
        assert dt.year == 2022
        assert dt.month == 1
        assert dt.day == 1

    def test_to_timestamp_with_timezone_aware_datetime(self):
        """Test to_timestamp with timezone-aware datetime."""
        dt = to_utc_datetime(2024, 1, 1, 12, 0, 0)
        timestamp = to_timestamp(dt)
        assert isinstance(timestamp, float)

        # Convert back to verify
        dt_back = from_timestamp(timestamp)
        assert dt_back == dt

    def test_to_timestamp_with_naive_datetime(self):
        """Test to_timestamp with naive datetime - covers line 106."""
        # Create naive datetime (no timezone) - intentionally naive for testing
        dt = datetime(2024, 1, 1, 12, 0, 0)  # noqa: DTZ001
        assert dt.tzinfo is None

        timestamp = to_timestamp(dt)
        assert isinstance(timestamp, float)

        # Should be treated as UTC
        dt_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        expected_timestamp = dt_utc.timestamp()
        assert timestamp == expected_timestamp

    def test_format_datetime_with_utc(self):
        """Test format_datetime with UTC datetime."""
        dt = to_utc_datetime(2024, 1, 1, 12, 0, 0)
        formatted = format_datetime(dt)
        assert formatted == "2024-01-01 12:00:00 UTC"

    def test_format_datetime_with_custom_format(self):
        """Test format_datetime with custom format."""
        dt = to_utc_datetime(2024, 1, 1, 12, 0, 0)
        formatted = format_datetime(dt, "%Y-%m-%d %H:%M")
        assert formatted == "2024-01-01 12:00"

    def test_format_datetime_with_naive_datetime(self):
        """Test format_datetime with naive datetime - covers line 127."""
        # Create naive datetime (no timezone) - intentionally naive for testing
        dt = datetime(2024, 1, 1, 12, 0, 0)  # noqa: DTZ001
        assert dt.tzinfo is None

        formatted = format_datetime(dt)
        assert formatted == "2024-01-01 12:00:00 UTC"

    def test_format_datetime_with_non_utc_timezone(self):
        """Test format_datetime with non-UTC timezone - covers line 129."""
        # Create datetime with different timezone (EST is UTC-5)
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 1, 7, 0, 0, tzinfo=est)  # 7 AM EST = 12 PM UTC

        formatted = format_datetime(dt)
        assert formatted == "2024-01-01 12:00:00 UTC"

    def test_parse_iso_datetime(self):
        """Test parse_iso_datetime function."""
        iso_string = "2024-01-01T12:00:00+00:00"
        dt = parse_iso_datetime(iso_string)
        assert dt.tzinfo == UTC
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12

    def test_parse_iso_datetime_with_z_suffix(self):
        """Test parse_iso_datetime with Z suffix."""
        iso_string = "2024-01-01T12:00:00Z"
        dt = parse_iso_datetime(iso_string)
        assert dt.tzinfo == UTC
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12

    def test_utc_constant(self):
        """Test UTC constant."""
        assert UTC is UTC  # noqa: PLR0124

    def test_all_exports(self):
        """Test that all public API functions are exported."""
        expected_exports = [
            "UTC",
            "format_datetime",
            "from_timestamp",
            "parse_iso_datetime",
            "to_timestamp",
            "to_utc_datetime",
            "utc_now",
            "utc_timestamp",
        ]

        assert set(__all__) == set(expected_exports)

    def test_round_trip_timestamp_conversion(self):
        """Test round-trip conversion between datetime and timestamp."""
        # Start with a known datetime
        original_dt = to_utc_datetime(2024, 6, 15, 10, 30, 45)

        # Convert to timestamp and back
        timestamp = to_timestamp(original_dt)
        converted_dt = from_timestamp(timestamp)

        # Should be the same (within microsecond precision)
        assert abs((converted_dt - original_dt).total_seconds()) < 0.001

    def test_timezone_consistency(self):
        """Test that all functions maintain UTC timezone consistency."""
        # Test that all datetime-returning functions return UTC
        now = utc_now()
        assert now.tzinfo == UTC

        dt_from_components = to_utc_datetime(2024, 1, 1)
        assert dt_from_components.tzinfo == UTC

        dt_from_timestamp = from_timestamp(1640995200)
        assert dt_from_timestamp.tzinfo == UTC

        dt_from_iso = parse_iso_datetime("2024-01-01T12:00:00Z")
        assert dt_from_iso.tzinfo == UTC
