"""Pytest configuration and shared fixtures for {{ cookiecutter.project_name }} tests.

This module provides:
- Test fixture paths and directories
- Pytest markers for test categorization
- Shared fixtures for common test resources
- Temporary directory management
"""

from pathlib import Path

import pytest


# ============================================================================
# Test Fixture Paths
# ============================================================================

# Root paths
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "test_fixtures"
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"


# ============================================================================
# Pytest Markers
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "unit: Unit tests (fast, isolated, no external dependencies)",
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests (moderate speed, may use fixtures)",
    )
    config.addinivalue_line(
        "markers",
        "slow: Slow tests (can be excluded with -m 'not slow')",
    )


# ============================================================================
# Fixture Directory Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory.

    Returns:
        Path object pointing to the test fixtures directory.
    """
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def benchmarks_dir() -> Path:
    """Return path to benchmarks directory.

    Returns:
        Path object pointing to the benchmarks directory.
    """
    return BENCHMARKS_DIR


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return temporary directory for test outputs.

    Creates and returns a clean temporary directory for each test to write
    output files.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        Path object pointing to the temporary output directory.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Return temporary directory for caching.

    Creates and returns a clean temporary cache directory for each test.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        Path object pointing to the temporary cache directory.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


# ============================================================================
# Logging Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Setup test logging configuration.

    Automatically applied to all tests to ensure consistent logging setup.
    """
    from {{ cookiecutter.project_slug }}.utils.logging import setup_logging

    setup_logging(level="DEBUG", json_logs=False, include_timestamp=False)
