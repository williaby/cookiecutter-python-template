"""Comprehensive tests for exceptions.py coverage gaps - targeting 0% coverage functions."""

import pytest

from src.agents.exceptions import AgentError, AgentExecutionError, AgentTimeoutError


class TestAgentTimeoutErrorCoverageGaps:
    """Test AgentTimeoutError methods with 0% coverage."""

    def test_processing_time_property(self):
        """Test processing_time property with 0% coverage."""
        # Test with processing_time provided
        error = AgentTimeoutError(
            message="Test timeout",
            timeout=30.0,
            processing_time=35.5,
            agent_id="test_agent",
            request_id="req-123",
        )

        assert error.processing_time == 35.5
        assert error.timeout == 30.0

    def test_processing_time_property_none(self):
        """Test processing_time property when None."""
        # Test without processing_time
        error = AgentTimeoutError(message="Test timeout", timeout=30.0, agent_id="test_agent", request_id="req-123")

        assert error.processing_time is None
        assert error.timeout == 30.0

    def test_timeout_property_none(self):
        """Test timeout property when None."""
        # Test without timeout
        error = AgentTimeoutError(
            message="Test timeout",
            processing_time=35.5,
            agent_id="test_agent",
            request_id="req-123",
        )

        assert error.timeout is None
        assert error.processing_time == 35.5

    def test_both_properties_none(self):
        """Test both properties when None."""
        # Test without timeout or processing_time
        error = AgentTimeoutError(message="Test timeout", agent_id="test_agent", request_id="req-123")

        assert error.timeout is None
        assert error.processing_time is None

    def test_context_built_correctly(self):
        """Test that context is built correctly with timeout and processing_time."""
        error = AgentTimeoutError(
            message="Test timeout with both values",
            timeout=45.0,
            processing_time=50.2,
            agent_id="test_agent",
            request_id="req-456",
        )

        # Verify context contains both values
        assert error.context["timeout"] == 45.0
        assert error.context["processing_time"] == 50.2
        assert error.error_code == "EXECUTION_TIMEOUT"

    def test_context_empty_when_no_values(self):
        """Test that context is empty when no timeout/processing_time provided."""
        error = AgentTimeoutError(message="Test timeout no values", agent_id="test_agent", request_id="req-789")

        # Context should be empty
        assert error.context == {}
        assert error.error_code == "EXECUTION_TIMEOUT"

    def test_default_message(self):
        """Test AgentTimeoutError with default message."""
        error = AgentTimeoutError(timeout=30.0, processing_time=35.0, agent_id="test_agent", request_id="req-default")

        assert error.message == "Agent execution timed out"
        assert error.timeout == 30.0
        assert error.processing_time == 35.0

    def test_minimal_initialization(self):
        """Test AgentTimeoutError with minimal parameters."""
        error = AgentTimeoutError()

        assert error.message == "Agent execution timed out"
        assert error.timeout is None
        assert error.processing_time is None
        assert error.agent_id is None
        assert error.request_id is None
        assert error.error_code == "EXECUTION_TIMEOUT"

    def test_inheritance_chain(self):
        """Test that AgentTimeoutError inherits correctly."""
        error = AgentTimeoutError(
            message="Test inheritance",
            timeout=20.0,
            processing_time=25.0,
            agent_id="inheritance_agent",
        )

        # Should inherit from AgentExecutionError
        assert isinstance(error, AgentExecutionError)
        assert isinstance(error, AgentError)
        assert isinstance(error, Exception)

    def test_to_dict_includes_timeout_data(self):
        """Test that to_dict includes timeout-specific data."""
        error = AgentTimeoutError(
            message="Test dict conversion",
            timeout=40.0,
            processing_time=45.0,
            agent_id="dict_agent",
            request_id="req-dict",
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "AgentTimeoutError"
        assert error_dict["message"] == "Test dict conversion"
        assert error_dict["error_code"] == "EXECUTION_TIMEOUT"
        assert error_dict["context"]["timeout"] == 40.0
        assert error_dict["context"]["processing_time"] == 45.0
        assert error_dict["agent_id"] == "dict_agent"
        assert error_dict["request_id"] == "req-dict"

    def test_str_representation_includes_timeout_data(self):
        """Test that string representation includes timeout data."""
        error = AgentTimeoutError(
            message="Test string representation",
            timeout=60.0,
            processing_time=65.0,
            agent_id="string_agent",
            request_id="req-string",
        )

        error_str = str(error)

        assert "[EXECUTION_TIMEOUT]" in error_str
        assert "Test string representation" in error_str
        assert "Agent: string_agent" in error_str
        assert "Request: req-string" in error_str
        assert "timeout" in error_str.lower()  # Context should be included

    @pytest.mark.parametrize(
        ("timeout", "processing_time", "expected_timeout", "expected_processing"),
        [
            (10.0, 12.0, 10.0, 12.0),
            (None, 15.0, None, 15.0),
            (20.0, None, 20.0, None),
            (None, None, None, None),
            (0.0, 0.0, 0.0, 0.0),
            (100.5, 99.9, 100.5, 99.9),
        ],
    )
    def test_property_combinations(self, timeout, processing_time, expected_timeout, expected_processing):
        """Test various combinations of timeout and processing_time values."""
        error = AgentTimeoutError(
            message="Parametrized test",
            timeout=timeout,
            processing_time=processing_time,
        )

        assert error.timeout == expected_timeout
        assert error.processing_time == expected_processing
