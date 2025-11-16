"""
Unit tests for error handling boundary conditions.

This module tests edge cases and boundary conditions in error handling
that might not be covered in standard tests, focusing on exception
propagation, error context preservation, and error recovery scenarios.
"""

import asyncio
from typing import Any

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import (
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    AgentRegistrationError,
    AgentTimeoutError,
    AgentValidationError,
    create_agent_error,
    handle_agent_error,
)
from src.agents.models import AgentInput, AgentOutput
from src.agents.registry import AgentRegistry


class TestErrorBoundaryConditions:
    """Test error handling boundary conditions and edge cases."""

    @pytest.fixture
    def error_prone_agent_class(self):
        """Agent class that can simulate various error conditions."""

        class ErrorProneAgent(BaseAgent):
            def __init__(self, config: dict[str, Any]):
                self.error_scenario = config.get("error_scenario", "none")
                super().__init__(config)

            def _validate_configuration(self) -> None:
                super()._validate_configuration()
                if self.error_scenario == "config_validation":
                    raise AgentConfigurationError(
                        message="Configuration validation failed",
                        error_code="CONFIG_VALIDATION_FAILED",
                        agent_id=self.agent_id,
                    )

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                scenario = self.error_scenario

                if scenario == "none":
                    return self._create_output("Success", confidence=1.0)
                if scenario == "value_error":
                    raise ValueError("Invalid value provided")
                if scenario == "key_error":
                    raise KeyError("missing_key")
                if scenario == "timeout_error":
                    raise TimeoutError("Operation timed out")
                if scenario == "generic_error":
                    raise Exception("Generic error")
                if scenario == "nested_error":
                    try:
                        raise ValueError("Inner error")
                    except ValueError as e:
                        raise RuntimeError("Outer error") from e
                elif scenario == "partial_output":
                    # Simulate partial output creation that fails
                    return AgentOutput(
                        content="",  # Empty content should cause validation error
                        confidence=1.0,
                        processing_time=0.1,
                        agent_id=self.agent_id,
                    )
                elif scenario == "invalid_confidence":
                    return AgentOutput(
                        content="Test",
                        confidence=2.0,  # Invalid confidence > 1.0
                        processing_time=0.1,
                        agent_id=self.agent_id,
                    )
                elif scenario == "negative_time":
                    return AgentOutput(
                        content="Test",
                        confidence=1.0,
                        processing_time=-1.0,  # Invalid negative time
                        agent_id=self.agent_id,
                    )

                return self._create_output("Unknown scenario", confidence=0.5)

        return ErrorProneAgent

    @pytest.fixture
    def registry(self):
        """Fresh registry for testing."""
        registry = AgentRegistry()
        yield registry
        registry.clear()

    @pytest.mark.unit
    def test_agent_error_serialization_boundary_cases(self):
        """Test AgentError serialization with various edge cases."""
        # Test with None values
        error = AgentError(message="Test error", error_code="TEST_ERROR", context=None, agent_id=None, request_id=None)

        error_dict = error.to_dict()
        assert error_dict["context"] == {}
        assert error_dict["agent_id"] is None
        assert error_dict["request_id"] is None

        # Test with complex context
        complex_context = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "unicode": "🤖",
            "special_chars": "!@#$%^&*()",
        }

        error_complex = AgentError(
            message="Complex error",
            error_code="COMPLEX_ERROR",
            context=complex_context,
            agent_id="test_agent",
            request_id="req-123",
        )

        error_dict_complex = error_complex.to_dict()
        assert error_dict_complex["context"] == complex_context
        assert error_dict_complex["agent_id"] == "test_agent"
        assert error_dict_complex["request_id"] == "req-123"

    @pytest.mark.unit
    def test_error_context_preservation_across_boundaries(self):
        """Test error context preservation across different error boundaries."""
        original_context = {"operation": "test", "user_id": "123"}

        # Test configuration error context
        config_error = AgentConfigurationError(
            message="Config failed",
            error_code="CONFIG_FAILED",
            context=original_context,
            agent_id="config_agent",
        )

        assert config_error.context == original_context
        assert config_error.agent_id == "config_agent"

        # Test execution error context
        exec_error = AgentExecutionError(
            message="Execution failed",
            error_code="EXEC_FAILED",
            context=original_context,
            agent_id="exec_agent",
            request_id="req-456",
        )

        assert exec_error.context == original_context
        assert exec_error.request_id == "req-456"

    @pytest.mark.unit
    def test_handle_agent_error_boundary_cases(self):
        """Test handle_agent_error with various exception types."""
        # Test with already wrapped AgentError
        original_error = AgentValidationError(
            message="Already wrapped",
            error_code="ALREADY_WRAPPED",
            agent_id="wrapped_agent",
        )

        handled = handle_agent_error(original_error, agent_id="different_agent")
        assert handled is original_error  # Should return the same error
        assert handled.agent_id == "wrapped_agent"  # Should not change agent_id

        # Test with ValueError
        value_error = ValueError("Invalid value")
        handled_value = handle_agent_error(value_error, agent_id="value_agent", request_id="req-789")

        assert isinstance(handled_value, AgentExecutionError)
        assert handled_value.agent_id == "value_agent"
        assert handled_value.request_id == "req-789"
        assert "Invalid value" in handled_value.message

        # Test with KeyError
        key_error = KeyError("missing_key")
        handled_key = handle_agent_error(key_error, agent_id="key_agent")

        assert isinstance(handled_key, AgentConfigurationError)
        assert handled_key.error_code == "MISSING_REQUIRED_CONFIG"

        # Test with TimeoutError
        timeout_error = TimeoutError("Timeout occurred")
        handled_timeout = handle_agent_error(timeout_error, agent_id="timeout_agent")

        assert isinstance(handled_timeout, AgentTimeoutError)
        assert handled_timeout.agent_id == "timeout_agent"

        # Test with unknown exception type
        unknown_error = RuntimeError("Unknown error")
        handled_unknown = handle_agent_error(unknown_error, agent_id="unknown_agent")

        assert isinstance(handled_unknown, AgentError)
        assert handled_unknown.error_code == "UNKNOWN_ERROR"
        assert handled_unknown.context["original_error"] == "RuntimeError"

    @pytest.mark.unit
    def test_create_agent_error_boundary_cases(self):
        """Test create_agent_error factory with various error types."""
        # Test with unknown error type
        unknown_error = create_agent_error(
            error_type="unknown_type",
            message="Unknown error type",
            agent_id="test_agent",
        )

        assert isinstance(unknown_error, AgentError)
        assert unknown_error.error_code == "UNKNOWN_ERROR"

        # Test with timeout error type
        timeout_error = create_agent_error(
            error_type="timeout",
            message="Custom timeout",
            agent_id="timeout_agent",
        )

        assert isinstance(timeout_error, AgentTimeoutError)
        assert timeout_error.error_code == "EXECUTION_TIMEOUT"

        # Test with empty context
        empty_context_error = create_agent_error(
            error_type="validation",
            message="Empty context test",
            context={},
            agent_id="empty_agent",
        )

        assert isinstance(empty_context_error, AgentValidationError)
        assert empty_context_error.context == {}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_propagation_in_agent_execution(self, error_prone_agent_class):
        """Test error propagation through agent execution boundaries."""
        # Test ValueError propagation
        config = {"agent_id": "value_error_agent", "error_scenario": "value_error"}
        agent = error_prone_agent_class(config)

        agent_input = AgentInput(content="Test value error")

        with pytest.raises(AgentExecutionError) as excinfo:
            await agent.process(agent_input)

        assert "Invalid value provided" in str(excinfo.value)
        assert excinfo.value.agent_id == "value_error_agent"
        assert excinfo.value.request_id == agent_input.request_id

        # Test KeyError propagation
        config_key = {"agent_id": "key_error_agent", "error_scenario": "key_error"}
        agent_key = error_prone_agent_class(config_key)

        with pytest.raises(AgentConfigurationError) as excinfo_key:
            await agent_key.process(agent_input)

        assert excinfo_key.value.error_code == "MISSING_REQUIRED_CONFIG"
        assert excinfo_key.value.agent_id == "key_error_agent"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nested_error_handling(self, error_prone_agent_class):
        """Test handling of nested exceptions."""
        config = {"agent_id": "nested_error_agent", "error_scenario": "nested_error"}
        agent = error_prone_agent_class(config)

        agent_input = AgentInput(content="Test nested error")

        with pytest.raises(AgentError) as excinfo:
            await agent.process(agent_input)

        # Should wrap the outer error (RuntimeError)
        assert "Outer error" in str(excinfo.value)
        assert excinfo.value.context["original_error"] == "RuntimeError"

    @pytest.mark.unit
    def test_configuration_validation_error_boundaries(self, error_prone_agent_class):
        """Test configuration validation error boundaries."""
        # Test configuration validation failure during initialization
        config = {"agent_id": "config_fail_agent", "error_scenario": "config_validation"}

        with pytest.raises(AgentConfigurationError) as excinfo:
            error_prone_agent_class(config)

        assert excinfo.value.error_code == "CONFIG_VALIDATION_FAILED"
        assert excinfo.value.agent_id == "config_fail_agent"

        # Test missing agent_id
        config_no_id = {"error_scenario": "none"}

        with pytest.raises(AgentConfigurationError) as excinfo_no_id:
            error_prone_agent_class(config_no_id)

        assert excinfo_no_id.value.error_code == "MISSING_REQUIRED_CONFIG"
        assert "agent_id" in excinfo_no_id.value.context["required_field"]

        # Test invalid agent_id format
        config_invalid_id = {"agent_id": "invalid-id-with-dashes!", "error_scenario": "none"}

        with pytest.raises(AgentConfigurationError) as excinfo_invalid:
            error_prone_agent_class(config_invalid_id)

        assert excinfo_invalid.value.error_code == "INVALID_CONFIG_VALUE"
        assert "invalid-id-with-dashes!" in excinfo_invalid.value.message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_output_validation_error_boundaries(self, error_prone_agent_class):
        """Test output validation error boundaries."""
        # Test empty content validation
        config = {"agent_id": "empty_content_agent", "error_scenario": "partial_output"}
        agent = error_prone_agent_class(config)

        agent_input = AgentInput(content="Test empty content")

        with pytest.raises((AgentExecutionError, AgentValidationError)):  # Should raise validation error
            await agent.process(agent_input)

        # Test invalid confidence value
        config_conf = {"agent_id": "invalid_conf_agent", "error_scenario": "invalid_confidence"}
        agent_conf = error_prone_agent_class(config_conf)

        with pytest.raises((AgentExecutionError, AgentValidationError)):  # Should raise validation error
            await agent_conf.process(agent_input)

        # Test negative processing time
        config_time = {"agent_id": "negative_time_agent", "error_scenario": "negative_time"}
        agent_time = error_prone_agent_class(config_time)

        with pytest.raises((AgentExecutionError, AgentValidationError)):  # Should raise validation error
            await agent_time.process(agent_input)

    @pytest.mark.unit
    def test_registry_error_boundary_conditions(self, error_prone_agent_class, registry):
        """Test registry error handling boundary conditions."""
        # Test duplicate registration
        registry.register("duplicate_agent")(error_prone_agent_class)

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("duplicate_agent")(error_prone_agent_class)

        assert excinfo.value.error_code == "DUPLICATE_AGENT_ID"
        assert "duplicate_agent" in excinfo.value.message

        # Test getting non-existent agent
        with pytest.raises(AgentRegistrationError) as excinfo_missing:
            registry.get_agent("non_existent_agent", {"agent_id": "non_existent_agent"})

        assert excinfo_missing.value.error_code == "AGENT_NOT_FOUND"

        # Test invalid agent class registration
        class InvalidAgent:  # Does not inherit from BaseAgent
            pass

        with pytest.raises(AgentRegistrationError) as excinfo_invalid:
            registry.register("invalid_agent")(InvalidAgent)

        assert excinfo_invalid.value.error_code == "INVALID_AGENT_CLASS"

        # Test agent instantiation failure
        registry.register("failing_agent")(error_prone_agent_class)

        with pytest.raises(AgentRegistrationError) as excinfo_instantiation:
            registry.get_agent("failing_agent", {"agent_id": "failing_agent", "error_scenario": "config_validation"})

        assert excinfo_instantiation.value.error_code == "INSTANTIATION_FAILED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_error_boundary_conditions(self, error_prone_agent_class):
        """Test timeout error handling boundary conditions."""
        # Test timeout with very short duration and slow scenario
        config = {"agent_id": "timeout_agent", "timeout": 0.001, "error_scenario": "none"}
        agent = error_prone_agent_class(config)

        # Override the execute method to add a delay
        original_execute = agent.execute

        async def slow_execute(agent_input):
            await asyncio.sleep(0.1)  # This will exceed the 0.001 timeout
            return await original_execute(agent_input)

        agent.execute = slow_execute

        agent_input = AgentInput(content="Test timeout")

        with pytest.raises(AgentTimeoutError) as excinfo:
            await agent.process(agent_input)

        assert excinfo.value.error_code == "EXECUTION_TIMEOUT"
        assert excinfo.value.agent_id == "timeout_agent"
        assert excinfo.value.request_id == agent_input.request_id

        # Test timeout context information
        assert "timeout" in excinfo.value.context
        assert excinfo.value.context["timeout"] == 0.001

    @pytest.mark.unit
    def test_error_string_representation_boundaries(self):
        """Test error string representation with various edge cases."""
        # Test with minimal information
        minimal_error = AgentError(message="Minimal error")
        assert "[UNKNOWN_ERROR] Minimal error" in str(minimal_error)

        # Test with all information
        complete_error = AgentError(
            message="Complete error",
            error_code="COMPLETE_ERROR",
            context={"key": "value"},
            agent_id="complete_agent",
            request_id="req-complete",
        )

        error_str = str(complete_error)
        assert "[COMPLETE_ERROR] Complete error" in error_str
        assert "Agent: complete_agent" in error_str
        assert "Request: req-complete" in error_str
        assert "Context: {'key': 'value'}" in error_str

        # Test with unicode characters
        unicode_error = AgentError(message="Unicode error 🤖", error_code="UNICODE_ERROR", agent_id="unicode_agent_🚀")

        unicode_str = str(unicode_error)
        assert "Unicode error 🤖" in unicode_str
        assert "unicode_agent_🚀" in unicode_str

    @pytest.mark.unit
    def test_agent_initialization_error_recovery(self, error_prone_agent_class):
        """Test agent initialization error recovery scenarios."""
        # Test that agent is not marked as initialized after failure
        config = {"agent_id": "init_fail_agent", "error_scenario": "config_validation"}

        with pytest.raises(AgentConfigurationError):
            error_prone_agent_class(config)

        # Test successful initialization after fixing config
        config_fixed = {"agent_id": "init_success_agent", "error_scenario": "none"}
        agent_fixed = error_prone_agent_class(config_fixed)

        assert agent_fixed._initialized is True
        assert agent_fixed.agent_id == "init_success_agent"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, error_prone_agent_class):
        """Test error handling under concurrent execution."""
        config = {"agent_id": "concurrent_error_agent", "error_scenario": "value_error"}
        agent = error_prone_agent_class(config)

        # Create multiple concurrent requests that will all fail
        tasks = []
        for i in range(5):
            agent_input = AgentInput(content=f"Concurrent error test {i}")
            task = agent.process(agent_input)
            tasks.append(task)

        # All tasks should fail with the same error type
        with pytest.raises(AgentExecutionError):
            await asyncio.gather(*tasks)

        # Test mixed success/failure scenarios
        config_mixed = {"agent_id": "mixed_agent", "error_scenario": "none"}
        agent_mixed = error_prone_agent_class(config_mixed)

        mixed_tasks = []
        for i in range(3):
            # Create inputs that will succeed
            agent_input = AgentInput(content=f"Success test {i}")
            task = agent_mixed.process(agent_input)
            mixed_tasks.append(task)

        # These should all succeed
        results = await asyncio.gather(*mixed_tasks)
        assert len(results) == 3
        assert all(result.content == "Success" for result in results)
