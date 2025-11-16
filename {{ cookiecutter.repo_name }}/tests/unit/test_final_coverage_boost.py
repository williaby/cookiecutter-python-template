"""
Final coverage boost tests.

This module contains focused tests to achieve the final 1.91% coverage improvement
needed to reach the 80% target.
"""

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentConfigurationError, AgentRegistrationError, AgentTimeoutError
from src.agents.models import AgentInput, AgentOutput
from src.agents.registry import AgentRegistry


class TestFinalCoverageBoost:
    """Test suite to boost final coverage percentage."""

    def test_base_agent_config_validation_edge_case(self):
        """Test BaseAgent _validate_configuration with non-dict config."""

        class TestAgent(BaseAgent):
            def __init__(self, config):
                # Skip normal initialization to test _validate_configuration directly
                self.config = config
                self.agent_id = "test_agent"

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=1.0, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent({"agent_id": "test_agent"})

        # Test with non-dict config - this should hit line 216 in base_agent.py
        agent.config = "not_a_dict"

        with pytest.raises(AgentConfigurationError):  # This should raise AgentConfigurationError
            agent._validate_configuration()

    def test_agent_timeout_error_with_processing_time(self):
        """Test AgentTimeoutError with processing_time parameter."""

        # This should hit line 334 in exceptions.py
        error = AgentTimeoutError(
            message="Test timeout",
            timeout=5.0,
            processing_time=5.1,  # This line should be covered
            agent_id="test_agent",
            request_id="test_request",
        )

        assert error.context["timeout"] == 5.0
        assert error.context["processing_time"] == 5.1
        assert error.agent_id == "test_agent"
        assert error.request_id == "test_request"

    def test_registry_missing_required_method(self):
        """Test registry validation for missing required methods."""

        class IncompleteAgent(BaseAgent):
            """Agent class missing required execute method."""

            def __init__(self, config):
                super().__init__(config)
                # Missing execute method - this should be caught by hasattr check

        registry = AgentRegistry()

        # This should hit line 199 in registry.py
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("incomplete_agent")(IncompleteAgent)

        assert excinfo.value.error_code == "MISSING_REQUIRED_METHOD"
        assert "execute" in excinfo.value.message

        # Clean up
        registry.clear()

    def test_agent_timeout_error_minimal(self):
        """Test AgentTimeoutError with minimal parameters."""

        # Test with only timeout to ensure proper branching
        error = AgentTimeoutError(message="Test timeout", timeout=10.0, agent_id="test_agent")

        assert error.context["timeout"] == 10.0
        assert "processing_time" not in error.context
        assert error.agent_id == "test_agent"

    def test_agent_timeout_error_only_processing_time(self):
        """Test AgentTimeoutError with only processing_time parameter."""

        # Test with only processing_time to ensure proper branching
        error = AgentTimeoutError(message="Test timeout", processing_time=2.5, agent_id="test_agent")

        assert error.context["processing_time"] == 2.5
        assert "timeout" not in error.context
        assert error.agent_id == "test_agent"
