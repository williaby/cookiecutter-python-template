"""
Final coverage push tests.

This module contains very targeted tests to push coverage over 80%.
"""

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentRegistrationError
from src.agents.models import AgentInput
from src.agents.registry import AgentRegistry


class TestFinalCoveragePush:
    """Test suite to push coverage over 80%."""

    def test_agent_input_context_validation_errors(self):
        """Test AgentInput context validation error paths."""

        # Test non-dict context
        with pytest.raises(ValueError, match="Context must be a dictionary"):
            AgentInput(content="test", context="not_a_dict", agent_id="test_agent")  # This should trigger the error

        # Test non-string context keys
        with pytest.raises(ValueError, match="All context keys must be strings"):
            AgentInput(content="test", context={123: "value"}, agent_id="test_agent")  # Non-string key

    def test_agent_input_config_overrides_validation_errors(self):
        """Test AgentInput config_overrides validation error paths."""

        # Test non-dict config_overrides
        with pytest.raises(ValueError, match="Config overrides must be a dictionary"):
            AgentInput(
                content="test",
                config_overrides="not_a_dict",
                agent_id="test_agent",  # This should trigger the error
            )

        # Test non-string config override keys
        with pytest.raises(ValueError, match="All config override keys must be strings"):
            AgentInput(content="test", config_overrides={123: "value"}, agent_id="test_agent")  # Non-string key

    def test_agent_input_code_content_length_validation(self):
        """Test AgentInput code content length validation."""

        # Test code content that exceeds 50,000 characters
        long_code = "x" * 50001

        with pytest.raises(ValueError, match="Code content cannot exceed 50,000 characters"):
            AgentInput(
                content=long_code,
                context={"content_type": "code"},  # This should trigger the length check
                agent_id="test_agent",
            )

    def test_registry_missing_execute_method_exact_path(self):
        """Test the exact path for missing execute method in registry."""

        class IncompleteAgent(BaseAgent):
            """Agent class that inherits from BaseAgent but lacks execute method."""

            def __init__(self, config):
                super().__init__(config)
                # This class intentionally does not define execute method
                # Remove the execute method to trigger the hasattr check
                if hasattr(self, "execute"):
                    delattr(self, "execute")

        registry = AgentRegistry()

        # This should trigger the hasattr check on line 198 and raise on line 199
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("incomplete_agent")(IncompleteAgent)

        assert excinfo.value.error_code == "MISSING_REQUIRED_METHOD"
        assert "execute" in excinfo.value.message

        # Clean up
        registry.clear()

    def test_registry_abstract_method_detection(self):
        """Test abstract method detection in registry validation."""

        class AbstractAgent(BaseAgent):
            """Agent class with abstract execute method."""

            def __init__(self, config):
                super().__init__(config)

            # This will be detected as abstract
            async def execute(self, agent_input):
                """Abstract execute method."""

        # Mark the execute method as abstract
        AbstractAgent.execute.__isabstractmethod__ = True

        registry = AgentRegistry()

        # This should trigger the abstract method detection on line 208
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("abstract_agent")(AbstractAgent)

        assert excinfo.value.error_code == "MISSING_REQUIRED_METHOD"
        assert "execute" in excinfo.value.message

        # Clean up
        registry.clear()
