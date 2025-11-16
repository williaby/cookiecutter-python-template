"""
Comprehensive coverage boost tests to push coverage above 80%.

This module contains targeted tests designed to exercise currently untested
code paths and achieve the required 80% test coverage threshold.
"""

import asyncio

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import (
    AgentConfigurationError,
    AgentExecutionError,
    AgentRegistrationError,
    AgentTimeoutError,
    AgentValidationError,
    handle_agent_error,
)
from src.agents.models import AgentConfig, AgentInput, AgentOutput
from src.agents.registry import AgentRegistry


class TestComprehensiveCoverageBoost:
    """Comprehensive test suite designed to maximize code coverage."""

    def test_agent_config_validation_comprehensive(self):
        """Test AgentConfig validation with various edge cases."""

        # Test valid config
        config = AgentConfig(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            config={"param1": "value1"},
            enabled=True,
        )
        assert config.agent_id == "test_agent"
        assert config.name == "Test Agent"
        assert config.description == "A test agent"
        assert config.enabled is True

        # Test agent_id validation errors
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            AgentConfig(agent_id="", name="Test", description="Test")

        with pytest.raises(ValueError, match="Agent ID must contain only alphanumeric characters and underscores"):
            AgentConfig(agent_id="test-agent", name="Test", description="Test")

        # Test name validation errors
        with pytest.raises(ValueError, match="Name cannot be empty"):
            AgentConfig(agent_id="test_agent", name="", description="Test")

        with pytest.raises(ValueError, match="Description cannot be empty"):
            AgentConfig(agent_id="test_agent", name="Test", description="")

    def test_agent_output_metadata_validation(self):
        """Test AgentOutput metadata validation edge cases."""

        # Test non-dict metadata
        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            AgentOutput(
                content="test",
                metadata="not_a_dict",
                confidence=0.9,
                processing_time=1.0,
                agent_id="test_agent",
            )

        # Test non-string metadata keys
        with pytest.raises(ValueError, match="All metadata keys must be strings"):
            AgentOutput(
                content="test",
                metadata={123: "value"},
                confidence=0.9,
                processing_time=1.0,
                agent_id="test_agent",
            )

    def test_agent_output_agent_id_validation(self):
        """Test AgentOutput agent_id validation."""

        # Test empty agent_id
        with pytest.raises(ValueError, match="Agent ID cannot be empty or whitespace-only"):
            AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id="")

        # Test invalid agent_id format
        with pytest.raises(ValueError, match="Agent ID must contain only alphanumeric characters and underscores"):
            AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id="test-agent")

    def test_agent_output_cross_field_validation(self):
        """Test AgentOutput cross-field validation."""

        # Test high processing time with high confidence (should not raise error but triggers validation)
        output = AgentOutput(content="test", confidence=0.96, processing_time=35.0, agent_id="test_agent")
        assert output.confidence == 0.96
        assert output.processing_time == 35.0

    def test_registry_comprehensive_error_paths(self):
        """Test registry error handling paths."""

        class InvalidAgent:
            """Class that's not a BaseAgent subclass."""

        registry = AgentRegistry()

        # Test registering non-BaseAgent class
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("invalid_agent")(InvalidAgent)

        assert excinfo.value.error_code == "INVALID_AGENT_CLASS"
        assert "BaseAgent" in excinfo.value.message

        # Test getting non-existent agent
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.get_agent_class("nonexistent")

        assert excinfo.value.error_code == "AGENT_NOT_FOUND"

        # Test getting agent instance for non-existent agent
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.get_agent("nonexistent", {"agent_id": "test"})

        assert excinfo.value.error_code == "AGENT_NOT_FOUND"

        # Clean up
        registry.clear()

    def test_registry_agent_instantiation_error(self):
        """Test registry agent instantiation error handling."""

        class FailingAgent(BaseAgent):
            def __init__(self, config):
                # This will fail during instantiation
                raise ValueError("Instantiation failed")

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id=self.agent_id)

        registry = AgentRegistry()
        registry.register("failing_agent")(FailingAgent)

        # Test instantiation failure
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.get_agent("failing_agent", {"agent_id": "failing_agent"})

        assert excinfo.value.error_code == "INSTANTIATION_FAILED"
        assert "Failed to instantiate agent" in excinfo.value.message

        # Clean up
        registry.clear()

    def test_registry_capability_operations(self):
        """Test registry capability operations comprehensively."""

        class CapabilityAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id=self.agent_id)

            def get_capabilities(self):
                return {
                    "input_types": ["text", "code"],
                    "output_types": ["analysis"],
                    "specialization": "security",
                    "supports_batch": True,
                }

        registry = AgentRegistry()
        registry.register("capability_agent")(CapabilityAgent)

        # Create instance to populate capabilities
        registry.get_agent("capability_agent", {"agent_id": "capability_agent"})

        # Test capability queries
        text_agents = registry.find_agents_by_type("text")
        code_agents = registry.find_agents_by_type("code")
        analysis_agents = registry.find_agents_by_type("text", "analysis")

        assert "capability_agent" in text_agents
        assert "capability_agent" in code_agents
        assert "capability_agent" in analysis_agents

        # Test custom capability filtering
        security_agents = registry.find_agents_by_capability("specialization", "security")
        batch_agents = registry.find_agents_by_capability("supports_batch", True)

        assert "capability_agent" in security_agents
        assert "capability_agent" in batch_agents

        # Test capability filtering with non-existent values
        empty_results = registry.find_agents_by_capability("nonexistent", "value")
        assert len(empty_results) == 0

        # Clean up
        registry.clear()

    def test_registry_status_reporting(self):
        """Test registry status reporting functionality."""

        class StatusAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id=self.agent_id)

            def get_capabilities(self):
                return {"input_types": ["text"]}

        registry = AgentRegistry()

        # Test empty registry status
        status = registry.get_registry_status()
        assert status["total_agents"] == 0
        assert status["cached_instances"] == 0
        assert status["agents_with_capabilities"] == 0

        # Register and create agents
        registry.register("status_agent1")(StatusAgent)
        registry.register("status_agent2")(StatusAgent)

        registry.get_cached_agent("status_agent1", {"agent_id": "status_agent1"})
        registry.get_cached_agent("status_agent2", {"agent_id": "status_agent2"})

        # Test populated registry status
        status = registry.get_registry_status()
        assert status["total_agents"] == 2
        assert status["cached_instances"] == 2
        assert status["agents_with_capabilities"] == 2
        assert set(status["registered_agents"]) == {"status_agent1", "status_agent2"}

        # Test unregistering
        registry.unregister("status_agent1")
        status = registry.get_registry_status()
        assert status["total_agents"] == 1
        assert "status_agent1" not in status["registered_agents"]

        # Clean up
        registry.clear()

    @pytest.mark.asyncio
    async def test_base_agent_configuration_override_edge_cases(self):
        """Test BaseAgent configuration override edge cases."""

        class ConfigTestAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Access config during execution to test override
                confidence = self.config.get("confidence", 0.9)
                processing_time = self.config.get("processing_time", 0.1)

                await asyncio.sleep(processing_time)

                return AgentOutput(
                    content="Config test result",
                    confidence=confidence,
                    processing_time=processing_time,
                    agent_id=self.agent_id,
                )

        agent = ConfigTestAgent({"agent_id": "config_test", "confidence": 0.8, "processing_time": 0.01})

        # Test with config overrides
        agent_input = AgentInput(
            content="Test config override",
            config_overrides={"confidence": 0.95, "processing_time": 0.02},
        )

        result = await agent.process(agent_input)

        # Should use overridden values
        assert result.confidence == 0.95
        assert result.processing_time >= 0.02

        # Original config should be restored
        assert agent.config["confidence"] == 0.8
        assert agent.config["processing_time"] == 0.01

    def test_base_agent_validation_edge_cases(self):
        """Test BaseAgent validation edge cases."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id=self.agent_id)

        # Test missing agent_id
        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent({})

        assert excinfo.value.error_code == "MISSING_REQUIRED_CONFIG"

        # Test empty agent_id
        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent({"agent_id": ""})

        assert excinfo.value.error_code == "MISSING_REQUIRED_CONFIG"

        # Test agent_id with invalid characters
        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent({"agent_id": "test-agent"})

        assert excinfo.value.error_code == "INVALID_CONFIG_VALUE"

    @pytest.mark.asyncio
    async def test_base_agent_timeout_scenarios(self):
        """Test BaseAgent timeout handling scenarios."""

        class TimeoutTestAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Sleep longer than timeout
                sleep_time = agent_input.context.get("sleep_time", 10.0) if agent_input.context else 10.0
                await asyncio.sleep(sleep_time)

                return AgentOutput(
                    content="Should not reach here",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

        # Test with very short timeout
        agent = TimeoutTestAgent({"agent_id": "timeout_test", "timeout": 0.01})
        agent_input = AgentInput(content="Test timeout", context={"sleep_time": 1.0})

        with pytest.raises(AgentTimeoutError) as excinfo:
            await agent.process(agent_input)

        assert excinfo.value.agent_id == "timeout_test"
        assert "timed out" in str(excinfo.value)

    def test_exception_handling_comprehensive(self):
        """Test exception handling and error conversion."""

        # Test ValueError conversion
        error = handle_agent_error(ValueError("Test error"), agent_id="test_agent")
        assert isinstance(error, AgentExecutionError)
        assert error.error_code == "EXECUTION_ERROR"
        assert error.agent_id == "test_agent"

        # Test TimeoutError conversion
        error = handle_agent_error(TimeoutError("Timeout occurred"), agent_id="test_agent")
        assert isinstance(error, AgentTimeoutError)
        assert error.agent_id == "test_agent"

        # Test KeyError conversion
        error = handle_agent_error(KeyError("missing_key"), agent_id="test_agent")
        assert isinstance(error, AgentConfigurationError)
        assert error.error_code == "MISSING_REQUIRED_CONFIG"

        # Test already AgentError (should return as-is)
        original_error = AgentValidationError(message="Original error", agent_id="test_agent")
        returned_error = handle_agent_error(original_error)
        assert returned_error is original_error

    @pytest.mark.asyncio
    async def test_agent_metrics_and_logging(self):
        """Test agent metrics and logging functionality."""

        class MetricsTestAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content="Metrics test result",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

        agent = MetricsTestAgent({"agent_id": "metrics_test"})
        agent_input = AgentInput(content="Test metrics")

        # Execute to trigger metrics
        result = await agent.process(agent_input)

        # Verify result
        assert result.content == "Metrics test result"
        assert result.agent_id == "metrics_test"

    def test_registry_edge_cases(self):
        """Test registry edge cases and error conditions."""

        registry = AgentRegistry()

        # Test contains operation
        assert "nonexistent" not in registry

        # Test length operation
        assert len(registry) == 0

        # Test listing empty registry
        assert registry.list_agents() == []

        # Test clearing empty registry
        registry.clear()  # Should not raise error

        # Test unregistering non-existent agent
        with pytest.raises(AgentRegistrationError):
            registry.unregister("nonexistent")

    def test_agent_input_edge_cases(self):
        """Test AgentInput edge cases and validation paths."""

        # Test with all optional fields
        agent_input = AgentInput(content="Test content")
        assert agent_input.content == "Test content"
        assert agent_input.context is None
        assert agent_input.config_overrides is None
        assert agent_input.request_id is not None
        assert agent_input.timestamp is not None

        # Test content whitespace stripping
        agent_input = AgentInput(content="  Test content  ")
        assert agent_input.content == "Test content"

        # Test with complex context
        complex_context = {"nested": {"key": "value"}, "list_data": [1, 2, 3], "boolean": True, "number": 42}
        agent_input = AgentInput(content="Test", context=complex_context)
        assert agent_input.context == complex_context

    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self):
        """Test concurrent agent operations for thread safety."""

        class ConcurrentTestAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.execution_count = 0

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                self.execution_count += 1
                await asyncio.sleep(0.01)  # Simulate work

                return AgentOutput(
                    content=f"Execution #{self.execution_count}",
                    confidence=0.9,
                    processing_time=0.01,
                    agent_id=self.agent_id,
                )

        registry = AgentRegistry()
        registry.register("concurrent_agent")(ConcurrentTestAgent)

        # Get cached agent instance
        agent = registry.get_cached_agent("concurrent_agent", {"agent_id": "concurrent_agent"})

        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            agent_input = AgentInput(content=f"Request {i}")
            task = agent.process(agent_input)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Verify all executions completed
        assert len(results) == 5
        assert all(result.agent_id == "concurrent_agent" for result in results)
        assert agent.execution_count == 5

        # Clean up
        registry.clear()
