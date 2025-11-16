"""
Final push to 80% coverage by targeting specific uncovered lines.

This module contains very specific tests designed to hit the remaining
uncovered lines to push coverage from 78.21% to above 80%.
"""

import asyncio

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentConfigurationError, AgentExecutionError, AgentTimeoutError
from src.agents.models import AgentConfig, AgentInput, AgentOutput
from src.agents.registry import AgentRegistry
from src.utils.observability import log_agent_event


class TestFinal80Push:
    """Test suite to push coverage to 80%."""

    def test_base_agent_initialization_edge_cases(self):
        """Test BaseAgent initialization edge cases for missing coverage."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id=self.agent_id)

        # Test with non-dict config to hit line 116-120
        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent("not_a_dict")

        assert excinfo.value.error_code == "INVALID_CONFIG_TYPE"
        assert "Configuration must be a dictionary" in str(excinfo.value)

    def test_base_agent_validate_configuration_custom(self):
        """Test BaseAgent _validate_configuration method."""

        class CustomValidationAgent(BaseAgent):
            def _validate_configuration(self) -> None:
                super()._validate_configuration()
                # Add custom validation that can fail
                if self.config.get("fail_validation"):
                    raise AgentConfigurationError(
                        message="Custom validation failed",
                        error_code="CUSTOM_VALIDATION_ERROR",
                        agent_id=self.agent_id,
                    )

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id=self.agent_id)

        # Test successful custom validation
        agent = CustomValidationAgent({"agent_id": "test_agent"})
        assert agent.agent_id == "test_agent"

        # Test failed custom validation
        with pytest.raises(AgentConfigurationError) as excinfo:
            CustomValidationAgent({"agent_id": "test_agent", "fail_validation": True})

        assert excinfo.value.error_code == "CUSTOM_VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_base_agent_execute_with_timeout_config_override(self):
        """Test BaseAgent timeout configuration override."""

        class SlowAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                await asyncio.sleep(0.5)  # Sleep for 500ms
                return AgentOutput(content="slow result", confidence=0.9, processing_time=0.5, agent_id=self.agent_id)

        agent = SlowAgent({"agent_id": "slow_agent", "timeout": 1.0})

        # Test with config override timeout that's shorter
        agent_input = AgentInput(
            content="Test timeout override",
            config_overrides={"timeout": 0.1},  # 100ms timeout, should timeout
        )

        with pytest.raises(AgentTimeoutError) as excinfo:
            await agent.process(agent_input)

        assert excinfo.value.agent_id == "slow_agent"

    def test_agent_input_model_validator_edge_cases(self):
        """Test AgentInput model validator edge cases."""

        # Test cross-field validation with code content type but normal length
        agent_input = AgentInput(content="short code", context={"content_type": "code"})
        assert agent_input.content == "short code"

        # Test cross-field validation with non-code content type
        agent_input = AgentInput(content="normal content", context={"content_type": "text"})
        assert agent_input.content == "normal content"

        # Test cross-field validation with code content exceeding limit
        long_code = "x" * 50001
        with pytest.raises(ValueError, match="Code content cannot exceed 50,000 characters"):
            AgentInput(content=long_code, context={"content_type": "code"})

    def test_agent_output_model_validator_edge_cases(self):
        """Test AgentOutput model validator edge cases."""

        # Test high processing time with high confidence (line 262-269)
        output = AgentOutput(
            content="test",
            confidence=0.96,  # High confidence
            processing_time=35.0,  # High processing time
            agent_id="test_agent",
        )
        # Should not raise error but triggers the validation logic
        assert output.confidence == 0.96
        assert output.processing_time == 35.0

        # Test normal processing time with high confidence
        output = AgentOutput(
            content="test",
            confidence=0.96,
            processing_time=5.0,
            agent_id="test_agent",  # Normal processing time
        )
        assert output.confidence == 0.96

    def test_registry_find_agents_by_capability_edge_cases(self):
        """Test registry capability finding edge cases."""

        class CapabilityAgent(BaseAgent):
            def __init__(self, config):
                self.test_capability = config.get("test_capability", "default")
                super().__init__(config)

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id=self.agent_id)

            def get_capabilities(self):
                return {"test_feature": self.test_capability, "supports_advanced": True, "version": 1.0}

        registry = AgentRegistry()
        registry.register("capability_agent")(CapabilityAgent)

        # Create instance to populate capabilities
        registry.get_agent("capability_agent", {"agent_id": "capability_agent", "test_capability": "advanced"})

        # Test finding by capability with exact match
        agents = registry.find_agents_by_capability("test_feature", "advanced")
        assert "capability_agent" in agents

        # Test finding by capability with no match
        agents = registry.find_agents_by_capability("test_feature", "nonexistent")
        assert len(agents) == 0

        # Test finding by capability that doesn't exist
        agents = registry.find_agents_by_capability("nonexistent_feature", "value")
        assert len(agents) == 0

        # Clean up
        registry.clear()

    def test_registry_cached_agent_edge_cases(self):
        """Test registry cached agent edge cases."""

        class CacheTestAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.instance_id = config.get("instance_id", "default")

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content=f"instance {self.instance_id}",
                    confidence=0.9,
                    processing_time=1.0,
                    agent_id=self.agent_id,
                )

        registry = AgentRegistry()
        registry.register("cache_agent")(CacheTestAgent)

        # Test caching with different configurations creates different instances
        config1 = {"agent_id": "cache_agent", "instance_id": "first"}
        config2 = {"agent_id": "cache_agent", "instance_id": "second"}

        agent1 = registry.get_cached_agent("cache_agent", config1)
        agent2 = registry.get_cached_agent("cache_agent", config2)

        # Should be different instances due to different configs
        assert agent1 is not agent2
        assert agent1.instance_id == "first"
        assert agent2.instance_id == "second"

        # Test that we can get instances with same agent_id
        agent1_again = registry.get_cached_agent("cache_agent", config1)
        assert agent1_again.agent_id == "cache_agent"
        assert agent1_again.instance_id == "first"

        # Clean up
        registry.clear()

    @pytest.mark.asyncio
    async def test_base_agent_metrics_edge_cases(self):
        """Test BaseAgent metrics collection edge cases."""

        class MetricsAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate different processing scenarios
                processing_time = agent_input.context.get("processing_time", 0.1) if agent_input.context else 0.1
                await asyncio.sleep(processing_time)

                if agent_input.context and agent_input.context.get("fail"):
                    raise ValueError("Simulated failure")

                return AgentOutput(
                    content="metrics test",
                    confidence=0.9,
                    processing_time=processing_time,
                    agent_id=self.agent_id,
                )

        agent = MetricsAgent({"agent_id": "metrics_agent"})

        # Test successful execution (increments success metrics)
        agent_input = AgentInput(content="success test", context={"processing_time": 0.05})
        result = await agent.process(agent_input)
        assert result.content == "metrics test"

        # Test failed execution (increments failure metrics)
        agent_input = AgentInput(content="failure test", context={"fail": True})
        with pytest.raises(AgentExecutionError, match="Simulated failure"):
            await agent.process(agent_input)

    def test_agent_input_content_validation_edge_cases(self):
        """Test AgentInput content validation edge cases."""

        # Test content with only whitespace
        with pytest.raises(ValueError, match="Content cannot be empty or whitespace-only"):
            AgentInput(content="   \t\n   ")

        # Test content with leading/trailing whitespace (should be stripped)
        agent_input = AgentInput(content="  test content  ")
        assert agent_input.content == "test content"

    def test_agent_output_content_validation_edge_cases(self):
        """Test AgentOutput content validation edge cases."""

        # Test content with only whitespace
        with pytest.raises(ValueError, match="Content cannot be empty or whitespace-only"):
            AgentOutput(content="   \t\n   ", confidence=0.9, processing_time=1.0, agent_id="test_agent")

        # Test content with leading/trailing whitespace (should be stripped)
        output = AgentOutput(content="  test output  ", confidence=0.9, processing_time=1.0, agent_id="test_agent")
        assert output.content == "test output"

    def test_agent_config_validation_edge_cases(self):
        """Test AgentConfig validation edge cases."""

        # Test agent_id with leading/trailing whitespace (should be stripped)
        config = AgentConfig(agent_id="  test_agent  ", name="  Test Agent  ", description="  A test agent  ")
        assert config.agent_id == "test_agent"
        assert config.name == "Test Agent"
        assert config.description == "A test agent"

    def test_observability_coverage_boost(self):
        """Test observability functions to boost coverage."""

        # Test basic event logging
        log_agent_event(event_type="test_event", agent_id="test_agent", message="Test message", level="info")

        # Test event logging with all parameters
        log_agent_event(
            event_type="complex_event",
            agent_id="test_agent",
            message="Complex message",
            level="error",
            request_id="test-request",
            processing_time=1.5,
            confidence=0.9,
            content_length=100,
            metadata_keys=["key1", "key2"],
            error_code="TEST_ERROR",
            error_type="ValueError",
            error_message="Test error",
        )

    @pytest.mark.asyncio
    async def test_base_agent_process_comprehensive(self):
        """Test BaseAgent process method comprehensively."""

        class ProcessTestAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.process_count = 0

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                self.process_count += 1

                # Test different processing scenarios
                if agent_input.context:
                    scenario = agent_input.context.get("scenario", "normal")
                    if scenario == "slow":
                        await asyncio.sleep(0.1)
                    elif scenario == "error":
                        raise ValueError("Test error")

                return AgentOutput(
                    content=f"Process #{self.process_count}",
                    confidence=0.9,
                    processing_time=0.05,
                    agent_id=self.agent_id,
                )

        agent = ProcessTestAgent({"agent_id": "process_agent"})

        # Test normal processing
        result = await agent.process(AgentInput(content="normal"))
        assert result.content == "Process #1"
        assert result.agent_id == "process_agent"

        # Test processing with context
        result = await agent.process(AgentInput(content="with context", context={"scenario": "slow"}))
        assert result.content == "Process #2"

        # Test processing with config overrides
        result = await agent.process(AgentInput(content="with overrides", config_overrides={"custom_param": "value"}))
        assert result.content == "Process #3"
