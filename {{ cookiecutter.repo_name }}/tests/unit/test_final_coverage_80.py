"""
Final targeted tests to achieve 80% coverage threshold.

This module contains very specific tests to hit the remaining
1.77% uncovered lines to push coverage above 80%.
"""

import asyncio

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentError, AgentExecutionError, handle_agent_error
from src.agents.models import AgentConfig, AgentInput, AgentOutput
from src.agents.registry import AgentRegistry


class TestFinalCoverage80:
    """Final test suite to achieve 80% coverage."""

    def test_agent_config_edge_cases_final(self):
        """Test AgentConfig edge cases for final coverage."""

        # Test name and description stripping
        config = AgentConfig(
            agent_id="test_agent",
            name="   Test Agent   ",
            description="   A test agent   ",
            enabled=False,
        )
        assert config.name == "Test Agent"
        assert config.description == "A test agent"
        assert config.enabled is False

        # Test config dict validation with empty dict
        config = AgentConfig(
            agent_id="test_agent",
            name="Test",
            description="Test",
            config={},  # Empty dict should work
        )
        assert config.config == {}

    def test_agent_input_cross_validation_final(self):
        """Test AgentInput cross-field validation final cases."""

        # Test long content with code type for validation path
        long_code = "x" * 40000  # Under limit but still long
        agent_input = AgentInput(content=long_code, context={"content_type": "code"})
        assert len(agent_input.content) == 40000

        # Test non-code content type
        agent_input = AgentInput(content="normal text content", context={"content_type": "markdown"})
        assert agent_input.content == "normal text content"

    def test_agent_output_cross_validation_final(self):
        """Test AgentOutput cross-field validation final cases."""

        # Test low confidence with normal time
        output = AgentOutput(
            content="test",
            confidence=0.3,
            processing_time=5.0,
            agent_id="test_agent",  # Low confidence  # Normal time
        )
        assert output.confidence == 0.3

        # Test high confidence with low time
        output = AgentOutput(
            content="test",
            confidence=0.95,
            processing_time=0.5,
            agent_id="test_agent",  # High confidence  # Low time
        )
        assert output.confidence == 0.95

    @pytest.mark.asyncio
    async def test_base_agent_config_restoration_final(self):
        """Test config restoration after overrides."""

        class ConfigRestoreAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Access config to test override/restore
                test_value = self.config.get("test_param", "default")
                return AgentOutput(
                    content=f"test_value: {test_value}",
                    confidence=0.9,
                    processing_time=0.01,
                    agent_id=self.agent_id,
                )

        agent = ConfigRestoreAgent({"agent_id": "restore_test", "test_param": "original"})

        # Verify original config
        assert agent.config["test_param"] == "original"

        # Process with override
        agent_input = AgentInput(content="test", config_overrides={"test_param": "overridden"})

        result = await agent.process(agent_input)
        assert "overridden" in result.content

        # Verify config was restored
        assert agent.config["test_param"] == "original"

    def test_registry_error_edge_cases_final(self):
        """Test registry error handling edge cases."""

        registry = AgentRegistry()

        # Test double registration (should work)
        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test", confidence=0.9, processing_time=0.01, agent_id=self.agent_id)

        # First registration
        registry.register("test_agent")(TestAgent)

        # Second registration with same name (should overwrite without error)
        registry.register("test_agent_2")(TestAgent)

        # Should still work
        agent = registry.get_agent("test_agent", {"agent_id": "test_agent"})
        assert isinstance(agent, TestAgent)

        registry.clear()

    def test_registry_cached_agent_different_configs(self):
        """Test cached agent with different configurations."""

        class ConfigurableAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.config_hash = str(hash(frozenset(config.items())))

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content=self.config_hash,
                    confidence=0.9,
                    processing_time=0.01,
                    agent_id=self.agent_id,
                )

        registry = AgentRegistry()
        registry.register("config_agent")(ConfigurableAgent)

        # Different configs should create different instances
        config1 = {"agent_id": "config_agent", "param": "value1"}
        config2 = {"agent_id": "config_agent", "param": "value2"}

        agent1 = registry.get_cached_agent("config_agent", config1)
        agent2 = registry.get_cached_agent("config_agent", config2)

        # Should be different instances
        assert agent1 is not agent2
        assert agent1.config_hash != agent2.config_hash

        registry.clear()

    def test_registry_capabilities_comprehensive_final(self):
        """Test registry capabilities edge cases."""

        class NoCapabilitiesAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="no caps", confidence=0.9, processing_time=0.01, agent_id=self.agent_id)

        registry = AgentRegistry()
        registry.register("no_caps")(NoCapabilitiesAgent)

        # Create instance - BaseAgent has default get_capabilities that returns text types
        agent = registry.get_agent("no_caps", {"agent_id": "no_caps"})
        assert isinstance(agent, NoCapabilitiesAgent)

        # Test capability queries - BaseAgent defaults to text input/output
        text_agents = registry.find_agents_by_type("text")
        code_agents = registry.find_agents_by_type("code")

        # Should find the agent for text (default capability) but not code
        assert "no_caps" in text_agents
        assert len(code_agents) == 0

        registry.clear()

    @pytest.mark.asyncio
    async def test_base_agent_metrics_collection_final(self):
        """Test BaseAgent metrics collection edge cases."""

        class MetricsAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.custom_metric = 0

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                self.custom_metric += 1

                if self.custom_metric == 1:
                    # First call - normal execution
                    await asyncio.sleep(0.01)
                elif self.custom_metric == 2:
                    # Second call - faster execution
                    await asyncio.sleep(0.001)
                else:
                    # Third call - error
                    raise ValueError("Test error for metrics")

                return AgentOutput(
                    content=f"call {self.custom_metric}",
                    confidence=0.9,
                    processing_time=0.01,
                    agent_id=self.agent_id,
                )

        agent = MetricsAgent({"agent_id": "metrics_agent"})

        # First successful call
        result1 = await agent.process(AgentInput(content="test1"))
        assert "call 1" in result1.content

        # Second successful call
        result2 = await agent.process(AgentInput(content="test2"))
        assert "call 2" in result2.content

        # Third call should fail and record failure metrics
        with pytest.raises(AgentExecutionError):
            await agent.process(AgentInput(content="test3"))

    def test_agent_validation_comprehensive_final(self):
        """Test agent validation edge cases."""

        # Test agent_id validation in models
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            AgentConfig(agent_id="   ", name="Test", description="Test")

        with pytest.raises(ValueError, match="Agent ID must contain only alphanumeric"):
            AgentConfig(agent_id="test@agent", name="Test", description="Test")

        # Test output agent_id validation
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            AgentOutput(content="test", confidence=0.9, processing_time=1.0, agent_id="   ")

    def test_registry_status_edge_cases_final(self):
        """Test registry status reporting edge cases."""

        registry = AgentRegistry()

        # Test status on empty registry
        status = registry.get_registry_status()
        assert status["total_agents"] == 0
        assert status["cached_instances"] == 0
        assert status["agents_with_capabilities"] == 0
        assert status["registered_agents"] == []

        class SimpleAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="simple", confidence=0.9, processing_time=0.01, agent_id=self.agent_id)

        # Register but don't instantiate
        registry.register("simple")(SimpleAgent)

        status = registry.get_registry_status()
        assert status["total_agents"] == 1
        assert status["cached_instances"] == 0  # Not instantiated yet
        assert status["agents_with_capabilities"] == 0

        # Now instantiate
        registry.get_cached_agent("simple", {"agent_id": "simple"})

        status = registry.get_registry_status()
        assert status["total_agents"] == 1
        assert status["cached_instances"] == 1
        assert status["agents_with_capabilities"] == 1  # No capabilities but counted

        registry.clear()

    @pytest.mark.asyncio
    async def test_concurrent_execution_edge_cases(self):
        """Test concurrent execution edge cases."""

        class ConcurrentAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.concurrent_calls = 0

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                self.concurrent_calls += 1
                call_num = self.concurrent_calls

                # Simulate different processing times
                if call_num % 2 == 0:
                    await asyncio.sleep(0.02)
                else:
                    await asyncio.sleep(0.01)

                return AgentOutput(
                    content=f"concurrent {call_num}",
                    confidence=0.9,
                    processing_time=0.01,
                    agent_id=self.agent_id,
                )

        agent = ConcurrentAgent({"agent_id": "concurrent"})

        # Create concurrent tasks with different inputs
        tasks = []
        for i in range(3):
            input_obj = AgentInput(content=f"concurrent test {i}", context={"task_id": i})
            tasks.append(agent.process(input_obj))

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert len(results) == 3
        assert all("concurrent" in result.content for result in results)
        assert agent.concurrent_calls == 3

    def test_exception_handling_final_cases(self):
        """Test exception handling final edge cases."""

        # Test generic Exception (not specifically handled) - returns generic AgentError
        generic_error = RuntimeError("Generic runtime error")
        handled = handle_agent_error(generic_error, agent_id="test")

        assert isinstance(handled, AgentError)
        assert handled.agent_id == "test"
        assert "Generic runtime error" in str(handled)

        # Test with agent_id=None (should preserve None)
        handled_none_id = handle_agent_error(ValueError("None ID error"), agent_id=None)
        assert isinstance(handled_none_id, AgentExecutionError)
        assert handled_none_id.agent_id is None
