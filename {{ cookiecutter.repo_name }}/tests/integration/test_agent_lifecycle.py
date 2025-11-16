"""
Integration tests for complete agent lifecycle.

This module tests the integration between all agent system components,
validating the complete flow from registration to execution following
the testing guide requirements.
"""

import asyncio
import time
from typing import Any

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import (
    AgentConfigurationError,
    AgentExecutionError,
    AgentRegistrationError,
    AgentTimeoutError,
)
from src.agents.models import AgentInput, AgentOutput
from src.agents.registry import AgentRegistry, agent_registry


class TestAgentLifecycleIntegration:
    """Integration tests for complete agent lifecycle."""

    @pytest.fixture
    def integration_registry(self):
        """Fresh registry for integration testing."""
        registry = AgentRegistry()
        yield registry
        registry.clear()

    @pytest.fixture
    def sample_integration_agent(self):
        """Sample agent class for integration testing."""

        class IntegrationTestAgent(BaseAgent):
            def __init__(self, config: dict[str, Any]):
                # Set custom attributes before parent init
                self.processing_history = []
                self.call_count = 0
                super().__init__(config)

            def _validate_configuration(self) -> None:
                super()._validate_configuration()
                # Custom validation for integration testing
                if self.config.get("require_special_param") and not self.config.get("special_param"):
                    raise AgentConfigurationError(
                        message="special_param is required when require_special_param is True",
                        error_code="MISSING_SPECIAL_PARAM",
                        agent_id=self.agent_id,
                    )

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                self.call_count += 1

                # Simulate processing based on input
                processing_time = self.config.get("processing_time", 0.1)
                await asyncio.sleep(processing_time)

                # Record processing history
                self.processing_history.append(
                    {
                        "request_id": agent_input.request_id,
                        "content_length": len(agent_input.content),
                        "context": agent_input.context,
                        "call_number": self.call_count,
                    },
                )

                # Generate response based on operation
                operation = agent_input.context.get("operation", "default") if agent_input.context else "default"

                if operation == "error":
                    raise ValueError("Simulated processing error")
                if operation == "timeout":
                    await asyncio.sleep(10)  # Will timeout

                response_content = f"Integration test response #{self.call_count}: {operation}"
                confidence = self.config.get("confidence", 0.9)

                return AgentOutput(
                    content=response_content,
                    metadata={
                        "operation": operation,
                        "call_count": self.call_count,
                        "processing_history_length": len(self.processing_history),
                    },
                    confidence=confidence,
                    processing_time=processing_time,
                    agent_id=self.agent_id,
                    request_id=agent_input.request_id,
                )

            def get_capabilities(self) -> dict[str, Any]:
                return {
                    "input_types": ["text"],
                    "output_types": ["text", "analysis"],
                    "operations": ["default", "analyze", "transform"],
                    "stateful": True,
                    "supports_history": True,
                }

        return IntegrationTestAgent

    @pytest.mark.integration
    def test_complete_agent_registration_and_retrieval(self, integration_registry, sample_integration_agent):
        """Test complete agent registration and retrieval lifecycle."""
        # Register agent
        decorated_class = integration_registry.register("integration_agent")(sample_integration_agent)

        # Verify registration
        assert decorated_class == sample_integration_agent
        assert "integration_agent" in integration_registry
        assert len(integration_registry) == 1

        # Retrieve agent class
        retrieved_class = integration_registry.get_agent_class("integration_agent")
        assert retrieved_class == sample_integration_agent

        # Get agent instance
        config = {"agent_id": "integration_agent", "processing_time": 0.05, "confidence": 0.95}
        agent = integration_registry.get_agent("integration_agent", config)

        # Verify agent instance
        assert isinstance(agent, sample_integration_agent)
        assert agent.agent_id == "integration_agent"
        assert agent.config["processing_time"] == 0.05
        assert agent.config["confidence"] == 0.95

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_agent_execution_flow(self, integration_registry, sample_integration_agent):
        """Test complete agent execution flow from registration to response."""
        # Register agent
        integration_registry.register("execution_agent")(sample_integration_agent)

        # Get agent instance
        config = {"agent_id": "execution_agent", "processing_time": 0.02, "confidence": 0.88}
        agent = integration_registry.get_agent("execution_agent", config)

        # Create test input
        agent_input = AgentInput(
            content="Test integration execution",
            context={"operation": "analyze"},
            config_overrides={"confidence": 0.92},
        )

        # Execute agent
        result = await agent.process(agent_input)

        # Verify execution result
        assert result.content == "Integration test response #1: analyze"
        assert result.confidence == 0.92  # Should use override
        assert result.agent_id == "execution_agent"
        assert result.request_id == agent_input.request_id
        assert result.metadata["operation"] == "analyze"
        assert result.metadata["call_count"] == 1

        # Verify agent state
        assert agent.call_count == 1
        assert len(agent.processing_history) == 1
        assert agent.processing_history[0]["request_id"] == agent_input.request_id

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_caching_and_reuse(self, integration_registry, sample_integration_agent):
        """Test agent caching and reuse across multiple requests."""
        # Register agent
        integration_registry.register("cached_agent")(sample_integration_agent)

        # Configuration
        config = {"agent_id": "cached_agent", "processing_time": 0.01, "confidence": 0.9}

        # Get cached agent instances
        agent1 = integration_registry.get_cached_agent("cached_agent", config)
        agent2 = integration_registry.get_cached_agent("cached_agent", config)

        # Should be same instance
        assert agent1 is agent2

        # Execute multiple requests
        inputs = [
            AgentInput(content="Request 1", context={"operation": "analyze"}),
            AgentInput(content="Request 2", context={"operation": "transform"}),
            AgentInput(content="Request 3", context={"operation": "default"}),
        ]

        results = []
        for agent_input in inputs:
            result = await agent1.process(agent_input)
            results.append(result)

        # Verify all requests processed by same agent instance
        assert len(results) == 3
        assert all(result.agent_id == "cached_agent" for result in results)
        assert agent1.call_count == 3
        assert len(agent1.processing_history) == 3

        # Verify call ordering
        for i, result in enumerate(results, 1):
            assert result.metadata["call_count"] == i

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_capability_discovery_and_matching(self, integration_registry, sample_integration_agent):
        """Test agent capability discovery and matching integration."""

        # Register multiple agents with different capabilities
        class AnalysisAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content="Analysis result",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

            def get_capabilities(self) -> dict[str, Any]:
                return {
                    "input_types": ["text", "code"],
                    "output_types": ["analysis", "report"],
                    "specialization": "security_analysis",
                    "supports_batch": True,
                }

        class TransformAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content="Transformed content",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

            def get_capabilities(self) -> dict[str, Any]:
                return {
                    "input_types": ["text"],
                    "output_types": ["text", "formatted"],
                    "specialization": "content_transformation",
                    "supports_batch": False,
                }

        # Register agents
        integration_registry.register("analysis_agent")(AnalysisAgent)
        integration_registry.register("transform_agent")(TransformAgent)
        integration_registry.register("integration_agent")(sample_integration_agent)

        # Create instances to populate capabilities
        integration_registry.get_agent("analysis_agent", {"agent_id": "analysis_agent"})
        integration_registry.get_agent("transform_agent", {"agent_id": "transform_agent"})
        integration_registry.get_agent("integration_agent", {"agent_id": "integration_agent"})

        # Test capability discovery
        text_agents = integration_registry.find_agents_by_type("text")
        code_agents = integration_registry.find_agents_by_type("code")
        analysis_output_agents = integration_registry.find_agents_by_type("text", "analysis")

        # Verify capability matching
        assert set(text_agents) == {"analysis_agent", "transform_agent", "integration_agent"}
        assert code_agents == ["analysis_agent"]
        assert set(analysis_output_agents) == {"analysis_agent", "integration_agent"}

        # Test custom capability filtering
        security_agents = integration_registry.find_agents_by_capability("specialization", "security_analysis")
        batch_agents = integration_registry.find_agents_by_capability("supports_batch", True)
        stateful_agents = integration_registry.find_agents_by_capability("stateful", True)

        assert security_agents == ["analysis_agent"]
        assert batch_agents == ["analysis_agent"]
        assert stateful_agents == ["integration_agent"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_configuration_validation_flow(self, integration_registry, sample_integration_agent):
        """Test agent configuration validation throughout the lifecycle."""
        # Register agent
        integration_registry.register("validation_agent")(sample_integration_agent)

        # Test successful configuration
        valid_config = {
            "agent_id": "validation_agent",
            "require_special_param": True,
            "special_param": "required_value",
        }

        agent = integration_registry.get_agent("validation_agent", valid_config)
        assert agent.config["special_param"] == "required_value"

        # Test configuration validation failure
        invalid_config = {
            "agent_id": "validation_agent",
            "require_special_param": True,
            # Missing special_param
        }

        with pytest.raises(AgentRegistrationError) as excinfo:
            integration_registry.get_agent("validation_agent", invalid_config)

        assert "Failed to instantiate agent" in str(excinfo.value)
        assert excinfo.value.error_code == "INSTANTIATION_FAILED"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_error_handling_integration(self, integration_registry, sample_integration_agent):
        """Test error handling integration across the system."""
        # Register agent
        integration_registry.register("error_agent")(sample_integration_agent)

        # Get agent instance
        config = {"agent_id": "error_agent"}
        agent = integration_registry.get_agent("error_agent", config)

        # Test execution error
        error_input = AgentInput(content="Test error handling", context={"operation": "error"})

        with pytest.raises(AgentExecutionError) as excinfo:
            await agent.process(error_input)

        assert "Simulated processing error" in str(excinfo.value)
        assert excinfo.value.agent_id == "error_agent"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_timeout_handling_integration(self, integration_registry, sample_integration_agent):
        """Test timeout handling integration."""
        # Register agent
        integration_registry.register("timeout_agent")(sample_integration_agent)

        # Get agent instance with short timeout
        config = {"agent_id": "timeout_agent", "timeout": 0.1}
        agent = integration_registry.get_agent("timeout_agent", config)

        # Test timeout scenario
        timeout_input = AgentInput(content="Test timeout handling", context={"operation": "timeout"})

        with pytest.raises(AgentTimeoutError) as excinfo:
            await agent.process(timeout_input)

        assert "Agent execution timed out" in str(excinfo.value)
        assert excinfo.value.agent_id == "timeout_agent"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, integration_registry, sample_integration_agent):
        """Test concurrent agent execution and thread safety."""
        # Register agent
        integration_registry.register("concurrent_agent")(sample_integration_agent)

        # Get agent instance
        config = {"agent_id": "concurrent_agent", "processing_time": 0.05}
        agent = integration_registry.get_cached_agent("concurrent_agent", config)

        # Create multiple inputs
        inputs = [AgentInput(content=f"Concurrent request {i}", context={"operation": "analyze"}) for i in range(5)]

        # Execute concurrently
        tasks = [agent.process(agent_input) for agent_input in inputs]
        results = await asyncio.gather(*tasks)

        # Verify all executions completed
        assert len(results) == 5
        assert all(result.agent_id == "concurrent_agent" for result in results)
        assert agent.call_count == 5
        assert len(agent.processing_history) == 5

        # Verify unique request IDs
        request_ids = {result.request_id for result in results}
        assert len(request_ids) == 5

    @pytest.mark.integration
    def test_agent_registry_status_and_cleanup(self, integration_registry, sample_integration_agent):
        """Test registry status reporting and cleanup integration."""
        # Register multiple agents
        integration_registry.register("agent1")(sample_integration_agent)
        integration_registry.register("agent2")(sample_integration_agent)

        # Create cached instances
        integration_registry.get_cached_agent("agent1", {"agent_id": "agent1"})
        integration_registry.get_cached_agent("agent2", {"agent_id": "agent2"})

        # Check status
        status = integration_registry.get_registry_status()
        assert status["total_agents"] == 2
        assert status["cached_instances"] == 2
        assert status["agents_with_capabilities"] == 2
        assert set(status["registered_agents"]) == {"agent1", "agent2"}

        # Test selective cleanup
        integration_registry.unregister("agent1")

        # Verify cleanup
        assert len(integration_registry) == 1
        assert "agent1" not in integration_registry
        assert "agent2" in integration_registry

        # Test complete cleanup
        integration_registry.clear()
        assert len(integration_registry) == 0
        assert integration_registry.get_registry_status()["total_agents"] == 0

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_performance_integration(self, integration_registry, sample_integration_agent):
        """Test agent performance characteristics in integration."""
        # Register agent
        integration_registry.register("perf_agent")(sample_integration_agent)

        # Get agent instance
        config = {"agent_id": "perf_agent", "processing_time": 0.01}
        agent = integration_registry.get_cached_agent("perf_agent", config)

        # Test processing time accuracy
        agent_input = AgentInput(content="Performance test")

        start_time = time.time()
        result = await agent.process(agent_input)
        end_time = time.time()

        # Verify timing
        actual_time = end_time - start_time
        assert actual_time >= 0.01  # Should be at least processing time
        assert actual_time < 0.1  # Should not be too much overhead
        assert result.processing_time >= 0.01

    @pytest.mark.integration
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_agent_security_integration(
        self,
        integration_registry,
        sample_integration_agent,
        security_test_inputs,
    ):
        """Test agent security handling in integration."""
        # Register agent
        integration_registry.register("security_agent")(sample_integration_agent)

        # Get agent instance
        config = {"agent_id": "security_agent"}
        agent = integration_registry.get_agent("security_agent", config)

        # Test security inputs
        for malicious_input in security_test_inputs[:5]:  # Test subset for integration
            agent_input = AgentInput(content=malicious_input, context={"operation": "analyze"})

            # Should process without errors
            result = await agent.process(agent_input)
            assert result.content.startswith("Integration test response")
            assert result.agent_id == "security_agent"

            # Verify malicious content doesn't cause issues
            assert "alert" not in result.content or malicious_input in result.content

    @pytest.mark.integration
    def test_global_registry_integration(self, sample_integration_agent):
        """Test integration with global registry instance."""
        # Use global registry

        # Store original state
        original_agents = list(agent_registry.list_agents())

        try:
            # Register agent in global registry
            agent_registry.register("global_agent")(sample_integration_agent)

            # Verify registration
            assert "global_agent" in agent_registry

            # Get agent instance
            config = {"agent_id": "global_agent"}
            agent = agent_registry.get_agent("global_agent", config)

            # Verify agent works
            assert isinstance(agent, sample_integration_agent)
            assert agent.agent_id == "global_agent"

        finally:
            # Cleanup global registry
            if "global_agent" in agent_registry:
                agent_registry.unregister("global_agent")

            # Verify cleanup
            assert agent_registry.list_agents() == original_agents

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_lifecycle_with_config_overrides(self, integration_registry, sample_integration_agent):
        """Test complete agent lifecycle with configuration overrides."""
        # Register agent
        integration_registry.register("override_agent")(sample_integration_agent)

        # Base configuration
        base_config = {"agent_id": "override_agent", "confidence": 0.8, "processing_time": 0.05}

        # Get agent instance
        agent = integration_registry.get_agent("override_agent", base_config)

        # Create input with config overrides
        agent_input = AgentInput(
            content="Test config overrides",
            context={"operation": "analyze"},
            config_overrides={"confidence": 0.95, "processing_time": 0.02},
        )

        # Execute with overrides
        result = await agent.process(agent_input)

        # Verify overrides were applied
        assert result.confidence == 0.95
        assert result.processing_time >= 0.02

        # Verify base config unchanged
        assert agent.config["confidence"] == 0.8
        assert agent.config["processing_time"] == 0.05

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_batch_processing_integration(self, integration_registry, sample_integration_agent):
        """Test batch processing integration across multiple agents."""
        # Register multiple agents
        integration_registry.register("batch_agent1")(sample_integration_agent)
        integration_registry.register("batch_agent2")(sample_integration_agent)

        # Get agent instances
        agent1 = integration_registry.get_agent("batch_agent1", {"agent_id": "batch_agent1"})
        agent2 = integration_registry.get_agent("batch_agent2", {"agent_id": "batch_agent2"})

        # Create batch inputs
        batch_inputs = [AgentInput(content=f"Batch item {i}", context={"batch_id": i}) for i in range(3)]

        # Process batch with both agents
        agent1_tasks = [agent1.process(inp) for inp in batch_inputs]
        agent2_tasks = [agent2.process(inp) for inp in batch_inputs]

        # Execute all tasks concurrently
        all_results = await asyncio.gather(*agent1_tasks, *agent2_tasks)

        # Verify batch processing
        assert len(all_results) == 6

        # Verify agent1 results
        agent1_results = all_results[:3]
        assert all(result.agent_id == "batch_agent1" for result in agent1_results)

        # Verify agent2 results
        agent2_results = all_results[3:]
        assert all(result.agent_id == "batch_agent2" for result in agent2_results)

        # Verify processing state
        assert agent1.call_count == 3
        assert agent2.call_count == 3
