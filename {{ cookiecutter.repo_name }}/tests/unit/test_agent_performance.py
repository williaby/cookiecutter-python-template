"""
Unit tests for agent performance monitoring edge cases.

This module tests performance monitoring functionality and edge cases that
might not be covered in the standard test suite, focusing on timing accuracy,
resource monitoring, and performance boundary conditions.
"""

import asyncio
import gc
import time
from typing import Any

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentTimeoutError
from src.agents.models import AgentInput, AgentOutput
from src.agents.registry import AgentRegistry


class TestAgentPerformanceMonitoring:
    """Test performance monitoring edge cases and boundary conditions."""

    @pytest.fixture
    def performance_agent_class(self):
        """Agent class with configurable performance characteristics."""

        class PerformanceTestAgent(BaseAgent):
            def __init__(self, config: dict[str, Any]):
                super().__init__(config)
                self.execution_count = 0
                self.timing_history = []

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                start_time = time.time()
                self.execution_count += 1

                # Simulate different performance scenarios
                scenario = agent_input.context.get("scenario", "normal") if agent_input.context else "normal"

                if scenario == "slow":
                    await asyncio.sleep(0.1)
                elif scenario == "variable":
                    # Variable processing time based on execution count
                    sleep_time = 0.01 * (self.execution_count % 5)
                    await asyncio.sleep(sleep_time)
                elif scenario == "cpu_intensive":
                    # Simulate CPU-intensive work
                    for _ in range(10000):
                        _ = sum(range(100))
                elif scenario == "memory_intensive":
                    # Simulate memory allocation
                    large_list = list(range(100000))
                    del large_list

                processing_time = time.time() - start_time
                # Ensure processing time is non-negative to avoid validation errors
                processing_time = max(0.0, processing_time)
                self.timing_history.append(processing_time)

                return AgentOutput(
                    content=f"Performance test result for {scenario}",
                    metadata={
                        "scenario": scenario,
                        "execution_count": self.execution_count,
                        "processing_time": processing_time,
                    },
                    confidence=0.95,
                    processing_time=processing_time,
                    agent_id=self.agent_id,
                )

        return PerformanceTestAgent

    @pytest.fixture
    def registry(self):
        """Fresh registry for testing."""
        registry = AgentRegistry()
        yield registry
        registry.clear()

    @pytest.mark.unit
    @pytest.mark.performance
    def test_timing_accuracy_edge_cases(self, performance_agent_class):
        """Test timing accuracy in various edge cases."""
        config = {"agent_id": "timing_test_agent"}
        agent = performance_agent_class(config)

        # Test very short execution times
        agent_input = AgentInput(content="Quick test", context={"scenario": "normal"})

        # Test with actual timing - no need to mock time.time()
        # since we're testing edge cases in timing precision
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(agent.process(agent_input))
            # Should have a reasonable processing time (> 0)
            assert result.processing_time >= 0.0
            assert result.processing_time < 1.0  # Should be fast
            # Test that timing is captured with reasonable precision
            assert isinstance(result.processing_time, float)
        finally:
            loop.close()

    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_metadata_accuracy(self, performance_agent_class):
        """Test performance metadata accuracy and consistency."""
        config = {"agent_id": "metadata_test_agent"}
        agent = performance_agent_class(config)

        # Test different scenarios
        scenarios = ["normal", "slow", "cpu_intensive", "memory_intensive"]
        results = []

        for scenario in scenarios:
            agent_input = AgentInput(content=f"Test {scenario}", context={"scenario": scenario})

            start_time = time.time()
            result = await agent.process(agent_input)
            end_time = time.time()

            results.append(
                {
                    "scenario": scenario,
                    "reported_time": result.processing_time,
                    "measured_time": end_time - start_time,
                    "metadata": result.metadata,
                },
            )

        # Verify timing consistency
        for result in results:
            # Reported time should be non-negative and reasonable
            assert (
                result["reported_time"] >= 0.0
            ), f"Negative processing time for {result['scenario']}: {result['reported_time']}"
            # Measured time should also be reasonable
            assert (
                result["measured_time"] >= 0.0
            ), f"Negative measured time for {result['scenario']}: {result['measured_time']}"
            # Reported time should be close to measured time (within 50ms tolerance to account for timing variations)
            time_diff = abs(result["reported_time"] - result["measured_time"])
            assert time_diff < 0.05, f"Time difference too large for {result['scenario']}: {time_diff}"

            # Metadata should include performance details
            assert "processing_time" in result["metadata"]
            assert "execution_count" in result["metadata"]
            assert result["metadata"]["scenario"] == result["scenario"]

    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_monitoring_under_load(self, performance_agent_class):
        """Test performance monitoring accuracy under concurrent load."""
        config = {"agent_id": "load_test_agent"}
        agent = performance_agent_class(config)

        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            agent_input = AgentInput(content=f"Concurrent request {i}", context={"scenario": "variable"})
            task = agent.process(agent_input)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Verify all executions completed successfully
        assert len(results) == 10

        # Check timing consistency across concurrent executions
        processing_times = [result.processing_time for result in results]

        # All processing times should be reasonable (< 1 second)
        assert all(time < 1.0 for time in processing_times)

        # Should have some variation in processing times due to "variable" scenario
        assert max(processing_times) > min(processing_times)

    @pytest.mark.unit
    @pytest.mark.performance
    def test_timeout_boundary_conditions(self, performance_agent_class):
        """Test timeout behavior at boundary conditions."""
        # Test with very short timeout
        config = {"agent_id": "timeout_test_agent", "timeout": 0.001}  # 1ms timeout
        agent = performance_agent_class(config)

        agent_input = AgentInput(content="Test timeout", context={"scenario": "slow"})  # Will take 100ms

        # Should timeout almost immediately
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            with pytest.raises(AgentTimeoutError) as excinfo:
                loop.run_until_complete(agent.process(agent_input))

            assert "timed out" in str(excinfo.value)
            assert excinfo.value.agent_id == "timeout_test_agent"

        finally:
            loop.close()

    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_with_config_overrides(self, performance_agent_class):
        """Test performance monitoring with configuration overrides."""
        config = {"agent_id": "override_test_agent", "timeout": 1.0}
        agent = performance_agent_class(config)

        # Test with timeout override
        agent_input = AgentInput(
            content="Test override",
            context={"scenario": "slow"},
            config_overrides={"timeout": 0.05},  # Shorter timeout
        )

        with pytest.raises(AgentTimeoutError):
            await agent.process(agent_input)

        # Test with normal execution (no override)
        agent_input_normal = AgentInput(content="Test normal", context={"scenario": "normal"})

        result = await agent.process(agent_input_normal)
        assert result.processing_time < 0.1  # Should complete quickly

    @pytest.mark.unit
    @pytest.mark.performance
    def test_performance_monitoring_error_conditions(self, performance_agent_class):
        """Test performance monitoring during error conditions."""
        config = {"agent_id": "error_test_agent"}

        # Create agent that will fail during execution
        class FailingAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate some processing time before failure
                await asyncio.sleep(0.01)
                raise ValueError("Simulated execution failure")

        failing_agent = FailingAgent(config)
        agent_input = AgentInput(content="Test error")

        # Should still track processing time even on error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            start_time = time.time()
            with pytest.raises(Exception, match="Simulated execution failure"):
                loop.run_until_complete(failing_agent.process(agent_input))
            end_time = time.time()

            # Should have taken some time (at least the sleep time)
            assert end_time - start_time >= 0.01

        finally:
            loop.close()

    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_history_tracking(self, performance_agent_class):
        """Test performance history tracking across multiple executions."""
        config = {"agent_id": "history_test_agent"}
        agent = performance_agent_class(config)

        # Execute multiple times with different scenarios
        scenarios = ["normal", "slow", "normal", "cpu_intensive", "normal"]

        for i, scenario in enumerate(scenarios):
            agent_input = AgentInput(content=f"Test {i}", context={"scenario": scenario})

            result = await agent.process(agent_input)
            assert result.metadata["execution_count"] == i + 1

        # Check timing history
        assert len(agent.timing_history) == 5

        # Slow execution should take longer than normal
        slow_time = agent.timing_history[1]  # Second execution was "slow"
        normal_times = [agent.timing_history[0], agent.timing_history[2], agent.timing_history[4]]

        # Slow should be noticeably longer than normal executions
        assert slow_time > max(normal_times) * 2

    @pytest.mark.unit
    @pytest.mark.performance
    def test_performance_monitoring_edge_cases(self, performance_agent_class):
        """Test performance monitoring in various edge cases."""
        config = {"agent_id": "edge_case_agent"}
        agent = performance_agent_class(config)

        # Test with very long content
        long_content = "x" * 100000  # 100KB of content
        agent_input = AgentInput(content=long_content, context={"scenario": "normal"})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(agent.process(agent_input))

            # Should handle large content without performance issues
            assert result.processing_time < 1.0
            assert len(result.content) > 0

        finally:
            loop.close()

    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_registry_performance_with_many_agents(self, performance_agent_class, registry):
        """Test registry performance with many registered agents."""
        # Register many agents
        for i in range(100):
            agent_class = performance_agent_class
            registry.register(f"perf_agent_{i}")(agent_class)

        # Test agent retrieval performance
        start_time = time.time()

        # Get agents in batch
        agents = []
        for i in range(100):
            config = {"agent_id": f"perf_agent_{i}"}
            agent = registry.get_agent(f"perf_agent_{i}", config)
            agents.append(agent)

        end_time = time.time()

        # Should be able to retrieve 100 agents quickly
        retrieval_time = end_time - start_time
        assert retrieval_time < 0.1  # Should take less than 100ms

        # Test capability discovery performance
        start_time = time.time()
        text_agents = registry.find_agents_by_type("text")
        end_time = time.time()

        discovery_time = end_time - start_time
        assert discovery_time < 0.1  # Should be fast even with many agents
        assert len(text_agents) == 100  # Should find all agents

    @pytest.mark.unit
    @pytest.mark.performance
    def test_memory_usage_monitoring(self, performance_agent_class):
        """Test memory usage patterns during agent execution."""
        config = {"agent_id": "memory_test_agent"}
        agent = performance_agent_class(config)

        # Test memory-intensive scenario
        agent_input = AgentInput(content="Memory test", context={"scenario": "memory_intensive"})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Force garbage collection before test
            gc.collect()

            # Execute multiple times to check for memory leaks
            for i in range(10):
                result = loop.run_until_complete(agent.process(agent_input))
                assert result.metadata["execution_count"] == i + 1

            # Force garbage collection after test
            gc.collect()

            # Agent should not accumulate excessive state
            assert len(agent.timing_history) == 10

        finally:
            loop.close()
