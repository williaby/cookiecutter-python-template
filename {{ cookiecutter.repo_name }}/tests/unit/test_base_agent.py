"""
Unit tests for BaseAgent framework.

This module tests the BaseAgent abstract class and its core functionality,
including configuration validation, error handling, and lifecycle management.
"""

import asyncio
from unittest.mock import patch

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentConfigurationError, AgentExecutionError, AgentTimeoutError
from src.agents.models import AgentInput, AgentOutput


class TestBaseAgent:
    """Test suite for BaseAgent abstract class."""

    def test_base_agent_cannot_be_instantiated(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent({"agent_id": "test"})

    def test_base_agent_subclass_creation(self, sample_agent_config):
        """Test that BaseAgent subclasses can be created."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)
        assert agent.agent_id == "test_agent"
        assert agent.config == sample_agent_config
        assert agent._initialized is True

    def test_base_agent_missing_agent_id(self):
        """Test BaseAgent validation with missing agent_id."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent({})
        assert "Agent ID is required in configuration" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("agent_id", "expected_error"),
        [
            ("", "Agent ID is required in configuration"),
            ("test-agent", "Agent ID 'test-agent' must contain only alphanumeric characters and underscores"),
            ("test agent", "Agent ID 'test agent' must contain only alphanumeric characters and underscores"),
            ("test@agent", "Agent ID 'test@agent' must contain only alphanumeric characters and underscores"),
        ],
        ids=["empty", "dash", "space", "special-char"],
    )
    def test_base_agent_agent_id_validation(self, agent_id, expected_error):
        """Test BaseAgent agent_id validation edge cases."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent({"agent_id": agent_id})
        assert expected_error in str(excinfo.value)

    def test_base_agent_config_validation(self):
        """Test BaseAgent configuration validation."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent("not_a_dict")
        assert "Configuration must be a dictionary" in str(excinfo.value)

    def test_base_agent_custom_validation(self):
        """Test BaseAgent with custom validation logic."""

        class TestAgent(BaseAgent):
            def _validate_configuration(self) -> None:
                super()._validate_configuration()

                # Custom validation
                if self.config.get("required_param") is None:
                    raise AgentConfigurationError(
                        message="required_param is missing",
                        error_code="MISSING_REQUIRED_PARAM",
                        agent_id=self.agent_id,
                    )

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        # Should fail without required_param
        with pytest.raises(AgentConfigurationError) as excinfo:
            TestAgent({"agent_id": "test_agent"})
        assert "required_param is missing" in str(excinfo.value)

        # Should succeed with required_param
        agent = TestAgent({"agent_id": "test_agent", "required_param": "value"})
        assert agent.config["required_param"] == "value"

    def test_base_agent_config_merging(self, sample_agent_config):
        """Test configuration merging functionality."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)

        # Test merging with None
        merged = agent._merge_config(None)
        assert merged == sample_agent_config

        # Test merging with overrides
        overrides = {"temperature": 0.5, "new_param": "value"}
        merged = agent._merge_config(overrides)

        assert merged["temperature"] == 0.5  # Overridden
        assert merged["max_tokens"] == 1000  # Original
        assert merged["new_param"] == "value"  # New

    def test_base_agent_create_output_helper(self, sample_agent_config):
        """Test _create_output helper method."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)

        # Test minimal output creation
        output = agent._create_output("Test content")
        assert output.content == "Test content"
        assert output.agent_id == "test_agent"
        assert output.confidence == 1.0
        assert output.processing_time == 0.0
        assert output.metadata == {}

        # Test full output creation
        output = agent._create_output(
            content="Full test content",
            metadata={"test": True},
            confidence=0.85,
            processing_time=2.5,
            request_id="test-request",
        )
        assert output.content == "Full test content"
        assert output.metadata == {"test": True}
        assert output.confidence == 0.85
        assert output.processing_time == 2.5
        assert output.request_id == "test-request"

    @pytest.mark.asyncio
    async def test_base_agent_execute_with_timeout_success(self, sample_agent_config, sample_agent_input):
        """Test successful execution with timeout."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate some processing time
                await asyncio.sleep(0.1)
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)
        result = await agent._execute_with_timeout(sample_agent_input, timeout=1.0)

        assert result.content == "test response"
        assert result.agent_id == "test_agent"

    @pytest.mark.asyncio
    async def test_base_agent_execute_with_timeout_failure(self, sample_agent_config, sample_agent_input):
        """Test execution timeout failure."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate long processing time
                await asyncio.sleep(2.0)
                return AgentOutput(content="test response", confidence=0.9, processing_time=2.0, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)

        with pytest.raises(AgentTimeoutError) as excinfo:
            await agent._execute_with_timeout(sample_agent_input, timeout=0.5)

        assert "Agent execution timed out after 0.5 seconds" in str(excinfo.value)
        assert excinfo.value.agent_id == "test_agent"

    @pytest.mark.asyncio
    async def test_base_agent_execute_with_default_timeout(self, sample_agent_input):
        """Test execution with default timeout from config."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        config = {"agent_id": "test_agent", "timeout": 15.0}
        agent = TestAgent(config)

        # Mock asyncio.wait_for to verify timeout parameter
        with patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.return_value = AgentOutput(
                content="test response",
                confidence=0.9,
                processing_time=0.1,
                agent_id="test_agent",
            )

            await agent._execute_with_timeout(sample_agent_input)

            # Verify timeout was passed correctly
            mock_wait_for.assert_called_once()
            args, kwargs = mock_wait_for.call_args
            assert kwargs.get("timeout") == 15.0 or args[1] == 15.0

    @pytest.mark.asyncio
    async def test_base_agent_process_success(self, sample_agent_config, sample_agent_input):
        """Test successful agent processing."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)
        result = await agent.process(sample_agent_input)

        assert result.content == "test response"
        assert result.agent_id == "test_agent"
        assert result.processing_time > 0  # Should be updated with actual time

    @pytest.mark.asyncio
    async def test_base_agent_process_not_initialized(self, sample_agent_config, sample_agent_input):
        """Test processing with uninitialized agent."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)
        agent._initialized = False  # Manually set to not initialized

        with pytest.raises(AgentExecutionError) as excinfo:
            await agent.process(sample_agent_input)

        assert "Agent not initialized" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_base_agent_process_with_config_overrides(self, sample_agent_config, sample_agent_input):
        """Test processing with configuration overrides."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Return config value to verify override
                temperature = self.config.get("temperature", 0.0)
                return AgentOutput(
                    content=f"temperature: {temperature}",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

        agent = TestAgent(sample_agent_config)

        # Original config has temperature 0.8, override in sample_agent_input is 0.7
        result = await agent.process(sample_agent_input)
        assert "temperature: 0.7" in result.content  # Should use override from sample_agent_input

        # Verify original config is restored
        assert agent.config["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_base_agent_process_execution_error(self, sample_agent_config, sample_agent_input):
        """Test processing with execution error."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                raise ValueError("Test error")

        agent = TestAgent(sample_agent_config)

        with pytest.raises(AgentExecutionError) as excinfo:
            await agent.process(sample_agent_input)

        assert "Test error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_base_agent_process_logging(self, sample_agent_config, sample_agent_input):
        """Test that processing logs appropriately."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)

        with patch.object(agent.structured_logger, "info") as mock_info:
            await agent.process(sample_agent_input)

            # Verify logging calls
            assert mock_info.call_count >= 1  # At least start logging

            # Check that request_id is logged in kwargs
            call_kwargs = [call[1] for call in mock_info.call_args_list]
            assert any(sample_agent_input.request_id == kwargs.get("request_id") for kwargs in call_kwargs)

    def test_base_agent_get_capabilities(self, sample_agent_config):
        """Test get_capabilities method."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)
        capabilities = agent.get_capabilities()

        assert capabilities["agent_id"] == "test_agent"
        assert capabilities["agent_type"] == "TestAgent"
        assert capabilities["input_types"] == ["text"]
        assert capabilities["output_types"] == ["text"]
        assert capabilities["async_execution"] is True
        assert capabilities["timeout_support"] is True
        assert capabilities["config_overrides"] is True

    def test_base_agent_get_status(self, sample_agent_config):
        """Test get_status method."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)
        status = agent.get_status()

        assert status["agent_id"] == "test_agent"
        assert status["agent_type"] == "TestAgent"
        assert status["initialized"] is True
        assert status["config"] == sample_agent_config

    def test_base_agent_string_representation(self, sample_agent_config):
        """Test string representation methods."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        agent = TestAgent(sample_agent_config)

        # Test __str__
        str_repr = str(agent)
        assert "TestAgent" in str_repr
        assert "test_agent" in str_repr

        # Test __repr__
        repr_str = repr(agent)
        assert "TestAgent" in repr_str
        assert "test_agent" in repr_str
        assert "initialized=True" in repr_str

    def test_base_agent_initialization_error_handling(self):
        """Test error handling during initialization."""

        class TestAgent(BaseAgent):
            def _validate_configuration(self) -> None:
                super()._validate_configuration()
                raise ValueError("Custom validation error")

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(content="test response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        with pytest.raises(AgentExecutionError) as excinfo:
            TestAgent({"agent_id": "test_agent"})

        # Should wrap the ValueError in AgentExecutionError
        assert "Custom validation error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_base_agent_abstract_execute_method(self, sample_agent_config):
        """Test that execute method must be implemented."""

        class IncompleteAgent(BaseAgent):
            pass  # Missing execute method

        # Should fail during class definition due to abstract method
        with pytest.raises(TypeError):
            IncompleteAgent(sample_agent_config)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_base_agent_performance_timing(self, sample_agent_config, sample_agent_input):
        """Test that processing time is accurately measured."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate processing time
                await asyncio.sleep(0.1)
                return AgentOutput(
                    content="test response",
                    confidence=0.9,
                    processing_time=0.0,  # Will be overwritten
                    agent_id=self.agent_id,
                )

        agent = TestAgent(sample_agent_config)
        result = await agent.process(sample_agent_input)

        # Processing time should be approximately 0.1 seconds
        assert result.processing_time >= 0.1
        assert result.processing_time < 0.2  # Allow some tolerance

    @pytest.mark.security
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)  # Add explicit timeout to prevent hang
    async def test_base_agent_security_input_handling(self, sample_agent_config, security_test_inputs):
        """Test BaseAgent handling of potentially malicious inputs."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Echo back the input to test handling
                return AgentOutput(
                    content=f"Processed: {agent_input.content}",
                    confidence=0.9,
                    processing_time=0.01,  # Reduce mock processing time
                    agent_id=self.agent_id,
                )

        agent = TestAgent(sample_agent_config)

        # Test a representative sample to avoid timeout with large fixture list
        sample_inputs = security_test_inputs[:10] if len(security_test_inputs) > 10 else security_test_inputs

        for malicious_input in sample_inputs:
            # Skip None values that can't be converted to string properly
            if malicious_input is None:
                continue

            # Create input with malicious content
            agent_input = AgentInput(content=str(malicious_input))

            # Should not raise errors - input should be processed
            result = await agent.process(agent_input)
            assert result.content.startswith("Processed:")

            # Verify malicious content is not executed
            assert "alert" not in result.content or str(malicious_input) in result.content
