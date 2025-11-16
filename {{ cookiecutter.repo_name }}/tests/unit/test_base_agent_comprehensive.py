"""Comprehensive unit tests for BaseAgent framework.

This module provides comprehensive unit test coverage for the BaseAgent class
and its associated functionality including configuration, validation, execution,
and error handling.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agents.base_agent import BaseAgent, BaseAgentType
from src.agents.exceptions import (
    AgentConfigurationError,
    AgentExecutionError,
    AgentTimeoutError,
)
from src.agents.models import AgentInput, AgentOutput


class TestBaseAgent:
    """Test suite for BaseAgent class."""

    def test_concrete_agent_implementation(self):
        """Test a concrete agent implementation."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output(
                    content="Test response",
                    metadata={"test": True},
                    confidence=0.9,
                    request_id=agent_input.request_id,
                )

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        assert agent.agent_id == "test_agent"
        assert agent.config == config
        assert agent._initialized is True
        assert agent.logger is not None

    def test_base_agent_initialization_valid_config(self):
        """Test BaseAgent initialization with valid configuration."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "valid_agent", "timeout": 30.0}
        agent = TestAgent(config)

        assert agent.agent_id == "valid_agent"
        assert agent.config == config
        assert agent._initialized is True
        assert agent.logger is not None

    def test_base_agent_initialization_invalid_config_type(self):
        """Test BaseAgent initialization with invalid config type raises error."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        with pytest.raises(AgentConfigurationError) as exc_info:
            TestAgent("invalid_config")

        assert "Configuration must be a dictionary" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_CONFIG_TYPE"

    def test_base_agent_initialization_missing_agent_id(self):
        """Test BaseAgent initialization without agent_id raises error."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"timeout": 30.0}  # Missing agent_id
        with pytest.raises(AgentConfigurationError) as exc_info:
            TestAgent(config)

        assert "Agent ID is required" in str(exc_info.value)
        assert exc_info.value.error_code == "MISSING_REQUIRED_CONFIG"

    def test_base_agent_initialization_empty_agent_id(self):
        """Test BaseAgent initialization with empty agent_id raises error."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": ""}
        with pytest.raises(AgentConfigurationError) as exc_info:
            TestAgent(config)

        assert "Agent ID is required" in str(exc_info.value)
        assert exc_info.value.error_code == "MISSING_REQUIRED_CONFIG"

    def test_base_agent_initialization_invalid_agent_id_format(self):
        """Test BaseAgent initialization with invalid agent_id format raises error."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "invalid-agent-id"}  # Hyphens not allowed
        with pytest.raises(AgentConfigurationError) as exc_info:
            TestAgent(config)

        assert "must contain only alphanumeric characters and underscores" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_CONFIG_VALUE"

    def test_validate_agent_id_valid_formats(self):
        """Test _validate_agent_id accepts valid formats."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        # Valid formats
        valid_ids = ["simple", "with_underscore", "with123numbers", "test_agent_123"]

        for valid_id in valid_ids:
            config = {"agent_id": valid_id}
            agent = TestAgent(config)
            assert agent.agent_id == valid_id

    def test_validate_agent_id_invalid_formats(self):
        """Test _validate_agent_id rejects invalid formats."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        # Invalid formats
        invalid_ids = ["with-hyphen", "with.dot", "with space", "with@symbol", "with#hash"]

        for invalid_id in invalid_ids:
            config = {"agent_id": invalid_id}
            with pytest.raises(AgentConfigurationError):
                TestAgent(config)

    def test_validate_configuration_base_implementation(self):
        """Test _validate_configuration base implementation."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        # Should not raise for valid dict config
        agent._validate_configuration()

        # Should raise for invalid config type
        agent.config = "invalid"
        with pytest.raises(AgentConfigurationError):
            agent._validate_configuration()

    def test_validate_configuration_custom_override(self):
        """Test _validate_configuration can be overridden in subclasses."""

        class TestAgent(BaseAgent):
            def _validate_configuration(self):
                super()._validate_configuration()
                if not self.config.get("required_param"):
                    raise AgentConfigurationError(
                        message="Required parameter missing",
                        error_code="MISSING_REQUIRED_PARAM",
                        agent_id=self.agent_id,
                    )

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        # Should fail without required param
        config = {"agent_id": "test_agent"}
        with pytest.raises(AgentConfigurationError) as exc_info:
            TestAgent(config)

        assert "Required parameter missing" in str(exc_info.value)
        assert exc_info.value.error_code == "MISSING_REQUIRED_PARAM"

        # Should succeed with required param
        config = {"agent_id": "test_agent", "required_param": "value"}
        agent = TestAgent(config)
        assert agent._initialized is True

    def test_merge_config_no_overrides(self):
        """Test _merge_config with no overrides returns original config."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent", "param": "value"}
        agent = TestAgent(config)

        merged = agent._merge_config(None)
        assert merged == config
        assert merged is not config  # Should be a copy

    def test_merge_config_with_overrides(self):
        """Test _merge_config merges overrides correctly."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent", "param1": "value1", "param2": "value2"}
        agent = TestAgent(config)

        overrides = {"param2": "overridden", "param3": "new_value"}
        merged = agent._merge_config(overrides)

        assert merged["agent_id"] == "test_agent"
        assert merged["param1"] == "value1"
        assert merged["param2"] == "overridden"
        assert merged["param3"] == "new_value"

    def test_merge_config_preserves_original(self):
        """Test _merge_config preserves original config."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent", "param": "original"}
        agent = TestAgent(config)

        overrides = {"param": "overridden"}
        merged = agent._merge_config(overrides)

        assert merged["param"] == "overridden"
        assert agent.config["param"] == "original"  # Original unchanged

    def test_create_output_minimal(self):
        """Test _create_output with minimal parameters."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        output = agent._create_output("Test content")

        assert isinstance(output, AgentOutput)
        assert output.content == "Test content"
        assert output.metadata == {}
        assert output.confidence == 1.0
        assert output.processing_time == 0.0
        assert output.agent_id == "test_agent"
        assert output.request_id is None

    def test_create_output_full_parameters(self):
        """Test _create_output with all parameters."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        metadata = {"key": "value", "count": 42}
        output = agent._create_output(
            content="Test content",
            metadata=metadata,
            confidence=0.85,
            processing_time=1.5,
            request_id="req-123",
        )

        assert output.content == "Test content"
        assert output.metadata == metadata
        assert output.confidence == 0.85
        assert output.processing_time == 1.5
        assert output.agent_id == "test_agent"
        assert output.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self):
        """Test _execute_with_timeout executes successfully within timeout."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate some processing time
                await asyncio.sleep(0.1)
                return self._create_output("Success", request_id=agent_input.request_id)

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test", request_id="req-123")
        result = await agent._execute_with_timeout(agent_input, timeout=1.0)

        assert result.content == "Success"
        assert result.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_times_out(self):
        """Test _execute_with_timeout raises AgentTimeoutError on timeout."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate long processing time
                await asyncio.sleep(2.0)
                return self._create_output("Success")

        config = {"agent_id": "test_agent", "timeout": 0.1}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test", request_id="req-123")
        with pytest.raises(AgentTimeoutError) as exc_info:
            await agent._execute_with_timeout(agent_input, timeout=0.1)

        assert "timed out after 0.1 seconds" in str(exc_info.value)
        assert exc_info.value.timeout == 0.1
        assert exc_info.value.agent_id == "test_agent"
        assert exc_info.value.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_uses_config_timeout(self):
        """Test _execute_with_timeout uses config timeout when none specified."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                await asyncio.sleep(0.2)
                return self._create_output("Success")

        config = {"agent_id": "test_agent", "timeout": 0.1}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test", request_id="req-123")
        with pytest.raises(AgentTimeoutError):
            await agent._execute_with_timeout(agent_input)  # No timeout specified

    @pytest.mark.asyncio
    async def test_execute_with_timeout_execution_error(self):
        """Test _execute_with_timeout handles execution errors."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                raise ValueError("Execution failed")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test", request_id="req-123")
        with pytest.raises(AgentExecutionError) as exc_info:
            await agent._execute_with_timeout(agent_input)

        assert exc_info.value.agent_id == "test_agent"
        assert exc_info.value.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_process_successful_execution(self):
        """Test process method with successful execution."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output(
                    content=f"Processed: {agent_input.content}",
                    metadata={"processed": True},
                    confidence=0.95,
                    request_id=agent_input.request_id,
                )

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test input", request_id="req-123")
        result = await agent.process(agent_input)

        assert result.content == "Processed: test input"
        assert result.metadata["processed"] is True
        assert result.confidence == 0.95
        assert result.request_id == "req-123"
        assert result.agent_id == "test_agent"
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_not_initialized_error(self):
        """Test process method fails when agent not initialized."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)
        agent._initialized = False  # Force uninitialized state

        agent_input = AgentInput(content="test", request_id="req-123")
        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.process(agent_input)

        assert "Agent not initialized" in str(exc_info.value)
        assert exc_info.value.error_code == "AGENT_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_process_with_config_overrides(self):
        """Test process method with configuration overrides."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                param_value = self.config.get("param", "default")
                return self._create_output(content=f"Param: {param_value}", request_id=agent_input.request_id)

        config = {"agent_id": "test_agent", "param": "original"}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test", request_id="req-123", config_overrides={"param": "overridden"})
        result = await agent.process(agent_input)

        assert result.content == "Param: overridden"
        assert agent.config["param"] == "original"  # Original config restored

    @pytest.mark.asyncio
    async def test_process_config_overrides_restoration(self):
        """Test process method restores original config after overrides."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                # Simulate an error during processing
                if agent_input.content == "error":
                    raise ValueError("Simulated error")
                return self._create_output("success")

        config = {"agent_id": "test_agent", "param": "original"}
        agent = TestAgent(config)

        agent_input = AgentInput(content="error", request_id="req-123", config_overrides={"param": "overridden"})

        with pytest.raises(AgentExecutionError):
            await agent.process(agent_input)

        # Original config should be restored even after error
        assert agent.config["param"] == "original"

    @pytest.mark.asyncio
    async def test_process_metrics_recording(self):
        """Test process method records metrics correctly."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("success")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        # Mock metrics collector
        with patch.object(agent, "metrics") as mock_metrics:
            agent_input = AgentInput(content="test", request_id="req-123")
            await agent.process(agent_input)

            # Verify success metrics were recorded
            mock_metrics.increment_counter.assert_called_with("agent_executions_success")
            mock_metrics.record_duration.assert_called_once()
            args = mock_metrics.record_duration.call_args[0]
            assert args[0] == "agent_execution_duration_seconds"
            assert args[1] > 0  # Processing time should be positive

    @pytest.mark.asyncio
    async def test_process_error_metrics_recording(self):
        """Test process method records error metrics correctly."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                raise ValueError("Execution failed")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        # Mock metrics collector
        with patch.object(agent, "metrics") as mock_metrics:
            agent_input = AgentInput(content="test", request_id="req-123")
            with pytest.raises(AgentExecutionError):
                await agent.process(agent_input)

            # Verify failure metrics were recorded
            mock_metrics.increment_counter.assert_called_with("agent_executions_failed")
            mock_metrics.record_duration.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_structured_logging(self):
        """Test process method uses structured logging."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("success")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        # Mock structured logger
        with patch.object(agent, "structured_logger") as mock_logger:
            agent_input = AgentInput(
                content="test content",
                request_id="req-123",
                config_overrides={"debug": True},  # Add config overrides to trigger debug logging
            )
            await agent.process(agent_input)

            # Verify structured logging calls
            assert mock_logger.info.called
            assert mock_logger.debug.called  # For config overrides

    def test_get_capabilities_default(self):
        """Test get_capabilities returns default capabilities."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        capabilities = agent.get_capabilities()

        assert capabilities["agent_id"] == "test_agent"
        assert capabilities["agent_type"] == "TestAgent"
        assert capabilities["input_types"] == ["text"]
        assert capabilities["output_types"] == ["text"]
        assert capabilities["async_execution"] is True
        assert capabilities["timeout_support"] is True
        assert capabilities["config_overrides"] is True

    def test_get_capabilities_custom_override(self):
        """Test get_capabilities can be overridden in subclasses."""

        class TestAgent(BaseAgent):
            def get_capabilities(self):
                base_capabilities = super().get_capabilities()
                base_capabilities.update(
                    {
                        "input_types": ["text", "json"],
                        "output_types": ["text", "analysis"],
                        "max_input_length": 10000,
                        "specialized_features": ["sentiment_analysis"],
                    },
                )
                return base_capabilities

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        capabilities = agent.get_capabilities()

        assert capabilities["input_types"] == ["text", "json"]
        assert capabilities["output_types"] == ["text", "analysis"]
        assert capabilities["max_input_length"] == 10000
        assert capabilities["specialized_features"] == ["sentiment_analysis"]

    def test_get_status(self):
        """Test get_status returns current agent status."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent", "param": "value"}
        agent = TestAgent(config)

        status = agent.get_status()

        assert status["agent_id"] == "test_agent"
        assert status["agent_type"] == "TestAgent"
        assert status["initialized"] is True
        assert status["config"] == config

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        str_repr = str(agent)
        assert str_repr == "TestAgent(agent_id='test_agent')"

        repr_repr = repr(agent)
        assert repr_repr == "TestAgent(agent_id='test_agent', initialized=True)"

    def test_base_agent_type_alias(self):
        """Test BaseAgentType alias is properly defined."""
        assert BaseAgentType is BaseAgent

    def test_module_exports(self):
        """Test module __all__ exports."""
        from src.agents.base_agent import __all__

        assert "BaseAgent" in __all__
        assert "BaseAgentType" in __all__

    @pytest.mark.asyncio
    async def test_initialization_logging(self):
        """Test agent initialization logs appropriate messages."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}

        # Mock structured logger and log_agent_event
        with (
            patch("src.agents.base_agent.create_structured_logger") as mock_create_logger,
            patch("src.agents.base_agent.log_agent_event") as mock_log_event,
        ):

            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger

            TestAgent(config)

            # Verify structured logger was created
            mock_create_logger.assert_called_with("agent.test_agent")

            # Verify initialization logging
            mock_logger.info.assert_called_once()
            mock_log_event.assert_called_once()

            # Verify log event details
            call_args = mock_log_event.call_args
            assert call_args[1]["event_type"] == "agent_initialization_success"
            assert call_args[1]["agent_id"] == "test_agent"

    @pytest.mark.asyncio
    async def test_initialization_error_logging(self):
        """Test agent initialization logs errors appropriately."""

        class TestAgent(BaseAgent):
            def _validate_configuration(self):
                raise ValueError("Custom validation error")

            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config = {"agent_id": "test_agent"}

        # Mock structured logger and log_agent_event
        with (
            patch("src.agents.base_agent.create_structured_logger") as mock_create_logger,
            patch("src.agents.base_agent.log_agent_event") as mock_log_event,
            patch("src.agents.base_agent.handle_agent_error") as mock_handle_error,
        ):

            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            mock_handle_error.return_value = AgentConfigurationError("Handled error")

            with pytest.raises(AgentConfigurationError):
                TestAgent(config)

            # Verify error logging
            mock_log_event.assert_called_once()
            call_args = mock_log_event.call_args
            assert call_args[1]["event_type"] == "agent_initialization_failed"
            assert call_args[1]["agent_id"] == "test_agent"
            assert call_args[1]["level"] == "error"

    @pytest.mark.asyncio
    async def test_process_timing_accuracy(self):
        """Test process method timing is accurate."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                await asyncio.sleep(0.1)  # Simulate 100ms processing
                return self._create_output("success")

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test", request_id="req-123")

        start_time = time.time()
        result = await agent.process(agent_input)
        end_time = time.time()

        # Processing time should be accurate within tolerance
        actual_duration = end_time - start_time
        assert abs(result.processing_time - actual_duration) < 0.05  # 50ms tolerance

    @pytest.mark.asyncio
    async def test_process_with_context_and_metadata(self):
        """Test process method handles context and metadata correctly."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                context_info = agent_input.context.get("info", "none") if agent_input.context else "none"
                return self._create_output(content=f"Context: {context_info}", metadata={"context_processed": True})

        config = {"agent_id": "test_agent"}
        agent = TestAgent(config)

        agent_input = AgentInput(content="test", context={"info": "test_context"}, request_id="req-123")

        result = await agent.process(agent_input)

        assert result.content == "Context: test_context"
        assert result.metadata["context_processed"] is True

    @pytest.mark.asyncio
    async def test_abstract_method_enforcement(self):
        """Test BaseAgent cannot be instantiated directly due to abstract method."""
        config = {"agent_id": "test_agent"}

        with pytest.raises(TypeError):
            BaseAgent(config)

    def test_config_isolation(self):
        """Test agent configurations are isolated between instances."""

        class TestAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return self._create_output("test")

        config1 = {"agent_id": "agent1", "param": "value1"}
        config2 = {"agent_id": "agent2", "param": "value2"}

        agent1 = TestAgent(config1)
        agent2 = TestAgent(config2)

        assert agent1.config["param"] == "value1"
        assert agent2.config["param"] == "value2"

        # Modify one config
        agent1.config["param"] = "modified"

        # Other agent should be unaffected
        assert agent2.config["param"] == "value2"
