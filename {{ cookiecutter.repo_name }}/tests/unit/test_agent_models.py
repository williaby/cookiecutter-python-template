"""
Unit tests for agent data models.

This module tests the AgentInput, AgentOutput, and AgentConfig models
following the testing guide requirements for comprehensive validation,
edge case testing, and security considerations.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.agents.models import AgentConfig, AgentInput, AgentOutput


class TestAgentInput:
    """Test suite for AgentInput model."""

    def test_agent_input_creation_success(self, sample_agent_input):
        """Test successful creation of AgentInput."""
        assert sample_agent_input.content == "This is a test input for the agent"
        assert sample_agent_input.context["language"] == "python"
        assert sample_agent_input.config_overrides["max_tokens"] == 500
        assert sample_agent_input.request_id is not None
        assert isinstance(sample_agent_input.timestamp, datetime)

    def test_agent_input_minimal_creation(self):
        """Test creation with minimal required fields."""
        agent_input = AgentInput(content="Test content")

        assert agent_input.content == "Test content"
        assert agent_input.context is None
        assert agent_input.config_overrides is None
        assert agent_input.request_id is not None
        assert isinstance(agent_input.timestamp, datetime)

    def test_agent_input_request_id_generation(self):
        """Test that request_id is auto-generated and unique."""
        input1 = AgentInput(content="Test 1")
        input2 = AgentInput(content="Test 2")

        assert input1.request_id != input2.request_id
        assert len(input1.request_id) > 0
        assert len(input2.request_id) > 0

    @pytest.mark.parametrize(
        ("content", "expected_error"),
        [
            ("", "Content cannot be empty or whitespace-only"),
            ("   ", "Content cannot be empty or whitespace-only"),
            ("\n\t  \n", "Content cannot be empty or whitespace-only"),
            ("a" * 100001, "String should have at most 100000 characters"),
        ],
        ids=["empty", "whitespace", "whitespace-mixed", "too-long"],
    )
    def test_agent_input_content_validation(self, content, expected_error):
        """Test content validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentInput(content=content)
        assert expected_error in str(excinfo.value)

    def test_agent_input_content_whitespace_trimming(self):
        """Test that content whitespace is trimmed."""
        agent_input = AgentInput(content="  Test content with whitespace  ")
        assert agent_input.content == "Test content with whitespace"

    @pytest.mark.parametrize(
        ("context", "expected_error"),
        [
            ("not_a_dict", "Context must be a dictionary"),
            ({123: "value"}, "All context keys must be strings"),
            ({"key": "value", 456: "another"}, "All context keys must be strings"),
        ],
        ids=["not-dict", "non-string-key", "mixed-key-types"],
    )
    def test_agent_input_context_validation(self, context, expected_error):
        """Test context validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentInput(content="Test", context=context)
        assert expected_error in str(excinfo.value)

    @pytest.mark.parametrize(
        ("config_overrides", "expected_error"),
        [
            ("not_a_dict", "Config overrides must be a dictionary"),
            ({123: "value"}, "All config override keys must be strings"),
            ({"key": "value", 456: "another"}, "All config override keys must be strings"),
        ],
        ids=["not-dict", "non-string-key", "mixed-key-types"],
    )
    def test_agent_input_config_overrides_validation(self, config_overrides, expected_error):
        """Test config_overrides validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentInput(content="Test", config_overrides=config_overrides)
        assert expected_error in str(excinfo.value)

    def test_agent_input_cross_field_validation_code_content(self):
        """Test cross-field validation for code content."""
        long_code = "def test():\n    pass\n" * 10000  # Over 50,000 characters

        with pytest.raises(ValidationError) as excinfo:
            AgentInput(content=long_code, context={"content_type": "code"})
        assert "Code content cannot exceed 50,000 characters" in str(excinfo.value)

    def test_agent_input_cross_field_validation_regular_content(self):
        """Test cross-field validation allows long regular content."""
        long_content = "This is regular text. " * 3000  # Over 50,000 characters

        # Should not raise an error for regular content
        agent_input = AgentInput(content=long_content)
        assert len(agent_input.content) > 50000

    @pytest.mark.security
    def test_agent_input_security_content(self, security_test_inputs):
        """Test AgentInput with potentially malicious content."""
        for malicious_input in security_test_inputs:
            try:
                # Most malicious inputs should be accepted for downstream processing
                agent_input = AgentInput(content=malicious_input)
                assert agent_input.content == malicious_input
            except ValidationError as e:
                error_str = str(e)
                # Some inputs should legitimately fail validation - this is correct security behavior
                if malicious_input in ["", " ", "\t\n\r", "\r\n\r\n"]:
                    # Empty/whitespace-only content should fail validation
                    assert "Content cannot be empty or whitespace-only" in error_str
                elif malicious_input is None:
                    # None values should fail type validation
                    assert "Input should be a valid string" in error_str
                else:
                    # Other malicious inputs should be accepted
                    raise

    def test_agent_input_serialization(self, sample_agent_input):
        """Test AgentInput serialization."""
        data = sample_agent_input.model_dump()

        assert isinstance(data, dict)
        assert data["content"] == sample_agent_input.content
        assert data["context"] == sample_agent_input.context
        assert data["config_overrides"] == sample_agent_input.config_overrides
        assert data["request_id"] == sample_agent_input.request_id

    def test_agent_input_deserialization(self, sample_agent_input):
        """Test AgentInput deserialization."""
        data = sample_agent_input.model_dump()
        reconstructed = AgentInput.model_validate(data)

        assert reconstructed.content == sample_agent_input.content
        assert reconstructed.context == sample_agent_input.context
        assert reconstructed.config_overrides == sample_agent_input.config_overrides
        assert reconstructed.request_id == sample_agent_input.request_id


class TestAgentOutput:
    """Test suite for AgentOutput model."""

    def test_agent_output_creation_success(self, sample_agent_output):
        """Test successful creation of AgentOutput."""
        assert sample_agent_output.content == "This is a test output from the agent"
        assert sample_agent_output.metadata["analysis_type"] == "security"
        assert sample_agent_output.confidence == 0.95
        assert sample_agent_output.processing_time == 1.234
        assert sample_agent_output.agent_id == "test_agent"
        assert sample_agent_output.request_id == "test-request-123"

    def test_agent_output_minimal_creation(self):
        """Test creation with minimal required fields."""
        agent_output = AgentOutput(content="Test output", confidence=0.8, processing_time=0.5, agent_id="test_agent")

        assert agent_output.content == "Test output"
        assert agent_output.metadata == {}
        assert agent_output.confidence == 0.8
        assert agent_output.processing_time == 0.5
        assert agent_output.agent_id == "test_agent"
        assert agent_output.request_id is None
        assert isinstance(agent_output.timestamp, datetime)

    @pytest.mark.parametrize(
        ("content", "expected_error"),
        [
            ("", "Content cannot be empty or whitespace-only"),
            ("   ", "Content cannot be empty or whitespace-only"),
            ("\n\t  \n", "Content cannot be empty or whitespace-only"),
            ("a" * 500001, "String should have at most 500000 characters"),
        ],
        ids=["empty", "whitespace", "whitespace-mixed", "too-long"],
    )
    def test_agent_output_content_validation(self, content, expected_error):
        """Test content validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentOutput(content=content, confidence=0.8, processing_time=0.5, agent_id="test_agent")
        assert expected_error in str(excinfo.value)

    def test_agent_output_content_whitespace_trimming(self):
        """Test that content whitespace is trimmed."""
        agent_output = AgentOutput(
            content="  Test output with whitespace  ",
            confidence=0.8,
            processing_time=0.5,
            agent_id="test_agent",
        )
        assert agent_output.content == "Test output with whitespace"

    @pytest.mark.parametrize(
        ("confidence", "expected_error"),
        [
            (-0.1, "Input should be greater than or equal to 0"),
            (1.1, "Input should be less than or equal to 1"),
            ("not_a_number", "Input should be a valid number"),
        ],
        ids=["negative", "too-high", "not-number"],
    )
    def test_agent_output_confidence_validation(self, confidence, expected_error):
        """Test confidence validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentOutput(content="Test", confidence=confidence, processing_time=0.5, agent_id="test_agent")
        assert expected_error in str(excinfo.value)

    @pytest.mark.parametrize(
        ("processing_time", "expected_error"),
        [
            (-0.1, "Input should be greater than or equal to 0"),
            ("not_a_number", "Input should be a valid number"),
        ],
        ids=["negative", "not-number"],
    )
    def test_agent_output_processing_time_validation(self, processing_time, expected_error):
        """Test processing_time validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentOutput(content="Test", confidence=0.8, processing_time=processing_time, agent_id="test_agent")
        assert expected_error in str(excinfo.value)

    @pytest.mark.parametrize(
        ("agent_id", "expected_error"),
        [
            ("", "Agent ID cannot be empty or whitespace-only"),
            ("   ", "Agent ID cannot be empty or whitespace-only"),
            ("test-agent", "Agent ID must contain only alphanumeric characters and underscores"),
            ("test agent", "Agent ID must contain only alphanumeric characters and underscores"),
            ("test@agent", "Agent ID must contain only alphanumeric characters and underscores"),
        ],
        ids=["empty", "whitespace", "dash", "space", "special-char"],
    )
    def test_agent_output_agent_id_validation(self, agent_id, expected_error):
        """Test agent_id validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentOutput(content="Test", confidence=0.8, processing_time=0.5, agent_id=agent_id)
        assert expected_error in str(excinfo.value)

    def test_agent_output_agent_id_whitespace_trimming(self):
        """Test that agent_id whitespace is trimmed."""
        agent_output = AgentOutput(content="Test", confidence=0.8, processing_time=0.5, agent_id="  test_agent  ")
        assert agent_output.agent_id == "test_agent"

    @pytest.mark.parametrize(
        ("metadata", "expected_error"),
        [
            ("not_a_dict", "Metadata must be a dictionary"),
            ({123: "value"}, "All metadata keys must be strings"),
            ({"key": "value", 456: "another"}, "All metadata keys must be strings"),
        ],
        ids=["not-dict", "non-string-key", "mixed-key-types"],
    )
    def test_agent_output_metadata_validation(self, metadata, expected_error):
        """Test metadata validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentOutput(content="Test", confidence=0.8, processing_time=0.5, agent_id="test_agent", metadata=metadata)
        assert expected_error in str(excinfo.value)

    def test_agent_output_cross_field_validation_warning(self):
        """Test cross-field validation for high processing time with high confidence."""
        # This should not raise an error but might log a warning
        agent_output = AgentOutput(content="Test", confidence=0.96, processing_time=35.0, agent_id="test_agent")

        # Should be created successfully
        assert agent_output.confidence == 0.96
        assert agent_output.processing_time == 35.0

    def test_agent_output_serialization(self, sample_agent_output):
        """Test AgentOutput serialization."""
        data = sample_agent_output.model_dump()

        assert isinstance(data, dict)
        assert data["content"] == sample_agent_output.content
        assert data["metadata"] == sample_agent_output.metadata
        assert data["confidence"] == sample_agent_output.confidence
        assert data["processing_time"] == sample_agent_output.processing_time
        assert data["agent_id"] == sample_agent_output.agent_id
        assert data["request_id"] == sample_agent_output.request_id

    def test_agent_output_deserialization(self, sample_agent_output):
        """Test AgentOutput deserialization."""
        data = sample_agent_output.model_dump()
        reconstructed = AgentOutput.model_validate(data)

        assert reconstructed.content == sample_agent_output.content
        assert reconstructed.metadata == sample_agent_output.metadata
        assert reconstructed.confidence == sample_agent_output.confidence
        assert reconstructed.processing_time == sample_agent_output.processing_time
        assert reconstructed.agent_id == sample_agent_output.agent_id
        assert reconstructed.request_id == sample_agent_output.request_id


class TestAgentConfig:
    """Test suite for AgentConfig model."""

    def test_agent_config_creation_success(self, sample_agent_config_model):
        """Test successful creation of AgentConfig."""
        assert sample_agent_config_model.agent_id == "test_agent"
        assert sample_agent_config_model.name == "Test Agent"
        assert sample_agent_config_model.description == "A test agent for unit testing"
        assert sample_agent_config_model.config["max_tokens"] == 1000
        assert sample_agent_config_model.enabled is True

    def test_agent_config_minimal_creation(self):
        """Test creation with minimal required fields."""
        agent_config = AgentConfig(
            agent_id="minimal_agent",
            name="Minimal Agent",
            description="A minimal agent configuration",
        )

        assert agent_config.agent_id == "minimal_agent"
        assert agent_config.name == "Minimal Agent"
        assert agent_config.description == "A minimal agent configuration"
        assert agent_config.config == {}
        assert agent_config.enabled is True

    def test_agent_config_disabled_agent(self):
        """Test creation with disabled agent."""
        agent_config = AgentConfig(
            agent_id="disabled_agent",
            name="Disabled Agent",
            description="A disabled agent",
            enabled=False,
        )

        assert agent_config.enabled is False

    @pytest.mark.parametrize(
        ("agent_id", "expected_error"),
        [
            ("", "Agent ID cannot be empty"),
            ("   ", "Agent ID cannot be empty"),
            ("test-agent", "Agent ID must contain only alphanumeric characters and underscores"),
            ("test agent", "Agent ID must contain only alphanumeric characters and underscores"),
            ("test@agent", "Agent ID must contain only alphanumeric characters and underscores"),
        ],
        ids=["empty", "whitespace", "dash", "space", "special-char"],
    )
    def test_agent_config_agent_id_validation(self, agent_id, expected_error):
        """Test agent_id validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(agent_id=agent_id, name="Test Agent", description="Test description")
        assert expected_error in str(excinfo.value)

    def test_agent_config_agent_id_whitespace_trimming(self):
        """Test that agent_id whitespace is trimmed."""
        agent_config = AgentConfig(agent_id="  test_agent  ", name="Test Agent", description="Test description")
        assert agent_config.agent_id == "test_agent"

    @pytest.mark.parametrize(
        ("name", "expected_error"),
        [
            ("", "Name cannot be empty"),
            ("   ", "Name cannot be empty"),
        ],
        ids=["empty", "whitespace"],
    )
    def test_agent_config_name_validation(self, name, expected_error):
        """Test name validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(agent_id="test_agent", name=name, description="Test description")
        assert expected_error in str(excinfo.value)

    def test_agent_config_name_whitespace_trimming(self):
        """Test that name whitespace is trimmed."""
        agent_config = AgentConfig(agent_id="test_agent", name="  Test Agent  ", description="Test description")
        assert agent_config.name == "Test Agent"

    @pytest.mark.parametrize(
        ("description", "expected_error"),
        [
            ("", "Description cannot be empty"),
            ("   ", "Description cannot be empty"),
        ],
        ids=["empty", "whitespace"],
    )
    def test_agent_config_description_validation(self, description, expected_error):
        """Test description validation edge cases."""
        with pytest.raises(ValidationError) as excinfo:
            AgentConfig(agent_id="test_agent", name="Test Agent", description=description)
        assert expected_error in str(excinfo.value)

    def test_agent_config_description_whitespace_trimming(self):
        """Test that description whitespace is trimmed."""
        agent_config = AgentConfig(agent_id="test_agent", name="Test Agent", description="  Test description  ")
        assert agent_config.description == "Test description"

    def test_agent_config_complex_config(self):
        """Test AgentConfig with complex configuration."""
        complex_config = {
            "max_tokens": 2000,
            "temperature": 0.7,
            "models": ["gpt-4", "gpt-3.5-turbo"],
            "features": {"sentiment_analysis": True, "language_detection": False},
            "thresholds": {"confidence": 0.8, "processing_time": 30.0},
        }

        agent_config = AgentConfig(
            agent_id="complex_agent",
            name="Complex Agent",
            description="An agent with complex configuration",
            config=complex_config,
        )

        assert agent_config.config == complex_config
        assert agent_config.config["features"]["sentiment_analysis"] is True
        assert agent_config.config["thresholds"]["confidence"] == 0.8

    def test_agent_config_serialization(self, sample_agent_config_model):
        """Test AgentConfig serialization."""
        data = sample_agent_config_model.model_dump()

        assert isinstance(data, dict)
        assert data["agent_id"] == sample_agent_config_model.agent_id
        assert data["name"] == sample_agent_config_model.name
        assert data["description"] == sample_agent_config_model.description
        assert data["config"] == sample_agent_config_model.config
        assert data["enabled"] == sample_agent_config_model.enabled

    def test_agent_config_deserialization(self, sample_agent_config_model):
        """Test AgentConfig deserialization."""
        data = sample_agent_config_model.model_dump()
        reconstructed = AgentConfig.model_validate(data)

        assert reconstructed.agent_id == sample_agent_config_model.agent_id
        assert reconstructed.name == sample_agent_config_model.name
        assert reconstructed.description == sample_agent_config_model.description
        assert reconstructed.config == sample_agent_config_model.config
        assert reconstructed.enabled == sample_agent_config_model.enabled

    @pytest.mark.security
    def test_agent_config_security_validation(self, security_test_inputs):
        """Test AgentConfig with potentially malicious inputs."""
        for malicious_input in security_test_inputs:
            # Test that malicious inputs are handled properly
            # The model should accept them as strings but not execute them
            try:
                agent_config = AgentConfig(
                    agent_id="security_test",
                    name=malicious_input,
                    description="Testing security",
                )
                assert agent_config.name == malicious_input
            except ValidationError:
                # Some malicious inputs might fail validation, which is expected
                pass

    @pytest.mark.parametrize(
        "config_value",
        [
            {"string": "value"},
            {"number": 123},
            {"boolean": True},
            {"list": [1, 2, 3]},
            {"dict": {"nested": "value"}},
            {"none": None},
        ],
        ids=["string", "number", "boolean", "list", "dict", "none"],
    )
    def test_agent_config_arbitrary_types(self, config_value):
        """Test AgentConfig with various config value types."""
        agent_config = AgentConfig(
            agent_id="test_agent",
            name="Test Agent",
            description="Test description",
            config=config_value,
        )

        assert agent_config.config == config_value
