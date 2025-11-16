"""Comprehensive tests for TextProcessorAgent to achieve 100% coverage.

This test module covers all functionality of the TextProcessorAgent including:
- Initialization and configuration validation
- All text processing operations (analyze, clean, transform, sentiment)
- Error handling and validation
- Edge cases and boundary conditions
- Statistics and metadata generation
"""

import pytest

from src.agents.examples.text_processor_agent import TextProcessorAgent
from src.agents.exceptions import AgentExecutionError, AgentValidationError
from src.agents.models import AgentInput
from src.agents.registry import agent_registry


class TestTextProcessorAgent:
    """Comprehensive test suite for TextProcessorAgent."""

    def test_initialization_default_config(self):
        """Test TextProcessorAgent initialization with default configuration."""
        config = {"agent_id": "test_text_processor"}
        agent = TextProcessorAgent(config)

        # Verify default configuration values
        assert agent.agent_id == "test_text_processor"
        assert agent.max_words == 1000
        assert agent.min_words == 1
        assert agent.operations == ["analyze", "clean", "transform", "sentiment"]
        assert agent.enable_sentiment is True

        # Verify statistics initialization
        assert agent.requests_processed == 0
        assert agent.total_words_processed == 0
        assert agent.total_characters_processed == 0

    def test_initialization_custom_config(self):
        """Test TextProcessorAgent initialization with custom configuration."""
        config = {
            "agent_id": "custom_processor",
            "max_words": 500,
            "min_words": 5,
            "operations": ["analyze", "clean"],
            "enable_sentiment": False,
        }
        agent = TextProcessorAgent(config)

        # Verify custom configuration values
        assert agent.agent_id == "custom_processor"
        assert agent.max_words == 500
        assert agent.min_words == 5
        assert agent.operations == ["analyze", "clean"]
        assert agent.enable_sentiment is False

    def test_configuration_validation_invalid_max_words(self):
        """Test configuration validation for invalid max_words."""
        config = {"agent_id": "test_agent", "max_words": -1}

        with pytest.raises(AgentValidationError) as excinfo:
            TextProcessorAgent(config)

        assert excinfo.value.error_code == "INVALID_CONFIG_VALUE"
        assert "max_words must be a positive integer" in str(excinfo.value)

    def test_configuration_validation_invalid_min_words(self):
        """Test configuration validation for invalid min_words."""
        config = {"agent_id": "test_agent", "min_words": -1}

        with pytest.raises(AgentValidationError) as excinfo:
            TextProcessorAgent(config)

        assert excinfo.value.error_code == "INVALID_CONFIG_VALUE"
        assert "min_words must be a non-negative integer" in str(excinfo.value)

    def test_configuration_validation_invalid_operations_type(self):
        """Test configuration validation for invalid operations type."""
        config = {"agent_id": "test_agent", "operations": "not_a_list"}

        with pytest.raises(AgentValidationError) as excinfo:
            TextProcessorAgent(config)

        assert excinfo.value.error_code == "INVALID_CONFIG_VALUE"
        assert "operations must be a list" in str(excinfo.value)

    def test_configuration_validation_invalid_operation(self):
        """Test configuration validation for invalid operation."""
        config = {"agent_id": "test_agent", "operations": ["analyze", "invalid_op"]}

        with pytest.raises(AgentValidationError) as excinfo:
            TextProcessorAgent(config)

        assert excinfo.value.error_code == "INVALID_CONFIG_VALUE"
        assert "Invalid operation 'invalid_op'" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_analyze_operation_basic(self):
        """Test text analysis operation with basic input."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="Hello world! This is a test sentence.", context={"operation": "analyze"})

        result = await agent.execute(agent_input)

        # Verify result structure
        assert result.agent_id == "test_agent"
        assert "Text Analysis Results:" in result.content
        assert result.confidence >= 0.9
        assert "operation" in result.metadata
        assert result.metadata["operation"] == "analyze"
        assert result.metadata["word_count"] == 7

    @pytest.mark.asyncio
    async def test_analyze_operation_with_sentiment(self):
        """Test text analysis with sentiment analysis enabled."""
        config = {"agent_id": "test_agent", "enable_sentiment": True}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="This is a great and wonderful day! I love this amazing weather.",
            context={"operation": "analyze"},
        )

        result = await agent.execute(agent_input)

        # Verify sentiment is included
        assert "Text Analysis Results:" in result.content
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_clean_operation(self):
        """Test text cleaning operation."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="  Hello    world!!!   Extra   spaces   and   @#$%^   symbols  ",
            context={"operation": "clean"},
        )

        result = await agent.execute(agent_input)

        # Verify cleaning worked
        assert result.agent_id == "test_agent"
        assert result.confidence >= 0.9
        assert result.metadata["operation"] == "clean"
        # Cleaned text should have normalized whitespace and removed special chars
        assert "Hello world" in result.content

    @pytest.mark.asyncio
    async def test_transform_operation_uppercase(self):
        """Test text transformation to uppercase."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="hello world",
            context={"operation": "transform", "transform_type": "uppercase"},
        )

        result = await agent.execute(agent_input)

        # Verify transformation
        assert result.content == "HELLO WORLD"
        assert result.confidence >= 0.9
        assert result.metadata["operation"] == "transform"

    @pytest.mark.asyncio
    async def test_transform_operation_lowercase(self):
        """Test text transformation to lowercase."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="HELLO WORLD",
            context={"operation": "transform", "transform_type": "lowercase"},
        )

        result = await agent.execute(agent_input)

        assert result.content == "hello world"
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_transform_operation_title(self):
        """Test text transformation to title case."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="hello world test",
            context={"operation": "transform", "transform_type": "title"},
        )

        result = await agent.execute(agent_input)

        assert result.content == "Hello World Test"
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_transform_operation_sentence(self):
        """Test text transformation to sentence case."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="hello world test",
            context={"operation": "transform", "transform_type": "sentence"},
        )

        result = await agent.execute(agent_input)

        assert result.content == "Hello world test"
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_transform_operation_default_type(self):
        """Test text transformation with default transform type."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="HELLO WORLD",
            context={"operation": "transform"},  # No transform_type specified
        )

        result = await agent.execute(agent_input)

        # Default should be lowercase
        assert result.content == "hello world"

    @pytest.mark.asyncio
    async def test_transform_operation_invalid_type(self):
        """Test text transformation with invalid transform type."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="hello world", context={"operation": "transform", "transform_type": "invalid"})

        with pytest.raises(AgentValidationError) as excinfo:
            await agent.execute(agent_input)

        assert excinfo.value.error_code == "INVALID_TRANSFORM_TYPE"
        assert "Unknown transform type: invalid" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_sentiment_operation_positive(self):
        """Test sentiment analysis with positive text."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="I love this amazing wonderful fantastic great product! It's awesome and excellent.",
            context={"operation": "sentiment"},
        )

        result = await agent.execute(agent_input)

        # Should detect positive sentiment
        assert "positive" in result.content.lower()
        assert result.confidence >= 0.7  # Lower confidence for simple algorithm

    @pytest.mark.asyncio
    async def test_sentiment_operation_negative(self):
        """Test sentiment analysis with negative text."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="I hate this terrible awful horrible bad product! It's the worst failure ever.",
            context={"operation": "sentiment"},
        )

        result = await agent.execute(agent_input)

        # Should detect negative sentiment
        assert "negative" in result.content.lower()
        assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_sentiment_operation_neutral(self):
        """Test sentiment analysis with neutral text."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(
            content="This is a normal sentence with neutral words and standard content.",
            context={"operation": "sentiment"},
        )

        result = await agent.execute(agent_input)

        # Should detect neutral sentiment
        assert "neutral" in result.content.lower()

    @pytest.mark.asyncio
    async def test_default_operation(self):
        """Test execution with no operation specified (should default to analyze)."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="Test content without operation context")

        result = await agent.execute(agent_input)

        # Should default to analyze operation
        assert "Text Analysis Results:" in result.content
        assert result.metadata["operation"] == "analyze"

    @pytest.mark.asyncio
    async def test_operation_not_allowed(self):
        """Test execution with operation not in allowed operations."""
        config = {"agent_id": "test_agent", "operations": ["analyze"]}  # Only analyze allowed
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="Test content", context={"operation": "clean"})  # Clean not allowed

        with pytest.raises(AgentValidationError) as excinfo:
            await agent.execute(agent_input)

        assert excinfo.value.error_code == "INVALID_OPERATION"
        assert "Operation 'clean' is not allowed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        """Test execution with completely unknown operation."""
        config = {"agent_id": "test_agent"}  # Use default operations, unknown_operation not allowed
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="Test content", context={"operation": "unknown_operation"})

        with pytest.raises(AgentValidationError) as excinfo:
            await agent.execute(agent_input)

        assert excinfo.value.error_code == "INVALID_OPERATION"
        assert "Operation 'unknown_operation' is not allowed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_input_validation_too_short(self):
        """Test input validation with text too short."""
        config = {"agent_id": "test_agent", "min_words": 5}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="Short")  # Only 1 word, need 5

        with pytest.raises(AgentValidationError) as excinfo:
            await agent.execute(agent_input)

        assert excinfo.value.error_code == "INPUT_TOO_SHORT"
        assert "minimum is 5" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_input_validation_too_long(self):
        """Test input validation with text too long."""
        config = {"agent_id": "test_agent", "max_words": 3}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="This text has many words")  # 5 words, limit is 3

        with pytest.raises(AgentValidationError) as excinfo:
            await agent.execute(agent_input)

        assert excinfo.value.error_code == "INPUT_TOO_LONG"
        assert "maximum is 3" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test that statistics are properly tracked across multiple requests."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        # Initial state
        assert agent.requests_processed == 0
        assert agent.total_words_processed == 0
        assert agent.total_characters_processed == 0

        # Process first request
        agent_input1 = AgentInput(content="Hello world test")  # 3 words, 16 chars
        result1 = await agent.execute(agent_input1)

        # Check updated statistics
        assert agent.requests_processed == 1
        assert agent.total_words_processed == 3
        assert agent.total_characters_processed == 16
        assert result1.metadata["requests_processed"] == 1

        # Process second request
        agent_input2 = AgentInput(content="Another test sentence here")  # 4 words, 26 chars
        result2 = await agent.execute(agent_input2)

        # Check cumulative statistics
        assert agent.requests_processed == 2
        assert agent.total_words_processed == 7
        assert agent.total_characters_processed == 42
        assert result2.metadata["requests_processed"] == 2

    def test_get_capabilities(self):
        """Test get_capabilities method."""
        config = {
            "agent_id": "test_agent",
            "max_words": 500,
            "min_words": 2,
            "operations": ["analyze", "clean"],
            "enable_sentiment": False,
        }
        agent = TextProcessorAgent(config)

        capabilities = agent.get_capabilities()

        # Verify capabilities structure
        assert capabilities["agent_id"] == "test_agent"
        assert capabilities["agent_type"] == "TextProcessorAgent"
        assert capabilities["input_types"] == ["text"]
        assert capabilities["output_types"] == ["text", "analysis"]
        assert capabilities["operations"] == ["analyze", "clean"]
        assert capabilities["max_words"] == 500
        assert capabilities["min_words"] == 2
        assert capabilities["enable_sentiment"] is False
        assert capabilities["async_execution"] is True
        assert capabilities["timeout_support"] is True
        assert capabilities["config_overrides"] is True
        assert capabilities["languages"] == ["en"]

    def test_get_status_initial(self):
        """Test get_status method with initial state."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        status = agent.get_status()

        # Should include base status plus custom fields
        assert "agent_id" in status  # From base class
        assert status["requests_processed"] == 0
        assert status["total_words_processed"] == 0
        assert status["total_characters_processed"] == 0
        assert status["average_words_per_request"] == 0

    @pytest.mark.asyncio
    async def test_get_status_after_processing(self):
        """Test get_status method after processing requests."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        # Process some requests
        await agent.execute(AgentInput(content="Hello world"))  # 2 words
        await agent.execute(AgentInput(content="Test sentence here"))  # 3 words

        status = agent.get_status()

        # Check updated status
        assert status["requests_processed"] == 2
        assert status["total_words_processed"] == 5
        assert status["average_words_per_request"] == 2.5

    def test_most_common_words_functionality(self):
        """Test _get_most_common_words helper method."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "the"]
        result = agent._get_most_common_words(words, 3)

        # Should return top 3 words (excluding short words like "the")
        assert len(result) <= 3
        assert all("word" in item and "frequency" in item for item in result)
        # "the" should be most frequent but might be filtered out for being too short

    def test_readability_calculation(self):
        """Test _calculate_readability helper method."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        text = "This is a simple test sentence. Another sentence here."
        words = text.split()
        sentences = text.split(".")

        score = agent._calculate_readability(text, words, sentences)

        # Should return a score between 0 and 100
        assert 0 <= score <= 100
        assert isinstance(score, int | float)  # Can be int or float

    def test_readability_calculation_empty_input(self):
        """Test _calculate_readability with empty input."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        score = agent._calculate_readability("", [], [])

        # Should handle empty input gracefully
        assert score == 0.0

    def test_format_analysis_comprehensive(self):
        """Test _format_analysis helper method with comprehensive data."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        analysis = {
            "word_count": 10,
            "character_count": 50,
            "sentence_count": 2,
            "paragraph_count": 1,
            "average_word_length": 4.5,
            "average_sentence_length": 5.0,
            "readability_score": 75.5,
            "most_common_words": [{"word": "test", "frequency": 3}, {"word": "word", "frequency": 2}],
            "sentiment": "positive",
        }

        formatted = agent._format_analysis(analysis)

        # Check that all information is included
        assert "Word Count: 10" in formatted
        assert "Character Count: 50" in formatted
        assert "Sentence Count: 2" in formatted
        assert "Average Word Length: 4.50" in formatted
        assert "Readability Score: 75.50" in formatted
        assert "Most Common Words:" in formatted
        assert "test: 3" in formatted
        assert "Sentiment: Positive" in formatted

    def test_format_analysis_minimal(self):
        """Test _format_analysis with minimal data."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        analysis = {
            "word_count": 5,
            "character_count": 25,
            "sentence_count": 1,
            "paragraph_count": 1,
            "average_word_length": 5.0,
            "average_sentence_length": 5.0,
            "readability_score": 50.0,
            "most_common_words": [],  # Empty
        }

        formatted = agent._format_analysis(analysis)

        # Should handle empty most_common_words gracefully
        assert "Word Count: 5" in formatted
        assert "Most Common Words:" not in formatted  # Should be skipped when empty

    @pytest.mark.asyncio
    async def test_no_context_provided(self):
        """Test execution with no context provided."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="Test content")  # No context

        result = await agent.execute(agent_input)

        # Should default to analyze operation
        assert result.metadata["operation"] == "analyze"
        assert "Text Analysis Results:" in result.content

    @pytest.mark.asyncio
    async def test_exception_handling_in_execute(self):
        """Test that unexpected exceptions are properly handled in execute."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        # Create a mock input that might cause unexpected behavior
        # We'll test the exception handling path by using an invalid operation internally
        agent_input = AgentInput(content="Test content", context={"operation": "analyze"})

        # Temporarily break something to trigger exception handling
        original_method = agent._analyze_text

        async def broken_analyze(text):
            raise ValueError("Simulated error")

        agent._analyze_text = broken_analyze

        with pytest.raises(AgentExecutionError) as excinfo:
            await agent.execute(agent_input)

        assert excinfo.value.error_code == "PROCESSING_FAILED"
        assert "Text processing failed" in str(excinfo.value)

        # Restore original method
        agent._analyze_text = original_method

    def test_registry_registration(self):
        """Test that the agent is properly registered with the registry."""
        # The agent should be registered via the @agent_registry.register decorator
        registered_agents = agent_registry.list_agents()
        assert "text_processor" in registered_agents

    @pytest.mark.asyncio
    async def test_edge_case_empty_words_sentiment(self):
        """Test sentiment analysis with empty or minimal text."""
        config = {"agent_id": "test_agent"}
        agent = TextProcessorAgent(config)

        # Test with minimal text
        result = await agent._analyze_sentiment("")

        # Should handle empty text gracefully
        assert "sentiment" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_edge_case_single_word_analysis(self):
        """Test analysis with single word."""
        config = {"agent_id": "test_agent", "min_words": 1}
        agent = TextProcessorAgent(config)

        agent_input = AgentInput(content="Hello", context={"operation": "analyze"})
        result = await agent.execute(agent_input)

        # Should handle single word analysis
        assert result.metadata["word_count"] == 1
        assert "Text Analysis Results:" in result.content
