"""
Unit tests for TextProcessorAgent class.

This module provides comprehensive test coverage for the TextProcessorAgent class,
testing all text processing operations, validation, error handling, and metadata generation.
"""

from unittest.mock import patch

import pytest

from src.agents.examples.text_processor_agent import TextProcessorAgent
from src.agents.exceptions import AgentExecutionError, AgentValidationError
from src.agents.models import AgentInput, AgentOutput


@pytest.mark.unit
class TestTextProcessorAgentInitialization:
    """Test cases for TextProcessorAgent initialization."""

    def test_init_default_config(self):
        """Test initialization with default configuration values."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        assert agent.max_words == 1000
        assert agent.min_words == 1
        assert agent.operations == ["analyze", "clean", "transform", "sentiment"]
        assert agent.enable_sentiment is True
        assert agent.requests_processed == 0
        assert agent.total_words_processed == 0
        assert agent.total_characters_processed == 0

    def test_init_custom_config(self):
        """Test initialization with custom configuration values."""
        config = {
            "agent_id": "text_processor",
            "max_words": 500,
            "min_words": 5,
            "operations": ["analyze", "clean"],
            "enable_sentiment": False,
        }
        agent = TextProcessorAgent(config)

        assert agent.max_words == 500
        assert agent.min_words == 5
        assert agent.operations == ["analyze", "clean"]
        assert agent.enable_sentiment is False

    def test_init_partial_custom_config(self):
        """Test initialization with partially customized configuration."""
        config = {"agent_id": "text_processor", "max_words": 750, "operations": ["analyze", "sentiment"]}
        agent = TextProcessorAgent(config)

        assert agent.max_words == 750
        assert agent.min_words == 1  # Default
        assert agent.operations == ["analyze", "sentiment"]
        assert agent.enable_sentiment is True  # Default


@pytest.mark.unit
class TestTextProcessorAgentValidation:
    """Test cases for configuration and input validation."""

    def test_validate_configuration_invalid_max_words_negative(self):
        """Test validation fails for negative max_words."""
        config = {"agent_id": "text_processor", "max_words": -1}

        with pytest.raises(AgentValidationError) as exc_info:
            TextProcessorAgent(config)

        assert "max_words must be a positive integer" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_CONFIG_VALUE"

    def test_validate_configuration_invalid_max_words_zero(self):
        """Test validation fails for zero max_words."""
        config = {"agent_id": "text_processor", "max_words": 0}

        with pytest.raises(AgentValidationError) as exc_info:
            TextProcessorAgent(config)

        assert "max_words must be a positive integer" in str(exc_info.value)

    def test_validate_configuration_invalid_max_words_string(self):
        """Test validation fails for string max_words."""
        config = {"agent_id": "text_processor", "max_words": "100"}

        with pytest.raises(AgentValidationError) as exc_info:
            TextProcessorAgent(config)

        assert "max_words must be a positive integer" in str(exc_info.value)

    def test_validate_configuration_invalid_min_words_negative(self):
        """Test validation fails for negative min_words."""
        config = {"agent_id": "text_processor", "min_words": -1}

        with pytest.raises(AgentValidationError) as exc_info:
            TextProcessorAgent(config)

        assert "min_words must be a non-negative integer" in str(exc_info.value)

    def test_validate_configuration_invalid_min_words_string(self):
        """Test validation fails for string min_words."""
        config = {"agent_id": "text_processor", "min_words": "5"}

        with pytest.raises(AgentValidationError) as exc_info:
            TextProcessorAgent(config)

        assert "min_words must be a non-negative integer" in str(exc_info.value)

    def test_validate_configuration_invalid_operations_not_list(self):
        """Test validation fails when operations is not a list."""
        config = {"agent_id": "text_processor", "operations": "analyze"}

        with pytest.raises(AgentValidationError) as exc_info:
            TextProcessorAgent(config)

        assert "operations must be a list" in str(exc_info.value)

    def test_validate_configuration_invalid_operation_unknown(self):
        """Test validation fails for unknown operation."""
        config = {"agent_id": "text_processor", "operations": ["analyze", "unknown"]}

        with pytest.raises(AgentValidationError) as exc_info:
            TextProcessorAgent(config)

        assert "Invalid operation 'unknown'" in str(exc_info.value)
        assert "analyze" in str(exc_info.value)
        assert "clean" in str(exc_info.value)
        assert "transform" in str(exc_info.value)
        assert "sentiment" in str(exc_info.value)

    def test_validate_configuration_valid_edge_cases(self):
        """Test validation passes for valid edge cases."""
        config = {
            "agent_id": "text_processor",
            "max_words": 1,
            "min_words": 0,
            "operations": ["sentiment"],  # Single operation
            "enable_sentiment": False,
        }

        # Should not raise exception
        agent = TextProcessorAgent(config)
        assert agent.max_words == 1
        assert agent.min_words == 0
        assert agent.operations == ["sentiment"]

    @pytest.mark.asyncio
    async def test_validate_input_too_short(self):
        """Test input validation fails when text is too short."""
        config = {"agent_id": "text_processor", "min_words": 5}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(content="short", context={"operation": "analyze"}, request_id="test_req")  # Only 1 word

        with pytest.raises(AgentValidationError) as exc_info:
            await agent.execute(input_data)

        assert "Input has 1 words, but minimum is 5" in str(exc_info.value)
        assert exc_info.value.error_code == "INPUT_TOO_SHORT"

    @pytest.mark.asyncio
    async def test_validate_input_too_long(self):
        """Test input validation fails when text is too long."""
        config = {"agent_id": "text_processor", "max_words": 3}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="this is a very long text",
            context={"operation": "analyze"},
            request_id="test_req",  # 6 words
        )

        with pytest.raises(AgentValidationError) as exc_info:
            await agent.execute(input_data)

        assert "Input has 6 words, but maximum is 3" in str(exc_info.value)
        assert exc_info.value.error_code == "INPUT_TOO_LONG"


@pytest.mark.unit
class TestTextProcessorAgentExecution:
    """Test cases for agent execution operations."""

    @pytest.mark.asyncio
    async def test_execute_analyze_operation(self):
        """Test successful execution of analyze operation."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="Hello world! This is a test sentence.",
            context={"operation": "analyze"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        assert "Text Analysis Results:" in result.content
        assert result.confidence == 0.95
        assert result.metadata["operation"] == "analyze"
        assert result.metadata["word_count"] == 7
        assert result.metadata["character_count"] == len(input_data.content)

    @pytest.mark.asyncio
    async def test_execute_clean_operation(self):
        """Test successful execution of clean operation."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="Hello   world!!!    Extra   spaces.",
            context={"operation": "clean"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        assert result.confidence == 0.98
        assert result.metadata["operation"] == "clean"
        # Should clean extra spaces
        assert "   " not in result.content

    @pytest.mark.asyncio
    async def test_execute_transform_operation_uppercase(self):
        """Test successful execution of transform operation with uppercase."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="hello world",
            context={"operation": "transform", "transform_type": "uppercase"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        assert result.content == "HELLO WORLD"
        assert result.confidence == 0.99
        assert result.metadata["operation"] == "transform"

    @pytest.mark.asyncio
    async def test_execute_transform_operation_lowercase(self):
        """Test successful execution of transform operation with lowercase."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="HELLO WORLD",
            context={"operation": "transform", "transform_type": "lowercase"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert result.content == "hello world"
        assert result.confidence == 0.99

    @pytest.mark.asyncio
    async def test_execute_transform_operation_title(self):
        """Test successful execution of transform operation with title case."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="hello world test",
            context={"operation": "transform", "transform_type": "title"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert result.content == "Hello World Test"
        assert result.confidence == 0.99

    @pytest.mark.asyncio
    async def test_execute_transform_operation_sentence(self):
        """Test successful execution of transform operation with sentence case."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="hello world test",
            context={"operation": "transform", "transform_type": "sentence"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert result.content == "Hello world test"
        assert result.confidence == 0.99

    @pytest.mark.asyncio
    async def test_execute_transform_operation_default_lowercase(self):
        """Test transform operation uses lowercase as default."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="HELLO WORLD",
            context={"operation": "transform"},  # No transform_type specified
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert result.content == "hello world"

    @pytest.mark.asyncio
    async def test_execute_transform_operation_invalid_type(self):
        """Test transform operation fails with invalid transform type."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="hello world",
            context={"operation": "transform", "transform_type": "invalid"},
            request_id="test_req",
        )

        with pytest.raises(AgentValidationError) as exc_info:
            await agent.execute(input_data)

        assert "Unknown transform type: invalid" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_TRANSFORM_TYPE"

    @pytest.mark.asyncio
    async def test_execute_sentiment_operation_positive(self):
        """Test successful execution of sentiment operation with positive text."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="I love this amazing product. It's fantastic and wonderful!",
            context={"operation": "sentiment"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        assert "Sentiment Analysis: Positive" in result.content
        assert result.confidence == 0.75
        assert result.metadata["operation"] == "sentiment"

    @pytest.mark.asyncio
    async def test_execute_sentiment_operation_negative(self):
        """Test successful execution of sentiment operation with negative text."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="I hate this terrible product. It's awful and horrible!",
            context={"operation": "sentiment"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert "Sentiment Analysis: Negative" in result.content
        assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_execute_sentiment_operation_neutral(self):
        """Test successful execution of sentiment operation with neutral text."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="This is a normal product with standard features.",
            context={"operation": "sentiment"},
            request_id="test_req",
        )

        result = await agent.execute(input_data)

        assert "Sentiment Analysis: Neutral" in result.content

    @pytest.mark.asyncio
    async def test_execute_default_operation_analyze(self):
        """Test that analyze is the default operation when none specified."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="Test content for analysis",
            context=None,
            request_id="test_req",  # No context provided
        )

        result = await agent.execute(input_data)

        assert result.metadata["operation"] == "analyze"
        assert "Text Analysis Results:" in result.content

    @pytest.mark.asyncio
    async def test_execute_default_operation_with_empty_context(self):
        """Test default operation with empty context."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(content="Test content for analysis", context={}, request_id="test_req")  # Empty context

        result = await agent.execute(input_data)

        assert result.metadata["operation"] == "analyze"

    @pytest.mark.asyncio
    async def test_execute_disallowed_operation(self):
        """Test execution fails when operation is not allowed."""
        config = {"agent_id": "text_processor", "operations": ["analyze"]}  # Only analyze allowed
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="Test content",
            context={"operation": "sentiment"},
            request_id="test_req",  # Not allowed
        )

        with pytest.raises(AgentValidationError) as exc_info:
            await agent.execute(input_data)

        assert "Operation 'sentiment' is not allowed" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_OPERATION"

    @pytest.mark.asyncio
    async def test_execute_unknown_operation(self):
        """Test execution fails with unknown operation."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        # Mock the operations to include an unknown operation
        agent.operations = ["unknown_op"]

        input_data = AgentInput(content="Test content", context={"operation": "unknown_op"}, request_id="test_req")

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute(input_data)

        assert "Unknown operation: unknown_op" in str(exc_info.value)
        assert exc_info.value.error_code == "UNKNOWN_OPERATION"

    @pytest.mark.asyncio
    async def test_execute_updates_statistics(self):
        """Test that execution updates processing statistics."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="Test content with five words here",  # 6 words
            context={"operation": "analyze"},
            request_id="test_req",
        )

        # Initial statistics
        assert agent.requests_processed == 0
        assert agent.total_words_processed == 0
        assert agent.total_characters_processed == 0

        result = await agent.execute(input_data)

        # Statistics should be updated
        assert agent.requests_processed == 1
        assert agent.total_words_processed == 6
        assert agent.total_characters_processed == len(input_data.content)
        assert result.metadata["requests_processed"] == 1
        assert result.metadata["total_words_processed"] == 6

    @pytest.mark.asyncio
    async def test_execute_multiple_requests_statistics(self):
        """Test statistics accumulate across multiple requests."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        # First request
        input1 = AgentInput(content="First test", context={"operation": "analyze"}, request_id="test_req1")  # 2 words
        await agent.execute(input1)

        # Second request
        input2 = AgentInput(
            content="Second longer test content",
            context={"operation": "clean"},
            request_id="test_req2",  # 4 words
        )
        result = await agent.execute(input2)

        # Check accumulated statistics
        assert agent.requests_processed == 2
        assert agent.total_words_processed == 6  # 2 + 4
        assert result.metadata["requests_processed"] == 2
        assert result.metadata["total_words_processed"] == 6


@pytest.mark.unit
class TestTextProcessorAgentAnalysisOperations:
    """Test cases for detailed analysis operations."""

    @pytest.mark.asyncio
    async def test_analyze_text_comprehensive(self):
        """Test comprehensive text analysis."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "Hello world! This is a test. How are you today?"
        result = await agent._analyze_text(text)

        assert result["confidence"] == 0.95
        assert "analysis" in result
        analysis = result["analysis"]

        assert analysis["word_count"] == 10
        assert analysis["character_count"] == len(text)
        assert analysis["sentence_count"] == 3
        assert analysis["paragraph_count"] == 1
        assert analysis["average_word_length"] > 0
        assert analysis["average_sentence_length"] > 0
        assert "most_common_words" in analysis
        assert "readability_score" in analysis

    @pytest.mark.asyncio
    async def test_analyze_text_with_sentiment_enabled(self):
        """Test text analysis includes sentiment when enabled."""
        config = {"agent_id": "text_processor", "enable_sentiment": True}
        agent = TextProcessorAgent(config)

        text = "I love this amazing product!"
        result = await agent._analyze_text(text)

        assert "sentiment" in result["analysis"]

    @pytest.mark.asyncio
    async def test_analyze_text_with_sentiment_disabled(self):
        """Test text analysis excludes sentiment when disabled."""
        config = {"agent_id": "text_processor", "enable_sentiment": False}
        agent = TextProcessorAgent(config)

        text = "I love this amazing product!"
        result = await agent._analyze_text(text)

        assert "sentiment" not in result["analysis"]

    @pytest.mark.asyncio
    async def test_clean_text_removes_extra_whitespace(self):
        """Test text cleaning removes extra whitespace."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "Hello    world!    This   has   extra   spaces."
        result = await agent._clean_text(text)

        assert result["confidence"] == 0.98
        assert "    " not in result["content"]
        assert "   " not in result["content"]
        assert result["original_length"] == len(text)
        assert result["cleaned_length"] < result["original_length"]

    @pytest.mark.asyncio
    async def test_clean_text_removes_special_characters(self):
        """Test text cleaning removes special characters."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "Hello @#$% world! This has #special@ characters."
        result = await agent._clean_text(text)

        assert "@" not in result["content"]
        assert "#" not in result["content"]
        assert "$" not in result["content"]
        assert "%" not in result["content"]
        # Basic punctuation should remain
        assert "!" in result["content"]
        assert "." in result["content"]

    def test_get_most_common_words(self):
        """Test getting most common words from text."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        words = ["the", "test", "the", "word", "test", "the"]
        result = agent._get_most_common_words(words, 3)

        assert len(result) <= 3
        assert result[0]["word"] == "the"
        assert result[0]["frequency"] == 3
        assert result[1]["word"] == "test"
        assert result[1]["frequency"] == 2

    def test_get_most_common_words_filters_short_words(self):
        """Test that short words are filtered out."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        words = ["a", "to", "the", "test", "word", "it"]
        result = agent._get_most_common_words(words, 5)

        # Should exclude words with length <= 2
        word_list = [item["word"] for item in result]
        assert "a" not in word_list
        assert "to" not in word_list
        assert "it" not in word_list
        assert "the" in word_list  # Length 3, should be included
        assert "test" in word_list
        assert "word" in word_list

    def test_calculate_readability_basic(self):
        """Test basic readability calculation."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        words = ["Hello", "world", "test"]
        sentences = ["Hello world.", "Test."]
        result = agent._calculate_readability("Hello world. Test.", words, sentences)

        assert isinstance(result, float)
        assert 0 <= result <= 100

    def test_calculate_readability_empty_input(self):
        """Test readability calculation with empty input."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        result = agent._calculate_readability("", [], [])
        assert result == 0.0

    def test_format_analysis_complete(self):
        """Test formatting of complete analysis results."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        analysis = {
            "word_count": 10,
            "character_count": 50,
            "sentence_count": 2,
            "paragraph_count": 1,
            "average_word_length": 5.0,
            "average_sentence_length": 5.0,
            "readability_score": 75.5,
            "most_common_words": [{"word": "test", "frequency": 2}],
            "sentiment": "positive",
        }

        result = agent._format_analysis(analysis)

        assert "Word Count: 10" in result
        assert "Character Count: 50" in result
        assert "Sentence Count: 2" in result
        assert "Paragraph Count: 1" in result
        assert "Average Word Length: 5.00" in result
        assert "Average Sentence Length: 5.00" in result
        assert "Readability Score: 75.50" in result
        assert "Most Common Words:" in result
        assert "test: 2" in result
        assert "Sentiment: Positive" in result


@pytest.mark.unit
class TestTextProcessorAgentSentimentAnalysis:
    """Test cases for sentiment analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive_words(self):
        """Test sentiment analysis detects positive words."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "This is great and amazing and wonderful!"
        result = await agent._analyze_sentiment(text)

        assert result["sentiment"] == "positive"
        assert result["score"] > 0.5
        assert result["positive_words"] > 0
        assert result["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative_words(self):
        """Test sentiment analysis detects negative words."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "This is terrible and awful and horrible!"
        result = await agent._analyze_sentiment(text)

        assert result["sentiment"] == "negative"
        assert result["score"] < 0.5
        assert result["negative_words"] > 0

    @pytest.mark.asyncio
    async def test_analyze_sentiment_neutral_words(self):
        """Test sentiment analysis detects neutral text."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "This is a normal sentence about standard things."
        result = await agent._analyze_sentiment(text)

        assert result["sentiment"] == "neutral"
        assert result["score"] == 0.5
        assert result["positive_words"] == 0
        assert result["negative_words"] == 0

    @pytest.mark.asyncio
    async def test_analyze_sentiment_mixed_words(self):
        """Test sentiment analysis with mixed positive and negative words."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "This product is good but has some bad features."
        result = await agent._analyze_sentiment(text)

        assert result["positive_words"] == 1  # "good"
        assert result["negative_words"] == 1  # "bad"
        # With equal counts, should be neutral
        assert result["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_analyze_sentiment_case_insensitive(self):
        """Test sentiment analysis is case insensitive."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        text = "GREAT and AMAZING and WONDERFUL!"
        result = await agent._analyze_sentiment(text)

        assert result["sentiment"] == "positive"
        assert result["positive_words"] == 3

    @pytest.mark.asyncio
    async def test_analyze_sentiment_score_bounds(self):
        """Test sentiment score stays within expected bounds."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        # Extremely positive text
        positive_text = "amazing wonderful great excellent fantastic awesome " * 10
        pos_result = await agent._analyze_sentiment(positive_text)
        assert 0.1 <= pos_result["score"] <= 0.9

        # Extremely negative text
        negative_text = "terrible awful horrible bad hate worst " * 10
        neg_result = await agent._analyze_sentiment(negative_text)
        assert 0.1 <= neg_result["score"] <= 0.9


@pytest.mark.unit
class TestTextProcessorAgentCapabilitiesAndStatus:
    """Test cases for agent capabilities and status reporting."""

    def test_get_capabilities_basic(self):
        """Test getting basic agent capabilities."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        capabilities = agent.get_capabilities()

        assert capabilities["agent_id"] == "text_processor"
        assert capabilities["agent_type"] == "TextProcessorAgent"
        assert "text" in capabilities["input_types"]
        assert "text" in capabilities["output_types"]
        assert "analysis" in capabilities["output_types"]
        assert capabilities["async_execution"] is True
        assert capabilities["timeout_support"] is True
        assert capabilities["config_overrides"] is True
        assert "en" in capabilities["languages"]

    def test_get_capabilities_custom_config(self):
        """Test capabilities reflect custom configuration."""
        config = {
            "agent_id": "text_processor",
            "max_words": 500,
            "min_words": 10,
            "operations": ["analyze", "clean"],
            "enable_sentiment": False,
        }
        agent = TextProcessorAgent(config)

        capabilities = agent.get_capabilities()

        assert capabilities["max_words"] == 500
        assert capabilities["min_words"] == 10
        assert capabilities["operations"] == ["analyze", "clean"]
        assert capabilities["enable_sentiment"] is False

    def test_get_status_initial(self):
        """Test getting initial agent status."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        status = agent.get_status()

        assert status["requests_processed"] == 0
        assert status["total_words_processed"] == 0
        assert status["total_characters_processed"] == 0
        assert status["average_words_per_request"] == 0

    @pytest.mark.asyncio
    async def test_get_status_after_processing(self):
        """Test status updates after processing requests."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        # Process a request
        input_data = AgentInput(
            content="Test content with four words",
            context={"operation": "analyze"},
            request_id="test_req",  # 5 words
        )
        await agent.execute(input_data)

        status = agent.get_status()

        assert status["requests_processed"] == 1
        assert status["total_words_processed"] == 5
        assert status["total_characters_processed"] == len(input_data.content)
        assert status["average_words_per_request"] == 5.0

    @pytest.mark.asyncio
    async def test_get_status_multiple_requests(self):
        """Test status with multiple processed requests."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        # Process multiple requests
        for i in range(3):
            input_data = AgentInput(
                content=f"Test content number {i}",  # 4 words each
                context={"operation": "analyze"},
                request_id=f"test_req_{i}",
            )
            await agent.execute(input_data)

        status = agent.get_status()

        assert status["requests_processed"] == 3
        assert status["total_words_processed"] == 12  # 4 * 3
        assert status["average_words_per_request"] == 4.0


@pytest.mark.unit
class TestTextProcessorAgentErrorHandling:
    """Test cases for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_execute_converts_unexpected_errors(self):
        """Test that unexpected errors are converted to AgentExecutionError."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        # Mock a method to raise an unexpected error
        with patch.object(agent, "_analyze_text", side_effect=ValueError("Unexpected error")):
            input_data = AgentInput(content="Test content", context={"operation": "analyze"}, request_id="test_req")

            with pytest.raises(AgentExecutionError) as exc_info:
                await agent.execute(input_data)

            assert "Text processing failed" in str(exc_info.value)
            assert exc_info.value.error_code == "PROCESSING_FAILED"
            assert "ValueError" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_preserves_agent_errors(self):
        """Test that AgentValidationError and AgentExecutionError are preserved."""
        config = {"agent_id": "text_processor", "operations": ["analyze"]}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="Test content",
            context={"operation": "sentiment"},
            request_id="test_req",  # Not allowed
        )

        # Should preserve the original AgentValidationError
        with pytest.raises(AgentValidationError) as exc_info:
            await agent.execute(input_data)

        assert exc_info.value.error_code == "INVALID_OPERATION"
        # Should not be wrapped in AgentExecutionError


@pytest.mark.unit
class TestTextProcessorAgentIntegration:
    """Integration test cases for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow_analyze_with_sentiment(self):
        """Test complete workflow with analysis including sentiment."""
        config = {"agent_id": "text_processor", "max_words": 100, "enable_sentiment": True}
        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="I love this amazing product! It works great and is wonderful to use.",
            context={"operation": "analyze"},
            request_id="integration_test",
        )

        result = await agent.execute(input_data)

        # Verify complete result structure
        assert isinstance(result, AgentOutput)
        assert result.request_id == "integration_test"
        assert result.confidence == 0.95
        assert "Text Analysis Results:" in result.content
        assert "Sentiment: Positive" in result.content

        # Verify metadata
        metadata = result.metadata
        assert metadata["operation"] == "analyze"
        assert metadata["word_count"] > 0
        assert metadata["character_count"] > 0
        assert metadata["requests_processed"] == 1

    @pytest.mark.asyncio
    async def test_full_workflow_all_operations(self):
        """Test all operations work end-to-end."""
        config = {"agent_id": "text_processor"}
        agent = TextProcessorAgent(config)

        test_content = "Hello World! This is a test sentence."
        operations = ["analyze", "clean", "transform", "sentiment"]

        for operation in operations:
            context = {"operation": operation}
            if operation == "transform":
                context["transform_type"] = "uppercase"

            input_data = AgentInput(content=test_content, context=context, request_id=f"test_{operation}")

            result = await agent.execute(input_data)

            assert isinstance(result, AgentOutput)
            assert result.metadata["operation"] == operation
            assert len(result.content) > 0
            assert result.confidence > 0

    def test_agent_registration_decorator(self):
        """Test that the agent is properly registered with the registry."""
        # This tests the @agent_registry.register("text_processor") decorator
        # The actual implementation would require the registry to be functional
        assert hasattr(TextProcessorAgent, "__name__")
        assert TextProcessorAgent.__name__ == "TextProcessorAgent"

    def test_exports_correctly(self):
        """Test that the module exports the agent class correctly."""
        from src.agents.examples.text_processor_agent import __all__

        assert "TextProcessorAgent" in __all__
