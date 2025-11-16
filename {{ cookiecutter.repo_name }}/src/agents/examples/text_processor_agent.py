"""
Example Text Processor Agent for PromptCraft-Hybrid.

This module demonstrates the complete usage of the agent framework,
including registration, configuration, error handling, and execution.
It serves as a reference implementation for creating new agents.

The agent performs basic text processing operations like:
- Text cleaning and normalization
- Word count and character analysis
- Simple sentiment analysis
- Text transformations (uppercase, lowercase, title case)

Architecture:
    The agent inherits from BaseAgent and implements the execute() method.
    It demonstrates configuration injection, error handling, and metadata
    generation. The agent is registered with the global registry for
    automatic discovery.

Example:
    ```python
    from src.agents.examples.text_processor_agent import TextProcessorAgent
    from src.agents.models import AgentInput
    from src.agents.registry import agent_registry

    # Get agent from registry
    agent = agent_registry.get_agent("text_processor", {"agent_id": "text_processor"})

    # Create input
    input_data = AgentInput(
        content="Hello World! This is a test.",
        context={"operation": "analyze"},
        config_overrides={"max_words": 10}
    )

    # Process
    result = await agent.process(input_data)
    print(result.content)  # Analysis results
    ```

Dependencies:
    - src.agents.base_agent: BaseAgent interface
    - src.agents.models: AgentInput, AgentOutput data models
    - src.agents.registry: Agent registration system
    - src.agents.exceptions: Error handling classes

Called by:
    - Example usage in documentation
    - Integration tests in tests/integration/
    - Framework demonstration scripts

Complexity: O(n) where n is the length of input text
"""

import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentExecutionError, AgentValidationError
from src.agents.models import AgentInput, AgentOutput
from src.agents.registry import agent_registry


@agent_registry.register("text_processor")
class TextProcessorAgent(BaseAgent):
    """
    Example agent that demonstrates text processing capabilities.

    This agent showcases all features of the PromptCraft agent framework:
    - Configuration injection and validation
    - Runtime configuration overrides
    - Error handling and logging
    - Metadata generation
    - Capability reporting

    The agent supports various text processing operations:
    - analyze: Perform text analysis (word count, character count, etc.)
    - clean: Clean and normalize text
    - transform: Apply text transformations (case changes)
    - sentiment: Basic sentiment analysis

    Configuration Parameters:
        max_words (int): Maximum number of words to process (default: 1000)
        min_words (int): Minimum number of words required (default: 1)
        operations (List[str]): Allowed operations (default: all)
        enable_sentiment (bool): Whether to enable sentiment analysis (default: True)

    Example:
        ```python
        config = {
            "agent_id": "text_processor",
            "max_words": 500,
            "operations": ["analyze", "clean"],
            "enable_sentiment": False
        }

        agent = TextProcessorAgent(config)

        input_data = AgentInput(
            content="Hello World! This is a test.",
            context={"operation": "analyze"}
        )

        result = await agent.process(input_data)
        ```
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the TextProcessorAgent.

        Args:
            config: Configuration dictionary containing agent parameters

        Raises:
            AgentConfigurationError: If required configuration is missing
        """
        # Set default configuration values before calling super().__init__()
        # This ensures they're available during _validate_configuration()
        self.max_words = config.get("max_words", 1000)
        self.min_words = config.get("min_words", 1)
        self.operations = config.get("operations", ["analyze", "clean", "transform", "sentiment"])
        self.enable_sentiment = config.get("enable_sentiment", True)

        # Initialize processing statistics
        self.requests_processed = 0
        self.total_words_processed = 0
        self.total_characters_processed = 0

        super().__init__(config)

        self.logger.info("TextProcessorAgent initialized", extra={"config": self.config})

    def _validate_configuration(self) -> None:
        """
        Validate agent-specific configuration.

        Raises:
            AgentConfigurationError: If configuration is invalid
        """
        super()._validate_configuration()

        # Validate max_words
        if not isinstance(self.max_words, int) or self.max_words <= 0:
            raise AgentValidationError(
                message="max_words must be a positive integer",
                error_code="INVALID_CONFIG_VALUE",
                context={"max_words": self.max_words},
                agent_id=self.agent_id,
            )

        # Validate min_words
        if not isinstance(self.min_words, int) or self.min_words < 0:
            raise AgentValidationError(
                message="min_words must be a non-negative integer",
                error_code="INVALID_CONFIG_VALUE",
                context={"min_words": self.min_words},
                agent_id=self.agent_id,
            )

        # Validate operations
        valid_operations = {"analyze", "clean", "transform", "sentiment"}
        if not isinstance(self.operations, list):
            raise AgentValidationError(
                message="operations must be a list",
                error_code="INVALID_CONFIG_VALUE",
                context={"operations": self.operations},
                agent_id=self.agent_id,
            )

        for op in self.operations:
            if op not in valid_operations:
                raise AgentValidationError(
                    message=f"Invalid operation '{op}'. Must be one of: {valid_operations}",
                    error_code="INVALID_CONFIG_VALUE",
                    context={"operation": op, "valid_operations": list(valid_operations)},
                    agent_id=self.agent_id,
                )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """
        Execute text processing on the provided input.

        Args:
            agent_input: Input data containing text to process

        Returns:
            AgentOutput: Processing results with metadata

        Raises:
            AgentValidationError: If input is invalid
            AgentExecutionError: If processing fails
        """
        try:
            # Validate input
            self._validate_input(agent_input)

            # Extract operation from context
            operation = agent_input.context.get("operation", "analyze") if agent_input.context else "analyze"

            # Check if operation is allowed
            if operation not in self.operations:
                raise AgentValidationError(
                    message=f"Operation '{operation}' is not allowed",
                    error_code="INVALID_OPERATION",
                    context={"operation": operation, "allowed_operations": self.operations},
                    agent_id=self.agent_id,
                    request_id=agent_input.request_id,
                )

            # Process the text based on operation
            if operation == "analyze":
                result = await self._analyze_text(agent_input.content)
            elif operation == "clean":
                result = await self._clean_text(agent_input.content)
            elif operation == "transform":
                transform_type = (
                    agent_input.context.get("transform_type", "lowercase") if agent_input.context else "lowercase"
                )
                result = await self._transform_text(agent_input.content, transform_type)
            elif operation == "sentiment":
                result = await self._analyze_sentiment(agent_input.content)
            else:
                raise AgentExecutionError(
                    message=f"Unknown operation: {operation}",
                    error_code="UNKNOWN_OPERATION",
                    context={"operation": operation},
                    agent_id=self.agent_id,
                    request_id=agent_input.request_id,
                )

            # Update statistics
            self.requests_processed += 1
            self.total_words_processed += len(agent_input.content.split())
            self.total_characters_processed += len(agent_input.content)

            # Create output with metadata
            metadata = {
                "operation": operation,
                "input_length": len(agent_input.content),
                "word_count": len(agent_input.content.split()),
                "character_count": len(agent_input.content),
                "requests_processed": self.requests_processed,
                "total_words_processed": self.total_words_processed,
                "total_characters_processed": self.total_characters_processed,
            }

            return self._create_output(
                content=result["content"],
                metadata=metadata,
                confidence=result["confidence"],
                request_id=agent_input.request_id,
            )

        except Exception as e:
            # Convert any unexpected errors to AgentExecutionError
            if not isinstance(e, AgentValidationError | AgentExecutionError):
                raise AgentExecutionError(
                    message=f"Text processing failed: {e!s}",
                    error_code="PROCESSING_FAILED",
                    context={"error": str(e), "error_type": type(e).__name__},
                    agent_id=self.agent_id,
                    request_id=agent_input.request_id,
                ) from e
            raise

    def _validate_input(self, agent_input: AgentInput) -> None:
        """
        Validate the input data.

        Args:
            agent_input: Input data to validate

        Raises:
            AgentValidationError: If input is invalid
        """
        # Check word count limits
        word_count = len(agent_input.content.split())

        if word_count < self.min_words:
            raise AgentValidationError(
                message=f"Input has {word_count} words, but minimum is {self.min_words}",
                error_code="INPUT_TOO_SHORT",
                context={"word_count": word_count, "min_words": self.min_words},
                agent_id=self.agent_id,
                request_id=agent_input.request_id,
            )

        if word_count > self.max_words:
            raise AgentValidationError(
                message=f"Input has {word_count} words, but maximum is {self.max_words}",
                error_code="INPUT_TOO_LONG",
                context={"word_count": word_count, "max_words": self.max_words},
                agent_id=self.agent_id,
                request_id=agent_input.request_id,
            )

    async def _analyze_text(self, text: str) -> dict[str, Any]:
        """
        Analyze text and return statistics.

        Args:
            text: Text to analyze

        Returns:
            Dict containing analysis results
        """
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        paragraphs = text.split("\n\n")

        # Calculate statistics
        analysis = {
            "word_count": len(words),
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(" ", "")),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "average_sentence_length": len(words) / len([s for s in sentences if s.strip()]) if sentences else 0,
            "most_common_words": self._get_most_common_words(words, 5),
            "readability_score": self._calculate_readability(text, words, sentences),
        }

        # Add sentiment analysis if enabled
        if self.enable_sentiment:
            sentiment = await self._analyze_sentiment(text)
            analysis["sentiment"] = sentiment["sentiment"]

        return {
            "content": f"Text Analysis Results:\n\n{self._format_analysis(analysis)}",
            "confidence": 0.95,
            "analysis": analysis,
        }

    async def _clean_text(self, text: str) -> dict[str, Any]:
        """
        Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Dict containing cleaned text
        """
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", text)

        # Remove special characters (keep basic punctuation)
        cleaned = re.sub(r"[^\w\s.!?,-]", "", cleaned)

        # Normalize case (optional)
        cleaned = cleaned.strip()

        return {"content": cleaned, "confidence": 0.98, "original_length": len(text), "cleaned_length": len(cleaned)}

    async def _transform_text(self, text: str, transform_type: str) -> dict[str, Any]:
        """
        Transform text based on specified type.

        Args:
            text: Text to transform
            transform_type: Type of transformation (uppercase, lowercase, title, sentence)

        Returns:
            Dict containing transformed text
        """
        if transform_type == "uppercase":
            transformed = text.upper()
        elif transform_type == "lowercase":
            transformed = text.lower()
        elif transform_type == "title":
            transformed = text.title()
        elif transform_type == "sentence":
            transformed = text.capitalize()
        else:
            raise AgentValidationError(
                message=f"Unknown transform type: {transform_type}",
                error_code="INVALID_TRANSFORM_TYPE",
                context={"transform_type": transform_type},
                agent_id=self.agent_id,
            )

        return {"content": transformed, "confidence": 0.99, "transform_type": transform_type}

    async def _analyze_sentiment(self, text: str) -> dict[str, Any]:
        """
        Perform basic sentiment analysis.

        Args:
            text: Text to analyze

        Returns:
            Dict containing sentiment analysis
        """
        # Simple sentiment analysis based on word lists
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "love",
            "like",
            "enjoy",
            "happy",
            "joy",
            "success",
            "win",
            "best",
            "awesome",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "dislike",
            "sad",
            "angry",
            "fail",
            "failure",
            "worst",
            "problem",
            "issue",
            "error",
            "wrong",
        }

        words = text.lower().split()
        # Clean punctuation from words before checking sentiment
        cleaned_words = [word.strip(".,!?;:\"'()[]{}") for word in words]
        positive_count = sum(1 for word in cleaned_words if word in positive_words)
        negative_count = sum(1 for word in cleaned_words if word in negative_words)

        # Calculate sentiment score
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(0.9, 0.5 + (positive_count - negative_count) / len(cleaned_words))
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(0.1, 0.5 - (negative_count - positive_count) / len(cleaned_words))
        else:
            sentiment = "neutral"
            score = 0.5

        return {
            "content": f"Sentiment Analysis: {sentiment.title()} (score: {score:.2f})",
            "confidence": 0.75,  # Lower confidence for simple algorithm
            "sentiment": sentiment,
            "score": score,
            "positive_words": positive_count,
            "negative_words": negative_count,
        }

    def _get_most_common_words(self, words: list[str], count: int) -> list[dict[str, Any]]:
        """
        Get the most common words in the text.

        Args:
            words: List of words
            count: Number of top words to return

        Returns:
            List of dictionaries with word and frequency
        """
        # Simple word frequency counting
        word_freq: dict[str, int] = {}
        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            # Skip short words (length <= 2 characters)
            min_word_length = 2
            if len(word_lower) > min_word_length:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [{"word": word, "frequency": freq} for word, freq in sorted_words[:count]]

    def _calculate_readability(self, _text: str, words: list[str], sentences: list[str]) -> float:
        """
        Calculate a simple readability score.

        Args:
            text: Original text
            words: List of words
            sentences: List of sentences

        Returns:
            Readability score (0-100, higher is easier to read)
        """
        if not words or not sentences:
            return 0.0

        # Simple readability calculation
        avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Simplified Flesch Reading Ease approximation
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 4.7)

        return max(0.0, min(100.0, score))

    def _format_analysis(self, analysis: dict[str, Any]) -> str:
        """
        Format analysis results for display.

        Args:
            analysis: Analysis results dictionary

        Returns:
            Formatted string representation
        """
        formatted = []

        # Basic statistics
        formatted.append(f"Word Count: {analysis['word_count']}")
        formatted.append(f"Character Count: {analysis['character_count']}")
        formatted.append(f"Sentence Count: {analysis['sentence_count']}")
        formatted.append(f"Paragraph Count: {analysis['paragraph_count']}")
        formatted.append(f"Average Word Length: {analysis['average_word_length']:.2f}")
        formatted.append(f"Average Sentence Length: {analysis['average_sentence_length']:.2f}")
        formatted.append(f"Readability Score: {analysis['readability_score']:.2f}")

        # Most common words
        if analysis["most_common_words"]:
            formatted.append("\nMost Common Words:")
            for word_info in analysis["most_common_words"]:
                formatted.append(f"  - {word_info['word']}: {word_info['frequency']}")

        # Sentiment (if available)
        if "sentiment" in analysis:
            formatted.append(f"\nSentiment: {analysis['sentiment'].title()}")

        return "\n".join(formatted)

    def get_capabilities(self) -> dict[str, Any]:
        """
        Get the agent's capabilities.

        Returns:
            Dict containing capability information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": "TextProcessorAgent",
            "input_types": ["text"],
            "output_types": ["text", "analysis"],
            "operations": self.operations,
            "max_words": self.max_words,
            "min_words": self.min_words,
            "enable_sentiment": self.enable_sentiment,
            "async_execution": True,
            "timeout_support": True,
            "config_overrides": True,
            "languages": ["en"],  # English only for this example
            "processing_types": ["analyze", "clean", "transform", "sentiment"],
        }

    def get_status(self) -> dict[str, Any]:
        """
        Get the agent's current status.

        Returns:
            Dict containing status information
        """
        status = super().get_status()
        status.update(
            {
                "requests_processed": self.requests_processed,
                "total_words_processed": self.total_words_processed,
                "total_characters_processed": self.total_characters_processed,
                "average_words_per_request": (
                    self.total_words_processed / self.requests_processed if self.requests_processed > 0 else 0
                ),
            },
        )
        return status


# Export the agent class
__all__ = ["TextProcessorAgent"]
