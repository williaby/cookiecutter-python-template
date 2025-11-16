"""Unit tests for core CreateProcessor implementation.

This module contains comprehensive unit tests for the CreateProcessor class
in Phase 1 Issue 4, focusing on the essential C.R.E.A.T.E. framework functionality.
"""

from unittest.mock import patch

import pytest

from src.core.create_processor_core import (
    CreateProcessor,
    CreateResponse,
    Domain,
    ValidationError,
)


class TestCreateProcessor:
    """Test cases for CreateProcessor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.processor = CreateProcessor()
        self.sample_prompt = "Help me write a professional email to a client about project delays"

    def test_init_default_config(self):
        """Test CreateProcessor initialization with default config."""
        processor = CreateProcessor()
        assert processor.config == {}
        assert processor.logger is not None

    def test_init_custom_config(self):
        """Test CreateProcessor initialization with custom config."""
        config = {"log_level": "DEBUG", "custom_setting": "value"}
        processor = CreateProcessor(config)
        assert processor.config == config

    def test_validate_input_valid_prompt(self):
        """Test input validation with valid prompt."""
        valid_prompt = "This is a valid prompt"
        result = self.processor.validate_input(valid_prompt)
        assert result is True

    def test_validate_input_empty_string(self):
        """Test input validation with empty string."""
        with pytest.raises(ValidationError, match="Input prompt must be a non-empty string"):
            self.processor.validate_input("")

    def test_validate_input_none(self):
        """Test input validation with None."""
        with pytest.raises(ValidationError, match="Input prompt must be a non-empty string"):
            self.processor.validate_input(None)

    def test_validate_input_whitespace_only(self):
        """Test input validation with whitespace only."""
        with pytest.raises(ValidationError, match="Input prompt cannot be empty or whitespace only"):
            self.processor.validate_input("   \n\t  ")

    def test_validate_input_too_long(self):
        """Test input validation with excessively long prompt."""
        long_prompt = "a" * 50001
        with pytest.raises(ValidationError, match="Input prompt exceeds maximum length"):
            self.processor.validate_input(long_prompt)

    def test_validate_input_injection_script(self):
        """Test input validation with script injection."""
        malicious_prompt = "Help me <script>alert('xss')</script> write code"
        with pytest.raises(ValidationError, match="Input contains potential injection pattern"):
            self.processor.validate_input(malicious_prompt)

    def test_validate_input_injection_javascript(self):
        """Test input validation with javascript injection."""
        malicious_prompt = "Help me javascript:alert('xss') write code"
        with pytest.raises(ValidationError, match="Input contains potential injection pattern"):
            self.processor.validate_input(malicious_prompt)

    def test_extract_context_with_role(self):
        """Test context extraction with role information."""
        prompt = "You are a senior software engineer. Help me design a database schema."
        context = self.processor._extract_context(prompt)
        assert context["role"] == "a senior software engineer"
        assert context["background"] is None
        assert context["goal"] is None

    def test_extract_context_with_background(self):
        """Test context extraction with background information."""
        prompt = "Given that you have expertise in machine learning, explain neural networks."
        context = self.processor._extract_context(prompt)
        assert context["background"] == "you have expertise in machine learning"

    def test_extract_context_with_goal(self):
        """Test context extraction with goal information."""
        prompt = "I need to create a comprehensive report on market trends."
        context = self.processor._extract_context(prompt)
        assert context["goal"] == "create a comprehensive report on market trends"

    def test_extract_context_empty_prompt(self):
        """Test context extraction with empty prompt."""
        context = self.processor._extract_context("")
        assert context["role"] is None
        assert context["background"] is None
        assert context["goal"] is None

    def test_generate_request_component_with_task(self):
        """Test request component generation with task."""
        prompt = "Please write a technical documentation for our API."
        context = {}
        request = self.processor._generate_request_component(prompt, context)
        assert request["task"] == "write a technical documentation for our API"

    def test_generate_request_component_with_deliverable(self):
        """Test request component generation with deliverable."""
        prompt = "Create a report. The deliverable should be a PDF document."
        context = {}
        request = self.processor._generate_request_component(prompt, context)
        assert request["deliverable"] == "a PDF document"

    def test_generate_examples_component_with_patterns(self):
        """Test examples component generation with patterns."""
        prompt = "Write like this: Hello Mr. Smith, Thank you for your email."
        examples = self.processor._generate_examples_component(prompt)
        assert "Hello Mr. Smith, Thank you for your email" in examples["patterns"]

    def test_generate_examples_component_empty_prompt(self):
        """Test examples component generation with empty prompt."""
        examples = self.processor._generate_examples_component("")
        assert examples["patterns"] == []

    def test_generate_augmentations_component_legal_domain(self):
        """Test augmentations component generation for legal domain."""
        augmentations = self.processor._generate_augmentations_component(Domain.LEGAL.value)
        assert augmentations["domain"] == "legal"
        assert "IRAC" in augmentations["frameworks"]
        assert "Legal precedents" in augmentations["sources"]

    def test_generate_augmentations_component_business_domain(self):
        """Test augmentations component generation for business domain."""
        augmentations = self.processor._generate_augmentations_component(Domain.BUSINESS.value)
        assert augmentations["domain"] == "business"
        assert "SWOT" in augmentations["frameworks"]
        assert "Market data" in augmentations["sources"]

    def test_generate_augmentations_component_technical_domain(self):
        """Test augmentations component generation for technical domain."""
        augmentations = self.processor._generate_augmentations_component(Domain.TECHNICAL.value)
        assert augmentations["domain"] == "technical"
        assert "Technical specifications" in augmentations["frameworks"]
        assert "Documentation" in augmentations["sources"]

    def test_generate_augmentations_component_general_domain(self):
        """Test augmentations component generation for general domain."""
        augmentations = self.processor._generate_augmentations_component(None)
        assert augmentations["domain"] == "general"
        assert augmentations["frameworks"] == []

    def test_generate_tone_format_component_with_tone(self):
        """Test tone format component generation with tone."""
        prompt = "Write in a formal tone for the board meeting."
        tone_format = self.processor._generate_tone_format_component(prompt)
        assert tone_format["tone"] == "tone for the board meeting"

    def test_generate_tone_format_component_default(self):
        """Test tone format component generation with defaults."""
        tone_format = self.processor._generate_tone_format_component("")
        assert tone_format["tone"] == "professional"
        assert tone_format["style"] == "clear"
        assert tone_format["format"] == "structured"

    def test_generate_evaluation_component(self):
        """Test evaluation component generation."""
        evaluation = self.processor._generate_evaluation_component()
        assert "Accuracy verification" in evaluation["quality_checks"]
        assert "Meets all requirements" in evaluation["success_criteria"]

    def test_apply_create_framework_basic(self):
        """Test applying C.R.E.A.T.E. framework to basic prompt."""
        prompt = "You are a teacher. Help me explain photosynthesis to students."
        context = {"domain": "academic"}

        components = self.processor.apply_create_framework(prompt, context)

        assert "context" in components
        assert "request" in components
        assert "examples" in components
        assert "augmentations" in components
        assert "tone_format" in components
        assert "evaluation" in components

        assert components["context"]["role"] == "a teacher"
        assert components["augmentations"]["domain"] == "academic"

    def test_build_enhanced_prompt_complete(self):
        """Test building enhanced prompt with complete components."""
        components = {
            "context": {
                "role": "technical writer",
                "background": "API documentation",
                "goal": "create clear documentation",
            },
            "request": {"task": "write API documentation", "deliverable": "markdown file"},
            "examples": {"patterns": ["GET /api/users", "POST /api/users"]},
            "augmentations": {"domain": "technical", "frameworks": ["OpenAPI"]},
            "tone_format": {"tone": "professional", "format": "structured"},
            "evaluation": {"quality_checks": ["Accuracy verification", "Completeness check"]},
        }

        enhanced_prompt = self.processor._build_enhanced_prompt(components)

        assert "## Context" in enhanced_prompt
        assert "Role: technical writer" in enhanced_prompt
        assert "## Request" in enhanced_prompt
        assert "Task: write API documentation" in enhanced_prompt
        assert "## Examples" in enhanced_prompt
        assert "## Augmentations" in enhanced_prompt
        assert "## Tone & Format" in enhanced_prompt
        assert "## Evaluation" in enhanced_prompt

    def test_build_enhanced_prompt_minimal(self):
        """Test building enhanced prompt with minimal components."""
        components = {
            "context": {"role": None, "background": None, "goal": None},
            "request": {"task": None, "deliverable": None},
            "examples": {"patterns": []},
            "augmentations": {"domain": "general", "frameworks": []},
            "tone_format": {"tone": "professional", "format": "structured"},
            "evaluation": {"quality_checks": ["Accuracy verification"]},
        }

        enhanced_prompt = self.processor._build_enhanced_prompt(components)

        assert "## Tone & Format" in enhanced_prompt
        assert "## Evaluation" in enhanced_prompt
        assert "## Context" not in enhanced_prompt
        assert "## Request" not in enhanced_prompt
        assert "## Examples" not in enhanced_prompt

    @pytest.mark.asyncio
    async def test_process_prompt_success(self):
        """Test successful prompt processing."""
        prompt = "You are a data scientist. Help me analyze customer churn data."

        response = await self.processor.process_prompt(prompt, "business")

        assert isinstance(response, CreateResponse)
        assert response.success is True
        assert response.errors == []
        assert response.enhanced_prompt != ""
        assert response.processing_time > 0
        assert response.framework_components is not None
        assert response.metadata["domain"] == "business"

    @pytest.mark.asyncio
    async def test_process_prompt_validation_error(self):
        """Test prompt processing with validation error."""
        response = await self.processor.process_prompt("", "general")

        assert isinstance(response, CreateResponse)
        assert response.success is False
        assert len(response.errors) > 0
        assert response.enhanced_prompt == ""
        assert "Input prompt must be a non-empty string" in response.errors[0]

    @pytest.mark.asyncio
    async def test_process_prompt_no_domain(self):
        """Test prompt processing without domain specification."""
        prompt = "Help me write a letter"

        response = await self.processor.process_prompt(prompt)

        assert isinstance(response, CreateResponse)
        assert response.success is True
        assert response.metadata["domain"] == "general"

    @pytest.mark.asyncio
    async def test_process_prompt_processing_time(self):
        """Test that processing time is recorded correctly."""
        prompt = "Create a business plan for a startup"

        response = await self.processor.process_prompt(prompt, "business")

        assert response.processing_time > 0
        assert response.processing_time < 5  # Should be fast for core implementation

    @pytest.mark.asyncio
    async def test_process_prompt_metadata_complete(self):
        """Test that response metadata is complete."""
        prompt = "Write a technical specification"

        response = await self.processor.process_prompt(prompt, "technical")

        assert "original_prompt_length" in response.metadata
        assert "enhanced_prompt_length" in response.metadata
        assert "domain" in response.metadata
        assert "timestamp" in response.metadata
        assert response.metadata["original_prompt_length"] == len(prompt)
        assert response.metadata["enhanced_prompt_length"] == len(response.enhanced_prompt)

    @pytest.mark.asyncio
    async def test_process_prompt_logging(self):
        """Test that processing generates appropriate log messages."""
        prompt = "Help me write documentation"

        with patch.object(self.processor.logger, "info") as mock_log:
            response = await self.processor.process_prompt(prompt, "technical")

            assert response.success is True
            mock_log.assert_called()

    @pytest.mark.asyncio
    async def test_process_prompt_error_logging(self):
        """Test that processing errors are logged appropriately."""
        with patch.object(self.processor.logger, "error") as mock_log:
            response = await self.processor.process_prompt("", "general")

            assert response.success is False
            mock_log.assert_called()
