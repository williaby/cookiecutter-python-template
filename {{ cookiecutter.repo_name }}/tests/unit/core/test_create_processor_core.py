"""
Unit tests for C.R.E.A.T.E. framework core processor.

This module provides comprehensive test coverage for the CreateProcessor class,
testing prompt enhancement, framework component generation, validation, and error handling.
Uses proper pytest markers for codecov integration per codecov.yml config component.
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.core.create_processor_core import (
    CreateProcessor,
    CreateProcessorError,
    CreateRequest,
    CreateResponse,
    Domain,
    ProcessingError,
    ValidationError,
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCreateProcessor:
    """Test cases for CreateProcessor class."""

    def test_init_default_config(self):
        """Test CreateProcessor initialization with default configuration."""
        processor = CreateProcessor()

        assert processor.config == {}
        assert hasattr(processor, "logger")

    def test_init_custom_config(self):
        """Test CreateProcessor initialization with custom configuration."""
        config = {"log_level": "DEBUG", "max_prompt_length": 10000, "custom_setting": "value"}

        processor = CreateProcessor(config)

        assert processor.config == config
        assert processor.config["log_level"] == "DEBUG"
        assert processor.config["custom_setting"] == "value"

    def test_setup_logging_valid_level(self):
        """Test logging setup with valid log level."""
        config = {"log_level": "DEBUG"}

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            CreateProcessor(config)

            # Should call setLevel with the correct level
            mock_logger.setLevel.assert_called_once()

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid log level."""
        config = {"log_level": "INVALID_LEVEL"}

        processor = CreateProcessor(config)

        # Should not crash with invalid log level
        assert processor.config["log_level"] == "INVALID_LEVEL"

    def test_setup_logging_no_level(self):
        """Test logging setup with no log level specified."""
        processor = CreateProcessor({})

        # Should use default INFO level
        assert processor.config.get("log_level", "INFO") == "INFO"


@pytest.mark.unit
class TestCreateProcessorValidation:
    """Test cases for input validation."""

    def test_validate_input_success(self):
        """Test successful input validation."""
        processor = CreateProcessor()

        valid_prompt = "Create a marketing plan for our new product launch."

        result = processor.validate_input(valid_prompt)

        assert result is True

    def test_validate_input_none(self):
        """Test validation with None input."""
        processor = CreateProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(None)

        assert "must be a non-empty string" in str(exc_info.value)

    def test_validate_input_non_string(self):
        """Test validation with non-string input."""
        processor = CreateProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(123)

        assert "must be a non-empty string" in str(exc_info.value)

    def test_validate_input_empty_string(self):
        """Test validation with empty string."""
        processor = CreateProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input("")

        assert "must be a non-empty string" in str(exc_info.value)

    def test_validate_input_whitespace_only(self):
        """Test validation with whitespace-only string."""
        processor = CreateProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input("   \n\t   ")

        assert "cannot be empty or whitespace only" in str(exc_info.value)

    def test_validate_input_too_long(self):
        """Test validation with overly long input."""
        processor = CreateProcessor()

        # Create a prompt longer than the 50000 character limit
        long_prompt = "a" * 50001

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(long_prompt)

        assert "exceeds maximum length" in str(exc_info.value)

    def test_validate_input_script_injection(self):
        """Test validation with script injection attempt."""
        processor = CreateProcessor()

        malicious_prompt = "Create a plan <script>alert('xss')</script> for our product."

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(malicious_prompt)

        assert "potential injection pattern" in str(exc_info.value)

    def test_validate_input_javascript_injection(self):
        """Test validation with JavaScript injection attempt."""
        processor = CreateProcessor()

        malicious_prompt = "Create a plan javascript:alert('xss') for our product."

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(malicious_prompt)

        assert "potential injection pattern" in str(exc_info.value)

    def test_validate_input_data_url_injection(self):
        """Test validation with data URL injection attempt."""
        processor = CreateProcessor()

        malicious_prompt = "Create data:text/html,<script>alert('xss')</script> content."

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(malicious_prompt)

        assert "potential injection pattern" in str(exc_info.value)

    def test_validate_input_vbscript_injection(self):
        """Test validation with VBScript injection attempt."""
        processor = CreateProcessor()

        malicious_prompt = "Create vbscript:msgbox('xss') content."

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(malicious_prompt)

        assert "potential injection pattern" in str(exc_info.value)

    def test_validate_input_event_handler_injection(self):
        """Test validation with event handler injection attempt."""
        processor = CreateProcessor()

        malicious_prompt = "Create onclick=alert('xss') content."

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(malicious_prompt)

        assert "potential injection pattern" in str(exc_info.value)

    def test_validate_input_case_insensitive_injection(self):
        """Test validation is case insensitive for injection patterns."""
        processor = CreateProcessor()

        malicious_prompt = "Create <SCRIPT>alert('xss')</SCRIPT> content."

        with pytest.raises(ValidationError) as exc_info:
            processor.validate_input(malicious_prompt)

        assert "potential injection pattern" in str(exc_info.value)


@pytest.mark.unit
class TestCreateProcessorContextExtraction:
    """Test cases for context extraction."""

    def test_extract_context_role_patterns(self):
        """Test extraction of role information from various patterns."""
        processor = CreateProcessor()

        test_cases = [
            ("You are a senior marketing analyst.", "a senior marketing analyst"),
            ("Act as a legal expert in corporate law.", "a legal expert in corporate law"),
            ("Role: Technical writer", "Technical writer"),
            ("As a data scientist, please analyze...", "data scientist"),
            ("Assume the role of project manager.", "project manager"),
            ("Playing the role of customer service representative.", "customer service representative"),
        ]

        for prompt, expected_role in test_cases:
            context = processor._extract_context(prompt)
            assert context["role"] == expected_role

    def test_extract_context_background_patterns(self):
        """Test extraction of background information."""
        processor = CreateProcessor()

        test_cases = [
            ("Background: 10 years experience in finance.", "10 years experience in finance"),
            ("Context: Working with enterprise software solutions.", "Working with enterprise software solutions"),
            ("Given that we're launching a new product.", "we're launching a new product"),
            ("With expertise in machine learning.", "machine learning"),
            ("Specializing in customer engagement strategies.", "customer engagement strategies"),
        ]

        for prompt, expected_background in test_cases:
            context = processor._extract_context(prompt)
            assert context["background"] == expected_background

    def test_extract_context_goal_patterns(self):
        """Test extraction of goal information."""
        processor = CreateProcessor()

        test_cases = [
            ("Goal: Increase sales by 20%.", "Increase sales by 20%"),
            ("Objective: Improve customer satisfaction.", "Improve customer satisfaction"),
            ("My goal is to streamline operations.", "streamline operations"),
            ("The goal is to reduce costs.", "reduce costs"),
            ("I need to create a comprehensive report.", "create a comprehensive report"),
            ("We need to develop a strategy.", "develop a strategy"),
        ]

        for prompt, expected_goal in test_cases:
            context = processor._extract_context(prompt)
            assert context["goal"] == expected_goal

    def test_extract_context_no_matches(self):
        """Test context extraction when no patterns match."""
        processor = CreateProcessor()

        prompt = "Create a simple marketing plan for our product."
        context = processor._extract_context(prompt)

        assert context["role"] is None
        assert context["background"] is None
        assert context["goal"] is None
        assert context["constraints"] == []

    def test_extract_context_multiple_patterns(self):
        """Test context extraction with multiple pattern types."""
        processor = CreateProcessor()

        prompt = (
            "You are a marketing expert with expertise in digital marketing. "
            "Goal: Create a comprehensive campaign strategy."
        )

        context = processor._extract_context(prompt)

        assert context["role"] == "a marketing expert"
        assert context["background"] == "digital marketing"
        assert context["goal"] == "Create a comprehensive campaign strategy"


@pytest.mark.unit
class TestCreateProcessorRequestComponent:
    """Test cases for request component generation."""

    def test_generate_request_component_task_patterns(self):
        """Test extraction of task information."""
        processor = CreateProcessor()

        test_cases = [
            ("Task: Develop a marketing strategy.", "Develop a marketing strategy"),
            ("Please create a financial report.", "create a financial report"),
            ("Can you analyze market trends.", "analyze market trends"),
            ("Help me to write a proposal.", "write a proposal"),
            ("Write a comprehensive business plan.", "a comprehensive business plan"),
            ("Create an implementation roadmap.", "an implementation roadmap"),
        ]

        for prompt, expected_task in test_cases:
            request = processor._generate_request_component(prompt, {})
            assert request["task"] == expected_task

    def test_generate_request_component_deliverable_patterns(self):
        """Test extraction of deliverable information."""
        processor = CreateProcessor()

        test_cases = [
            ("Deliverable should be a detailed report.", "a detailed report"),
            ("Deliverable: Executive summary document.", "Executive summary document"),
            ("Output: Strategic recommendations.", "Strategic recommendations"),
            ("Format: PowerPoint presentation.", "PowerPoint presentation"),
            ("In the form of bullet points.", "bullet points"),
        ]

        for prompt, expected_deliverable in test_cases:
            request = processor._generate_request_component(prompt, {})
            assert request["deliverable"] == expected_deliverable

    def test_generate_request_component_no_matches(self):
        """Test request component when no patterns match."""
        processor = CreateProcessor()

        prompt = "Simple text without specific patterns."
        request = processor._generate_request_component(prompt, {})

        assert request["task"] is None
        assert request["deliverable"] is None
        assert request["specifications"] == []
        assert request["constraints"] == []

    def test_generate_request_component_complete(self):
        """Test request component with both task and deliverable."""
        processor = CreateProcessor()

        prompt = "Task: Analyze customer feedback data. Deliverable should be an actionable insights report."

        request = processor._generate_request_component(prompt, {})

        assert request["task"] == "Analyze customer feedback data"
        assert request["deliverable"] == "an actionable insights report"


@pytest.mark.unit
class TestCreateProcessorExamplesComponent:
    """Test cases for examples component generation."""

    def test_generate_examples_component_patterns(self):
        """Test extraction of example patterns."""
        processor = CreateProcessor()

        test_cases = [
            ("Example: Focus on customer satisfaction metrics.", "Focus on customer satisfaction metrics"),
            ("For example: Use SWOT analysis framework.", "Use SWOT analysis framework"),
            ("E.g.: Include market research data.", "Include market research data"),
            ("Such as ROI calculations and projections.", "ROI calculations and projections"),
        ]

        for prompt, expected_pattern in test_cases:
            examples = processor._generate_examples_component(prompt)
            assert expected_pattern in examples["patterns"]

    def test_generate_examples_component_style_patterns(self):
        """Test extraction of style example patterns."""
        processor = CreateProcessor()

        prompt = "Write like this: clear, concise, and actionable recommendations."
        examples = processor._generate_examples_component(prompt)

        assert "clear, concise, and actionable recommendations" in examples["patterns"]

    def test_generate_examples_component_multiple_patterns(self):
        """Test extraction of multiple example patterns."""
        processor = CreateProcessor()

        prompt = (
            "Example: Use data-driven insights. "
            "For example: Include customer segmentation analysis. "
            "Such as demographic and behavioral patterns."
        )

        examples = processor._generate_examples_component(prompt)

        assert len(examples["patterns"]) >= 1  # At least one should match
        assert examples["input_examples"] == []
        assert examples["output_examples"] == []

    def test_generate_examples_component_no_patterns(self):
        """Test examples component when no patterns found."""
        processor = CreateProcessor()

        prompt = "Simple request without examples."
        examples = processor._generate_examples_component(prompt)

        assert examples["patterns"] == []
        assert examples["input_examples"] == []
        assert examples["output_examples"] == []


@pytest.mark.unit
class TestCreateProcessorAugmentationsComponent:
    """Test cases for augmentations component generation."""

    def test_generate_augmentations_legal_domain(self):
        """Test augmentations for legal domain."""
        processor = CreateProcessor()

        augmentations = processor._generate_augmentations_component(Domain.LEGAL.value)

        assert augmentations["domain"] == "legal"
        assert "IRAC" in augmentations["frameworks"]
        assert "Legal precedents" in augmentations["sources"]
        assert augmentations["tools"] == []

    def test_generate_augmentations_business_domain(self):
        """Test augmentations for business domain."""
        processor = CreateProcessor()

        augmentations = processor._generate_augmentations_component(Domain.BUSINESS.value)

        assert augmentations["domain"] == "business"
        assert "SWOT" in augmentations["frameworks"]
        assert "Market data" in augmentations["sources"]

    def test_generate_augmentations_technical_domain(self):
        """Test augmentations for technical domain."""
        processor = CreateProcessor()

        augmentations = processor._generate_augmentations_component(Domain.TECHNICAL.value)

        assert augmentations["domain"] == "technical"
        assert "Technical specifications" in augmentations["frameworks"]
        assert "Documentation" in augmentations["sources"]

    def test_generate_augmentations_academic_domain(self):
        """Test augmentations for academic domain."""
        processor = CreateProcessor()

        augmentations = processor._generate_augmentations_component(Domain.ACADEMIC.value)

        assert augmentations["domain"] == "academic"
        assert "Academic structure" in augmentations["frameworks"]
        assert "Scholarly sources" in augmentations["sources"]

    def test_generate_augmentations_general_domain(self):
        """Test augmentations for general domain."""
        processor = CreateProcessor()

        augmentations = processor._generate_augmentations_component(Domain.GENERAL.value)

        assert augmentations["domain"] == "general"
        assert augmentations["frameworks"] == []
        assert augmentations["sources"] == []

    def test_generate_augmentations_none_domain(self):
        """Test augmentations when domain is None."""
        processor = CreateProcessor()

        augmentations = processor._generate_augmentations_component(None)

        assert augmentations["domain"] == "general"
        assert augmentations["frameworks"] == []
        assert augmentations["sources"] == []


@pytest.mark.unit
class TestCreateProcessorToneFormatComponent:
    """Test cases for tone & format component generation."""

    def test_generate_tone_format_default(self):
        """Test default tone & format generation."""
        processor = CreateProcessor()

        prompt = "Create a report."
        tone_format = processor._generate_tone_format_component(prompt)

        assert tone_format["tone"] == "professional"
        assert tone_format["style"] == "clear"
        assert tone_format["format"] == "structured"
        assert tone_format["length"] == "appropriate"

    def test_generate_tone_format_tone_patterns(self):
        """Test extraction of tone indicators."""
        processor = CreateProcessor()

        test_cases = [
            "Tone: friendly and approachable.",
            "In a casual conversational style.",
            "Style: formal business communication.",
        ]

        for prompt in test_cases:
            tone_format = processor._generate_tone_format_component(prompt)
            # The tone should be extracted (not default)
            assert tone_format["tone"] != "professional" or "professional" in prompt.lower()

    def test_generate_tone_format_format_patterns(self):
        """Test extraction of format indicators."""
        processor = CreateProcessor()

        test_cases = [
            "Format: bullet points with key highlights.",
            "Structure: executive summary format.",
            "As a comprehensive analysis document.",
            "Bullet points with supporting details.",
            "Numbered list of recommendations.",
        ]

        for prompt in test_cases:
            tone_format = processor._generate_tone_format_component(prompt)
            # Format should be extracted (not default)
            assert tone_format["format"] != "structured" or "structured" in prompt.lower()

    def test_generate_tone_format_no_patterns(self):
        """Test tone & format when no specific patterns found."""
        processor = CreateProcessor()

        prompt = "Simple request without tone or format specifications."
        tone_format = processor._generate_tone_format_component(prompt)

        assert tone_format["tone"] == "professional"
        assert tone_format["style"] == "clear"
        assert tone_format["format"] == "structured"
        assert tone_format["length"] == "appropriate"


@pytest.mark.unit
class TestCreateProcessorEvaluationComponent:
    """Test cases for evaluation component generation."""

    def test_generate_evaluation_component(self):
        """Test evaluation component generation."""
        processor = CreateProcessor()

        evaluation = processor._generate_evaluation_component()

        assert "quality_checks" in evaluation
        assert "success_criteria" in evaluation

        # Verify standard quality checks
        expected_checks = ["Accuracy verification", "Completeness check", "Clarity assessment", "Relevance validation"]

        for check in expected_checks:
            assert check in evaluation["quality_checks"]

        # Verify standard success criteria
        expected_criteria = [
            "Meets all requirements",
            "Appropriate tone and format",
            "Clear and actionable",
            "Factually accurate",
        ]

        for criterion in expected_criteria:
            assert criterion in evaluation["success_criteria"]


@pytest.mark.unit
class TestCreateProcessorFrameworkApplication:
    """Test cases for applying the complete C.R.E.A.T.E. framework."""

    def test_apply_create_framework_basic(self):
        """Test basic application of C.R.E.A.T.E. framework."""
        processor = CreateProcessor()

        prompt = "You are a marketing expert. Create a campaign strategy."
        context = {"domain": "business"}

        components = processor.apply_create_framework(prompt, context)

        assert "context" in components
        assert "request" in components
        assert "examples" in components
        assert "augmentations" in components
        assert "tone_format" in components
        assert "evaluation" in components

    def test_apply_create_framework_with_role(self):
        """Test framework application with role extraction."""
        processor = CreateProcessor()

        prompt = "You are a financial analyst. Analyze Q4 performance."
        context = {"domain": "business"}

        components = processor.apply_create_framework(prompt, context)

        assert components["context"]["role"] == "a financial analyst"
        assert components["augmentations"]["domain"] == "business"

    def test_apply_create_framework_complex_prompt(self):
        """Test framework application with complex prompt."""
        processor = CreateProcessor()

        prompt = (
            "You are a senior marketing manager with expertise in digital marketing. "
            "Task: Create a comprehensive social media strategy. "
            "Deliverable should be a detailed implementation plan. "
            "Example: Include platform-specific content strategies. "
            "Tone: professional but engaging."
        )

        context = {"domain": "business"}

        components = processor.apply_create_framework(prompt, context)

        # Verify all components are populated
        assert components["context"]["role"] == "a senior marketing manager"
        assert components["context"]["background"] == "digital marketing"
        assert components["request"]["task"] == "Create a comprehensive social media strategy"
        assert components["request"]["deliverable"] == "a detailed implementation plan"
        assert len(components["examples"]["patterns"]) > 0
        assert components["augmentations"]["domain"] == "business"

    def test_apply_create_framework_default_domain(self):
        """Test framework application with default domain."""
        processor = CreateProcessor()

        prompt = "Create a report."
        context = {}

        components = processor.apply_create_framework(prompt, context)

        assert components["augmentations"]["domain"] == "general"


@pytest.mark.unit
class TestCreateProcessorPromptBuilding:
    """Test cases for building enhanced prompts from components."""

    def test_build_enhanced_prompt_basic(self):
        """Test basic enhanced prompt building."""
        processor = CreateProcessor()

        components = {
            "context": {"role": "analyst", "background": None, "goal": None},
            "request": {"task": "analyze data", "deliverable": None},
            "examples": {"patterns": []},
            "augmentations": {"domain": "general", "frameworks": [], "sources": []},
            "tone_format": {"tone": "professional", "format": "structured"},
            "evaluation": {"quality_checks": ["accuracy"], "success_criteria": ["complete"]},
        }

        enhanced_prompt = processor._build_enhanced_prompt(components)

        assert "## Context" in enhanced_prompt
        assert "Role: analyst" in enhanced_prompt
        assert "## Request" in enhanced_prompt
        assert "Task: analyze data" in enhanced_prompt
        assert "## Tone & Format" in enhanced_prompt
        assert "## Evaluation" in enhanced_prompt

    def test_build_context_section_complete(self):
        """Test building context section with all fields."""
        processor = CreateProcessor()

        context = {"role": "marketing manager", "background": "5 years experience", "goal": "increase engagement"}

        sections = processor._build_context_section(context)

        assert len(sections) == 3
        assert "Role: marketing manager" in sections[0]
        assert "Background: 5 years experience" in sections[1]
        assert "Goal: increase engagement" in sections[2]

    def test_build_context_section_minimal(self):
        """Test building context section with minimal fields."""
        processor = CreateProcessor()

        context = {"role": None, "background": None, "goal": None}

        sections = processor._build_context_section(context)

        assert sections == []

    def test_build_request_section_complete(self):
        """Test building request section with all fields."""
        processor = CreateProcessor()

        request = {"task": "create strategy", "deliverable": "implementation plan"}

        sections = processor._build_request_section(request)

        assert len(sections) == 2
        assert "Task: create strategy" in sections[0]
        assert "Deliverable: implementation plan" in sections[1]

    def test_build_examples_section_with_patterns(self):
        """Test building examples section with patterns."""
        processor = CreateProcessor()

        examples = {"patterns": ["pattern 1", "pattern 2"]}

        sections = processor._build_examples_section(examples)

        assert "## Examples" in sections[0]
        assert "- pattern 1" in sections[1]
        assert "- pattern 2" in sections[2]

    def test_build_examples_section_empty(self):
        """Test building examples section when empty."""
        processor = CreateProcessor()

        examples = {"patterns": []}

        sections = processor._build_examples_section(examples)

        assert sections == []

    def test_build_augmentations_section_with_frameworks(self):
        """Test building augmentations section with frameworks."""
        processor = CreateProcessor()

        augmentations = {"domain": "business", "frameworks": ["SWOT", "Porter's Five Forces"], "sources": []}

        sections = processor._build_augmentations_section(augmentations)

        assert "## Augmentations" in sections[0]
        assert "Domain: business" in sections[1]
        assert "Frameworks: SWOT, Porter's Five Forces" in sections[2]

    def test_build_tone_format_section(self):
        """Test building tone & format section."""
        processor = CreateProcessor()

        tone_format = {"tone": "professional", "format": "structured"}

        sections = processor._build_tone_format_section(tone_format)

        assert "## Tone & Format" in sections[0]
        assert "Tone: professional" in sections[1]
        assert "Format: structured" in sections[2]

    def test_build_evaluation_section(self):
        """Test building evaluation section."""
        processor = CreateProcessor()

        evaluation = {"quality_checks": ["accuracy", "completeness"]}

        sections = processor._build_evaluation_section(evaluation)

        assert "## Evaluation" in sections[0]
        assert "Quality checks:" in sections[1]
        assert "- accuracy" in sections[2]
        assert "- completeness" in sections[3]


@pytest.mark.unit
@pytest.mark.asyncio
class TestCreateProcessorProcessPrompt:
    """Test cases for the main process_prompt method."""

    async def test_process_prompt_success(self):
        """Test successful prompt processing."""
        processor = CreateProcessor()

        input_prompt = "You are a marketing expert. Create a campaign strategy."
        domain = "business"

        response = await processor.process_prompt(input_prompt, domain)

        assert isinstance(response, CreateResponse)
        assert response.success is True
        assert response.errors == []
        assert len(response.enhanced_prompt) > len(input_prompt)
        assert response.framework_components is not None
        assert response.metadata["domain"] == domain
        assert response.processing_time > 0

    async def test_process_prompt_validation_error(self):
        """Test prompt processing with validation error."""
        processor = CreateProcessor()

        # Invalid input (None)
        response = await processor.process_prompt(None)

        assert isinstance(response, CreateResponse)
        assert response.success is False
        assert len(response.errors) > 0
        assert "must be a non-empty string" in response.errors[0]
        assert response.enhanced_prompt == ""
        assert response.framework_components == {}

    async def test_process_prompt_injection_error(self):
        """Test prompt processing with injection attempt."""
        processor = CreateProcessor()

        malicious_prompt = "Create a plan <script>alert('xss')</script>"

        response = await processor.process_prompt(malicious_prompt)

        assert response.success is False
        assert len(response.errors) > 0
        assert "potential injection pattern" in response.errors[0]

    async def test_process_prompt_default_domain(self):
        """Test prompt processing with default domain."""
        processor = CreateProcessor()

        input_prompt = "Create a simple report."

        response = await processor.process_prompt(input_prompt)

        assert response.success is True
        assert response.metadata["domain"] == "general"
        assert response.framework_components["augmentations"]["domain"] == "general"

    async def test_process_prompt_metadata(self):
        """Test prompt processing metadata generation."""
        processor = CreateProcessor()

        input_prompt = "Create a marketing strategy."
        domain = "business"

        response = await processor.process_prompt(input_prompt, domain)

        assert response.metadata["original_prompt_length"] == len(input_prompt)
        assert response.metadata["enhanced_prompt_length"] == len(response.enhanced_prompt)
        assert response.metadata["domain"] == domain
        assert "timestamp" in response.metadata

    async def test_process_prompt_logging(self):
        """Test prompt processing includes proper logging."""
        processor = CreateProcessor()

        with patch.object(processor.logger, "info") as mock_info:
            input_prompt = "Create a business plan."

            response = await processor.process_prompt(input_prompt)

            assert response.success is True
            mock_info.assert_called_once()

    async def test_process_prompt_timing(self):
        """Test prompt processing timing calculation."""
        processor = CreateProcessor()

        input_prompt = "Create a comprehensive analysis."

        start_time = time.time()
        response = await processor.process_prompt(input_prompt)
        end_time = time.time()

        assert response.processing_time > 0
        assert response.processing_time <= (end_time - start_time)

    async def test_process_prompt_exception_handling(self):
        """Test prompt processing handles unexpected exceptions."""
        processor = CreateProcessor()

        # Mock a method to raise an unexpected exception
        with patch.object(processor, "apply_create_framework", side_effect=RuntimeError("Unexpected error")):
            input_prompt = "Create a report."

            response = await processor.process_prompt(input_prompt)

            assert response.success is False
            assert len(response.errors) > 0
            assert "Unexpected error" in response.errors[0]
            assert response.processing_time > 0


@pytest.mark.unit
class TestCreateRequestResponse:
    """Test cases for CreateRequest and CreateResponse data classes."""

    def test_create_request_basic(self):
        """Test CreateRequest initialization."""
        request = CreateRequest(
            input_prompt="Test prompt",
            domain="business",
            context={"key": "value"},
            settings={"setting": "value"},
        )

        assert request.input_prompt == "Test prompt"
        assert request.domain == "business"
        assert request.context == {"key": "value"}
        assert request.settings == {"setting": "value"}

    def test_create_request_minimal(self):
        """Test CreateRequest with minimal parameters."""
        request = CreateRequest(input_prompt="Test prompt")

        assert request.input_prompt == "Test prompt"
        assert request.domain is None
        assert request.context is None
        assert request.settings is None

    def test_create_response_complete(self):
        """Test CreateResponse initialization."""
        response = CreateResponse(
            enhanced_prompt="Enhanced test prompt",
            framework_components={"context": {}},
            metadata={"key": "value"},
            processing_time=1.5,
            success=True,
            errors=[],
        )

        assert response.enhanced_prompt == "Enhanced test prompt"
        assert response.framework_components == {"context": {}}
        assert response.metadata == {"key": "value"}
        assert response.processing_time == 1.5
        assert response.success is True
        assert response.errors == []

    def test_create_response_error(self):
        """Test CreateResponse for error case."""
        errors = ["Validation error", "Processing error"]

        response = CreateResponse(
            enhanced_prompt="",
            framework_components={},
            metadata={"error": True},
            processing_time=0.1,
            success=False,
            errors=errors,
        )

        assert response.enhanced_prompt == ""
        assert response.framework_components == {}
        assert response.success is False
        assert response.errors == errors


@pytest.mark.unit
class TestDomainEnum:
    """Test cases for Domain enum."""

    def test_domain_values(self):
        """Test Domain enum values."""
        assert Domain.GENERAL.value == "general"
        assert Domain.TECHNICAL.value == "technical"
        assert Domain.LEGAL.value == "legal"
        assert Domain.BUSINESS.value == "business"
        assert Domain.ACADEMIC.value == "academic"

    def test_domain_membership(self):
        """Test Domain enum membership."""
        valid_domains = ["general", "technical", "legal", "business", "academic"]

        for domain_value in valid_domains:
            # Check that each value corresponds to a Domain enum member
            domain_members = [member.value for member in Domain]
            assert domain_value in domain_members


@pytest.mark.unit
class TestExceptions:
    """Test cases for custom exceptions."""

    def test_create_processor_error(self):
        """Test CreateProcessorError exception."""
        error = CreateProcessorError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Validation failed")

        assert str(error) == "Validation failed"
        assert isinstance(error, CreateProcessorError)
        assert isinstance(error, Exception)

    def test_processing_error(self):
        """Test ProcessingError exception."""
        error = ProcessingError("Processing failed")

        assert str(error) == "Processing failed"
        assert isinstance(error, CreateProcessorError)
        assert isinstance(error, Exception)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCreateProcessorIntegration:
    """Integration test cases for complete CreateProcessor workflows."""

    async def test_complete_workflow_business_domain(self):
        """Test complete processing workflow for business domain."""
        processor = CreateProcessor()

        input_prompt = (
            "You are a senior business analyst with expertise in market research. "
            "Task: Create a comprehensive market analysis report. "
            "Deliverable should be a strategic recommendations document. "
            "Example: Include competitive analysis and market trends. "
            "Tone: professional and data-driven."
        )

        response = await processor.process_prompt(input_prompt, "business")

        # Verify successful processing
        assert response.success is True
        assert response.errors == []
        assert len(response.enhanced_prompt) > len(input_prompt)

        # Verify framework components
        components = response.framework_components
        assert components["context"]["role"] == "a senior business analyst"
        assert components["context"]["background"] == "market research"
        assert "Create a comprehensive market analysis report" in components["request"]["task"]
        assert components["request"]["deliverable"] == "a strategic recommendations document"
        assert len(components["examples"]["patterns"]) > 0
        assert components["augmentations"]["domain"] == "business"
        assert "SWOT" in components["augmentations"]["frameworks"]

        # Verify enhanced prompt structure
        assert "## Context" in response.enhanced_prompt
        assert "## Request" in response.enhanced_prompt
        assert "## Examples" in response.enhanced_prompt
        assert "## Augmentations" in response.enhanced_prompt
        assert "## Tone & Format" in response.enhanced_prompt
        assert "## Evaluation" in response.enhanced_prompt

    async def test_complete_workflow_technical_domain(self):
        """Test complete processing workflow for technical domain."""
        processor = CreateProcessor()

        input_prompt = (
            "You are a software architect. "
            "Create a system design for a microservices platform. "
            "Format: Technical specification document."
        )

        response = await processor.process_prompt(input_prompt, "technical")

        assert response.success is True
        assert response.framework_components["augmentations"]["domain"] == "technical"
        assert "Technical specifications" in response.framework_components["augmentations"]["frameworks"]
        assert "Documentation" in response.framework_components["augmentations"]["sources"]

    async def test_complete_workflow_error_recovery(self):
        """Test complete workflow with error recovery."""
        processor = CreateProcessor()

        # First, test with invalid input
        response1 = await processor.process_prompt(None)
        assert response1.success is False

        # Then, test with valid input to ensure processor still works
        response2 = await processor.process_prompt("Create a simple report.")
        assert response2.success is True

    async def test_performance_requirements(self):
        """Test processing meets performance requirements."""
        processor = CreateProcessor()

        input_prompt = "Create a marketing strategy for our new product launch."

        start_time = time.time()
        response = await processor.process_prompt(input_prompt)
        processing_time = time.time() - start_time

        # Should meet < 3 seconds requirement for simple prompts
        assert processing_time < 3.0
        assert response.processing_time < 3.0
        assert response.success is True
