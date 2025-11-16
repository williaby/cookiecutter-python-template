"""Core CreateProcessor implementation for C.R.E.A.T.E. framework prompt processing.

This module implements the essential CreateProcessor class for Phase 1 Issue 4,
providing the basic API contract for C.R.E.A.T.E. framework prompt enhancement.

API Contract:
- process_prompt(input_prompt: str, domain: str = None) -> dict
- apply_create_framework(prompt: str, context: dict) -> dict
- validate_input(input_prompt: str) -> bool

Security Features:
- Input validation preventing injection attacks
- Defensive security practices with logging
- Type safety with comprehensive type hints

Performance Targets:
- < 3 seconds response time for simple prompts
- Comprehensive error handling and logging
- Basic timing metrics for performance monitoring
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Try to import structlog, fall back to standard logging if not available
try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# Core exceptions for create processor
class CreateProcessorError(Exception):
    """Base exception for CreateProcessor errors."""


class ValidationError(CreateProcessorError):
    """Exception raised for validation failures."""


class ProcessingError(CreateProcessorError):
    """Exception raised for processing failures."""


# Domain definitions for prompt processing
class Domain(Enum):
    """Available domains for prompt processing."""

    GENERAL = "general"
    TECHNICAL = "technical"
    LEGAL = "legal"
    BUSINESS = "business"
    ACADEMIC = "academic"


@dataclass
class CreateRequest:
    """Request object for C.R.E.A.T.E. framework processing."""

    input_prompt: str
    domain: str | None = None
    context: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None


@dataclass
class CreateResponse:
    """Response object for C.R.E.A.T.E. framework processing."""

    enhanced_prompt: str
    framework_components: dict[str, Any]
    metadata: dict[str, Any]
    processing_time: float
    success: bool
    errors: list[str]


class CreateProcessor:
    """Core processor for C.R.E.A.T.E. framework prompt enhancement.

    This class implements the basic functionality required for Phase 1 Issue 4,
    focusing on the six C.R.E.A.T.E. components without advanced features.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the CreateProcessor.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logger
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get("log_level", "INFO")
        if hasattr(logging, log_level):
            logging.getLogger().setLevel(getattr(logging, log_level))

    def validate_input(self, input_prompt: str) -> bool:
        """Validate input prompt for security and basic requirements.

        Args:
            input_prompt: The input prompt to validate.

        Returns:
            True if input is valid, False otherwise.

        Raises:
            ValidationError: If input fails validation.
        """
        if not input_prompt or not isinstance(input_prompt, str):
            raise ValidationError("Input prompt must be a non-empty string")

        if len(input_prompt.strip()) == 0:
            raise ValidationError("Input prompt cannot be empty or whitespace only")

        max_prompt_length = 50000  # Reasonable limit
        if len(input_prompt) > max_prompt_length:
            raise ValidationError(f"Input prompt exceeds maximum length of {max_prompt_length} characters")

        # Check for potential injection patterns
        injection_patterns = [
            r"<script\s*>",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"on\w+\s*=",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, input_prompt, re.IGNORECASE):
                raise ValidationError(f"Input contains potential injection pattern: {pattern}")

        return True

    def _extract_context(self, prompt: str) -> dict[str, Any]:
        """Extract context information from the prompt.

        Args:
            prompt: The input prompt.

        Returns:
            Dictionary containing context information.
        """
        context: dict[str, Any] = {
            "role": None,
            "background": None,
            "goal": None,
            "constraints": [],
        }

        # Simple pattern matching for context extraction
        role_patterns = [
            r"(?:you are|act as|role:)\s*((?:a\s+)?.*?)(?:\s+with|\.|,|$)",
            r"(?:as a)\s+(.*?)(?:\s+with|\.|,|$)",
            r"(?:assume the role of|playing the role of)\s+(.*?)(?:\s+with|\.|,|$)",
        ]

        for pattern in role_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                context["role"] = match.group(1).strip()
                break

        # Extract background information
        background_patterns = [
            r"(?:background:|context:)\s*(.*?)(?:\.|$)",
            r"(?:given that|considering that)\s+(.*?)(?:,|\.)",
            r"(?:with expertise in|specializing in)\s+(.*?)(?:\.|,|$)",
        ]

        for pattern in background_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                context["background"] = match.group(1).strip()
                break

        # Extract goal information
        goal_patterns = [
            r"(?:goal:|objective:|aim:|purpose:)\s*(.*?)(?:\.|$)",
            r"(?:my goal is to|the goal is to)\s+(.*?)(?:\.|$)",
            r"(?:I need to|we need to)\s+(.*?)(?:\.|$)",
        ]

        for pattern in goal_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                context["goal"] = match.group(1).strip()
                break

        return context

    def _generate_request_component(self, prompt: str, _context: dict[str, Any]) -> dict[str, Any]:
        """Generate the Request component of C.R.E.A.T.E. framework.

        Args:
            prompt: The input prompt.
            context: Extracted context information.

        Returns:
            Dictionary containing request component.
        """
        request: dict[str, Any] = {
            "task": None,
            "deliverable": None,
            "specifications": [],
            "constraints": [],
        }

        # Extract task information
        task_patterns = [
            r"(?:task:|please|can you|help me to) (.*?)(?:\.|$)",
            r"(?:write|create|generate|produce) (.*?)(?:\.|$)",
        ]

        for pattern in task_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                request["task"] = match.group(1).strip()
                break

        # Extract deliverable information
        deliverable_patterns = [
            r"(?:deliverable should be|deliverable:|output:|result should be)\s+(.*?)(?:\.|$)",
            r"(?:format:|in the form of)\s+(.*?)(?:\.|$)",
        ]

        for pattern in deliverable_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                request["deliverable"] = match.group(1).strip()
                break

        return request

    def _generate_examples_component(self, prompt: str) -> dict[str, Any]:
        """Generate the Examples component of C.R.E.A.T.E. framework.

        Args:
            prompt: The input prompt.

        Returns:
            Dictionary containing examples component.
        """
        examples: dict[str, Any] = {
            "input_examples": [],
            "output_examples": [],
            "patterns": [],
        }

        # Look for example patterns in the prompt
        example_patterns = [
            r"(?:example:|for example:|e\.g\.:|such as)\s+(.*?)(?:\.|$)",
            r"(?:write like this:|like this:|similar to)\s+(.*)",  # Greedy capture for examples
        ]

        for pattern in example_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                captured_text = match.group(1).strip().rstrip(".")
                examples["patterns"].append(captured_text)

        return examples

    def _generate_augmentations_component(self, domain: str | None) -> dict[str, Any]:
        """Generate the Augmentations component of C.R.E.A.T.E. framework.

        Args:
            domain: The domain for augmentation.

        Returns:
            Dictionary containing augmentations component.
        """
        augmentations = {
            "domain": domain or "general",
            "frameworks": [],
            "tools": [],
            "sources": [],
        }

        # Add domain-specific augmentations
        if domain == Domain.LEGAL.value:
            augmentations["frameworks"] = ["IRAC"]
            augmentations["sources"] = ["Legal precedents"]
        elif domain == Domain.BUSINESS.value:
            augmentations["frameworks"] = ["SWOT"]
            augmentations["sources"] = ["Market data"]
        elif domain == Domain.TECHNICAL.value:
            augmentations["frameworks"] = ["Technical specifications"]
            augmentations["sources"] = ["Documentation"]
        elif domain == Domain.ACADEMIC.value:
            augmentations["frameworks"] = ["Academic structure"]
            augmentations["sources"] = ["Scholarly sources"]

        return augmentations

    def _generate_tone_format_component(self, prompt: str) -> dict[str, Any]:
        """Generate the Tone & Format component of C.R.E.A.T.E. framework.

        Args:
            prompt: The input prompt.

        Returns:
            Dictionary containing tone and format component.
        """
        tone_format = {
            "tone": "professional",
            "style": "clear",
            "format": "structured",
            "length": "appropriate",
        }

        # Extract tone indicators
        tone_patterns = [
            r"(?:tone:|in a|style:)\s*(?:formal|informal|casual|professional|friendly)?\s*(.*?)(?:\.|,|$)",
        ]

        for pattern in tone_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                tone_format["tone"] = match.group(1).strip()
                break

        # Extract format indicators
        format_patterns = [
            r"(?:format:|structure:|as a) (.*?)(?:\.|,|$)",
            r"(?:bullet points|numbered list|paragraph|essay) (.*?)(?:\.|,|$)",
        ]

        for pattern in format_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                tone_format["format"] = match.group(1).strip()
                break

        return tone_format

    def _generate_evaluation_component(self) -> dict[str, Any]:
        """Generate the Evaluation component of C.R.E.A.T.E. framework.

        Returns:
            Dictionary containing evaluation component.
        """
        return {
            "quality_checks": [
                "Accuracy verification",
                "Completeness check",
                "Clarity assessment",
                "Relevance validation",
            ],
            "success_criteria": [
                "Meets all requirements",
                "Appropriate tone and format",
                "Clear and actionable",
                "Factually accurate",
            ],
        }

    def apply_create_framework(self, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        """Apply the C.R.E.A.T.E. framework to enhance a prompt.

        Args:
            prompt: The input prompt to enhance.
            context: Additional context information.

        Returns:
            Dictionary containing the enhanced prompt components.
        """
        extracted_context = self._extract_context(prompt)
        domain = context.get("domain", "general")

        # Generate each component of the C.R.E.A.T.E. framework
        return {
            "context": extracted_context,
            "request": self._generate_request_component(prompt, extracted_context),
            "examples": self._generate_examples_component(prompt),
            "augmentations": self._generate_augmentations_component(domain),
            "tone_format": self._generate_tone_format_component(prompt),
            "evaluation": self._generate_evaluation_component(),
        }

    def _build_enhanced_prompt(self, components: dict[str, Any]) -> str:
        """Build the enhanced prompt from C.R.E.A.T.E. components.

        Args:
            components: Dictionary of C.R.E.A.T.E. components.

        Returns:
            Enhanced prompt string.
        """
        sections = []

        # Build each section using helper methods
        sections.extend(self._build_context_section(components["context"]))
        sections.extend(self._build_request_section(components["request"]))
        sections.extend(self._build_examples_section(components["examples"]))
        sections.extend(self._build_augmentations_section(components["augmentations"]))
        sections.extend(self._build_tone_format_section(components["tone_format"]))
        sections.extend(self._build_evaluation_section(components["evaluation"]))

        return "\n\n".join(sections)

    def _build_context_section(self, context: dict[str, Any]) -> list[str]:
        """Build the context section of the prompt."""
        sections = []
        if context["role"]:
            sections.append(f"## Context\nRole: {context['role']}")
            if context["background"]:
                sections.append(f"Background: {context['background']}")
            if context["goal"]:
                sections.append(f"Goal: {context['goal']}")
        return sections

    def _build_request_section(self, request: dict[str, Any]) -> list[str]:
        """Build the request section of the prompt."""
        sections = []
        if request["task"]:
            sections.append(f"## Request\nTask: {request['task']}")
            if request["deliverable"]:
                sections.append(f"Deliverable: {request['deliverable']}")
        return sections

    def _build_examples_section(self, examples: dict[str, Any]) -> list[str]:
        """Build the examples section of the prompt."""
        sections = []
        if examples["patterns"]:
            sections.append("## Examples")
            for pattern in examples["patterns"]:
                sections.append(f"- {pattern}")
        return sections

    def _build_augmentations_section(self, augmentations: dict[str, Any]) -> list[str]:
        """Build the augmentations section of the prompt."""
        sections = []
        if augmentations["frameworks"]:
            sections.append("## Augmentations")
            sections.append(f"Domain: {augmentations['domain']}")
            if augmentations["frameworks"]:
                sections.append(f"Frameworks: {', '.join(augmentations['frameworks'])}")
        return sections

    def _build_tone_format_section(self, tone_format: dict[str, Any]) -> list[str]:
        """Build the tone & format section of the prompt."""
        return [
            "## Tone & Format",
            f"Tone: {tone_format['tone']}",
            f"Format: {tone_format['format']}",
        ]

    def _build_evaluation_section(self, evaluation: dict[str, Any]) -> list[str]:
        """Build the evaluation section of the prompt."""
        sections = ["## Evaluation", "Quality checks:"]
        for check in evaluation["quality_checks"]:
            sections.append(f"- {check}")
        return sections

    async def process_prompt(self, input_prompt: str, domain: str | None = None) -> CreateResponse:
        """Process a prompt using the C.R.E.A.T.E. framework.

        Args:
            input_prompt: The input prompt to process.
            domain: Optional domain for specialized processing.

        Returns:
            CreateResponse containing the enhanced prompt and metadata.
        """
        start_time = time.time()
        errors: list[str] = []

        try:
            # Validate input
            self.validate_input(input_prompt)

            # Prepare context
            context = {"domain": domain or "general"}

            # Apply C.R.E.A.T.E. framework
            components = self.apply_create_framework(input_prompt, context)

            # Build enhanced prompt
            enhanced_prompt = self._build_enhanced_prompt(components)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create response
            response = CreateResponse(
                enhanced_prompt=enhanced_prompt,
                framework_components=components,
                metadata={
                    "original_prompt_length": len(input_prompt),
                    "enhanced_prompt_length": len(enhanced_prompt),
                    "domain": domain or "general",
                    "timestamp": time.time(),
                },
                processing_time=processing_time,
                success=True,
                errors=errors,
            )

            self.logger.info(
                "Successfully processed prompt",
                original_length=len(input_prompt),
                enhanced_length=len(enhanced_prompt),
                processing_time=processing_time,
                domain=domain,
            )

            return response

        except ValidationError as e:
            errors.append(str(e))
            self.logger.error("Validation error", error=str(e))
        except Exception as e:
            errors.append(str(e))
            self.logger.exception("Unexpected error during processing", error=str(e))

        # Return error response
        return CreateResponse(
            enhanced_prompt="",
            framework_components={},
            metadata={"error": True},
            processing_time=time.time() - start_time,
            success=False,
            errors=errors,
        )
