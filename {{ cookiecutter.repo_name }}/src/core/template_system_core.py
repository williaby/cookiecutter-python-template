"""Core template system implementation for C.R.E.A.T.E. framework.

This module provides the essential template management functionality for Phase 1 Issue 4,
implementing basic template loading and validation without advanced features.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Available template types."""

    BUSINESS = "business"
    TECHNICAL = "technical"
    LEGAL = "legal"
    ACADEMIC = "academic"
    CREATIVE = "creative"


@dataclass
class TemplateMetadata:
    """Template metadata information."""

    name: str
    description: str
    template_type: TemplateType
    version: str
    author: str
    created_date: str
    modified_date: str
    tags: list[str]


class TemplateSchema(BaseModel):
    """Pydantic schema for template validation."""

    metadata: dict[str, Any] = Field(..., description="Template metadata")
    variables: dict[str, Any] = Field(..., description="Template variables")
    structure: dict[str, Any] = Field(..., description="Template structure")
    examples: list[dict[str, Any]] = Field(default_factory=list, description="Usage examples")


class TemplateManager:
    """Core template management system.

    This class provides basic template loading, validation, and management
    functionality for the C.R.E.A.T.E. framework.
    """

    def __init__(self, templates_dir: str = "knowledge/create_agent/templates") -> None:
        """Initialize the template manager.

        Args:
            templates_dir: Directory containing template files.
        """
        self.templates_dir = Path(templates_dir)
        self.templates: dict[str, TemplateSchema] = {}
        self.logger = logger
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all templates from the templates directory."""
        if not self.templates_dir.exists():
            self.logger.warning("Templates directory does not exist: %s", self.templates_dir)
            return

        for template_file in self.templates_dir.glob("**/*.yaml"):
            try:
                self._load_template(template_file)
            except Exception as e:
                self.logger.error("Failed to load template %s: %s", template_file, e)

    def _load_template(self, template_file: Path) -> None:
        """Load a single template file.

        Args:
            template_file: Path to the template file.
        """
        try:
            with template_file.open(encoding="utf-8") as f:
                template_data = yaml.safe_load(f)

            # Validate template structure
            template = TemplateSchema(**template_data)
            template_name = template_file.stem
            self.templates[template_name] = template

            self.logger.info("Loaded template: %s", template_name)

        except yaml.YAMLError as e:
            self.logger.error("YAML error in template %s: %s", template_file, e)
            raise
        except ValidationError as e:
            self.logger.error("Validation error in template %s: %s", template_file, e)
            raise
        except Exception as e:
            self.logger.error("Unexpected error loading template %s: %s", template_file, e)
            raise

    def get_template(self, name: str) -> TemplateSchema | None:
        """Get a template by name.

        Args:
            name: Template name.

        Returns:
            Template schema or None if not found.
        """
        return self.templates.get(name)

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template names.
        """
        return list(self.templates.keys())

    def get_templates_by_type(self, template_type: TemplateType) -> list[str]:
        """Get templates by type.

        Args:
            template_type: Template type to filter by.

        Returns:
            List of template names matching the type.
        """
        matching_templates = []
        for name, template in self.templates.items():
            if template.metadata.get("type") == template_type.value:
                matching_templates.append(name)
        return matching_templates

    def validate_template(self, template_data: dict[str, Any]) -> bool:
        """Validate template data against schema.

        Args:
            template_data: Template data to validate.

        Returns:
            True if valid, False otherwise.
        """
        try:
            TemplateSchema(**template_data)
            return True
        except ValidationError as e:
            self.logger.error("Template validation failed: %s", e)
            return False

    def create_template_structure(self, name: str, template_type: TemplateType) -> dict[str, Any]:
        """Create basic template structure.

        Args:
            name: Template name.
            template_type: Template type.

        Returns:
            Basic template structure.
        """
        return {
            "metadata": {
                "name": name,
                "description": f"Template for {name}",
                "type": template_type.value,
                "version": "1.0",
                "author": "PromptCraft",
                "created_date": "2025-01-01",
                "modified_date": "2025-01-01",
                "tags": [template_type.value],
            },
            "variables": {
                "title": {
                    "type": "string",
                    "description": "Template title",
                    "required": True,
                },
                "content": {
                    "type": "string",
                    "description": "Main content",
                    "required": True,
                },
            },
            "structure": {
                "sections": [
                    {
                        "name": "introduction",
                        "description": "Introduction section",
                        "template": "## Introduction\n\n{content}",
                    },
                ],
            },
            "examples": [
                {
                    "name": f"Basic {name} example",
                    "variables": {
                        "title": f"Sample {name}",
                        "content": "Sample content for this template",
                    },
                },
            ],
        }


class TemplateProcessor:
    """Core template processing functionality.

    This class handles template rendering and variable substitution
    for the C.R.E.A.T.E. framework.
    """

    def __init__(self, template_manager: TemplateManager) -> None:
        """Initialize the template processor.

        Args:
            template_manager: Template manager instance.
        """
        self.template_manager = template_manager
        self.logger = logger

    def process_template(self, name: str, variables: dict[str, Any]) -> str:
        """Process a template with variables.

        Args:
            name: Template name.
            variables: Variables to substitute.

        Returns:
            Processed template content.

        Raises:
            ValueError: If template not found or processing fails.
        """
        template = self.template_manager.get_template(name)
        if not template:
            raise ValueError(f"Template not found: {name}")

        try:
            # Basic variable substitution
            return self._substitute_variables(template, variables)
        except Exception as e:
            self.logger.error("Template processing failed for %s: %s", name, e)
            raise

    def _substitute_variables(self, template: TemplateSchema, variables: dict[str, Any]) -> str:
        """Substitute variables in template.

        Args:
            template: Template schema.
            variables: Variables to substitute.

        Returns:
            Processed template content.
        """
        # Simple implementation - build content from sections
        content_parts = []

        for section in template.structure.get("sections", []):
            section_template = section.get("template", "")
            try:
                # Basic string formatting
                section_content = section_template.format(**variables)
                content_parts.append(section_content)
            except KeyError as e:
                self.logger.warning("Missing variable %s in template", e)
                content_parts.append(section_template)

        return "\n\n".join(content_parts)

    def validate_variables(self, template_name: str, variables: dict[str, Any]) -> bool:
        """Validate variables against template requirements.

        Args:
            template_name: Template name.
            variables: Variables to validate.

        Returns:
            True if valid, False otherwise.
        """
        template = self.template_manager.get_template(template_name)
        if not template:
            return False

        template_vars = template.variables
        for var_name, var_config in template_vars.items():
            if var_config.get("required", False) and var_name not in variables:
                self.logger.error("Required variable missing: %s", var_name)
                return False

        return True

    def get_template_info(self, name: str) -> dict[str, Any]:
        """Get template information.

        Args:
            name: Template name.

        Returns:
            Template information.
        """
        template = self.template_manager.get_template(name)
        if not template:
            return {}

        return {
            "name": name,
            "metadata": template.metadata,
            "variables": template.variables,
            "examples": template.examples,
        }
