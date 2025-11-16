"""Unit tests for core template system implementation.

This module contains comprehensive unit tests for the template system
in Phase 1 Issue 4, focusing on template management and processing.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from src.core.template_system_core import (
    TemplateManager,
    TemplateProcessor,
    TemplateSchema,
    TemplateType,
)


class TestTemplateManager:
    """Test cases for TemplateManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.templates_dir = Path(self.temp_dir) / "templates"
        self.templates_dir.mkdir()
        self.manager = TemplateManager(str(self.templates_dir))

    def test_init_with_valid_directory(self):
        """Test TemplateManager initialization with valid directory."""
        manager = TemplateManager(str(self.templates_dir))
        assert manager.templates_dir == self.templates_dir
        assert isinstance(manager.templates, dict)

    def test_init_with_nonexistent_directory(self):
        """Test TemplateManager initialization with nonexistent directory."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        manager = TemplateManager(str(nonexistent_dir))
        assert manager.templates == {}

    def test_load_valid_template(self):
        """Test loading a valid template file."""
        template_data = {
            "metadata": {"name": "test", "type": "business"},
            "variables": {"title": {"type": "string", "required": True}},
            "structure": {"sections": []},
            "examples": [],
        }

        template_file = self.templates_dir / "test.yaml"
        with template_file.open("w") as f:
            yaml.dump(template_data, f)

        manager = TemplateManager(str(self.templates_dir))
        assert "test" in manager.templates
        assert isinstance(manager.templates["test"], TemplateSchema)

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        template_file = self.templates_dir / "invalid.yaml"
        with template_file.open("w") as f:
            f.write("invalid: yaml: content: [")

        with patch.object(TemplateManager, "_load_template") as mock_load:
            mock_load.side_effect = yaml.YAMLError("Invalid YAML")
            manager = TemplateManager(str(self.templates_dir))
            assert manager.templates == {}

    def test_get_template_existing(self):
        """Test getting an existing template."""
        template_data = {
            "metadata": {"name": "test", "type": "business"},
            "variables": {"title": {"type": "string", "required": True}},
            "structure": {"sections": []},
            "examples": [],
        }

        template_file = self.templates_dir / "test.yaml"
        with template_file.open("w") as f:
            yaml.dump(template_data, f)

        manager = TemplateManager(str(self.templates_dir))
        template = manager.get_template("test")
        assert template is not None
        assert template.metadata["name"] == "test"

    def test_get_template_nonexistent(self):
        """Test getting a nonexistent template."""
        template = self.manager.get_template("nonexistent")
        assert template is None

    def test_list_templates(self):
        """Test listing all templates."""
        # Create multiple templates
        for i in range(3):
            template_data = {
                "metadata": {"name": f"test{i}", "type": "business"},
                "variables": {"title": {"type": "string", "required": True}},
                "structure": {"sections": []},
                "examples": [],
            }

            template_file = self.templates_dir / f"test{i}.yaml"
            with template_file.open("w") as f:
                yaml.dump(template_data, f)

        manager = TemplateManager(str(self.templates_dir))
        templates = manager.list_templates()
        assert len(templates) == 3
        assert "test0" in templates
        assert "test1" in templates
        assert "test2" in templates

    def test_get_templates_by_type(self):
        """Test getting templates by type."""
        # Create templates of different types
        types = ["business", "technical", "business"]
        for i, template_type in enumerate(types):
            template_data = {
                "metadata": {"name": f"test{i}", "type": template_type},
                "variables": {"title": {"type": "string", "required": True}},
                "structure": {"sections": []},
                "examples": [],
            }

            template_file = self.templates_dir / f"test{i}.yaml"
            with template_file.open("w") as f:
                yaml.dump(template_data, f)

        manager = TemplateManager(str(self.templates_dir))
        business_templates = manager.get_templates_by_type(TemplateType.BUSINESS)
        assert len(business_templates) == 2
        assert "test0" in business_templates
        assert "test2" in business_templates

    def test_validate_template_valid(self):
        """Test validating a valid template."""
        template_data = {
            "metadata": {"name": "test", "type": "business"},
            "variables": {"title": {"type": "string", "required": True}},
            "structure": {"sections": []},
            "examples": [],
        }

        result = self.manager.validate_template(template_data)
        assert result is True

    def test_validate_template_invalid(self):
        """Test validating an invalid template."""
        template_data = {
            "metadata": {"name": "test"},
            # Missing required fields
        }

        result = self.manager.validate_template(template_data)
        assert result is False

    def test_create_template_structure(self):
        """Test creating basic template structure."""
        structure = self.manager.create_template_structure("test_template", TemplateType.BUSINESS)

        assert structure["metadata"]["name"] == "test_template"
        assert structure["metadata"]["type"] == "business"
        assert "variables" in structure
        assert "structure" in structure
        assert "examples" in structure


class TestTemplateProcessor:
    """Test cases for TemplateProcessor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.templates_dir = Path(self.temp_dir) / "templates"
        self.templates_dir.mkdir()

        # Create a test template
        template_data = {
            "metadata": {"name": "test", "type": "business"},
            "variables": {
                "title": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
            },
            "structure": {
                "sections": [
                    {
                        "name": "header",
                        "template": "# {title}",
                    },
                    {
                        "name": "body",
                        "template": "{content}",
                    },
                ],
            },
            "examples": [],
        }

        template_file = self.templates_dir / "test.yaml"
        with template_file.open("w") as f:
            yaml.dump(template_data, f)

        self.manager = TemplateManager(str(self.templates_dir))
        self.processor = TemplateProcessor(self.manager)

    def test_process_template_success(self):
        """Test successful template processing."""
        variables = {"title": "Test Title", "content": "Test content"}
        result = self.processor.process_template("test", variables)

        assert "# Test Title" in result
        assert "Test content" in result

    def test_process_template_missing_template(self):
        """Test processing nonexistent template."""
        variables = {"title": "Test Title", "content": "Test content"}

        with pytest.raises(ValueError, match="Template not found"):
            self.processor.process_template("nonexistent", variables)

    def test_process_template_missing_variables(self):
        """Test processing template with missing variables."""
        variables = {"title": "Test Title"}  # Missing 'content'

        # Should still process but with placeholder
        result = self.processor.process_template("test", variables)
        assert "# Test Title" in result

    def test_validate_variables_success(self):
        """Test successful variable validation."""
        variables = {"title": "Test Title", "content": "Test content"}
        result = self.processor.validate_variables("test", variables)
        assert result is True

    def test_validate_variables_missing_required(self):
        """Test variable validation with missing required variable."""
        variables = {"title": "Test Title"}  # Missing required 'content'
        result = self.processor.validate_variables("test", variables)
        assert result is False

    def test_validate_variables_nonexistent_template(self):
        """Test variable validation for nonexistent template."""
        variables = {"title": "Test Title", "content": "Test content"}
        result = self.processor.validate_variables("nonexistent", variables)
        assert result is False

    def test_get_template_info(self):
        """Test getting template information."""
        info = self.processor.get_template_info("test")

        assert info["name"] == "test"
        assert "metadata" in info
        assert "variables" in info
        assert "examples" in info

    def test_get_template_info_nonexistent(self):
        """Test getting information for nonexistent template."""
        info = self.processor.get_template_info("nonexistent")
        assert info == {}


class TestTemplateSchema:
    """Test cases for TemplateSchema validation."""

    def test_valid_schema(self):
        """Test valid template schema."""
        data = {
            "metadata": {"name": "test", "type": "business"},
            "variables": {"title": {"type": "string", "required": True}},
            "structure": {"sections": []},
            "examples": [],
        }

        schema = TemplateSchema(**data)
        assert schema.metadata["name"] == "test"
        assert schema.variables["title"]["type"] == "string"

    def test_invalid_schema_missing_required(self):
        """Test invalid schema with missing required fields."""
        data = {
            "metadata": {"name": "test"},
            # Missing required fields
        }

        with pytest.raises(ValidationError):
            TemplateSchema(**data)

    def test_schema_with_defaults(self):
        """Test schema with default values."""
        data = {
            "metadata": {"name": "test", "type": "business"},
            "variables": {"title": {"type": "string", "required": True}},
            "structure": {"sections": []},
            # examples will use default empty list
        }

        schema = TemplateSchema(**data)
        assert schema.examples == []


class TestTemplateType:
    """Test cases for TemplateType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert TemplateType.BUSINESS.value == "business"
        assert TemplateType.TECHNICAL.value == "technical"
        assert TemplateType.LEGAL.value == "legal"
        assert TemplateType.ACADEMIC.value == "academic"
        assert TemplateType.CREATIVE.value == "creative"

    def test_enum_membership(self):
        """Test enum membership."""
        assert TemplateType.BUSINESS in TemplateType
        assert "invalid" not in [t.value for t in TemplateType]
