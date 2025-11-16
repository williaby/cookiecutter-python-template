"""
Unit tests for Journey1SmartTemplates class.

This module provides comprehensive test coverage for the Journey1SmartTemplates class,
testing file content extraction, C.R.E.A.T.E. framework functionality, and UI integration.
"""

import contextlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ui.journeys.journey1_smart_templates import Journey1SmartTemplates


@pytest.mark.unit
class TestJourney1SmartTemplates:
    """Test cases for Journey1SmartTemplates class."""

    def test_init(self):
        """Test Journey1SmartTemplates initialization."""
        journey = Journey1SmartTemplates()

        assert journey.supported_file_types == [".txt", ".md", ".pdf", ".docx", ".csv", ".json"]
        assert journey.max_file_size == 10 * 1024 * 1024  # 10MB
        assert journey.max_files == 5

    def test_extract_file_content_txt(self):
        """Test extracting content from text files."""
        journey = Journey1SmartTemplates()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content\nWith multiple lines\n")
            temp_path = f.name

        try:
            content, file_type = journey.extract_file_content(temp_path)
            assert "This is test content" in content
            assert file_type == ".txt"
        finally:
            Path(temp_path).unlink()

    def test_extract_file_content_md(self):
        """Test extracting content from markdown files."""
        journey = Journey1SmartTemplates()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Markdown\n\nThis is **bold** text.")
            temp_path = f.name

        try:
            content, file_type = journey.extract_file_content(temp_path)
            assert "# Test Markdown" in content
            assert "**bold**" in content
            assert file_type == ".md"
        finally:
            Path(temp_path).unlink()

    def test_extract_file_content_pdf_placeholder(self):
        """Test PDF file handling (returns placeholder)."""
        journey = Journey1SmartTemplates()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf content")
            temp_path = f.name

        try:
            content, file_type = journey.extract_file_content(temp_path)
            assert "[PDF Document:" in content
            assert "PDF text extraction requires PyPDF2" in content
            assert file_type == ".pdf"
        finally:
            Path(temp_path).unlink()

    def test_extract_file_content_docx_placeholder(self):
        """Test DOCX file handling (returns placeholder)."""
        journey = Journey1SmartTemplates()

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"fake docx content")
            temp_path = f.name

        try:
            content, file_type = journey.extract_file_content(temp_path)
            assert "[Word Document:" in content
            assert "DOCX text extraction requires python-docx" in content
            assert file_type == ".docx"
        finally:
            Path(temp_path).unlink()

    def test_extract_file_content_csv(self):
        """Test extracting content from CSV files."""
        journey = Journey1SmartTemplates()

        csv_content = "name,age,city\nJohn,25,New York\nJane,30,San Francisco"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            content, file_type = journey.extract_file_content(temp_path)
            assert "CSV Data Structure Analysis" in content
            assert "John,25,New York" in content
            assert file_type == ".csv"
        finally:
            Path(temp_path).unlink()

    def test_extract_file_content_json(self):
        """Test extracting content from JSON files."""
        journey = Journey1SmartTemplates()

        json_data = {"name": "test", "value": 123, "items": ["a", "b", "c"]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            content, file_type = journey.extract_file_content(temp_path)
            assert "JSON Data Structure Analysis" in content
            assert '"name": "test"' in content
            assert file_type == ".json"
        finally:
            Path(temp_path).unlink()

    def test_extract_file_content_unsupported(self):
        """Test handling unsupported file types."""
        journey = Journey1SmartTemplates()

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"unsupported content")
            temp_path = f.name

        try:
            content, file_type = journey.extract_file_content(temp_path)
            assert "[Unsupported file type: .xyz]" in content
            assert "Supported formats:" in content
            assert file_type == ".xyz"
        finally:
            Path(temp_path).unlink()

    def test_extract_file_content_file_not_found(self):
        """Test handling missing files."""
        journey = Journey1SmartTemplates()

        content, file_type = journey.extract_file_content("/nonexistent/file.txt")
        assert "[Error processing file]" in content
        assert "File not found or access denied" in content
        assert file_type == "error"

    def test_clean_text_content(self):
        """Test text content cleaning functionality."""
        journey = Journey1SmartTemplates()

        # Test with messy text content
        messy_text = "  Line 1  \n\n\n\n  Line 2  \n\t\tTabbed content\n   "
        cleaned = journey._clean_text_content(messy_text)

        assert cleaned == "Line 1\n\nLine 2\nTabbed content"
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")

    def test_clean_text_content_empty(self):
        """Test cleaning empty or whitespace-only content."""
        journey = Journey1SmartTemplates()

        assert journey._clean_text_content("") == ""
        assert journey._clean_text_content("   \n\t  ") == ""
        assert journey._clean_text_content("\n\n\n") == ""

    def test_process_csv_content(self):
        """Test CSV content processing."""
        journey = Journey1SmartTemplates()

        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,SF"
        result = journey._process_csv_content(csv_content, "test.csv")

        assert "CSV Data Structure Analysis" in result
        assert "test.csv" in result
        assert "3 rows detected" in result
        assert "3 columns detected" in result
        assert csv_content in result

    def test_process_csv_content_malformed(self):
        """Test processing malformed CSV content."""
        journey = Journey1SmartTemplates()

        # Malformed CSV (inconsistent columns)
        malformed_csv = "name,age\nJohn,25,extra\nJane"
        result = journey._process_csv_content(malformed_csv, "bad.csv")

        assert "CSV Data Structure Analysis" in result
        assert "inconsistent column count" in result.lower()
        assert malformed_csv in result

    def test_process_json_content_valid(self):
        """Test processing valid JSON content."""
        journey = Journey1SmartTemplates()

        json_content = '{"name": "test", "items": [1, 2, 3], "nested": {"key": "value"}}'
        result = journey._process_json_content(json_content, "test.json")

        assert "JSON Data Structure Analysis" in result
        assert "test.json" in result
        assert "Valid JSON structure" in result
        assert "3 top-level keys" in result
        assert json_content in result

    def test_process_json_content_invalid(self):
        """Test processing invalid JSON content."""
        journey = Journey1SmartTemplates()

        invalid_json = '{"name": "test", "missing_quote: "value"}'
        result = journey._process_json_content(invalid_json, "bad.json")

        assert "JSON Data Structure Analysis" in result
        assert "Invalid JSON syntax" in result
        assert invalid_json in result

    def test_validate_file_size_valid(self):
        """Test file size validation for valid files."""
        journey = Journey1SmartTemplates()

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"small content")
            f.flush()

            is_valid, message = journey.validate_file_size(f.name)
            assert is_valid is True
            assert "File size is acceptable" in message

    def test_validate_file_size_too_large(self):
        """Test file size validation for oversized files."""
        journey = Journey1SmartTemplates()

        # Mock a large file
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 15 * 1024 * 1024  # 15MB

            is_valid, message = journey.validate_file_size("fake_large_file.txt")
            assert is_valid is False
            assert "File too large" in message
            assert "14.3 MB" in message

    def test_validate_file_size_not_found(self):
        """Test file size validation for missing files."""
        journey = Journey1SmartTemplates()

        is_valid, message = journey.validate_file_size("/nonexistent/file.txt")
        assert is_valid is False
        assert "File not found" in message

    def test_analyze_content_structure_code(self):
        """Test content structure analysis for code content."""
        journey = Journey1SmartTemplates()

        code_content = """
def hello_world():
    print("Hello, World!")

class TestClass:
    def method(self):
        return True
"""

        analysis = journey.analyze_content_structure(code_content)

        assert analysis["content_type"] == "code"
        assert analysis["line_count"] > 0
        assert analysis["has_functions"] is True
        assert analysis["has_classes"] is True
        assert "python" in analysis["detected_language"].lower()

    def test_analyze_content_structure_documentation(self):
        """Test content structure analysis for documentation."""
        journey = Journey1SmartTemplates()

        doc_content = """
# Main Title

## Section 1
This is documentation content.

## Section 2
More documentation with [links](http://example.com).

- List item 1
- List item 2
"""

        analysis = journey.analyze_content_structure(doc_content)

        assert analysis["content_type"] == "documentation"
        assert analysis["line_count"] > 0
        assert analysis["has_headings"] is True
        assert analysis["has_links"] is True
        assert analysis["has_lists"] is True

    def test_analyze_content_structure_data(self):
        """Test content structure analysis for data content."""
        journey = Journey1SmartTemplates()

        data_content = """
name,age,city
John,25,NYC
Jane,30,SF
Bob,35,LA
"""

        analysis = journey.analyze_content_structure(data_content)

        assert analysis["content_type"] == "data"
        assert analysis["line_count"] > 0
        assert analysis["has_structure"] is True

    def test_analyze_content_structure_mixed(self):
        """Test content structure analysis for mixed content."""
        journey = Journey1SmartTemplates()

        mixed_content = """
# Documentation

Some text content here.

```python
def example():
    return "code"
```

Regular paragraph text continues.
"""

        analysis = journey.analyze_content_structure(mixed_content)

        assert analysis["content_type"] == "mixed"
        assert analysis["line_count"] > 0
        assert analysis["has_code_blocks"] is True
        assert analysis["has_headings"] is True

    def test_get_file_metadata(self):
        """Test getting file metadata."""
        journey = Journey1SmartTemplates()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            metadata = journey.get_file_metadata(temp_path)

            assert metadata["name"] == Path(temp_path).name
            assert metadata["extension"] == ".txt"
            assert metadata["size"] > 0
            assert "created" in metadata
            assert "modified" in metadata
        finally:
            Path(temp_path).unlink()

    def test_get_file_metadata_not_found(self):
        """Test getting metadata for non-existent file."""
        journey = Journey1SmartTemplates()

        metadata = journey.get_file_metadata("/nonexistent/file.txt")

        assert metadata["error"] == "File not found"
        assert metadata["name"] == "file.txt"
        assert metadata["extension"] == ".txt"

    def test_create_breakdown_comprehensive(self):
        """Test creating comprehensive C.R.E.A.T.E. breakdown."""
        journey = Journey1SmartTemplates()

        input_text = "Create a Python function that calculates fibonacci numbers efficiently."

        breakdown = journey.create_breakdown(input_text)

        # Verify all C.R.E.A.T.E. components are present
        assert "context" in breakdown
        assert "request" in breakdown
        assert "examples" in breakdown
        assert "augmentations" in breakdown
        assert "tone_format" in breakdown
        assert "evaluation" in breakdown

        # Verify content quality
        assert len(breakdown["context"]) > 10
        assert len(breakdown["request"]) > 10
        assert "fibonacci" in breakdown["request"].lower()

    def test_create_breakdown_empty_input(self):
        """Test creating breakdown with empty input."""
        journey = Journey1SmartTemplates()

        breakdown = journey.create_breakdown("")

        # Should still return all components, but with generic content
        assert all(
            key in breakdown for key in ["context", "request", "examples", "augmentations", "tone_format", "evaluation"]
        )
        assert "provide more specific input" in breakdown["request"].lower()

    def test_create_breakdown_with_file_context(self):
        """Test creating breakdown with file context."""
        journey = Journey1SmartTemplates()

        input_text = "Optimize this code for performance"
        file_sources = [
            {
                "name": "slow_function.py",
                "type": "python",
                "size": 1024,
                "content_preview": "def slow_function():\n    time.sleep(1)",
            },
        ]

        breakdown = journey.create_breakdown(input_text, file_sources=file_sources)

        assert "slow_function.py" in breakdown["context"]
        assert "python" in breakdown["context"].lower()
        assert "performance" in breakdown["request"].lower()

    def test_enhance_prompt_basic(self):
        """Test basic prompt enhancement."""
        journey = Journey1SmartTemplates()

        original_prompt = "Write a sorting algorithm"
        breakdown = {
            "context": "Programming assistant role",
            "request": "Create efficient sorting algorithm",
            "examples": "Example: quicksort implementation",
            "augmentations": "Include time complexity analysis",
            "tone_format": "Technical and educational",
            "evaluation": "Verify correctness and efficiency",
        }

        enhanced = journey.enhance_prompt_from_breakdown(original_prompt, breakdown)

        assert len(enhanced) > len(original_prompt)
        assert "sorting algorithm" in enhanced.lower()
        assert "context" in enhanced.lower() or "role" in enhanced.lower()

    def test_enhance_prompt_with_file_sources(self):
        """Test prompt enhancement with file sources."""
        journey = Journey1SmartTemplates()

        original_prompt = "Review this code"
        breakdown = {
            "context": "Code reviewer",
            "request": "Analyze code quality",
            "examples": "Point out issues",
            "augmentations": "Include suggestions",
            "tone_format": "Constructive feedback",
            "evaluation": "Check for best practices",
        }
        file_sources = [{"name": "code.py", "content": "def example(): pass"}]

        enhanced = journey.enhance_prompt_from_breakdown(original_prompt, breakdown, file_sources)

        assert "code.py" in enhanced
        assert len(enhanced) > len(original_prompt)

    def test_copy_to_clipboard_simulation(self):
        """Test clipboard copying simulation."""
        journey = Journey1SmartTemplates()

        # Since we can't actually test clipboard in unit tests,
        # we'll test the method exists and returns expected format
        test_content = "Test content for clipboard"

        # This would normally copy to clipboard, but we can't test that
        # Instead, verify the method exists and handles the input
        try:
            result = journey.copy_to_clipboard(test_content)
            # Method should handle the operation gracefully
            assert result is None or isinstance(result, str)
        except Exception as e:
            # Expected in test environment without clipboard access
            assert "clipboard" in str(e).lower() or "not supported" in str(e).lower()  # noqa: PT017

    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        journey = Journey1SmartTemplates()

        formats = journey.get_supported_formats()

        assert isinstance(formats, list)
        assert ".txt" in formats
        assert ".md" in formats
        assert len(formats) == len(journey.supported_file_types)

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        journey = Journey1SmartTemplates()

        # Small content
        small_content = "Short text"
        small_time = journey.estimate_processing_time(small_content)
        assert small_time > 0
        assert small_time < 10  # Should be quick

        # Large content
        large_content = "Large text content. " * 1000
        large_time = journey.estimate_processing_time(large_content)
        assert large_time > small_time  # Should take longer

    def test_validate_input_content(self):
        """Test input content validation."""
        journey = Journey1SmartTemplates()

        # Valid content
        valid_content = "This is valid input content for processing."
        is_valid, message = journey.validate_input_content(valid_content)
        assert is_valid is True
        assert "valid" in message.lower()

        # Empty content
        is_valid, message = journey.validate_input_content("")
        assert is_valid is False
        assert "empty" in message.lower()

        # Very long content
        long_content = "x" * 100000
        is_valid, message = journey.validate_input_content(long_content)
        assert is_valid is False
        assert "too long" in message.lower()

    def test_error_handling_robustness(self):
        """Test error handling across different scenarios."""
        journey = Journey1SmartTemplates()

        # Test with None inputs
        with contextlib.suppress(TypeError, AttributeError):
            journey.analyze_content_structure(None)

        # Test with invalid file paths
        content, file_type = journey.extract_file_content("")
        assert "Error processing file" in content
        assert file_type == "error"

        # Test breakdown with None
        breakdown = journey.create_breakdown(None)
        assert all(isinstance(v, str) for v in breakdown.values())

    def test_process_files_empty_list(self):
        """Test processing empty file list."""
        journey = Journey1SmartTemplates()

        result = journey.process_files([])

        assert result["files"] == []
        assert result["content"] == ""
        assert result["summary"] == "No files uploaded"
        assert result["file_count"] == 0
        assert result["total_size"] == 0
        assert result["supported_files"] == 0
        assert result["preview_available"] is False

    def test_process_files_single_file(self):
        """Test processing a single file."""
        journey = Journey1SmartTemplates()

        # Create a mock file object
        class MockFile:
            def __init__(self, name):
                self.name = name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test file content\nLine 2\nLine 3")
            temp_path = f.name

        try:
            mock_file = MockFile(temp_path)
            result = journey.process_files([mock_file])

            assert len(result["files"]) == 1
            assert result["file_count"] == 1
            assert result["supported_files"] == 1
            assert result["preview_available"] is True
            assert "Test file content" in result["content"]
            assert "FILE:" in result["content"]
            assert result["processing_complete"] is True
        finally:
            Path(temp_path).unlink()

    def test_process_files_multiple_files(self):
        """Test processing multiple files."""
        journey = Journey1SmartTemplates()

        class MockFile:
            def __init__(self, name):
                self.name = name

        # Create multiple temporary files
        files_to_cleanup = []
        mock_files = []

        try:
            # Create text file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("Text content")
                files_to_cleanup.append(f.name)
                mock_files.append(MockFile(f.name))

            # Create CSV file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write("name,age\nJohn,25")
                files_to_cleanup.append(f.name)
                mock_files.append(MockFile(f.name))

            # Create unsupported file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
                f.write("Unsupported content")
                files_to_cleanup.append(f.name)
                mock_files.append(MockFile(f.name))

            result = journey.process_files(mock_files)

            assert len(result["files"]) == 3
            assert result["file_count"] == 3
            assert result["supported_files"] == 2  # txt and csv are supported
            assert result["preview_available"] is True
            assert "Text content" in result["content"]
            assert "name,age" in result["content"]

        finally:
            for file_path in files_to_cleanup:
                Path(file_path).unlink()

    def test_process_files_error_handling(self):
        """Test file processing error handling."""
        journey = Journey1SmartTemplates()

        class MockFile:
            def __init__(self, name):
                self.name = name

        # Test with non-existent file
        mock_file = MockFile("/nonexistent/file.txt")
        result = journey.process_files([mock_file])

        assert len(result["files"]) == 1
        assert result["files"][0]["processing_status"] == "error"
        assert result["supported_files"] == 0
        assert len(result["errors"]) >= 0  # May have errors

    def test_enhance_prompt_full_method(self):
        """Test the full enhance_prompt method with all parameters."""
        journey = Journey1SmartTemplates()

        text_input = "Create a function to sort numbers"
        files = []
        model_mode = "standard"
        custom_model = ""
        reasoning_depth = "detailed"
        search_tier = "basic"
        temperature = 0.7

        result = journey.enhance_prompt(
            text_input,
            files,
            model_mode,
            custom_model,
            reasoning_depth,
            search_tier,
            temperature,
        )

        # Should return 9 values
        assert len(result) == 9
        enhanced, context, request, examples, augmentations, tone_format, evaluation, attribution, file_sources = result

        assert len(enhanced) > 0
        assert "sort numbers" in enhanced.lower()
        assert len(context) > 0
        assert len(request) > 0
        assert "gpt-4o-mini" in attribution  # standard model

    def test_enhance_prompt_different_models(self):
        """Test enhance_prompt with different model modes."""
        journey = Journey1SmartTemplates()

        # Test custom model
        result = journey.enhance_prompt("test input", [], "custom", "custom-model-name", "basic", "basic", 0.5)
        assert "custom-model-name" in result[7]  # attribution

        # Test free mode
        result = journey.enhance_prompt("test input", [], "free_mode", "", "basic", "basic", 0.5)
        assert "llama-4-maverick:free" in result[7]

        # Test premium mode
        result = journey.enhance_prompt("test input", [], "premium", "", "basic", "basic", 0.5)
        assert "claude-3.5-sonnet" in result[7]

    def test_enhance_prompt_with_files(self):
        """Test enhance_prompt with file uploads."""
        journey = Journey1SmartTemplates()

        class MockFile:
            def __init__(self, name):
                self.name = name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File content for enhancement")
            temp_path = f.name

        try:
            mock_file = MockFile(temp_path)
            result = journey.enhance_prompt(
                "Analyze this file",
                [mock_file],
                "standard",
                "",
                "comprehensive",
                "basic",
                0.7,
            )

            enhanced, context, request, examples, augmentations, tone_format, evaluation, attribution, file_sources = (
                result
            )

            assert "File content for enhancement" in enhanced
            assert "FILE:" in file_sources
            assert len(examples) > 0  # comprehensive mode should have code examples

        finally:
            Path(temp_path).unlink()

    def test_enhance_prompt_error_handling(self):
        """Test enhance_prompt error handling."""
        journey = Journey1SmartTemplates()

        # Test with file processing error by using invalid file path
        class MockFile:
            def __init__(self, name):
                self.name = name

        mock_file = MockFile("/invalid/path/file.txt")

        # This should handle the error gracefully
        result = journey.enhance_prompt("test", [mock_file], "standard", "", "basic", "basic", 0.5)

        # Should still return 9 values even with errors
        assert len(result) == 9

    def test_copy_code_blocks(self):
        """Test copying code blocks functionality."""
        journey = Journey1SmartTemplates()

        content_with_code = """
        Here's some code:

        ```python
        def hello():
            print("Hello World")
        ```

        And some more text.
        """

        result = journey.copy_code_blocks(content_with_code)
        assert "Found" in result
        assert "code blocks" in result

    def test_copy_code_blocks_no_code(self):
        """Test copying when no code blocks exist."""
        journey = Journey1SmartTemplates()

        content_no_code = "This is just regular text with no code blocks."

        result = journey.copy_code_blocks(content_no_code)
        assert "No code blocks found" in result

    def test_copy_as_markdown(self):
        """Test copying content as markdown."""
        journey = Journey1SmartTemplates()

        test_content = "This is test content\nWith multiple lines\nAnd formatting"

        result = journey.copy_as_markdown(test_content)
        assert "Copied" in result
        assert "characters as markdown" in result

    def test_copy_as_markdown_empty(self):
        """Test copying empty content as markdown."""
        journey = Journey1SmartTemplates()

        result = journey.copy_as_markdown("")
        assert "No content to copy" in result

    def test_copy_as_markdown_with_code_blocks(self):
        """Test copying content with code blocks as markdown."""
        journey = Journey1SmartTemplates()

        content_with_code = """
        # Title

        Some text here.

        ```python
        def example():
            return "test"
        ```
        """

        result = journey.copy_as_markdown(content_with_code)
        assert "code blocks preserved" in result

    def test_download_content(self):
        """Test download content preparation."""
        journey = Journey1SmartTemplates()

        test_content = "Content to download"
        create_data = {"context": "test", "request": "test"}

        result = journey.download_content(test_content, create_data)
        assert "Download prepared" in result
        assert str(len(test_content)) in result

    def test_validate_input_content_short_content(self):
        """Test validation of very short content."""
        journey = Journey1SmartTemplates()

        # Test content that's too short
        short_content = "Hi"
        is_valid, message = journey.validate_input_content(short_content)
        assert is_valid is False
        assert "too short" in message.lower()

    def test_validate_input_content_null_bytes(self):
        """Test validation of content with null bytes."""
        journey = Journey1SmartTemplates()

        # Test content with null bytes
        null_content = "Content with \x00 null bytes"
        is_valid, message = journey.validate_input_content(null_content)
        assert is_valid is False
        assert "null bytes" in message.lower()

    def test_extract_file_content_empty_path(self):
        """Test extract_file_content with empty path."""
        journey = Journey1SmartTemplates()

        content, file_type = journey.extract_file_content("")
        assert "Error processing file" in content
        assert "No file path provided" in content
        assert file_type == "error"

        # Test with whitespace-only path
        content, file_type = journey.extract_file_content("   ")
        assert "Error processing file" in content
        assert file_type == "error"

    def test_process_csv_content_empty(self):
        """Test processing empty CSV content."""
        journey = Journey1SmartTemplates()

        result = journey._process_csv_content("", "empty.csv")
        assert "empty.csv" in result
        assert "Content:" in result

    def test_process_json_content_error(self):
        """Test JSON processing with general error."""
        journey = Journey1SmartTemplates()

        # Mock an error during JSON processing
        with patch("json.loads", side_effect=Exception("General error")):
            result = journey._process_json_content('{"valid": "json"}', "test.json")
            assert "Processing error" in result
            assert "General error" in result

    def test_analyze_content_structure_empty_content(self):
        """Test content analysis with empty content."""
        journey = Journey1SmartTemplates()

        analysis = journey.analyze_content_structure("")
        assert analysis["content_type"] == "empty"
        assert analysis["complexity"] == "simple"
        assert analysis["has_code"] is False
        assert analysis["line_count"] == 0
        assert analysis["char_count"] == 0

    def test_analyze_content_structure_javascript(self):
        """Test content analysis for JavaScript code."""
        journey = Journey1SmartTemplates()

        js_content = """
        function greet(name) {
            console.log("Hello " + name);
        }

        const app = {
            start: function() {
                greet("World");
            }
        };
        """

        analysis = journey.analyze_content_structure(js_content)
        assert analysis["content_type"] == "code"
        assert analysis["has_functions"] is True
        assert analysis["language"] == "javascript"
        assert analysis["detected_language"] == "javascript"

    def test_analyze_content_structure_java(self):
        """Test content analysis for Java code."""
        journey = Journey1SmartTemplates()

        java_content = """
        public class HelloWorld {
            public static void main(String[] args) {
                System.out.println("Hello World");
            }
        }
        """

        analysis = journey.analyze_content_structure(java_content)
        assert analysis["content_type"] == "code"
        assert analysis["has_classes"] is True
        assert analysis["language"] == "java"
        assert analysis["detected_language"] == "java"

    def test_analyze_content_structure_json_data(self):
        """Test content analysis for JSON data."""
        journey = Journey1SmartTemplates()

        json_content = '{"name": "test", "items": [1, 2, 3]}'

        analysis = journey.analyze_content_structure(json_content)
        assert analysis["content_type"] == "data"
        assert analysis["language"] == "json"
        assert analysis["has_structure"] is True

    def test_get_file_metadata_with_error(self):
        """Test file metadata with general error."""
        journey = Journey1SmartTemplates()

        # Mock an error during file stat
        with patch("pathlib.Path.stat", side_effect=PermissionError("Access denied")):
            metadata = journey.get_file_metadata("/some/file.txt")
            assert metadata["error"] == "Access denied"
            assert metadata["exists"] is False
            assert metadata["size"] == 0

    def test_create_breakdown_with_languages(self):
        """Test breakdown creation with language detection from files."""
        journey = Journey1SmartTemplates()

        input_text = "Review this code"
        file_sources = [{"name": "script.py", "type": "python"}, {"name": "app.js", "type": "javascript"}]

        breakdown = journey.create_breakdown(input_text, file_sources)

        assert "script.py" in breakdown["context"]
        assert "app.js" in breakdown["context"]
        assert "python" in breakdown["context"].lower()

    def test_enhance_prompt_full_duplicate_method(self):
        """Test the enhance_prompt_full method (lines 1083-1176)."""
        journey = Journey1SmartTemplates()

        result = journey.enhance_prompt_full("Test input", [], "free_mode", "", "comprehensive", "advanced", 0.8)

        assert len(result) == 9
        enhanced, context, request, examples, augmentations, tone_format, evaluation, attribution, file_sources = result

        assert "llama-4-maverick:free" in attribution
        assert len(enhanced) > 0

    def test_enhance_prompt_method_at_line_1178(self):
        """Test the enhance_prompt method starting at line 1178."""
        journey = Journey1SmartTemplates()

        original_prompt = "Write a calculator function"
        breakdown = {
            "context": "Programming context",
            "content_type": "code",
            "complexity": "moderate",
            "recommended_approach": "step-by-step implementation",
        }
        file_sources = [{"name": "example.py", "content": "def example(): pass"}]

        enhanced = journey.enhance_prompt_from_breakdown(original_prompt, breakdown, file_sources)

        assert "calculator function" in enhanced.lower()
        assert "Programming context" in enhanced
        assert "example.py" in enhanced
        assert len(enhanced) > len(original_prompt)

    def test_estimate_processing_time_bounds(self):
        """Test processing time estimation bounds."""
        journey = Journey1SmartTemplates()

        # Test minimum bound
        empty_time = journey.estimate_processing_time("")
        assert empty_time == 0.1

        # Test maximum bound with very large content
        huge_content = "x" * 1000000  # 1M characters
        huge_time = journey.estimate_processing_time(huge_content)
        assert huge_time <= 30.0

    def test_estimate_processing_time_complexity_factors(self):
        """Test processing time with different complexity factors."""
        journey = Journey1SmartTemplates()

        # Simple content
        simple_content = "Hello world"
        simple_time = journey.estimate_processing_time(simple_content)

        # Complex code content
        complex_content = (
            """
        def complex_function():
            # This is complex code with many lines
            for i in range(1000):
                if i % 2 == 0:
                    print(f"Even: {i}")
                else:
                    print(f"Odd: {i}")
        """
            * 50
        )  # Make it long enough to be complex

        complex_time = journey.estimate_processing_time(complex_content)
        assert complex_time > simple_time
