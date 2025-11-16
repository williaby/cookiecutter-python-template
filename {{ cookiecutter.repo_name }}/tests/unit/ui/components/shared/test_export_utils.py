"""
Unit tests for ExportUtils class.

This module provides comprehensive test coverage for the ExportUtils class,
testing all export formats, code block extraction, and file preparation functionality.
"""

import json

import pytest

from src.ui.components.shared.export_utils import ExportUtils


@pytest.mark.unit
class TestExportUtils:
    """Test cases for ExportUtils class."""

    def test_init(self):
        """Test ExportUtils initialization."""
        export_utils = ExportUtils()
        assert export_utils.export_formats == ["text", "markdown", "json"]

    def test_export_journey1_content_markdown_format(self, sample_export_data, mock_datetime):
        """Test Journey 1 export in markdown format."""
        export_utils = ExportUtils()

        result = export_utils.export_journey1_content(
            enhanced_prompt=sample_export_data["enhanced_prompt"],
            create_breakdown=sample_export_data["create_breakdown"],
            model_info=sample_export_data["model_info"],
            file_sources=sample_export_data["file_sources"],
            session_data=sample_export_data["session_data"],
            format_type="markdown",
        )

        assert "# Enhanced Prompt Export" in result
        assert "Journey 1: Smart Templates" in result
        assert sample_export_data["enhanced_prompt"] in result
        assert "C.R.E.A.T.E. Framework Breakdown" in result
        assert "claude-3-5-sonnet-20241022" in result
        assert "$0.0123" in result
        assert "test_export.py" in result

    def test_export_journey1_content_json_format(self, sample_export_data, mock_datetime):
        """Test Journey 1 export in JSON format."""
        export_utils = ExportUtils()

        result = export_utils.export_journey1_content(
            enhanced_prompt=sample_export_data["enhanced_prompt"],
            create_breakdown=sample_export_data["create_breakdown"],
            model_info=sample_export_data["model_info"],
            file_sources=sample_export_data["file_sources"],
            session_data=sample_export_data["session_data"],
            format_type="json",
        )

        # Verify it's valid JSON
        data = json.loads(result)
        assert data["journey"] == "Journey 1: Smart Templates"
        assert data["enhanced_prompt"] == sample_export_data["enhanced_prompt"]
        assert data["model_info"]["model"] == "claude-3-5-sonnet-20241022"
        assert len(data["file_sources"]) == 2

    def test_export_journey1_content_text_format(self, sample_export_data, mock_datetime):
        """Test Journey 1 export in text format."""
        export_utils = ExportUtils()

        result = export_utils.export_journey1_content(
            enhanced_prompt=sample_export_data["enhanced_prompt"],
            create_breakdown=sample_export_data["create_breakdown"],
            model_info=sample_export_data["model_info"],
            file_sources=sample_export_data["file_sources"],
            session_data=sample_export_data["session_data"],
            format_type="text",
        )

        assert "ENHANCED PROMPT EXPORT" in result
        assert "Journey 1: Smart Templates" in result
        assert sample_export_data["enhanced_prompt"] in result
        assert "C.R.E.A.T.E. FRAMEWORK BREAKDOWN:" in result
        assert "SOURCE FILES USED:" in result

    def test_export_journey1_content_with_no_file_sources(self, sample_export_data, mock_datetime):
        """Test export when no file sources are provided."""
        export_utils = ExportUtils()

        result = export_utils.export_journey1_content(
            enhanced_prompt=sample_export_data["enhanced_prompt"],
            create_breakdown=sample_export_data["create_breakdown"],
            model_info=sample_export_data["model_info"],
            file_sources=[],
            session_data=sample_export_data["session_data"],
            format_type="markdown",
        )

        assert "📄 Source Files" not in result
        assert sample_export_data["enhanced_prompt"] in result

    def test_export_as_json_success(self, sample_export_data):
        """Test successful JSON export."""
        export_utils = ExportUtils()

        test_data = {"test": "data", "number": 123}
        result = export_utils._export_as_json(test_data)

        # Verify it's valid and formatted JSON
        parsed = json.loads(result)
        assert parsed == test_data
        assert "  " in result  # Check for indentation

    def test_export_as_json_with_error(self, caplog):
        """Test JSON export with serialization error."""
        export_utils = ExportUtils()

        # Create an object that can't be JSON serialized
        class NonSerializable:
            pass

        test_data = {"bad": NonSerializable()}
        result = export_utils._export_as_json(test_data)

        # Should return error JSON
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Export failed:" in parsed["error"]

    def test_extract_code_blocks_multiple_languages(self, complex_content_sample):
        """Test extracting code blocks from content with multiple languages."""
        export_utils = ExportUtils()

        code_blocks = export_utils.extract_code_blocks(complex_content_sample)

        assert len(code_blocks) == 4

        # Check Python block
        python_block = code_blocks[0]
        assert python_block["language"] == "python"
        assert "def calculate_score" in python_block["content"]
        assert python_block["has_functions"] is True
        assert len(python_block["comments"]) >= 2  # Docstring and inline comment

        # Check JavaScript block
        js_block = code_blocks[1]
        assert js_block["language"] == "javascript"
        assert "function processData" in js_block["content"]
        assert js_block["has_functions"] is True

        # Check SQL block
        sql_block = code_blocks[2]
        assert sql_block["language"] == "sql"
        assert "SELECT * FROM users" in sql_block["content"]

        # Check plain text block
        text_block = code_blocks[3]
        assert text_block["language"] == "text"
        assert "No special formatting" in text_block["content"]

    def test_extract_code_blocks_no_blocks(self):
        """Test extracting code blocks from content with no code blocks."""
        export_utils = ExportUtils()

        content = "This is just plain text with no code blocks."
        code_blocks = export_utils.extract_code_blocks(content)

        assert code_blocks == []

    def test_detect_language_python(self):
        """Test Python language detection."""
        export_utils = ExportUtils()

        python_code = "def hello():\n    print('Hello')"
        language = export_utils._detect_language(python_code)
        assert language == "python"

    def test_detect_language_javascript(self):
        """Test JavaScript language detection."""
        export_utils = ExportUtils()

        js_code = "function hello() {\n    console.log('Hello');\n}"
        language = export_utils._detect_language(js_code)
        assert language == "javascript"

    def test_detect_language_java(self):
        """Test Java language detection."""
        export_utils = ExportUtils()

        java_code = "class HelloWorld {\n    public static void main() {}\n}"
        language = export_utils._detect_language(java_code)
        assert language == "java"

    def test_detect_language_cpp(self):
        """Test C++ language detection."""
        export_utils = ExportUtils()

        cpp_code = "#include <iostream>\nint main() { return 0; }"
        language = export_utils._detect_language(cpp_code)
        assert language == "cpp"

    def test_detect_language_sql(self):
        """Test SQL language detection."""
        export_utils = ExportUtils()

        sql_code = "SELECT id FROM users WHERE active = true"
        language = export_utils._detect_language(sql_code)
        assert language == "sql"

    def test_detect_language_html(self):
        """Test HTML language detection."""
        export_utils = ExportUtils()

        html_code = "<html><div>Hello</div></html>"
        language = export_utils._detect_language(html_code)
        assert language == "html"

    def test_detect_language_json(self):
        """Test JSON language detection."""
        export_utils = ExportUtils()

        json_code = '{"name": "test", "value": 123}'
        language = export_utils._detect_language(json_code)
        assert language == "json"

    def test_detect_language_yaml(self):
        """Test YAML language detection."""
        export_utils = ExportUtils()

        yaml_code = "---\nname: test\nvalue: 123"
        language = export_utils._detect_language(yaml_code)
        assert language == "yaml"

    def test_detect_language_default(self):
        """Test default language detection for unknown code."""
        export_utils = ExportUtils()

        unknown_code = "Some random text that doesn't match any pattern"
        language = export_utils._detect_language(unknown_code)
        assert language == "text"

    def test_extract_comments_python(self):
        """Test extracting comments from Python code."""
        export_utils = ExportUtils()

        python_code = '''def test():
    """This is a docstring."""
    x = 1  # This is a comment
    # Another comment
    return x'''

        comments = export_utils._extract_comments(python_code, "python")

        assert "This is a docstring." in comments
        assert "This is a comment" in comments
        assert "Another comment" in comments

    def test_extract_comments_javascript(self):
        """Test extracting comments from JavaScript code."""
        export_utils = ExportUtils()

        js_code = """function test() {
    // This is a single line comment
    var x = 1; /* This is a block comment */
    return x;
}"""

        comments = export_utils._extract_comments(js_code, "javascript")

        assert "This is a single line comment" in comments
        assert "This is a block comment" in comments

    def test_extract_comments_html(self):
        """Test extracting comments from HTML code."""
        export_utils = ExportUtils()

        html_code = """<html>
    <!-- This is an HTML comment -->
    <div>Content</div>
    <!-- Another comment -->
</html>"""

        comments = export_utils._extract_comments(html_code, "html")

        assert "This is an HTML comment" in comments
        assert "Another comment" in comments

    def test_extract_comments_no_comments(self):
        """Test extracting comments from code with no comments."""
        export_utils = ExportUtils()

        code = "def test():\n    return 42"
        comments = export_utils._extract_comments(code, "python")

        assert comments == []

    def test_has_functions_python(self):
        """Test function detection in Python code."""
        export_utils = ExportUtils()

        assert export_utils._has_functions("def test():\n    pass", "python") is True
        assert export_utils._has_functions("x = 1\ny = 2", "python") is False

    def test_has_functions_javascript(self):
        """Test function detection in JavaScript code."""
        export_utils = ExportUtils()

        assert export_utils._has_functions("function test() { return 1; }", "javascript") is True
        assert export_utils._has_functions("var x = 1;", "javascript") is False

    def test_has_functions_java(self):
        """Test function detection in Java/C++ code."""
        export_utils = ExportUtils()

        assert export_utils._has_functions("public void test() { }", "java") is True
        assert export_utils._has_functions("int x = 1;", "java") is False

    def test_assess_complexity_simple(self):
        """Test complexity assessment for simple code."""
        export_utils = ExportUtils()

        simple_code = "def test():\n    return 1"
        complexity = export_utils._assess_complexity(simple_code)
        assert complexity == "simple"

    def test_assess_complexity_moderate(self):
        """Test complexity assessment for moderate code."""
        export_utils = ExportUtils()

        moderate_code = "\n".join([f"line {i}" for i in range(10)])
        complexity = export_utils._assess_complexity(moderate_code)
        assert complexity == "moderate"

    def test_assess_complexity_complex(self):
        """Test complexity assessment for complex code."""
        export_utils = ExportUtils()

        complex_code = "\n".join([f"line {i}" for i in range(25)])
        complexity = export_utils._assess_complexity(complex_code)
        assert complexity == "complex"

    def test_format_code_blocks_for_export(self, sample_code_blocks):
        """Test formatting code blocks for export."""
        export_utils = ExportUtils()

        result = export_utils.format_code_blocks_for_export(sample_code_blocks)

        assert "CODE BLOCKS EXPORT" in result
        assert "Total blocks: 2" in result
        assert "Languages: python, javascript" in result
        assert "Block 1: PYTHON" in result
        assert "Block 2: JAVASCRIPT" in result
        assert "def test_example():" in result
        assert "function getData()" in result

    def test_format_code_blocks_for_export_empty(self):
        """Test formatting empty code blocks list."""
        export_utils = ExportUtils()

        result = export_utils.format_code_blocks_for_export([])
        assert result == "No code blocks found."

    def test_copy_code_as_markdown(self, sample_code_blocks):
        """Test copying code blocks as markdown."""
        export_utils = ExportUtils()

        result = export_utils.copy_code_as_markdown(sample_code_blocks)

        assert "# Code Blocks Export" in result
        assert "Total blocks: 2" in result
        assert "## Block 1: Python" in result
        assert "## Block 2: Javascript" in result
        assert "```python" in result
        assert "```javascript" in result
        assert "Test function docstring." in result

    def test_copy_code_as_markdown_empty(self):
        """Test copying empty code blocks as markdown."""
        export_utils = ExportUtils()

        result = export_utils.copy_code_as_markdown([])
        assert result == "No code blocks found."

    def test_prepare_download_file(self, mock_datetime):
        """Test preparing file for download."""
        export_utils = ExportUtils()

        filename = export_utils.prepare_download_file("content", "test file", "txt")

        assert filename == "test file_20240101_120000.txt"

    def test_prepare_download_file_clean_filename(self, mock_datetime):
        """Test preparing file with special characters in filename."""
        export_utils = ExportUtils()

        filename = export_utils.prepare_download_file("content", "test@file#name!", "md")

        assert filename == "testfilename_20240101_120000.md"

    def test_prepare_download_file_different_formats(self, mock_datetime):
        """Test preparing files with different formats."""
        export_utils = ExportUtils()

        txt_file = export_utils.prepare_download_file("content", "test", "txt")
        md_file = export_utils.prepare_download_file("content", "test", "md")
        json_file = export_utils.prepare_download_file("content", "test", "json")

        assert txt_file.endswith(".txt")
        assert md_file.endswith(".md")
        assert json_file.endswith(".json")

    def test_export_with_file_size_formatting(self, mock_datetime):
        """Test export with file size formatting in file sources."""
        export_utils = ExportUtils()

        file_sources = [
            {"name": "large_file.py", "type": "python", "size": 2097152},  # 2MB
            {"name": "small_file.py", "type": "python", "size": 1024},  # 1KB
        ]

        sample_data = {
            "enhanced_prompt": "Test prompt",
            "create_breakdown": {"context": "Test"},
            "model_info": {"model": "test"},
            "session_data": {"total_cost": 0.01},
        }

        result = export_utils.export_journey1_content(
            enhanced_prompt=sample_data["enhanced_prompt"],
            create_breakdown=sample_data["create_breakdown"],
            model_info=sample_data["model_info"],
            file_sources=file_sources,
            session_data=sample_data["session_data"],
            format_type="markdown",
        )

        assert "2.0MB" in result
        assert "0.0MB" in result  # Small file shows as 0.0MB

    def test_edge_case_empty_strings(self, mock_datetime):
        """Test export with empty string inputs."""
        export_utils = ExportUtils()

        result = export_utils.export_journey1_content(
            enhanced_prompt="",
            create_breakdown={"context": "", "request": ""},
            model_info={"model": "", "cost": 0},
            file_sources=[],
            session_data={"total_cost": 0},
            format_type="text",
        )

        assert "ENHANCED PROMPT EXPORT" in result
        assert result.count("N/A") >= 2  # Missing values show as N/A

    def test_code_extraction_edge_cases(self):
        """Test code extraction with edge cases."""
        export_utils = ExportUtils()

        # Test with malformed code blocks
        malformed_content = "```python\nincomplete code block"
        blocks = export_utils.extract_code_blocks(malformed_content)
        assert len(blocks) == 0

        # Test with nested code blocks (new secure algorithm behavior)
        # Note: After security fix, this now splits on internal ``` which is safer
        nested_content = "```python\ndef test():\n    code = '''```nested```'''\n```"
        blocks = export_utils.extract_code_blocks(nested_content)
        assert len(blocks) == 2  # Security fix changes behavior: safer but different
        # First block contains the code up to the first internal ```
        assert "def test():" in blocks[0]["content"]
        # Second block contains the remainder after the internal ```
        assert "'''" in blocks[1]["content"]

    def test_language_detection_edge_cases(self):
        """Test language detection with edge cases."""
        export_utils = ExportUtils()

        # Test mixed language indicators
        mixed = "def function() { return 'test'; }"
        assert export_utils._detect_language(mixed) == "python"  # def takes precedence

        # Test empty/whitespace code
        assert export_utils._detect_language("") == "text"
        assert export_utils._detect_language("   \n\t   ") == "text"

        # Test case sensitivity
        assert export_utils._detect_language("SELECT id FROM users") == "sql"
        assert export_utils._detect_language("select id from users") == "sql"
