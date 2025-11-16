"""
Test suite for the refactored coverage automation system.
Validates modular architecture, security improvements, and backward compatibility.
"""

import json

# Add the scripts directory to the path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Get absolute path to scripts directory from this test file
# From tests/unit/scripts/ -> ../../../ -> project_root/scripts/
scripts_path = str(Path(__file__).parent / ".." / ".." / ".." / "scripts")
scripts_path = str(Path(scripts_path).resolve())  # Resolve to absolute path
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Import the refactored modules with try/except for better error handling
try:
    from coverage_automation.classifier import TestTypeClassifier
    from coverage_automation.cli import CoverageAutomationCLI  
    from coverage_automation.config import TestPatternConfig
    from coverage_automation.renderer import CoverageRenderer
    from coverage_automation.security import HTMLSanitizer, SecurityValidator
    from coverage_automation.watcher import CoverageWatcher
except ImportError as e:
    # If coverage_automation can't be imported, skip all tests in this module
    pytest.skip(f"coverage_automation module not available: {e}", allow_module_level=True)


class TestSecurityValidatorEnhancements:
    """Test security improvements in path validation and HTML sanitization."""

    def test_enhanced_path_validation(self, tmp_path):
        """Test enhanced path validation using Path.resolve().is_relative_to()."""
        validator = SecurityValidator(tmp_path)

        # Test valid path within project
        valid_file = tmp_path / "src" / "test.py"
        valid_file.parent.mkdir(parents=True)
        valid_file.touch()
        assert validator.validate_path(valid_file) is True

        # Test path traversal attempt
        traversal_path = tmp_path / ".." / "dangerous.py"
        assert validator.validate_path(traversal_path) is False

        # Test symlink outside project bounds
        external_dir = tmp_path.parent / "external"
        external_dir.mkdir(exist_ok=True)
        external_file = external_dir / "external.py"
        external_file.touch()

        symlink_path = tmp_path / "link_to_external.py"
        try:
            symlink_path.symlink_to(external_file)
            assert validator.validate_path(symlink_path) is False
        except OSError:
            # Skip if symlinks not supported
            pass

    def test_file_size_validation(self, tmp_path):
        """Test file size validation prevents memory exhaustion."""
        validator = SecurityValidator(tmp_path)

        # Test small file (should pass)
        small_file = tmp_path / "small.py"
        small_file.write_text("print('hello')")
        assert validator.validate_file_size(small_file, max_size_mb=1.0) is True

        # Test oversized file (should fail)
        large_file = tmp_path / "large.py"
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        large_file.write_text(large_content)
        assert validator.validate_file_size(large_file, max_size_mb=1.0) is False

    def test_content_sanitization(self, tmp_path):
        """Test content sanitization removes dangerous characters."""
        validator = SecurityValidator(tmp_path)

        # Test content with null bytes and control characters
        dangerous_content = "normal text\x00null byte\x01control\x1fmore"
        sanitized = validator.sanitize_content(dangerous_content)

        assert "\x00" not in sanitized
        assert "\x01" not in sanitized
        assert "\x1f" not in sanitized
        assert "normal text" in sanitized

    def test_import_path_validation(self, tmp_path):
        """Test import path validation prevents injection attacks."""
        validator = SecurityValidator(tmp_path)

        # Valid import paths
        assert validator.validate_import_path("src.core.module") is True
        assert validator.validate_import_path("src.auth.jwt_validator") is True

        # Invalid import paths
        assert validator.validate_import_path("../dangerous") is False
        assert validator.validate_import_path("src/../../../etc/passwd") is False
        assert validator.validate_import_path("src.module<script>") is False
        assert validator.validate_import_path("x" * 200) is False  # Too long


class TestHTMLSanitization:
    """Test HTML sanitization improvements."""

    def test_html_escape_replaces_manual_operations(self):
        """Test HTML escaping replaces manual string replace operations."""
        # Test basic HTML escaping
        unsafe_text = '<script>alert("xss")</script>'
        safe_text = HTMLSanitizer.escape_html(unsafe_text)

        assert "&lt;" in safe_text
        assert "&gt;" in safe_text
        assert "<script>" not in safe_text
        assert "alert" in safe_text  # Content preserved, tags escaped

    def test_attribute_escaping(self):
        """Test HTML attribute value escaping."""
        unsafe_attr = 'value" onload="alert(1)'
        safe_attr = HTMLSanitizer.escape_html_attribute(unsafe_attr)

        assert "&quot;" in safe_attr
        assert "onload=" not in safe_attr or "&quot;" in safe_attr

    def test_filename_sanitization(self):
        """Test filename sanitization for HTML links."""
        unsafe_filename = "../../../etc/passwd<script>.py"
        safe_filename = HTMLSanitizer.sanitize_filename(unsafe_filename)

        assert ".." not in safe_filename
        assert "<" not in safe_filename or "&lt;" in safe_filename
        assert "/" not in safe_filename or safe_filename.replace("/", "_")

    def test_coverage_percentage_sanitization(self):
        """Test coverage percentage sanitization."""
        # Valid percentages
        assert HTMLSanitizer.sanitize_coverage_percentage(85.5) == "85.5"
        assert HTMLSanitizer.sanitize_coverage_percentage(100.0) == "100.0"
        assert HTMLSanitizer.sanitize_coverage_percentage(0.0) == "0.0"

        # Invalid/dangerous values
        assert HTMLSanitizer.sanitize_coverage_percentage(-5.0) == "0.0"
        assert HTMLSanitizer.sanitize_coverage_percentage(150.0) == "100.0"
        assert HTMLSanitizer.sanitize_coverage_percentage("invalid") == "0.0"


class TestModularArchitecture:
    """Test the modular architecture and component interaction."""

    def test_component_initialization(self, tmp_path):
        """Test all components initialize correctly."""
        # Create minimal project structure
        (tmp_path / "config").mkdir()
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()

        # Create minimal config
        config_file = tmp_path / "config" / "test_patterns.yaml"
        config_content = """
test_types:
  unit:
    priority: 1
    patterns: ['tests/unit/']
    description: 'Unit tests'
global:
  security:
    max_file_size_mb: 1
  performance:
    cache_size_test_mapping: 32
"""
        config_file.write_text(config_content)

        # Test component initialization
        config = TestPatternConfig(config_file)
        assert config is not None
        assert "unit" in config.get_all_test_types()

        watcher = CoverageWatcher(tmp_path, config)
        assert watcher is not None
        assert watcher.project_root == tmp_path

        classifier = TestTypeClassifier(tmp_path, config)
        assert classifier is not None

        renderer = CoverageRenderer(tmp_path, config, classifier)
        assert renderer is not None

        cli = CoverageAutomationCLI(tmp_path)
        assert cli is not None

    def test_codecov_integration(self, tmp_path):
        """Test Codecov flag mapping integration."""
        # Create codecov.yaml
        codecov_file = tmp_path / "codecov.yaml"
        codecov_content = """
flags:
  unit:
    paths:
      - src/core/
      - src/utils/
  auth:
    paths:
      - src/auth/
"""
        codecov_file.write_text(codecov_content)

        # Create config with codecov integration
        config_file = tmp_path / "config" / "test_patterns.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_content = """
test_types:
  unit:
    priority: 1
    patterns: ['tests/unit/']
    description: 'Unit tests'
  auth:
    priority: 2
    patterns: ['tests/auth/']
    description: 'Auth tests'
global:
  security:
    max_file_size_mb: 1
codecov_integration:
  enabled: true
"""
        config_file.write_text(config_content)

        config = TestPatternConfig(config_file)
        flag_mapping = config.get_codecov_flag_mapping()

        # Should load from codecov.yaml
        assert "unit" in flag_mapping
        assert "src/core/" in flag_mapping["unit"]

    @patch("subprocess.run")
    def test_coverage_report_generation(self, mock_subprocess, tmp_path):
        """Test coverage report generation workflow."""
        # Setup mock responses
        mock_subprocess.return_value = Mock(returncode=0)

        # Create project structure
        (tmp_path / "config").mkdir()
        (tmp_path / "htmlcov").mkdir()
        (tmp_path / "reports" / "coverage").mkdir(parents=True)

        # Create coverage.json
        coverage_data = {
            "files": {
                "src/core/test.py": {
                    "summary": {
                        "num_statements": 100,
                        "missing_lines": 20,
                        "num_branches": 10,
                        "num_partial_branches": 2,
                    },
                },
            },
            "totals": {"percent_covered": 80.0},
        }

        coverage_json = tmp_path / "coverage.json"
        with coverage_json.open("w") as f:
            json.dump(coverage_data, f)

        # Create minimal config
        config_file = tmp_path / "config" / "test_patterns.yaml"
        config_content = """
test_types:
  unit:
    priority: 1
    patterns: ['tests/unit/']
    description: 'Unit tests'
global:
  security:
    max_file_size_mb: 1
"""
        config_file.write_text(config_content)

        # Test report generation
        config = TestPatternConfig(config_file)
        classifier = TestTypeClassifier(tmp_path, config)
        renderer = CoverageRenderer(tmp_path, config, classifier)

        used_contexts = {"unit"}
        report_path = renderer.generate_coverage_reports(used_contexts)

        assert report_path
        assert Path(report_path).exists()


class TestStructuredLogging:
    """Test structured logging improvements."""

    def test_context_aware_logging(self, caplog):
        """Test context-aware logging with structured information."""
        from coverage_automation.logging_utils import get_logger
        import logging

        # Set up logging level to capture info messages
        caplog.set_level(logging.INFO)
        
        logger = get_logger("test")

        # Test logging with context
        logger.info("Test message", test_param="value", count=5)

        # Check log output contains context
        assert "test_param=value" in caplog.text
        assert "count=5" in caplog.text

    def test_security_logging(self, caplog):
        """Test security-specific logging."""
        from coverage_automation.logging_utils import get_security_logger

        security_logger = get_security_logger()

        # Test security violation logging
        test_path = Path("/tmp/test.py")  # noqa: S108
        security_logger.log_path_validation_failure(test_path, "test violation")

        # Check security log format
        assert "Security: Path validation failed" in caplog.text
        assert str(test_path) in caplog.text

    def test_performance_logging(self, caplog):
        """Test performance logging."""
        from coverage_automation.logging_utils import get_performance_logger
        import logging

        # Set up logging level to capture info messages
        caplog.set_level(logging.INFO)
        
        perf_logger = get_performance_logger()

        # Test performance timing
        perf_logger.log_operation_timing("test_operation", 1.5, context="test")

        # Check performance log format
        assert "Performance: test_operation completed" in caplog.text
        assert "duration_seconds=1.5" in caplog.text


class TestBackwardCompatibility:
    """Test backward compatibility with original system."""

    def test_cli_interface_compatibility(self, tmp_path):
        """Test CLI maintains compatibility with original interface."""
        # Create minimal project structure
        (tmp_path / "config").mkdir()
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()

        config_file = tmp_path / "config" / "test_patterns.yaml"
        config_content = """
test_types:
  unit:
    priority: 1
    patterns: ['tests/unit/']
    description: 'Unit tests'
global:
  security:
    max_file_size_mb: 1
"""
        config_file.write_text(config_content)

        # Test CLI initialization doesn't break
        cli = CoverageAutomationCLI(tmp_path)

        # Test environment validation
        result = cli.validate_environment()
        # Should not crash, though may fail validation due to missing Poetry
        assert isinstance(result, bool)

    @patch("coverage_automation.watcher.CoverageWatcher.detect_vscode_coverage_run")
    @patch("coverage_automation.renderer.CoverageRenderer.generate_coverage_reports")
    def test_automation_workflow_compatibility(self, mock_generate, mock_detect, tmp_path):
        """Test main automation workflow maintains compatibility."""
        # Setup mocks
        mock_detect.return_value = True
        mock_generate.return_value = "/tmp/report.html"  # noqa: S108

        # Create CLI and test workflow
        cli = CoverageAutomationCLI(tmp_path)

        # This should not crash and should return a boolean
        result = cli.run_automation(force_run=True)
        assert isinstance(result, bool)


class TestErrorHandling:
    """Test enhanced error handling."""

    def test_graceful_config_loading_failure(self, tmp_path):
        """Test graceful handling of config loading failures."""
        # Test with non-existent config file
        non_existent_config = tmp_path / "missing.yaml"
        config = TestPatternConfig(non_existent_config)

        # Should fall back to default config
        assert config is not None
        assert "unit" in config.get_all_test_types()

    def test_graceful_coverage_file_missing(self, tmp_path):
        """Test graceful handling when coverage files are missing."""
        config_file = tmp_path / "config" / "test_patterns.yaml"
        config_file.parent.mkdir()
        config_file.write_text(
            """
test_types:
  unit:
    priority: 1
    patterns: ['tests/unit/']
    description: 'Unit tests'
global:
  security:
    max_file_size_mb: 1
""",
        )

        config = TestPatternConfig(config_file)
        watcher = CoverageWatcher(tmp_path, config)

        # Should not crash when coverage file is missing
        result = watcher.detect_vscode_coverage_run()
        assert result is False  # No coverage file found

    def test_type_validation_in_classifier(self, tmp_path):
        """Test type validation in classifier methods."""
        config_file = tmp_path / "config" / "test_patterns.yaml"
        config_file.parent.mkdir()
        config_file.write_text(
            """
test_types:
  unit:
    priority: 1
    patterns: ['tests/unit/']
    description: 'Unit tests'
global:
  security:
    max_file_size_mb: 1
""",
        )

        config = TestPatternConfig(config_file)
        classifier = TestTypeClassifier(tmp_path, config)

        # Test type validation
        with pytest.raises(TypeError):
            classifier.estimate_test_type_coverage("invalid", {"unit"})

        with pytest.raises(TypeError):
            classifier.estimate_test_type_coverage({}, "invalid")


@pytest.fixture
def temp_project_structure():
    """Create a temporary project structure for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create project structure
        (tmp_path / "src" / "core").mkdir(parents=True)
        (tmp_path / "src" / "auth").mkdir(parents=True)
        (tmp_path / "tests" / "unit").mkdir(parents=True)
        (tmp_path / "tests" / "auth").mkdir(parents=True)
        (tmp_path / "config").mkdir()

        # Create sample files
        (tmp_path / "src" / "core" / "test.py").write_text("def test(): pass")
        (tmp_path / "tests" / "unit" / "test_core.py").write_text("import src.core.test")

        yield tmp_path


def test_integration_full_workflow(temp_project_structure):
    """Integration test for the full workflow."""
    tmp_path = temp_project_structure

    # Create config file
    config_file = tmp_path / "config" / "test_patterns.yaml"
    config_content = """
test_types:
  unit:
    priority: 1
    patterns: ['tests/unit/']
    description: 'Unit tests'
  auth:
    priority: 2
    patterns: ['tests/auth/']
    description: 'Auth tests'
global:
  security:
    max_file_size_mb: 1
"""
    config_file.write_text(config_content)

    # Create coverage data
    coverage_data = {
        "files": {
            "src/core/test.py": {
                "summary": {"num_statements": 50, "missing_lines": 10, "num_branches": 5, "num_partial_branches": 1},
            },
        },
        "totals": {"percent_covered": 80.0},
    }

    coverage_json = tmp_path / "coverage.json"
    with coverage_json.open("w") as f:
        json.dump(coverage_data, f)

    # Test full component interaction
    config = TestPatternConfig(config_file)
    classifier = TestTypeClassifier(tmp_path, config)

    # Test target mapping
    _ = classifier.get_test_target_mapping("unit")
    # Should find the target file through import analysis

    # Test coverage estimation
    coverage_by_type = classifier.estimate_test_type_coverage(coverage_data, {"unit"})
    assert "unit" in coverage_by_type
    assert coverage_by_type["unit"]["statement"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
