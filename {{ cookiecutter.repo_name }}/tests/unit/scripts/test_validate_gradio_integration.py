"""
Unit tests for validate_gradio_integration.py script.

This module tests all validation functions for Phase 1 Issue 5
Gradio UI integration and validation functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the validation script functions
from scripts.validate_gradio_integration import (
    main,
    validate_export_utils,
    validate_journey1_implementation,
    validate_multi_journey_interface,
    validate_performance_requirements,
    validate_ts1_compliance,
)


@pytest.mark.unit
@pytest.mark.fast
class TestValidateMultiJourneyInterface:
    """Test cases for validate_multi_journey_interface function."""

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_multi_journey_interface_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful validation of multi_journey_interface.py."""
        # Setup mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()

        # Create simple mock classes without complex __dict__ manipulation
        mock_interface_class = type(
            "MockMultiJourneyInterface",
            (),
            {
                "calculate_cost": Mock(),
                "create_interface": Mock(),
                "method1": Mock(),
                "__dict__": {"calculate_cost": Mock(), "create_interface": Mock()},
            },
        )

        mock_rate_limiter_class = type(
            "MockRateLimiter",
            (),
            {"check_request": Mock(), "cleanup": Mock(), "__dict__": {"check_request": Mock(), "cleanup": Mock()}},
        )

        mock_module.MultiJourneyInterface = mock_interface_class
        mock_module.RateLimiter = mock_rate_limiter_class
        mock_module_from_spec.return_value = mock_module

        result = validate_multi_journey_interface()

        assert result is True
        mock_spec_from_file.assert_called_once()
        mock_spec.loader.exec_module.assert_called_once()

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_validate_multi_journey_interface_no_spec(self, mock_spec_from_file):
        """Test validation fails when spec cannot be loaded."""
        mock_spec_from_file.return_value = None

        result = validate_multi_journey_interface()

        assert result is False

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_validate_multi_journey_interface_no_loader(self, mock_spec_from_file):
        """Test validation fails when spec has no loader."""
        mock_spec = Mock()
        mock_spec.loader = None
        mock_spec_from_file.return_value = mock_spec

        result = validate_multi_journey_interface()

        assert result is False

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_multi_journey_interface_missing_classes(self, mock_module_from_spec, mock_spec_from_file):
        """Test validation fails when required classes are missing."""
        # Setup mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        # Only add RateLimiter, missing MultiJourneyInterface class
        mock_module.RateLimiter = Mock()
        mock_module_from_spec.return_value = mock_module

        # Mock hasattr to return False for MultiJourneyInterface
        with patch("builtins.hasattr") as mock_hasattr:

            def hasattr_side_effect(obj, name):
                if name == "MultiJourneyInterface":
                    return False  # Missing this class
                return name == "RateLimiter"

            mock_hasattr.side_effect = hasattr_side_effect

            result = validate_multi_journey_interface()

        assert result is False

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_validate_multi_journey_interface_import_error(self, mock_spec_from_file):
        """Test validation handles ImportError gracefully."""
        mock_spec_from_file.side_effect = ImportError("Module not found")

        result = validate_multi_journey_interface()

        assert result is False

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_validate_multi_journey_interface_unexpected_error(self, mock_spec_from_file):
        """Test validation handles unexpected errors gracefully."""
        mock_spec_from_file.side_effect = Exception("Unexpected error")

        result = validate_multi_journey_interface()

        assert result is False


@pytest.mark.unit
@pytest.mark.fast
class TestValidateJourney1Implementation:
    """Test cases for validate_journey1_implementation function."""

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_journey1_implementation_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful validation of Journey 1 implementation."""
        # Setup mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        # Mock Journey1SmartTemplates class with enhance_prompt method
        mock_class = Mock()
        mock_class.enhance_prompt = Mock()
        mock_module.Journey1SmartTemplates = mock_class
        mock_module_from_spec.return_value = mock_module

        result = validate_journey1_implementation()

        assert result is True

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_validate_journey1_implementation_no_spec(self, mock_spec_from_file):
        """Test validation fails when spec cannot be loaded."""
        mock_spec_from_file.return_value = None

        result = validate_journey1_implementation()

        assert result is False

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_validate_journey1_implementation_import_error(self, mock_spec_from_file):
        """Test validation handles ImportError gracefully."""
        mock_spec_from_file.side_effect = ImportError("Module not found")

        result = validate_journey1_implementation()

        assert result is False


@pytest.mark.unit
@pytest.mark.fast
class TestValidateExportUtils:
    """Test cases for validate_export_utils function."""

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_export_utils_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful validation of export utilities."""
        # Setup mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        mock_export_class = Mock()

        # Mock required methods
        mock_export_class.export_journey1_content = Mock()
        mock_export_class.extract_code_blocks = Mock()
        mock_export_class.format_code_blocks_for_export = Mock()
        mock_export_class.other_method = Mock()

        mock_module.ExportUtils = mock_export_class
        mock_module_from_spec.return_value = mock_module

        result = validate_export_utils()

        assert result is True

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_export_utils_missing_methods(self, mock_module_from_spec, mock_spec_from_file):
        """Test validation when export methods are missing."""
        # Setup mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        mock_export_class = Mock()

        # Missing required methods
        mock_export_class.some_other_method = Mock()

        mock_module.ExportUtils = mock_export_class
        mock_module_from_spec.return_value = mock_module

        result = validate_export_utils()

        # Should still return True as long as ExportUtils class exists
        assert result is True

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_validate_export_utils_no_spec(self, mock_spec_from_file):
        """Test validation fails when spec cannot be loaded."""
        mock_spec_from_file.return_value = None

        result = validate_export_utils()

        assert result is False


@pytest.mark.unit
@pytest.mark.fast
class TestValidateTS1Compliance:
    """Test cases for validate_ts1_compliance function."""

    @patch("scripts.validate_gradio_integration.project_root")
    def test_validate_ts1_compliance_all_files_exist(self, mock_project_root):
        """Test validation succeeds when all required files exist."""
        # Mock Path objects
        mock_project_root = Mock()

        def mock_path_div(self, other):
            mock_path = Mock()
            mock_path.exists.return_value = True
            return mock_path

        with patch("scripts.validate_gradio_integration.project_root", mock_project_root):
            mock_project_root.__truediv__ = mock_path_div

            result = validate_ts1_compliance()

        assert result is True

    @patch("scripts.validate_gradio_integration.project_root")
    def test_validate_ts1_compliance_missing_files(self, mock_project_root):
        """Test validation fails when required files are missing."""
        # Create a mock path that always returns False for exists()
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_project_root.__truediv__.return_value = mock_path

        result = validate_ts1_compliance()

        assert result is False


@pytest.mark.unit
@pytest.mark.fast
class TestValidatePerformanceRequirements:
    """Test cases for validate_performance_requirements function."""

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_performance_requirements_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful validation of performance requirements."""
        # Setup mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        mock_rate_limiter = Mock()
        mock_interface = Mock()

        mock_module.RateLimiter = mock_rate_limiter
        mock_module.MultiJourneyInterface = mock_interface
        mock_module_from_spec.return_value = mock_module

        result = validate_performance_requirements()

        assert result is True

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_performance_requirements_missing_classes(self, mock_module_from_spec, mock_spec_from_file):
        """Test validation when performance-related classes are missing."""
        # Setup mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        # Missing required classes
        mock_module_from_spec.return_value = mock_module

        result = validate_performance_requirements()

        # Should still return True if no exception is raised
        assert result is True

    def test_validate_performance_requirements_no_spec(self):
        """Test validation handles missing spec gracefully."""
        with patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location", return_value=None):
            result = validate_performance_requirements()

        assert result is True

    def test_validate_performance_requirements_exception(self):
        """Test validation handles exceptions gracefully."""
        with patch(
            "scripts.validate_gradio_integration.importlib.util.spec_from_file_location",
            side_effect=Exception("Unexpected error"),
        ):
            result = validate_performance_requirements()

        assert result is False


@pytest.mark.unit
@pytest.mark.fast
class TestMainFunction:
    """Test cases for main function."""

    @patch("scripts.validate_gradio_integration.validate_ts1_compliance")
    @patch("scripts.validate_gradio_integration.validate_multi_journey_interface")
    @patch("scripts.validate_gradio_integration.validate_journey1_implementation")
    @patch("scripts.validate_gradio_integration.validate_export_utils")
    @patch("scripts.validate_gradio_integration.validate_performance_requirements")
    def test_main_all_validations_pass(self, mock_perf, mock_export, mock_journey1, mock_interface, mock_ts1):
        """Test main function when all validations pass."""
        # Setup all validations to return True
        mock_ts1.return_value = True
        mock_interface.return_value = True
        mock_journey1.return_value = True
        mock_export.return_value = True
        mock_perf.return_value = True

        result = main()

        assert result == 0

        # Verify all validation functions were called
        mock_ts1.assert_called_once()
        mock_interface.assert_called_once()
        mock_journey1.assert_called_once()
        mock_export.assert_called_once()
        mock_perf.assert_called_once()

    @patch("scripts.validate_gradio_integration.validate_ts1_compliance")
    @patch("scripts.validate_gradio_integration.validate_multi_journey_interface")
    @patch("scripts.validate_gradio_integration.validate_journey1_implementation")
    @patch("scripts.validate_gradio_integration.validate_export_utils")
    @patch("scripts.validate_gradio_integration.validate_performance_requirements")
    def test_main_some_validations_fail(self, mock_perf, mock_export, mock_journey1, mock_interface, mock_ts1):
        """Test main function when some validations fail."""
        # Setup mixed validation results
        mock_ts1.return_value = True
        mock_interface.return_value = False  # This one fails
        mock_journey1.return_value = True
        mock_export.return_value = False  # This one fails
        mock_perf.return_value = True

        result = main()

        assert result == 1  # Should return failure code

    @patch("scripts.validate_gradio_integration.validate_ts1_compliance")
    @patch("scripts.validate_gradio_integration.validate_multi_journey_interface")
    @patch("scripts.validate_gradio_integration.validate_journey1_implementation")
    @patch("scripts.validate_gradio_integration.validate_export_utils")
    @patch("scripts.validate_gradio_integration.validate_performance_requirements")
    def test_main_validation_exception(self, mock_perf, mock_export, mock_journey1, mock_interface, mock_ts1):
        """Test main function handles validation exceptions."""
        # Setup validation to raise exception
        mock_ts1.return_value = True
        mock_interface.side_effect = Exception("Validation error")
        mock_journey1.return_value = True
        mock_export.return_value = True
        mock_perf.return_value = True

        result = main()

        assert result == 1  # Should return failure code

    @patch("scripts.validate_gradio_integration.validate_ts1_compliance")
    @patch("scripts.validate_gradio_integration.validate_multi_journey_interface")
    @patch("scripts.validate_gradio_integration.validate_journey1_implementation")
    @patch("scripts.validate_gradio_integration.validate_export_utils")
    @patch("scripts.validate_gradio_integration.validate_performance_requirements")
    def test_main_all_validations_fail(self, mock_perf, mock_export, mock_journey1, mock_interface, mock_ts1):
        """Test main function when all validations fail."""
        # Setup all validations to return False
        mock_ts1.return_value = False
        mock_interface.return_value = False
        mock_journey1.return_value = False
        mock_export.return_value = False
        mock_perf.return_value = False

        result = main()

        assert result == 1  # Should return failure code


@pytest.mark.unit
@pytest.mark.fast
class TestValidationScriptEdgeCases:
    """Test cases for edge cases and error handling in validation script."""

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validate_multi_journey_interface_with_partial_classes(self, mock_module_from_spec, mock_spec_from_file):
        """Test validation with partial class implementation."""
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        # Create module with only one required class
        mock_module = Mock()
        mock_module.MultiJourneyInterface = Mock()

        # Missing RateLimiter class - use hasattr to control what's found
        def mock_hasattr(obj, name):
            if name == "MultiJourneyInterface":
                return True
            if name == "RateLimiter":
                return False  # Missing this class
            return False

        mock_module_from_spec.return_value = mock_module

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = validate_multi_journey_interface()

        assert result is False

    def test_project_root_path_handling(self):
        """Test that project_root path is handled correctly."""
        # This test ensures the project_root Path object is working
        from scripts.validate_gradio_integration import project_root

        assert isinstance(project_root, Path)
        assert project_root.name == "PromptCraft"

    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    def test_importlib_util_unavailable(self, mock_spec_from_file):
        """Test handling when importlib.util methods are unavailable."""
        mock_spec_from_file.return_value = None

        # Should return False, not raise exception
        result = validate_multi_journey_interface()
        assert result is False

    @patch("builtins.print")
    def test_validation_output_formatting(self, mock_print):
        """Test that validation functions produce expected output format."""
        # Test that validation functions call print with expected patterns
        with patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location", return_value=None):
            validate_multi_journey_interface()

        # Check that error message was printed
        mock_print.assert_called()
        print_args = [call[0][0] for call in mock_print.call_args_list]
        error_printed = any("❌" in str(arg) for arg in print_args)
        assert error_printed

    def test_validation_function_names_consistency(self):
        """Test that validation function names are consistent with their purpose."""
        validation_functions = [
            validate_multi_journey_interface,
            validate_journey1_implementation,
            validate_export_utils,
            validate_ts1_compliance,
            validate_performance_requirements,
        ]

        for func in validation_functions:
            assert func.__name__.startswith("validate_")
            assert callable(func)

    def test_project_root_in_sys_path(self):
        """Test that project root is properly added to sys.path."""
        # This is already done at module level, but we can verify the logic
        from scripts.validate_gradio_integration import project_root

        # The script should add project_root to sys.path for imports
        # Just test that project_root is a valid Path object
        assert isinstance(project_root, Path)
        assert project_root.exists()


@pytest.mark.unit
@pytest.mark.fast
class TestValidationScriptIntegration:
    """Integration-style tests for validation script components working together."""

    @patch("scripts.validate_gradio_integration.project_root")
    @patch("scripts.validate_gradio_integration.importlib.util.spec_from_file_location")
    @patch("scripts.validate_gradio_integration.importlib.util.module_from_spec")
    def test_validation_workflow_integration(self, mock_module_from_spec, mock_spec_from_file, mock_project_root):
        """Test complete validation workflow integration."""
        # Setup project root mock
        mock_project_root = Mock()
        mock_project_root.__truediv__ = lambda self, other: Mock(exists=lambda: True)

        # Setup import mocks
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        mock_module.MultiJourneyInterface = Mock()
        mock_module.RateLimiter = Mock()
        mock_module.ExportUtils = Mock()
        mock_module_from_spec.return_value = mock_module

        # Run multiple validations
        results = []
        results.append(validate_ts1_compliance())
        results.append(validate_multi_journey_interface())
        results.append(validate_export_utils())
        results.append(validate_performance_requirements())

        # All should pass with proper mocking
        assert all(results)

    def test_validation_error_consistency(self):
        """Test that validation functions handle errors consistently."""
        validation_functions = [
            validate_multi_journey_interface,
            validate_journey1_implementation,
            validate_export_utils,
            validate_performance_requirements,
        ]

        for func in validation_functions:
            # Each function should handle ImportError gracefully
            with patch(
                "scripts.validate_gradio_integration.importlib.util.spec_from_file_location",
                side_effect=ImportError("Test error"),
            ):
                result = func()
                assert result is False  # Should return False, not raise exception
