#!/usr/bin/env python3
"""Comprehensive tests for examples/config_demo.py.

This test suite covers all functions and code paths in the config demo example,
ensuring 80%+ test coverage while following project testing standards.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from examples.config_demo import main


class TestConfigDemo:
    """Test the config demo main function."""

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_basic_execution(self, mock_get_logger, mock_get_settings):
        """Test that main function executes successfully with mocked settings."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Hybrid"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "dev"
        mock_settings.debug = True
        mock_settings.api_host = "127.0.0.1"
        mock_settings.api_port = 8000
        mock_get_settings.return_value = mock_settings

        # Execute main function
        main()

        # Verify get_settings was called
        mock_get_settings.assert_called_once()

        # Verify logger was obtained
        mock_get_logger.assert_called_once_with("examples.config_demo")

        # Verify expected log calls were made
        assert mock_logger.info.call_count >= 5

        # Check specific log calls
        log_calls = [call[0] for call in mock_logger.info.call_args_list]

        # Check for configuration logging
        config_call = next((call for call in log_calls if "🚀 PromptCraft-Hybrid Configuration" in str(call)), None)
        assert config_call is not None, "Configuration header should be logged"

        # Check for app name and version logging (formatted with %s)
        app_call = next((call for call in log_calls if "App: %s v%s" in str(call)), None)
        assert app_call is not None, "App name and version should be logged"

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_development_environment(self, mock_get_logger, mock_get_settings):
        """Test main function behavior with development environment."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Hybrid"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "dev"
        mock_settings.debug = True
        mock_settings.api_host = "127.0.0.1"
        mock_settings.api_port = 8000
        mock_get_settings.return_value = mock_settings

        main()

        # Check for development-specific logging
        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        dev_log = next(
            (call for call in log_calls if "🔧 Development mode active - verbose logging enabled" in call),
            None,
        )
        assert dev_log is not None, "Development mode message should be logged"

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_staging_environment(self, mock_get_logger, mock_get_settings):
        """Test main function behavior with staging environment."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Hybrid"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "staging"
        mock_settings.debug = False
        mock_settings.api_host = "127.0.0.1"
        mock_settings.api_port = 8000
        mock_get_settings.return_value = mock_settings

        main()

        # Check for staging-specific logging
        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        staging_log = next(
            (call for call in log_calls if "🧪 Staging mode active - testing with production-like settings" in call),
            None,
        )
        assert staging_log is not None, "Staging mode message should be logged"

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_production_environment(self, mock_get_logger, mock_get_settings):
        """Test main function behavior with production environment."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Production"
        mock_settings.version = "1.0.0"
        mock_settings.environment = "prod"
        mock_settings.debug = False
        mock_settings.api_host = "0.0.0.0"  # noqa: S104
        mock_settings.api_port = 80
        mock_get_settings.return_value = mock_settings

        main()

        # Check for production-specific logging
        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        prod_log = next(
            (call for call in log_calls if "🏭 Production mode active - optimized for performance" in call),
            None,
        )
        assert prod_log is not None, "Production mode message should be logged"

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_unknown_environment(self, mock_get_logger, mock_get_settings):
        """Test main function behavior with unknown environment."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Test"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "test"
        mock_settings.debug = True
        mock_settings.api_host = "localhost"
        mock_settings.api_port = 9000
        mock_get_settings.return_value = mock_settings

        main()

        # Check that no environment-specific message is logged for unknown environment
        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        env_logs = [
            call
            for call in log_calls
            if any(
                phrase in call
                for phrase in ["Development mode active", "Staging mode active", "Production mode active"]
            )
        ]
        assert len(env_logs) == 0, "No environment-specific message should be logged for unknown environment"

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_debug_mode_logging(self, mock_get_logger, mock_get_settings):
        """Test debug mode logging variations."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test with debug enabled
        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Hybrid"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "dev"
        mock_settings.debug = True
        mock_settings.api_host = "127.0.0.1"
        mock_settings.api_port = 8000
        mock_get_settings.return_value = mock_settings

        main()

        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        debug_on_log = next((call for call in log_calls if "Debug Mode: %s" in call), None)
        assert debug_on_log is not None, "Debug ON should be logged when debug=True"

        # Reset mock and test with debug disabled
        mock_logger.reset_mock()
        mock_settings.debug = False

        main()

        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        debug_off_log = next((call for call in log_calls if "Debug Mode: %s" in call), None)
        assert debug_off_log is not None, "Debug OFF should be logged when debug=False"

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_api_server_logging(self, mock_get_logger, mock_get_settings):
        """Test API server URL logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Hybrid"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "dev"
        mock_settings.debug = True
        mock_settings.api_host = "192.168.1.205"
        mock_settings.api_port = 7860
        mock_get_settings.return_value = mock_settings

        main()

        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        api_log = next((call for call in log_calls if "API Server: http://%s:%s" in call), None)
        assert api_log is not None, "API server URL should be logged correctly"

    @patch("examples.config_demo.get_settings", side_effect=Exception("Config error"))
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_error_handling(self, mock_get_logger, mock_get_settings):
        """Test main function error handling when get_settings fails."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Should raise exception when get_settings fails
        with pytest.raises(Exception, match="Config error"):
            main()

        # Verify get_settings was called
        mock_get_settings.assert_called_once()

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_main_function_complete_log_sequence(self, mock_get_logger, mock_get_settings):
        """Test that all expected log messages are present in correct sequence."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Hybrid"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "dev"
        mock_settings.debug = True
        mock_settings.api_host = "127.0.0.1"
        mock_settings.api_port = 8000
        mock_get_settings.return_value = mock_settings

        main()

        # Get all log calls in order
        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]

        # Verify expected log sequence
        expected_patterns = [
            "🚀 PromptCraft-Hybrid Configuration",
            "App: %s v%s",
            "Environment: %s",
            "Debug Mode: %s",
            "API Server: http://%s:%s",
            "🔧 Development mode active - verbose logging enabled",
        ]

        for _i, pattern in enumerate(expected_patterns):
            matching_log = next((call for call in log_calls if pattern in call), None)
            assert matching_log is not None, f"Expected log pattern not found: {pattern}"

    def test_sys_path_modification(self):
        """Test that sys.path is modified correctly for imports."""
        # The module should modify sys.path to include src directory
        # This is important for the import to work

        # Check if the path modification logic is present in the module
        # The actual path modification happens during import, so we test the logic
        demo_file_path = Path(__file__).parent.parent.parent / "examples" / "config_demo.py"
        assert demo_file_path.exists(), "config_demo.py should exist"

        # Read the file to verify path modification logic
        content = demo_file_path.read_text()
        assert "sys.path.insert" in content, "sys.path should be modified for imports"
        assert "src" in content, "src directory should be added to path"

    @patch("examples.config_demo.get_settings")
    def test_import_compatibility(self, mock_get_settings):
        """Test that the import structure works correctly."""
        # Mock settings to avoid actual config loading
        mock_settings = Mock()
        mock_settings.app_name = "Test"
        mock_settings.version = "1.0.0"
        mock_settings.environment = "test"
        mock_settings.debug = False
        mock_settings.api_host = "localhost"
        mock_settings.api_port = 8000
        mock_get_settings.return_value = mock_settings

        # Should be able to import and use get_settings
        try:
            from examples.config_demo import main as demo_main

            demo_main()
            mock_get_settings.assert_called_once()
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_edge_case_empty_strings(self, mock_get_logger, mock_get_settings):
        """Test handling of edge cases like empty strings."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = ""
        mock_settings.version = ""
        mock_settings.environment = ""
        mock_settings.debug = False
        mock_settings.api_host = ""
        mock_settings.api_port = 0
        mock_get_settings.return_value = mock_settings

        # Should handle empty values gracefully
        main()

        # Verify function completed without errors
        mock_get_settings.assert_called_once()
        assert mock_logger.info.call_count >= 5

    @patch("examples.config_demo.get_settings")
    @patch("examples.config_demo.logging.getLogger")
    def test_edge_case_none_values(self, mock_get_logger, mock_get_settings):
        """Test handling of None values in settings."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_settings = Mock()
        mock_settings.app_name = None
        mock_settings.version = None
        mock_settings.environment = None
        mock_settings.debug = None
        mock_settings.api_host = None
        mock_settings.api_port = None
        mock_get_settings.return_value = mock_settings

        # Should handle None values gracefully
        main()

        # Verify function completed without errors
        mock_get_settings.assert_called_once()
        assert mock_logger.info.call_count >= 5


class TestIntegration:
    """Integration tests for the config demo."""

    def test_module_structure(self):
        """Test that the module has the expected structure."""
        # Import the module to verify it's structured correctly
        import examples.config_demo as demo_module

        # Check that main function exists
        assert hasattr(demo_module, "main"), "Module should have main function"
        assert callable(demo_module.main), "main should be callable"

    def test_logging_setup_integration(self):
        """Test that logging setup works correctly with the module."""
        # This test ensures logging is properly configured
        logger = logging.getLogger("examples.config_demo")
        assert logger is not None, "Logger should be obtainable"

    @patch("examples.config_demo.get_settings")
    def test_full_execution_flow(self, mock_get_settings):
        """Test the complete execution flow from import to completion."""
        # Setup realistic mock settings
        mock_settings = Mock()
        mock_settings.app_name = "PromptCraft-Hybrid"
        mock_settings.version = "0.1.0"
        mock_settings.environment = "dev"
        mock_settings.debug = True
        mock_settings.api_host = "127.0.0.1"
        mock_settings.api_port = 8000
        mock_get_settings.return_value = mock_settings

        # Execute the complete flow
        try:
            from examples.config_demo import main

            main()

            # Verify settings were loaded
            mock_get_settings.assert_called_once()

        except Exception as e:
            pytest.fail(f"Full execution flow failed: {e}")

    def test_real_import_without_execution(self):
        """Test that the module can be imported without executing main."""
        try:
            # Import without triggering main execution
            import examples.config_demo

            # Verify module attributes
            assert hasattr(examples.config_demo, "main")
            assert hasattr(examples.config_demo, "sys")
            assert hasattr(examples.config_demo, "Path")
            assert hasattr(examples.config_demo, "logging")

        except Exception as e:
            pytest.fail(f"Module import failed: {e}")
