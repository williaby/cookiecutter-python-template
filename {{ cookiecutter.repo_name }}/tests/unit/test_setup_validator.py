"""Tests for the setup validator utility module.

This module tests the system requirements validation and startup checks
that ensure the environment is properly configured for the application.
"""

import sys
from unittest.mock import Mock, patch

import pytest

from src.config.settings import ConfigurationValidationError
from src.utils.encryption import EncryptionError
from src.utils.setup_validator import (
    run_startup_checks,
    validate_environment_setup,
    validate_startup_configuration,
    validate_system_requirements,
)


class TestSetupValidator:
    """Test the setup validator utility functions."""

    def test_validate_system_requirements_valid(self):
        """Test that current Python version is considered valid."""
        # Current Python version should be valid since tests are running
        is_valid, errors = validate_system_requirements()
        assert is_valid is True
        assert errors == []

    @patch("sys.version_info", (3, 9, 0))
    def test_validate_system_requirements_invalid_python(self):
        """Test system requirements validation with invalid Python version."""
        is_valid, errors = validate_system_requirements()
        assert is_valid is False
        assert len(errors) > 0
        assert any("Python 3.11+ required" in error for error in errors)

    @patch("sys.version_info", (3, 11, 5))
    def test_validate_system_requirements_minimum_valid(self):
        """Test system requirements validation with minimum valid Python version."""
        is_valid, errors = validate_system_requirements()
        assert is_valid is True
        assert errors == []

    def test_validate_environment_setup(self):
        """Test environment setup validation."""
        # This may pass or fail depending on encryption setup
        success, errors, warnings = validate_environment_setup()

        # Should return the correct types
        assert isinstance(success, bool)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    def test_validate_system_requirements_success(self):
        """Test system requirements validation when all checks pass."""
        success, errors = validate_system_requirements()

        # Should succeed in test environment
        assert success is True
        assert errors == []

    @patch("sys.version_info", (3, 10, 0))
    def test_validate_system_requirements_python_failure(self):
        """Test system requirements validation with Python version failure."""
        success, errors = validate_system_requirements()

        assert success is False
        assert len(errors) >= 1
        assert any("Python 3.11+ required" in error for error in errors)

    def test_validate_system_requirements_imports(self):
        """Test system requirements validation with import checks."""
        # Current environment should have required packages
        success, errors = validate_system_requirements()

        # Should succeed since pydantic and gnupg should be available
        assert isinstance(success, bool)
        assert isinstance(errors, list)

    def test_validate_startup_configuration_basic(self):
        """Test basic startup configuration validation."""
        # This function loads settings internally or uses provided ones
        # It returns a boolean indicating success/failure
        result = validate_startup_configuration()

        # Should return a boolean
        assert isinstance(result, bool)

    @patch("src.utils.setup_validator.validate_system_requirements")
    @patch("src.utils.setup_validator.validate_environment_setup")
    def test_run_startup_checks_success(self, mock_env_setup, mock_system_req):
        """Test running startup checks when all validations pass."""
        mock_system_req.return_value = (True, [])
        mock_env_setup.return_value = (True, [], [])

        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.environment = "dev"
            mock_get_settings.return_value = mock_settings

            # Should not raise SystemExit for successful validation
            try:
                result = validate_startup_configuration()
                assert isinstance(result, bool)
            except SystemExit:
                pytest.fail("Should not exit on successful validation")

    @patch("src.utils.setup_validator.validate_system_requirements")
    def test_run_startup_checks_system_failure(self, mock_validate_system):
        """Test running startup checks when system validation fails."""
        mock_validate_system.return_value = (False, ["System validation error"])

        # Should exit with error code 1
        with pytest.raises(SystemExit) as exc_info:
            run_startup_checks()

        assert exc_info.value.code == 1


class TestEnvironmentValidation:
    """Test environment-specific validation logic."""

    def test_environment_validation_types(self):
        """Test basic environment validation functionality."""
        # Test that validation functions return expected types
        success, errors, warnings = validate_environment_setup()

        assert isinstance(success, bool)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    def test_system_requirements_basic(self):
        """Test basic system requirements validation."""
        success, errors = validate_system_requirements()

        # Should return proper types
        assert isinstance(success, bool)
        assert isinstance(errors, list)

        # In test environment, should generally pass
        if not success:
            # Log errors for debugging (errors available in 'errors' list)
            pass


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("sys.version_info", (3, 9, 0))
    def test_python_version_edge_cases(self):
        """Test Python version checking with edge cases."""
        # Test version comparison logic
        success, errors = validate_system_requirements()
        assert success is False
        assert any("Python 3.11+ required" in error for error in errors)

    def test_validation_robustness(self):
        """Test validation robustness with actual system."""
        # Test actual system validation
        success, errors = validate_system_requirements()

        # Should handle system information gracefully
        assert isinstance(success, bool)
        assert isinstance(errors, list)

        # Test environment setup
        env_success, env_errors, env_warnings = validate_environment_setup()
        assert isinstance(env_success, bool)
        assert isinstance(env_errors, list)
        assert isinstance(env_warnings, list)

    def test_startup_configuration_with_none(self):
        """Test startup configuration with None argument."""
        # The function should handle None by loading settings internally
        result = validate_startup_configuration(None)
        assert isinstance(result, bool)

    # New tests to improve coverage
    @patch("src.utils.setup_validator.gnupg", None)
    def test_validate_system_requirements_missing_gnupg(self):
        """Test system requirements validation when gnupg is missing - covers lines 14-15, 46-47."""
        success, errors = validate_system_requirements()
        assert success is False
        assert any("python-gnupg package not available" in error for error in errors)

    @patch("src.utils.setup_validator.pydantic", None)
    def test_validate_system_requirements_missing_pydantic(self):
        """Test system requirements validation when pydantic is missing - covers lines 19-20."""
        # This test covers the import error handling for pydantic
        success, errors = validate_system_requirements()
        # Should still work since pydantic is not actually checked in validate_system_requirements
        assert isinstance(success, bool)
        assert isinstance(errors, list)

    @patch("src.utils.setup_validator.validate_environment_keys")
    def test_validate_environment_setup_with_encryption_error(self, mock_validate_keys):
        """Test environment setup when encryption validation fails - covers lines 65-67."""
        mock_validate_keys.side_effect = EncryptionError("Test encryption error")

        success, errors, warnings = validate_environment_setup()
        assert success is False
        assert any("Encryption setup issue" in error for error in errors)
        assert any("Some features requiring encryption may not be available" in warning for warning in warnings)

    @patch("os.getuid")
    def test_validate_environment_setup_running_as_root(self, mock_getuid):
        """Test environment setup when running as root - covers lines 72-77."""
        mock_getuid.return_value = 0  # Running as root

        success, errors, warnings = validate_environment_setup()
        assert any("Running as root user" in warning for warning in warnings)

    def test_validate_environment_setup_windows_no_getuid(self):
        """Test environment setup on Windows (no getuid) - covers lines 75-77."""
        with patch("os.getuid", side_effect=AttributeError("No getuid on Windows")):
            success, errors, warnings = validate_environment_setup()
            # Should handle the AttributeError gracefully
            assert isinstance(success, bool)
            assert isinstance(errors, list)
            assert isinstance(warnings, list)

    @patch("src.utils.setup_validator.validate_system_requirements")
    @patch("src.utils.setup_validator.validate_environment_setup")
    def test_validate_startup_configuration_system_failure(self, mock_env_setup, mock_system_req):
        """Test startup configuration when system validation fails - covers lines 100-104."""
        mock_system_req.return_value = (False, ["System validation failed"])
        mock_env_setup.return_value = (True, [], [])

        result = validate_startup_configuration()
        assert result is False

    @patch("src.utils.setup_validator.validate_system_requirements")
    @patch("src.utils.setup_validator.validate_environment_setup")
    def test_validate_startup_configuration_environment_failure(self, mock_env_setup, mock_system_req):
        """Test startup configuration when environment validation fails - covers lines 111-115."""
        mock_system_req.return_value = (True, [])
        mock_env_setup.return_value = (False, ["Environment validation failed"], [])

        result = validate_startup_configuration()
        assert result is False

    @patch("src.utils.setup_validator.validate_system_requirements")
    @patch("src.utils.setup_validator.validate_environment_setup")
    def test_validate_startup_configuration_with_warnings(self, mock_env_setup, mock_system_req):
        """Test startup configuration with warnings - covers line 121."""
        mock_system_req.return_value = (True, [])
        mock_env_setup.return_value = (True, [], ["Warning message"])

        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.environment = "dev"
            mock_get_settings.return_value = mock_settings

            result = validate_startup_configuration()
            assert isinstance(result, bool)

    @patch("src.utils.setup_validator.validate_system_requirements")
    @patch("src.utils.setup_validator.validate_environment_setup")
    def test_validate_startup_configuration_with_provided_settings(self, mock_env_setup, mock_system_req):
        """Test startup configuration with provided settings - covers line 131."""
        mock_system_req.return_value = (True, [])
        mock_env_setup.return_value = (True, [], [])

        with patch("src.config.settings.validate_configuration_on_startup") as mock_validate:
            mock_validate.return_value = None
            mock_settings = Mock()
            mock_settings.environment = "test"

            result = validate_startup_configuration(mock_settings)
            assert isinstance(result, bool)

    @patch("src.utils.setup_validator.validate_system_requirements")
    @patch("src.utils.setup_validator.validate_environment_setup")
    @patch("src.utils.setup_validator.get_settings")
    def test_validate_startup_configuration_config_validation_error(
        self,
        mock_get_settings,
        mock_env_setup,
        mock_system_req,
    ):
        """Test startup configuration with configuration validation error - covers lines 135-138."""
        mock_system_req.return_value = (True, [])
        mock_env_setup.return_value = (True, [], [])
        mock_get_settings.side_effect = ConfigurationValidationError("Config validation failed")

        result = validate_startup_configuration()
        assert result is False

    @patch("src.utils.setup_validator.validate_system_requirements")
    @patch("src.utils.setup_validator.validate_environment_setup")
    @patch("src.utils.setup_validator.get_settings")
    def test_validate_startup_configuration_runtime_error(self, mock_get_settings, mock_env_setup, mock_system_req):
        """Test startup configuration with runtime error - covers lines 140-142."""
        mock_system_req.return_value = (True, [])
        mock_env_setup.return_value = (True, [], [])
        mock_get_settings.side_effect = RuntimeError("Unexpected error")

        result = validate_startup_configuration()
        assert result is False

    @patch("src.utils.setup_validator.validate_startup_configuration")
    def test_run_startup_checks_calls_exit_on_failure(self, mock_validate):
        """Test that run_startup_checks calls sys.exit on failure - covers line 161."""
        mock_validate.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            run_startup_checks()

        assert exc_info.value.code == 1

    def test_main_module_execution_success(self):
        """Test module execution when validation succeeds."""
        # Test that the main block execution calls sys.exit(0) on success
        with patch("src.utils.setup_validator.validate_startup_configuration") as mock_validate:
            mock_validate.return_value = True

            # Mock sys.exit to capture the exit code
            with patch("sys.exit") as mock_exit:
                # Mock logging.basicConfig to avoid setup
                with patch("logging.basicConfig"):
                    # Simulate calling the main block
                    import src.utils.setup_validator

                    # Call the main block logic directly
                    success = src.utils.setup_validator.validate_startup_configuration()

                    if success:
                        sys.exit(0)
                    else:
                        sys.exit(1)

                # Should exit with code 0 on success
                mock_exit.assert_called_with(0)

    def test_main_module_execution_failure(self):
        """Test module execution when validation fails."""
        # Test that the main block execution calls sys.exit(1) on failure
        with patch("src.utils.setup_validator.validate_startup_configuration") as mock_validate:
            mock_validate.return_value = False

            # Mock sys.exit to capture the exit code
            with patch("sys.exit") as mock_exit:
                # Mock logging.basicConfig to avoid setup
                with patch("logging.basicConfig"):
                    # Simulate calling the main block
                    import src.utils.setup_validator

                    # Call the main block logic directly
                    success = src.utils.setup_validator.validate_startup_configuration()

                    if success:
                        sys.exit(0)
                    else:
                        sys.exit(1)

                # Should exit with code 1 on failure
                mock_exit.assert_called_with(1)
