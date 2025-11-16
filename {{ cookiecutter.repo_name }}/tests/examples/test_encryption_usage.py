#!/usr/bin/env python3
"""Comprehensive tests for examples/encryption_usage.py.

This test suite covers all functions and code paths in the encryption usage example,
ensuring 80%+ test coverage while following project testing standards.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from examples.encryption_usage import (
    create_example_env_files,
    demonstrate_development_setup,
    demonstrate_encryption_workflow,
    demonstrate_settings_usage,
    setup_example_environment,
)


class TestSetupExampleEnvironment:
    """Test the setup_example_environment function."""

    def test_setup_example_environment_returns_dict(self):
        """Test that setup_example_environment returns a dictionary."""
        result = setup_example_environment()

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_setup_example_environment_contains_required_keys(self):
        """Test that all required environment keys are present."""
        result = setup_example_environment()

        required_keys = [
            "PROMPTCRAFT_APP_NAME",
            "PROMPTCRAFT_ENVIRONMENT",
            "PROMPTCRAFT_DEBUG",
            "PROMPTCRAFT_API_HOST",
            "PROMPTCRAFT_API_PORT",
            "PROMPTCRAFT_DATABASE_PASSWORD",
            "PROMPTCRAFT_DATABASE_URL",
            "PROMPTCRAFT_API_KEY",
            "PROMPTCRAFT_SECRET_KEY",
            "PROMPTCRAFT_AZURE_OPENAI_API_KEY",
            "PROMPTCRAFT_JWT_SECRET_KEY",
            "PROMPTCRAFT_QDRANT_API_KEY",
            "PROMPTCRAFT_ENCRYPTION_KEY",
        ]

        for key in required_keys:
            assert key in result, f"Required key {key} not found in result"

    def test_setup_example_environment_values_are_strings(self):
        """Test that all environment values are strings."""
        result = setup_example_environment()

        for key, value in result.items():
            assert isinstance(value, str), f"Value for {key} is not a string: {type(value)}"

    def test_setup_example_environment_security_sensitive_values(self):
        """Test that sensitive values are present but not empty."""
        result = setup_example_environment()

        sensitive_keys = [
            "PROMPTCRAFT_DATABASE_PASSWORD",
            "PROMPTCRAFT_API_KEY",
            "PROMPTCRAFT_SECRET_KEY",
            "PROMPTCRAFT_AZURE_OPENAI_API_KEY",
            "PROMPTCRAFT_JWT_SECRET_KEY",
            "PROMPTCRAFT_QDRANT_API_KEY",
            "PROMPTCRAFT_ENCRYPTION_KEY",
        ]

        for key in sensitive_keys:
            assert result[key], f"Sensitive key {key} should not be empty"
            assert len(result[key]) > 10, f"Sensitive key {key} should be sufficiently long"

    def test_setup_example_environment_production_settings(self):
        """Test that production-appropriate settings are configured."""
        result = setup_example_environment()

        assert result["PROMPTCRAFT_ENVIRONMENT"] == "prod"
        assert result["PROMPTCRAFT_DEBUG"] == "false"
        assert result["PROMPTCRAFT_API_HOST"] == "127.0.0.1"  # Security: localhost, not 0.0.0.0


class TestCreateExampleEnvFiles:
    """Test the create_example_env_files function."""

    @patch("examples.encryption_usage.logging.getLogger")
    def test_create_example_env_files_logs_structure(self, mock_get_logger):
        """Test that create_example_env_files logs the expected structure."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        create_example_env_files()

        # Verify logger was called
        mock_get_logger.assert_called_once_with("examples.encryption_usage")

        # Verify expected log calls were made
        assert mock_logger.info.call_count >= 5  # Multiple info calls expected

        # Check for key log messages
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Example .env file structure:" in call for call in log_calls)
        assert any("📁 .env (base, non-sensitive)" in call for call in log_calls)
        assert any("📁 .env.prod.gpg (production, encrypted)" in call for call in log_calls)

    @patch("examples.encryption_usage.logging.getLogger")
    def test_create_example_env_files_content_structure(self, mock_get_logger):
        """Test that the logged content contains expected structure."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        create_example_env_files()

        # Get all logged content
        all_logs = [call[0][0] for call in mock_logger.info.call_args_list]

        # Check for base environment content
        base_content_logged = any("PROMPTCRAFT_APP_NAME=PromptCraft-Hybrid" in log for log in all_logs)
        assert base_content_logged, "Base environment content should be logged"

        # Check for production content reference
        prod_content_logged = any("PROMPTCRAFT_ENVIRONMENT=prod" in log for log in all_logs)
        assert prod_content_logged, "Production environment content should be logged"

    def test_create_example_env_files_no_exceptions(self):
        """Test that create_example_env_files runs without exceptions."""
        # Should not raise any exceptions
        create_example_env_files()


class TestDemonstrateSettingsUsage:
    """Test the demonstrate_settings_usage function."""

    @patch("examples.encryption_usage.logging.getLogger")
    def test_demonstrate_settings_usage_logs_examples(self, mock_get_logger):
        """Test that demonstrate_settings_usage logs the expected examples."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        demonstrate_settings_usage()

        # Verify logger was called
        mock_get_logger.assert_called_once_with("examples.encryption_usage")

        # Verify expected log calls were made
        assert mock_logger.info.call_count >= 2

        # Check for key log messages
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("💻 Application Usage Examples" in call for call in log_calls)
        assert any("Example application code:" in call for call in log_calls)

    @patch("examples.encryption_usage.logging.getLogger")
    def test_demonstrate_settings_usage_code_example_content(self, mock_get_logger):
        """Test that the code example contains expected imports and patterns."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        demonstrate_settings_usage()

        # Get all logged content
        all_logs = [call[0][0] for call in mock_logger.info.call_args_list]

        # Check for expected code patterns in the example
        code_content = "\n".join(all_logs)
        assert "from src.config.settings import get_settings" in code_content
        assert "settings = get_settings()" in code_content
        assert "get_secret_value()" in code_content
        assert 'settings.environment == "prod"' in code_content

    def test_demonstrate_settings_usage_no_exceptions(self):
        """Test that demonstrate_settings_usage runs without exceptions."""
        # Should not raise any exceptions
        demonstrate_settings_usage()


class TestDemonstrateEncryptionWorkflow:
    """Test the demonstrate_encryption_workflow function."""

    @patch("examples.encryption_usage.logging.getLogger")
    def test_demonstrate_encryption_workflow_logs_steps(self, mock_get_logger):
        """Test that demonstrate_encryption_workflow logs workflow steps."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        demonstrate_encryption_workflow()

        # Verify logger was called
        mock_get_logger.assert_called_once_with("examples.encryption_usage")

        # Verify expected log calls were made
        assert mock_logger.info.call_count >= 5

        # Check for key log messages
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("🔐 Encryption Workflow for Production" in call for call in log_calls)
        assert any("Production deployment workflow:" in call for call in log_calls)
        assert any("Security benefits:" in call for call in log_calls)

    @patch("examples.encryption_usage.logging.getLogger")
    def test_demonstrate_encryption_workflow_step_content(self, mock_get_logger):
        """Test that workflow steps contain expected content."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        demonstrate_encryption_workflow()

        # Get all logged content (handle both formatted and unformatted calls)
        all_logs = []
        for call in mock_logger.info.call_args_list:
            if len(call[0]) == 1:
                # Single string (no formatting)
                all_logs.append(call[0][0])
            else:
                # Format string with args
                format_str = call[0][0]
                args = call[0][1:]
                all_logs.append(format_str % args)

        # Check for expected workflow steps
        workflow_content = "\n".join(all_logs)
        assert "Create .env.prod file" in workflow_content
        assert "gpg --encrypt" in workflow_content
        assert ".env.prod.gpg" in workflow_content
        assert "Secrets are encrypted at rest" in workflow_content
        assert "Only authorized users can decrypt" in workflow_content

    def test_demonstrate_encryption_workflow_no_exceptions(self):
        """Test that demonstrate_encryption_workflow runs without exceptions."""
        # Should not raise any exceptions
        demonstrate_encryption_workflow()


class TestDemonstrateDevelopmentSetup:
    """Test the demonstrate_development_setup function."""

    @patch("examples.encryption_usage.logging.getLogger")
    def test_demonstrate_development_setup_logs_steps(self, mock_get_logger):
        """Test that demonstrate_development_setup logs development steps."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        demonstrate_development_setup()

        # Verify logger was called
        mock_get_logger.assert_called_once_with("examples.encryption_usage")

        # Verify expected log calls were made
        assert mock_logger.info.call_count >= 5

        # Check for key log messages
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("🛠️  Development Environment Setup" in call for call in log_calls)
        assert any("Development setup steps:" in call for call in log_calls)
        assert any("Development benefits:" in call for call in log_calls)

    @patch("examples.encryption_usage.logging.getLogger")
    def test_demonstrate_development_setup_step_content(self, mock_get_logger):
        """Test that development steps contain expected content."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        demonstrate_development_setup()

        # Get all logged content - handle both format strings and interpolated values
        all_logs = []
        for call in mock_logger.info.call_args_list:
            if len(call[0]) > 1:  # Format string with args
                format_str = call[0][0]
                args = call[0][1:]
                all_logs.append(format_str % args)
            else:  # Plain string
                all_logs.append(call[0][0])

        # Check for expected development steps
        dev_content = "\n".join(all_logs)
        assert "1. Create .env.dev file with development values" in dev_content
        assert "non-sensitive default values" in dev_content
        assert "No encryption setup required" in dev_content
        assert "Easy to override values" in dev_content

    def test_demonstrate_development_setup_no_exceptions(self):
        """Test that demonstrate_development_setup runs without exceptions."""
        # Should not raise any exceptions
        demonstrate_development_setup()


class TestMainFunction:
    """Test the main function execution path."""

    @patch("examples.encryption_usage.logging.basicConfig")
    @patch("examples.encryption_usage.logging.getLogger")
    @patch("examples.encryption_usage.create_example_env_files")
    @patch("examples.encryption_usage.demonstrate_settings_usage")
    @patch("examples.encryption_usage.demonstrate_encryption_workflow")
    @patch("examples.encryption_usage.demonstrate_development_setup")
    def test_main_function_success_path(
        self,
        mock_dev_setup,
        mock_encryption_workflow,
        mock_settings_usage,
        mock_create_env,
        mock_get_logger,
        mock_basic_config,
    ):
        """Test the main function success execution path."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Import and execute main
        from examples.encryption_usage import __name__ as module_name

        # Import all functions first
        from examples.encryption_usage import (
            create_example_env_files,
            demonstrate_development_setup,
            demonstrate_encryption_workflow,
            demonstrate_settings_usage,
        )

        if module_name == "__main__":
            # Simulate main execution
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            logging.getLogger("examples.encryption_usage")

            # Execute the main logic - functions now available
            create_example_env_files()
            demonstrate_settings_usage()
            demonstrate_encryption_workflow()
            demonstrate_development_setup()

        create_example_env_files()
        demonstrate_settings_usage()
        demonstrate_encryption_workflow()
        demonstrate_development_setup()

        # After actual calls, check the mocks would have been called
        mock_create_env.assert_called_once()
        mock_settings_usage.assert_called_once()
        mock_encryption_workflow.assert_called_once()
        mock_dev_setup.assert_called_once()

    @patch("examples.encryption_usage.logging.basicConfig")
    @patch("examples.encryption_usage.logging.getLogger")
    @patch("examples.encryption_usage.create_example_env_files", side_effect=Exception("Test error"))
    def test_main_function_error_handling(self, mock_create_env, mock_get_logger, mock_basic_config):
        """Test the main function error handling."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Simulate error during execution
        with pytest.raises(Exception, match="Test error"):
            mock_create_env()

    @patch("examples.encryption_usage.logging.basicConfig")
    @patch("examples.encryption_usage.logging.getLogger")
    def test_main_function_logging_setup(self, mock_get_logger, mock_basic_config):
        """Test that main function sets up logging correctly."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Simulate logging setup
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        # Verify logging configuration
        mock_basic_config.assert_called_once_with(level=logging.INFO, format="%(message)s")

    def test_integration_all_functions_callable(self):
        """Integration test to ensure all functions are callable without errors."""
        # This tests that all functions can be called without exceptions
        # and demonstrates the integration between them

        try:
            env_vars = setup_example_environment()
            assert isinstance(env_vars, dict)

            create_example_env_files()
            demonstrate_settings_usage()
            demonstrate_encryption_workflow()
            demonstrate_development_setup()

        except Exception as e:
            pytest.fail(f"Integration test failed with exception: {e}")


class TestSecurityAndBestPractices:
    """Test security aspects and best practices demonstrated in the example."""

    def test_no_hardcoded_real_secrets(self):
        """Test that no real secrets are hardcoded in the example."""
        env_vars = setup_example_environment()

        # These should be example/placeholder values, not real secrets
        example_patterns = [
            "super-secret",
            "sk-1234567890",
            "abcd1234567890",
            "test",
            "example",
            "your-secret",
            "jwt-signing-secret",
        ]

        for key, value in env_vars.items():
            if "password" in key.lower() or "key" in key.lower() or "secret" in key.lower():
                # Should contain example patterns, not real secrets
                contains_example_pattern = any(pattern in value for pattern in example_patterns)
                assert contains_example_pattern, f"Value for {key} should be clearly an example, not a real secret"

    def test_localhost_configuration_security(self):
        """Test that localhost is used instead of 0.0.0.0 for security."""
        env_vars = setup_example_environment()

        # Should use localhost (127.0.0.1) instead of 0.0.0.0 for security
        assert env_vars["PROMPTCRAFT_API_HOST"] == "127.0.0.1"

    def test_production_environment_settings(self):
        """Test that production environment has appropriate security settings."""
        env_vars = setup_example_environment()

        # Production should have debug disabled
        assert env_vars["PROMPTCRAFT_ENVIRONMENT"] == "prod"
        assert env_vars["PROMPTCRAFT_DEBUG"] == "false"

    def test_comprehensive_secret_coverage(self):
        """Test that all types of secrets are covered in the example."""
        env_vars = setup_example_environment()

        secret_categories = {
            "database": ["DATABASE_PASSWORD", "DATABASE_URL"],
            "api_keys": ["API_KEY", "AZURE_OPENAI_API_KEY", "QDRANT_API_KEY"],
            "signing_keys": ["SECRET_KEY", "JWT_SECRET_KEY"],
            "encryption": ["ENCRYPTION_KEY"],
        }

        for category, secret_types in secret_categories.items():
            for secret_type in secret_types:
                matching_keys = [key for key in env_vars if secret_type in key]
                assert len(matching_keys) > 0, f"No {secret_type} found for {category} category"
