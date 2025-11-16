"""Tests for encryption integration in configuration settings.

This module tests the encryption functionality added to the configuration system,
including SecretStr handling, encrypted file loading, and graceful degradation.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.settings import (
    ApplicationSettings,
    _load_encrypted_env_file,
    reload_settings,
    validate_encryption_available,
)
from src.utils.encryption import EncryptionError, GPGError


class TestEncryptionIntegration:
    """Test encryption integration in configuration settings."""

    def test_secret_str_fields_exist(self):
        """Test that SecretStr fields are properly defined."""
        # Clear environment to test defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        # Check that secret fields are defined and None by default
        assert settings.database_password is None
        assert settings.database_url is None
        assert settings.api_key is None
        assert settings.secret_key is None
        assert settings.azure_openai_api_key is None
        assert settings.jwt_secret_key is None
        assert settings.qdrant_api_key is None
        assert settings.encryption_key is None

    def test_secret_str_validation_empty(self):
        """Test that empty SecretStr values are rejected."""
        with pytest.raises(ValueError, match="Secret values cannot be empty strings"):
            ApplicationSettings(api_key="")

    def test_secret_str_validation_valid(self):
        """Test that valid SecretStr values are accepted."""
        settings = ApplicationSettings(api_key="valid-key-123")
        assert settings.api_key is not None
        assert settings.api_key.get_secret_value() == "valid-key-123"

    def test_secret_str_representation(self):
        """Test that SecretStr values are properly hidden in string representation."""
        settings = ApplicationSettings(api_key="secret-key")
        settings_str = str(settings)
        # Secret values should be hidden
        assert "secret-key" not in settings_str
        assert "SecretStr" in settings_str or "**********" in settings_str

    @patch("src.config.settings.validate_environment_keys")
    def test_validate_encryption_available_success(self, mock_validate):
        """Test encryption validation when keys are available."""
        mock_validate.return_value = None
        assert validate_encryption_available() is True

    @patch("src.config.settings.validate_environment_keys")
    def test_validate_encryption_available_failure(self, mock_validate):
        """Test encryption validation when keys are not available."""
        mock_validate.side_effect = EncryptionError("No GPG keys found")
        assert validate_encryption_available() is False

    @patch("src.config.settings.load_encrypted_env")
    def test_load_encrypted_env_file_success(self, mock_load_encrypted):
        """Test successful loading of encrypted environment file."""
        # Mock the encrypted env loader
        mock_load_encrypted.return_value = {
            "PROMPTCRAFT_API_KEY": "encrypted-api-key",
            "PROMPTCRAFT_SECRET_KEY": "encrypted-secret-key",
            "OTHER_KEY": "should-be-ignored",
        }

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".gpg", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Test the function
            result = _load_encrypted_env_file(tmp_path)

            # Should only include PROMPTCRAFT_ prefixed variables with prefix removed
            expected = {
                "api_key": "encrypted-api-key",
                "secret_key": "encrypted-secret-key",
            }
            assert result == expected

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("src.config.settings.load_encrypted_env")
    def test_load_encrypted_env_file_not_found(self, mock_load_encrypted):
        """Test loading encrypted file when file doesn't exist."""
        non_existent_path = Path("/non/existent/file.gpg")
        result = _load_encrypted_env_file(non_existent_path)
        assert result == {}
        mock_load_encrypted.assert_not_called()

    @patch("src.config.settings.load_encrypted_env")
    def test_load_encrypted_env_file_gpg_error(self, mock_load_encrypted):
        """Test loading encrypted file when GPG decryption fails."""
        mock_load_encrypted.side_effect = GPGError("Decryption failed")

        with tempfile.NamedTemporaryFile(suffix=".gpg", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = _load_encrypted_env_file(tmp_path)
            assert result == {}  # Should return empty dict on error

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("src.config.settings.validate_encryption_available")
    @patch("src.config.settings._detect_environment")
    def test_get_settings_production_warning(
        self,
        mock_detect_env,
        mock_validate_encryption,
        caplog,
    ):
        """Test that production environment warns when encryption is unavailable."""
        mock_detect_env.return_value = "prod"
        mock_validate_encryption.return_value = False

        with caplog.at_level(logging.WARNING):
            # Clear global settings to force reinitialization
            reload_settings()

        # Check that warning was logged
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("production environment detected" in msg.lower() for msg in warning_messages)

    @patch("src.config.settings.validate_encryption_available")
    @patch("src.config.settings._detect_environment")
    def test_get_settings_development_info(
        self,
        mock_detect_env,
        mock_validate_encryption,
        caplog,
    ):
        """Test that development environment shows info when encryption is unavailable."""
        mock_detect_env.return_value = "dev"
        mock_validate_encryption.return_value = False

        with caplog.at_level(logging.INFO):
            # Clear global settings to force reinitialization
            reload_settings()

        # Check that info was logged
        info_messages = [record.message for record in caplog.records if record.levelno >= logging.INFO]
        assert any("development environment" in msg.lower() for msg in info_messages)

    @patch("src.config.settings.validate_encryption_available")
    @patch("src.config.settings._detect_environment")
    def test_get_settings_encryption_available_no_log(
        self,
        mock_detect_env,
        mock_validate_encryption,
        caplog,
    ):
        """Test that no warning is logged when encryption is available."""
        mock_detect_env.return_value = "prod"
        mock_validate_encryption.return_value = True

        with caplog.at_level(logging.WARNING):
            # Clear global settings to force reinitialization
            reload_settings()

            # Should not log any warnings about encryption
            warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
            assert not any("encryption" in msg.lower() for msg in warning_messages)

    def test_environment_variable_overrides(self):
        """Test that environment variables override encrypted file values."""
        # Set environment variable
        os.environ["PROMPTCRAFT_API_KEY"] = "env-override-key"

        try:
            # Reload settings to pick up environment variable
            settings = reload_settings()

            # Environment variable should override
            assert settings.api_key is not None
            assert settings.api_key.get_secret_value() == "env-override-key"

        finally:
            # Clean up
            os.environ.pop("PROMPTCRAFT_API_KEY", None)
            reload_settings()

    def test_backward_compatibility(self):
        """Test that existing non-secret fields still work."""
        settings = ApplicationSettings(
            app_name="Test App",
            environment="dev",
            debug=True,
            api_host="localhost",
            api_port=3000,
        )

        assert settings.app_name == "Test App"
        assert settings.environment == "dev"
        assert settings.debug is True
        assert settings.api_host == "localhost"
        assert settings.api_port == 3000
