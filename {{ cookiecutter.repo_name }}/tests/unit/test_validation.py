"""Tests for the validation module."""

from unittest.mock import patch

import pytest

from src.config.settings import ApplicationSettings, ConfigurationValidationError
from src.config.validation import validate_configuration_on_startup


class TestValidationModule:
    """Test the validation module functions."""

    @patch("src.config.settings.validate_configuration_on_startup")
    def test_validate_configuration_on_startup_success(self, mock_validate) -> None:
        """Test successful configuration validation."""
        mock_validate.return_value = None
        settings = ApplicationSettings()

        # Should not raise an exception
        validate_configuration_on_startup(settings)
        mock_validate.assert_called_once_with(settings)

    @patch("src.config.settings.validate_configuration_on_startup")
    def test_validate_configuration_on_startup_failure(self, mock_validate) -> None:
        """Test configuration validation failure."""
        mock_validate.side_effect = ConfigurationValidationError("Test error")
        settings = ApplicationSettings()

        # Should raise the ConfigurationValidationError
        with pytest.raises(ConfigurationValidationError):
            validate_configuration_on_startup(settings)
        mock_validate.assert_called_once_with(settings)
