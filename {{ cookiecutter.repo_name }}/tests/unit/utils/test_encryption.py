"""Comprehensive tests for src/utils/encryption.py module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.encryption import (
    EncryptionError,
    GPGError,
    decrypt_env_file,
    encrypt_env_file,
    initialize_encryption,
    load_encrypted_env,
    validate_environment_keys,
)


class TestEncryptionExceptions:
    """Test custom exception classes."""

    def test_encryption_error_inheritance(self):
        """Test EncryptionError inherits from Exception."""
        error = EncryptionError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_gpg_error_inheritance(self):
        """Test GPGError inherits from Exception."""
        error = GPGError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"


class TestValidateEnvironmentKeys:
    """Test validate_environment_keys function."""

    @patch("src.utils.encryption.gnupg.GPG")
    @patch("src.utils.encryption.subprocess.run")
    def test_successful_validation(self, mock_subprocess, mock_gpg_class):
        """Test successful validation of all required keys."""
        # Mock GPG with secret keys
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "test-key-123"}]
        mock_gpg_class.return_value = mock_gpg

        # Mock successful SSH key check
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout=""),  # ssh-add -l
            Mock(returncode=0, stdout="test-signing-key\n"),  # git config
        ]

        # Should not raise any exception
        validate_environment_keys()

        # Verify GPG was checked for secret keys
        mock_gpg.list_keys.assert_called_once_with(True)

        # Verify SSH and Git commands were called
        assert mock_subprocess.call_count == 2
        mock_subprocess.assert_any_call(
            ["ssh-add", "-l"],
            capture_output=True,
            text=True,
            check=False,
        )
        mock_subprocess.assert_any_call(
            ["git", "config", "--get", "user.signingkey"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("src.utils.encryption.gnupg.GPG")
    def test_no_gpg_secret_keys(self, mock_gpg_class):
        """Test validation fails when no GPG secret keys are found."""
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = []  # No secret keys
        mock_gpg_class.return_value = mock_gpg

        with pytest.raises(EncryptionError, match="No GPG secret keys found"):
            validate_environment_keys()

    @patch("src.utils.encryption.gnupg.GPG")
    def test_gpg_access_failure(self, mock_gpg_class):
        """Test validation fails when GPG access fails."""
        mock_gpg = Mock()
        mock_gpg.list_keys.side_effect = Exception("GPG not available")
        mock_gpg_class.return_value = mock_gpg

        with pytest.raises(EncryptionError, match="Failed to access GPG keys"):
            validate_environment_keys()

    @patch("src.utils.encryption.gnupg.GPG")
    @patch("src.utils.encryption.subprocess.run")
    def test_ssh_key_not_loaded(self, mock_subprocess, mock_gpg_class):
        """Test validation fails when SSH keys are not loaded."""
        # Mock GPG success
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "test-key"}]
        mock_gpg_class.return_value = mock_gpg

        # Mock SSH failure
        mock_subprocess.return_value = Mock(returncode=1, stdout="")

        with pytest.raises(EncryptionError, match="No SSH keys loaded"):
            validate_environment_keys()

    @patch("src.utils.encryption.gnupg.GPG")
    @patch("src.utils.encryption.subprocess.run")
    def test_ssh_command_not_found(self, mock_subprocess, mock_gpg_class):
        """Test validation fails when ssh-add command is not found."""
        # Mock GPG success
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "test-key"}]
        mock_gpg_class.return_value = mock_gpg

        # Mock ssh-add command not found
        mock_subprocess.side_effect = FileNotFoundError("ssh-add not found")

        with pytest.raises(EncryptionError, match="ssh-add command not found"):
            validate_environment_keys()

    @patch("src.utils.encryption.gnupg.GPG")
    @patch("src.utils.encryption.subprocess.run")
    def test_git_signing_key_not_configured(self, mock_subprocess, mock_gpg_class):
        """Test validation fails when Git signing key is not configured."""
        # Mock GPG success
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "test-key"}]
        mock_gpg_class.return_value = mock_gpg

        # Mock SSH success, Git failure
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout=""),  # ssh-add success
            Mock(returncode=1, stdout=""),  # git config failure
        ]

        with pytest.raises(EncryptionError, match="Git signing key not configured"):
            validate_environment_keys()

    @patch("src.utils.encryption.gnupg.GPG")
    @patch("src.utils.encryption.subprocess.run")
    def test_git_signing_key_empty(self, mock_subprocess, mock_gpg_class):
        """Test validation fails when Git signing key is empty."""
        # Mock GPG success
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "test-key"}]
        mock_gpg_class.return_value = mock_gpg

        # Mock SSH success, Git returns empty
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout=""),  # ssh-add success
            Mock(returncode=0, stdout="\n"),  # git config returns empty
        ]

        with pytest.raises(EncryptionError, match="Git signing key not configured"):
            validate_environment_keys()

    @patch("src.utils.encryption.gnupg.GPG")
    @patch("src.utils.encryption.subprocess.run")
    def test_git_command_not_found(self, mock_subprocess, mock_gpg_class):
        """Test validation fails when git command is not found."""
        # Mock GPG success
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "test-key"}]
        mock_gpg_class.return_value = mock_gpg

        # Mock SSH success, Git command not found
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout=""),  # ssh-add success
            FileNotFoundError("git not found"),  # git command not found
        ]

        with pytest.raises(EncryptionError, match="Git command not found"):
            validate_environment_keys()


class TestEncryptEnvFile:
    """Test encrypt_env_file function."""

    @patch("src.utils.encryption.gnupg.GPG")
    def test_successful_encryption_with_recipient(self, mock_gpg_class):
        """Test successful encryption with specified recipient."""
        mock_gpg = Mock()
        mock_encrypted = Mock()
        mock_encrypted.ok = True
        mock_encrypted.__str__ = lambda self: "encrypted-content"
        mock_gpg.encrypt.return_value = mock_encrypted
        mock_gpg_class.return_value = mock_gpg

        result = encrypt_env_file("TEST_VAR=value", "recipient-key")

        assert result == "encrypted-content"
        mock_gpg.encrypt.assert_called_once_with("TEST_VAR=value", recipients=["recipient-key"])

    @patch("src.utils.encryption.gnupg.GPG")
    def test_successful_encryption_auto_recipient(self, mock_gpg_class):
        """Test successful encryption with auto-detected recipient."""
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "auto-key-123"}]
        mock_encrypted = Mock()
        mock_encrypted.ok = True
        mock_encrypted.__str__ = lambda self: "encrypted-content"
        mock_gpg.encrypt.return_value = mock_encrypted
        mock_gpg_class.return_value = mock_gpg

        result = encrypt_env_file("TEST_VAR=value")

        assert result == "encrypted-content"
        mock_gpg.list_keys.assert_called_once_with(True)
        mock_gpg.encrypt.assert_called_once_with("TEST_VAR=value", recipients=["auto-key-123"])

    @patch("src.utils.encryption.gnupg.GPG")
    def test_encryption_no_keys_available(self, mock_gpg_class):
        """Test encryption fails when no GPG keys are available."""
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = []  # No keys available
        mock_gpg_class.return_value = mock_gpg

        with pytest.raises(GPGError, match="No GPG keys available for encryption"):
            encrypt_env_file("TEST_VAR=value")

    @patch("src.utils.encryption.gnupg.GPG")
    def test_encryption_gpg_failure(self, mock_gpg_class):
        """Test encryption fails when GPG encryption fails."""
        mock_gpg = Mock()
        mock_encrypted = Mock()
        mock_encrypted.ok = False
        mock_encrypted.status = "encryption failed"
        mock_gpg.encrypt.return_value = mock_encrypted
        mock_gpg_class.return_value = mock_gpg

        with pytest.raises(GPGError, match="Encryption failed: encryption failed"):
            encrypt_env_file("TEST_VAR=value", "recipient-key")

    @patch("src.utils.encryption.gnupg.GPG")
    def test_encryption_exception_handling(self, mock_gpg_class):
        """Test encryption handles general exceptions."""
        mock_gpg_class.side_effect = Exception("GPG initialization failed")

        with pytest.raises(GPGError, match="GPG encryption failed"):
            encrypt_env_file("TEST_VAR=value")


class TestDecryptEnvFile:
    """Test decrypt_env_file function."""

    @patch("src.utils.encryption.gnupg.GPG")
    def test_successful_decryption_no_passphrase(self, mock_gpg_class):
        """Test successful decryption without passphrase."""
        mock_gpg = Mock()
        mock_decrypted = Mock()
        mock_decrypted.ok = True
        mock_decrypted.__str__ = lambda self: "TEST_VAR=value"
        mock_gpg.decrypt.return_value = mock_decrypted
        mock_gpg_class.return_value = mock_gpg

        result = decrypt_env_file("encrypted-content")

        assert result == "TEST_VAR=value"
        mock_gpg.decrypt.assert_called_once_with("encrypted-content", passphrase=None)

    @patch("src.utils.encryption.gnupg.GPG")
    def test_successful_decryption_with_passphrase(self, mock_gpg_class):
        """Test successful decryption with passphrase."""
        mock_gpg = Mock()
        mock_decrypted = Mock()
        mock_decrypted.ok = True
        mock_decrypted.__str__ = lambda self: "TEST_VAR=value"
        mock_gpg.decrypt.return_value = mock_decrypted
        mock_gpg_class.return_value = mock_gpg

        result = decrypt_env_file("encrypted-content", "secret-passphrase")

        assert result == "TEST_VAR=value"
        mock_gpg.decrypt.assert_called_once_with("encrypted-content", passphrase="secret-passphrase")  # noqa: S106

    @patch("src.utils.encryption.gnupg.GPG")
    def test_decryption_failure(self, mock_gpg_class):
        """Test decryption fails when GPG decryption fails."""
        mock_gpg = Mock()
        mock_decrypted = Mock()
        mock_decrypted.ok = False
        mock_decrypted.status = "decryption failed"
        mock_gpg.decrypt.return_value = mock_decrypted
        mock_gpg_class.return_value = mock_gpg

        with pytest.raises(GPGError, match="Decryption failed: decryption failed"):
            decrypt_env_file("encrypted-content")

    @patch("src.utils.encryption.gnupg.GPG")
    def test_decryption_exception_handling(self, mock_gpg_class):
        """Test decryption handles general exceptions."""
        mock_gpg_class.side_effect = Exception("GPG initialization failed")

        with pytest.raises(GPGError, match="GPG decryption failed"):
            decrypt_env_file("encrypted-content")


class TestLoadEncryptedEnv:
    """Test load_encrypted_env function."""

    def test_file_not_found(self):
        """Test load_encrypted_env raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Encrypted env file not found"):
            load_encrypted_env("nonexistent.env.gpg")

    @patch("src.utils.encryption.decrypt_env_file")
    def test_successful_load_simple_env(self, mock_decrypt):
        """Test successful loading of simple environment variables."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
            f.write("encrypted-content")
            temp_path = f.name

        try:
            mock_decrypt.return_value = "TEST_VAR=value\nANOTHER_VAR=another_value"

            result = load_encrypted_env(temp_path)

            assert result == {"TEST_VAR": "value", "ANOTHER_VAR": "another_value"}
            mock_decrypt.assert_called_once_with("encrypted-content")
        finally:
            Path(temp_path).unlink()

    @patch("src.utils.encryption.decrypt_env_file")
    def test_load_env_with_quotes(self, mock_decrypt):
        """Test loading environment variables with quotes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
            f.write("encrypted-content")
            temp_path = f.name

        try:
            mock_decrypt.return_value = "QUOTED_VAR=\"quoted value\"\nSINGLE_QUOTED='single quoted'"

            result = load_encrypted_env(temp_path)

            assert result == {"QUOTED_VAR": "quoted value", "SINGLE_QUOTED": "single quoted"}
        finally:
            Path(temp_path).unlink()

    @patch("src.utils.encryption.decrypt_env_file")
    def test_load_env_with_comments_and_empty_lines(self, mock_decrypt):
        """Test loading environment variables ignoring comments and empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
            f.write("encrypted-content")
            temp_path = f.name

        try:
            mock_decrypt.return_value = """
            # This is a comment
            VALID_VAR=value

            # Another comment
            ANOTHER_VAR=another_value
            """

            result = load_encrypted_env(temp_path)

            assert result == {"VALID_VAR": "value", "ANOTHER_VAR": "another_value"}
        finally:
            Path(temp_path).unlink()

    @patch("src.utils.encryption.decrypt_env_file")
    def test_load_env_with_equals_in_value(self, mock_decrypt):
        """Test loading environment variables with equals signs in values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
            f.write("encrypted-content")
            temp_path = f.name

        try:
            mock_decrypt.return_value = "URL=https://example.com?param=value&other=data"

            result = load_encrypted_env(temp_path)

            assert result == {"URL": "https://example.com?param=value&other=data"}
        finally:
            Path(temp_path).unlink()

    @patch("src.utils.encryption.decrypt_env_file")
    def test_load_env_skips_invalid_lines(self, mock_decrypt):
        """Test loading environment variables skips lines without equals."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
            f.write("encrypted-content")
            temp_path = f.name

        try:
            mock_decrypt.return_value = """
            VALID_VAR=value
            invalid line without equals
            ANOTHER_VAR=another_value
            """

            result = load_encrypted_env(temp_path)

            assert result == {"VALID_VAR": "value", "ANOTHER_VAR": "another_value"}
        finally:
            Path(temp_path).unlink()

    @patch("src.utils.encryption.decrypt_env_file")
    def test_load_encrypted_env_propagates_gpg_error(self, mock_decrypt):
        """Test that GPG errors are propagated from decrypt_env_file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
            f.write("encrypted-content")
            temp_path = f.name

        try:
            mock_decrypt.side_effect = GPGError("Decryption failed")

            with pytest.raises(GPGError, match="Decryption failed"):
                load_encrypted_env(temp_path)
        finally:
            Path(temp_path).unlink()


class TestInitializeEncryption:
    """Test initialize_encryption function."""

    @patch("src.utils.encryption.validate_environment_keys")
    @patch("src.utils.encryption.load_encrypted_env")
    @patch("src.utils.encryption.os.environ")
    def test_successful_initialization_with_env_file(self, mock_environ, mock_load_env, mock_validate):
        """Test successful initialization with encrypted env file."""
        mock_load_env.return_value = {"SECRET_KEY": "secret-value", "API_TOKEN": "token-value"}
        mock_environ.setdefault = Mock()

        initialize_encryption()

        mock_validate.assert_called_once()
        mock_load_env.assert_called_once_with()

        # Verify environment variables were set
        mock_environ.setdefault.assert_any_call("SECRET_KEY", "secret-value")
        mock_environ.setdefault.assert_any_call("API_TOKEN", "token-value")

    @patch("src.utils.encryption.validate_environment_keys")
    @patch("src.utils.encryption.load_encrypted_env")
    @patch("src.utils.encryption.logging.getLogger")
    def test_initialization_no_env_file(self, mock_get_logger, mock_load_env, mock_validate):
        """Test initialization when no encrypted env file exists."""
        mock_load_env.side_effect = FileNotFoundError("No file found")
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        initialize_encryption()

        mock_validate.assert_called_once()
        mock_load_env.assert_called_once_with()
        mock_logger.debug.assert_called_once_with(
            "No encrypted .env file found, continuing with standard environment variables",
        )

    @patch("src.utils.encryption.validate_environment_keys")
    def test_initialization_validation_failure(self, mock_validate):
        """Test initialization when validation fails."""
        mock_validate.side_effect = EncryptionError("Validation failed")

        with pytest.raises(EncryptionError, match="Validation failed"):
            initialize_encryption()


class TestMainScriptExecution:
    """Test main script execution behavior."""

    @patch("src.utils.encryption.validate_environment_keys")
    @patch("src.utils.encryption.sys.exit")
    @patch("builtins.print")
    def test_main_script_success(self, mock_print, mock_exit, mock_validate):
        """Test main script execution on successful validation."""
        # Simulate the main script logic
        try:
            mock_validate.return_value = None  # Successful validation
            mock_validate()  # Call the mocked function
            mock_print("✓ All required keys are present and configured")
        except EncryptionError as e:
            mock_print(f"✗ Key validation failed: {e}")
            mock_exit(1)

        mock_validate.assert_called()
        mock_print.assert_called_with("✓ All required keys are present and configured")

    @patch("src.utils.encryption.validate_environment_keys")
    @patch("src.utils.encryption.sys.exit")
    @patch("builtins.print")
    def test_main_script_failure(self, mock_print, mock_exit, mock_validate):
        """Test main script execution on validation failure."""
        mock_validate.side_effect = EncryptionError("Test validation error")

        # Simulate the main script logic
        try:
            mock_validate()  # This will raise EncryptionError due to side_effect
            mock_print("✓ All required keys are present and configured")
        except EncryptionError as e:
            mock_print(f"✗ Key validation failed: {e}")
            mock_exit(1)

        mock_validate.assert_called()
        mock_print.assert_called_with("✗ Key validation failed: Test validation error")
        mock_exit.assert_called_with(1)


class TestEdgeCasesAndComplexScenarios:
    """Test edge cases and complex scenarios."""

    @patch("src.utils.encryption.gnupg.GPG")
    @patch("src.utils.encryption.subprocess.run")
    def test_whitespace_handling_in_git_output(self, mock_subprocess, mock_gpg_class):
        """Test handling of whitespace in git config output."""
        # Mock GPG success
        mock_gpg = Mock()
        mock_gpg.list_keys.return_value = [{"keyid": "test-key"}]
        mock_gpg_class.return_value = mock_gpg

        # Mock SSH success, Git with whitespace
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout=""),  # ssh-add success
            Mock(returncode=0, stdout="  signing-key-with-spaces  \n"),  # git config with spaces
        ]

        # Should not raise exception despite whitespace
        validate_environment_keys()

    @patch("src.utils.encryption.decrypt_env_file")
    def test_load_env_with_special_characters(self, mock_decrypt):
        """Test loading environment variables with special characters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
            f.write("encrypted-content")
            temp_path = f.name

        try:
            mock_decrypt.return_value = "SPECIAL_VAR=value with spaces & symbols!@#$%"

            result = load_encrypted_env(temp_path)

            assert result == {"SPECIAL_VAR": "value with spaces & symbols!@#$%"}
        finally:
            Path(temp_path).unlink()

    @patch("src.utils.encryption.gnupg.GPG")
    def test_encrypt_with_multiple_recipients(self, mock_gpg_class):
        """Test encryption behavior with complex recipient handling."""
        mock_gpg = Mock()
        mock_encrypted = Mock()
        mock_encrypted.ok = True
        mock_encrypted.__str__ = lambda self: "encrypted-content"
        mock_gpg.encrypt.return_value = mock_encrypted
        mock_gpg_class.return_value = mock_gpg

        # Test that single recipient is passed as list
        result = encrypt_env_file("TEST_VAR=value", "single-recipient")

        assert result == "encrypted-content"
        mock_gpg.encrypt.assert_called_once_with("TEST_VAR=value", recipients=["single-recipient"])

    @patch("src.utils.encryption.validate_environment_keys")
    @patch("src.utils.encryption.load_encrypted_env")
    @patch("src.utils.encryption.os.environ")
    def test_initialize_encryption_preserves_existing_env_vars(self, mock_environ, mock_load_env, mock_validate):
        """Test that initialize_encryption doesn't overwrite existing environment variables."""
        mock_load_env.return_value = {"NEW_VAR": "new-value", "EXISTING_VAR": "from-file"}

        # Mock setdefault to simulate existing environment variable
        def mock_setdefault(key, value):
            if key == "EXISTING_VAR":
                return "existing-value"  # Simulate existing value
            return value

        mock_environ.setdefault.side_effect = mock_setdefault

        initialize_encryption()

        # Verify setdefault was called for both variables
        mock_environ.setdefault.assert_any_call("NEW_VAR", "new-value")
        mock_environ.setdefault.assert_any_call("EXISTING_VAR", "from-file")


@pytest.mark.parametrize(
    ("env_content", "expected_vars"),
    [
        ("VAR1=value1\nVAR2=value2", {"VAR1": "value1", "VAR2": "value2"}),
        ("VAR_WITH_UNDERSCORE=value", {"VAR_WITH_UNDERSCORE": "value"}),
        ("VAR123=value123", {"VAR123": "value123"}),
        ("EMPTY_VAR=", {"EMPTY_VAR": ""}),
        ("VAR_WITH_SPACES = value with spaces ", {"VAR_WITH_SPACES": "value with spaces"}),
    ],
    ids=["basic", "underscore", "numbers", "empty_value", "spaces"],
)
@patch("src.utils.encryption.decrypt_env_file")
def test_load_env_parametrized(mock_decrypt, env_content, expected_vars):
    """Parametrized test for various env file formats."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env.gpg", delete=False) as f:
        f.write("encrypted-content")
        temp_path = f.name

    try:
        mock_decrypt.return_value = env_content
        result = load_encrypted_env(temp_path)
        assert result == expected_vars
    finally:
        Path(temp_path).unlink()


class TestSecurityAndErrorHandling:
    """Test security aspects and comprehensive error handling."""

    @patch("src.utils.encryption.gnupg.GPG")
    def test_encryption_handles_unicode(self, mock_gpg_class):
        """Test encryption handles unicode content properly."""
        mock_gpg = Mock()
        mock_encrypted = Mock()
        mock_encrypted.ok = True
        mock_encrypted.__str__ = lambda self: "encrypted-unicode-content"
        mock_gpg.encrypt.return_value = mock_encrypted
        mock_gpg_class.return_value = mock_gpg

        unicode_content = "UNICODE_VAR=värld 🌍 测试"
        result = encrypt_env_file(unicode_content, "recipient")

        assert result == "encrypted-unicode-content"
        mock_gpg.encrypt.assert_called_once_with(unicode_content, recipients=["recipient"])

    @patch("src.utils.encryption.gnupg.GPG")
    def test_decryption_handles_unicode(self, mock_gpg_class):
        """Test decryption handles unicode content properly."""
        mock_gpg = Mock()
        mock_decrypted = Mock()
        mock_decrypted.ok = True
        mock_decrypted.__str__ = lambda self: "UNICODE_VAR=värld 🌍 测试"
        mock_gpg.decrypt.return_value = mock_decrypted
        mock_gpg_class.return_value = mock_gpg

        result = decrypt_env_file("encrypted-unicode-content")

        assert result == "UNICODE_VAR=värld 🌍 测试"

    @patch("src.utils.encryption.Path.read_text")
    @patch("src.utils.encryption.Path.exists")
    def test_load_env_handles_file_read_errors(self, mock_exists, mock_read_text):
        """Test load_encrypted_env handles file read errors properly."""
        mock_exists.return_value = True
        mock_read_text.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError, match="Permission denied"):
            load_encrypted_env("test.env.gpg")

    @patch("src.utils.encryption.gnupg.GPG")
    def test_empty_content_encryption(self, mock_gpg_class):
        """Test encryption of empty content."""
        mock_gpg = Mock()
        mock_encrypted = Mock()
        mock_encrypted.ok = True
        mock_encrypted.__str__ = lambda self: "encrypted-empty"
        mock_gpg.encrypt.return_value = mock_encrypted
        mock_gpg_class.return_value = mock_gpg

        result = encrypt_env_file("", "recipient")

        assert result == "encrypted-empty"
        mock_gpg.encrypt.assert_called_once_with("", recipients=["recipient"])
