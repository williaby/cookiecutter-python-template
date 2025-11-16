"""
Encryption utilities for PromptCraft-Hybrid.

This module provides encryption/decryption capabilities for sensitive data,
following the pattern established in ledgerbase for secure .env file handling.
"""

import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import gnupg


class EncryptionError(Exception):
    """Raised when encryption/decryption operations fail."""


class GPGError(Exception):
    """Raised when GPG operations fail."""


def validate_environment_keys() -> None:
    """
    Validate that required GPG and SSH keys are present.

    Raises:
        EncryptionError: If required keys are missing or not configured.
    """
    # Validate GPG key
    try:
        gpg = gnupg.GPG()
        secret_keys = gpg.list_keys(True)
        if not secret_keys:
            raise EncryptionError(
                "No GPG secret keys found. GPG key required for .env encryption.",
            )
    except Exception as e:
        raise EncryptionError(f"Failed to access GPG keys: {e}") from e

    # Validate SSH key
    try:
        result = subprocess.run(  # nosec B603, B607, S607
            ["ssh-add", "-l"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise EncryptionError(
                "No SSH keys loaded. SSH key required for signed commits.",
            )
    except FileNotFoundError as e:
        raise EncryptionError("ssh-add command not found. SSH tools required.") from e

    # Validate Git signing configuration
    try:
        result = subprocess.run(  # nosec B603, B607, S607
            ["git", "config", "--get", "user.signingkey"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise EncryptionError(
                "Git signing key not configured. Required for signed commits.",
            )
    except FileNotFoundError as e:
        raise EncryptionError("Git command not found.") from e


def encrypt_env_file(content: str, recipient: str | None = None) -> str:
    """
    Encrypt .env file content using GPG.

    Args:
        content: The .env file content to encrypt
        recipient: GPG key identifier (if None, uses default key)

    Returns:
        Encrypted content as string

    Raises:
        GPGError: If encryption fails
    """
    try:
        gpg = gnupg.GPG()

        if recipient is None:
            # Use first available secret key
            secret_keys = gpg.list_keys(True)
            if not secret_keys:
                raise GPGError("No GPG keys available for encryption")
            recipient = secret_keys[0]["keyid"]

        encrypted_data = gpg.encrypt(content, recipients=[recipient])

        if not encrypted_data.ok:
            raise GPGError(f"Encryption failed: {encrypted_data.status}")

        return str(encrypted_data)

    except Exception as e:
        raise GPGError(f"GPG encryption failed: {e}") from e


def decrypt_env_file(encrypted_content: str, passphrase: str | None = None) -> str:
    """
    Decrypt .env file content using GPG.

    Args:
        encrypted_content: The encrypted content to decrypt
        passphrase: GPG key passphrase (if None, assumes agent or no passphrase)

    Returns:
        Decrypted content as string

    Raises:
        GPGError: If decryption fails
    """
    try:
        gpg = gnupg.GPG()
        decrypted_data = gpg.decrypt(encrypted_content, passphrase=passphrase)

        if not decrypted_data.ok:
            raise GPGError(f"Decryption failed: {decrypted_data.status}")

        return str(decrypted_data)

    except Exception as e:
        raise GPGError(f"GPG decryption failed: {e}") from e


def load_encrypted_env(env_file_path: str = ".env.gpg") -> dict[str, str]:
    """
    Load and decrypt environment variables from encrypted file.

    Args:
        env_file_path: Path to encrypted .env file

    Returns:
        Dictionary of environment variables

    Raises:
        FileNotFoundError: If env file doesn't exist
        GPGError: If decryption fails
    """
    env_path = Path(env_file_path)

    if not env_path.exists():
        raise FileNotFoundError(f"Encrypted env file not found: {env_file_path}")

    encrypted_content = env_path.read_text()

    decrypted_content = decrypt_env_file(encrypted_content)

    # Parse .env format
    env_vars = {}
    for orig_line in decrypted_content.strip().split("\n"):
        line = orig_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env_vars[key.strip()] = value.strip().strip("\"'")

    return env_vars


def initialize_encryption() -> None:
    """
    Initialize encryption system and validate environment.

    This should be called during application startup to ensure
    all required keys are present and properly configured.
    """
    validate_environment_keys()

    # Optionally load encrypted environment variables
    try:
        env_vars = load_encrypted_env()
        for key, value in env_vars.items():
            os.environ.setdefault(key, value)
    except FileNotFoundError:
        # No encrypted env file found, which is okay for development
        # Log this as debug info to maintain visibility without cluttering logs
        logging.getLogger(__name__).debug(
            "No encrypted .env file found, continuing with standard environment variables",
        )


if __name__ == "__main__":
    # Quick validation script
    try:
        validate_environment_keys()
    except EncryptionError:
        sys.exit(1)
