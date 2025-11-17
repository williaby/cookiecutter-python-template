#!/usr/bin/env python3
"""Pre-generation hook for cookiecutter Python template.

Validates user inputs before project generation begins.
Runs before any files are created.
"""

import re
import sys


def validate_project_slug(project_slug: str) -> bool:
    """Validate that project_slug follows Python package naming conventions.

    Args:
        project_slug: The project slug to validate

    Returns:
        True if valid, False otherwise
    """
    # Must be valid Python package name: lowercase, underscores, no hyphens
    pattern = r"^[a-z][a-z0-9_]*$"
    return bool(re.match(pattern, project_slug))


def validate_email(email: str) -> bool:
    """Validate email address format.

    Args:
        email: The email address to validate

    Returns:
        True if valid, False otherwise
    """
    # Basic email validation
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_github_username(username: str) -> bool:
    """Validate GitHub username format.

    Args:
        username: The GitHub username to validate

    Returns:
        True if valid, False otherwise
    """
    # GitHub username rules: alphanumeric and hyphens, no consecutive hyphens
    pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"
    return bool(re.match(pattern, username))


def main() -> None:
    """Run pre-generation validation checks."""
    # Get cookiecutter variables
    project_slug = "{{ cookiecutter.project_slug }}"
    author_email = "{{ cookiecutter.author_email }}"
    github_username = "{{ cookiecutter.github_username }}"
    python_version = "{{ cookiecutter.python_version }}"

    errors = []

    # Validate project_slug
    if not validate_project_slug(project_slug):
        errors.append(
            f"ERROR: '{project_slug}' is not a valid Python package name.\n"
            "       Must be lowercase with underscores (e.g., 'my_project')."
        )

    # Validate email
    if not validate_email(author_email):
        errors.append(
            f"ERROR: '{author_email}' is not a valid email address.\n"
            "       Format: user@example.com"
        )

    # Validate GitHub username
    if not validate_github_username(github_username):
        errors.append(
            f"ERROR: '{github_username}' is not a valid GitHub username.\n"
            "       Must be alphanumeric with hyphens (no consecutive hyphens)."
        )

    # Validate Python version
    valid_versions = ["3.11", "3.12", "3.13"]
    if python_version not in valid_versions:
        errors.append(
            f"ERROR: '{python_version}' is not a supported Python version.\n"
            f"       Supported versions: {', '.join(valid_versions)}"
        )

    # Print all errors and exit if any found
    if errors:
        print("\n" + "=" * 60)
        print("COOKIECUTTER PRE-GENERATION VALIDATION FAILED")
        print("=" * 60 + "\n")
        for error in errors:
            print(error)
        print("\n" + "=" * 60)
        sys.exit(1)

    print("✓ Pre-generation validation passed")


if __name__ == "__main__":
    main()
