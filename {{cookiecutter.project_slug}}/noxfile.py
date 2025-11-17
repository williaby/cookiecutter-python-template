{%- if cookiecutter.include_nox == "yes" %}
"""Nox-UV sessions for local testing and documentation workflows.

Nox-UV uses UV for fast virtual environment creation and package installation.
This file defines sessions for documentation validation, building, and serving,
as well as multi-Python version testing.

Usage:
    nox -s fm          # Validate and autofix front matter
    nox -s docs        # Build documentation
    nox -s serve       # Serve documentation locally
    nox -s docstrings  # Check docstring coverage
    nox -s validate    # Run all validation checks
    nox -s reuse       # Check REUSE compliance
    nox -s sbom        # Generate SBOM
    nox -s scan        # Scan SBOM for vulnerabilities
    nox -s compliance  # Run all compliance checks
    nox -s assuredoss  # Validate Google Assured OSS credentials
    nox -s test        # Run tests across multiple Python versions
    nox -s lint        # Run Ruff linting across multiple Python versions
    nox -s typecheck   # Run MyPy type checking across multiple Python versions
"""

import nox
import nox_uv

# Use nox-uv for faster environment creation
nox_uv.register()

# Default sessions and options
nox.options.sessions = ["test", "lint", "docs"]
nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "uv"


@nox.session(python="{{cookiecutter.python_version}}")
def fm(session: nox.Session) -> None:
    """Validate and autofix front matter in documentation.

    This session installs the required dependencies and runs the front matter
    validation script with autofix enabled.
    """
    session.install("pydantic>=2.0", "python-frontmatter>=1.1", "ruamel.yaml>=0.18")
    session.run("python", "tools/validate_front_matter.py", "docs", "--fix")


@nox.session(python="{{cookiecutter.python_version}}")
def docs(session: nox.Session) -> None:
    """Build documentation with MkDocs.

    This session installs the project with docs dependencies and builds
    the documentation in strict mode.
    """
    session.install("-e", ".[dev]")
    session.run("mkdocs", "build", "--strict")


@nox.session(python="{{cookiecutter.python_version}}")
def serve(session: nox.Session) -> None:
    """Serve documentation locally for development.

    This session starts the MkDocs development server with live reloading.
    Access at http://127.0.0.1:8000
    """
    session.install("-e", ".[dev]")
    session.run("mkdocs", "serve")


@nox.session(python="{{cookiecutter.python_version}}")
def docstrings(session: nox.Session) -> None:
    """Check docstring coverage with interrogate and pydocstyle.

    This session validates that docstrings meet the Google style convention
    and that coverage meets the minimum threshold.
    """
    session.install("pydocstyle>=6.3", "interrogate>=1.7")
    session.run("pydocstyle", "src/")
    session.run("interrogate", "-c", "pyproject.toml", "src/")


@nox.session(python="{{cookiecutter.python_version}}")
def validate(session: nox.Session) -> None:
    """Run all validation checks for documentation.

    This session combines front matter validation, docstring checks,
    and documentation building to ensure everything is correct.
    """
    session.install(
        "-e",
        ".[dev]",
        "pydantic>=2.0",
        "python-frontmatter>=1.1",
        "ruamel.yaml>=0.18",
        "pydocstyle>=6.3",
        "interrogate>=1.7",
    )
    session.run("python", "tools/validate_front_matter.py", "docs", "--fix")
    session.run("pydocstyle", "src/")
    session.run("interrogate", "-c", "pyproject.toml", "src/")
    session.run("mkdocs", "build", "--strict")


@nox.session(python="{{cookiecutter.python_version}}")
def reuse(session: nox.Session) -> None:
    """Check REUSE compliance.

    This session uses the REUSE tool to verify that all files have proper
    licensing information according to the REUSE specification.
    Requires Docker to be installed and running.
    """
    session.run(
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{session.posargs[0] if session.posargs else '.'}:/data",
        "fsfe/reuse:latest",
        "lint",
        external=True,
    )


@nox.session(python="{{cookiecutter.python_version}}")
def reuse_spdx(session: nox.Session) -> None:
    """Generate REUSE SPDX document.

    This session generates an SPDX document from the REUSE metadata.
    The SPDX file is saved to reuse-spdx.json in the current directory.
    Requires Docker to be installed and running.
    """
    session.run(
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{session.posargs[0] if session.posargs else '.'}:/data",
        "fsfe/reuse:latest",
        "spdx",
        "--output",
        "/data/reuse-spdx.json",
        external=True,
    )
    session.log("SPDX document generated: reuse-spdx.json")


@nox.session(python="{{cookiecutter.python_version}}")
def sbom(session: nox.Session) -> None:
    """Generate CycloneDX SBOM using UV.

    This session generates Software Bill of Materials (SBOM) in CycloneDX format
    for runtime and development dependency sets using UV's pip-compatible interface.
    """
    session.install("cyclonedx-bom==4.6.1")

    # Generate runtime SBOM (production dependencies only)
    # Export production dependencies to requirements.txt format
    session.run(
        "uv",
        "pip",
        "compile",
        "pyproject.toml",
        "--output-file",
        "requirements-runtime.txt",
        "--no-dev",
        external=True,
    )
    session.run(
        "cyclonedx-py",
        "requirements",
        "requirements-runtime.txt",
        "--of",
        "json",
        "-o",
        "sbom-runtime.json",
    )
    session.log("Runtime SBOM generated: sbom-runtime.json")

    # Generate development SBOM (all dependencies including dev)
    session.run(
        "uv",
        "export",
        "--format",
        "requirements-txt",
        "--output-file",
        "requirements-all.txt",
        "--no-hashes",
        external=True,
    )
    session.run(
        "cyclonedx-py",
        "requirements",
        "requirements-all.txt",
        "--of",
        "json",
        "-o",
        "sbom-complete.json",
    )
    session.log("Complete SBOM generated: sbom-complete.json")

    # Clean up temporary files
    import pathlib
    pathlib.Path("requirements-runtime.txt").unlink(missing_ok=True)
    pathlib.Path("requirements-all.txt").unlink(missing_ok=True)


@nox.session(python="{{cookiecutter.python_version}}")
def scan(session: nox.Session) -> None:
    """Scan SBOM for vulnerabilities.

    This session uses Trivy to scan the generated SBOMs for known vulnerabilities.
    Requires Docker to be installed and running.
    Requires SBOM files to be generated first (run 'nox -s sbom').
    """
    import pathlib

    sbom_file = session.posargs[0] if session.posargs else "sbom-runtime.json"

    if not pathlib.Path(sbom_file).exists():
        session.error(f"SBOM file not found: {sbom_file}. Run 'nox -s sbom' first.")

    session.run(
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{pathlib.Path().absolute()}:/workspace",
        "aquasec/trivy:latest",
        "sbom",
        f"/workspace/{sbom_file}",
        "--severity",
        "CRITICAL,HIGH",
        "--format",
        "table",
        external=True,
    )


@nox.session(python="{{cookiecutter.python_version}}")
def compliance(session: nox.Session) -> None:
    """Run all compliance checks.

    This session runs REUSE compliance checks and generates SBOMs
    for comprehensive compliance validation.
    """
    session.log("Running REUSE compliance check...")
    reuse(session)

    session.log("Generating SBOMs...")
    sbom(session)

    session.log("Scanning runtime SBOM for vulnerabilities...")
    scan(session)

    session.log("All compliance checks completed successfully!")


@nox.session(python="{{cookiecutter.python_version}}")
def assuredoss(session: nox.Session) -> None:
    """Validate Google Assured OSS credentials and configuration.

    This session validates that:
    - Google Cloud credentials are properly configured
    - Project ID is set correctly
    - Assured OSS is accessible
    - Package listing works

    Requires .env file with:
    - GOOGLE_CLOUD_PROJECT
    - GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS_B64
    """
    session.install("-e", ".[dev]")
    session.run("python", "scripts/validate_assuredoss.py")


@nox.session(python=["3.11", "3.12", "3.13"])
def test(session: nox.Session) -> None:
    """Run tests across multiple Python versions.

    This session runs the full test suite with coverage reporting
    across Python 3.11, 3.12, and 3.13 to ensure compatibility.
    """
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "-v",
        "--cov=src",
        "--cov-report=xml",
        "--cov-report=term-missing:skip-covered",
        "--cov-fail-under={{cookiecutter.code_coverage_target}}",
        "tests/",
    )


@nox.session(python=["3.11", "3.12", "3.13"])
def lint(session: nox.Session) -> None:
    """Run linting across multiple Python versions.

    This session runs Ruff linting to ensure code quality
    across all supported Python versions.
    """
    session.install("-e", ".[dev]")
    session.run("ruff", "check", ".", "--config=pyproject.toml")
    session.run("ruff", "format", "--check")


@nox.session(python=["3.11", "3.12", "3.13"])
def typecheck(session: nox.Session) -> None:
    """Run type checking across multiple Python versions.

    This session runs MyPy type checking to ensure type safety
    across all supported Python versions.
    """
    session.install("-e", ".[dev]")
    {% if cookiecutter.use_mypy == "yes" %}
    session.run("mypy", "src", "--config-file=pyproject.toml")
    {% else %}
    session.log("MyPy type checking is disabled in this project")
    {% endif %}
{%- else %}
"""Nox sessions - NOT CONFIGURED

To enable nox sessions, set include_nox to "yes" in cookiecutter.json

Common nox sessions include:
    - fm: Validate and autofix front matter
    - docs: Build documentation
    - serve: Serve documentation locally
    - docstrings: Check docstring coverage
    - reuse: Check REUSE compliance
    - sbom: Generate SBOM
    - scan: Scan SBOM for vulnerabilities
    - compliance: Run all compliance checks
"""
{%- endif %}
