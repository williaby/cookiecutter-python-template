# {{cookiecutter.project_name}}

{%- if cookiecutter.include_security_scanning == "yes" %}

## Status & Quality

[![Python {{cookiecutter.python_version}}](https://img.shields.io/badge/python-{{cookiecutter.python_version}}-blue.svg)](https://www.python.org/downloads/)
[![License: {{cookiecutter.license}}](https://img.shields.io/badge/License-{{cookiecutter.license}}-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
{%- if cookiecutter.include_codecov == "yes" %}
[![codecov](https://codecov.io/gh/{{cookiecutter.github_org_or_user}}/{{cookiecutter.project_slug}}/graph/badge.svg)](https://codecov.io/gh/{{cookiecutter.github_org_or_user}}/{{cookiecutter.project_slug}})
{%- endif %}
{%- if cookiecutter.include_github_actions == "yes" %}

[![CI Pipeline](https://github.com/{{cookiecutter.github_org_or_user}}/{{cookiecutter.project_slug}}/actions/workflows/ci.yml/badge.svg)](https://github.com/{{cookiecutter.github_org_or_user}}/{{cookiecutter.project_slug}}/actions/workflows/ci.yml)
[![Security Analysis](https://github.com/{{cookiecutter.github_org_or_user}}/{{cookiecutter.project_slug}}/actions/workflows/security-analysis.yml/badge.svg)](https://github.com/{{cookiecutter.github_org_or_user}}/{{cookiecutter.project_slug}}/actions/workflows/security-analysis.yml)
{%- endif %}
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/williaby/.github/blob/main/CODE_OF_CONDUCT.md)

{%- endif %}

---

## Overview

{{cookiecutter.project_short_description}}

This project provides:
- Core functionality for {{cookiecutter.project_short_description.lower()}}
- Production-ready code with comprehensive testing
- Well-documented API and architecture
- Security-first development practices

## Features

- **High Quality**: {{cookiecutter.code_coverage_target}}%+ test coverage enforced via CI
- **Type Safe**: Full type hints with MyPy strict mode
- **Well Documented**: Clear docstrings and comprehensive guides
- **Developer Friendly**: Pre-commit hooks, automated formatting, linting
- **Security First**: Dependency scanning, security analysis, SBOM generation
{%- if cookiecutter.include_cli == "yes" %}
- **CLI Tool**: Command-line interface via {{cookiecutter.cli_tool_name}}
{%- endif %}
{%- if cookiecutter.include_ml_dependencies == "yes" %}
- **ML Ready**: Optional ML dependencies with PyTorch support
{%- endif %}

## Quick Start

### Prerequisites

- Python {{cookiecutter.python_version}}+
- UV for dependency management

### Installation

```bash
# Clone repository
git clone {{cookiecutter.repo_url}}.git
cd {{cookiecutter.project_slug}}

# Install dependencies
uv sync

# Install with dev tools (recommended)
uv sync --all-extras

{%- if cookiecutter.include_ml_dependencies == "yes" %}
# Install with ML dependencies
uv sync --all-extras,ml
{%- endif %}

# Setup pre-commit hooks (required)
uv run pre-commit install
```

### Basic Usage

```python
# Import and use the package
from {{cookiecutter.project_slug}} import YourModule

# Example: Create an instance and use it
module = YourModule()
result = module.process()
print(result)
```

{%- if cookiecutter.include_cli == "yes" %}

### CLI Usage

```bash
# Display help
uv run {{cookiecutter.cli_tool_name}} --help

# Use the CLI tool
uv run {{cookiecutter.cli_tool_name}} command --option value

# Example: Process input file
uv run {{cookiecutter.cli_tool_name}} process input.txt --output result.json
```

{%- endif %}

## Google Assured OSS Integration

This project uses **Google Assured OSS** as the primary package source, with PyPI as a fallback. Assured OSS provides vetted, secure open-source packages with Google's security guarantees.

### Why Assured OSS?

- **Security**: All packages are scanned and verified by Google
- **Supply Chain Protection**: Reduced risk of malicious packages
- **Compliance**: Meets enterprise security requirements
- **Automatic Fallback**: Seamlessly falls back to PyPI when needed

### Setup Instructions

1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Configure Google Cloud Project**:
   ```bash
   # Edit .env and set your GCP project ID
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   ```

3. **Setup Authentication** (choose one method):

   **Option A: Service Account JSON File** (local development)
   ```bash
   # Download service account key from GCP Console
   # Set the file path in .env
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

   **Option B: Base64 Encoded Credentials** (CI/CD recommended)
   ```bash
   # Encode your service account JSON
   base64 -w 0 service-account-key.json

   # Set the base64 string in .env
   GOOGLE_APPLICATION_CREDENTIALS_B64=<paste-base64-here>
   ```

4. **Validate Configuration**:
   ```bash
   # Run the validation script
   uv run python scripts/validate_assuredoss.py

   # Or use nox
   nox -s assuredoss
   ```

### Service Account Permissions

Your service account needs the following IAM role:
- `roles/artifactregistry.reader` (Artifact Registry Reader)

### Disabling Assured OSS

To use only PyPI (not recommended for production):

```bash
# In .env file
USE_ASSURED_OSS=false
```

### Troubleshooting

**Q: Packages not found in Assured OSS?**
- UV automatically falls back to PyPI for packages not in Assured OSS
- No action needed - this is expected behavior

**Q: Authentication errors?**
- Verify your service account has Artifact Registry Reader role
- Check that GOOGLE_CLOUD_PROJECT is set correctly
- Ensure credentials file/base64 is valid JSON

**Q: How to see which packages are available?**
- Run `nox -s assuredoss` to list all available packages
- Visit: https://cloud.google.com/assured-open-source-software/docs/supported-packages

## Development

### Setup Development Environment

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Setup pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest -v

# Run with coverage
uv run pytest --cov={{cookiecutter.project_slug}} --cov-report=html

# Run all quality checks
uv run pre-commit run --all-files
```

### Code Quality Standards

All code must meet these requirements:

- **Formatting**: Ruff ({{cookiecutter.code_coverage_target}} char limit)
- **Linting**: Ruff with comprehensive rules
- **Type Checking**: MyPy strict mode
- **Testing**: Pytest with {{cookiecutter.code_coverage_target}}%+ coverage
- **Security**: Bandit + dependency scanning
- **Documentation**: Docstrings on all public APIs

### Running Tests

```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_module.py -v

# Run with coverage report
uv run pytest --cov={{cookiecutter.project_slug}} --cov-report=term-missing

# Run tests in parallel
uv run pytest -n auto
```

### Quality Checks

```bash
# Format code
uv run ruff format src tests

# Lint and auto-fix
uv run ruff check --fix src tests

# Type checking
uv run mypy src

# Security scanning
uv run bandit -r src

# Check dependencies
uv run safety check
```

## Project Structure

```
{{cookiecutter.project_slug}}/
├── src/{{cookiecutter.project_slug}}/     # Main package
│   ├── __init__.py
│   ├── core.py                           # Core functionality
│   └── utils/                            # Utility modules
├── tests/                                # Test suite
│   ├── unit/                             # Unit tests
│   └── integration/                      # Integration tests
├── docs/                                 # Documentation
│   ├── ADRs/                             # Architecture Decision Records
│   ├── planning/                         # Project planning docs
│   └── guides/                           # User guides
├── pyproject.toml                        # Dependencies & tool config
├── README.md                             # This file
├── CONTRIBUTING.md                       # Contribution guidelines
└── LICENSE                               # License
```

## Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)**: How to contribute to the project
- **[docs/ADRs/README.md](docs/ADRs/README.md)**: Architecture Decision Records documentation
- **[docs/planning/project-plan-template.md](docs/planning/project-plan-template.md)**: Project planning guide

### Writing Documentation

- Use Markdown for all documentation
- Include code examples for clarity
- Update README.md when adding major features
- Maintain architecture documentation (see [docs/ADRs/](docs/ADRs/))

## Testing

### Testing Policy

All new functionality must include tests:

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Coverage**: Maintain {{cookiecutter.code_coverage_target}}%+ coverage
- **Markers**: Use pytest markers (`@pytest.mark.unit`, `@pytest.mark.integration`)

### Test Guidelines

```bash
# Run all tests
uv run pytest -v

# Run only unit tests
uv run pytest -v -m unit

# Run only integration tests
uv run pytest -v -m integration

# Run with coverage requirements
uv run pytest --cov={{cookiecutter.project_slug}} --cov-fail-under={{cookiecutter.code_coverage_target}}
```

## Security

### Security-First Development

- Validate all inputs
- Use secure defaults
- Scan dependencies regularly
- Report vulnerabilities responsibly

### Reporting Security Issues

Please report security vulnerabilities to {{cookiecutter.author_email}} rather than using the public issue tracker.

See the [williaby Security Policy](https://github.com/williaby/.github/blob/main/SECURITY.md) for complete disclosure policy and response timelines.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code quality standards
- Testing requirements
- Git workflow and commit conventions
- Pull request process

### Quick Checklist Before Submitting PR

- [ ] Code follows style guide (Ruff format + lint)
- [ ] All tests pass with {{cookiecutter.code_coverage_target}}%+ coverage
- [ ] MyPy type checking passes
- [ ] Docstrings added for new public APIs
- [ ] CHANGELOG.md updated (if significant change)
- [ ] Commits follow conventional commit format

## Versioning

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backwards-compatible functionality additions
- **PATCH** version: Backwards-compatible bug fixes

Current version: **{{cookiecutter.version}}**

## License

{{cookiecutter.license}} License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues]({{cookiecutter.repo_url}}/issues)
{%- if cookiecutter.include_github_actions == "yes" %}
- **Discussions**: [GitHub Discussions]({{cookiecutter.repo_url}}/discussions)
{%- endif %}
- **Email**: {{cookiecutter.author_email}}

## Acknowledgments

Thank you to all contributors and the open-source community!

---

**Made with by [{{cookiecutter.author_name}}](https://github.com/{{cookiecutter.github_username}})**
