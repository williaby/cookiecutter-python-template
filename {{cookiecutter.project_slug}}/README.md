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
{%- if cookiecutter.include_code_of_conduct == "yes" %}
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
{%- endif %}

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
- Poetry 1.7+ for dependency management

### Installation

```bash
# Clone repository
git clone {{cookiecutter.repo_url}}.git
cd {{cookiecutter.project_slug}}

# Install dependencies
poetry install

# Install with dev tools (recommended)
poetry install --with dev

{%- if cookiecutter.include_ml_dependencies == "yes" %}
# Install with ML dependencies
poetry install --with dev,ml
{%- endif %}

# Setup pre-commit hooks (required)
poetry run pre-commit install
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
poetry run {{cookiecutter.cli_tool_name}} --help

# Use the CLI tool
poetry run {{cookiecutter.cli_tool_name}} command --option value

# Example: Process input file
poetry run {{cookiecutter.cli_tool_name}} process input.txt --output result.json
```

{%- endif %}

## Development

### Setup Development Environment

```bash
# Install all dependencies including dev tools
poetry install --with dev

# Setup pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov={{cookiecutter.project_slug}} --cov-report=html

# Run all quality checks
poetry run pre-commit run --all-files
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
poetry run pytest -v

# Run specific test file
poetry run pytest tests/unit/test_module.py -v

# Run with coverage report
poetry run pytest --cov={{cookiecutter.project_slug}} --cov-report=term-missing

# Run tests in parallel
poetry run pytest -n auto
```

### Quality Checks

```bash
# Format code
poetry run ruff format src tests

# Lint and auto-fix
poetry run ruff check --fix src tests

# Type checking
poetry run mypy src

# Security scanning
poetry run bandit -r src

# Check dependencies
poetry run safety check
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
poetry run pytest -v

# Run only unit tests
poetry run pytest -v -m unit

# Run only integration tests
poetry run pytest -v -m integration

# Run with coverage requirements
poetry run pytest --cov={{cookiecutter.project_slug}} --cov-fail-under={{cookiecutter.code_coverage_target}}
```

## Security

### Security-First Development

- Validate all inputs
- Use secure defaults
- Scan dependencies regularly
- Report vulnerabilities responsibly

### Reporting Security Issues

Please report security vulnerabilities to {{cookiecutter.author_email}} rather than using the public issue tracker.

{%- if cookiecutter.include_security_policy == "yes" %}
See [SECURITY.md](SECURITY.md) for complete disclosure policy.
{%- endif %}

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
