# Contributing to {{cookiecutter.project_name}}

Thank you for your interest in contributing to {{cookiecutter.project_name}}! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Commit Convention](#commit-convention)
- [Pull Request Process](#pull-request-process)
- [Questions?](#questions)

## Code of Conduct

This project adheres to the [williaby organization Code of Conduct](https://github.com/williaby/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to {{cookiecutter.author_email}}.

## Getting Started

### Prerequisites

- Python {{cookiecutter.python_version}} or higher
- UV 1.7+ for dependency management
- Git

{%- if cookiecutter.use_pre_commit == "yes" %}
- GPG key configured for commit signing (recommended)
{%- endif %}

### Development Environment Setup

```bash
# Clone the repository
git clone {{cookiecutter.repo_url}}.git
cd {{cookiecutter.project_slug}}

# Install dependencies with UV
uv sync --all-extras

{%- if cookiecutter.include_ml_dependencies == "yes" %}
# Install ML dependencies (if needed)
uv sync --all-extras,ml
{%- endif %}

{%- if cookiecutter.use_pre_commit == "yes" %}
# Setup pre-commit hooks (REQUIRED)
uv run pre-commit install

# Verify installation
uv run pytest -v
{%- if cookiecutter.use_ruff == "yes" %}
uv run ruff check src tests
{%- endif %}
{%- if cookiecutter.use_mypy == "yes" %}
uv run mypy src
{%- endif %}
{%- endif %}
```

### Project Structure

```
{{cookiecutter.project_slug}}/
├── src/{{cookiecutter.project_slug}}/       # Main package
│   ├── __init__.py
│   ├── core.py                           # Core functionality
│   └── utils/                            # Utility modules
├── tests/                                # Test suite
│   ├── unit/                             # Unit tests
│   └── integration/                      # Integration tests
├── docs/                                 # Documentation
│   ├── ADRs/                             # Architecture Decision Records
│   ├── planning/                         # Project planning
│   └── guides/                           # User guides
└── pyproject.toml                        # Dependencies & config
```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Create and checkout a new branch
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/issue-description

# For documentation
git checkout -b docs/documentation-update
```

### Branch Naming Convention

- `feature/` - New features or enhancements
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or improvements
- `chore/` - Maintenance tasks

### 2. Make Your Changes

- Write clean, readable code following [PEP 8](https://peps.python.org/pep-0008/)
- Add type hints to all functions and methods
- Update documentation for API changes
- Add tests for new functionality
- Ensure all tests pass locally

### 3. Run Quality Checks

Before committing, ensure all quality checks pass:

```bash
{%- if cookiecutter.use_ruff == "yes" %}
# Format code
uv run ruff format src tests

# Lint code
uv run ruff check --fix src tests
{%- endif %}

{%- if cookiecutter.use_mypy == "yes" %}
# Type checking
uv run mypy src
{%- endif %}

# Run tests with coverage
uv run pytest --cov={{cookiecutter.project_slug}} --cov-report=term-missing

{%- if cookiecutter.use_pre_commit == "yes" %}
# Run all pre-commit hooks manually
uv run pre-commit run --all-files
{%- endif %}
```

## Code Quality Standards

All contributions MUST meet these requirements:

### Formatting

{%- if cookiecutter.use_ruff == "yes" %}
- **Tool**: Ruff
- **Line Length**: 88 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Sorted with Ruff
- **Quotes**: Double quotes for strings
- **Verification**: `uv run ruff format src tests`
{%- else %}
- **Tool**: Black
- **Line Length**: 88 characters
- **Indentation**: 4 spaces (no tabs)
- **Verification**: `uv run black src tests`
{%- endif %}

### Linting

{%- if cookiecutter.use_ruff == "yes" %}
- **Tool**: Ruff with project configuration
- **Rules**: Comprehensive rule set (see `pyproject.toml`)
- **Auto-fix**: `uv run ruff check --fix src tests`
- **Verification**: `uv run ruff check src tests`
{%- else %}
- **Tool**: Pylint or similar
- **Verification**: Run configured linter
{%- endif %}

### Type Checking

{%- if cookiecutter.use_mypy == "yes" %}
- **Tool**: MyPy strict mode for `src/`
- **Coverage**: All public functions must have type hints
- **Verification**: `uv run mypy src`
{%- else %}
- **Tool**: Type hints recommended
- **Coverage**: Add type hints for clarity
{%- endif %}

### Security

- **No Hardcoded Secrets**: Use environment variables or secure vaults
- **Input Validation**: Validate all user inputs and file paths
- **Path Sanitization**: Use `pathlib.Path.resolve()` to prevent directory traversal
- **Dependency Security**: Run `uv run safety check` before submitting PRs

### Type Hints

All functions must include type hints:

```python
from pathlib import Path
from typing import Optional, Any

def process_data(
    input_path: Path,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Process data and return results.

    Args:
        input_path: Path to the input data
        output_dir: Optional output directory for results

    Returns:
        Dictionary containing processed data and metadata

    Raises:
        FileNotFoundError: If input_path does not exist
        ValueError: If data format is not supported
    """
    pass
```

### Documentation

- **Docstrings**: Use Google-style docstrings for all public APIs
- **Comments**: Explain *why*, not *what* (code should be self-documenting)
- **README Updates**: Update README.md for significant feature additions
- **Architecture Docs**: Add/update ADRs for architectural changes

## Testing Requirements

### Testing Policy

All new functionality MUST include corresponding tests:

- **Unit tests**: Required for all new functions/classes
- **Integration tests**: Required for new modules/workflows
- **Coverage**: Must maintain ≥{{cookiecutter.code_coverage_target}}% overall coverage
- **Test types**: Use pytest markers (`@pytest.mark.unit`, `@pytest.mark.integration`)

### Test Guidelines

- Test both success and failure cases
- Test edge cases and boundary conditions
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Include docstrings explaining test purpose
- Use fixtures for common setup

### Minimum Coverage

- **Overall Coverage**: {{cookiecutter.code_coverage_target}}% minimum (enforced by CI)
- **New Code**: 90% coverage for new features
- **Critical Paths**: 100% coverage for security-sensitive code

### Test Categories

```bash
# Run all tests
uv run pytest -v

# Run only unit tests
uv run pytest -v -m unit

# Run only integration tests
uv run pytest -v -m integration

# Run tests with coverage report
uv run pytest --cov={{cookiecutter.project_slug}} --cov-report=html
# Open htmlcov/index.html to view coverage report

# Run specific test file
uv run pytest tests/unit/test_module.py -v
```

### Writing Tests

```python
import pytest
from pathlib import Path
from {{cookiecutter.project_slug}}.core import YourModule

def test_module_initialization():
    """Test module initializes correctly."""
    module = YourModule()
    assert module is not None

def test_module_processes_data(tmp_path: Path):
    """Test module processes data correctly."""
    module = YourModule()
    input_file = tmp_path / "input.txt"
    input_file.write_text("test data")

    result = module.process(input_file)

    assert result is not None
    assert result["status"] == "success"
```

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic changes)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

### Examples

```bash
# Feature addition
feat(core): add data processing function

Implements new data processing capabilities with comprehensive
validation and error handling.

Refs: #42

# Bug fix
fix(utils): handle edge case in validation

Previously, certain input patterns caused incorrect results.
Now we properly handle all edge cases.

Fixes: #56

# Breaking change
feat(core)!: redesign API for better ergonomics

BREAKING CHANGE: API has been redesigned for improved usability.
See migration guide in docs/migration/v1.0.0.md
```

## Pull Request Process

### Before Submitting

- [ ] Branch is up-to-date with `main`
- [ ] All tests pass locally
- [ ] Code coverage meets minimum requirements ({{cookiecutter.code_coverage_target}}%)
{%- if cookiecutter.use_pre_commit == "yes" %}
- [ ] Pre-commit hooks pass
{%- endif %}
- [ ] Documentation is updated
- [ ] Commits follow conventional commit format

### Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - **Clear title**: Following conventional commit format
   - **Description**: What changes were made and why
   - **Issue reference**: `Fixes #123` or `Refs #456`
   - **Testing notes**: How reviewers can test the changes
   - **Breaking changes**: Clearly documented (if applicable)

3. **Wait for CI checks**:
{%- if cookiecutter.include_github_actions == "yes" %}
   - All GitHub Actions workflows must pass
{%- endif %}
   - Test coverage must meet requirements

4. **Address review feedback**:
   - Respond to all reviewer comments
   - Push additional commits to the same branch
   - Request re-review when ready

5. **Merge**:
   - Maintainer will merge when approved
   - Follow conventional commits for final commit message

### PR Review Criteria

Reviewers will check:

- **Code Quality**: Follows style guide and best practices
- **Tests**: Adequate test coverage and meaningful tests
- **Documentation**: Clear docstrings and updated docs
- **Security**: No security vulnerabilities introduced
- **Performance**: No performance regressions
- **Compatibility**: Maintains backward compatibility (or documents breaking changes)

## Reporting Issues

### Reporting Bugs

Use the bug report template and include:

- **Description**: Clear description of the bug
- **Reproduction Steps**: Minimal steps to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, package versions
- **Logs**: Relevant error messages or stack traces

### Requesting Features

Use the feature request template and include:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Additional Context**: Screenshots or examples

## Questions?

- **General Questions**: Open a [GitHub Discussion]({{cookiecutter.repo_url}}/discussions)
- **Bug Reports**: Open a [GitHub Issue]({{cookiecutter.repo_url}}/issues)
- **Security Issues**: See [williaby Security Policy](https://github.com/williaby/.github/blob/main/SECURITY.md)
- **Email**: {{cookiecutter.author_email}}

## Recognition

Contributors are recognized in:

- Repository contributors page
- Release notes for significant contributions
- Project documentation

Thank you for contributing to {{cookiecutter.project_name}}!
