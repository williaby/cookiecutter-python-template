# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Project Context**: Extends global CLAUDE.md standards from `~/.claude/CLAUDE.md`. Only project-specific configurations documented below.

## Claude Code Supervisor Role (CRITICAL)

**Claude Code acts as the SUPERVISOR for all development tasks and MUST:**

1. **Always Use TodoWrite Tool**: Create and maintain TODO lists for ALL tasks to track progress
2. **Assign Tasks to Agents**: Each TODO item should be assigned to a specialized agent via Zen MCP Server
3. **Review Agent Work**: Validate all agent outputs before proceeding to next steps
4. **Use Temporary Reference Files**: Create `.tmp-` prefixed files in `tmp_cleanup/` folder to store detailed context that might be lost during compaction
5. **Maintain Continuity**: Use reference files to preserve TODO details across conversation compactions

### Agent Assignment Patterns

```bash
# Always assign TODO items to appropriate agents:
- Security tasks → Security Agent (via mcp__zen-core__secaudit)
- Code reviews → Code Review Agent (via mcp__zen-core__codereview)
- Testing → Test Engineer Agent (via mcp__zen-core__testgen)
- Documentation → Documentation Agent (via mcp__zen-core__docgen)
- Debugging → Debug Agent (via mcp__zen-core__debug)
- Analysis → Analysis Agent (via mcp__zen-core__analyze)
- Refactoring → Refactor Agent (via mcp__zen-core__refactor)
```

### Temporary Reference Files (Anti-Compaction Strategy)

**ALWAYS create temporary reference files when:**
- TODO list contains >5 items
- Complex implementation details need preservation
- Multi-step workflows span multiple conversation turns
- Agent assignments and progress need tracking

**Naming Convention**: `tmp_cleanup/.tmp-{task-type}-{timestamp}.md` (e.g., `tmp_cleanup/.tmp-feature-impl-20250205.md`)

### Supervisor Workflow Patterns (MANDATORY)

**Every development task MUST follow this pattern:**

1. **Create TODO List**: Use TodoWrite tool to break down the task into specific, actionable items
2. **Agent Assignment**: Assign each TODO item to the most appropriate specialized agent
3. **Progress Tracking**: Mark items as in_progress when assigned, completed when validated
4. **Reference File Creation**: For complex tasks, create `.tmp-` reference files immediately
5. **Agent Output Validation**: Review all agent work before marking items complete

**For complex tasks requiring multiple agents:**

1. **Sequential Dependencies**: Use TodoWrite to show dependencies between tasks
2. **Parallel Execution**: Assign independent tasks to multiple agents simultaneously
3. **Integration Points**: Create specific TODO items for integrating agent outputs
4. **Quality Gates**: Assign review tasks to appropriate agents after implementation

## Project Overview

**Project Name**: {{cookiecutter.project_name}}

**Short Description**: {{cookiecutter.project_short_description}}

**Repository**: {{cookiecutter.repo_url}}

**Author**: {{cookiecutter.author_name}} <{{cookiecutter.author_email}}>

This project follows best practices for Python development with emphasis on:
- **Security First**: Always validate keys, encrypt secrets, scan dependencies
- **Code Quality**: Maintain consistent standards across the codebase
- **Testing**: High coverage (minimum {{cookiecutter.code_coverage_target}}%) and comprehensive test suites
- **Documentation**: Clear, current, and well-formatted documentation

## Development Philosophy (MANDATORY)

**Security First** → **Quality Standards** → **Documentation** → **Testing** → **Collaboration**

### Core Principles

1. **Security First**: Always validate keys, encrypt secrets, scan dependencies
2. **Reuse First**: Check existing repositories for solutions before building new code
3. **Configure, Don't Build**: Prefer configuration and orchestration over custom implementation
4. **Quality Standards**: Maintain consistent code quality across all projects
5. **Documentation**: Keep documentation current and well-formatted
6. **Testing**: Maintain high test coverage and run tests before commits
7. **Collaboration**: Use consistent Git workflows and clear commit messages

## Naming Conventions (MANDATORY COMPLIANCE)

**Core Components:**
- **Module Names**: snake_case (e.g., `data_loader`, `model_trainer`)
- **Classes**: PascalCase (e.g., `DataProcessor`, `ModelConfig`)
- **Functions**: snake_case (e.g., `load_data`, `train_model`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_BATCH_SIZE`, `MAX_RETRIES`)

**Code & Files:**
- **Python Files**: snake_case.py
- **Test Files**: test_*.py or *_test.py
- **Git Branches**: kebab-case with prefixes (e.g., `feature/add-model`, `fix/data-validation`)
- **Directories**: snake_case

## Common Commands

### Development Workflow

```bash
# Install dependencies (includes dev tools)
poetry install --with dev

# Install with optional dependency groups
poetry install --with dev,ml       # ML dependencies if included
poetry install --with dev,api      # API framework if included

# Setup pre-commit hooks (required before first commit)
poetry run pre-commit install

# Run CLI tool (if CLI is included)
poetry run {{cookiecutter.cli_tool_name}} --help
```

### Testing

```bash
# Run all tests with coverage ({{cookiecutter.code_coverage_target}}% minimum enforced)
poetry run pytest -v

# Run specific test categories
poetry run pytest -v -m unit               # Unit tests only
poetry run pytest -v -m integration        # Integration tests only
poetry run pytest -v -m "not slow"         # Exclude slow tests

# Run single test file
poetry run pytest tests/unit/test_module.py -v

# Run single test function
poetry run pytest tests/unit/test_module.py::test_function_name -v

# Run with coverage report
poetry run pytest --cov=src --cov-report=html --cov-report=term-missing

# Run tests in parallel (faster for large suites)
poetry run pytest -n auto
```

### Code Quality

```bash
# Format code (required before commit)
poetry run ruff format src tests

# Lint and auto-fix
poetry run ruff check --fix src tests

# Type checking (strict mode)
poetry run mypy src

# Docstring coverage check
poetry run interrogate -v src

# Run all pre-commit hooks manually
poetry run pre-commit run --all-files

# Security scanning
poetry run bandit -r src
poetry run safety check
```

### Docstring Coverage

```bash
# Check docstring coverage (minimum {{cookiecutter.docstring_coverage_target}}%)
poetry run interrogate -v src

# Generate detailed report
poetry run interrogate -v -o src > docstring_report.txt
```

### Mutation Testing

```bash
# Run mutation tests to validate test suite effectiveness
poetry run mutmut run

# Show results
poetry run mutmut results

# Run specific tests during mutation
poetry run mutmut run --tests tests/unit/
```

{% if cookiecutter.include_nox == "yes" %}
### Nox Sessions

```bash
# List available Nox sessions
poetry run nox -l

# Run all sessions
poetry run nox

# Run specific session
poetry run nox -s test
```
{% endif %}

### Security Requirements (MANDATORY)

```bash
# Key validation (SHOULD pass for secure commits)
gpg --list-secret-keys  # For GPG-signed commits
ssh-add -l              # For SSH key-based auth

# Security scanning
poetry run bandit -r src                    # Python security analysis
poetry run safety check                     # Dependency vulnerability check

# Optional: OSV Scanner (if installed locally)
osv-scanner --lockfile=poetry.lock
```

## Architecture Overview

### Module Structure

The project is organized with clear separation of concerns:

```
src/{{cookiecutter.project_slug}}/
├── __init__.py                 # Package initialization
├── main.py or cli.py          # Entry point (if CLI included)
├── config.py                  # Configuration management
├── core/                      # Core functionality
│   ├── __init__.py
│   ├── module_one.py
│   └── module_two.py
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── logging.py
│   └── helpers.py
└── schemas.py                 # Pydantic models (if used)

tests/
├── unit/                      # Unit tests
├── integration/               # Integration tests
└── conftest.py               # Pytest fixtures
```

### Data Flow Pattern

When designing your application:

1. **Input Validation**: Use Pydantic models to validate inputs
2. **Processing**: Core business logic in dedicated modules
3. **Error Handling**: Structured error handling with appropriate logging
4. **Output Generation**: Serialize results in expected formats

### Configuration Management

Use Pydantic Settings for environment-based configuration:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_name: str = "{{cookiecutter.project_name}}"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
```

## Project-Specific Standards

### Coverage Requirements

- **Minimum Code Coverage**: {{cookiecutter.code_coverage_target}}% enforced via `--cov-fail-under={{cookiecutter.code_coverage_target}}`
- **Docstring Coverage**: {{cookiecutter.docstring_coverage_target}}% minimum via `interrogate`
- **MyPy**: Strict type checking on `src/`, relaxed mode on `tests/`
- **Pre-commit**: All hooks must pass before commit (Ruff format, Ruff lint, MyPy, Bandit)

### Performance Targets (Project-Specific)

Define and track performance metrics:

| Metric | Target | Notes |
|--------|--------|-------|
| Test Execution | <30s | Full suite with coverage |
| API Response (if applicable) | <100ms | P95 latency |
| Memory Usage | <500MB | Single worker |

*Update this table with your actual targets*

### JSON/Data Schema

If using Pydantic for data validation, document your schema models:

```python
from pydantic import BaseModel

class DataModel(BaseModel):
    """Model description with required and optional fields."""
    required_field: str
    optional_field: str | None = None
```

### Quality Metrics

Track and maintain:

- **Test Coverage**: Run `poetry run pytest --cov=src` before commits
- **Type Safety**: Strict MyPy checks on all source code
- **Code Style**: Ruff formatting and linting must pass
- **Security**: Bandit and Safety scans with zero critical issues

## Testing Standards

### Test Organization

```bash
tests/
├── unit/                    # Fast, isolated tests
├── integration/             # Tests requiring external services/setup
├── conftest.py             # Shared fixtures and configuration
└── fixtures/               # Test data and helper objects
```

### Test Markers

Use pytest markers to categorize tests:

```bash
# Available markers:
- @pytest.mark.unit           # Fast unit tests
- @pytest.mark.integration    # Integration tests (slower)
- @pytest.mark.slow           # Long-running tests
- @pytest.mark.benchmark      # Performance benchmarks
```

### Fixture Management

```python
# conftest.py - Shared fixtures
import pytest

@pytest.fixture
def sample_data():
    """Fixture providing test data."""
    return {"key": "value"}
```

### Coverage Exclusions

Lines marked with `# pragma: no cover` are excluded from coverage calculations. Use sparingly:

```python
if __name__ == "__main__":  # pragma: no cover
    main()
```

## Pre-Commit Linting Checklist

Before committing ANY changes, ensure:

- [ ] **TODO Management**: Was TodoWrite used for task tracking?
- [ ] **Agent Assignment**: Were tasks assigned to appropriate specialized agents?
- [ ] **Reference Files**: Were temporary reference files created for complex tasks?
- [ ] **Agent Validation**: Was all agent work reviewed and validated?
- [ ] **Code Quality**: Ruff formatting, Ruff linting, MyPy type checking passed
- [ ] **Security Scans**: Bandit and Safety checks completed successfully
- [ ] **Test Coverage**: All tests pass with minimum {{cookiecutter.code_coverage_target}}% coverage
- [ ] **Docstring Coverage**: Minimum {{cookiecutter.docstring_coverage_target}}% docstring coverage
- [ ] **Configuration**: `.env` file properly configured with encrypted secrets (if applicable)
- [ ] **Git Signing**: Commits are signed (optional but recommended)
- [ ] **Documentation**: Code changes include relevant documentation updates
- [ ] **File-Specific Linting**: Appropriate linter run for modified file types

## CI/CD Pipeline

### GitHub Actions

The project uses GitHub Actions for continuous integration:

- **Triggers**: PRs and pushes to main/develop branches
- **Jobs**: Testing, code quality checks, security scans, documentation builds
- **Coverage Reports**: Uploaded to Codecov

{% if cookiecutter.include_github_actions == "yes" %}

**Workflow Files**:
- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/security-analysis.yml` - Security scanning
- `.github/workflows/docs.yml` - Documentation build (if applicable)

{% endif %}

### Quality Gates

All of the following must pass before merging:

1. All tests pass with {{cookiecutter.code_coverage_target}}%+ coverage
2. Ruff format and lint checks pass
3. MyPy type checking passes
4. Bandit security scan passes with no critical issues
5. Safety dependency scan passes

## Troubleshooting

### Tests Failing

```bash
# Check coverage threshold
poetry run pytest --cov=src --cov-report=term-missing

# Run specific failing test
poetry run pytest tests/path/to/test.py::test_name -v

# Check pre-commit hooks
poetry run pre-commit run --all-files

# Run tests in verbose mode
poetry run pytest -vv
```

### Type Errors

```bash
# MyPy strict mode on src/
poetry run mypy src

# Check specific file
poetry run mypy src/{{cookiecutter.project_slug}}/module.py

# Show all errors
poetry run mypy src --show-error-codes
```

### Import Errors

```bash
# Verify PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH

# Check installed packages
poetry show

# Reinstall dependencies
poetry install --with dev
```

### Pre-Commit Failures

```bash
# Run all hooks manually
poetry run pre-commit run --all-files

# Run specific hook
poetry run pre-commit run ruff-format --all-files

# Bypass hooks (NOT recommended)
git commit --no-verify  # Use only when necessary
```

### Performance Issues

For performance troubleshooting:

```bash
# Profile test execution
poetry run pytest --durations=10

# Check import times
python -X importtime -c "import {{cookiecutter.project_slug}}" 2>&1 | grep "{{cookiecutter.project_slug}}"

# Memory profiling (if memory_profiler installed)
pip install memory-profiler
python -m memory_profiler script.py
```

### Dependency Issues

```bash
# Check for conflicts
poetry check

# Update all dependencies
poetry update

# Update specific package
poetry update package-name

# Check security vulnerabilities
poetry run safety check
```

## Key Technologies

**Core Stack**:
- **Python**: {{cookiecutter.python_version}}+
- **Package Manager**: Poetry 2.0+
- **Pydantic**: v2.0+ for validation and serialization

**Testing & Quality**:
- **pytest**: Testing framework with coverage
- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability scanning

{% if cookiecutter.include_cli == "yes" %}
**CLI Framework**:
- **Click**: Command-line interface framework
{% endif %}

{% if cookiecutter.include_api_framework == "yes" %}
**API Framework** (if included):
- **FastAPI**: Modern API framework
- **Uvicorn**: ASGI server
{% endif %}

{% if cookiecutter.include_ml_dependencies == "yes" %}
**Machine Learning** (if included):
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **TensorBoard**: Model visualization (if applicable)
{% endif %}

**Documentation**:
{% if cookiecutter.use_mkdocs == "yes" %}
- **MkDocs**: Static documentation generator
- **Material for MkDocs**: Professional documentation theme
{% else %}
- **Sphinx**: Documentation generator (if applicable)
{% endif %}

**Development Tools**:
- **pre-commit**: Git hooks framework
- **Interrogate**: Docstring coverage checker
- **mutmut**: Mutation testing for test quality

## Additional Resources

- **pytest Documentation**: https://docs.pytest.org/
- **Pydantic Documentation**: https://docs.pydantic.dev/latest/
- **Poetry Documentation**: https://python-poetry.org/docs/
- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **MyPy Documentation**: https://mypy.readthedocs.io/

---

*This configuration extends global CLAUDE.md standards. For detailed specifications on security, testing, and Git workflows, see `~/.claude/CLAUDE.md` and referenced files in `/standards/` directory.*
