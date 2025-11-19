# CLAUDE.md

Guidance for Claude Code when working with this repository.

> **Context**: Extends global CLAUDE.md from `~/.claude/CLAUDE.md`. Only project-specific configurations below.

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

**Naming Convention**: `tmp_cleanup/.tmp-{task-type}-{timestamp}.md` (e.g., `tmp_cleanup/.tmp-feature-implementation-20250117.md`)

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

**Name**: {{cookiecutter.project_name}}
**Description**: {{cookiecutter.project_short_description}}
**Author**: {{cookiecutter.author_name}} <{{cookiecutter.author_email}}>
**Repository**: {{cookiecutter.repo_url}}

## Core Principles

1. **Security First**: Validate keys, encrypt secrets, scan dependencies before commits
2. **Code Quality**: Maintain {{cookiecutter.code_coverage_target}}% test coverage, pass all linters
3. **Documentation**: Keep docs current, use clear docstrings
4. **Testing**: High coverage, comprehensive test suites

## Development Workflow

### Quick Start
```bash
# Setup
uv sync --all-extras
uv run pre-commit install

# Development cycle
uv run pytest -v              # Run tests
uv run ruff format .          # Format code
uv run ruff check . --fix     # Lint and fix
uv run mypy src/              # Type check
```

### Before Commit (MANDATORY)
```bash
# All must pass:
uv run pytest --cov=src --cov-fail-under={{cookiecutter.code_coverage_target}}
uv run ruff check .
uv run mypy src/
uv run bandit -r src
pre-commit run --all-files        # Or: git commit (runs automatically)
```

## Code Standards

### Naming Conventions
- **Modules/Functions**: `snake_case` (e.g., `data_loader`, `process_file`)
- **Classes**: `PascalCase` (e.g., `DataProcessor`, `ModelConfig`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_PORT`)
- **Files**: `snake_case.py`, test files: `test_*.py`
- **Branches**: `feature/description`, `fix/description`, `docs/description`

### Docstrings (Google Style)
```python
def process_data(input_path: str, max_rows: int = 1000) -> dict[str, Any]:
    """Process data from input file.

    Args:
        input_path: Path to input file
        max_rows: Maximum rows to process (default: 1000)

    Returns:
        Dictionary with processing results

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is invalid
    """
```

## Project Structure

```
src/{{cookiecutter.project_slug}}/
├── __init__.py              # Package initialization
{% if cookiecutter.include_cli == "yes" -%}
├── cli.py                   # CLI entry point
{% endif -%}
├── core/                    # Core business logic
│   ├── __init__.py
│   └── config.py           # Configuration
├── middleware/              # Middleware components
│   └── __init__.py
└── utils/                   # Utilities
    ├── __init__.py
    ├── financial.py        # Financial utilities (Decimal precision)
    └── logging.py          # Structured logging

tests/
├── unit/                   # Unit tests
├── integration/            # Integration tests
├── conftest.py            # Pytest fixtures
└── test_example.py        # Example tests demonstrating best practices
```

## Testing Guidelines

### Coverage Requirements
- **Minimum Coverage**: {{cookiecutter.code_coverage_target}}%
- **Focus**: Core logic, edge cases, error paths
- **Exclude**: CLI entry points, simple getters/setters

### Test Structure
```python
def test_function_name_should_expected_behavior():
    """Test that function_name produces expected_behavior."""
    # Arrange
    input_data = ...

    # Act
    result = function_name(input_data)

    # Assert
    assert result == expected_value
```

## Security Requirements

### Pre-Commit Security Checks
```bash
uv run bandit -r src         # Python security analysis
uv run safety check          # Dependency vulnerabilities
{% if cookiecutter.include_gitleaks == "yes" -%}
gitleaks detect --no-git        # Secrets detection
{% endif -%}
```

### Key Management
- **Never commit**: API keys, passwords, tokens, certificates
- **Use**: `.env` files (in `.gitignore`) with `python-dotenv`
- **Example**:
  ```python
  from pydantic_settings import BaseSettings

  class Settings(BaseSettings):
      api_key: str

      class Config:
          env_file = ".env"
  ```

{% if cookiecutter.use_decimal_precision == "yes" -%}
## Financial Calculations (CRITICAL)

**Always use `Decimal` for money** - never use `float`:

```python
from decimal import Decimal

# CORRECT
price = Decimal('19.99')
quantity = Decimal('3')
total = price * quantity  # Decimal('59.97')

# WRONG - floating point errors
price = 19.99  # float
total = price * 3  # 59.97000000000001
```

See `src/{{cookiecutter.project_slug}}/utils/financial.py` for utilities.
{% endif -%}

## Git Workflow

### Commit Messages (Conventional Commits)
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
- `feat(api): add user authentication endpoint`
- `fix(parser): handle empty input files correctly`
- `docs(readme): update installation instructions`

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch (if using)
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

## CI/CD Pipeline

### GitHub Actions Workflows
{% if cookiecutter.include_github_actions == "yes" -%}
1. **CI** (`.github/workflows/ci.yml`): Testing, linting, type checking
2. **Security** (`.github/workflows/security-analysis.yml`): CodeQL, Bandit, Safety, OSV
{% if cookiecutter.use_mkdocs == "yes" -%}
3. **Docs** (`.github/workflows/docs.yml`): Build and deploy documentation
{% endif -%}
4. **Publish** (`.github/workflows/publish-pypi.yml`): PyPI release automation
{% endif -%}

### Quality Gates (Must Pass)
- ✅ All tests pass ({{cookiecutter.code_coverage_target}}% coverage)
- ✅ Ruff linting (no errors)
- ✅ MyPy type checking (src/ only)
- ✅ Security scans (no high/critical issues)
- ✅ Pre-commit hooks pass

## Configuration Management

Use Pydantic Settings for environment-based config:

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    project_name: str = "{{cookiecutter.project_name}}"
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## Common Tasks

### Add New Dependency
```bash
uv add package-name              # Production dependency
uv add --dev package-name        # Development dependency
```

### Update Dependencies
```bash
uv sync --upgrade                        # Update all
uv sync --upgrade-package package-name   # Update specific package
```

### Run Tests with Coverage
```bash
uv run pytest -v --cov=src --cov-report=html
open htmlcov/index.html              # View coverage report
```

{% if cookiecutter.use_mkdocs == "yes" -%}
### Build Documentation
```bash
uv run mkdocs serve              # Local preview
uv run mkdocs build              # Build static site
```
{% endif -%}

## Troubleshooting

### Pre-commit Hooks Failing
```bash
pre-commit run --all-files           # Run all hooks manually
pre-commit clean                     # Clean cache
pre-commit install --install-hooks   # Reinstall hooks
```

### UV Lock Issues
```bash
uv lock                          # Regenerate lock file
uv sync                          # Reinstall dependencies
```

### MyPy Type Errors
```bash
uv run mypy src/ --show-error-codes  # See error codes
# Add `# type: ignore[error-code]` for specific issues
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Test Suite | <30s | Full suite with coverage |
| CI Pipeline | <5min | All checks |
| Code Coverage | {{cookiecutter.code_coverage_target}}% | Enforced in CI |

*Update with project-specific targets as needed*

## Additional Resources

- **UV Docs**: https://docs.astral.sh/uv/
- **Cruft Docs**: https://cruft.github.io/cruft/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **Ruff Docs**: https://docs.astral.sh/ruff/
- **Conventional Commits**: https://www.conventionalcommits.org/
{% if cookiecutter.use_mkdocs == "yes" -%}
- **MkDocs Material**: https://squidfunk.github.io/mkdocs-material/
{% endif -%}

---

**Last Updated**: {% now 'utc', '%Y-%m-%d' %}
**Template Version**: {{cookiecutter.version}}
