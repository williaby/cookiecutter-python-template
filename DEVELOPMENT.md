# Development Guide for Cookiecutter Python Template

This guide explains how to **develop and maintain the template itself**, not projects generated from it.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/cookiecutter-python-template.git
cd cookiecutter-python-template

# 2. Install development dependencies
uv sync --extra dev

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Make your changes to template files

# 5. Test template generation
uv run cruft create . --no-input --output-dir /tmp/test-project
cd /tmp/test-project
uv sync --all-extras
uv run pytest -v
cd -
rm -rf /tmp/test-project

# 6. Run quality checks
uv run pre-commit run --all-files

# 7. Commit your changes
git add .
git commit -m "feat(template): your change description"
```

## Repository Structure

### Critical Directories

```
cookiecutter-python-template/
├── cookiecutter.json                # Template configuration (40+ variables)
├── hooks/                           # Pre/post-generation hooks (REAL PYTHON CODE)
│   ├── pre_gen_project.py          # Input validation before generation
│   └── post_gen_project.py         # Cleanup and setup after generation
├── {{cookiecutter.project_slug}}/  # Template directory (JINJA2 TEMPLATES)
│   ├── src/
│   ├── tests/
│   ├── pyproject.toml              # Template's pyproject.toml (has Jinja2)
│   └── ...
├── pyproject.toml                  # THIS REPO'S config (for developing the template)
├── .pre-commit-config.yaml         # Quality checks for the template repo
└── DEVELOPMENT.md                  # This file
```

### Understanding the Difference

| File/Directory | Type | Purpose | Linting |
|----------------|------|---------|---------|
| `hooks/*.py` | **Real Python** | Runs during template generation | ✅ Full linting |
| `{{cookiecutter.project_slug}}/*` | **Jinja2 Templates** | Copied to generated projects | ⚠️  Excluded from most checks |
| `pyproject.toml` (root) | **Real Config** | Manages THIS repo's dependencies | ✅ Used by tools |
| `{{cookiecutter.project_slug}}/pyproject.toml` | **Jinja2 Template** | Becomes generated project's config | ⚠️  Has `{{ }}` syntax |

## Development Workflow

### 1. Setting Up Development Environment

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Verify installation
uv run ruff --version
uv run mypy --version
uv run cookiecutter --version
uv run cruft --version

# Install pre-commit hooks
uv run pre-commit install
```

### 2. Making Changes to the Template

#### Modifying Template Files (`{{cookiecutter.project_slug}}/`)

**Important**: These files use Jinja2 syntax and will be copied to generated projects.

```bash
# Example: Add a new dependency to generated projects
# Edit: {{cookiecutter.project_slug}}/pyproject.toml

[project]
dependencies = [
    "requests>=2.31.0",
    # Add your new dependency here
]
```

**Jinja2 Syntax Examples**:
```jinja2
# Conditional sections
{% if cookiecutter.include_cli == "yes" -%}
dependencies = ["click>=8.1.0"]
{% endif -%}

# Variable substitution
project_name = "{{ cookiecutter.project_name }}"

# Loops
{% for feature in cookiecutter.optional_features -%}
# {{ feature }}
{% endfor -%}
```

#### Modifying Hooks (`hooks/`)

**Important**: These are real Python files that execute during template generation.

```bash
# Example: Add validation to pre_gen_project.py

def validate_new_field(value: str) -> bool:
    """Validate a new field."""
    return bool(re.match(r'^[a-z]+$', value))

def main() -> None:
    new_field = "{{ cookiecutter.new_field }}"
    if not validate_new_field(new_field):
        errors.append(f"ERROR: Invalid value: {new_field}")
```

**Quality Requirements for Hooks**:
- ✅ Must pass Ruff linting
- ✅ Must pass MyPy type checking
- ✅ Must pass Bandit security scan
- ✅ Must be formatted with Black

### 3. Testing Template Generation

#### Basic Test

```bash
# Generate with default values
uv run cruft create . --no-input --output-dir /tmp/test-project

# Verify generated project structure
ls -la /tmp/test-project

# Clean up
rm -rf /tmp/test-project
```

#### Test with Custom Values

```bash
# Generate with specific configuration
uv run cruft create . --output-dir /tmp/test-custom

# Or use config file
cat > /tmp/test-config.yaml <<EOF
default_context:
  project_name: "My Test Project"
  include_cli: "yes"
  use_mkdocs: "yes"
  include_docker: "yes"
EOF

uv run cruft create . --config-file /tmp/test-config.yaml --output-dir /tmp/test-custom
```

#### Comprehensive Test (Verify Generated Project Works)

```bash
# 1. Generate project
uv run cruft create . --no-input --output-dir /tmp/test-full
cd /tmp/test-full

# 2. Install dependencies
uv sync --all-extras

# 3. Run quality checks
uv run ruff check .
uv run mypy src/
uv run bandit -r src/

# 4. Run tests
uv run pytest -v --cov=src --cov-fail-under=80

# 5. Verify pre-commit works
uv run pre-commit install
uv run pre-commit run --all-files

# 6. Clean up
cd -
rm -rf /tmp/test-full
```

### 4. Running Quality Checks on Template Repository

```bash
# Run all pre-commit hooks (recommended before commit)
uv run pre-commit run --all-files

# Run individual hooks
uv run pre-commit run black --all-files
uv run pre-commit run ruff --all-files
uv run pre-commit run mypy --all-files

# Run linters directly (faster for iteration)
uv run ruff check hooks/
uv run ruff format hooks/
uv run mypy hooks/
uv run bandit -r hooks/

# Check for hardcoded values that should be templated
uv run pre-commit run check-hardcoded-values
```

### 5. Adding New Template Variables

#### Step 1: Add to cookiecutter.json

```json
{
  "project_name": "My Project",
  "new_variable": "default_value",
  "new_choice_variable": ["option1", "option2", "option3"]
}
```

#### Step 2: Use in Template Files

```jinja2
# In {{cookiecutter.project_slug}}/README.md
# {{ cookiecutter.project_name }}

{% if cookiecutter.new_choice_variable == "option1" -%}
Special content for option1
{% endif -%}
```

#### Step 3: Add Validation (Optional)

```python
# In hooks/pre_gen_project.py
def validate_new_variable(value: str) -> bool:
    """Validate new_variable format."""
    return len(value) >= 3

def main() -> None:
    new_var = "{{ cookiecutter.new_variable }}"
    if not validate_new_variable(new_var):
        errors.append("ERROR: new_variable must be at least 3 characters")
```

#### Step 4: Add Cleanup Logic (If Conditional)

```python
# In hooks/post_gen_project.py
def cleanup_conditional_files() -> None:
    if "{{ cookiecutter.new_choice_variable }}" != "option1":
        remove_file(Path("optional_feature_file.py"))
```

#### Step 5: Document in README

Update `README.md` with the new variable and its purpose.

### 6. Adding New Optional Features

Follow the established pattern for conditional features:

1. **Add feature flag** to `cookiecutter.json`:
   ```json
   {
     "include_new_feature": ["yes", "no"]
   }
   ```

2. **Add feature files** to template with conditional blocks:
   ```jinja2
   {% if cookiecutter.include_new_feature == "yes" -%}
   # Feature-specific content
   {% endif -%}
   ```

3. **Add cleanup logic** in `hooks/post_gen_project.py`:
   ```python
   if "{{ cookiecutter.include_new_feature }}" == "no":
       remove_file(Path("src/{{ cookiecutter.project_slug }}/feature.py"))
   ```

4. **Update success message** in `hooks/post_gen_project.py`:
   ```python
   if include_new_feature:
       optional_features.append("New Feature")
   ```

## Common Development Tasks

### Update Pre-commit Hook Versions

```bash
# Update to latest versions
uv run pre-commit autoupdate

# Test updated hooks
uv run pre-commit run --all-files

# Commit if successful
git add .pre-commit-config.yaml
git commit -m "chore: update pre-commit hook versions"
```

### Add New Dependency to Generated Projects

```bash
# 1. Edit {{cookiecutter.project_slug}}/pyproject.toml
# Add dependency to [project.dependencies] or [project.optional-dependencies]

# 2. Test generation
uv run cruft create . --no-input --output-dir /tmp/test
cd /tmp/test
uv sync --all-extras  # Verify dependency resolves
cd -
rm -rf /tmp/test

# 3. Document in README if significant
```

### Debug Hook Failures

```bash
# 1. Enable verbose output
uv run cruft create . --no-input --output-dir /tmp/debug -vv

# 2. Check hook syntax
python3 -m py_compile hooks/pre_gen_project.py
python3 -m py_compile hooks/post_gen_project.py

# 3. Test hooks directly (manual)
cd /tmp
# Manually set cookiecutter variables and run hook
python3 /path/to/hooks/pre_gen_project.py
```

### Test Jinja2 Template Syntax

```bash
# Validate Jinja2 syntax
python3 -c "
from jinja2 import Template
import sys
with open('{{cookiecutter.project_slug}}/pyproject.toml') as f:
    try:
        Template(f.read())
        print('✓ Syntax valid')
    except Exception as e:
        print(f'✗ Syntax error: {e}')
        sys.exit(1)
"
```

## Git Workflow

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Add/update tests
- `chore`: Maintenance tasks

**Scopes** (for this template repo):
- `template`: Changes to template files
- `hooks`: Changes to generation hooks
- `config`: Changes to cookiecutter.json
- `ci`: CI/CD workflow changes
- `docs`: Documentation updates

**Examples**:
```bash
feat(template): add Docker multi-stage build support
fix(hooks): correct git initialization on Windows
docs(readme): update installation instructions
chore(config): add python 3.13 support
```

### Branch Strategy

- `main`: Stable, tested template versions
- `develop`: Integration branch (if using)
- `feature/*`: New template features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

## Testing Checklist

Before committing template changes:

- [ ] Template generates successfully with default values
- [ ] Template generates successfully with all features enabled
- [ ] Template generates successfully with all features disabled
- [ ] Generated project passes all quality checks (Ruff, MyPy, Bandit)
- [ ] Generated project tests pass
- [ ] Pre-commit hooks pass on template repository
- [ ] Hooks pass all linters (Ruff, MyPy, Black, Bandit)
- [ ] No hardcoded values that should be templated
- [ ] Documentation updated (README.md, this file)
- [ ] CHANGELOG.md updated (if significant change)

## Troubleshooting

### Template Generation Fails

```bash
# Problem: Template generation fails with Jinja2 error
# Solution: Check for unescaped {{ }} in template files

# Problem: Hook validation fails
# Solution: Check hooks/pre_gen_project.py validation logic

# Problem: Variables not rendering
# Solution: Verify variable name in cookiecutter.json matches usage
```

### Pre-commit Hooks Fail

```bash
# Problem: markdownlint fails
# Solution: Requires Node.js - install with: apt-get install nodejs npm

# Problem: Hook permanently fails
# Solution: Skip temporarily: SKIP=hook-id git commit -m "message"

# Problem: Need to update hooks
# Solution: pre-commit autoupdate
```

### Generated Project Has Issues

```bash
# Problem: Generated project missing files
# Solution: Check hooks/post_gen_project.py cleanup logic

# Problem: Jinja2 variables not rendered
# Solution: Ensure variable exists in cookiecutter.json

# Problem: Dependency conflicts
# Solution: Test dependency resolution in isolated environment
```

## CI/CD (Future)

Planned GitHub Actions workflows for template validation:

- **Template Generation Test**: Generate project with all configurations
- **Quality Checks**: Run all linters on hooks
- **Generated Project Test**: Ensure generated project passes its own tests
- **Multi-Python Test**: Test with Python 3.10, 3.11, 3.12, 3.13

## Resources

- **Cookiecutter Docs**: https://cookiecutter.readthedocs.io/
- **Cruft Docs**: https://cruft.github.io/cruft/
- **Jinja2 Docs**: https://jinja.palletsprojects.com/
- **UV Docs**: https://docs.astral.sh/uv/
- **Ruff Docs**: https://docs.astral.sh/ruff/
- **Pre-commit Docs**: https://pre-commit.com/

## Getting Help

- **Template Issues**: Open issue in this repository
- **Generated Project Issues**: Check CLAUDE.md in generated project
- **Questions**: See README.md for contact information

---

**Remember**: This is a **meta-repository** - changes here affect all future projects generated from the template. Test thoroughly!
