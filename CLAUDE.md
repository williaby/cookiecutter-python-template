# CLAUDE.md

Guidance for Claude Code when working with this cookiecutter template repository.

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

**Naming Convention**: `tmp_cleanup/.tmp-{task-type}-{timestamp}.md` (e.g., `tmp_cleanup/.tmp-template-enhancement-20250117.md`)

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

**Name**: Modern Python Cookiecutter Template
**Description**: Comprehensive cookiecutter template for modern Python projects with UV, Ruff, and best practices
**Repository**: https://github.com/yourusername/cookiecutter-python-template

This is a **meta-repository** - it generates other Python projects. When working on this repository:

### Template Structure

```
cookiecutter-python-template/
├── cookiecutter.json                    # Template variables
├── hooks/                               # Pre/post-generation hooks
│   ├── pre_gen_project.py
│   └── post_gen_project.py
├── {{cookiecutter.project_slug}}/       # Template directory (becomes the generated project)
│   ├── CLAUDE.md                        # Template's CLAUDE.md (for generated projects)
│   ├── pyproject.toml.jinja2           # Jinja2 templated config files
│   └── src/
└── docs/                                # Template documentation
```

### Key Principles for Template Development

1. **Template vs Generated Project**: Changes to `{{cookiecutter.project_slug}}/` affect generated projects, not this repository
2. **Jinja2 Templating**: Use `{{ cookiecutter.variable }}` for dynamic content
3. **Test Generation**: Always test template generation with `cruft create .` before committing
4. **Documentation**: Update both template docs and root docs when making changes
5. **Backwards Compatibility**: Consider existing projects using this template

## Development Workflow

### Quick Start

```bash
# Install dependencies
uv sync

# Test template generation (from parent directory)
cd /tmp
cruft create /home/byron/dev/cookiecutter-python-template --no-input

# Or with custom values
cruft create /home/byron/dev/cookiecutter-python-template
```

### Template Testing Checklist

Before committing template changes:

```bash
# 1. Generate test project
cd /tmp
cruft create /home/byron/dev/cookiecutter-python-template --no-input
cd my_project

# 2. Verify generated project works
uv sync --all-extras
uv run pre-commit install
uv run pytest -v
uv run ruff check .
uv run mypy src/

# 3. Clean up test project
cd /tmp && rm -rf my_project
```

## Template-Specific Standards

### Jinja2 Templating Conventions

```jinja2
{# Comments use {# #} syntax #}

{% if cookiecutter.include_cli == "yes" -%}
# Conditional sections use -%} to strip whitespace
{% endif -%}

# Variables use {{ }} syntax
project_name = "{{ cookiecutter.project_name }}"

# Filters can transform values
author = "{{ cookiecutter.author_name | title }}"
```

### Cookiecutter Variables

**Core Variables** (in `cookiecutter.json`):
- `project_name`: Human-readable project name
- `project_slug`: Python package name (snake_case)
- `author_name`, `author_email`: Author information
- `python_version`: Target Python version
- Feature flags: `include_cli`, `use_mkdocs`, `include_github_actions`, etc.

### Hook Development

**Pre-generation hooks** (`hooks/pre_gen_project.py`):
- Validate input parameters
- Check system requirements
- Prevent generation if requirements not met

**Post-generation hooks** (`hooks/post_gen_project.py`):
- Initialize git repository
- Run initial setup commands
- Clean up unnecessary files based on feature flags

### Common Template Tasks

#### Add New Template Variable

1. Add to `cookiecutter.json`:
   ```json
   {
     "new_variable": "default_value"
   }
   ```

2. Use in template files:
   ```jinja2
   {% if cookiecutter.new_variable == "value" -%}
   # conditional content
   {% endif -%}
   ```

3. Document in `README.md`

4. Test generation with new variable

#### Add New Optional Feature

1. Add feature flag to `cookiecutter.json`:
   ```json
   {
     "include_new_feature": ["yes", "no"]
   }
   ```

2. Add conditional blocks in template files:
   ```jinja2
   {% if cookiecutter.include_new_feature == "yes" -%}
   # Feature-specific content
   {% endif -%}
   ```

3. Add cleanup logic in `hooks/post_gen_project.py`:
   ```python
   if "{{ cookiecutter.include_new_feature }}" != "yes":
       # Remove feature-specific files
       remove_file("path/to/feature/file")
   ```

4. Test both enabled and disabled states

#### Update Template Files

When modifying files in `{{cookiecutter.project_slug}}/`:

1. **Identify scope**: Does this affect all projects or specific features?
2. **Add conditionals**: Use Jinja2 if statements for optional content
3. **Test generation**: Generate multiple test projects with different configurations
4. **Update documentation**: Both template README and generated project README
5. **Version consideration**: Note if this is a breaking change

## Testing Strategy

### Manual Testing

```bash
# Test default configuration
cd /tmp && cruft create /home/byron/dev/cookiecutter-python-template --no-input

# Test with CLI enabled
cd /tmp && cruft create /home/byron/dev/cookiecutter-python-template \
  --config-file <(echo 'default_context: {"include_cli": "yes"}')

# Test with all features enabled
cd /tmp && cruft create /home/byron/dev/cookiecutter-python-template \
  --config-file <(echo 'default_context: {"include_cli": "yes", "use_mkdocs": "yes", "include_github_actions": "yes"}')
```

### Automated Testing (Future Enhancement)

Consider adding:
- CI workflow to test template generation
- Pytest tests for hooks
- Validation of generated project structure

## Documentation Standards

### Template Documentation (Root README.md)

Focus on:
- How to **use** the template
- Available configuration options
- Requirements and prerequisites
- Examples of generated projects

### Generated Project Documentation ({{cookiecutter.project_slug}}/README.md)

Focus on:
- How to **use** the generated project
- Development workflow
- Project-specific commands
- Architecture and design decisions

## Git Workflow

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Scopes for template repository:**
- `template`: Changes to template files in `{{cookiecutter.project_slug}}/`
- `hooks`: Changes to generation hooks
- `config`: Changes to `cookiecutter.json`
- `docs`: Documentation updates
- `ci`: CI/CD workflow changes

**Examples:**
- `feat(template): add pytest-xdist for parallel testing`
- `fix(hooks): correct git initialization in post-generation`
- `docs(readme): update installation instructions`
- `refactor(template): simplify pyproject.toml structure`

### Branch Strategy

- `main`: Stable, tested template versions
- `develop`: Integration branch for new features
- `feature/*`: New template features or enhancements
- `fix/*`: Bug fixes in template generation
- `docs/*`: Documentation improvements

## Pre-Commit Checklist

Before committing template changes:

- [ ] **TODO Management**: Was TodoWrite used for task tracking?
- [ ] **Agent Assignment**: Were tasks assigned to appropriate specialized agents?
- [ ] **Reference Files**: Were temporary reference files created for complex tasks?
- [ ] **Agent Validation**: Was all agent work reviewed and validated?
- [ ] **Template Generation**: Test project generation succeeds with multiple configurations
- [ ] **Generated Project Works**: Generated project passes all quality checks
- [ ] **Hook Testing**: Pre/post generation hooks execute successfully
- [ ] **Documentation**: Updated both template and generated project docs
- [ ] **Backwards Compatibility**: Considered impact on existing template users
- [ ] **Variable Validation**: New variables documented in cookiecutter.json and README
- [ ] **Jinja2 Syntax**: All template files have valid Jinja2 syntax
- [ ] **Feature Flags**: Conditional logic works for all combinations

## Troubleshooting

### Template Generation Fails

```bash
# Enable verbose output
cruft create /home/byron/dev/cookiecutter-python-template -vv

# Check hook errors
# Hooks are in hooks/pre_gen_project.py and hooks/post_gen_project.py
python hooks/pre_gen_project.py  # Test pre-gen hook

# Validate cookiecutter.json syntax
python -m json.tool cookiecutter.json
```

### Jinja2 Syntax Errors

```bash
# Test Jinja2 template syntax
python -c "
from jinja2 import Template
with open('{{cookiecutter.project_slug}}/pyproject.toml.jinja2') as f:
    Template(f.read())
print('Syntax OK')
"
```

### Generated Project Has Issues

```bash
# Generate test project with verbose output
cd /tmp
cruft create /home/byron/dev/cookiecutter-python-template --no-input -vv

# Check generated files
cd my_project
find . -type f -name "*.py" | xargs python -m py_compile  # Check Python syntax
uv sync  # Check dependency resolution
```

## Common Pitfalls

1. **Forgetting to test template generation**: Always generate test project before committing
2. **Hardcoding paths**: Use `{{ cookiecutter.project_slug }}` for package references
3. **Missing conditionals**: Features should be optional with proper feature flags
4. **Hook errors**: Ensure hooks are idempotent and handle edge cases
5. **Whitespace issues**: Use `{%- %}` to control Jinja2 whitespace stripping
6. **Circular dependencies**: Avoid template variables depending on other variables

## Performance Considerations

Template generation should be fast:
- **Target**: <5 seconds for complete project generation
- **Optimize hooks**: Minimize file I/O and external commands
- **Lazy initialization**: Only install/setup what's needed based on feature flags

## Security Considerations

Templates can execute arbitrary code:
- **Review hooks carefully**: `pre_gen_project.py` and `post_gen_project.py` execute during generation
- **Validate user input**: Sanitize cookiecutter variables in hooks
- **No secrets in template**: Never commit API keys or credentials
- **Secure defaults**: Generated projects should have secure configurations by default

## SonarCloud Integration

Both the template repository and generated projects support SonarCloud for continuous code quality analysis.

### Template Repository

**Configuration**:

- **Organization**: `williaby`
- **Project Key**: `williaby_cookiecutter-python-template`
- **Analysis Method**: CI-Based (GitHub Actions)
- **Workflow**: `.github/workflows/sonarcloud.yml`
- **Configuration**: `sonar-project.properties`
- **Dashboard**: <https://sonarcloud.io/project/overview?id=williaby_cookiecutter-python-template>

**What's Analyzed**:

- Hook files (`hooks/*.py`)
- Template files (`{{cookiecutter.project_slug}}/`)
- Code quality metrics (bugs, code smells, maintainability)
- Security vulnerabilities and hotspots
- Hook file test coverage (when tests exist)

**Quality Standards**:

- Quality gate must pass before merging PRs
- Security rating must be A or B
- Maintainability rating must be A or B
- No critical or high-severity vulnerabilities

### Generated Projects

When users create projects with `include_sonarcloud=yes`:

**Files Created**:

- `.github/workflows/sonarcloud.yml` - CI-based analysis workflow
- `sonar-project.properties` - Project configuration

**Setup Required**:

1. User creates SonarCloud project for their repository
2. `SONAR_TOKEN` secret configured (organization or repository level)
3. Project key: `{github_org}_{project_slug}` (auto-configured in templates)

**Features**:

- **Quality Gate Enforcement**: Fails PR if quality standards not met
- **PR Decoration**: Shows issues directly in pull request comments
- **Coverage Integration**: Pytest coverage reports uploaded to SonarCloud
- **Security Analysis**: Continuous vulnerability detection
- **Code Smells**: Maintainability issue detection
- **Technical Debt**: Quantified effort to fix issues

**Best Practices**:

- Always use CI-Based analysis (not Automatic) for Python projects
- Configure quality gates before first analysis
- Review SonarCloud issues before merging PRs
- Aim for Quality Gate = "Passed" on all branches
- Monitor coverage trends (target: 80%+ as configured in template)

**Troubleshooting**:

- If workflow fails: Check `SONAR_TOKEN` secret exists
- If no coverage: Verify pytest runs and generates `coverage.xml`
- If analysis incomplete: Check `sonar-project.properties` source paths
- If quality gate fails: Review specific issues in SonarCloud dashboard

## Additional Resources

- **Cookiecutter Docs**: https://cookiecutter.readthedocs.io/
- **Cruft Docs**: https://cruft.github.io/cruft/
- **Jinja2 Docs**: https://jinja.palletsprojects.com/
- **UV Docs**: https://docs.astral.sh/uv/
- **Template Examples**: See `docs/` for detailed examples

---

**Repository Type**: Cookiecutter Template (Meta-Repository)
**Generated Project Type**: Modern Python CLI/Library
**Template Version**: See `cookiecutter.json` for version information
