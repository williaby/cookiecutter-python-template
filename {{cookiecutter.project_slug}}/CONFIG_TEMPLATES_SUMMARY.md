# Configuration File Templates Summary

This document describes the comprehensive configuration file templates created for the cookiecutter Python template.

## Files Created

All configuration files have been created in `/home/user/cookiecutter-python-template/{{cookiecutter.project_slug}}/`

### 1. **codecov.yml** - Code Coverage Configuration
- **Purpose**: Configure Codecov for coverage tracking and reporting
- **Cookiecutter Variables**:
  - `{{cookiecutter.project_name}}` - Project name in comments
  - `{{cookiecutter.project_short_description}}` - Description in comments
  - `{{cookiecutter.project_slug}}` - Module path for coverage tracking
  - `{{cookiecutter.code_coverage_target}}` - Coverage target percentage
- **Features**:
  - Component-based coverage tracking (ingestion, detection, correction, output, utils)
  - Multi-flag tracking for unit, integration, and Python version tests
  - Comprehensive ignore patterns for tests, cache, documentation, etc.
  - Pull request comment configuration with change requirements
  - Annotations enabled for detailed coverage reports

### 2. **renovate.json** - Dependency Management
- **Purpose**: Configure automated dependency updates via Renovate Bot
- **Cookiecutter Variables**:
  - `{{cookiecutter.github_username}}` - GitHub assignees and reviewers
- **Features**:
  - Extends recommended Renovate configuration
  - Dependency dashboard, semantic commits, semver preservation
  - Package rules for:
    - Grouped Python dependencies and dev dependencies
    - Auto-merged GitHub Actions updates (minor/patch only)
    - High-priority security updates (not auto-merged)
    - Critical security updates with prPriority 10
  - GitHub Actions commit SHA pinning
  - Poetry and GitHub Actions managers enabled
  - Weekly lock file maintenance on Monday mornings

### 3. **REUSE.toml** - License Compliance
- **Purpose**: Centralized license information via REUSE specification
- **Cookiecutter Variables**:
  - `{{cookiecutter.project_slug}}` - Package name
  - `{{cookiecutter.author_name}}` - Copyright holder
  - `{{cookiecutter.repo_url}}` - Download location
  - `{{cookiecutter.license}}` - Primary license for source code
- **License Categories**:
  - Source code (Python, tools, tests): `{{cookiecutter.license}}` (user-selected)
  - Documentation: CC-BY-4.0
  - Configuration files: CC0-1.0
  - Data files and models: ODbL-1.0
  - Build artifacts and generated files: CC0-1.0
  - Benchmark results and metadata: CC0-1.0

### 4. **osv-scanner.toml** - CVE Exception Handling
- **Purpose**: Document vulnerability exceptions and false positives
- **Features**:
  - Template structure with clear guidelines for adding exceptions
  - Three example categories (commented out):
    - Withdrawn vulnerabilities
    - Fixed in current version
    - Not used in project
  - Explicit instructions for proper exception documentation
  - Section for project-specific exceptions
- **No Cookiecutter Variables** - File is generic for all projects

### 5. **mkdocs.yml** - Documentation Configuration (Conditional)
- **Purpose**: Configure MkDocs Material theme for project documentation
- **Condition**: Only generated if `{{cookiecutter.use_mkdocs}} == "yes"`
- **Cookiecutter Variables**:
  - `{{cookiecutter.project_name}}` - Site name and title
  - `{{cookiecutter.project_short_description}}` - Site description
  - `{{cookiecutter.author_name}}` - Author and copyright holder
  - `{{cookiecutter.docs_url}}` - Documentation site URL
  - `{{cookiecutter.repo_url}}` - Repository URL
  - `{{cookiecutter.github_org_or_user}}` - GitHub org/user in repo name
  - `{{cookiecutter.project_slug}}` - Project slug for repo name
- **Features**:
  - Material theme with light/dark mode palette
  - Navigation features: instant loading, tracking, tabs, sections, expand, indexes
  - Code features: copy button, annotations, syntax highlighting
  - 10 plugins: search, autorefs, mkdocstrings, section-index, git-revision-date-localized, gen-files
  - Comprehensive Markdown extensions for Python documentation
  - Google-style docstring parsing via mkdocstrings
  - Pydantic constraint rendering via griffe_pydantic
  - Standard navigation structure (User Guide, API Reference, Development, Project, Security, Operational)
  - Strict mode disabled for repo root file links

### 6. **noxfile.py** - Automation Sessions (Conditional)
- **Purpose**: Define Nox automation sessions for testing, documentation, and compliance
- **Condition**: Only generated if `{{cookiecutter.include_nox}} == "yes"`
- **Cookiecutter Variables**:
  - `{{cookiecutter.python_version}}` - Python version for all sessions
- **Sessions Included**:
  - **fm**: Validate and autofix front matter in documentation
  - **docs**: Build documentation with MkDocs in strict mode
  - **serve**: Serve documentation locally with live reloading
  - **docstrings**: Check docstring coverage with pydocstyle and interrogate
  - **validate**: Run all validation checks (front matter, docstrings, docs build)
  - **reuse**: Check REUSE compliance (Docker-based)
  - **reuse_spdx**: Generate REUSE SPDX document (Docker-based)
  - **sbom**: Generate CycloneDX SBOM (runtime, dev, complete)
  - **scan**: Scan SBOM for vulnerabilities with Trivy (Docker-based)
  - **compliance**: Run all compliance checks (REUSE, SBOM, scan)
- **Features**:
  - Reuses existing virtualenvs across sessions
  - External tool support for Docker-based operations
  - Comprehensive docstrings with usage examples
  - Path argument support for flexible execution
  - Error handling for missing SBOM files

## Cookiecutter Variables Used

All templates use the following cookiecutter variables from `cookiecutter.json`:

| Variable | Type | Default | Used In |
|----------|------|---------|---------|
| `{{cookiecutter.project_name}}` | string | "My Python Project" | codecov.yml, mkdocs.yml |
| `{{cookiecutter.project_slug}}` | auto | Derived from project_name | codecov.yml, renovate.json, REUSE.toml, mkdocs.yml, noxfile.py |
| `{{cookiecutter.project_short_description}}` | string | "A short description..." | codecov.yml, mkdocs.yml |
| `{{cookiecutter.author_name}}` | string | "Your Name" | REUSE.toml, mkdocs.yml |
| `{{cookiecutter.github_username}}` | string | "yourusername" | renovate.json |
| `{{cookiecutter.github_org_or_user}}` | auto | Derived from github_username | mkdocs.yml |
| `{{cookiecutter.license}}` | choice | Apache-2.0 (default) | REUSE.toml |
| `{{cookiecutter.repo_url}}` | auto | Derived from github info | REUSE.toml, mkdocs.yml |
| `{{cookiecutter.docs_url}}` | auto | Derived from project_slug | mkdocs.yml |
| `{{cookiecutter.code_coverage_target}}` | string | "80" | codecov.yml |
| `{{cookiecutter.python_version}}` | choice | 3.12 (default) | noxfile.py |
| `{{cookiecutter.use_mkdocs}}` | choice | "yes" | mkdocs.yml (conditional) |
| `{{cookiecutter.include_nox}}` | choice | "yes" | noxfile.py (conditional) |

## Conditional Rendering

Two files use Jinja2 conditional rendering:

### mkdocs.yml
```jinja2
{%- if cookiecutter.use_mkdocs == "yes" %}
  [Full mkdocs configuration]
{%- else %}
  # mkdocs.yml - NOT CONFIGURED
{%- endif %}
```

### noxfile.py
```jinja2
{%- if cookiecutter.include_nox == "yes" %}
  """Full nox sessions"""
{%- else %}
  """Nox sessions - NOT CONFIGURED"""
{%- endif %}
```

## Component-Based Coverage Tracking

The `codecov.yml` template includes component tracking for:

- **ingestion**: PDF/Image Ingestion module (`src/{{cookiecutter.project_slug}}/ingestion/**`)
- **detection**: IQA Detection module (`src/{{cookiecutter.project_slug}}/detection/**`)
- **correction**: Image Correction module (`src/{{cookiecutter.project_slug}}/correction/**`)
- **output**: JSON Output module (`src/{{cookiecutter.project_slug}}/output/**`)
- **utils**: Utilities module (`src/{{cookiecutter.project_slug}}/utils/**`)

These can be customized per project by editing the generated `codecov.yml` file.

## Integration with Source Project

All templates are based on the image-preprocessing-detector project structure and best practices:

- Follows project naming conventions (snake_case for modules, PascalCase for classes)
- Includes component-based architecture patterns
- Supports Material theme for documentation
- Includes comprehensive compliance and security tooling
- Compatible with Poetry package manager
- Supports GitHub Actions CI/CD workflows

## Usage in Generated Projects

When cookiecutter generates a new project from the template:

1. **codecov.yml** - Automatically configured for project coverage thresholds
2. **renovate.json** - Automated dependency updates with proper assignees
3. **REUSE.toml** - License compliance tracking with project-specific copyright
4. **osv-scanner.toml** - Ready for project-specific vulnerability exceptions
5. **mkdocs.yml** - Generated only if `use_mkdocs == "yes"`
6. **noxfile.py** - Generated only if `include_nox == "yes"`

All paths, URLs, and configuration values are automatically substituted with project-specific values during generation.
