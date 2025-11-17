# GitHub Actions Workflows

Comprehensive CI/CD workflows for the `{% raw %}{{ cookiecutter.project_name }}{% endraw %}` project, based on production best practices from the image-preprocessing-detector project.

## Overview

This directory contains four GitHub Actions workflow files that automate code quality, security, testing, documentation, and publishing:

| Workflow | File | Purpose | Trigger |
|----------|------|---------|---------|
| **CI** | `ci.yml` | Main CI pipeline with testing and quality checks | PR/Push to main/develop |
| **Security Analysis** | `security-analysis.yml` | Comprehensive security scanning | PR/Schedule/Manual |
| **Documentation** | `docs.yml` | Documentation building and deployment | Changes to docs/ (if enabled) |
| **PyPI Publishing** | `publish-pypi.yml` | Release automation to PyPI | Release creation/Manual |

## Workflow Details

### 1. CI Pipeline (`ci.yml`)

**Purpose**: Code quality, testing, and formatting validation

**Jobs**:
- **setup-optimized**: Dependencies installation with Poetry caching
- **test**: Comprehensive pytest suite with coverage reporting (target: {% raw %}{{ cookiecutter.code_coverage_target }}{% endraw %}%)
- **quality-checks**:
  {% if cookiecutter.use_mypy == "yes" %}- MyPy strict type checking{% endif %}
  {% if cookiecutter.use_ruff == "yes" %}- Ruff code formatting and linting{% endif %}
- **ci-gate**: Status check for branch protection

**Triggers**:
- Pull requests to `main`, `develop`, or `feature/**` branches
- Pushes to `main` or `develop`
- Manual workflow dispatch

**Concurrency**: Only one CI run per branch at a time (cancels in-progress runs)

**Features**:
- Multi-version Python testing: `{% raw %}{{ cookiecutter.python_version }}{% endraw %}`
- Poetry caching for faster builds
- System dependency validation
- Coverage reporting with `--cov-fail-under={% raw %}{{ cookiecutter.code_coverage_target }}{% endraw %}`
- Disk cleanup to prevent runner overload
- Step Security hardening with audit policy

**Key Outputs**:
- `junit.xml` - Test results
- `coverage.xml` - Coverage report
- `htmlcov/` - HTML coverage dashboard

---

### 2. Security Analysis (`security-analysis.yml`)

**Purpose**: Consolidated security scanning and vulnerability detection

**Jobs**:
- **detect-changes**: Filter to only run on security-relevant file changes
- **codeql-analysis**: GitHub CodeQL static analysis with security-extended + quality queries
- **dependency-security**: GitHub Dependency Review for PR-based vulnerability detection
- **security-scanning**:
  - Bandit: Python static security analysis
  - Safety: Dependency CVE scanning
- **osv-scanner**: Multi-ecosystem vulnerability detection (respects `osv-scanner.toml`)

**Triggers**:
{% if cookiecutter.include_security_scanning == "yes" %}- Pull requests to main/develop/feature branches
- Weekly schedule (Monday 2:30 AM UTC)
- Manual workflow dispatch{% else %}- Disabled by default
- Enable by setting `include_security_scanning` to "yes"{% endif %}

**Conditional Rendering**:
- **If `include_security_scanning == "yes"`**: Full security scanning pipeline
- **If `include_security_scanning == "no"`**: Minimal security gate

**Key Features**:
- CodeQL: Security-extended + security-and-quality query suites
- Dependency Review: Blocks moderate+ severity vulnerabilities
- OSV Scanner: JSON output with severity parsing
- Path-based filtering: Only runs on Python/dependency files
- Step Security hardening on all jobs

**Key Outputs**:
- `bandit-report.json` - Bandit findings
- `safety-report.json` - Safety CVE scan
- `osv-results.json` - OSV vulnerability scan
- GitHub Code Scanning tab: CodeQL results

---

### 3. Documentation (`docs.yml`)

**Purpose**: Documentation validation, building, and GitHub Pages deployment

{% if cookiecutter.use_mkdocs == "yes" %}**Jobs**:
- **validate**: Front matter validation (optional autofix)
- **docstrings**: Docstring style and coverage checks
- **build**: MkDocs site build
- **links**: Broken link detection with Lychee
- **deploy**: Deploy to GitHub Pages (main branch only)

**Triggers**:
- Pull requests with changes to docs/, mkdocs.yml, src/, etc.
- Pushes to main/develop with documentation changes
- Only runs if MkDocs configuration exists

**Features**:
- Front matter validation with optional autofix
- Docstring coverage target: {% raw %}{{ cookiecutter.docstring_coverage_target }}{% endraw %}%
- MkDocs strict build (catches missing pages, broken internal links)
- Lychee link checker (accepts 200, 206, 429 status codes)
- GitHub Pages deployment on pushes to main

**Key Outputs**:
- `mkdocs-site/` - Built documentation site
- `front-matter-report.json` - Validation results
- `link-check-report/` - Lychee results
- GitHub Pages: Live documentation

{% else %}**Status**: Documentation pipeline is disabled

To enable MkDocs documentation:
1. Set `use_mkdocs` to "yes" in `cookiecutter.json`
2. Install MkDocs dependencies: `poetry add mkdocs --group dev`
3. Create `mkdocs.yml` in project root
4. Add documentation files to `docs/` directory

{% endif %}

---

### 4. PyPI Publishing (`publish-pypi.yml`)

**Purpose**: Release automation to PyPI and TestPyPI using Trusted Publishing (OIDC)

**Jobs**:
- **build**: Build source distribution and wheel
- **publish-to-pypi**: Publish to production PyPI (on releases)
- **publish-to-testpypi**: Publish to TestPyPI (manual workflow dispatch)

**Triggers**:
- Release creation (type: published)
- Manual workflow dispatch with `use_testpypi` input

**Setup Instructions**:

1. **Configure PyPI Trusted Publisher**:
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new publisher"
   - Fill in:
     - **PyPI Project Name**: `{% raw %}{{ cookiecutter.pypi_package_name }}{% endraw %}`
     - **Owner**: `{% raw %}{{ cookiecutter.github_org_or_user }}{% endraw %}`
     - **Repository**: `{% raw %}{{ cookiecutter.project_slug }}{% endraw %}`
     - **Workflow**: `publish-pypi.yml`
   - Click "Add"

2. **Configure TestPyPI Trusted Publisher** (Optional):
   - Same steps as above, but at https://test.pypi.org/manage/account/publishing/

3. **Create a Release**:
   - Tag commit: `git tag v1.0.0`
   - Push tag: `git push origin v1.0.0`
   - Create GitHub Release from tag
   - Workflow automatically publishes to PyPI

**Features**:
- Trusted Publishing (OIDC) - No hardcoded API tokens
- Package verification with twine
- Artifact caching (5 days)
- Separate TestPyPI and PyPI jobs
- Automatic GitHub Step Summary with install instructions

**Key Outputs**:
- `python-package-distributions/` - Built artifacts (wheel + sdist)
- GitHub Release notes with installation instructions

---

## Cookiecutter Variables

All workflows use Jinja2 template variables that are automatically filled during project generation:

### Configuration Variables:

| Variable | Used In | Purpose | Default |
|----------|---------|---------|---------|
| `{% raw %}{{ cookiecutter.project_name }}{% endraw %}` | All | Project display name | Your Project |
| `{% raw %}{{ cookiecutter.project_slug }}{% endraw %}` | All | GitHub repository name | from project_name |
| `{% raw %}{{ cookiecutter.python_version }}{% endraw %}` | All | Python version for tests | 3.12 |
| `{% raw %}{{ cookiecutter.github_org_or_user }}{% endraw %}` | publish-pypi | GitHub org/user for PyPI | Your GitHub username |
| `{% raw %}{{ cookiecutter.pypi_package_name }}{% endraw %}` | publish-pypi | PyPI package name | project_slug with underscores |
| `{% raw %}{{ cookiecutter.repo_url }}{% endraw %}` | docs | Repository URL | GitHub URL |

### Feature Flags:

| Variable | Used In | Purpose | Type |
|----------|---------|---------|------|
| `include_github_actions` | All | Enable GitHub Actions workflows | yes/no |
| `include_security_scanning` | security-analysis.yml | Enable security scanning | yes/no |
| `include_codecov` | ci.yml | Enable Codecov uploads | yes/no |
| `use_ruff` | ci.yml | Enable Ruff linting | yes/no |
| `use_mypy` | ci.yml | Enable MyPy type checking | yes/no |
| `use_mkdocs` | docs.yml | Enable MkDocs documentation | yes/no |

### Quality Targets:

| Variable | Used In | Default |
|----------|---------|---------|
| `code_coverage_target` | ci.yml | 80 |
| `docstring_coverage_target` | docs.yml | 85 |

---

## Security Features

### Step Security Hardening

All workflows include **step-security/harden-runner** for supply chain security:

```yaml
- name: Harden the runner
  uses: step-security/harden-runner@v2.10.1
  with:
    egress-policy: audit  # Audit all outbound network calls
```

This prevents:
- Exfiltration of secrets via network calls
- Unauthorized dependency downloads
- Supply chain attacks

### Permissions Model

All workflows follow the **principle of least privilege**:

```yaml
permissions: read-all  # Default for most jobs
```

Specific permissions are granted only when needed:
- `id-token: write` - For OIDC (Trusted Publishing)
- `security-events: write` - For CodeQL uploads
- `pull-requests: write` - For Dependency Review comments
- `contents: write` - For GitHub Pages deployment

### Security Scanning

{% if cookiecutter.include_security_scanning == "yes" %}The security-analysis workflow provides:
- **CodeQL**: Advanced static analysis with GitHub's vulnerability patterns
- **Bandit**: Python-specific security scanning
- **Safety**: Known vulnerability detection in dependencies
- **OSV Scanner**: Multi-ecosystem vulnerability matching

All findings are available in GitHub Security tab.
{% else %}Security scanning is currently disabled. Enable by setting `include_security_scanning` to "yes".
{% endif %}

---

## Concurrency & Performance

### Concurrency Control

**CI Pipeline** (`ci.yml`):
```yaml
concurrency:
  group: ci-{% raw %}${{ github.ref }}{% endraw %}
  cancel-in-progress: true
```
- Only one CI run per branch
- Older runs are cancelled when new commits arrive
- Reduces resource waste and execution time

### Caching Strategy

**Poetry Dependencies**:
- Cache key: `py-deps-{% raw %}{{ runner.os }}-{{ poetry.lock hash }}{% endraw %}`
- Scope: `.venv` + `~/.cache/pypoetry`
- Hit rate: ~95% for unchanged dependencies

**Benefits**:
- CI time reduced by ~50% on cache hit
- Reduced PyPI API calls
- Faster local development setup

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| CI (cache hit) | ~3-5 min | Most common case |
| CI (cache miss) | ~8-12 min | After major dependency update |
| Security scan | ~5-10 min | Runs in parallel with CI |
| Docs build | ~2-3 min | Only on doc changes |
| PyPI publish | ~2-3 min | On release |

---

## Customization Guide

### Adding Custom Jobs

Example: Add a custom linting step to ci.yml

```yaml
- name: Custom Linter
  run: |
    poetry run your-linter src/
```

Insert into the `quality-checks` job.

### Modifying Thresholds

**Coverage threshold** (in ci.yml):
```yaml
--cov-fail-under={% raw %}{{ cookiecutter.code_coverage_target }}{% endraw %}
```
Edit `cookiecutter.json` or the workflow directly.

**Docstring coverage target** (in docs.yml):
```yaml
# target: {% raw %}{{ cookiecutter.docstring_coverage_target }}{% endraw %}%
```

### Disabling Workflows

Set feature flags in `cookiecutter.json` before running cookiecutter:
- `"include_security_scanning": "no"` - Disables security-analysis.yml
- `"use_mkdocs": "no"` - Disables docs.yml
- `"use_mypy": "no"` - Disables MyPy checks
- `"use_ruff": "no"` - Disables Ruff checks

---

## Troubleshooting

### CI Pipeline Failures

**Coverage below threshold**:
```bash
poetry run pytest --cov=src --cov-report=term-missing
# Identify uncovered lines and add tests
```

**MyPy errors**:
```bash
poetry run mypy src --config-file=pyproject.toml
# Fix type annotations in reported files
```

**Ruff formatting issues**:
```bash
poetry run ruff format src tests
poetry run ruff check --fix src tests
```

### Security Scan Failures

**OSV Scanner - HIGH/CRITICAL vulnerabilities**:
1. Check `osv-results.json` artifact for details
2. Update vulnerable package: `poetry update package-name`
3. If unfixable, add exception to `osv-scanner.toml`

**CodeQL findings**:
1. Review findings in GitHub Security tab
2. Evaluate severity and false positive rate
3. Fix critical/high severity issues

### Documentation Build Failures

**Front matter validation**:
```bash
poetry run python tools/validate_front_matter.py docs --fix
```

**Broken links**:
1. Review `link-check-report/` artifact
2. Update links in markdown files
3. Note: External link timeouts are not failures

### PyPI Publishing Issues

**Trusted Publisher not configured**:
1. Ensure PyPI publisher is configured (see setup instructions)
2. Repository name must exactly match
3. Workflow name must be `publish-pypi.yml`

**Package already exists**:
- Version must be unique
- Use semantic versioning (e.g., v1.0.1)

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [step-security/harden-runner](https://github.com/step-security/harden-runner)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [CodeQL Documentation](https://codeql.github.com/)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [Poetry Documentation](https://python-poetry.org/)

---

## License

These workflows are part of `{% raw %}{{ cookiecutter.project_name }}{% endraw %}` and follow the same license.

For questions or issues, refer to the project's contributing guidelines or create an issue on GitHub.
