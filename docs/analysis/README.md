# Cookiecutter Python Template

A comprehensive, production-ready Python project template based on best practices from real-world projects. Features modern tooling, extensive CI/CD pipelines, security scanning, and professional documentation infrastructure.

## 🎯 Features

### Core Infrastructure
- **📦 Modern Package Management**: Poetry with PEP 621 compliance
- **🧪 Comprehensive Testing**: pytest with coverage (configurable threshold), parallel execution, property-based testing
- **🔍 Code Quality**: Ruff (unified linting & formatting), MyPy (strict type checking), interrogate (docstring coverage)
- **🔒 Security First**: Bandit, Safety, OSV-Scanner, CodeQL, Dependabot, REUSE compliance
- **📚 Professional Documentation**: MkDocs Material theme, auto-generated API docs, ADR templates, project plan templates
- **🚀 CI/CD Ready**: 4 comprehensive GitHub Actions workflows with StepSecurity hardening

### Optional Features (Configurable)
- **🖥️ CLI Framework**: Click-based command-line interface
- **🤖 ML Dependencies**: PyTorch, NumPy, scikit-learn stack
- **🌐 API Framework**: FastAPI with Uvicorn
- **🔄 Automation**: Nox for local task automation
- **🐳 Containerization**: Docker support (optional)

### Development Tools
- **Pre-commit Hooks**: Automated code quality checks before every commit
- **Front Matter Validation**: Pydantic-based documentation metadata validation
- **Renovate Integration**: Automated dependency updates with security prioritization
- **Component-Based Coverage**: Track coverage by module with Codecov integration
- **SBOM Generation**: CycloneDX software bill of materials
- **Mutation Testing**: mutmut for test quality assessment

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (3.12 recommended)
- [Cookiecutter](https://github.com/cookiecutter/cookiecutter) >= 2.0.0
- [Poetry](https://python-poetry.org/) >= 1.5.0 (for generated projects)
- Git

```bash
# Install cookiecutter
pip install cookiecutter

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -
```

### Generate a New Project

```bash
# From GitHub (recommended)
cookiecutter gh:williaby/cookiecutter-python-template

# From local directory
cookiecutter /path/to/cookiecutter-python-template

# With defaults file
cookiecutter gh:williaby/cookiecutter-python-template --config-file my-defaults.yaml
```

### Configuration Prompts

You'll be prompted for the following information:

| Prompt | Description | Example |
|--------|-------------|---------|
| **project_name** | Human-readable project name | "My Python Project" |
| **project_slug** | Python package name (snake_case) | "my_python_project" |
| **project_short_description** | One-line project description | "A short description" |
| **author_name** | Your full name | "Jane Doe" |
| **author_email** | Your email address | "jane@example.com" |
| **github_username** | Your GitHub username | "janedoe" |
| **version** | Initial version number | "0.1.0" |
| **python_version** | Python version (3.11/3.12/3.13) | "3.12" |
| **license** | License type | "MIT" |
| **include_cli** | Add Click CLI framework? | "yes" |
| **include_ml_dependencies** | Add ML/DL libraries? | "no" |
| **include_api_framework** | Add FastAPI framework? | "no" |
| **use_mkdocs** | Use MkDocs for documentation? | "yes" |
| **code_coverage_target** | Minimum coverage percentage | "80" |
| **docstring_coverage_target** | Minimum docstring coverage | "85" |
| **include_github_actions** | Generate CI/CD workflows? | "yes" |
| **include_security_scanning** | Include security scanning? | "yes" |
| **include_nox** | Include Nox automation? | "yes" |

## 📋 What Gets Generated

### Project Structure

```
my_python_project/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Main CI pipeline
│   │   ├── security-analysis.yml     # Security scanning
│   │   ├── docs.yml                  # Documentation
│   │   └── publish-pypi.yml          # Release automation
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── docs/
│   ├── guides/                       # User guides
│   ├── api/                          # API reference (auto-generated)
│   ├── development/                  # Developer docs
│   ├── ADRs/                         # Architecture Decision Records
│   │   ├── README.md
│   │   └── adr-template.md
│   ├── planning/
│   │   └── project-plan-template.md
│   └── _data/
│       ├── tags.yml                  # Allowed doc tags
│       └── owners.yml                # Doc ownership
├── src/
│   └── my_python_project/
│       ├── __init__.py
│       ├── cli.py                    # CLI entry point (optional)
│       ├── core/
│       │   └── config.py             # Configuration
│       └── utils/
│           └── logging.py            # Structured logging
├── tests/
│   ├── conftest.py                   # Pytest configuration
│   ├── unit/
│   ├── integration/
│   └── test_example.py
├── tools/
│   ├── validate_front_matter.py     # Doc validation
│   └── frontmatter_contract/        # Pydantic models
├── .pre-commit-config.yaml
├── pyproject.toml                    # PEP 621 + Poetry
├── poetry.lock
├── codecov.yml                       # Coverage config
├── renovate.json                     # Dependency updates
├── osv-scanner.toml                  # CVE exceptions
├── REUSE.toml                        # License compliance
├── mkdocs.yml                        # Documentation config
├── noxfile.py                        # Automation tasks
├── CLAUDE.md                         # Claude Code guidance
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
└── LICENSE
```

### GitHub Actions Workflows

**ci.yml** - Main CI Pipeline (~12 minutes):
- Setup job with Poetry caching
- Test job with pytest and coverage
- Quality checks (Ruff, MyPy)
- CI gate for branch protection

**security-analysis.yml** - Security Scanning (~20 minutes):
- CodeQL analysis (security + quality)
- Bandit security scanning
- Safety dependency CVE scanning
- OSV-Scanner vulnerability detection

**docs.yml** - Documentation Pipeline:
- Front matter validation
- Docstring coverage checks
- MkDocs build in strict mode
- Lychee link checking
- GitHub Pages deployment

**publish-pypi.yml** - Release Automation:
- Trusted Publisher (OIDC) setup
- TestPyPI and PyPI publishing
- Artifact verification

## 🛠️ Post-Generation Steps

After generating your project:

### 1. Navigate to Project

```bash
cd my_python_project
```

### 2. Install Dependencies

```bash
# Install core + dev dependencies
poetry install --with dev

# With ML dependencies
poetry install --with dev,ml

# With API dependencies
poetry install --with dev,api
```

### 3. Setup Pre-commit Hooks

```bash
poetry run pre-commit install
```

### 4. Run Tests

```bash
# All tests with coverage
poetry run pytest -v

# Fast tests only (exclude slow)
poetry run pytest -v -m "not slow"

# Parallel execution
poetry run pytest -v -n auto
```

### 5. Build Documentation (if MkDocs enabled)

```bash
# Build docs
poetry run mkdocs build

# Serve locally
poetry run mkdocs serve
# Open http://127.0.0.1:8000
```

### 6. Setup GitHub Repository

```bash
# Initialize git (done automatically by hook)
git remote add origin https://github.com/yourusername/my_python_project.git
git push -u origin main

# Or use GitHub CLI
gh repo create my_python_project --public --source=. --remote=origin --push
```

### 7. Configure GitHub Settings

**Branch Protection** (for main branch):
- Require PR before merging
- Require status checks: `ci-gate`, `security-analysis`, `docs-build`
- Require conversation resolution

**GitHub Apps to Install**:
- **Renovate**: Automated dependency updates
- **Codecov**: Coverage reporting (if `include_codecov == "yes"`)
- **Dependabot**: Security alerts (already configured)

**PyPI Trusted Publisher** (for releases):
1. Go to https://pypi.org/manage/account/publishing/
2. Add publisher: `your-username/my_python_project` with workflow `publish-pypi.yml`
3. Create release on GitHub to trigger publish

## 🔧 Configuration Files Explained

### pyproject.toml
- **Dual Format**: PEP 621 + Poetry for maximum compatibility
- **Consolidated Tooling**: Ruff replaces Black, pydocstyle, and partial Bandit
- **Strict MyPy**: Type checking on `src/`, relaxed on `tests/`
- **Coverage Enforcement**: Configurable threshold (default 80%)
- **Docstring Coverage**: interrogate with configurable target (default 85%)
- **Mutation Testing**: mutmut configuration for test quality

### .pre-commit-config.yaml
- **9 Hooks**: File checks, Ruff format/lint, MyPy, Markdown/YAML linting, interrogate, Bandit, Safety
- **Auto-update**: Monthly via pre-commit.ci
- **Local Hooks**: Custom front matter validation (if MkDocs enabled)

### codecov.yml
- **Component Tracking**: Separate coverage for each module
- **Multi-Flag**: Track unit vs integration test coverage
- **Quality Range**: 70-95% with configurable thresholds
- **PR Comments**: Automatic coverage reports on pull requests

### renovate.json
- **Smart Grouping**: Dependencies grouped by type (runtime, dev, actions)
- **Security First**: Critical updates flagged and prioritized
- **Auto-merge**: Safe minor/patch updates for GitHub Actions
- **Lock File Maintenance**: Weekly lock file updates

### mkdocs.yml (if enabled)
- **Material Theme**: Modern, responsive with dark/light modes
- **10+ Plugins**: Search, git revision dates, auto-generated API docs
- **15+ Extensions**: Admonitions, code highlighting, tables, footnotes
- **Google-Style Docstrings**: Automatic API documentation with griffe-pydantic

### noxfile.py (if enabled)
- **10 Sessions**: fm, docs, serve, docstrings, validate, reuse, reuse_spdx, sbom, scan, compliance
- **Docker Integration**: REUSE compliance and Trivy vulnerability scanning
- **SBOM Generation**: CycloneDX format for supply chain security

## 📚 Key Technologies

### Development
- **Poetry**: Dependency management and packaging
- **Ruff**: Ultra-fast linting and formatting (10-100× faster than traditional tools)
- **MyPy**: Static type checking with strict mode
- **pytest**: Testing framework with fixtures and plugins
- **pre-commit**: Git hook automation

### Documentation
- **MkDocs**: Static site generator
- **Material Theme**: Beautiful, responsive documentation
- **mkdocstrings**: Automatic API reference from docstrings
- **griffe-pydantic**: Pydantic model documentation

### Security
- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanner
- **OSV-Scanner**: Multi-ecosystem vulnerability detection
- **CodeQL**: Advanced semantic code analysis

### CI/CD
- **GitHub Actions**: Workflow automation
- **StepSecurity**: Workflow hardening
- **Codecov**: Coverage reporting
- **Trusted Publisher**: Secure PyPI publishing with OIDC

## 🎨 Customization

### Using a Defaults File

Create `my-defaults.yaml`:

```yaml
default_context:
  author_name: "Jane Doe"
  author_email: "jane@example.com"
  github_username: "janedoe"
  python_version: "3.12"
  license: "MIT"
  include_cli: "yes"
  use_mkdocs: "yes"
  code_coverage_target: "85"
```

Then generate:

```bash
cookiecutter gh:williaby/cookiecutter-python-template --config-file my-defaults.yaml
```

### Customizing Generated Projects

After generation, customize:

1. **Update Dependencies**: Edit `pyproject.toml` → `[project.dependencies]`
2. **Adjust Coverage**: Edit `pyproject.toml` → `[tool.pytest.ini_options]` → `--cov-fail-under`
3. **Configure Ruff**: Edit `pyproject.toml` → `[tool.ruff.lint]` → `select`/`ignore`
4. **Add MkDocs Plugins**: Edit `mkdocs.yml` → `plugins`
5. **Customize Workflows**: Edit `.github/workflows/*.yml`

## 🤝 Contributing

Contributions are welcome! To contribute to this cookiecutter template:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Test by generating a project: `cookiecutter /path/to/your-fork`
5. Commit with conventional commits: `git commit -m "feat: add new feature"`
6. Push and create a Pull Request

## 📖 Documentation

- **CLAUDE.md**: Guidance for Claude Code when working with generated projects
- **ADR Templates**: Architecture Decision Record format and examples
- **Project Plan Template**: Comprehensive project planning structure
- **Generated README**: Each project gets a customized README with badges and instructions

## 🔗 Related Projects

This template is based on patterns from:
- [image-preprocessing-detector](https://github.com/williaby/image-preprocessing-detector): Production RAG document processing pipeline

## 📄 License

This template is licensed under MIT License. Generated projects will use the license you select during generation.

## 🙏 Acknowledgments

- Inspired by production Python projects and industry best practices
- Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter)
- Based on real-world patterns from the image-preprocessing-detector project

---

**Generated with ❤️ by Cookiecutter Python Template**
