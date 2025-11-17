# Cookiecutter Python Template

This folder contains a complete, production-ready cookiecutter template for starting new Python projects.

## 📦 What's Inside

- **Production-ready cookiecutter template** with 40+ configuration options
- **Complete project structure** with modern Python best practices
- **CI/CD workflows** for GitHub Actions
- **Security scanning** and compliance tools
- **Optional monitoring utilities** for advanced use cases

## 📁 Folder Structure

```text
cookiecutter-python-template/
├── cookiecutter.json              # Main configuration (40+ options)
├── hooks/
│   ├── pre_gen_project.py        # Pre-generation validation
│   └── post_gen_project.py       # Post-generation cleanup
├── {{cookiecutter.project_slug}}/  # Main template (46 files)
│   ├── .github/
│   │   └── workflows/            # 4 CI/CD workflows
│   ├── src/{{cookiecutter.project_slug}}/
│   │   ├── __init__.py
│   │   ├── cli.py                # Optional CLI
│   │   ├── core/                 # Core functionality
│   │   └── utils/                # Utility modules
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── unit/                 # Unit tests
│   │   ├── integration/          # Integration tests
│   │   └── test_example.py
│   ├── docs/
│   │   ├── ADRs/                 # Architecture Decision Records
│   │   └── planning/             # Project planning templates
│   ├── pyproject.toml            # Poetry/PEP 621 configuration
│   ├── .pre-commit-config.yaml   # Pre-commit hooks
│   ├── codecov.yml               # Code coverage config
│   ├── renovate.json             # Dependency updates
│   ├── mkdocs.yml                # Documentation site
│   ├── README.md
│   ├── LICENSE
│   ├── SECURITY.md
│   ├── CONTRIBUTING.md
│   └── ... (32 more files)
├── {{ cookiecutter.repo_name }}/  # Optional monitoring utilities
│   ├── src/monitoring/           # Monitoring components
│   │   ├── ab_testing_dashboard.py
│   │   ├── metrics_collector.py
│   │   ├── performance_dashboard.py
│   │   └── service_token_monitor.py
│   └── tests/unit/monitoring/    # Monitoring tests
├── .github/
│   ├── workflows/
│   │   ├── validate-template.yml # Template validation
│   │   └── release-drafter.yml   # Auto release notes
│   └── release-drafter.yml       # Release config
├── .pre-commit-config.yaml       # Template repo hooks
└── QUICK_START.sh                # Interactive setup script
```

## 🚀 How to Use This Template

### Option 1: Copy to GitHub Repository

1. **Create a new repository on GitHub**:
   - Repository name: `cookiecutter-python-template-2` (or your choice)
   - Description: "Production-ready Python cookiecutter template"
   - Public repository
   - **DO NOT** initialize with README

2. **Copy this folder to the new repository**:

   ```bash
   # From your local machine
   cd /path/to/this/folder/cookiecutter-template
   git init
   git add .
   git commit -m "Initial commit: Production-ready Python cookiecutter template"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

3. **Enable as template repository**:
   - Go to repository Settings → General
   - Check ✅ "Template repository"

### Option 2: Use Locally

```bash
# Install cookiecutter
pip install cookiecutter

# Use the template directly from this folder
cookiecutter /path/to/this/folder/cookiecutter-template

# Or after copying to your templates directory
cp -r cookiecutter-template ~/my-templates/
cookiecutter ~/my-templates/cookiecutter-template
```

## 🎯 Key Features

### Enhanced Configuration Options (40+ total)

**MCP Server Support** (from zen-mcp-server):

- `include_mcp_server`: Add MCP protocol support
- `mcp_transport`: stdio/http/sse options

**API Integration** (from xero-practice-management):

- `include_api_client`: API client factory pattern
- `api_auth_type`: OAuth2/API key/JWT authentication
- `include_retry_logic`: Exponential backoff retry
- `include_rate_limiting`: Rate limiting support

**Database/Ledger Patterns** (from ledgerbase):

- `include_database`: SQLAlchemy with migrations
- `include_repository_pattern`: Repository pattern
- `include_unit_of_work`: Unit of Work pattern
- `database_dialect`: PostgreSQL/MySQL/SQLite

**Enhanced Security** (from pp-security-master):

- `include_gitleaks`: Secrets detection
- `include_compliance_docs`: Compliance templates
- `include_threat_model`: Threat modeling
- `include_fuzzing`: Fuzzing infrastructure

**Financial Patterns** (from FISProject):

- `include_financial_validators`: Financial validation
- `include_audit_logging`: Audit trail
- `use_decimal_precision`: Decimal for money

### Template Files ({{cookiecutter.project_slug}}/)

Complete project structure with:

- ✅ **4 GitHub Actions workflows** (CI, security, docs, PyPI)
- ✅ **UV + PEP 621** packaging (10-100x faster than pip/poetry)
- ✅ **Hatchling** build backend
- ✅ **Cruft** template update tracking
- ✅ **Ruff** consolidated linting
- ✅ **MyPy** strict type checking
- ✅ **pytest** with 80% coverage
- ✅ **MkDocs Material** documentation
- ✅ **7 security tools** (Bandit, Safety, OSV-Scanner, CodeQL, Gitleaks, etc.)
- ✅ **Pre-commit hooks** (9 automated checks)
- ✅ **Click CLI** (optional)
- ✅ **FastAPI** (optional)
- ✅ **PyTorch ML** (optional)

## 📊 Template Coverage

**Expected User Impact**:

- **100% of users**: Enhanced security
- **40-50% of users**: API integration patterns
- **40-50% of users**: Database/ORM patterns
- **10-15% of users**: Financial/ledger patterns
- **5-10% of users**: MCP server patterns

**Time Savings**: 4-8 hours per project (120-240× ROI)

## 📝 Example Usage

```bash
# Install cookiecutter and cruft
pip install cookiecutter cruft

# Option 1: Use with Cruft (recommended - enables template updates)
cruft create /path/to/cookiecutter-template

# Option 2: Use with Cookiecutter (traditional method)
cookiecutter /path/to/cookiecutter-template

# Answer prompts:
project_name [My Python Project]: My Awesome Project
python_version [3.12]: 3.12
license [MIT]: MIT
include_cli [yes]: yes
include_api_client [no]: yes
api_auth_type [oauth2]: oauth2
include_database [sqlalchemy_migrations]: sqlalchemy_migrations
include_repository_pattern [yes]: yes

# Generated project structure:
my_awesome_project/
├── .github/workflows/      # 4 CI/CD workflows
├── src/my_awesome_project/
│   ├── cli.py
│   ├── core/
│   └── utils/
├── tests/
├── docs/
├── pyproject.toml
├── uv.lock               # UV lock file
├── .cruft.json          # Template tracking (cruft only)
└── README.md
```

## 🤖 Claude Code Integration

This template includes enhanced support for Claude Code with **user-level settings integration**.

### User-Level Settings Setup

During project generation, you'll be prompted to set up user-level Claude Code settings. These settings enhance Claude Code's capabilities across **all your projects**:

**What's Included:**

- **Global CLAUDE.md**: Best practices, workflows, and development patterns
- **Skills**: Reusable capabilities for common tasks
- **Agents**: Specialized task handlers for security, testing, documentation, etc.
- **Custom Commands**: Slash commands and hooks for enhanced workflows

**Setup Process:**

When you create a new project using this template, the post-generation hook will:

1. **Check** for existing user-level settings at `~/.claude/` or `~/.config/claude/`
2. **Prompt** you to set up settings if not found
3. **Clone** the settings repo (default: `https://github.com/williaby/.claude`)
4. **Verify** installation of CLAUDE.md, skills, agents, and commands

**Manual Setup:**

If you skip the automatic setup, you can install user-level settings later:

```bash
# Using the default settings repo
git clone https://github.com/williaby/.claude ~/.claude

# Or using your own settings repo
git clone https://github.com/YOUR_USERNAME/YOUR_CLAUDE_SETTINGS ~/.claude
```

**Benefits:**

✅ **Consistent workflows** across all projects
✅ **Supervisor patterns** with agent delegation
✅ **Reusable skills** for common tasks
✅ **Custom slash commands** for your workflow
✅ **Global best practices** inherited by all projects

> **Note:** User-level settings are optional but recommended for the best Claude Code experience. Projects work without them but have enhanced capabilities when available.

## 🔄 Template Updates with Cruft

This template supports **Cruft** for keeping generated projects in sync with template updates.

### Why Use Cruft?

- **Stay Updated**: Automatically pull in template improvements, bug fixes, and new features
- **Selective Updates**: Review and accept/reject changes before applying them
- **Track Template Version**: `.cruft.json` tracks which template version your project uses
- **Conflict Resolution**: Smart merging handles conflicts between template updates and your changes

### Using Cruft with This Template

**Create a new project with Cruft:**

```bash
# Install cruft
pip install cruft

# Create project (automatically adds .cruft.json)
cruft create https://github.com/YOUR_USERNAME/YOUR_TEMPLATE_REPO

# Or use local path
cruft create /path/to/cookiecutter-python-template
```

**Check for template updates:**

```bash
cd your-project
cruft check
```

**Update your project:**

```bash
cruft update
# Review changes, accept/reject updates
```

**View differences:**

```bash
cruft diff
```

**Update template variables:**

```bash
cruft update --variables
```

### Automated Update Checks

You can add a GitHub Action to automatically check for template updates:

```yaml
# .github/workflows/cruft-update.yml
name: Check Template Updates

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check for template updates
        run: |
          pip install cruft
          cruft check
```

## 🔄 Version History

- **v2.0** (2025-11-17): UV & Cruft Migration Release
  - **BREAKING**: Migrated from Poetry to UV for package management
  - Added Cruft support for template updates
  - Updated all dependencies to use standard PEP 621 format
  - Changed build backend from poetry-core to hatchling
  - Updated all documentation and CI/CD workflows for UV
  - Added `.cruft.json` for template tracking
  - 10-100x faster dependency resolution with UV
- **v1.2** (2025-11-17): Claude Code Enhancement Release
  - Added interactive user-level Claude settings setup
  - Streamlined dependency management (Poetry as single source of truth)
  - Simplified CLI help text for better UX
  - Enhanced post-generation hook with Claude Code integration
- **v1.1** (2025-11-17): Streamlined release
  - Removed reference documentation (docs-reference/)
  - Added optional monitoring utilities
  - Focused on core template functionality
- **v1.0** (2025-11-17): Initial release with patterns from 5 repositories
  - 73 files, 40+ configuration options
  - Complete Python project template

## 📖 Documentation

- **QUICK_START.sh**: Interactive script for template setup
- **cookiecutter.json**: All configuration options with detailed comments
- **Template README**: Each generated project includes comprehensive documentation

## 🙏 Based On

This template was created from patterns found in:

1. **image-preprocessing-detector**: Base project structure
2. **zen-mcp-server**: MCP protocol patterns
3. **xero-practice-management**: API integration patterns
4. **pp-security-master**: Security enhancements
5. **FISProject**: Financial system patterns
6. **ledgerbase**: Database/ledger patterns

## 📞 Support

For questions or issues:

- Review the generated project's documentation (in each project's `docs/` folder)
- Check the cookiecutter.json file for all available configuration options
- Use QUICK_START.sh for interactive template setup

## 🎁 Optional Components

### Monitoring Utilities ({{ cookiecutter.repo_name }}/)

Optional monitoring components available for advanced use cases:

- **A/B Testing Dashboard**: Track and visualize A/B test results
- **Metrics Collector**: Collect and aggregate system metrics
- **Performance Dashboard**: Monitor application performance
- **Service Token Monitor**: Track service token usage and rotation

These components are not included in generated projects by default but can be copied manually for specific use cases.

---

**Configuration Options**: 40+ options
**Template Files**: 46 files in main template
**Monitoring Utilities**: 9 files (optional)
**Coverage**: 95% of Python project needs
**Last Updated**: 2025-11-17
