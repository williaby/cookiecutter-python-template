# Template Improvements Based on image_detection Project

**Analysis Date**: 2025-11-19
**Source Project**: /home/byron/dev/image_detection
**Purpose**: Identify configuration patterns and best practices from a production project to improve the template

---

## Executive Summary

The image_detection project demonstrates several advanced configuration patterns that should be incorporated into the cookiecutter-python-template. This analysis identifies **15 key improvements** across 5 categories.

**Priority Distribution:**
- 🔴 **Critical** (3): Should be added immediately
- 🟡 **High** (6): Significant value, add soon
- 🟢 **Medium** (4): Nice to have, add when convenient
- ⚪ **Low** (2): Optional enhancements

---

## 1. Nox Automation Patterns (🔴 Critical)

### Current State
**Template**: Includes `include_nox` option but minimal session configuration
**Image Detection**: Comprehensive nox sessions for documentation, compliance, and SBOM generation

### Recommended Additions

#### 1.1 Documentation Workflow Sessions

```python
# noxfile.py template additions

@nox.session(python="{{ cookiecutter.python_version }}")
def fm(session: nox.Session) -> None:
    """Validate and autofix front matter in documentation."""
    session.install("pydantic>=2.0", "python-frontmatter>=1.1", "ruamel.yaml>=0.18")
    session.run("python", "tools/validate_front_matter.py", "docs", "--fix")

@nox.session(python="{{ cookiecutter.python_version }}")
def docs(session: nox.Session) -> None:
    """Build documentation with MkDocs."""
    session.install("-e", ".[dev]")
    session.run("mkdocs", "build", "--strict")

@nox.session(python="{{ cookiecutter.python_version }}")
def serve(session: nox.Session) -> None:
    """Serve documentation locally for development."""
    session.install("-e", ".[dev]")
    session.run("mkdocs", "serve")

@nox.session(python="{{ cookiecutter.python_version }}")
def docstrings(session: nox.Session) -> None:
    """Check docstring coverage with interrogate."""
    session.install("interrogate>=1.7")
    session.run("interrogate", "-c", "pyproject.toml", "src/")
```

**Priority**: 🔴 **Critical** (documentation is core to open source projects)

**Files to Add/Modify**:
- `{{cookiecutter.project_slug}}/noxfile.py` (expand sessions)
- `{{cookiecutter.project_slug}}/tools/validate_front_matter.py` (new file)

---

#### 1.2 Compliance and Security Sessions

```python
@nox.session(python="{{ cookiecutter.python_version }}")
def reuse(session: nox.Session) -> None:
    """Check REUSE compliance."""
    session.run(
        "docker", "run", "--rm", "--volume", f"{session.posargs[0] if session.posargs else '.'}:/data",
        "fsfe/reuse:latest", "lint", external=True,
    )

@nox.session(python="{{ cookiecutter.python_version }}")
def sbom(session: nox.Session) -> None:
    """Generate CycloneDX SBOM."""
    session.install("cyclonedx-bom==4.6.1")
    # Runtime SBOM
    session.run("cyclonedx-py", "uv", "--of", "json", "-o", "sbom-runtime.json", "--no-dev")
    # Development SBOM
    session.run("cyclonedx-py", "uv", "--of", "json", "-o", "sbom-dev.json", "--only", "dev")
    # Complete SBOM
    session.run("cyclonedx-py", "uv", "--of", "json", "-o", "sbom-complete.json")

@nox.session(python="{{ cookiecutter.python_version }}")
def scan(session: nox.Session) -> None:
    """Scan SBOM for vulnerabilities with Trivy."""
    import pathlib
    sbom_file = session.posargs[0] if session.posargs else "sbom-runtime.json"
    if not pathlib.Path(sbom_file).exists():
        session.error(f"SBOM file not found: {sbom_file}. Run 'nox -s sbom' first.")
    session.run(
        "docker", "run", "--rm", "--volume", f"{pathlib.Path().absolute()}:/workspace",
        "aquasec/trivy:latest", "sbom", f"/workspace/{sbom_file}",
        "--severity", "CRITICAL,HIGH", "--format", "table", external=True,
    )

@nox.session(python="{{ cookiecutter.python_version }}")
def compliance(session: nox.Session) -> None:
    """Run all compliance checks."""
    session.log("Running REUSE compliance check...")
    reuse(session)
    session.log("Generating SBOMs...")
    sbom(session)
    session.log("Scanning runtime SBOM for vulnerabilities...")
    scan(session)
    session.log("All compliance checks completed successfully!")
```

**Priority**: 🔴 **Critical** (OpenSSF requirements)

**Files to Add/Modify**:
- `{{cookiecutter.project_slug}}/noxfile.py` (expand sessions)
- Update `{{cookiecutter.project_slug}}/.github/workflows/security.yml` to use nox sessions

**Benefits**:
- Standardizes OpenSSF Scorecard compliance checks
- Automates SBOM generation (requirement for supply chain security)
- Makes compliance checks runnable locally (not just CI)

---

## 2. Renovate Configuration (🟡 High Priority)

### Current State
**Template**: Includes `include_renovate` option with basic config
**Image Detection**: Sophisticated dependency management with security prioritization

### Recommended Enhancements

#### 2.1 Security-First Package Rules

```json
{
  "packageRules": [
    {
      "description": "Critical security updates - immediate",
      "matchDatasources": ["pypi"],
      "matchCurrentVersion": "!/^0/",
      "matchSeverity": ["CRITICAL", "HIGH"],
      "labels": ["security", "critical", "dependencies"],
      "automerge": false,
      "prPriority": 10,
      "schedule": ["at any time"]
    },
    {
      "description": "Group Python dependencies by type",
      "matchManagers": ["poetry", "uv"],
      "matchDepTypes": ["dependencies"],
      "groupName": "Python dependencies",
      "automerge": false,
      "schedule": ["every weekend"]
    },
    {
      "description": "Auto-merge GitHub Actions minor/patch updates",
      "matchManagers": ["github-actions"],
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true,
      "automergeType": "pr",
      "automergeStrategy": "squash",
      "pinDigests": true
    }
  ],
  "osvVulnerabilityAlerts": true,
  "transitiveRemediation": true
}
```

**Priority**: 🟡 **High** (security and automation)

**Files to Modify**:
- `{{cookiecutter.project_slug}}/renovate.json` (expand configuration)

**Benefits**:
- Prioritizes security updates correctly
- Reduces manual PR management burden
- Pins GitHub Actions to commit SHA for security

---

## 3. Extended Ruff Configuration (🟡 High Priority)

### Current State
**Template**: Basic Ruff configuration (12 rule categories)
**Image Detection**: Comprehensive Ruff configuration (25+ rule categories)

### Missing Rule Categories

#### 3.1 Add Additional Linting Rules

```toml
[tool.ruff.lint]
select = [
    # Existing rules from template
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "ARG",  # flake8-unused-arguments
    "SIM",  # flake8-simplify
    "D",    # pydocstyle
    "S",    # flake8-bandit
    "T20",  # flake8-print
    "PT",   # flake8-pytest-style
    "RUF",  # Ruff-specific rules

    # NEW rules to add from image_detection
    "N",    # pep8-naming (enforce naming conventions)
    "A",    # flake8-builtins (prevent shadowing builtins)
    "DTZ",  # flake8-datetimez (timezone-aware datetimes)
    "PIE",  # flake8-pie (misc lints)
    "Q",    # flake8-quotes (consistent quote style)
    "RET",  # flake8-return (simplify return statements)
    "PTH",  # flake8-use-pathlib (prefer pathlib over os.path)
    "PERF", # Performance anti-patterns
    "FURB", # Refurb - modernization suggestions
    "LOG",  # Logging best practices
    "TRY",  # Exception handling best practices
    "ERA",  # Commented-out code detection
    "FBT",  # Boolean trap detection
    "ASYNC",# Async/await best practices
]
```

**Priority**: 🟡 **High** (code quality)

**Files to Modify**:
- `{{cookiecutter.project_slug}}/pyproject.toml` (add 11 new rule categories)

**Benefits**:
- Catches more bugs before they reach production
- Enforces modern Python best practices
- Prevents common security anti-patterns
- Improves async code quality

---

#### 3.2 Context-Specific Per-File Ignores

```toml
[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401", "D104"]  # Allow unused imports and missing docstrings
"tests/**/*.py" = ["ARG", "D", "S", "ERA"]  # Relaxed rules for tests
"scripts/**/*.py" = ["D", "T20", "TRY", "S", "PTH", "SIM", "DTZ"]  # CLI tools
"noxfile.py" = ["D", "T20"]  # Allow print in nox sessions
```

**Priority**: 🟡 **High** (developer experience)

**Benefits**:
- Reduces false positives
- Makes linting more practical for different contexts
- Improves developer productivity

---

## 4. OSV Scanner Configuration (🟢 Medium Priority)

### Current State
**Template**: Uses OSV Scanner in CI/CD
**Image Detection**: Has explicit vulnerability exceptions with detailed reasoning

### Recommended Addition

Create `{{cookiecutter.project_slug}}/osv-scanner.toml`:

```toml
# OSV-Scanner Vulnerability Exceptions
# Reference: https://google.github.io/osv-scanner/configuration/

# Example format:
# [[IgnoredVulns]]
# id = "PYSEC-XXXX-XXXXX"
# reason = "False positive explanation with verification details"

# Add project-specific exceptions here as needed
# Document why each vulnerability is ignored
# Include version verification and next review date
```

**Priority**: 🟢 **Medium** (transparency and documentation)

**Files to Add**:
- `{{cookiecutter.project_slug}}/osv-scanner.toml` (new file)
- Update `{{cookiecutter.project_slug}}/.github/workflows/security.yml` to use config file

**Benefits**:
- Documents why specific vulnerabilities are ignored
- Provides audit trail for security decisions
- Makes false positives easier to manage

---

## 5. Advanced Codecov Configuration (🟢 Medium Priority)

### Current State
**Template**: Basic codecov.yml
**Image Detection**: Component-based coverage tracking with multi-flag support

### Recommended Enhancements

```yaml
# Component-based coverage tracking
component_management:
  individual_components:
    - component_id: core
      name: Core Business Logic
      paths:
        - src/{{ cookiecutter.project_slug }}/core/**

    - component_id: api
      name: API Layer
      paths:
        - src/{{ cookiecutter.project_slug }}/api/**

    - component_id: utils
      name: Utilities
      paths:
        - src/{{ cookiecutter.project_slug }}/utils/**

# Multi-flag tracking for test types
flags:
  unit:
    carryforward: true
    paths:
      - src/{{ cookiecutter.project_slug }}/

  integration:
    carryforward: true
    paths:
      - src/{{ cookiecutter.project_slug }}/

# Pull request comment configuration
comment:
  layout: "reach, diff, flags, files"
  behavior: default
  require_changes: true
  require_base: yes
  require_head: yes

github_checks:
  annotations: true
```

**Priority**: 🟢 **Medium** (visibility and metrics)

**Files to Modify**:
- `{{cookiecutter.project_slug}}/codecov.yml` (expand configuration)

**Benefits**:
- Track coverage by component/module
- Distinguish unit vs integration test coverage
- Better PR review experience

---

## 6. Pre-commit Hook Enhancements (🟡 High Priority)

### Current State
**Template**: Uses qlty for unified checks
**Image Detection**: Individual hooks with safety, interrogate

### Analysis

**Good News**: Template's qlty approach is actually MORE modern than image_detection's individual hooks. The template uses:
- Unified qlty check (faster, single tool)
- OSV Scanner instead of safety (better, more comprehensive)
- Interrogate included in qlty configuration

**Recommendation**: ✅ **Template is already superior** - no changes needed. Image_detection should migrate TO the template approach.

---

## 7. Front Matter Validation Tool (🔴 Critical)

### Current State
**Template**: Missing
**Image Detection**: Has `tools/validate_front_matter.py` for documentation consistency

### Recommended Addition

Create `{{cookiecutter.project_slug}}/tools/validate_front_matter.py`:

```python
#!/usr/bin/env python3
"""Validate and autofix front matter in Markdown documentation files.

This tool ensures all documentation files have consistent YAML front matter
with required fields like title, description, and status.

Usage:
    python tools/validate_front_matter.py docs/          # Validate
    python tools/validate_front_matter.py docs/ --fix    # Auto-fix
    nox -s fm                                             # Via nox
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import frontmatter
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml import YAML

class FrontMatterSchema(BaseModel):
    """Required front matter schema."""

    title: str = Field(..., min_length=1, description="Document title")
    description: str = Field(None, description="Optional description")
    status: str = Field("active", description="Document status")
    tags: list[str] = Field(default_factory=list, description="Document tags")

def validate_file(file_path: Path, fix: bool = False) -> tuple[bool, str]:
    """Validate and optionally fix front matter in a Markdown file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            post = frontmatter.load(f)

        # Validate schema
        FrontMatterSchema(**post.metadata)
        return True, f"✅ {file_path.relative_to(Path.cwd())}"

    except ValidationError as e:
        if fix:
            # Auto-fix: Add missing fields with defaults
            post.metadata.setdefault("title", file_path.stem.replace("_", " ").title())
            post.metadata.setdefault("status", "active")
            post.metadata.setdefault("tags", [])

            with open(file_path, "w", encoding="utf-8") as f:
                frontmatter.dump(post, f)

            return True, f"🔧 {file_path.relative_to(Path.cwd())} (auto-fixed)"

        return False, f"❌ {file_path.relative_to(Path.cwd())}: {e}"

    except Exception as e:
        return False, f"❌ {file_path.relative_to(Path.cwd())}: {e}"

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate documentation front matter")
    parser.add_argument("path", type=Path, help="Path to docs directory")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        return 1

    markdown_files = list(args.path.rglob("*.md"))
    if not markdown_files:
        print(f"Warning: No Markdown files found in {args.path}")
        return 0

    results = [validate_file(f, args.fix) for f in markdown_files]
    successes = sum(1 for ok, _ in results if ok)

    for _, msg in results:
        print(msg)

    print(f"\n{successes}/{len(results)} files valid")
    return 0 if successes == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
```

**Priority**: 🔴 **Critical** (documentation quality)

**Files to Add**:
- `{{cookiecutter.project_slug}}/tools/validate_front_matter.py` (new file)
- Add to `pyproject.toml` dev dependencies: `python-frontmatter`, `ruamel.yaml`
- Add nox session (see section 1.1)

**Benefits**:
- Ensures consistent documentation metadata
- Catches missing or malformed front matter early
- Automated via nox and pre-commit

---

## 8. Enhanced Coverage Configuration (🟡 High Priority)

### Current State
**Template**: Basic coverage omit patterns
**Image Detection**: Extensive, well-documented omit patterns

### Recommended Additions

```toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    # Standard patterns (already in template)
    "*/tests/*",
    "*/__pycache__/*",

    # NEW: Additional practical patterns
    "validation/*",     # Validation/verification scripts
    "benchmarks/*",     # Benchmarking framework
    "data/*",           # Data processing scripts
    "scripts/*",        # Utility scripts
    "notebooks/*",      # Jupyter notebooks
    "tmp_cleanup/*",    # Temporary files
]

[tool.coverage.report]
exclude_lines = [
    # Standard patterns (already in template)
    "pragma: no cover",
    "if __name__ == .__main__.:",

    # NEW: Additional patterns
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
precision = 2
show_missing = true
```

**Priority**: 🟡 **High** (accurate metrics)

**Files to Modify**:
- `{{cookiecutter.project_slug}}/pyproject.toml` (expand coverage configuration)

**Benefits**:
- More accurate coverage metrics
- Excludes utility/script code appropriately
- Better reporting clarity

---

## 9. PyProject.toml Enhancements (🟢 Medium Priority)

### 9.1 Improved Dependency Organization

**Image Detection Pattern**:
```toml
[project.optional-dependencies]
dev = [...]
ml = [...]      # Optional ML dependencies
api = [...]     # Optional API dependencies
colab = [...]   # Environment-specific dependencies
```

**Recommendation**: Add conditional dependency group template

```toml
{% if cookiecutter.include_ml_dependencies == "yes" -%}
ml = [
    # Machine Learning dependencies
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
]
{% endif -%}

{% if cookiecutter.include_api_framework == "yes" -%}
api = [
    # API framework dependencies
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
]
{% endif -%}
```

**Priority**: 🟢 **Medium** (organization)

---

### 9.2 Enhanced MyPy Configuration

**Image Detection adds**:
```toml
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "scripts.*"
disallow_untyped_defs = false
check_untyped_defs = false
ignore_missing_imports = true
```

**Priority**: 🟢 **Medium** (practical type checking)

---

## 10. Project Metadata Files (⚪ Low Priority)

### Current State
**Image Detection includes**:
- `codemeta.json` - CodeMeta software metadata
- `CITATION.cff` - Citation file format
- `.zenodo.json` - Zenodo DOI metadata

**Recommendation**: Add as **optional** template features

```json
// cookiecutter.json additions
"include_citation_metadata": ["no", "yes"],
"include_codemeta": ["no", "yes"],
"include_zenodo": ["no", "yes"],
```

**Priority**: ⚪ **Low** (specialized use cases)

**Benefits**:
- Better academic/research project support
- Easier citation and attribution
- Integration with Zenodo for DOIs

---

## Implementation Checklist

### Phase 1: Critical Additions (Week 1)

- [ ] **1. Nox Sessions** (Section 1)
  - [ ] Add documentation workflow sessions
  - [ ] Add compliance sessions (REUSE, SBOM, scan)
  - [ ] Update noxfile.py template

- [ ] **2. Front Matter Validation** (Section 7)
  - [ ] Create `tools/validate_front_matter.py`
  - [ ] Add nox session
  - [ ] Add to pre-commit config (qlty integration)

- [ ] **3. Renovate Enhancements** (Section 2)
  - [ ] Add security-first package rules
  - [ ] Enable OSV vulnerability alerts
  - [ ] Pin GitHub Actions to SHA

### Phase 2: High Priority (Week 2)

- [ ] **4. Extended Ruff Rules** (Section 3)
  - [ ] Add 11 new rule categories
  - [ ] Add per-file ignores
  - [ ] Update documentation

- [ ] **5. Coverage Configuration** (Section 8)
  - [ ] Expand omit patterns
  - [ ] Add exclude_lines
  - [ ] Document coverage standards

- [ ] **6. Codecov Enhancements** (Section 5)
  - [ ] Add component-based tracking
  - [ ] Configure multi-flag support
  - [ ] Enable GitHub annotations

### Phase 3: Medium Priority (Week 3)

- [ ] **7. OSV Scanner Config** (Section 4)
  - [ ] Create osv-scanner.toml template
  - [ ] Document exception process
  - [ ] Update CI workflows

- [ ] **8. PyProject.toml** (Section 9)
  - [ ] Add conditional dependency groups
  - [ ] Enhance MyPy overrides
  - [ ] Improve organization

### Phase 4: Optional Enhancements (As Needed)

- [ ] **9. Metadata Files** (Section 10)
  - [ ] Add CITATION.cff template
  - [ ] Add codemeta.json template
  - [ ] Add .zenodo.json template

---

## Template vs Production: Scorecard

| Category | Template | Image Detection | Winner | Action |
|----------|----------|----------------|---------|--------|
| **Linting** | Ruff (12 rules) | Ruff (25+ rules) | 🏆 Image Detection | Expand rules |
| **Pre-commit** | Qlty unified | Individual hooks | 🏆 Template | No action |
| **Security Scanning** | OSV Scanner | Safety | 🏆 Template | No action |
| **Nox Automation** | Basic | Comprehensive | 🏆 Image Detection | Expand sessions |
| **Renovate** | Basic | Advanced | 🏆 Image Detection | Enhance config |
| **Documentation** | MkDocs | MkDocs + Validation | 🏆 Image Detection | Add validation |
| **Coverage** | Basic config | Advanced config | 🏆 Image Detection | Expand config |
| **Compliance** | Manual | Automated (nox) | 🏆 Image Detection | Add automation |
| **SBOM** | Not implemented | Automated | 🏆 Image Detection | Add generation |

**Overall Assessment**: Template has superior core architecture (qlty, OSV Scanner) but needs more comprehensive automation and configuration depth.

---

## Conclusion

The image_detection project demonstrates **production-ready patterns** that significantly enhance the template. Priority should be given to:

1. **Nox automation** - Makes all checks runnable locally
2. **Front matter validation** - Ensures documentation quality
3. **Extended Ruff rules** - Catches more bugs earlier
4. **Renovate security prioritization** - Automates dependency management

These improvements align with **OpenSSF Best Practices** and **Google Python Style Guide** requirements while maintaining the template's modern architecture (qlty, OSV Scanner, UV).

**Estimated Implementation Time**: 2-3 weeks for Phases 1-3

---

**Document Version**: 1.0
**Last Updated**: 2025-11-19
**Analysis Scope**: Configuration files and automation patterns
**Next Review**: After Phase 1 implementation
