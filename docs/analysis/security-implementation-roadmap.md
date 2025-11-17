# Security Implementation Roadmap for Cookiecutter Template
## Detailed Action Items & Code Examples

---

## PART 1: PRE-COMMIT HOOKS CONFIGURATION

### Action 1A: Add Gitleaks Hook
**File**: `{{cookiecutter.project_slug}}/.pre-commit-config.yaml`

```yaml
{% if cookiecutter.include_security_scanning == "yes" %}
  # Secrets detection with Gitleaks
  - repo: https://github.com/gitleaks/gitleaks-action
    rev: v2.4.0
    hooks:
      - id: gitleaks
{% endif %}
```

### Action 1B: Add Bandit Configuration
**File**: `{{cookiecutter.project_slug}}/pyproject.toml`

```toml
[tool.bandit]
exclude_dirs = [
    "tests",
    "build",
    "dist",
    ".venv",
    "venv"
]
tests = [
    "B201",  # Flask debug enable
    "B301",  # Pickle usage
    "B302",  # Marshal usage
    "B303",  # MD5 usage
    "B304",  # Cipher algorithms
    "B305",  # Cipher modes
    "B306",  # Temporary file usage
    "B307",  # eval() usage
    "B308",  # Mark safe usage
    "B309",  # HTMLParser XSS
    "B310",  # urllib URL usage
    "B311",  # Random usage
    "B312",  # Telnet usage
    "B313",  # XML parse unsafe
    "B314",  # Pickle usage
    "B315",  # Cookie secure
    "B316",  # Temporary file
    "B317",  # YAML load
    "B318",  # XML parse
    "B319",  # CSV reader
    "B320",  # XML entity
    "B321",  # FTP server
    "B322",  # Input usage
    "B323",  # Unverified context
    "B324",  # Hash functions
    "B325",  # Tempfile usage
]
assert_used = true
```

### Action 1C: Add Safety Hook
**Already in cookiecutter template** - verify it's enabled:

```yaml
{% if cookiecutter.include_security_scanning == "yes" %}
  - repo: https://github.com/Lucas-C/pre-commit-hooks-safety
    rev: v1.3.3
    hooks:
      - id: python-safety-dependencies-check
        files: pyproject.toml
{% endif %}
```

---

## PART 2: CI/CD SECURITY WORKFLOWS

### Action 2A: Create Security Analysis Workflow
**File**: `{{cookiecutter.project_slug}}/.github/workflows/security.yml`

```yaml
name: Security Analysis

on:
  pull_request:
    branches:
      - main
      - develop
  schedule:
    - cron: '30 2 * * 1'  # Weekly Monday
  workflow_dispatch:

permissions: read-all

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    timeout-minutes: 20
    permissions:
      contents: read
      security-events: write

    steps:
      - name: Harden the runner
        uses: step-security/harden-runner@v2.10.1
        with:
          egress-policy: audit

      - name: Checkout code
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
          queries: security-extended

      - name: Perform CodeQL analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: /language:python

  bandit-safety:
    name: Bandit & Safety Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run Bandit
        run: poetry run bandit -r src -f json -o bandit-report.json || true

      - name: Run Safety
        run: poetry run safety check --json || true

  dependency-check:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    permissions:
      contents: read
      pull-requests: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: moderate
```

### Action 2B: Create OSV-Scanner Workflow
**File**: `{{cookiecutter.project_slug}}/.github/workflows/osv-scan.yml`

```yaml
name: OSV Scanner

on:
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 3 * * 1'
  workflow_dispatch:

jobs:
  osv-scanner:
    name: OSV Vulnerability Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run OSV Scanner
        uses: google/osv-scanner-action/osv-scanner-action@v2.2.4
        with:
          scan-args: |-
            --lockfile=poetry.lock
            --format=json
            --output=osv-results.json
        continue-on-error: true

      - name: Check for HIGH/CRITICAL vulnerabilities
        run: |
          python3 << 'PYTHON_EOF'
          import json
          import sys

          try:
              with open('osv-results.json', 'r') as f:
                  results = json.load(f)

              high_crit = []
              for result in results.get('results', []):
                  for pkg in result.get('packages', []):
                      for vuln in pkg.get('vulnerabilities', []):
                          severity = vuln.get('severity', 'UNKNOWN').upper()
                          if severity in ['HIGH', 'CRITICAL']:
                              high_crit.append(f"{vuln.get('id')} ({severity})")

              if high_crit:
                  print(f"❌ Found {len(high_crit)} HIGH/CRITICAL vulnerabilities")
                  sys.exit(1)
              else:
                  print("✅ No HIGH/CRITICAL vulnerabilities found")
          except Exception as e:
              print(f"⚠️ Error parsing results: {e}")
              sys.exit(1)
          PYTHON_EOF
```

---

## PART 3: SECURITY DOCUMENTATION

### Action 3A: Create SECURITY.md Template
**File**: `{{cookiecutter.project_slug}}/SECURITY.md`

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| {{cookiecutter.version}} | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### GitHub Private Vulnerability Reporting

Use GitHub's private vulnerability reporting:
https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/security/advisories/new

### Email

Alternatively, email: {{cookiecutter.security_email}}

Include:
- Vulnerability type
- Affected code path
- Steps to reproduce
- Proof-of-concept (if available)
- Impact assessment

## Response Timeline

- **Acknowledgment**: 7 days
- **Assessment**: 14 days
- **Fix Timeline**:
  - Critical: 30 days
  - High: 60 days
  - Medium: 60 days
  - Low: Next release

## Security Tools

| Tool | Purpose |
|------|---------|
| **CodeQL** | Semantic code analysis |
| **Bandit** | Python security scanning |
| **Safety** | Dependency vulnerability check |
| **OSV-Scanner** | Multi-ecosystem vulnerability detection |
| **MyPy** | Type safety |

All tools run automatically on every commit via pre-commit and CI/CD.

## Security Best Practices for Users

- Keep dependencies updated: `poetry update`
- Run security scans: `poetry run bandit -r src`
- Check vulnerabilities: `poetry run safety check`
- Review advisories: https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/security/advisories

## Common Vulnerability Mitigations

### Input Validation
- All inputs validated via Pydantic schemas
- File size limits enforced
- Type checking with MyPy

### Dependency Security
- Regular updates via Poetry
- Automated vulnerability scanning
- OSV-Scanner for comprehensive coverage

### Code Quality
- Type safety (MyPy strict mode)
- Security linting (Bandit)
- 80%+ test coverage minimum

### No Dangerous Patterns
- No eval() or exec() usage
- No pickle deserialization
- No shell=True execution
- No hardcoded secrets
```

### Action 3B: Create COMPLIANCE.md Template
**File**: `{{cookiecutter.project_slug}}/COMPLIANCE.md`

```markdown
# Compliance & Audit

## Regulatory Frameworks

### OWASP Top 10 Mapping

| OWASP Risk | Status | Mitigation |
|-----------|--------|-----------|
| Injection | Mitigated | Pydantic input validation |
| Broken Auth | N/A | No authentication system |
| Sensitive Data | Mitigated | No sensitive data storage |
| XXE | Mitigated | XML parsing hardened |
| Broken Access | N/A | Local processing only |
| Security Misconfiguration | Mitigated | Strict linting |
| XSS | N/A | No web interface |
| Insecure Deserialization | Mitigated | JSON only via Pydantic |
| Vulnerable Components | Mitigated | OSV-Scanner CI/CD |
| Logging/Monitoring | Implemented | Structured logging |

### CWE Coverage

CodeQL scans for:
- CWE-89: SQL Injection
- CWE-78: Command Injection
- CWE-22: Path Traversal
- CWE-798: Hardcoded Secrets
- CWE-502: Insecure Deserialization
- CWE-327: Weak Cryptography
- CWE-95: Code Injection

## Audit Trail

### Git History
- All commits signed with GPG
- Branch protection enforced
- Code review required

### Dependency Updates
- Automated tracking via Dependabot
- Security advisories monitored
- Lock file maintained

### Security Scanning
- CodeQL: Every PR + weekly
- OSV-Scanner: Every PR + weekly
- Bandit/Safety: Every commit (pre-commit)

## Metrics & Reporting

Track these security metrics:
- Vulnerability resolution time
- Code coverage percentage
- Type coverage (MyPy)
- Security findings by severity
```

### Action 3C: Create Threat Model Template
**File**: `{{cookiecutter.project_slug}}/docs/THREAT_MODEL.md`

```markdown
# Threat Model

## Assets

- Application code
- User input data
- Configuration/secrets
- Dependencies

## Threat Actors

- External attackers
- Malicious input sources
- Compromised dependencies

## Attack Vectors

### 1. Malicious Input
- **Threat**: Inject malicious code/commands
- **Likelihood**: High
- **Impact**: Code execution
- **Mitigation**: 
  - Input validation via Pydantic
  - Type checking with MyPy
  - No eval/exec usage

### 2. Dependency Compromise
- **Threat**: Vulnerable dependencies
- **Likelihood**: Medium
- **Impact**: Code execution
- **Mitigation**:
  - OSV-Scanner CI/CD
  - Regular updates
  - Dependency pinning

### 3. Secrets Exposure
- **Threat**: API keys leaked in code/logs
- **Likelihood**: Medium
- **Impact**: Account compromise
- **Mitigation**:
  - detect-private-key hook
  - Environment variables
  - Log masking

## Risk Matrix

| Risk | Likelihood | Impact | Priority |
|------|-----------|--------|----------|
| Input injection | High | Critical | 1 |
| Dependency exploit | Medium | High | 2 |
| Secret exposure | Medium | High | 2 |
| Type errors | Low | Medium | 3 |
```

---

## PART 4: SECURITY TESTING

### Action 4A: Create Security Test Structure
**File**: `{{cookiecutter.project_slug}}/tests/security/test_input_validation.py`

```python
"""Security tests for input validation."""

import pytest

def test_no_eval_usage():
    """Verify eval() is not used in source code."""
    import os
    import re

    dangerous_patterns = [
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'pickle\.loads',
        r'yaml\.load\s*\(',
        r'subprocess\.run.*shell=True',
    ]

    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    for pattern in dangerous_patterns:
                        if re.search(pattern, content):
                            pytest.fail(
                                f"Dangerous pattern '{pattern}' found in {filepath}"
                            )


def test_no_hardcoded_secrets():
    """Verify no hardcoded secrets in source code."""
    import os
    import re

    secret_patterns = [
        r'password\s*[=:]\s*["\'][\w]{4,}["\']',
        r'api_key\s*[=:]\s*["\'][\w]{4,}["\']',
        r'secret\s*[=:]\s*["\'][\w]{4,}["\']',
    ]

    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    for i, line in enumerate(f, 1):
                        for pattern in secret_patterns:
                            if (re.search(pattern, line, re.IGNORECASE) and
                                'example' not in line.lower()):
                                pytest.fail(
                                    f"Potential hardcoded secret at {filepath}:{i}"
                                )
```

### Action 4B: Create Secrets Handling Tests
**File**: `{{cookiecutter.project_slug}}/tests/security/test_secrets_handling.py`

```python
"""Security tests for secrets handling."""

import os
import pytest
from unittest.mock import patch

def test_env_variables_loaded():
    """Verify environment variables are properly loaded."""
    # Should use pydantic-settings or python-dotenv
    assert hasattr(os, 'environ'), "Environment variables not accessible"


def test_no_secrets_in_logs(caplog):
    """Verify secrets are not logged."""
    import logging

    logger = logging.getLogger(__name__)

    # This should be masked
    api_key = "sk-1234567890abcdef"
    logger.info(f"Using API key: {api_key}")

    # Verify the actual key is not in logs
    assert api_key not in caplog.text or "[MASKED]" in caplog.text


def test_env_file_not_committed():
    """Verify .env is in .gitignore."""
    with open('.gitignore', 'r') as f:
        gitignore_content = f.read()
        assert '.env' in gitignore_content, ".env not in .gitignore"
        assert '*.b64' in gitignore_content, "Encoded secrets not ignored"
        assert '*service-account*' in gitignore_content, "Service accounts not ignored"
```

---

## PART 5: FUZZING SETUP

### Action 5A: Create Fuzzing Harness Template
**File**: `{{cookiecutter.project_slug}}/fuzz/fuzz_core_functionality.py`

```python
"""Fuzzing harness for core functionality.

This harness uses Atheris to fuzz the main processing functions
with coverage-guided input generation.
"""

import atheris
import sys
from typing import Any

# Import your core functionality here
# from {{cookiecutter.project_slug}}.core import process_input


@atheris.instrument_func
def fuzz_core_processor(data: bytes) -> None:
    """Fuzz the core processor with arbitrary input."""
    try:
        # Example: if processing strings
        try:
            input_str = data.decode('utf-8')
        except UnicodeDecodeError:
            input_str = data.hex()

        # Call your core function
        # result = process_input(input_str)

        # Don't crash on any exception - fuzzer needs to continue
    except Exception:
        # Expected for malformed input
        pass


def main(argv: list[str]) -> int:
    """Run fuzzing."""
    # Load input corpus if available
    corpus_path = "fuzz/corpus"

    atheris.Setup(argv, fuzz_core_processor)
    atheris.Fuzz()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

### Action 5B: Create Fuzzing Configuration
**File**: `{{cookiecutter.project_slug}}/.clusterfuzzlite/project.yaml`

```yaml
homepage: "https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}"
primary_contact: "{{cookiecutter.author_email}}"
language: python
fuzzing_engines:
  - libfuzzer
sanitizers:
  - address
  - memory
  - undefined
  - integer
architectures:
  - x86_64
help_wanted: false
```

---

## PART 6: VULNERABILITY EXCEPTION MANAGEMENT

### Action 6A: Create OSV-Scanner Exception Template
**File**: `{{cookiecutter.project_slug}}/osv-scanner.toml`

```toml
# OSV-Scanner Vulnerability Exceptions
# This file documents false positives and non-applicable vulnerabilities
# Reference: https://google.github.io/osv-scanner/configuration/

# Example: Bleach - Outdated vulnerability data (using newer version)
# [[IgnoredVulns]]
# id = "PYSEC-2020-28"
# reason = "Fixed in bleach 3.1.2 (March 2020). Current version is significantly newer. Not vulnerable."

# Notes:
# - All exceptions must include detailed justification
# - Review quarterly or when dependencies update
# - Format: [[IgnoredVulns]] with id and reason fields
```

---

## PART 7: COOKIECUTTER CONFIGURATION

### Action 7A: Update cookiecutter.json
```json
{
    "project_name": "My Project",
    "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}",
    "author_name": "Author Name",
    "author_email": "author@example.com",
    "security_email": "security@example.com",
    "github_username": "github_username",
    "version": "0.1.0",
    "python_version": "3.12",
    "use_pre_commit": "yes",
    "use_ruff": "yes",
    "use_mypy": "yes",
    "include_security_scanning": "yes",
    "include_fuzzing": "no",
    "include_compliance_docs": "yes",
    "cwe_focus_areas": "all"
}
```

---

## IMPLEMENTATION CHECKLIST

- [ ] Phase 1: Documentation
  - [ ] SECURITY.md created
  - [ ] COMPLIANCE.md created
  - [ ] THREAT_MODEL.md created

- [ ] Phase 2: Pre-Commit Hooks
  - [ ] Gitleaks added
  - [ ] Bandit configured
  - [ ] Safety check enabled
  - [ ] MyPy enabled

- [ ] Phase 3: CI/CD Workflows
  - [ ] security.yml workflow created
  - [ ] osv-scan.yml workflow created
  - [ ] Branch protection rules configured
  - [ ] SARIF report upload enabled

- [ ] Phase 4: Security Testing
  - [ ] tests/security directory created
  - [ ] Input validation tests added
  - [ ] Secrets handling tests added
  - [ ] Security checklist test added

- [ ] Phase 5: Fuzzing (Optional)
  - [ ] Fuzzing harness created
  - [ ] ClusterFuzzLite configured
  - [ ] CIFuzz workflow added

- [ ] Phase 6: Vulnerability Management
  - [ ] osv-scanner.toml template created
  - [ ] Exception review process documented
  - [ ] Update policy established

- [ ] Phase 7: Cookiecutter Configuration
  - [ ] Template variables updated
  - [ ] Conditional content working
  - [ ] Test generation verified

---

**Total Implementation Time**: 2-3 weeks
**Effort Level**: Medium
**Priority**: High
