# Org-Level Reusable Workflows Handoff Document

**Date**: 2025-11-22
**From**: Cookiecutter Template Repository
**To**: Org `.github` Repository Team
**Purpose**: Convert template workflows to org-level reusable workflows

---

## Executive Summary

The cookiecutter-python-template repository contains several GitHub Actions workflows that would benefit from being centralized as org-level reusable workflows. This document provides the current implementations, suggested interfaces, and conversion guidance.

**Benefits of Centralization:**
- Single point of maintenance for security updates
- Consistent behavior across all org projects
- Reduced template complexity (thin callers only)
- Easier to roll out new checks org-wide
- Version-controlled workflow changes

---

## Workflows Recommended for Conversion

### Priority 1: Security & Compliance (Critical)

| Workflow | Current Location | Suggested Reusable Name |
|----------|------------------|-------------------------|
| Security Analysis | `security-analysis.yml` | `python-security.yml` |
| OpenSSF Scorecard | `scorecard.yml` | `python-scorecard.yml` |
| SBOM Generation | `sbom.yml` | `python-sbom.yml` |
| SLSA Provenance | `slsa-provenance.yml` | `python-slsa.yml` |
| REUSE Compliance | `reuse.yml` | `python-reuse.yml` |

### Priority 2: Quality & Testing

| Workflow | Current Location | Suggested Reusable Name |
|----------|------------------|-------------------------|
| Mutation Testing | `mutation-testing.yml` | `python-mutation.yml` |
| Python Compatibility | `python-compatibility.yml` | `python-compatibility.yml` |
| Codecov Upload | `codecov.yml` | `python-codecov.yml` |
| Container Security | `container-security.yml` | `python-container-security.yml` |

### Priority 3: Release & Publishing

| Workflow | Current Location | Suggested Reusable Name |
|----------|------------------|-------------------------|
| PyPI Publish | `publish-pypi.yml` | `python-publish.yml` |
| PR Validation | `pr-validation.yml` | `python-pr-validation.yml` |

---

## Detailed Workflow Specifications

### 1. Security Analysis (`python-security.yml`)

**Purpose**: Comprehensive security scanning with Bandit, Safety, Gitleaks, and OSV Scanner.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/security-analysis.yml`

**Suggested Interface**:

```yaml
name: Python Security Analysis

on:
  workflow_call:
    inputs:
      python-version:
        description: 'Python version to use'
        required: false
        default: '3.12'
        type: string
      source-directory:
        description: 'Source code directory'
        required: false
        default: 'src'
        type: string
      enable-bandit:
        description: 'Enable Bandit SAST scanning'
        required: false
        default: true
        type: boolean
      enable-safety:
        description: 'Enable Safety dependency scanning'
        required: false
        default: true
        type: boolean
      enable-gitleaks:
        description: 'Enable Gitleaks secret scanning'
        required: false
        default: true
        type: boolean
      enable-osv:
        description: 'Enable OSV Scanner'
        required: false
        default: true
        type: boolean
      bandit-severity:
        description: 'Minimum Bandit severity (low, medium, high)'
        required: false
        default: 'medium'
        type: string
      fail-on-findings:
        description: 'Fail workflow on security findings'
        required: false
        default: true
        type: boolean
    secrets:
      GITHUB_TOKEN:
        required: true
```

**Jobs to Include**:
- `bandit` - SAST scanning with SARIF upload
- `safety` - Dependency vulnerability scanning
- `gitleaks` - Secret detection
- `osv-scanner` - OSV database vulnerability check
- `security-summary` - Aggregated results

---

### 2. OpenSSF Scorecard (`python-scorecard.yml`)

**Purpose**: OpenSSF Scorecard security health metrics.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/scorecard.yml`

**Suggested Interface**:

```yaml
name: OpenSSF Scorecard

on:
  workflow_call:
    inputs:
      publish-results:
        description: 'Publish results to OpenSSF'
        required: false
        default: true
        type: boolean
    secrets:
      SCORECARD_TOKEN:
        required: false
        description: 'Token for publishing (defaults to GITHUB_TOKEN)'
```

**Notes**:
- Minimal customization needed
- Should run on default branch only
- SARIF upload to Security tab

---

### 3. SBOM Generation (`python-sbom.yml`)

**Purpose**: Generate Software Bill of Materials using CycloneDX and Syft.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/sbom.yml`

**Suggested Interface**:

```yaml
name: SBOM Generation

on:
  workflow_call:
    inputs:
      python-version:
        description: 'Python version'
        required: false
        default: '3.12'
        type: string
      output-format:
        description: 'SBOM format (cyclonedx-json, spdx-json)'
        required: false
        default: 'cyclonedx-json'
        type: string
      include-syft:
        description: 'Also generate Syft SBOM'
        required: false
        default: true
        type: boolean
      artifact-retention-days:
        description: 'Days to retain SBOM artifacts'
        required: false
        default: 90
        type: number
    outputs:
      sbom-artifact-name:
        description: 'Name of uploaded SBOM artifact'
        value: ${{ jobs.sbom.outputs.artifact-name }}
```

---

### 4. SLSA Provenance (`python-slsa.yml`)

**Purpose**: Generate SLSA Level 3 provenance attestations for supply chain security.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/slsa-provenance.yml`

**Suggested Interface**:

```yaml
name: SLSA Provenance

on:
  workflow_call:
    inputs:
      python-version:
        description: 'Python version'
        required: false
        default: '3.12'
        type: string
      slsa-level:
        description: 'SLSA level (1, 2, 3)'
        required: false
        default: 3
        type: number
    secrets:
      GITHUB_TOKEN:
        required: true
    outputs:
      provenance-artifact:
        description: 'Name of provenance artifact'
        value: ${{ jobs.provenance.outputs.artifact-name }}
```

**Notes**:
- Uses `slsa-framework/slsa-github-generator`
- Triggered on releases
- Attestation uploaded to GitHub

---

### 5. REUSE Compliance (`python-reuse.yml`)

**Purpose**: Validate REUSE 3.0 specification compliance for licensing.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/reuse.yml`

**Suggested Interface**:

```yaml
name: REUSE Compliance

on:
  workflow_call:
    inputs:
      generate-spdx:
        description: 'Generate SPDX report on success'
        required: false
        default: true
        type: boolean
      required-licenses:
        description: 'Comma-separated list of required license files'
        required: false
        default: ''
        type: string
      artifact-retention-days:
        description: 'Days to retain SPDX artifact'
        required: false
        default: 90
        type: number
```

**Jobs to Include**:
- `reuse-check` - FSFE REUSE compliance check
- `validate-licenses` - Verify required license files exist
- `generate-spdx` - Generate SPDX report (conditional)

---

### 6. Mutation Testing (`python-mutation.yml`)

**Purpose**: Validate test effectiveness using mutation testing with mutmut.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/mutation-testing.yml`

**Suggested Interface**:

```yaml
name: Mutation Testing

on:
  workflow_call:
    inputs:
      python-version:
        description: 'Python version'
        required: false
        default: '3.12'
        type: string
      source-directory:
        description: 'Source directory to mutate'
        required: false
        default: 'src'
        type: string
      mutation-threshold:
        description: 'Minimum mutation score (%)'
        required: false
        default: 80
        type: number
      timeout-minutes:
        description: 'Maximum mutation testing time'
        required: false
        default: 60
        type: number
      post-pr-comment:
        description: 'Post results as PR comment'
        required: false
        default: true
        type: boolean
    secrets:
      GITHUB_TOKEN:
        required: true
    outputs:
      mutation-score:
        description: 'Achieved mutation score'
        value: ${{ jobs.mutation.outputs.score }}
      killed:
        description: 'Number of killed mutations'
        value: ${{ jobs.mutation.outputs.killed }}
      survived:
        description: 'Number of survived mutations'
        value: ${{ jobs.mutation.outputs.survived }}
```

**Notes**:
- Weekly scheduled runs recommended
- Manual trigger with threshold override
- HTML report artifact generation

---

### 7. Python Compatibility (`python-compatibility.yml`)

**Purpose**: Matrix testing across Python versions and operating systems.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/python-compatibility.yml`

**Suggested Interface**:

```yaml
name: Python Compatibility

on:
  workflow_call:
    inputs:
      python-versions:
        description: 'JSON array of Python versions'
        required: false
        default: '["3.10", "3.11", "3.12", "3.13"]'
        type: string
      operating-systems:
        description: 'JSON array of OS to test'
        required: false
        default: '["ubuntu-latest"]'
        type: string
      include-macos:
        description: 'Include macOS in matrix'
        required: false
        default: false
        type: boolean
      include-windows:
        description: 'Include Windows in matrix'
        required: false
        default: false
        type: boolean
      primary-python-version:
        description: 'Primary Python version for cross-platform'
        required: false
        default: '3.12'
        type: string
      test-command:
        description: 'Test command to run'
        required: false
        default: 'uv run pytest -v'
        type: string
```

---

### 8. Codecov Upload (`python-codecov.yml`)

**Purpose**: Secure coverage upload triggered after CI completion.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/codecov.yml`

**Suggested Interface**:

```yaml
name: Codecov Upload

on:
  workflow_call:
    inputs:
      coverage-artifact-name:
        description: 'Name of coverage artifact from CI'
        required: false
        default: 'coverage-report'
        type: string
      coverage-file:
        description: 'Coverage file name'
        required: false
        default: 'coverage.xml'
        type: string
      flags:
        description: 'Codecov flags'
        required: false
        default: ''
        type: string
      fail-on-error:
        description: 'Fail workflow on upload error'
        required: false
        default: false
        type: boolean
    secrets:
      CODECOV_TOKEN:
        required: true
```

**Security Notes**:
- Must NOT checkout source code
- Downloads artifact only
- Prevents malicious PR code execution

---

### 9. Container Security (`python-container-security.yml`)

**Purpose**: Container image scanning with Trivy and Dockerfile linting with Hadolint.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/container-security.yml`

**Suggested Interface**:

```yaml
name: Container Security

on:
  workflow_call:
    inputs:
      dockerfile-path:
        description: 'Path to Dockerfile'
        required: false
        default: 'Dockerfile'
        type: string
      image-name:
        description: 'Image name for scanning'
        required: false
        default: 'app'
        type: string
      trivy-severity:
        description: 'Trivy severity threshold'
        required: false
        default: 'CRITICAL,HIGH'
        type: string
      enable-hadolint:
        description: 'Enable Hadolint Dockerfile linting'
        required: false
        default: true
        type: boolean
      enable-sbom:
        description: 'Generate container SBOM'
        required: false
        default: true
        type: boolean
      fail-on-vulnerabilities:
        description: 'Fail on vulnerability findings'
        required: false
        default: true
        type: boolean
    secrets:
      GITHUB_TOKEN:
        required: true
```

---

### 10. PyPI Publish (`python-publish.yml`)

**Purpose**: Publish packages to PyPI using trusted publishing (OIDC).

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/publish-pypi.yml`

**Suggested Interface**:

```yaml
name: PyPI Publish

on:
  workflow_call:
    inputs:
      python-version:
        description: 'Python version for build'
        required: false
        default: '3.12'
        type: string
      publish-to-testpypi:
        description: 'Also publish to TestPyPI'
        required: false
        default: false
        type: boolean
      environment:
        description: 'GitHub environment for publishing'
        required: false
        default: 'pypi'
        type: string
      attestations:
        description: 'Generate package attestations'
        required: false
        default: true
        type: boolean
    secrets:
      PYPI_TOKEN:
        required: false
        description: 'PyPI token (optional if using trusted publishing)'
```

**Notes**:
- Prefer trusted publishing (OIDC) over tokens
- Build isolation with separate job
- Attestation generation for supply chain

---

### 11. PR Validation (`python-pr-validation.yml`)

**Purpose**: Comprehensive PR validation checks.

**Current Implementation**: `{{cookiecutter.project_slug}}/.github/workflows/pr-validation.yml`

**Suggested Interface**:

```yaml
name: PR Validation

on:
  workflow_call:
    inputs:
      python-version:
        description: 'Python version'
        required: false
        default: '3.12'
        type: string
      # Feature flags for optional checks
      enable-dependency-review:
        description: 'Enable dependency review'
        required: false
        default: true
        type: boolean
      enable-license-check:
        description: 'Enable license compliance check'
        required: false
        default: true
        type: boolean
      enable-dead-code:
        description: 'Enable vulture dead code detection'
        required: false
        default: true
        type: boolean
      enable-pip-audit:
        description: 'Enable pip-audit vulnerability check'
        required: false
        default: true
        type: boolean
      enable-changelog:
        description: 'Enable changelog enforcement'
        required: false
        default: true
        type: boolean
      enable-link-check:
        description: 'Enable documentation link checking'
        required: false
        default: false
        type: boolean
      enable-pr-size:
        description: 'Enable PR size labeling'
        required: false
        default: true
        type: boolean
      # Configuration
      denied-licenses:
        description: 'Comma-separated denied licenses'
        required: false
        default: 'GPL-3.0,AGPL-3.0,SSPL-1.0,BUSL-1.1'
        type: string
      changelog-path:
        description: 'Path to changelog file'
        required: false
        default: 'CHANGELOG.md'
        type: string
      skip-changelog-labels:
        description: 'Labels that skip changelog requirement'
        required: false
        default: 'skip-changelog,dependencies,documentation'
        type: string
    secrets:
      GITHUB_TOKEN:
        required: true
```

**Jobs to Include**:
- `dependency-review` - Dependency vulnerability review
- `license-check` - License compliance
- `dead-code` - Vulture dead code detection
- `pip-audit` - Python vulnerability scanning
- `pr-size` - PR size labeling
- `changelog` - Changelog enforcement
- `link-check` - Documentation link validation
- `validation-summary` - Aggregated status

---

## Template Caller Examples

Once workflows are converted, the template will use thin callers:

### Example: Security Analysis Caller

```yaml
# {{cookiecutter.project_slug}}/.github/workflows/security-analysis.yml
name: Security Analysis

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 6 * * 1'

permissions:
  contents: read
  security-events: write

jobs:
  security:
    uses: {{ cookiecutter.github_org_or_user }}/.github/.github/workflows/python-security.yml@main
    with:
      python-version: '{{ cookiecutter.python_version }}'
      source-directory: 'src'
    secrets: inherit
```

### Example: Mutation Testing Caller

```yaml
# {{cookiecutter.project_slug}}/.github/workflows/mutation-testing.yml
name: Mutation Testing

on:
  schedule:
    - cron: '0 2 * * 0'
  workflow_dispatch:
    inputs:
      mutation_threshold:
        description: 'Minimum mutation score (%)'
        default: '80'
  pull_request:
    paths:
      - 'src/**/*.py'
      - 'tests/**/*.py'

permissions:
  contents: read
  pull-requests: write

jobs:
  mutation:
    uses: {{ cookiecutter.github_org_or_user }}/.github/.github/workflows/python-mutation.yml@main
    with:
      python-version: '{{ cookiecutter.python_version }}'
      source-directory: 'src'
      mutation-threshold: {% raw %}${{ github.event.inputs.mutation_threshold || 80 }}{% endraw %}
    secrets: inherit
```

---

## Migration Checklist

For each workflow conversion:

- [ ] Review current implementation in template
- [ ] Design reusable workflow interface (inputs/outputs/secrets)
- [ ] Implement reusable workflow in org `.github` repo
- [ ] Test with a sample repository
- [ ] Update template to use thin caller
- [ ] Update `post_gen_project.py` cleanup logic if needed
- [ ] Document in org `.github` README
- [ ] Notify dependent repositories

---

## Files Reference

All current workflow implementations are located at:

```
cookiecutter-python-template/
└── {{cookiecutter.project_slug}}/
    └── .github/
        └── workflows/
            ├── ci.yml                    # Already calls org-level
            ├── cifuzzy.yml               # Keep in template (project-specific)
            ├── codecov.yml               # Convert to reusable
            ├── container-security.yml    # Convert to reusable
            ├── docs.yml                  # Consider for reusable
            ├── mutation-testing.yml      # Convert to reusable
            ├── pr-validation.yml         # Convert to reusable
            ├── publish-pypi.yml          # Convert to reusable
            ├── python-compatibility.yml  # Convert to reusable
            ├── release.yml               # Consider for reusable
            ├── reuse.yml                 # Convert to reusable
            ├── sbom.yml                  # Convert to reusable
            ├── scorecard.yml             # Convert to reusable
            ├── security-analysis.yml     # Convert to reusable
            ├── slsa-provenance.yml       # Convert to reusable
            └── sonarcloud.yml            # Keep in template (project-specific)
```

---

## Questions for Review

1. **Naming Convention**: Should reusable workflows use `python-` prefix or another convention?

2. **Versioning**: How should we version reusable workflows? Tags, branches, or SHA pinning?

3. **Breaking Changes**: What's the process for introducing breaking changes to reusable workflow interfaces?

4. **Default Values**: Are the suggested default values appropriate for all org projects?

5. **Secrets Management**: Should we use `secrets: inherit` or explicit secret passing?

6. **Documentation**: Where should reusable workflow documentation live?

---

## Contact

For questions about current implementations, contact the cookiecutter-python-template maintainers.

For questions about org-level workflow infrastructure, contact the `.github` repository maintainers.
