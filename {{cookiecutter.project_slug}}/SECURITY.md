# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported          |
| ------- | ------------------ |
| {{cookiecutter.version}} | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

### GitHub Private Vulnerability Reporting

Use GitHub's private vulnerability reporting feature:
https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/security/advisories/new

### Email

Alternatively, email security reports to: {{cookiecutter.author_email}}

Include:
- Type of vulnerability
- Full path to affected source file(s)
- Location of affected code (tag/branch/commit)
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if possible)
- Impact assessment

## Response Timeline

- **Acknowledgment**: Within 7 days
- **Initial Assessment**: Within 14 days
- **Fix Timeline**:
  - Critical: Within 30 days
  - High: Within 60 days
  - Medium: Within 60 days
  - Low: Next release cycle

## Disclosure Policy

- Security advisories published after fix is available
- CVE requested for significant vulnerabilities
- Credit given to reporters (unless anonymity requested)

## Security Update Process

1. Fix developed in private fork
2. Fix tested and reviewed
3. Security advisory published
4. Patched version released
5. Public disclosure with CVE (if applicable)

## Security Best Practices for Users

- Keep dependencies updated: `poetry update`
- Run security scans: `poetry run bandit -r src`
- Check for known vulnerabilities: `poetry run safety check`
- Review security advisories: https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/security/advisories

## Automated Security Tools

This project uses the following automated security tools:

| Tool | Purpose | Integration |
|------|---------|-------------|
| **Bandit** | Python security vulnerability scanning | Pre-commit hook + CI/CD |
| **Safety** | Dependency vulnerability checking | Pre-commit hook + CI/CD |
| **MyPy** | Static type checking (prevents type-related bugs) | Pre-commit hook + CI/CD |
| **Pydantic** | Runtime data validation and type safety | Core dependency |
| **Poetry** | Dependency lock file with cryptographic hashes | Build system |

All security tools run automatically on every commit via pre-commit hooks and in the CI/CD pipeline.

## Security Design Principles

This project follows secure development practices:

### Input Validation
- All file inputs validated for type and size
- JSON schema validation via Pydantic

### Dependency Security
- Regular dependency updates via Poetry
- Automated vulnerability scanning (Safety, Bandit)
- Minimal dependency footprint

### Code Quality
- Type safety via MyPy strict mode
- Comprehensive test coverage (80%+)
- Security-focused linting with Bandit

## Secret Management

The repository is continuously scanned for accidental secret exposure. Local developers should run security scanning pre-commit hooks to prevent committing credentials. No private keys or tokens should be stored in this repository.
