# Security Policy

## Reporting Vulnerabilities

**DO NOT** open public issues for security vulnerabilities.

### How to Report

Email security reports to: **{{ cookiecutter.author_email }}**

Include the following information:
- **Description**: Clear description of the vulnerability
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Potential Impact**: Assessment of the security impact
- **Suggested Fix**: Proposed solution (if any)
- **Affected Versions**: Which versions are affected

### Response Timeline

- **Initial Acknowledgment**: Within 14 days
- **Status Updates**: Every 7 days until resolved
- **Fix Timeline**: Within 90 days for critical vulnerabilities

## Supported Versions

| Version | Supported          | End of Support |
| ------- | ------------------ | -------------- |
| 1.x     | :white_check_mark: | TBD            |
| < 1.0   | :x:                | N/A            |

## Security Best Practices

When contributing to this project:

1. **Never commit secrets**: Use environment variables or encrypted configuration
2. **Validate inputs**: Always validate and sanitize user inputs
3. **Use latest dependencies**: Keep dependencies up to date
4. **Follow secure coding practices**: See CONTRIBUTING.md for guidelines
5. **Run security scans**: Use `poetry run bandit -r src` before commits

## Security Features

This project implements the following security measures:

- ✅ **Dependency Scanning**: Automated checks via `safety` and `bandit`
- ✅ **Secret Detection**: Pre-commit hooks prevent credential leaks
- ✅ **Signed Commits**: GPG-signed commits required
- ✅ **Encrypted Secrets**: GPG-encrypted `.env` files
- ✅ **Static Analysis**: Ruff and MyPy enforce code quality
- ✅ **HTTPS Only**: All external communications use HTTPS

## Known Security Considerations

### Authentication
- Service accounts use JWT tokens with short expiration
- API keys are stored encrypted and rotated regularly

### Data Protection
- Sensitive data is encrypted at rest
- PII is handled according to GDPR requirements

### Network Security
- All API endpoints use HTTPS
- Rate limiting enabled on public endpoints

## Security Advisories

Security advisories will be published at:
- GitHub Security Advisories: https://github.com/{{ cookiecutter.author_name }}/{{ cookiecutter.repo_name }}/security/advisories
- Project changelog: CHANGELOG.md

## OpenSSF Best Practices

This project follows [OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/):
- Automated testing with >80% coverage
- Static analysis and security scanning
- Signed releases
- Documented security processes

## Contact

For security concerns, contact: {{ cookiecutter.author_email }}

For general questions: https://github.com/{{ cookiecutter.author_name }}/{{ cookiecutter.repo_name }}/discussions
