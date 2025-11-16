# Contributing to {{ cookiecutter.repo_name }}

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Process

### Getting Started

1. **Fork the repository**
   ```bash
   gh repo fork {{ cookiecutter.author_name }}/{{ cookiecutter.repo_name }}
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/{{ cookiecutter.repo_name }}.git
   cd {{ cookiecutter.repo_name }}
   ```

3. **Install dependencies**
   ```bash
   poetry install
   poetry run pre-commit install
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/123-description
   ```

### Development Workflow

1. **Make changes** following our code standards (see below)
2. **Add tests** for new functionality (minimum 80% coverage)
3. **Run quality checks**
   ```bash
   poetry run black .
   poetry run ruff check --fix .
   poetry run mypy src
   poetry run pytest --cov=src --cov-report=term-missing
   poetry run pre-commit run --all-files
   ```
4. **Commit with signing**
   ```bash
   git commit -S -m "feat: description"
   ```
5. **Push and open a PR**
   ```bash
   git push -u origin feature/123-description
   gh pr create
   ```

## Code Standards

### Python Code Quality
- **Formatting**: Black (88 characters line length)
- **Linting**: Ruff with comprehensive rules
- **Type Checking**: MyPy with strict mode
- **Documentation**: Google-style docstrings

### Testing Requirements
- **Minimum Coverage**: 80% test coverage required
- **Test Organization**: Follow `tests/` directory structure mirroring `src/`
- **Test Naming**: `test_<function_name>_<scenario>.py`
- **TDD Encouraged**: Write tests first when practical

### Security Standards
- **No Secrets**: Never commit API keys, passwords, or credentials
- **Input Validation**: Validate and sanitize all user inputs
- **Dependencies**: Keep dependencies up to date
- **Security Scans**: Run `poetry run bandit -r src` before commits

### Git Commit Standards
- **Conventional Commits**: Use `feat:`, `fix:`, `docs:`, `refactor:`, etc.
- **Signed Commits**: All commits must be GPG-signed
- **Atomic Commits**: One logical change per commit
- **Descriptive Messages**: Clear, concise commit messages

Example commit messages:
```
feat: add user authentication with JWT
fix: resolve race condition in payment processing
docs: update API documentation for v2 endpoints
refactor: simplify database query logic
test: add integration tests for auth flow
```

## Testing

### Running Tests
```bash
# Run all tests with coverage
poetry run pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
poetry run pytest tests/test_specific.py -v

# Run tests matching pattern
poetry run pytest -k "test_auth" -v

# Run with debugging
poetry run pytest --pdb
```

### Writing Tests
```python
"""Example test structure."""
import pytest
from your_module import function_to_test


def test_function_success_case():
    """Test successful execution of function."""
    result = function_to_test(valid_input)
    assert result == expected_output


def test_function_error_handling():
    """Test error handling in function."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"key": "value"}
```

## Documentation

### Code Documentation
- **Docstrings**: All public functions, classes, and modules
- **Type Hints**: Use type hints for all function signatures
- **Comments**: Explain complex logic and design decisions
- **Examples**: Include usage examples in docstrings

### Documentation Updates
- **README.md**: Update for user-facing changes
- **CHANGELOG.md**: Document all changes with version
- **API Docs**: Update for API changes
- **Security Implications**: Document in SECURITY.md if applicable

## Pull Request Process

### Before Submitting
- [ ] All tests pass with >80% coverage
- [ ] Pre-commit hooks pass successfully
- [ ] No linting or type checking errors
- [ ] Code is properly documented
- [ ] CHANGELOG.md updated
- [ ] Security implications documented (if applicable)
- [ ] Commits are signed

### PR Description Template
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Security Considerations
Describe any security implications

## Checklist
- [ ] Tests pass (>80% coverage)
- [ ] Pre-commit hooks pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commits are signed
```

### Review Process
1. **Automated Checks**: CI/CD must pass
2. **Code Review**: At least one approving review required
3. **Security Review**: Required for security-related changes
4. **Maintainer Approval**: Final approval from maintainers

## Code of Conduct

### Our Standards
- **Respectful**: Treat everyone with respect
- **Collaborative**: Work together constructively
- **Inclusive**: Welcome diverse perspectives
- **Professional**: Maintain professional communication

### Unacceptable Behavior
- Harassment or discrimination
- Personal attacks
- Trolling or insulting comments
- Publishing others' private information

## Getting Help

### Resources
- **Documentation**: https://github.com/{{ cookiecutter.author_name }}/{{ cookiecutter.repo_name }}/wiki
- **Discussions**: https://github.com/{{ cookiecutter.author_name }}/{{ cookiecutter.repo_name }}/discussions
- **Issues**: https://github.com/{{ cookiecutter.author_name }}/{{ cookiecutter.repo_name }}/issues

### Questions
For questions or support:
1. Check existing documentation
2. Search existing issues
3. Open a discussion thread
4. Contact maintainers: {{ cookiecutter.author_email }}

## OpenSSF Compliance

This project follows [OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/):
- Automated testing and security scanning
- Signed commits and releases
- Documented security processes
- Active vulnerability management

See SECURITY.md for security-specific guidelines.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to {{ cookiecutter.repo_name }}! 🎉
