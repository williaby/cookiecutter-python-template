# Pull Request

## Type of Change
<!-- Mark the appropriate box with an "x" -->
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Maintenance (dependency updates, refactoring, etc.)
- [ ] 🔒 Security update

## Description
<!-- Provide a brief description of the changes -->

## Testing
<!-- Describe the tests that you ran to verify your changes -->
- [ ] Tests pass locally with `poetry run pytest`
- [ ] Code follows the style guidelines (`poetry run black .` and `poetry run ruff check .`)
- [ ] Type checking passes (`poetry run mypy src`)
- [ ] Security scans pass (`poetry run safety check` and `poetry run bandit -r src`)

## Dependency Updates (if applicable)
<!-- For Renovate bot PRs, this section is auto-populated -->

### Change Summary
<!-- Brief description of dependency changes -->

### Security Assessment

- [ ] No known vulnerabilities in updated dependencies
- [ ] All dependencies scanned with pip-audit and OSV scanner
- [ ] Requirements.txt updated with cryptographic hashes
- [ ] SBOM generated for supply chain verification

### Auto-merge Criteria
<!-- This PR will auto-merge if: -->
- [ ] All CI checks pass (tests, linting, security scans)
- [ ] Test coverage remains ≥80%
- [ ] Only patch/minor updates (major updates require manual review)
- [ ] No breaking changes detected
- [ ] Has `automerge` label applied

### Labels Applied
<!-- Auto-applied by Renovate based on update type -->
- `type:dependencies` - Dependency update
- `automerge` or `requires-review` - Merge strategy
- `scope:patch|minor|major` - Update scope
- `security` - Security-related update (if applicable)

## Checklist
<!-- Ensure all items are checked before requesting review -->
- [ ] Code follows project style guidelines
- [ ] Self-review of the code has been performed
- [ ] Code is properly commented, particularly in hard-to-understand areas
- [ ] Corresponding changes to documentation have been made
- [ ] Changes generate no new warnings
- [ ] Tests have been added that prove the fix is effective or that the feature works
- [ ] New and existing unit tests pass locally
- [ ] Any dependent changes have been merged and published

## Additional Context
<!-- Add any other context, screenshots, or relevant information -->

---

## For Reviewers

### Security Review (Required for major updates)

- [ ] Dependencies reviewed for known vulnerabilities
- [ ] License compatibility verified
- [ ] No suspicious packages or maintainers
- [ ] Update aligns with security policies

### Auto-merge Validation

✅ **This PR will auto-merge if:**

- All status checks pass
- Has `automerge` label
- Created by renovate[bot]
- Not a draft PR

🔍 **Manual review required if:**

- Major version updates
- Has `requires-review` label
- Security concerns identified
- Test coverage drops below 80%

---

*🤖 This template supports automated dependency management with security-hardened auto-merge workflows.*
