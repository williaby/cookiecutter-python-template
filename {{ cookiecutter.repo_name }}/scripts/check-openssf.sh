#!/bin/bash
# OpenSSF Compliance Checker
# Quick verification of OpenSSF best practices compliance
# Usage: ./scripts/check-openssf.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Checking OpenSSF Best Practices Compliance...${NC}\n"

# Track compliance status
COMPLIANCE_PASSED=true

# Required files check
echo -e "${BLUE}📄 Required Files:${NC}"
FILES=("LICENSE" "SECURITY.md" "CONTRIBUTING.md" "CHANGELOG.md" "README.md")
for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✅ $file exists${NC}"
    else
        echo -e "${RED}❌ $file missing${NC}"
        COMPLIANCE_PASSED=false
    fi
done

# Security checks
echo -e "\n${BLUE}🔒 Security Checks:${NC}"

# Check for poetry
if command -v poetry &> /dev/null; then
    # Safety check for known vulnerabilities
    if poetry run safety check --json > /dev/null 2>&1; then
        echo -e "${GREEN}✅ No known vulnerabilities (safety)${NC}"
    else
        VULNERABILITIES=$(poetry run safety check 2>&1 | grep -c "vulnerability" || true)
        if [[ $VULNERABILITIES -gt 0 ]]; then
            echo -e "${RED}⚠️  $VULNERABILITIES vulnerabilities found (safety)${NC}"
            COMPLIANCE_PASSED=false
        else
            echo -e "${GREEN}✅ No known vulnerabilities (safety)${NC}"
        fi
    fi

    # Bandit security scan
    if poetry run bandit -r src -f json > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Bandit security scan passed${NC}"
    else
        ISSUES=$(poetry run bandit -r src 2>&1 | grep -c "Issue:" || true)
        if [[ $ISSUES -gt 0 ]]; then
            echo -e "${YELLOW}⚠️  $ISSUES security issues found (bandit)${NC}"
        else
            echo -e "${GREEN}✅ Bandit security scan passed${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Poetry not found - skipping dependency security checks${NC}"
fi

# Git security checks
echo -e "\n${BLUE}🔐 Git Security:${NC}"

# Check if commits are signed
if git log -1 --show-signature > /dev/null 2>&1; then
    if git log -1 --show-signature 2>&1 | grep -q "Good signature"; then
        echo -e "${GREEN}✅ Latest commit is GPG-signed${NC}"
    else
        echo -e "${YELLOW}⚠️  Latest commit may not be properly signed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Unable to verify commit signatures${NC}"
fi

# Check for secrets in repository
if command -v gitleaks &> /dev/null; then
    if gitleaks detect --no-git --verbose > /dev/null 2>&1; then
        echo -e "${GREEN}✅ No secrets detected (gitleaks)${NC}"
    else
        echo -e "${RED}❌ Potential secrets detected (gitleaks)${NC}"
        COMPLIANCE_PASSED=false
    fi
else
    echo -e "${YELLOW}⚠️  gitleaks not installed - skipping secret detection${NC}"
fi

# Test coverage check
echo -e "\n${BLUE}🧪 Test Coverage:${NC}"

if command -v poetry &> /dev/null; then
    if [[ -f "pyproject.toml" ]] && [[ -d "tests" ]]; then
        # Run tests with coverage
        if poetry run pytest --cov=src --cov-report=term-missing --cov-fail-under=80 > /dev/null 2>&1; then
            COVERAGE=$(poetry run pytest --cov=src --cov-report=term 2>&1 | grep "TOTAL" | awk '{print $NF}' || echo "N/A")
            echo -e "${GREEN}✅ Test coverage: $COVERAGE (meets 80% requirement)${NC}"
        else
            COVERAGE=$(poetry run pytest --cov=src --cov-report=term 2>&1 | grep "TOTAL" | awk '{print $NF}' || echo "N/A")
            echo -e "${RED}❌ Test coverage: $COVERAGE (below 80% requirement)${NC}"
            COMPLIANCE_PASSED=false
        fi
    else
        echo -e "${YELLOW}⚠️  Tests not configured or test directory missing${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Poetry not found - skipping test coverage check${NC}"
fi

# Pre-commit hooks check
echo -e "\n${BLUE}🪝 Pre-commit Hooks:${NC}"

if [[ -f ".pre-commit-config.yaml" ]]; then
    echo -e "${GREEN}✅ Pre-commit config exists${NC}"

    if command -v pre-commit &> /dev/null; then
        if poetry run pre-commit run --all-files > /dev/null 2>&1; then
            echo -e "${GREEN}✅ All pre-commit hooks passing${NC}"
        else
            echo -e "${RED}❌ Pre-commit hook failures detected${NC}"
            COMPLIANCE_PASSED=false
        fi
    else
        echo -e "${YELLOW}⚠️  pre-commit not installed - skipping hook validation${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .pre-commit-config.yaml not found${NC}"
fi

# Code quality checks
echo -e "\n${BLUE}📝 Code Quality:${NC}"

if command -v poetry &> /dev/null; then
    # Black formatting check
    if poetry run black --check . > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Code formatting (black) compliant${NC}"
    else
        echo -e "${YELLOW}⚠️  Code formatting issues found (black)${NC}"
    fi

    # Ruff linting check
    if poetry run ruff check . > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Linting (ruff) passed${NC}"
    else
        LINT_ISSUES=$(poetry run ruff check . 2>&1 | grep -c "error" || true)
        echo -e "${YELLOW}⚠️  $LINT_ISSUES linting issues found (ruff)${NC}"
    fi

    # MyPy type checking
    if [[ -d "src" ]]; then
        if poetry run mypy src > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Type checking (mypy) passed${NC}"
        else
            TYPE_ERRORS=$(poetry run mypy src 2>&1 | grep -c "error" || true)
            echo -e "${YELLOW}⚠️  $TYPE_ERRORS type errors found (mypy)${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Poetry not found - skipping code quality checks${NC}"
fi

# Documentation check
echo -e "\n${BLUE}📚 Documentation:${NC}"

# Check README has minimum content
if [[ -f "README.md" ]]; then
    LINES=$(wc -l < README.md)
    if [[ $LINES -gt 10 ]]; then
        echo -e "${GREEN}✅ README.md has substantial content ($LINES lines)${NC}"
    else
        echo -e "${YELLOW}⚠️  README.md appears minimal ($LINES lines)${NC}"
    fi
fi

# Check for API documentation
if [[ -d "docs" ]] || [[ -f "mkdocs.yml" ]] || [[ -f "docs/index.md" ]]; then
    echo -e "${GREEN}✅ Documentation directory/config found${NC}"
else
    echo -e "${YELLOW}⚠️  No dedicated documentation found${NC}"
fi

# HTTPS check
echo -e "\n${BLUE}🌐 HTTPS Configuration:${NC}"

# Check if project URLs use HTTPS
if [[ -f "pyproject.toml" ]]; then
    if grep -q "https://" pyproject.toml; then
        echo -e "${GREEN}✅ Project URLs use HTTPS${NC}"
    else
        echo -e "${YELLOW}⚠️  Some URLs may not use HTTPS${NC}"
    fi
fi

# Version control check
echo -e "\n${BLUE}📦 Version Control:${NC}"

if [[ -d ".git" ]]; then
    echo -e "${GREEN}✅ Git repository initialized${NC}"

    # Check for .gitignore
    if [[ -f ".gitignore" ]]; then
        echo -e "${GREEN}✅ .gitignore present${NC}"
    else
        echo -e "${YELLOW}⚠️  .gitignore missing${NC}"
    fi

    # Check for branch protection (requires GitHub CLI)
    if command -v gh &> /dev/null; then
        if gh api repos/:owner/:repo/branches/main/protection > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Branch protection enabled${NC}"
        else
            echo -e "${YELLOW}⚠️  Branch protection not configured${NC}"
        fi
    fi
else
    echo -e "${RED}❌ Not a git repository${NC}"
    COMPLIANCE_PASSED=false
fi

# Summary
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [[ "$COMPLIANCE_PASSED" == true ]]; then
    echo -e "${GREEN}✨ OpenSSF Compliance Check: PASSED${NC}"
    echo -e "${GREEN}Your project meets core OpenSSF best practices!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  OpenSSF Compliance Check: FAILED${NC}"
    echo -e "${YELLOW}Some critical requirements are not met. Please address the issues above.${NC}"
    exit 1
fi
