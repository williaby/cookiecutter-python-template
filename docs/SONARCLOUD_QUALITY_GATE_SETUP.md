# SonarCloud Quality Gate Setup Guide

Quick reference for setting up the custom "LLM-Aware" quality gate in SonarCloud.

## Prerequisites

- SonarCloud account with admin access to your organization
- Project already created in SonarCloud
- `SONAR_TOKEN` configured in GitHub repository secrets

## Step 1: Access Quality Gates

1. Log in to [SonarCloud](https://sonarcloud.io)
2. Navigate to **Organization Settings** (top right) → **Quality Gates**
3. Click **Create** button to create a new quality gate

## Step 2: Create LLM-Aware Quality Gate

### Basic Information

- **Name**: `LLM-Aware`
- **Description**: `Enhanced quality gate with LLM governance integration`

### Quality Gate Conditions

Add the following conditions by clicking **Add Condition**:

#### BLOCKER Conditions (Must Pass)

These conditions will block PRs:

| Metric | Operator | Threshold | Description |
|--------|----------|-----------|-------------|
| Security Rating | is worse than | A | Security vulnerabilities |
| Reliability Rating | is worse than | A | Bug detection |
| Security Hotspots Reviewed | is less than | 100% | All hotspots must be reviewed |

#### CRITICAL Conditions (Should Pass)

These conditions indicate serious issues:

| Metric | Operator | Threshold | Description |
|--------|----------|-----------|-------------|
| Coverage on New Code | is less than | 80% | Test coverage |
| Duplicated Lines (%) on New Code | is greater than | 3% | Code duplication |
| Maintainability Rating on New Code | is worse than | A | Code smells |

#### LLM Governance Conditions

Map these to LLM tags (handled by CI script):

| Metric | Operator | Threshold | Maps To |
|--------|----------|-----------|---------|
| Code Smells (CRITICAL) | is greater than | 0 | #LLM-LOGIC |
| Duplicated Blocks | is greater than | 0 | #LLM-SCAFFOLD |
| Coverage | is less than | {{ cookiecutter.code_coverage_target }}% | #LLM-TEST-FIRST |

## Step 3: Set as Default (Optional)

- Toggle **Set as Default** if you want all projects to use this gate
- Or apply per-project (see Step 4)

## Step 4: Apply to Project

### Option A: During Project Creation

1. Create new project in SonarCloud
2. Select **LLM-Aware** quality gate during setup

### Option B: For Existing Project

1. Go to **Project Settings** → **Quality Gate**
2. Select **LLM-Aware** from dropdown
3. Click **Save**

## Step 5: Verify Configuration

Test the quality gate:

```bash
# Run local analysis (requires SONAR_TOKEN)
export SONAR_TOKEN="your-token-here"

# For template repository
python scripts/check_quality_gate.py \
  --project-key "williaby_cookiecutter-python-template" \
  --org "williaby"

# For generated projects (from project root)
python scripts/check_quality_gate.py
```

Expected output:
```
================================================================================
THREE-LAYER GOVERNANCE REPORT
================================================================================

Layer 1: Production Runtime Risks (RAD)
--------------------------------------------------------------------------------
✅ No unverified production risk tags
   Status: PASSED

Layer 2: LLM Development Debt
--------------------------------------------------------------------------------
✅ No unverified LLM debt tags
   Status: PASSED

Layer 3: Automated Code Quality (SonarQube)
--------------------------------------------------------------------------------
✅ Quality Gate: PASSED
...
```

## Step 6: Configure CI Integration

The quality gate is automatically enforced by the CI workflow:

### For Template Repository

File: `.github/workflows/ci.yml`

```yaml
sonarqube-quality-gate:
  name: SonarQube Quality Gate
  runs-on: ubuntu-latest
  steps:
    - name: SonarCloud Scan
      uses: SonarSource/sonarcloud-github-action@master
      env:
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

    - name: Wait for Quality Gate
      uses: sonarsource/sonarqube-quality-gate-action@master
      env:
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

### For Generated Projects

Quality gate enforcement is automatically included when:
- `include_github_actions: yes`
- `include_sonarcloud: yes`

## Troubleshooting

### Quality Gate Not Appearing

**Problem**: Custom quality gate doesn't show up in project settings

**Solution**:
1. Verify you have admin access to the organization
2. Check quality gate is not set as "Default" if you want per-project control
3. Clear browser cache and reload SonarCloud

### Quality Gate Always Passes

**Problem**: Quality gate passes even with obvious issues

**Solution**:
1. Check that conditions are correctly configured (use "is less than" for thresholds that should be low)
2. Verify coverage data is being uploaded (check `coverage.xml` artifact)
3. Run analysis with verbose logging: `sonar-scanner -X`

### Quality Gate Blocks Valid PRs

**Problem**: Quality gate fails for legitimate changes

**Solution**:
1. Review specific conditions that failed in SonarCloud dashboard
2. Check if thresholds are too strict for your project
3. Consider adjusting thresholds (e.g., 80% → 70% coverage)
4. Temporarily disable specific conditions during migration

### CI Workflow Fails

**Problem**: CI workflow fails with "SONAR_TOKEN not found"

**Solution**:
1. Add `SONAR_TOKEN` to repository secrets:
   - Go to repository **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `SONAR_TOKEN`
   - Value: Your SonarCloud token (from [https://sonarcloud.io/account/security](https://sonarcloud.io/account/security))

2. For organization-level token (recommended for multiple repos):
   - Go to organization **Settings** → **Secrets and variables** → **Actions**
   - Create organization-level secret
   - All repos in org can use it

## Advanced Configuration

### Custom Rules for LLM Patterns

Add custom rules to detect LLM-specific patterns:

1. **Organization Settings** → **Quality Profiles** → **Python**
2. Click **Activate More** → **Custom Rules**
3. Create rules for:
   - Hardcoded localhost URLs
   - Mock data patterns
   - TODO-LLM-* tag detection

### Notification Configuration

Set up notifications for quality gate failures:

1. **Project Settings** → **Webhooks**
2. Add webhook URL (Slack, Discord, etc.)
3. Select events: "Quality Gate Status Changed"

### Integration with GitHub Status Checks

Enable PR decoration:

1. **Organization Settings** → **GitHub Integration**
2. Enable **Automatic Analysis** (optional)
3. Enable **PR Decoration**
4. Configure status check: "SonarCloud Quality Gate"

## Quality Gate Maintenance

### Monthly Review

- Review failed quality gate conditions
- Adjust thresholds based on team velocity
- Update rules for new LLM patterns

### When to Update

Update quality gate when:
- New SonarQube rules are released
- Team agrees on new standards
- LLM patterns evolve
- Coverage targets change

## Reference Links

- [SonarCloud Quality Gates Documentation](https://docs.sonarcloud.io/improving/quality-gates/)
- [SonarQube Metrics Reference](https://docs.sonarqube.org/latest/user-guide/metric-definitions/)
- [GitHub Actions Integration](https://docs.sonarcloud.io/advanced-setup/ci-based-analysis/github-actions/)

## Quick Commands

```bash
# Check quality gate locally
python scripts/check_quality_gate.py --project-key YOUR_PROJECT_KEY

# View quality gate status (requires jq)
curl -u "${SONAR_TOKEN}:" \
  "https://sonarcloud.io/api/qualitygates/project_status?projectKey=YOUR_PROJECT_KEY" | jq

# List all quality gates
curl -u "${SONAR_TOKEN}:" \
  "https://sonarcloud.io/api/qualitygates/list" | jq

# Get project quality gate
curl -u "${SONAR_TOKEN}:" \
  "https://sonarcloud.io/api/qualitygates/get_by_project?project=YOUR_PROJECT_KEY" | jq
```

---

**Next Steps:**
1. Create quality gate in SonarCloud UI (Steps 1-2)
2. Apply to your project (Step 4)
3. Test with `check_quality_gate.py` script (Step 5)
4. Push changes to trigger CI workflow (Step 6)
