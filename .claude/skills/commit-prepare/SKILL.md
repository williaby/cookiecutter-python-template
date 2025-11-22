---
name: commit-prepare
description: "Prepare git commit messages for template changes following conventional commits. Activates on: commit, prepare commit, commit this, commit message, ready to commit"
---

# Commit Preparation Skill

Automatically prepare commit messages for template changes following conventional commits.

## Activation

This skill activates on keywords:
- "commit", "prepare commit", "commit this"
- "commit message", "write commit"
- "stage and commit", "commit changes"
- "what should I commit", "ready to commit"

## Workflow

When activated, follow these steps:

### 1. Gather Context

```bash
git status
git diff --cached --stat
git diff --cached
```

### 2. Analyze Template Changes

Determine:
- **Type**: feat, fix, docs, refactor, chore
- **Scope**: template, hooks, config, ci, docs
- **Impact**: Breaking changes for template users?

### 3. Generate Commit Message

#### Template-Specific Scopes

| Scope | Files | Example |
|-------|-------|---------|
| `template` | `{{cookiecutter.project_slug}}/**` | `feat(template): add health checks` |
| `hooks` | `hooks/*.py` | `fix(hooks): handle Windows paths` |
| `config` | `cookiecutter.json` | `chore(config): add new variable` |
| `ci` | `.github/workflows/**` | `ci: add template validation` |
| `docs` | `*.md`, `docs/**` | `docs: update installation guide` |

#### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type Reference

| Type | When to Use |
|------|-------------|
| `feat` | New template feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `refactor` | Code restructuring |
| `chore` | Maintenance |
| `ci` | CI/CD changes |

### 4. Handle Staging

Ask about unstaged changes:
- Stage all? (`git add .`)
- Stage specific files?
- Review first?

### 5. Execute Commit

Use HEREDOC for multi-line:

```bash
git commit -m "$(cat <<'EOF'
feat(template): add CodeRabbit integration

Add .coderabbit.yaml configuration for AI code reviews.
Includes path-specific review instructions and auto-labeling.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 6. Post-Commit

- Show `git log -1`
- Suggest push if ready
- Note remaining changes

## Example Interactions

### Template Change

**User**: "commit this"

**Claude**: Staged changes:
- `{{cookiecutter.project_slug}}/.coderabbit.yaml` (new)
- `cookiecutter.json` (modified)
- `hooks/post_gen_project.py` (modified)

**Suggested commit**:
```
feat(template): add CodeRabbit AI code review integration

- Add .coderabbit.yaml with assertive profile
- Add include_coderabbit feature flag
- Add cleanup logic in post-gen hook
```

---

### Hook Fix

**User**: "ready to commit"

**Claude**: Staged: `hooks/post_gen_project.py`

**Suggested commit**:
```
fix(hooks): handle missing directory gracefully

Check if directory exists before attempting removal.
Prevents error when optional features are disabled.
```

---

### Multiple Scopes

**User**: "commit these changes"

**Claude**: Changes span multiple areas:
1. `cookiecutter.json` - new variables
2. `hooks/post_gen_project.py` - cleanup logic
3. `README.md` - documentation

Recommend separate commits:
1. `chore(config): add Linear integration variables`
2. `feat(hooks): add Linear cleanup logic`
3. `docs: document Linear integration`

Or single combined commit:
```
feat: add Linear project management integration

- Add include_linear and linear_team_key variables
- Add cleanup logic for Linear files
- Document setup in README
```

Which approach?

## Attribution

Always include:
```
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Safety Rules

- **Never** amend pushed commits
- **Never** skip hooks without explicit request
- **Always** show message before executing
- **Check** for hardcoded paths/usernames
- **Verify** Jinja2 syntax in template files
