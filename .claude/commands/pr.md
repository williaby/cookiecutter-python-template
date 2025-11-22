# Prepare Pull Request

Analyze the current branch and prepare a PR description for template changes.

## Instructions

1. **Gather context** by running these commands:
   - `git status` - see uncommitted changes
   - `git log $(git merge-base HEAD main)..HEAD --oneline` - commits on this branch
   - `git diff $(git merge-base HEAD main)..HEAD --stat` - files changed summary

2. **Analyze the changes**:
   - What template components were modified?
   - Does this affect `cookiecutter.json`, hooks, or generated files?
   - Are there breaking changes for existing template users?
   - What testing was done?

3. **Generate PR description** using this template:

```markdown
## Summary

<!-- Brief description: what changed and why -->

## Changes

- **Component**: What changed and why

## Scope

- [ ] `cookiecutter.json` (template variables)
- [ ] `hooks/` (generation hooks)
- [ ] `{{cookiecutter.project_slug}}/` (generated project files)
- [ ] Documentation

## Impact

- ✅ [Key benefit]
- ✅ No breaking changes to existing template users

## Testing

- [ ] Template generates with default options
- [ ] Template generates with features enabled
- [ ] Generated project passes `uv run pytest`

## Notes

<!-- Optional: known issues, follow-up work -->
```

4. **Output the PR description** ready to copy-paste.

5. **Suggest a PR title** following conventional commits:
   - `feat(template):` for new template features
   - `fix(hooks):` for hook fixes
   - `docs:` for documentation
   - `chore(config):` for cookiecutter.json changes

## Scope Guidance

| Files Changed | Scope |
|--------------|-------|
| `cookiecutter.json` | config |
| `hooks/*.py` | hooks |
| `{{cookiecutter.project_slug}}/**` | template |
| `README.md`, `CLAUDE.md` | docs |
| `.github/workflows/**` | ci |
