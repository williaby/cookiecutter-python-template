## Summary

<!-- Brief description: what changed and why -->
@coderabbitai summary

{%- if cookiecutter.include_linear == "yes" %}

## Linear Issue

Closes {{ cookiecutter.linear_team_key }}-
{%- endif %}

## Changes

<!-- Key changes made (delete examples) -->
- **Component**: What changed and why

## Impact

<!-- Expected outcomes (fill in the blanks) -->
- ✅
- ✅ No breaking changes

## Testing

- [ ] Tests pass (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check`)
{%- if cookiecutter.include_api_framework == "yes" %}
- [ ] API manually tested
{%- endif %}

## Notes

<!-- Optional: anything reviewers should know, known issues, follow-up work -->

---
{%- if cookiecutter.include_coderabbit == "yes" %}
<!-- @coderabbitai review -->
{%- endif %}
