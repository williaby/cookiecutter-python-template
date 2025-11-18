{% if cookiecutter.include_api_framework == "yes" -%}
"""API package for {{ cookiecutter.project_name }}.

This package contains FastAPI routers and API-related functionality.
"""

from __future__ import annotations

from {{ cookiecutter.project_slug }}.api.health import router as health_router

__all__ = ["health_router"]
{% endif -%}
