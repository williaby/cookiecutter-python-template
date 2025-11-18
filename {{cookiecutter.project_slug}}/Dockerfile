{% if cookiecutter.include_docker == "yes" -%}
# Multi-stage Dockerfile for {{ cookiecutter.project_name }}
# Optimized for production with security best practices and minimal image size

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:{{ cookiecutter.python_version }}-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies to a virtual environment
# This creates .venv/ which we'll copy to the final stage
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:{{ cookiecutter.python_version }}-slim

# Metadata labels (OCI standard)
LABEL org.opencontainers.image.title="{{ cookiecutter.project_name }}"
LABEL org.opencontainers.image.description="{{ cookiecutter.project_short_description }}"
LABEL org.opencontainers.image.version="{{ cookiecutter.version }}"
LABEL org.opencontainers.image.authors="{{ cookiecutter.author_name }} <{{ cookiecutter.author_email }}>"
LABEL org.opencontainers.image.url="{{ cookiecutter.repo_url }}"
LABEL org.opencontainers.image.source="{{ cookiecutter.repo_url }}"
LABEL org.opencontainers.image.licenses="{{ cookiecutter.license }}"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src

# Switch to non-root user
USER appuser

# Expose port (default for FastAPI/web apps)
{% if cookiecutter.include_api_framework == "yes" -%}
EXPOSE 8000
{% else -%}
# EXPOSE 8000  # Uncomment if running a web server
{% endif -%}

{% if cookiecutter.include_api_framework == "yes" -%}
# Health check - adjust endpoint based on your app
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default command - run web server
CMD ["uvicorn", "{{ cookiecutter.project_slug }}.main:app", "--host", "0.0.0.0", "--port", "8000"]
{% elif cookiecutter.include_cli == "yes" -%}
# Default command - run CLI (override as needed)
ENTRYPOINT ["{{ cookiecutter.cli_tool_name }}"]
CMD ["--help"]
{% else -%}
# Default command - adjust based on your application
CMD ["python", "-m", "{{ cookiecutter.project_slug }}"]
{% endif -%}

# =============================================================================
# Build Arguments (optional, for build-time configuration)
# =============================================================================
# Example:
# ARG BUILD_ENV=production
# ENV ENVIRONMENT=${BUILD_ENV}

# =============================================================================
# Multi-architecture support
# =============================================================================
# Build for multiple platforms:
# docker buildx build --platform linux/amd64,linux/arm64 -t myimage:latest .
{% endif -%}
