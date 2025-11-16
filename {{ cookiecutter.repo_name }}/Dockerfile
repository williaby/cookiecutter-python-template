FROM python:3.11-slim AS builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements with hash verification
COPY requirements-docker.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-docker.txt

# Multi-stage build for minimal final image
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r promptcraft && \
    useradd -r -g promptcraft -u 1000 -m -s /bin/bash promptcraft

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R promptcraft:promptcraft /app

WORKDIR /app

# Copy application code
COPY --chown=promptcraft:promptcraft src/ ./src/
COPY --chown=promptcraft:promptcraft config/ ./config/

# Security: Don't run as root
USER promptcraft

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Set security headers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Expose only necessary port
EXPOSE 7860

# Run with minimal privileges
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
