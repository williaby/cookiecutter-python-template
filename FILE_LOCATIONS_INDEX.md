# File Locations Index
## Centralized Code Pattern Analysis

### All File Paths (Absolute for Reference)

```
/home/user/cookiecutter-python-template/{{cookiecutter.project_slug}}/
```

---

## Security & Authentication Patterns

### 1. Security Middleware
**Path:** `/src/{{cookiecutter.project_slug}}/middleware/security.py`
**Lines:** 39-100 (SecurityHeadersMiddleware)
**Pattern:** OWASP security headers implementation
**Status:** ✅ Code exists, needs centralization
**Move To:** `api-gateway/`

**Key Code:**
```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    # Implements: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
    # HSTS, CSP, Referrer-Policy, Permissions-Policy
```

### 2. Rate Limiting Middleware
**Path:** `/src/{{cookiecutter.project_slug}}/middleware/security.py`
**Lines:** 102-173 (RateLimitMiddleware)
**Pattern:** In-memory rate limiter (per-IP)
**Status:** ⚠️ Not production-ready (in-memory only)
**Move To:** `api-gateway/` with Redis backend

**Key Code:**
```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        self.requests: dict[str, list[float]] = defaultdict(list)
```

### 3. SSRF Prevention Middleware
**Path:** `/src/{{cookiecutter.project_slug}}/middleware/security.py`
**Lines:** 176-225 (SSRFPreventionMiddleware)
**Pattern:** Block internal IP ranges and cloud metadata
**Status:** ✅ Code exists, needs enhancement
**Move To:** `api-gateway/` or `security-service/`

**Key Code:**
```python
class SSRFPreventionMiddleware(BaseHTTPMiddleware):
    BLOCKED_HOSTS = {"localhost", "127.0.0.1", "169.254.169.254"}
    BLOCKED_RANGES = ["10.", "172.16.", "192.168."]
```

### 4. Security Middleware Composition
**Path:** `/src/{{cookiecutter.project_slug}}/middleware/security.py`
**Lines:** 228-327 (add_security_middleware function)
**Pattern:** Middleware factory/composition pattern
**Status:** ✅ Good example for centralization

### 5. CORS Middleware
**Path:** `/src/{{cookiecutter.project_slug}}/middleware/security.py`
**Lines:** 273-282
**Pattern:** Starlette CORSMiddleware configuration
**Status:** ✅ Per-project (should be centralized)
**Move To:** `api-gateway/`

### 6. User Context Tracking
**Path:** `/src/{{cookiecutter.project_slug}}/core/sentry.py`
**Lines:** 330-368 (set_user_context function)
**Pattern:** Associate errors with users in Sentry
**Status:** ✅ Code exists, should move to audit-logging-service
**Related Functions:** 
- `capture_exception()` (Lines 240-282)
- `capture_message()` (Lines 285-327)
- `add_breadcrumb()` (Lines 370-403)

---

## Notification & Email Patterns

### 1. Email Sending Task
**Path:** `/src/{{cookiecutter.project_slug}}/jobs/worker.py`
**Lines:** 80-109 (send_email_task function)
**Pattern:** Background job for email sending
**Status:** ⚠️ TODO implementation (skeleton only)
**Missing:** Email provider integration, templates, tracking
**Move To:** `notification-service/`

**Key Code:**
```python
async def send_email_task(
    ctx: dict[str, Any],
    recipient: str,
    subject: str,
    body: str,
) -> dict:
    """Send email asynchronously."""
    logger.info("sending_email", recipient=recipient, subject=subject)
    # TODO: Integrate with your email provider
```

### 2. File Processing Task
**Path:** `/src/{{cookiecutter.project_slug}}/jobs/worker.py`
**Lines:** 112-146 (process_file_upload function)
**Pattern:** Background job for file processing
**Status:** ⚠️ Skeleton implementation
**Missing:** S3 integration, virus scanning, encryption
**Move To:** `storage-service/`

### 3. Task Enqueuing
**Path:** `/src/{{cookiecutter.project_slug}}/jobs/worker.py`
**Lines:** 262-291 (enqueue_task function)
**Pattern:** Wrapper for ARQ job enqueueing
**Status:** ✅ Basic implementation works
**Move To:** Could be part of `job-queue-service/` or `notification-service/`

### 4. Background Task Scheduling
**Path:** `/src/{{cookiecutter.project_slug}}/jobs/worker.py`
**Lines:** 231-233 (WorkerSettings.cron_jobs)
**Pattern:** Cron job configuration
**Example:** `cron(cleanup_old_data, hour=2, minute=0)`
**Move To:** `job-queue-service/` or keep per-project

### 5. Worker Lifecycle Hooks
**Path:** `/src/{{cookiecutter.project_slug}}/jobs/worker.py`
**Lines:** 178-210 (startup and shutdown functions)
**Pattern:** Worker initialization and cleanup
**Status:** ✅ Good pattern to follow

---

## Configuration & Secrets Patterns

### 1. Settings/Configuration
**Path:** `/src/{{cookiecutter.project_slug}}/core/config.py`
**Lines:** 1-35 (Settings class)
**Pattern:** Pydantic Settings with environment variables
**Status:** ✅ Good pattern, but static only
**Missing:** Dynamic config, feature flags, secrets
**Move To:** Keep as base, integrate with `config-service/` and `secrets-manager/`

**Key Code:**
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="{{ cookiecutter.project_slug }}_",
        case_sensitive=False,
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = False
    include_timestamp: bool = True

settings = Settings()
```

### 2. Sentry Configuration
**Path:** `/src/{{cookiecutter.project_slug}}/core/sentry.py`
**Lines:** 36-150 (init_sentry function)
**Pattern:** SDK initialization with hooks and integrations
**Status:** ✅ Excellent pattern to follow
**Why Good:** Hooks, PII filtering, environment-aware

---

## Logging & Audit Patterns

### 1. Structured Logging Setup
**Path:** `/src/{{cookiecutter.project_slug}}/utils/logging.py`
**Lines:** 23-97 (setup_logging function)
**Pattern:** Structlog configuration with JSON/Rich output
**Status:** ✅ Good foundation
**Missing:** Audit-specific events, compliance logging

**Key Code:**
```python
def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure structured logging for the application."""
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        # ... more processors
    ]
    
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
```

### 2. Logger Getter
**Path:** `/src/{{cookiecutter.project_slug}}/utils/logging.py`
**Lines:** 100-123 (get_logger function)
**Pattern:** Factory for structured loggers
**Usage:** `logger = get_logger(__name__)`

### 3. Performance Logging
**Path:** `/src/{{cookiecutter.project_slug}}/utils/logging.py`
**Lines:** 126-162 (log_performance function)
**Pattern:** Standardized performance metric logging
**Status:** ✅ Should be centralized
**Move To:** Central metrics service

**Key Code:**
```python
def log_performance(
    logger: Any,
    operation: str,
    duration_ms: float,
    success: bool = True,
    **context: Any,
) -> None:
    """Log performance metrics for an operation."""
    logger.info(
        "performance",
        operation=operation,
        duration_ms=round(duration_ms, 2),
        success=success,
        **context,
    )
```

### 4. Financial Audit Logging
**Path:** `/src/{{cookiecutter.project_slug}}/utils/financial.py`
**Lines:** 130-146 (format_audit_amount function)
**Pattern:** Format amounts for audit logs
**Status:** ⚠️ Very minimal (just formatting)
**Missing:** Actual transaction audit logging
**Move To:** `audit-logging-service/`

---

## API Health Check Patterns

### 1. Liveness Probe
**Path:** `/src/{{cookiecutter.project_slug}}/api/health.py`
**Lines:** 64-82 (liveness function)
**Pattern:** Simple "is app running?" check
**Status:** ✅ Good K8s pattern

### 2. Readiness Probe
**Path:** `/src/{{cookiecutter.project_slug}}/api/health.py`
**Lines:** 188-233 (readiness function)
**Pattern:** Check all critical dependencies
**Status:** ✅ Good foundation
**Checks Included:** Database, cache (optional), external services (optional)

### 3. Database Health Check
**Path:** `/src/{{cookiecutter.project_slug}}/api/health.py`
**Lines:** 86-114 (check_database function)
**Pattern:** Test database connectivity with latency
**Status:** ✅ Good pattern

### 4. Cache Health Check
**Path:** `/src/{{cookiecutter.project_slug}}/api/health.py`
**Lines:** 118-144 (check_cache function)
**Pattern:** Redis/cache connectivity check
**Status:** ✅ Placeholder (needs implementation)

### 5. External Service Health Check
**Path:** `/src/{{cookiecutter.project_slug}}/api/health.py`
**Lines:** 147-175 (check_external_service function)
**Pattern:** HTTP check against external API
**Status:** ⚠️ TODO (needs actual HTTP client)
**Missing:** Retry logic, circuit breaker, caching

---

## Caching Patterns

### 1. Redis Connection Pool
**Path:** `/src/{{cookiecutter.project_slug}}/core/cache.py`
**Lines:** 54-86 (get_redis function)
**Pattern:** Global Redis connection management
**Status:** ✅ Good pattern (singleton pool)

### 2. Caching Decorator
**Path:** `/src/{{cookiecutter.project_slug}}/core/cache.py`
**Lines:** 107-177 (cached decorator)
**Pattern:** Function result caching with TTL
**Usage:** `@cached(ttl=300, key_prefix="user")`
**Status:** ✅ Good pattern to follow

### 3. Cache Invalidation Decorator
**Path:** `/src/{{cookiecutter.project_slug}}/core/cache.py`
**Lines:** 180-212 (cache_invalidate decorator)
**Pattern:** Automatic cache invalidation on updates
**Usage:** `@cache_invalidate("user:*")`
**Status:** ✅ Good pattern

### 4. Cache Operations
**Path:** `/src/{{cookiecutter.project_slug}}/core/cache.py`
**Lines:** 220-318
**Functions:**
- `get_cached()` - Get value from cache
- `set_cached()` - Set value in cache
- `delete_cached()` - Delete from cache
- `invalidate_pattern()` - Invalidate by pattern
**Status:** ✅ Good utilities

### 5. Cache Warming
**Path:** `/src/{{cookiecutter.project_slug}}/core/cache.py`
**Lines:** 326-373 (warm_cache function)
**Pattern:** Pre-load cache on startup
**Usage:** Warm popular data before serving traffic
**Status:** ✅ Good pattern

### 6. Cache Statistics
**Path:** `/src/{{cookiecutter.project_slug}}/core/cache.py`
**Lines:** 429-453 (get_cache_stats function)
**Pattern:** Get hit rate, memory usage, connected clients
**Status:** ✅ Good for monitoring

---

## Central Services Directory

**Path:** `/central-services/`

### Existing Service Documentation
1. **Auth Service**: `/central-services/auth-service/README.md`
2. **Notification Service**: `/central-services/notification-service/README.md`
3. **Config Service**: `/central-services/config-service/` (skeleton)
4. **Feature Flags**: `/central-services/feature-flags/` (skeleton)
5. **API Gateway**: `/central-services/api-gateway/` (skeleton)
6. **Secrets Manager**: `/central-services/secrets-manager/` (skeleton)
7. **Storage Service**: `/central-services/storage-service/` (skeleton)

---

## Command Line Interface

### 1. CLI Entry Point
**Path:** `/src/{{cookiecutter.project_slug}}/cli.py`
**Lines:** 1-94
**Pattern:** Click-based CLI with structured logging integration
**Status:** ✅ Good pattern
**Example:** Tool name configured in cookiecutter

---

## Summary by Priority

### IMMEDIATE (Files ready to move/centralize)
1. **SecurityHeadersMiddleware** - `/middleware/security.py:39-100`
2. **RateLimitMiddleware** - `/middleware/security.py:102-173`
3. **SSRFPreventionMiddleware** - `/middleware/security.py:176-225`
4. **CORS Middleware** - `/middleware/security.py:273-282`
5. **Sentry Integration** - `/core/sentry.py` (full file)
6. **Structured Logging** - `/utils/logging.py:23-97`
7. **Cache Pattern** - `/core/cache.py` (full file - good reference)

### HIGH PRIORITY (Needs implementation in central services)
1. **Email Sending** - `/jobs/worker.py:80-109`
2. **File Processing** - `/jobs/worker.py:112-146`
3. **Configuration** - `/core/config.py` (integrate with config-service)

### MEDIUM PRIORITY (Good patterns, enhance with central service)
1. **Health Checks** - `/api/health.py` (use as base)
2. **Performance Logging** - `/utils/logging.py:126-162`
3. **Financial Audit** - `/utils/financial.py:130-146`

### REFERENCE ONLY (Follow these patterns)
1. **Sentry Initialization** - `/core/sentry.py:36-150`
2. **Worker Lifecycle** - `/jobs/worker.py:178-210`
3. **Cache Management** - `/core/cache.py` (full file)

---

**Last Updated:** 2025-11-18
**Total Files Analyzed:** 12
**Total Code Patterns Identified:** 35+
