# Code Patterns for Centralization: Analysis Report
## Cookiecutter Python Template

### Executive Summary

This analysis identifies **7 major categories** of code patterns that would benefit from centralization into shared services. The template already demonstrates excellent planning through the `central-services/` directory, which outlines the intended architecture. Below is a detailed breakdown of existing patterns with specific locations and recommendations.

---

## 1. AUTHENTICATION/AUTHORIZATION CODE

### Current State: Minimal (Template Phase)
The main template currently provides **security middleware** but no explicit authentication implementation, expecting projects to integrate with the planned central auth service.

**Files Analyzed:**
- `/middleware/security.py` - Security headers, rate limiting, SSRF prevention
- `/core/sentry.py` - User context tracking (partial)

### Identified Patterns:

#### Pattern 1.1: Security Headers Middleware
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/middleware/security.py` (Lines 39-100)

```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        # X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
        # HSTS, CSP, Referrer-Policy, Permissions-Policy
        return response
```

**Why Centralize:**
- Duplicated across every project
- Headers should be consistent organization-wide
- Easier to update security policies centrally

**Central Service:** `auth-service/` or new `security-service/`

---

#### Pattern 1.2: Rate Limiting Middleware
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/middleware/security.py` (Lines 102-173)

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Track per-IP rate limiting
        # Return 429 if exceeded
```

**Why Centralize:**
- Currently in-memory (lost on restart)
- Should be backed by Redis for production
- Better suited in API Gateway service
- Prevents per-project inconsistencies

**Central Service:** `api-gateway/` with Redis backend

**Suggested Implementation:**
```python
# Central service would provide:
# - Distributed rate limiting (Redis-backed)
# - Multiple strategies (fixed-window, sliding-window, token-bucket)
# - Per-user, per-IP, per-endpoint customization
# - Analytics and alerting
```

---

#### Pattern 1.3: User Context in Error Tracking
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/core/sentry.py` (Lines 330-368)

```python
def set_user_context(
    user_id: str | None = None,
    email: str | None = None,
    username: str | None = None,
    **kwargs: Any,
) -> None:
    """Set user context for error tracking."""
    user_data = {}
    # ... build user data
    sentry_sdk.set_user(user_data)
```

**Why Centralize:**
- Pattern shows user tracking for compliance
- Should be part of centralized audit logging
- Needs standardization across org

**Central Service:** Future `audit-logging-service/`

---

## 2. EMAIL/NOTIFICATION CODE

### Current State: Placeholder Pattern
The template provides a **skeleton** for notification patterns in the background jobs module.

**Files Analyzed:**
- `/jobs/worker.py` - Background job patterns including `send_email_task()`
- `/central-services/notification-service/` - Documentation and design

### Identified Patterns:

#### Pattern 2.1: Email Sending via Background Jobs
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/jobs/worker.py` (Lines 80-109)

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
    # Example with SendGrid, AWS SES, etc.
    # await send_email_via_provider(recipient, subject, body)
    
    await asyncio.sleep(1)  # Simulate email sending
    return {"status": "sent", "recipient": recipient, ...}
```

**Why Centralize:**
- Email provider integration duplicated per project
- Template handling should be centralized
- Delivery tracking across org
- Compliance (unsubscribe, opt-out) management
- Cost optimization and reputation management

**Current Pattern Issues:**
- No template management
- No delivery tracking
- No unsubscribe handling
- No rate limiting
- No A/B testing

**Central Service:** `notification-service/` (Already documented)

---

#### Pattern 2.2: Task Enqueuing Pattern
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/jobs/worker.py` (Lines 262-291)

```python
async def enqueue_task(
    redis: ArqRedis,
    task_name: str,
    *args: Any,
    **kwargs: Any,
) -> str:
    """Enqueue a background task."""
    job = await redis.enqueue_job(task_name, *args, **kwargs)
    logger.info("task_enqueued", task=task_name, job_id=job.job_id)
    return job.job_id
```

**Why Centralize:**
- Job queue infrastructure should be shared
- Monitoring and alerting for failures
- Retry strategies standardized
- Dead letter queue management

**Central Service:** Could be part of `notification-service/` or separate `job-queue-service/`

---

## 3. FILE UPLOAD/STORAGE CODE

### Current State: Not Implemented
The template does NOT currently include S3 or file upload patterns, but the `central-services/` directory contains a placeholder for this.

**Files Analyzed:**
- `/jobs/worker.py` - Contains `process_file_upload()` placeholder (Lines 112-146)

### Identified Patterns:

#### Pattern 3.1: File Upload Processing
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/jobs/worker.py` (Lines 112-146)

```python
async def process_file_upload(
    ctx: dict[str, Any],
    file_id: str,
    file_path: str,
) -> dict:
    """Process uploaded file in background."""
    logger.info("processing_file", file_id=file_id, path=file_path)
    
    try:
        # Example: Read and process file
        # with open(file_path, 'rb') as f:
        #     data = f.read()
        #     # Process data...
        
        await asyncio.sleep(3)  # Simulate processing
        return {
            "status": "completed",
            "file_id": file_id,
            "processed_at": datetime.utcnow().isoformat(),
            "records_processed": 1000,
        }
    except Exception as e:
        logger.error("file_processing_failed", file_id=file_id, error=str(e))
        raise
```

**Why Centralize:**
- S3 credentials duplicated per project
- File processing logic (virus scanning, validation) should be shared
- CDN integration centralized
- Storage quota management
- Compliance (data retention, encryption)

**Missing from Template:**
1. S3 client initialization
2. File validation logic
3. Virus scanning integration
4. Encryption at rest
5. Access control
6. Audit logging for file operations

**Central Service:** `storage-service/` (Already outlined in central-services/)

---

## 4. EXTERNAL API CLIENTS

### Current State: Not Implemented
The template does NOT provide reusable API client patterns. The external service health check is a placeholder.

**Files Analyzed:**
- `/api/health.py` - Contains `check_external_service()` placeholder (Lines 147-175)
- `/core/sentry.py` - Example of external service integration (Sentry SDK)

### Identified Patterns:

#### Pattern 4.1: External API Health Check Pattern
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/api/health.py` (Lines 147-175)

```python
async def check_external_service() -> ReadinessCheck:
    """Check external API/service connectivity."""
    start = time.time()
    try:
        # Example external service check
        # import httpx
        # async with httpx.AsyncClient() as client:
        #     response = await client.get("https://api.example.com/health", timeout=2.0)
        #     response.raise_for_status()
        
        latency_ms = (time.time() - start) * 1000
        return ReadinessCheck(
            name="external_api",
            status=True,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        # ... error handling
```

**Why Centralize:**
- API client patterns duplicated across projects
- Error handling and retry logic inconsistent
- Authentication to external services scattered
- Rate limiting per external service
- Circuit breaker patterns
- Request/response logging

**Missing Patterns:**
1. HTTP client pooling and reuse
2. Request signing/authentication
3. Retry strategies (exponential backoff)
4. Timeout management
5. Request/response validation
6. Caching strategies
7. Circuit breaker pattern

**Suggested Central Service:**
- Create `api-clients/` service with SDKs for common integrations
- Provide base client class with:
  - Automatic retries
  - Circuit breaker
  - Request signing
  - Response caching
  - Error handling

---

#### Pattern 4.2: Sentry Integration (External Service Example)
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/core/sentry.py` (Full file)

**Existing Implementation Quality:**
```python
def init_sentry(
    dsn: str | None = None,
    environment: str | None = None,
    release: str | None = None,
    traces_sample_rate: float = 0.1,
    # ... more config
) -> None:
    """Initialize Sentry error tracking and performance monitoring."""
    
    integrations: list[Any] = [
        LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        # FastAPI integration
        # SQLAlchemy integration
    ]
    
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        integrations=integrations,
        before_send=before_send_hook,  # Custom filtering
        before_breadcrumb=before_breadcrumb_hook,
    )
```

**Why This IS Good (Already Centralized):**
- ✅ Proper hook architecture for data filtering
- ✅ Configurable per environment
- ✅ Handles PII filtering (GDPR compliance)
- ✅ Structured error context

**For OTHER External APIs:**
- Should follow this pattern
- Centralize SDK initialization
- Provide configuration management
- Handle authentication

---

## 5. COMMON MIDDLEWARE

### Current State: Good Foundation
The template provides excellent middleware examples that can be moved to central service.

**Files Analyzed:**
- `/middleware/security.py` - Complete middleware suite
- `/middleware/__init__.py` - Middleware exports

### Identified Patterns:

#### Pattern 5.1: CORS Middleware
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/middleware/security.py` (Lines 273-282)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or [],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
    max_age=3600,
)
```

**Why Centralize:**
- CORS policies should be org-wide consistent
- Often managed by API Gateway
- Configuration should be centralized
- Easier to update security policies

---

#### Pattern 5.2: SSRF Prevention Middleware
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/middleware/security.py` (Lines 176-225)

```python
class SSRFPreventionMiddleware(BaseHTTPMiddleware):
    """Prevent Server-Side Request Forgery (SSRF) attacks."""
    
    BLOCKED_HOSTS = {
        "localhost", "127.0.0.1", "0.0.0.0",
        "169.254.169.254",  # AWS metadata
        "metadata.google.internal",  # GCP metadata
    }
    
    BLOCKED_RANGES = [
        "10.", "172.16.", "192.168.",  # Private ranges
        "169.254.", "::1", "fc00::",  # Special ranges
    ]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Check for SSRF patterns in query parameters
```

**Why Centralize:**
- SSRF patterns should be consistent org-wide
- Cloud-provider-specific (AWS, GCP, Azure)
- Should be updated when new vulnerabilities discovered
- Better as API Gateway rule

---

#### Pattern 5.3: Security Headers Composition
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/middleware/security.py` (Lines 228-327)

```python
def add_security_middleware(
    app: FastAPI,
    *,
    enable_https_redirect: bool = False,
    enable_rate_limiting: bool = True,
    enable_ssrf_prevention: bool = True,
    allowed_origins: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
    rate_limit_rpm: int = 60,
) -> None:
    """Add all security middleware to FastAPI application."""
    # HTTPS redirect
    # Trusted hosts
    # CORS
    # Security headers
    # Rate limiting
    # SSRF prevention
```

**Why Centralize:**
- Ensures consistent security posture across all projects
- Single point for security updates
- Easier compliance audits
- Can be managed via API Gateway instead

**Central Service:** `api-gateway/` should provide this

---

## 6. CONFIGURATION MANAGEMENT

### Current State: Good Pattern, but Could Be Centralized

**Files Analyzed:**
- `/core/config.py` - Pydantic Settings implementation
- `pyproject.toml` - Project configuration

### Identified Patterns:

#### Pattern 6.1: Settings via Environment Variables
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/core/config.py`

```python
class Settings(BaseSettings):
    """Configuration settings for the application."""
    
    model_config = SettingsConfigDict(
        env_prefix="{{ cookiecutter.project_slug }}_",
        case_sensitive=False,
        extra="ignore",
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = False
    include_timestamp: bool = True

# A single, global instance of the settings
settings = Settings()
```

**Why Centralize:**
- Settings are duplicated across projects
- Dynamic config changes require redeployment
- Feature flags buried in environment variables
- No audit trail for config changes
- Hard to manage secrets

**Current Limitations:**
1. No secrets management (uses .env files)
2. No dynamic configuration
3. No feature flags
4. No audit trail
5. No multi-tenant support

**Centralized Service Needed:** `config-service/` and `secrets-manager/`

---

#### Pattern 6.2: Dynamic Configuration Service
**File:** `/central-services/config-service/` (Directory exists, needs implementation)

**Should Implement:**
```python
class DynamicConfig:
    """Fetch config from central service"""
    
    async def get_config(self, key: str) -> Any:
        """Get configuration value from central service"""
        # Implement caching with TTL
        # Watch for changes
        # Support namespacing
        
    async def get_feature_flag(self, flag_name: str) -> bool:
        """Get feature flag status"""
        # Support gradual rollout
        # Support per-user targeting
        # Support A/B testing
```

---

#### Pattern 6.3: Secrets Management
**File:** `/central-services/secrets-manager/` (Directory exists, needs implementation)

**Needed Pattern:**
```python
class SecretsManager:
    """Fetch secrets from Vault or cloud provider"""
    
    async def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from vault"""
        # Implement rotation
        # Implement audit logging
        # Implement TTL
        # Support version tracking
```

---

## 7. AUDIT LOGGING

### Current State: Partial Implementation

**Files Analyzed:**
- `/utils/logging.py` - Structured logging with structlog
- `/utils/financial.py` - Financial utilities with audit hooks
- `/core/sentry.py` - Breadcrumb and event tracking

### Identified Patterns:

#### Pattern 7.1: Structured Logging Foundation
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/utils/logging.py`

```python
def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure structured logging for the application."""
    
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        # ... more processors
    ]
    
    if json_logs:
        # Production: JSON logs for easy parsing and aggregation
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: Rich console output with colors
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
```

**Good Patterns:**
- ✅ JSON output for production
- ✅ Structured logging (key-value pairs)
- ✅ Environment-aware configuration
- ✅ Timestamp inclusion

**Missing Patterns:**
1. No audit logging specific module
2. No event classification
3. No compliance logging
4. No data sensitivity marking
5. No retention policy
6. No log aggregation client

---

#### Pattern 7.2: Performance Logging
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/utils/logging.py` (Lines 126-162)

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

**Why Centralize:**
- Performance metrics should be centralized
- Aggregated across all services
- Trend analysis and alerting
- SLA tracking

---

#### Pattern 7.3: Financial Audit Logging (if enabled)
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/utils/financial.py` (Lines 130-146)

```python
{% if cookiecutter.include_audit_logging == "yes" -%}
def format_audit_amount(amount: Decimal) -> str:
    """Format amount for audit logging."""
    return f"${amount:,.2f}"
{% endif -%}
```

**Why Centralize:**
- Financial transactions require audit trail
- Compliance requirement (PCI-DSS, etc.)
- Should be immutable log
- Need tamper-proof storage

**Missing Implementation:**
```python
# What should exist but doesn't:

async def audit_log_financial_transaction(
    transaction_id: str,
    amount: Decimal,
    currency: str,
    from_account: str,
    to_account: str,
    reason: str,
    user_id: str,
    timestamp: datetime,
) -> None:
    """Log financial transaction for audit trail"""
    # Must be immutable
    # Must include all parties
    # Must be tamper-proof
    # Must include authorization
```

---

#### Pattern 7.4: Sentry-Based Event Tracking
**File:** `{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/core/sentry.py` (Lines 285-327)

```python
def capture_message(
    message: str,
    *,
    level: str = "info",
    tags: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Capture a message (not an exception) to Sentry."""
    # Use for non-error events that you want to track
    # Allows custom event tracking

def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: dict[str, Any] | None = None,
) -> None:
    """Add a breadcrumb (event leading up to an error)."""
    # Track sequence of events
```

**Why Centralize:**
- Event tracking should have consistent categories
- Org-wide event taxonomy
- Cross-service event correlation
- Event replay for debugging

---

## SUMMARY TABLE

| Category | Pattern | Current Location | Centralization Target | Priority |
|----------|---------|------------------|----------------------|----------|
| **Auth** | Security Headers | `middleware/security.py` | `api-gateway/` | High |
| **Auth** | Rate Limiting | `middleware/security.py` | `api-gateway/` | High |
| **Auth** | User Context Tracking | `core/sentry.py` | `audit-logging-service/` | Medium |
| **Notifications** | Email Sending | `jobs/worker.py` | `notification-service/` | High |
| **Notifications** | Task Queueing | `jobs/worker.py` | `notification-service/` or `job-queue-service/` | High |
| **Storage** | File Upload Processing | `jobs/worker.py` | `storage-service/` | Medium |
| **API Clients** | Health Check Pattern | `api/health.py` | `api-clients/` library | Medium |
| **API Clients** | External Service Integration | `core/sentry.py` | Centralized SDK management | Medium |
| **Middleware** | CORS | `middleware/security.py` | `api-gateway/` | High |
| **Middleware** | SSRF Prevention | `middleware/security.py` | `api-gateway/` | High |
| **Config** | Settings from Env | `core/config.py` | `config-service/` | High |
| **Config** | Dynamic Configuration | Not implemented | `config-service/` | High |
| **Config** | Secrets Management | Uses .env files | `secrets-manager/` | High |
| **Logging** | Structured Logging | `utils/logging.py` | Centralized logging service | Medium |
| **Logging** | Performance Metrics | `utils/logging.py` | Central metrics service | Medium |
| **Logging** | Financial Audit | `utils/financial.py` | `audit-logging-service/` | High |
| **Logging** | Event Tracking | `core/sentry.py` | Event service/Kafka | Medium |

---

## RECOMMENDATIONS

### Phase 1: Critical Infrastructure (Week 1-2)
1. **Auth Service** - Already well-documented in `central-services/`
   - Implement JWT token management
   - Role-based access control
   - Move all security headers & rate limiting here

2. **Secrets Manager** - Deploy Vault or cloud secrets
   - Migrate from .env files
   - Implement secret rotation

### Phase 2: Core Services (Week 3-4)
1. **Notification Service** - Already documented
   - Email/SMS/Push notifications
   - Template management
   - Delivery tracking

2. **Configuration Service**
   - Dynamic config without redeployment
   - Feature flags
   - Multi-tenant support

### Phase 3: Support Services (Week 5-6)
1. **Audit Logging Service**
   - Centralized event logging
   - Compliance logging
   - Immutable transaction logs

2. **API Gateway** - Consolidate middleware
   - Rate limiting
   - CORS management
   - SSRF prevention
   - Request/response validation

### Phase 4: Optional/Specialized (Week 7+)
1. **Storage Service** - S3 integration
2. **API Clients Library** - Common external service patterns
3. **Feature Flags Service** - Already outlined
4. **Job Queue Service** - Task distribution

---

## Implementation Notes

### Why This Matters
- **Consistency**: Same patterns across all 50+ projects
- **Security**: Single point for security updates
- **Compliance**: Easier to audit and maintain compliance
- **Cost**: Centralized resources = economies of scale
- **Maintenance**: Update once, benefits all projects

### Next Steps
1. Review this analysis with platform team
2. Prioritize service implementations
3. Create client SDKs for each service
4. Update cookiecutter template to integrate with central services
5. Create migration guides for existing projects

