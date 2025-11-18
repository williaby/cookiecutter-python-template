# API Gateway Service

Central entry point for all backend services with routing, security, and observability.

## 🎯 Purpose

Unified API gateway for microservices architecture providing routing, authentication, rate limiting, and monitoring.

## ✨ Features

- **Request Routing**: Route to appropriate backend service
- **Load Balancing**: Distribute traffic across service instances
- **Authentication**: Integrate with central auth service
- **Rate Limiting**: Distributed rate limiting (Redis-backed)
- **CORS Handling**: Centralized CORS policies
- **Request/Response Transformation**: Modify headers, body
- **Circuit Breaking**: Prevent cascade failures
- **API Versioning**: Support multiple API versions
- **Observability**: Metrics, logging, tracing
- **Caching**: Cache responses at edge

## 📦 Code to Migrate from Template

### From `{{cookiecutter.project_slug}}/middleware/security.py`

**Rate Limiting** (Lines 102-173):
```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    # Current: Per-project, in-memory (not distributed)
    # Move to: API Gateway with Redis backend
```

**CORS Configuration** (Lines 273-282):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be centralized policy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Security Headers** (Lines 39-100):
```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    # OWASP security headers
    # Better at API Gateway (single point of enforcement)
```

**SSRF Prevention** (Lines 176-225):
```python
class SSRFPreventionMiddleware(BaseHTTPMiddleware):
    """Prevent Server-Side Request Forgery attacks."""
    # Block requests to internal IPs
    # More effective at gateway layer
```

### Why Move to Gateway?

**Current State** (per-project):
```
Client → [Rate Limit] → [CORS] → [Auth] → [Security Headers] → Backend
         ↑ Per-project middleware (duplicated in every service)
```

**With API Gateway**:
```
Client → API Gateway [All middleware once] → Backend Services
         ↑ Single enforcement point, no duplication
```

## 🏗️ Gateway Options

### 1. Kong (Recommended)
- Open-source, plugin-based
- Great for microservices
- Extensive plugin ecosystem

### 2. Traefik
- Cloud-native, dynamic configuration
- Excellent for Kubernetes
- Automatic service discovery

### 3. AWS API Gateway
- Managed service (no ops)
- Integrated with AWS ecosystem
- Pay-per-use pricing

### 4. Envoy
- High-performance proxy
- Service mesh ready
- CNCF graduated project

## 🔌 Example Configuration

### Kong Configuration

```yaml
# services/backend-service.yaml
_format_version: "3.0"

services:
  - name: user-service
    url: http://user-service:8000
    routes:
      - name: user-routes
        paths:
          - /api/v1/users
    plugins:
      - name: rate-limiting
        config:
          minute: 100
          hour: 10000
      - name: jwt
        config:
          uri_param_names:
            - jwt
      - name: cors
        config:
          origins:
            - https://app.your-domain.com
```

### Traefik Configuration

```yaml
# traefik.yml
http:
  routers:
    user-service:
      rule: "PathPrefix(`/api/v1/users`)"
      service: user-service
      middlewares:
        - auth
        - ratelimit
        - cors

  middlewares:
    auth:
      forwardAuth:
        address: "https://auth.your-domain.com/verify"
    ratelimit:
      rateLimit:
        average: 100
        burst: 200
    cors:
      headers:
        accessControlAllowOrigins:
          - "https://app.your-domain.com"
```

## 🚀 Deployment Pattern

### With API Gateway (Recommended)

```
┌──────────────┐
│   Clients    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│       API Gateway (Kong)         │
│  • Authentication                │
│  • Rate Limiting                 │
│  • CORS                          │
│  • Load Balancing                │
│  • Circuit Breaking              │
└──────┬───────────────────────────┘
       │
       ├─────► User Service (8001)
       ├─────► Order Service (8002)
       ├─────► Payment Service (8003)
       └─────► Notification Service (8004)
```

### Without Gateway (Current - Not Recommended)

```
┌──────────────┐
│   Clients    │
└──────┬───────┘
       │
       ├─────► User Service → [Middleware] → Logic
       ├─────► Order Service → [Middleware] → Logic
       ├─────► Payment Service → [Middleware] → Logic
       └─────► Notification Service → [Middleware] → Logic
                ↑ Middleware duplicated in every service
```

## 📋 Migration from Per-Service Middleware

### Step 1: Deploy API Gateway
```bash
# Using Kong
docker run -d --name kong-gateway \
  -p 8000:8000 -p 8001:8001 \
  kong:latest
```

### Step 2: Configure Routes
```bash
# Add service
curl -i -X POST http://localhost:8001/services/ \
  --data name=user-service \
  --data url=http://user-service:8000

# Add route
curl -i -X POST http://localhost:8001/services/user-service/routes \
  --data paths[]=/api/v1/users
```

### Step 3: Add Plugins (Middleware)
```bash
# Rate limiting
curl -X POST http://localhost:8001/services/user-service/plugins \
  --data name=rate-limiting \
  --data config.minute=100

# JWT auth
curl -X POST http://localhost:8001/services/user-service/plugins \
  --data name=jwt
```

### Step 4: Remove Per-Service Middleware
```python
# Before (each service has middleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CORSMiddleware)
# ... etc

# After (clean backend service)
app = FastAPI()
# No middleware - handled by gateway
```

### Step 5: Update Client Routing
```python
# Before
API_BASE = "https://user-service.your-domain.com"

# After
API_BASE = "https://api.your-domain.com"  # Gateway endpoint
```

## 🔒 Security Benefits

- **Single Enforcement Point**: One place to apply security policies
- **Consistent Policies**: Same rules across all services
- **DDoS Protection**: Rate limiting at edge
- **Hide Backend**: Clients don't know backend topology
- **WAF Integration**: Web Application Firewall at gateway

## 📊 Observability

### Metrics
- Request rate (per service, per endpoint)
- Response times (p50, p95, p99)
- Error rates (4xx, 5xx)
- Rate limit hits
- Active connections

### Logging
```json
{
  "timestamp": "2025-11-18T12:00:00Z",
  "client_ip": "1.2.3.4",
  "method": "GET",
  "path": "/api/v1/users/123",
  "service": "user-service",
  "status": 200,
  "duration_ms": 45,
  "user_id": "user-123"
}
```

### Tracing
- Distributed tracing with OpenTelemetry
- See full request path through services
- Identify bottlenecks

## 🎯 When to Use API Gateway

**Use Gateway When:**
- ✅ Microservices architecture (2+ services)
- ✅ Need consistent security across services
- ✅ Want centralized rate limiting
- ✅ Need load balancing
- ✅ Service-to-service auth

**Skip Gateway When:**
- ❌ Single monolithic application
- ❌ Internal-only APIs (no public exposure)
- ❌ Very simple architecture
- ❌ Extreme performance requirements (extra hop)

## 📦 Template Integration

After deploying gateway, update cookiecutter template:

```json
// cookiecutter.json
{
  "use_api_gateway": ["no", "yes"],
  "api_gateway_url": "https://api.your-domain.com",
  "skip_middleware": ["no", "yes"]  // Skip if gateway handles it
}
```

Generated projects would:
- Skip rate limiting middleware (gateway handles it)
- Skip CORS middleware (gateway handles it)
- Skip some security headers (gateway adds them)
- Focus on business logic only

---

**Status**: 🟡 Recommended for microservices
**Priority**: High (if 2+ services)
**Best Practice**: Deploy early in microservices journey
