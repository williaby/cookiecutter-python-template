# Authentication Service

Central JWT authentication and authorization service for all projects.

## 🎯 Purpose

Provides centralized user authentication, token management, and authorization for all projects in the organization.

## ✨ Features

- **JWT Token Management**: Issue, verify, refresh, and revoke tokens
- **Multiple Auth Methods**: OAuth2, API keys, username/password
- **Role-Based Access Control (RBAC)**: Flexible permission system
- **Multi-Tenancy**: Support for multiple organizations/tenants
- **Token Rotation**: Automatic key rotation for security
- **Audit Logging**: Track all authentication events
- **Rate Limiting**: Prevent brute force attacks
- **Session Management**: Track active sessions per user

## 📦 Components

This directory should contain:

```
auth-service/
├── README.md (this file)
├── src/
│   ├── auth_service/
│   │   ├── api/          # FastAPI endpoints
│   │   ├── models/       # User, Token, Permission models
│   │   ├── services/     # Business logic
│   │   ├── middleware/   # Auth middleware
│   │   └── utils/        # JWT utilities, crypto
│   └── tests/
├── client/               # Client SDK (separate package)
│   ├── auth_client/
│   │   ├── __init__.py
│   │   ├── client.py     # Main client class
│   │   ├── middleware.py # FastAPI/Flask middleware
│   │   └── decorators.py # @require_auth, @require_role
│   └── pyproject.toml
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── k8s/              # Kubernetes manifests
│   └── terraform/        # Infrastructure as code
├── examples/
│   ├── fastapi-integration.py
│   ├── flask-integration.py
│   └── cli-client.py
└── docs/
    ├── API.md            # API documentation
    ├── DEPLOYMENT.md     # Deployment guide
    └── SECURITY.md       # Security considerations
```

## 🔌 Client SDK Usage

### Installation

```bash
# Add to your project
uv add auth-service-client

# Or from private registry
uv add auth-service-client --index-url https://pypi.your-domain.com/simple
```

### FastAPI Integration

```python
from fastapi import FastAPI, Depends
from auth_client import AuthClient, require_auth, require_role

app = FastAPI()
auth = AuthClient(url="https://auth.your-domain.com")

@app.get("/public")
async def public_endpoint():
    """No authentication required"""
    return {"message": "Hello, world!"}

@app.get("/protected")
async def protected_endpoint(user = Depends(require_auth)):
    """Requires valid JWT token"""
    return {"message": f"Hello, {user.username}!"}

@app.get("/admin")
async def admin_endpoint(user = Depends(require_role("admin"))):
    """Requires admin role"""
    return {"message": "Admin dashboard"}

@app.post("/login")
async def login(username: str, password: str):
    """Login and get JWT token"""
    token = await auth.login(username, password)
    return {"access_token": token.access_token, "refresh_token": token.refresh_token}
```

### Environment Variables

```bash
# .env
AUTH_SERVICE_URL=https://auth.your-domain.com
AUTH_SERVICE_API_KEY=your-service-api-key  # For service-to-service auth
AUTH_JWT_PUBLIC_KEY_URL=https://auth.your-domain.com/.well-known/jwks.json
```

## 🔐 API Endpoints

### Authentication

```
POST   /auth/login                 # Username/password login
POST   /auth/logout                # Logout (invalidate token)
POST   /auth/refresh               # Refresh access token
POST   /auth/register              # User registration (if enabled)
POST   /auth/verify                # Verify token validity
```

### OAuth2

```
GET    /oauth2/authorize           # OAuth2 authorization
POST   /oauth2/token               # Token exchange
GET    /oauth2/userinfo            # Get user info from token
```

### User Management

```
GET    /users/{user_id}            # Get user details
PUT    /users/{user_id}            # Update user
DELETE /users/{user_id}            # Delete user
GET    /users/{user_id}/sessions   # List active sessions
DELETE /users/{user_id}/sessions   # Revoke all sessions
```

### Roles & Permissions

```
GET    /roles                      # List all roles
POST   /roles                      # Create role
GET    /roles/{role_id}            # Get role details
PUT    /roles/{role_id}            # Update role
DELETE /roles/{role_id}            # Delete role

POST   /users/{user_id}/roles      # Assign role to user
DELETE /users/{user_id}/roles/{role_id}  # Remove role
```

### API Keys

```
POST   /api-keys                   # Create API key (for services)
GET    /api-keys                   # List API keys
DELETE /api-keys/{key_id}          # Revoke API key
```

## 🏗️ Architecture

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Roles table
CREATE TABLE roles (
    id UUID PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB,  -- {"resource": ["read", "write", "delete"]}
    created_at TIMESTAMP DEFAULT NOW()
);

-- User roles (many-to-many)
CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id),
    role_id UUID REFERENCES roles(id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);

-- Sessions/Tokens
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    token_hash VARCHAR(255) NOT NULL,  -- Hashed JTI
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- API Keys (for service-to-service)
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions JSONB,
    expires_at TIMESTAMP,
    revoked_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit log
CREATE TABLE auth_events (
    id UUID PRIMARY KEY,
    user_id UUID,
    event_type VARCHAR(50),  -- login, logout, token_refresh, etc.
    ip_address VARCHAR(45),
    user_agent TEXT,
    success BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### JWT Token Structure

```json
{
  "jti": "unique-token-id",
  "sub": "user-id",
  "username": "john.doe",
  "email": "john@example.com",
  "roles": ["user", "developer"],
  "permissions": {
    "projects": ["read", "write"],
    "users": ["read"]
  },
  "iss": "https://auth.your-domain.com",
  "aud": ["api.your-domain.com"],
  "iat": 1700000000,
  "exp": 1700003600,
  "type": "access"
}
```

## 🚀 Deployment

### Docker Compose (Development)

```bash
cd deploy/
docker-compose up -d
```

### Kubernetes (Production)

```bash
cd deploy/k8s/
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
```

### Environment Configuration

```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/authdb
JWT_SECRET_KEY=your-secret-key-min-32-chars
JWT_ALGORITHM=RS256  # Recommended: RS256 for asymmetric
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# Optional
ENABLE_REGISTRATION=false
ENABLE_OAUTH2=true
OAUTH2_PROVIDERS=google,github
PASSWORD_MIN_LENGTH=12
REQUIRE_EMAIL_VERIFICATION=true
MAX_LOGIN_ATTEMPTS=5
LOGIN_RATE_LIMIT=10/minute

# For asymmetric JWT (recommended)
JWT_PRIVATE_KEY_PATH=/secrets/jwt-private.pem
JWT_PUBLIC_KEY_PATH=/secrets/jwt-public.pem
```

## 🔒 Security Considerations

### Token Security
- Use **RS256** (asymmetric) for production (not HS256)
- Rotate signing keys regularly (every 90 days)
- Short-lived access tokens (15 minutes)
- Longer refresh tokens (30 days) with rotation
- Store token JTI in database for revocation

### Password Security
- bcrypt with cost factor 12+
- Password complexity requirements
- Rate limiting on login attempts
- Account lockout after 5 failed attempts
- Password history (prevent reuse)

### Network Security
- TLS/HTTPS only (no HTTP)
- Certificate pinning for client SDKs
- IP allowlisting for admin endpoints

### Audit & Monitoring
- Log all authentication events
- Alert on suspicious patterns (many failed logins)
- Track session activity
- Regular security audits

## 📊 Monitoring

### Metrics to Track
- Login success/failure rate
- Token issuance rate
- Active sessions count
- Average token lifetime
- Failed login attempts by IP
- API key usage

### Health Checks
```
GET /health/live      # Service is running
GET /health/ready     # Database connected, ready to serve
GET /health/startup   # Initialization complete
```

### Logging
- Structured JSON logs
- PII filtering (don't log passwords!)
- Correlation IDs for tracing
- Ship to central logging (Loki/ELK)

## 🧪 Testing

```bash
# Run tests
uv run pytest

# Load testing
locust -f tests/load/auth_load_test.py

# Security testing
uv run bandit -r src/
uv run safety check
```

## 📈 Performance

### Caching
- Cache public keys (JWKS) for 1 hour
- Cache user roles/permissions for 5 minutes
- Use Redis for session storage

### Database Optimization
- Index on username, email, token_hash
- Partition auth_events by date
- Archive old sessions monthly

### Scalability
- Stateless design (scales horizontally)
- Read replicas for token verification
- CDN for public keys (.well-known/jwks.json)

## 🔄 Migration from Per-Project Auth

### Step 1: Deploy Auth Service
```bash
# Set up central auth service
docker-compose up -d
# Or K8s deployment
```

### Step 2: Migrate Users
```python
# Export users from old system
from old_auth import export_users
users = export_users()

# Import to central auth
from auth_client import AuthClient
auth = AuthClient()
for user in users:
    await auth.import_user(
        username=user.username,
        email=user.email,
        password_hash=user.password_hash,  # Keep existing hashes
        roles=user.roles
    )
```

### Step 3: Update Projects
```python
# Before (per-project JWT)
from jose import jwt
from app.auth import verify_token

@app.get("/protected")
def protected(token: str):
    user = verify_token(token)  # Local verification
    return {"user": user}

# After (central auth)
from auth_client import require_auth

@app.get("/protected")
def protected(user = Depends(require_auth)):  # Central verification
    return {"user": user}
```

### Step 4: Deprecate Old Auth
- Set up parallel running period (both systems active)
- Monitor auth service metrics
- Gradually migrate projects
- Deprecate old auth code after all projects migrated

## 🤝 Contributing

See main template CONTRIBUTING.md for guidelines.

## 📞 Support

- **Issues**: https://github.com/YOUR_ORG/auth-service/issues
- **Slack**: #auth-service
- **Email**: platform-team@your-domain.com

---

**Status**: ✅ Production
**Version**: 1.0.0
**Last Updated**: 2025-11-18
