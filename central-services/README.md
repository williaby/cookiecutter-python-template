# Centralized Services

This folder contains reference implementations and starter code for centralized infrastructure services that support multiple projects.

## 🎯 Purpose

Rather than implementing common infrastructure in every project, these centralized services provide:
- **Single source of truth** for shared functionality
- **Consistency** across all projects
- **Reduced maintenance burden** (update once, benefits all projects)
- **Better security** (centralized access control, audit trails)
- **Cost efficiency** (shared resources, economies of scale)

## 📁 Service Catalog

### 1. **Auth Service** (`auth-service/`)
Central authentication and authorization service supporting JWT, OAuth2, and API keys.

**Status**: ✅ Production
**Use When**: All projects that need user authentication
**Replaces**: Per-project JWT middleware, OAuth implementations

### 2. **Secrets Manager** (`secrets-manager/`)
Centralized secrets storage and rotation service (Vault-compatible).

**Status**: 🟡 Recommended
**Use When**: Production deployments
**Replaces**: `.env` files, hardcoded credentials

### 3. **API Gateway** (`api-gateway/`)
Central API gateway with routing, rate limiting, and security policies.

**Status**: 🟡 Recommended for microservices
**Use When**: Multiple backend services need unified entry point
**Replaces**: Per-project rate limiting, CORS handling

### 4. **Notification Service** (`notification-service/`)
Unified email, SMS, and push notification service.

**Status**: 🟡 Recommended
**Use When**: Projects need to send communications
**Replaces**: Per-project email/SMS integrations

### 5. **Feature Flags** (`feature-flags/`)
Central feature flag management for gradual rollouts and A/B testing.

**Status**: 🔵 Optional
**Use When**: Need dynamic feature toggling across services
**Replaces**: Hardcoded feature toggles, environment variables

### 6. **Storage Service** (`storage-service/`)
S3-compatible object storage service with CDN integration.

**Status**: 🔵 Optional
**Use When**: Projects need file storage/uploads
**Replaces**: Per-project S3/blob storage integration

### 7. **Config Service** (`config-service/`)
Dynamic configuration management without redeployment.

**Status**: 🔵 Optional
**Use When**: Need runtime config changes
**Replaces**: Static `.env` files

## 🚀 Getting Started

### For Service Maintainers

Each service directory contains:
- `README.md` - Service overview and API documentation
- `src/` - Service implementation
- `deploy/` - Deployment configurations (Docker, K8s)
- `client/` - Client SDK for consuming projects
- `examples/` - Integration examples

### For Project Developers

To integrate a centralized service into your project:

1. **Add the client SDK** to your project:
   ```bash
   uv add git+https://github.com/YOUR_ORG/auth-service.git#subdirectory=client
   ```

2. **Configure environment variables**:
   ```bash
   # .env
   AUTH_SERVICE_URL=https://auth.your-domain.com
   AUTH_SERVICE_API_KEY=your-api-key
   ```

3. **Use in your code**:
   ```python
   from auth_service_client import AuthClient

   auth = AuthClient()
   user = await auth.verify_token(token)
   ```

## 🏗️ Architecture Patterns

### Monolith Projects
For monolithic applications, centralize:
- ✅ Authentication (JWT service)
- ✅ Secrets Management
- ✅ Observability (Sentry org)
- 🟡 Notifications (if sending emails/SMS)

Keep per-project:
- Application logic
- Database
- Caching (if app-specific)

### Microservices Projects
For microservice architectures, centralize:
- ✅ All services listed above
- ✅ API Gateway (required)
- ✅ Service mesh/registry
- ✅ Message broker

Keep per-project:
- Service-specific logic
- Service-specific databases (often)
- Health checks

## 📊 Migration Strategy

### Phase 1: Authentication (Week 1-2)
1. Deploy central auth service
2. Migrate existing projects to use auth client SDK
3. Deprecate per-project JWT implementations

### Phase 2: Secrets (Week 3-4)
1. Deploy Vault or cloud secrets manager
2. Migrate secrets from `.env` files
3. Update CI/CD to pull from secrets manager

### Phase 3: Observability (Week 5-6)
1. Set up central Sentry organization
2. Configure central logging (Loki/ELK)
3. Deploy Prometheus + Grafana

### Phase 4: API Gateway (Week 7-8)
1. Deploy Kong/Traefik
2. Move rate limiting to gateway
3. Consolidate CORS policies

### Phase 5+: Optional Services
Deploy as needed based on project requirements.

## 🔗 Integration with Cookiecutter Template

The main cookiecutter template can be configured to integrate with these services:

```json
// cookiecutter.json
{
  "use_central_auth": ["no", "yes"],
  "central_auth_url": "https://auth.your-domain.com",

  "use_secrets_manager": ["none", "vault", "aws", "azure"],
  "vault_url": "https://vault.your-domain.com",

  // ... other options
}
```

When enabled, the generated project will include:
- Client SDK dependencies
- Configuration templates
- Integration examples
- Health check endpoints

## 📖 Documentation

Each service has detailed documentation in its subdirectory:
- API reference
- Deployment guides
- Client SDK documentation
- Security considerations
- Performance tuning

## 🤝 Contributing

When adding a new centralized service:

1. Create service directory: `central-services/my-service/`
2. Follow standard structure (src, deploy, client, examples)
3. Add entry to this README
4. Create client SDK for easy integration
5. Update cookiecutter template if appropriate

## 📞 Support

For issues with centralized services:
- **Auth Service**: Create issue in auth-service repo
- **General Questions**: Discussion in main template repo
- **Integration Help**: See `examples/` in each service directory

---

**Last Updated**: 2025-11-18
**Maintained By**: Platform Team
