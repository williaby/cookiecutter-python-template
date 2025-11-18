# Secrets Manager Service

Centralized secrets storage, rotation, and access control.

## 🎯 Purpose

Replace `.env` files and hardcoded secrets with secure, audited secret management.

## ✨ Features

- **Encrypted Storage**: AES-256 encryption at rest
- **Access Control**: Fine-grained permissions per secret
- **Automatic Rotation**: Scheduled key rotation
- **Audit Trail**: Track all secret access
- **Versioning**: Keep secret history
- **Dynamic Secrets**: Generate temporary credentials
- **Integration**: Vault-compatible API

## 📦 Code to Migrate from Template

### From `.env` files

**MIGRATE ALL SENSITIVE VALUES**:
```bash
# DON'T KEEP IN .env (migrate to Vault)
DATABASE_URL=postgresql://user:password@host/db  # ← MOVE TO VAULT
API_KEY=sk-1234567890  # ← MOVE TO VAULT
JWT_SECRET_KEY=super-secret  # ← MOVE TO VAULT
SENTRY_DSN=https://key@sentry.io/project  # ← MOVE TO VAULT

# OK TO KEEP IN .env (non-sensitive)
LOG_LEVEL=INFO
ENVIRONMENT=production
WORKERS=4
```

**Replace with**:
```python
from secrets_client import SecretsClient

secrets = SecretsClient()
db_url = await secrets.get("database/url")
api_key = await secrets.get("external/api_key")
```

## 🔌 Client SDK

```python
from secrets_client import SecretsClient

client = SecretsClient(url="https://vault.your-domain.com")

# Get secret
db_password = await client.get("database/postgres/password")

# Get multiple secrets
secrets = await client.get_many([
    "database/postgres/password",
    "external/stripe/api_key",
    "jwt/private_key"
])

# Set secret (admin only)
await client.set(
    "database/postgres/password",
    value="new-password",
    metadata={"rotated_at": "2025-11-18"}
)

# Dynamic secrets (temporary credentials)
aws_creds = await client.get_dynamic("aws/sts/developer")
# Returns: {"access_key": "...", "secret_key": "...", "expires_in": 3600}
```

## 🏗️ Architecture

Options:
1. **HashiCorp Vault** (recommended, open-source)
2. **AWS Secrets Manager**
3. **Azure Key Vault**
4. **Google Secret Manager**
5. **Custom implementation** (use template if needed)

## 🚀 Migration Strategy

### Phase 1: Deploy Vault
```bash
docker run -d -p 8200:8200 vault:latest
vault operator init
vault operator unseal
```

### Phase 2: Migrate Secrets
```bash
# Export from .env
cat .env | while read line; do
    key=$(echo $line | cut -d= -f1)
    value=$(echo $line | cut -d= -f2-)
    vault kv put secret/myapp/$key value="$value"
done
```

### Phase 3: Update Projects
```python
# Before
import os
DATABASE_URL = os.getenv("DATABASE_URL")

# After
from secrets_client import get_secret
DATABASE_URL = await get_secret("database/url")
```

## 🔒 Security Benefits

- **No secrets in git** (no accidental commits)
- **No secrets in CI logs** (fetch at runtime)
- **Access auditing** (who accessed what, when)
- **Automatic rotation** (reduce breach window)
- **Encrypted at rest** (protection if storage compromised)

## 📋 Migration Checklist

- [ ] Deploy Vault/Secrets Manager
- [ ] Configure access policies
- [ ] Migrate database credentials
- [ ] Migrate API keys (Stripe, AWS, etc.)
- [ ] Migrate JWT signing keys
- [ ] Update CI/CD to fetch secrets
- [ ] Remove secrets from `.env` files
- [ ] Update projects to use secrets client
- [ ] Enable audit logging
- [ ] Set up rotation schedule

---

**Status**: 🟡 Recommended
**Priority**: High (security critical)
