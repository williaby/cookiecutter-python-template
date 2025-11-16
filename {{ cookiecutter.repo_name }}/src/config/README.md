# Configuration System - Phase 2: Environment-Specific Loading

This module implements the Phase 2 Core Configuration System for PromptCraft-Hybrid with environment-specific loading support.

## Features

- **Environment Detection**: Automatic detection of dev/staging/prod environments
- **File Loading Hierarchy**: Proper precedence for configuration sources
- **Graceful Degradation**: Works without any .env files using sensible defaults
- **Singleton Pattern**: Efficient settings management across application lifecycle
- **Type Safety**: Full Pydantic validation with clear error messages

## Configuration Precedence

The system loads configuration with the following precedence (highest to lowest):

1. **Environment Variables** (highest priority)
2. **`.env.{environment}` file** (e.g., `.env.dev`, `.env.staging`, `.env.prod`)
3. **`.env` file** (base configuration)
4. **Pydantic Field Defaults** (lowest priority)

## Usage

### Basic Usage

```python
from config import get_settings

# Get settings instance (singleton pattern)
settings = get_settings()

print(f"Running {settings.app_name} v{settings.version}")
print(f"Environment: {settings.environment}")
print(f"API: http://{settings.api_host}:{settings.api_port}")
```

### Environment-Specific Configuration

Create environment-specific .env files in the project root:

#### `.env.dev` (Development)
```bash
PROMPTCRAFT_APP_NAME=PromptCraft-Hybrid-Dev
PROMPTCRAFT_ENVIRONMENT=dev
PROMPTCRAFT_DEBUG=true
PROMPTCRAFT_API_PORT=7860
```

#### `.env.staging` (Staging)
```bash
PROMPTCRAFT_APP_NAME=PromptCraft-Hybrid-Staging
PROMPTCRAFT_ENVIRONMENT=staging
PROMPTCRAFT_DEBUG=false
PROMPTCRAFT_API_PORT=8000
```

#### `.env.prod` (Production)
```bash
PROMPTCRAFT_APP_NAME=PromptCraft-Hybrid
PROMPTCRAFT_ENVIRONMENT=prod
PROMPTCRAFT_DEBUG=false
PROMPTCRAFT_API_PORT=8000
```

### Environment Variable Overrides

You can override any setting using environment variables with the `PROMPTCRAFT_` prefix:

```bash
# Override environment
export PROMPTCRAFT_ENVIRONMENT=staging

# Override API port
export PROMPTCRAFT_API_PORT=9000

# Run your application
python your_app.py
```

### Runtime Settings Reload

For testing or when configuration changes during runtime:

```python
from config import reload_settings

# Force reload settings from environment and files
settings = reload_settings()
```

## Available Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `app_name` | str | "PromptCraft-Hybrid" | Application name for logging and identification |
| `version` | str | "0.1.0" | Application version string |
| `environment` | str | "dev" | Deployment environment (dev/staging/prod) |
| `debug` | bool | `True` | Whether debug mode is enabled |
| `api_host` | str | "0.0.0.0" | Host address for the API server |
| `api_port` | int | 8000 | Port number for the API server |

## Environment Detection

The system automatically detects the current environment using this priority:

1. `PROMPTCRAFT_ENVIRONMENT` environment variable
2. `PROMPTCRAFT_ENVIRONMENT` setting in `.env` file
3. Default to `"dev"`

## Integration with Phase 3 (Encryption)

This configuration system is designed to integrate seamlessly with Phase 3 encryption:

- Settings factory pattern supports encrypted .env file loading
- Environment detection enables encryption key selection
- Graceful degradation ensures development works without encryption setup

## Files Created

- `/src/config/settings.py` - Core configuration schema and loading logic
- `/src/config/__init__.py` - Module exports
- `/.env.dev` - Development environment configuration
- `/.env.staging` - Staging environment configuration
- `/.env.prod` - Production environment configuration
- `/examples/config_demo.py` - Usage demonstration

## Next Steps

Phase 3 will add:
- GPG encryption for sensitive .env files
- Key management integration
- Encrypted settings validation
- Environment-specific encryption keys
