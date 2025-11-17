# Xero API Integration: Code Examples for Cookiecutter Template

This document provides concrete code examples extracted from Xero Python ecosystem analysis. These can be used as templates for the cookiecutter-python-template.

## Table of Contents

1. [API Client Initialization](#1-api-client-initialization)
2. [Configuration Management](#2-configuration-management)
3. [Authentication Patterns](#3-authentication-patterns)
4. [Data Models](#4-data-models)
5. [Error Handling](#5-error-handling)
6. [API Operations](#6-api-operations)
7. [Testing Fixtures](#7-testing-fixtures)
8. [Middleware & Utilities](#8-middleware--utilities)

---

## 1. API Client Initialization

### 1.1 Standard API Client Factory

```python
# src/myproject/api/client.py
"""API client factory and initialization."""

from typing import Optional, Callable
from xero_python.api_client import ApiClient, Configuration, OAuth2Token
import logging

logger = logging.getLogger(__name__)


class APIClientFactory:
    """Factory for creating and configuring API clients."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_getter: Optional[Callable] = None,
        token_saver: Optional[Callable] = None,
        debug: bool = False
    ):
        """
        Initialize API client factory.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            token_getter: Callable to retrieve persisted token
            token_saver: Callable to save token
            debug: Enable debug logging
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_getter = token_getter
        self.token_saver = token_saver
        self.debug = debug
        
        # Validate required credentials
        if not self.client_id or not self.client_secret:
            raise ValueError("client_id and client_secret are required")
    
    def create(self) -> ApiClient:
        """
        Create configured API client.
        
        Returns:
            Configured ApiClient instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Create configuration
        config = Configuration(
            debug=self.debug,
            oauth2_token=OAuth2Token(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        )
        
        # Create API client
        api_client = ApiClient(
            configuration=config,
            pool_threads=1
        )
        
        # Attach token getter/saver decorators if provided
        if self.token_getter:
            api_client.oauth2_token_getter(self.token_getter)
            
        if self.token_saver:
            api_client.oauth2_token_saver(self.token_saver)
        
        logger.info("API client initialized successfully")
        return api_client


def create_api_client_from_config(config) -> ApiClient:
    """
    Factory function to create API client from config object.
    
    Args:
        config: Configuration object with API credentials
        
    Returns:
        Configured ApiClient
    """
    # Example with token storage
    def get_token():
        """Retrieve token from config/storage."""
        return getattr(config, 'oauth2_token', None)
    
    def save_token(token):
        """Save token to config/storage."""
        config.oauth2_token = token
    
    factory = APIClientFactory(
        client_id=config.XERO_CLIENT_ID,
        client_secret=config.XERO_CLIENT_SECRET,
        token_getter=get_token,
        token_saver=save_token,
        debug=config.DEBUG
    )
    
    return factory.create()
```

### 1.2 Multi-API Support

```python
# src/myproject/api/__init__.py
"""API module with multi-endpoint support."""

from .client import APIClientFactory, create_api_client_from_config
from .accounting import AccountingAPI
from .assets import AssetsAPI
from .projects import ProjectsAPI


class XeroAPI:
    """High-level Xero API wrapper with multiple endpoints."""
    
    def __init__(self, api_client):
        """Initialize with configured API client."""
        self.api_client = api_client
        
        # Initialize endpoint handlers
        self.accounting = AccountingAPI(api_client)
        self.assets = AssetsAPI(api_client)
        self.projects = ProjectsAPI(api_client)
    
    @classmethod
    def from_config(cls, config):
        """Create XeroAPI instance from config."""
        api_client = create_api_client_from_config(config)
        return cls(api_client)


# src/myproject/api/accounting.py
from xero_python.accounting import AccountingApi
from typing import List
import logging

logger = logging.getLogger(__name__)


class AccountingAPI:
    """Accounting API endpoint wrapper."""
    
    def __init__(self, api_client):
        """Initialize with API client."""
        self._api = AccountingApi(api_client)
    
    def get_invoices(
        self,
        xero_tenant_id: str,
        status: str = "DRAFT,SUBMITTED,AUTHORISED,SUBMITTED"
    ) -> List:
        """
        Get invoices for tenant.
        
        Args:
            xero_tenant_id: Xero tenant identifier
            status: Invoice status filter
            
        Returns:
            List of invoices
        """
        try:
            invoices = self._api.get_invoices(
                xero_tenant_id,
                where=f'Status="{status}"'
            )
            logger.info(f"Retrieved {len(invoices)} invoices")
            return invoices
            
        except Exception as e:
            logger.error(f"Failed to fetch invoices: {str(e)}")
            raise
    
    def create_invoice(self, xero_tenant_id: str, invoice_data: dict):
        """Create new invoice."""
        try:
            result = self._api.create_invoices(xero_tenant_id, invoice_data)
            logger.info("Invoice created successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to create invoice: {str(e)}")
            raise
```

---

## 2. Configuration Management

### 2.1 Pydantic-Based Configuration

```python
# src/myproject/config.py
"""Application configuration with Pydantic v2."""

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


class APIConfig(BaseModel):
    """API configuration."""
    
    client_id: str = Field(..., min_length=1)
    client_secret: str = Field(..., min_length=1)
    base_url: str = "https://api.xero.com"
    timeout: int = Field(default=30, ge=1, le=300)
    pool_threads: int = Field(default=1, ge=1)
    debug: bool = False


class TokenConfig(BaseModel):
    """Token configuration."""
    
    storage_type: str = Field(default="file")  # file, redis, database
    storage_path: Optional[str] = Field(default=None)
    auto_refresh: bool = True
    refresh_threshold_seconds: int = 300  # Refresh 5min before expiry


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    xero_client_id: str = Field(..., alias="XERO_CLIENT_ID")
    xero_client_secret: str = Field(..., alias="XERO_CLIENT_SECRET")
    xero_base_url: str = Field(
        default="https://api.xero.com",
        alias="XERO_BASE_URL"
    )
    xero_timeout: int = Field(default=30, alias="XERO_TIMEOUT")
    
    # Token Configuration
    token_storage_type: str = Field(
        default="file",
        alias="TOKEN_STORAGE_TYPE"
    )
    token_storage_path: Optional[str] = Field(
        default=None,
        alias="TOKEN_STORAGE_PATH"
    )
    
    # Application Settings
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def api_config(self) -> APIConfig:
        """Get API configuration."""
        return APIConfig(
            client_id=self.xero_client_id,
            client_secret=self.xero_client_secret,
            base_url=self.xero_base_url,
            timeout=self.xero_timeout,
            debug=self.debug
        )
    
    @property
    def token_config(self) -> TokenConfig:
        """Get token configuration."""
        return TokenConfig(
            storage_type=self.token_storage_type,
            storage_path=self.token_storage_path or None
        )
    
    def validate_credentials(self):
        """Validate required credentials."""
        if not self.xero_client_id:
            raise ValueError("XERO_CLIENT_ID is required")
        if not self.xero_client_secret:
            raise ValueError("XERO_CLIENT_SECRET is required")
    
    @classmethod
    def from_env(cls):
        """Create settings from environment."""
        return cls()


# Usage
if __name__ == "__main__":
    settings = Settings.from_env()
    settings.validate_credentials()
    print(f"API Config: {settings.api_config}")
```

### 2.2 Example .env File

```bash
# .env.example
# Xero OAuth2 Configuration
XERO_CLIENT_ID=your-client-id-here
XERO_CLIENT_SECRET=your-client-secret-here
XERO_BASE_URL=https://api.xero.com
XERO_TIMEOUT=30

# Token Management
TOKEN_STORAGE_TYPE=file
TOKEN_STORAGE_PATH=./var/tokens/xero_token.json

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
```

---

## 3. Authentication Patterns

### 3.1 Token Storage Abstraction

```python
# src/myproject/auth/token_storage.py
"""Token storage abstraction layer."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TokenStorage(ABC):
    """Abstract base class for token storage."""
    
    @abstractmethod
    def get(self) -> Optional[Dict[str, Any]]:
        """Retrieve stored token."""
        pass
    
    @abstractmethod
    def save(self, token: Dict[str, Any]) -> None:
        """Save token."""
        pass
    
    @abstractmethod
    def delete(self) -> None:
        """Delete stored token."""
        pass
    
    @abstractmethod
    def exists(self) -> bool:
        """Check if token exists."""
        pass


class FileTokenStorage(TokenStorage):
    """File-based token storage."""
    
    def __init__(self, path: str = "./var/tokens/xero_token.json"):
        """Initialize file storage."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def get(self) -> Optional[Dict[str, Any]]:
        """Retrieve token from file."""
        try:
            if not self.path.exists():
                return None
            
            with open(self.path, 'r') as f:
                token = json.load(f)
            
            logger.debug(f"Token retrieved from {self.path}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to retrieve token: {str(e)}")
            return None
    
    def save(self, token: Dict[str, Any]) -> None:
        """Save token to file."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.path, 'w') as f:
                json.dump(token, f, indent=2)
            
            # Restrict permissions to owner only
            self.path.chmod(0o600)
            logger.debug(f"Token saved to {self.path}")
            
        except Exception as e:
            logger.error(f"Failed to save token: {str(e)}")
            raise
    
    def delete(self) -> None:
        """Delete token file."""
        try:
            if self.path.exists():
                self.path.unlink()
                logger.info(f"Token deleted from {self.path}")
        except Exception as e:
            logger.error(f"Failed to delete token: {str(e)}")


class RedisTokenStorage(TokenStorage):
    """Redis-based token storage for distributed systems."""
    
    def __init__(self, redis_client, key: str = "xero:oauth2:token"):
        """Initialize Redis storage."""
        self.redis = redis_client
        self.key = key
    
    def get(self) -> Optional[Dict[str, Any]]:
        """Retrieve token from Redis."""
        try:
            token_json = self.redis.get(self.key)
            if not token_json:
                return None
            return json.loads(token_json)
        except Exception as e:
            logger.error(f"Failed to retrieve token from Redis: {str(e)}")
            return None
    
    def save(self, token: Dict[str, Any]) -> None:
        """Save token to Redis with TTL."""
        try:
            token_json = json.dumps(token)
            # Set expiry based on token's expires_in
            ttl = token.get('expires_in', 3600) + 3600  # Add 1hr buffer
            self.redis.setex(self.key, ttl, token_json)
            logger.debug(f"Token saved to Redis with {ttl}s TTL")
        except Exception as e:
            logger.error(f"Failed to save token to Redis: {str(e)}")
            raise
    
    def delete(self) -> None:
        """Delete token from Redis."""
        try:
            self.redis.delete(self.key)
            logger.info("Token deleted from Redis")
        except Exception as e:
            logger.error(f"Failed to delete token from Redis: {str(e)}")


def create_token_storage(storage_type: str, **kwargs) -> TokenStorage:
    """Factory function to create token storage."""
    if storage_type == "file":
        return FileTokenStorage(kwargs.get("path", "./var/tokens/xero_token.json"))
    elif storage_type == "redis":
        return RedisTokenStorage(kwargs["redis_client"], kwargs.get("key"))
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
```

### 3.2 OAuth2 Flow Handler

```python
# src/myproject/auth/oauth2.py
"""OAuth2 authentication flow handler."""

from typing import Optional, Dict, Any
from urllib.parse import urlencode
import logging

logger = logging.getLogger(__name__)


class OAuth2Handler:
    """Handle OAuth2 authentication flow."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        auth_endpoint: str = "https://login.xero.com/identity/connect/authorize",
        token_endpoint: str = "https://identity.xero.com/connect/token"
    ):
        """Initialize OAuth2 handler."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_endpoint = auth_endpoint
        self.token_endpoint = token_endpoint
    
    def get_authorization_url(
        self,
        state: str,
        scope: str = "openid profile email offline_access"
    ) -> str:
        """
        Generate OAuth2 authorization URL.
        
        Args:
            state: CSRF protection state string
            scope: OAuth2 scopes
            
        Returns:
            Authorization URL
        """
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "state": state
        }
        
        url = f"{self.auth_endpoint}?{urlencode(params)}"
        logger.debug(f"Generated authorization URL")
        return url
    
    def exchange_code_for_token(
        self,
        code: str,
        http_client
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from callback
            http_client: HTTP client instance
            
        Returns:
            Token dictionary
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            response = http_client.post(self.token_endpoint, data=data)
            response.raise_for_status()
            token = response.json()
            
            logger.info("Token obtained from authorization code")
            return token
            
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {str(e)}")
            raise
    
    def refresh_token(
        self,
        refresh_token: str,
        http_client
    ) -> Dict[str, Any]:
        """
        Refresh OAuth2 access token.
        
        Args:
            refresh_token: Refresh token
            http_client: HTTP client instance
            
        Returns:
            New token dictionary
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            response = http_client.post(self.token_endpoint, data=data)
            response.raise_for_status()
            token = response.json()
            
            logger.info("Token refreshed successfully")
            return token
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {str(e)}")
            raise
```

---

## 4. Data Models

### 4.1 Pydantic Models with Validation

```python
# src/myproject/models/accounting.py
"""Accounting data models with Pydantic v2."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class InvoiceStatus(str, Enum):
    """Invoice status enumeration."""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    AUTHORISED = "AUTHORISED"
    PAID = "PAID"


class LineItem(BaseModel):
    """Invoice line item."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    line_item_id: Optional[str] = Field(None, alias="LineItemID")
    description: str
    quantity: float = Field(gt=0)
    unit_amount: float = Field(ge=0, alias="UnitAmount")
    account_code: str = Field(alias="AccountCode")
    
    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        """Validate description length."""
        if not v or len(v) > 4000:
            raise ValueError("Description must be 1-4000 characters")
        return v


class Contact(BaseModel):
    """Contact data model."""
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "contact_id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "John Doe",
                "email_address": "john@example.com"
            }
        }
    )
    
    contact_id: Optional[str] = Field(None, alias="ContactID")
    name: str = Field(min_length=1, max_length=255)
    email_address: Optional[str] = Field(None, alias="EmailAddress")
    phone: Optional[str] = Field(None)
    is_customer: bool = Field(default=True, alias="IsCustomer")
    created_utc: Optional[datetime] = Field(None, alias="UpdatedUtc")


class Invoice(BaseModel):
    """Invoice data model."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    invoice_id: Optional[str] = Field(None, alias="InvoiceID")
    invoice_number: str = Field(min_length=1, alias="InvoiceNumber")
    status: InvoiceStatus = Field(alias="Status")
    contact: Contact
    line_items: List[LineItem] = Field(default_factory=list, alias="LineItems")
    due_date: Optional[datetime] = Field(None, alias="DueDate")
    reference: Optional[str] = Field(None)
    
    @property
    def total(self) -> float:
        """Calculate invoice total."""
        return sum(item.quantity * item.unit_amount for item in self.line_items)


# src/myproject/models/base.py
"""Base model with common fields."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class AuditableModel(BaseModel):
    """Base model with audit fields."""
    
    created_at: Optional[datetime] = Field(None)
    updated_at: Optional[datetime] = Field(None)
    created_by: Optional[str] = Field(None)
    updated_by: Optional[str] = Field(None)


class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    
    created_utc: Optional[datetime] = Field(None)
    updated_utc: Optional[datetime] = Field(None)
```

---

## 5. Error Handling

### 5.1 Custom Exception Hierarchy

```python
# src/myproject/exceptions.py
"""Custom exception classes for API errors."""

from typing import Optional, Dict, Any


class APIError(Exception):
    """Base API error."""
    
    def __init__(
        self,
        message: str,
        status_code: int,
        response_body: str = "",
        request_id: Optional[str] = None
    ):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            response_body: API response body
            request_id: Request ID for tracking
        """
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id
        super().__init__(self.message)
    
    def __str__(self):
        """String representation."""
        s = f"[{self.status_code}] {self.message}"
        if self.request_id:
            s += f" (Request ID: {self.request_id})"
        return s


class AuthenticationError(APIError):
    """OAuth2 authentication failure (401)."""
    pass


class AuthorizationError(APIError):
    """Insufficient permissions (403)."""
    pass


class ValidationError(APIError):
    """Request validation failed (400)."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize validation error."""
        super().__init__(message, 400, **kwargs)
        self.validation_errors = validation_errors or {}


class NotFoundError(APIError):
    """Resource not found (404)."""
    pass


class ConflictError(APIError):
    """Resource conflict (409)."""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded (429)."""
    
    def __init__(
        self,
        message: str,
        retry_after: int,
        **kwargs
    ):
        """Initialize rate limit error."""
        super().__init__(message, 429, **kwargs)
        self.retry_after = retry_after


class ServerError(APIError):
    """Server error (5xx)."""
    pass


class TimeoutError(APIError):
    """Request timeout."""
    pass


def map_http_error(status_code: int, message: str, response_body: str = "") -> APIError:
    """Map HTTP status code to appropriate exception."""
    
    error_mapping = {
        400: ValidationError,
        401: AuthenticationError,
        403: AuthorizationError,
        404: NotFoundError,
        409: ConflictError,
        429: RateLimitError,
    }
    
    # Get appropriate error class
    error_class = error_mapping.get(status_code, ServerError if status_code >= 500 else APIError)
    
    # Special handling for rate limit
    if status_code == 429:
        return error_class(message, retry_after=60)
    
    return error_class(message, status_code, response_body)
```

### 5.2 Error Handling in API Operations

```python
# src/myproject/api/operations.py
"""API operations with error handling."""

from .exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError
)
import logging
import time

logger = logging.getLogger(__name__)


def fetch_invoices_with_retry(
    api_client,
    xero_tenant_id: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> list:
    """
    Fetch invoices with exponential backoff retry logic.
    
    Args:
        api_client: Configured API client
        xero_tenant_id: Xero tenant ID
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
        
    Returns:
        List of invoices
        
    Raises:
        APIError: After retries exhausted
    """
    
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            accounting_api = AccountingApi(api_client)
            invoices = accounting_api.get_invoices(xero_tenant_id)
            
            logger.info(f"Successfully fetched {len(invoices)} invoices")
            return invoices
            
        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e.message}")
            # Don't retry auth errors - require manual intervention
            raise
            
        except RateLimitError as e:
            # Calculate backoff time
            wait_time = e.retry_after or (2 ** attempt) * backoff_factor
            
            logger.warning(
                f"Rate limited. Waiting {wait_time}s before retry "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            
            time.sleep(wait_time)
            attempt += 1
            last_error = e
            
        except ValidationError as e:
            logger.error(f"Validation error: {e.message}")
            logger.debug(f"Validation errors: {e.validation_errors}")
            # Don't retry validation errors
            raise
            
        except APIError as e:
            # Retry other API errors with backoff
            wait_time = (2 ** attempt) * backoff_factor
            
            logger.warning(
                f"API error ({e.status_code}): {e.message}. "
                f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
            )
            
            time.sleep(wait_time)
            attempt += 1
            last_error = e
            
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            raise
    
    # All retries exhausted
    logger.error(f"Failed to fetch invoices after {max_retries} retries")
    raise last_error or APIError("Failed to fetch invoices", 500)
```

---

## 6. API Operations

### 6.1 Complete API Operation Example

```python
# src/myproject/api/invoices.py
"""Invoice operations."""

from typing import List, Optional
from datetime import datetime
import logging

from xero_python.accounting import AccountingApi
from xero_python.accounting.models import Invoice as XeroInvoice
from xero_python.accounting.models import LineItem as XeroLineItem
from xero_python.accounting.models import Contact as XeroContact

from myproject.models import Invoice, Contact, LineItem
from myproject.exceptions import (
    APIError,
    NotFoundError,
    ValidationError,
    RateLimitError
)

logger = logging.getLogger(__name__)


class InvoiceOperations:
    """Invoice operations wrapper."""
    
    def __init__(self, api_client):
        """Initialize invoice operations."""
        self.api = AccountingApi(api_client)
    
    def get_invoices(
        self,
        xero_tenant_id: str,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Invoice]:
        """
        Get invoices for tenant.
        
        Args:
            xero_tenant_id: Xero tenant ID
            status: Filter by status (comma-separated)
            limit: Maximum invoices to return
            
        Returns:
            List of Invoice models
            
        Raises:
            APIError: On API failure
        """
        try:
            # Build where clause
            where = None
            if status:
                where = f'Status="{status}"'
            
            # Fetch invoices from API
            response = self.api.get_invoices(
                xero_tenant_id,
                where=where
            )
            
            # Convert to our models
            invoices = [
                self._xero_invoice_to_model(invoice)
                for invoice in (response.invoices or [])
            ]
            
            logger.info(f"Fetched {len(invoices)} invoices")
            return invoices
            
        except RateLimitError:
            raise  # Re-raise rate limit errors
            
        except Exception as e:
            logger.error(f"Failed to fetch invoices: {str(e)}")
            raise APIError(f"Failed to fetch invoices: {str(e)}", 500)
    
    def create_invoice(
        self,
        xero_tenant_id: str,
        invoice: Invoice
    ) -> Invoice:
        """
        Create new invoice.
        
        Args:
            xero_tenant_id: Xero tenant ID
            invoice: Invoice data
            
        Returns:
            Created invoice with ID
            
        Raises:
            ValidationError: On invalid data
            APIError: On API failure
        """
        try:
            # Convert to Xero model
            xero_invoice = self._model_to_xero_invoice(invoice)
            
            # Create invoice
            response = self.api.create_invoices(
                xero_tenant_id,
                xero_invoice
            )
            
            # Return created invoice
            created = response.invoices[0] if response.invoices else None
            if not created:
                raise APIError("No invoice returned from create", 500)
            
            logger.info(f"Invoice created: {created.invoice_id}")
            return self._xero_invoice_to_model(created)
            
        except ValidationError:
            raise  # Re-raise validation errors
            
        except Exception as e:
            logger.error(f"Failed to create invoice: {str(e)}")
            raise APIError(f"Failed to create invoice: {str(e)}", 500)
    
    def _xero_invoice_to_model(self, xero_invoice: XeroInvoice) -> Invoice:
        """Convert Xero invoice to our model."""
        return Invoice(
            invoice_id=xero_invoice.invoice_id,
            invoice_number=xero_invoice.invoice_number,
            status=xero_invoice.status,
            contact=Contact(
                contact_id=xero_invoice.contact.contact_id,
                name=xero_invoice.contact.name,
                email_address=xero_invoice.contact.email_address
            ),
            line_items=[
                LineItem(
                    description=item.description,
                    quantity=item.quantity,
                    unit_amount=item.unit_amount,
                    account_code=item.account_code
                )
                for item in (xero_invoice.line_items or [])
            ]
        )
    
    def _model_to_xero_invoice(self, invoice: Invoice) -> XeroInvoice:
        """Convert our model to Xero invoice."""
        return XeroInvoice(
            invoice_number=invoice.invoice_number,
            contact=XeroContact(name=invoice.contact.name),
            line_items=[
                XeroLineItem(
                    description=item.description,
                    quantity=item.quantity,
                    unit_amount=item.unit_amount,
                    account_code=item.account_code
                )
                for item in invoice.line_items
            ]
        )
```

---

## 7. Testing Fixtures

### 7.1 Pytest Fixtures

```python
# tests/conftest.py
"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from xero_python.api_client import OAuth2Token

from myproject.config import Settings
from myproject.api.client import APIClientFactory
from myproject.auth.token_storage import FileTokenStorage


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return Settings(
        xero_client_id="test-client-id",
        xero_client_secret="test-client-secret",
        debug=True
    )


@pytest.fixture
def mock_oauth2_token():
    """Provide mock OAuth2 token."""
    token = Mock(spec=OAuth2Token)
    token.access_token = "test-access-token"
    token.refresh_token = "test-refresh-token"
    token.expires_in = 3600
    token.scope = ["openid", "profile", "email"]
    return token


@pytest.fixture
def mock_api_client(mock_oauth2_token):
    """Provide mock API client."""
    client = MagicMock()
    client.oauth2_token = mock_oauth2_token
    return client


@pytest.fixture
def token_storage(tmp_path):
    """Provide temporary token storage."""
    storage_path = tmp_path / "tokens" / "token.json"
    return FileTokenStorage(str(storage_path))


@pytest.fixture
def sample_token():
    """Provide sample OAuth2 token."""
    return {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
        "scope": "openid profile email offline_access",
        "token_type": "Bearer"
    }


@pytest.fixture
def api_client_factory(test_config):
    """Provide API client factory."""
    return APIClientFactory(
        client_id=test_config.xero_client_id,
        client_secret=test_config.xero_client_secret
    )
```

### 7.2 Mock Response Fixtures

```python
# tests/fixtures/api_responses.py
"""Mock API responses for testing."""

from dataclasses import dataclass
from typing import List


@dataclass
class MockInvoice:
    """Mock invoice object."""
    invoice_id: str = "123e4567-e89b-12d3-a456-426614174000"
    invoice_number: str = "INV001"
    status: str = "AUTHORISED"
    total: float = 1000.00


@dataclass
class MockContact:
    """Mock contact object."""
    contact_id: str = "456e7890-f12g-34hi-j567-890123456789"
    name: str = "John Doe"
    email_address: str = "john@example.com"


def create_mock_invoice_response(invoice_count: int = 1) -> dict:
    """Create mock invoice API response."""
    return {
        "invoices": [
            MockInvoice(
                invoice_id=f"invoice-{i}",
                invoice_number=f"INV{i:03d}"
            )
            for i in range(invoice_count)
        ]
    }


# tests/unit/test_invoices.py
"""Unit tests for invoice operations."""

import pytest
from unittest.mock import patch, Mock
from myproject.api.invoices import InvoiceOperations
from myproject.models import Invoice
from myproject.exceptions import APIError


class TestInvoiceOperations:
    """Tests for InvoiceOperations."""
    
    @pytest.fixture
    def operations(self, mock_api_client):
        """Provide invoice operations."""
        with patch('myproject.api.invoices.AccountingApi') as mock_api:
            mock_api.return_value = Mock(spec=AccountingApi)
            return InvoiceOperations(mock_api_client)
    
    def test_get_invoices_success(self, operations):
        """Test successful invoice fetch."""
        # Arrange
        xero_tenant_id = "test-tenant-id"
        
        with patch.object(operations.api, 'get_invoices') as mock_get:
            mock_get.return_value = create_mock_invoice_response(2)
            
            # Act
            invoices = operations.get_invoices(xero_tenant_id)
            
            # Assert
            assert len(invoices) == 2
            assert invoices[0].invoice_number == "INV-0"
            mock_get.assert_called_once()
    
    def test_get_invoices_with_status_filter(self, operations):
        """Test invoice fetch with status filter."""
        # Arrange
        xero_tenant_id = "test-tenant-id"
        
        with patch.object(operations.api, 'get_invoices') as mock_get:
            mock_get.return_value = create_mock_invoice_response(1)
            
            # Act
            operations.get_invoices(xero_tenant_id, status="AUTHORISED")
            
            # Assert
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert 'where' in call_kwargs
    
    def test_create_invoice_validation(self, operations):
        """Test invoice creation with validation."""
        # Arrange
        invoice = Invoice(
            invoice_number="",  # Invalid: empty
            status="DRAFT",
            contact=Mock(),
            line_items=[]
        )
        
        # Act & Assert
        with pytest.raises(ValueError):
            operations.create_invoice("tenant-id", invoice)
```

---

## 8. Middleware & Utilities

### 8.1 Rate Limit Monitor

```python
# src/myproject/middleware/rate_limit.py
"""Rate limit monitoring and enforcement."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class RateLimitMonitor:
    """Monitor API rate limits."""
    
    def __init__(
        self,
        daily_limit: int = 5000,
        warning_threshold: float = 0.1
    ):
        """
        Initialize rate limit monitor.
        
        Args:
            daily_limit: Daily API call limit
            warning_threshold: Warn when this % of limit remains (0.1 = 10%)
        """
        self.daily_limit = daily_limit
        self.warning_threshold = warning_threshold
        self.reset_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        self.calls_made = 0
    
    def check_response(self, response_headers: Dict[str, str]) -> None:
        """
        Check response headers for rate limit info.
        
        Args:
            response_headers: HTTP response headers
        """
        try:
            # Extract rate limit info from headers
            remaining = int(
                response_headers.get('x-rate-limit-remaining', self.daily_limit)
            )
            reset = int(
                response_headers.get('x-rate-limit-reset', 0)
            )
            
            self.calls_made = self.daily_limit - remaining
            
            if reset:
                self.reset_time = datetime.utcfromtimestamp(reset)
            
            # Check if approaching limit
            if remaining < self.daily_limit * self.warning_threshold:
                logger.warning(
                    f"Rate limit warning: {remaining}/{self.daily_limit} "
                    f"calls remaining. Reset at {self.reset_time}"
                )
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse rate limit headers: {e}")
    
    def should_request_proceed(self) -> bool:
        """Check if safe to make request."""
        return self.calls_made < self.daily_limit
    
    def get_calls_remaining(self) -> int:
        """Get estimated remaining calls."""
        return self.daily_limit - self.calls_made


class RateLimitMiddleware:
    """Middleware to enforce rate limits."""
    
    def __init__(self, monitor: RateLimitMonitor):
        """Initialize middleware."""
        self.monitor = monitor
    
    def before_request(self, request) -> Optional[Exception]:
        """Check rate limits before request."""
        if not self.monitor.should_request_proceed():
            seconds_until_reset = (
                self.monitor.reset_time - datetime.utcnow()
            ).total_seconds()
            
            from myproject.exceptions import RateLimitError
            return RateLimitError(
                "Rate limit exceeded",
                retry_after=int(seconds_until_reset)
            )
        return None
    
    def after_response(self, response) -> None:
        """Update monitor with response info."""
        self.monitor.check_response(response.headers)
```

### 8.2 Request Retry Handler

```python
# src/myproject/middleware/retry.py
"""Retry logic with exponential backoff."""

import time
import logging
from typing import Callable, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 1.0,
        backoff_max: int = 60,
        retryable_exceptions: Tuple[Type[Exception], ...] = None
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of attempts
            backoff_factor: Exponential backoff multiplier
            backoff_max: Maximum backoff time in seconds
            retryable_exceptions: Exception types to retry on
        """
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self.retryable_exceptions = retryable_exceptions or (Exception,)


def retry(config: RetryConfig = None):
    """
    Decorator to add retry logic to function.
    
    Args:
        config: RetryConfig instance
        
    Returns:
        Decorator function
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after "
                            f"{config.max_attempts} attempts: {str(e)}"
                        )
                        raise
                    
                    # Calculate backoff
                    backoff = min(
                        (2 ** (attempt - 1)) * config.backoff_factor,
                        config.backoff_max
                    )
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/"
                        f"{config.max_attempts}). "
                        f"Retrying in {backoff}s: {str(e)}"
                    )
                    
                    time.sleep(backoff)
            
            raise last_exception
        
        return wrapper
    return decorator


# Usage example
@retry(RetryConfig(max_attempts=3, backoff_factor=1.5))
def fetch_with_retry(api_client, tenant_id):
    """Fetch data with automatic retry."""
    return api_client.get_invoices(tenant_id)
```

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Status**: Ready for Template Integration
