"""Authentication configuration for PromptCraft.

This module extends the core application settings with authentication-specific
configuration including Cloudflare Access integration, email whitelisting,
and role mapping.
"""

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class AuthenticationConfig(BaseModel):
    """Authentication configuration settings."""

    # Cloudflare Access Configuration
    cloudflare_access_enabled: bool = Field(
        default=True,
        description="Whether Cloudflare Access authentication is enabled",
    )

    cloudflare_team_domain: str = Field(
        default="",
        description="Cloudflare team domain (e.g., 'myteam' for myteam.cloudflareaccess.com)",
    )

    cloudflare_jwks_url: str = Field(
        default="",
        description="Cloudflare JWKS endpoint URL for JWT validation",
    )

    cloudflare_audience: str | None = Field(
        default=None,
        description="Expected audience (aud) claim in Cloudflare JWT tokens",
    )

    cloudflare_issuer: str | None = Field(
        default=None,
        description="Expected issuer (iss) claim in Cloudflare JWT tokens",
    )

    # JWT Configuration
    jwt_algorithm: str = Field(
        default="RS256",
        description="JWT algorithm for signature verification",
    )

    jwt_leeway: int = Field(
        default=10,
        description="JWT clock skew tolerance in seconds",
    )

    # JWKS Caching Configuration
    jwks_cache_ttl: int = Field(
        default=3600,  # 1 hour
        description="JWKS cache TTL in seconds",
    )

    jwks_cache_max_size: int = Field(
        default=10,
        description="Maximum number of cached JWKS entries",
    )

    jwks_timeout: int = Field(
        default=10,
        description="JWKS HTTP request timeout in seconds",
    )

    # Email Whitelist Configuration
    email_whitelist: list[str] = Field(
        default_factory=list,
        description="List of allowed email addresses or domains (use @domain.com for domain)",
    )

    email_whitelist_enabled: bool = Field(
        default=True,
        description="Whether email whitelist validation is enabled",
    )

    # Role Mapping Configuration
    admin_emails: list[str] = Field(
        default_factory=list,
        description="List of email addresses that should have admin role",
    )

    admin_email_domains: list[str] = Field(
        default_factory=list,
        description="List of email domains that should have admin role (without @)",
    )

    # Rate Limiting Configuration
    rate_limiting_enabled: bool = Field(
        default=True,
        description="Whether rate limiting is enabled for authentication",
    )

    rate_limit_requests: int = Field(
        default=100,
        description="Number of requests allowed per rate limit window",
    )

    rate_limit_window: int = Field(
        default=3600,  # 1 hour
        description="Rate limit window in seconds",
    )

    rate_limit_key_func: str = Field(
        default="email",
        description="Rate limiting key function (ip, email, or user)",
    )

    # Session Configuration
    session_cookie_name: str = Field(
        default="promptcraft_session",
        description="Name of the session cookie",
    )

    session_cookie_secure: bool = Field(
        default=True,
        description="Whether session cookies should be marked as secure",
    )

    session_cookie_httponly: bool = Field(
        default=True,
        description="Whether session cookies should be HTTP-only",
    )

    session_cookie_samesite: str = Field(
        default="lax",
        description="SameSite attribute for session cookies",
    )

    # Error Handling Configuration
    auth_error_detail_enabled: bool = Field(
        default=False,
        description="Whether to include detailed error messages in authentication failures",
    )

    auth_logging_enabled: bool = Field(
        default=True,
        description="Whether to log authentication events",
    )

    auth_metrics_enabled: bool = Field(
        default=True,
        description="Whether to collect authentication metrics",
    )

    @field_validator("cloudflare_jwks_url")
    @classmethod
    def validate_jwks_url(cls, v: str, info: ValidationInfo) -> str:
        """Validate and auto-generate JWKS URL if needed."""
        if not v and info.data.get("cloudflare_team_domain"):
            # Auto-generate JWKS URL from team domain
            team_domain = info.data["cloudflare_team_domain"]
            return f"https://{team_domain}.cloudflareaccess.com/cdn-cgi/access/certs"
        return v

    @field_validator("email_whitelist")
    @classmethod
    def validate_email_whitelist(cls, v: list[str]) -> list[str]:
        """Validate email whitelist entries."""
        validated = []
        for email_entry in v:
            email_addr = email_entry.strip().lower()
            if email_addr:
                # Basic validation for email format or domain format
                if email_addr.startswith("@"):
                    # Domain entry - must have at least one dot after @
                    if "." in email_addr[1:]:
                        validated.append(email_addr)
                elif "@" in email_addr:
                    # Email entry - basic format check
                    validated.append(email_addr)
        return validated

    @field_validator("admin_email_domains")
    @classmethod
    def validate_admin_domains(cls, v: list[str]) -> list[str]:
        """Validate admin email domains."""
        validated = []
        for domain_entry in v:
            domain_name = domain_entry.strip().lower()
            if domain_name and "." in domain_name:
                # Remove @ prefix if present
                if domain_name.startswith("@"):
                    domain_name = domain_name[1:]
                validated.append(domain_name)
        return validated

    @field_validator("rate_limit_key_func")
    @classmethod
    def validate_rate_limit_key_func(cls, v: str) -> str:
        """Validate rate limit key function."""
        allowed_values = ["ip", "email", "user"]
        if v not in allowed_values:
            raise ValueError(f"rate_limit_key_func must be one of: {allowed_values}")
        return v

    @field_validator("session_cookie_samesite")
    @classmethod
    def validate_samesite(cls, v: str) -> str:
        """Validate SameSite cookie attribute."""
        allowed_values = ["strict", "lax", "none"]
        if v.lower() not in allowed_values:
            raise ValueError(f"session_cookie_samesite must be one of: {allowed_values}")
        return v.lower()

    def get_jwks_url(self) -> str:
        """Get the JWKS URL, auto-generating if needed."""
        if self.cloudflare_jwks_url:
            return self.cloudflare_jwks_url
        if self.cloudflare_team_domain:
            return f"https://{self.cloudflare_team_domain}.cloudflareaccess.com/cdn-cgi/access/certs"
        raise ValueError("Either cloudflare_jwks_url or cloudflare_team_domain must be configured")

    def is_admin_email(self, email: str) -> bool:
        """Check if an email should have admin role.

        Args:
            email: Email address to check

        Returns:
            True if email should have admin role, False otherwise
        """
        email_lower = email.lower()

        # Check exact email matches
        if email_lower in [admin.lower() for admin in self.admin_emails]:
            return True

        # Check domain matches
        email_domain = email_lower.split("@")[-1] if "@" in email_lower else ""
        return email_domain in [domain.lower() for domain in self.admin_email_domains]
