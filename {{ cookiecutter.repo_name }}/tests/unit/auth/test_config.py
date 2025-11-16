"""
Unit tests for AuthenticationConfig class.

This module provides comprehensive test coverage for the AuthenticationConfig class,
testing all field validators, methods, configuration options, and edge cases.
Uses proper pytest markers for codecov integration per codecov.yml auth component.
"""

import pytest
from pydantic import ValidationError

from src.auth.config import AuthenticationConfig


@pytest.mark.auth
class TestAuthenticationConfigInitialization:
    """Test cases for AuthenticationConfig initialization."""

    def test_init_default_values(self):
        """Test initialization with default configuration values."""
        config = AuthenticationConfig()

        # Cloudflare Access defaults
        assert config.cloudflare_access_enabled is True
        assert config.cloudflare_team_domain == ""
        assert config.cloudflare_jwks_url == ""
        assert config.cloudflare_audience is None
        assert config.cloudflare_issuer is None

        # JWT defaults
        assert config.jwt_algorithm == "RS256"
        assert config.jwt_leeway == 10

        # JWKS caching defaults
        assert config.jwks_cache_ttl == 3600
        assert config.jwks_cache_max_size == 10
        assert config.jwks_timeout == 10

        # Email whitelist defaults
        assert config.email_whitelist == []
        assert config.email_whitelist_enabled is True

        # Role mapping defaults
        assert config.admin_emails == []
        assert config.admin_email_domains == []

        # Rate limiting defaults
        assert config.rate_limiting_enabled is True
        assert config.rate_limit_requests == 100
        assert config.rate_limit_window == 3600
        assert config.rate_limit_key_func == "email"

        # Session defaults
        assert config.session_cookie_name == "promptcraft_session"
        assert config.session_cookie_secure is True
        assert config.session_cookie_httponly is True
        assert config.session_cookie_samesite == "lax"

        # Error handling defaults
        assert config.auth_error_detail_enabled is False
        assert config.auth_logging_enabled is True
        assert config.auth_metrics_enabled is True

    def test_init_custom_values(self):
        """Test initialization with custom configuration values."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=False,
            cloudflare_team_domain="testteam",
            jwt_algorithm="HS256",
            jwt_leeway=30,
            jwks_cache_ttl=7200,
            email_whitelist=["admin@example.com", "@company.com"],
            rate_limit_requests=50,
            session_cookie_name="custom_session",
            auth_error_detail_enabled=True,
        )

        assert config.cloudflare_access_enabled is False
        assert config.cloudflare_team_domain == "testteam"
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_leeway == 30
        assert config.jwks_cache_ttl == 7200
        assert "admin@example.com" in config.email_whitelist
        assert "@company.com" in config.email_whitelist
        assert config.rate_limit_requests == 50
        assert config.session_cookie_name == "custom_session"
        assert config.auth_error_detail_enabled is True

    def test_init_partial_custom_values(self):
        """Test initialization with partially customized configuration."""
        config = AuthenticationConfig(cloudflare_team_domain="myteam", jwt_leeway=5, email_whitelist_enabled=False)

        # Custom values
        assert config.cloudflare_team_domain == "myteam"
        assert config.jwt_leeway == 5
        assert config.email_whitelist_enabled is False

        # Default values should remain
        assert config.cloudflare_access_enabled is True
        assert config.jwt_algorithm == "RS256"
        assert config.rate_limit_requests == 100


@pytest.mark.auth
class TestAuthenticationConfigValidators:
    """Test cases for field validators."""

    def test_validate_jwks_url_auto_generation(self):
        """Test JWKS URL auto-generation from team domain."""
        config = AuthenticationConfig(
            cloudflare_team_domain="myteam",
            cloudflare_jwks_url="",  # Empty, should be auto-generated
        )

        expected_url = "https://myteam.cloudflareaccess.com/cdn-cgi/access/certs"
        assert config.cloudflare_jwks_url == expected_url

    def test_validate_jwks_url_explicit_value(self):
        """Test JWKS URL with explicit value provided."""
        explicit_url = "https://custom.example.com/jwks"
        config = AuthenticationConfig(cloudflare_team_domain="myteam", cloudflare_jwks_url=explicit_url)

        # Should use explicit value, not auto-generate
        assert config.cloudflare_jwks_url == explicit_url

    def test_validate_jwks_url_no_team_domain(self):
        """Test JWKS URL validation when no team domain provided."""
        config = AuthenticationConfig(cloudflare_team_domain="", cloudflare_jwks_url="")

        # Should remain empty when no team domain
        assert config.cloudflare_jwks_url == ""

    def test_validate_email_whitelist_valid_entries(self):
        """Test email whitelist validation with valid entries."""
        config = AuthenticationConfig(
            email_whitelist=[
                "user@example.com",
                "@company.com",
                "  admin@test.org  ",  # With whitespace
                "CAPS@EXAMPLE.COM",  # With uppercase
            ],
        )

        assert "user@example.com" in config.email_whitelist
        assert "@company.com" in config.email_whitelist
        assert "admin@test.org" in config.email_whitelist
        assert "caps@example.com" in config.email_whitelist

    def test_validate_email_whitelist_invalid_entries(self):
        """Test email whitelist validation filters out invalid entries."""
        config = AuthenticationConfig(
            email_whitelist=[
                "user@example.com",  # Valid
                "@",  # Invalid - no domain
                "@nodot",  # Invalid - no dot in domain
                "invalid-email",  # Invalid - no @ symbol
                "",  # Invalid - empty
                "   ",  # Invalid - whitespace only
                "@valid.com",  # Valid
            ],
        )

        # Should only contain valid entries
        assert "user@example.com" in config.email_whitelist
        assert "@valid.com" in config.email_whitelist
        assert len(config.email_whitelist) == 2

    def test_validate_email_whitelist_empty_list(self):
        """Test email whitelist validation with empty list."""
        config = AuthenticationConfig(email_whitelist=[])

        assert config.email_whitelist == []

    def test_validate_admin_domains_valid_entries(self):
        """Test admin email domains validation with valid entries."""
        config = AuthenticationConfig(
            admin_email_domains=[
                "company.com",
                "@admin.org",  # With @ prefix
                "  test.net  ",  # With whitespace
                "EXAMPLE.COM",  # With uppercase
            ],
        )

        assert "company.com" in config.admin_email_domains
        assert "admin.org" in config.admin_email_domains  # @ removed
        assert "test.net" in config.admin_email_domains
        assert "example.com" in config.admin_email_domains

    def test_validate_admin_domains_invalid_entries(self):
        """Test admin email domains validation filters out invalid entries."""
        config = AuthenticationConfig(
            admin_email_domains=[
                "valid.com",  # Valid
                "nodot",  # Invalid - no dot
                "",  # Invalid - empty
                "   ",  # Invalid - whitespace only
                "another.org",  # Valid
            ],
        )

        # Should only contain valid entries
        assert "valid.com" in config.admin_email_domains
        assert "another.org" in config.admin_email_domains
        assert len(config.admin_email_domains) == 2

    def test_validate_rate_limit_key_func_valid_values(self):
        """Test rate limit key function validation with valid values."""
        for valid_value in ["ip", "email", "user"]:
            config = AuthenticationConfig(rate_limit_key_func=valid_value)
            assert config.rate_limit_key_func == valid_value

    def test_validate_rate_limit_key_func_invalid_value(self):
        """Test rate limit key function validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            AuthenticationConfig(rate_limit_key_func="invalid")

        error_details = exc_info.value.errors()[0]
        assert error_details["type"] == "value_error"
        assert "must be one of: ['ip', 'email', 'user']" in str(error_details["ctx"]["error"])

    def test_validate_samesite_valid_values(self):
        """Test SameSite cookie attribute validation with valid values."""
        for valid_value in ["strict", "lax", "none"]:
            config = AuthenticationConfig(session_cookie_samesite=valid_value)
            assert config.session_cookie_samesite == valid_value.lower()

    def test_validate_samesite_case_insensitive(self):
        """Test SameSite validation is case insensitive."""
        config = AuthenticationConfig(session_cookie_samesite="STRICT")
        assert config.session_cookie_samesite == "strict"

    def test_validate_samesite_invalid_value(self):
        """Test SameSite validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            AuthenticationConfig(session_cookie_samesite="invalid")

        error_details = exc_info.value.errors()[0]
        assert error_details["type"] == "value_error"
        assert "must be one of: ['strict', 'lax', 'none']" in str(error_details["ctx"]["error"])


@pytest.mark.auth
class TestAuthenticationConfigMethods:
    """Test cases for AuthenticationConfig methods."""

    def test_get_jwks_url_with_explicit_url(self):
        """Test get_jwks_url when explicit URL is configured."""
        explicit_url = "https://custom.example.com/jwks"
        config = AuthenticationConfig(cloudflare_jwks_url=explicit_url)

        result = config.get_jwks_url()
        assert result == explicit_url

    def test_get_jwks_url_with_team_domain(self):
        """Test get_jwks_url auto-generation from team domain."""
        config = AuthenticationConfig(cloudflare_team_domain="myteam", cloudflare_jwks_url="")

        result = config.get_jwks_url()
        expected = "https://myteam.cloudflareaccess.com/cdn-cgi/access/certs"
        assert result == expected

    def test_get_jwks_url_no_configuration(self):
        """Test get_jwks_url raises error when no configuration provided."""
        config = AuthenticationConfig(cloudflare_team_domain="", cloudflare_jwks_url="")

        with pytest.raises(
            ValueError,
            match="Either cloudflare_jwks_url or cloudflare_team_domain must be configured",
        ) as exc_info:
            config.get_jwks_url()

        assert "Either cloudflare_jwks_url or cloudflare_team_domain must be configured" in str(exc_info.value)

    def test_is_admin_email_exact_match(self):
        """Test is_admin_email with exact email matches."""
        config = AuthenticationConfig(admin_emails=["admin@company.com", "super@test.org"])

        assert config.is_admin_email("admin@company.com") is True
        assert config.is_admin_email("super@test.org") is True
        assert config.is_admin_email("user@company.com") is False

    def test_is_admin_email_case_insensitive(self):
        """Test is_admin_email is case insensitive for exact matches."""
        config = AuthenticationConfig(admin_emails=["Admin@Company.Com"])

        assert config.is_admin_email("admin@company.com") is True
        assert config.is_admin_email("ADMIN@COMPANY.COM") is True
        assert config.is_admin_email("Admin@Company.Com") is True

    def test_is_admin_email_domain_match(self):
        """Test is_admin_email with domain matches."""
        config = AuthenticationConfig(admin_email_domains=["company.com", "admin.org"])

        assert config.is_admin_email("user@company.com") is True
        assert config.is_admin_email("admin@admin.org") is True
        assert config.is_admin_email("test@external.com") is False

    def test_is_admin_email_domain_case_insensitive(self):
        """Test is_admin_email domain matching is case insensitive."""
        config = AuthenticationConfig(admin_email_domains=["Company.Com"])

        assert config.is_admin_email("user@company.com") is True
        assert config.is_admin_email("USER@COMPANY.COM") is True

    def test_is_admin_email_combined_rules(self):
        """Test is_admin_email with both exact emails and domain rules."""
        config = AuthenticationConfig(admin_emails=["specific@external.com"], admin_email_domains=["company.com"])

        # Should match exact email
        assert config.is_admin_email("specific@external.com") is True

        # Should match domain rule
        assert config.is_admin_email("anyone@company.com") is True

        # Should not match
        assert config.is_admin_email("other@external.com") is False

    def test_is_admin_email_invalid_email_format(self):
        """Test is_admin_email with invalid email format."""
        config = AuthenticationConfig(admin_emails=["admin@company.com"], admin_email_domains=["company.com"])

        # Invalid email formats should return False
        assert config.is_admin_email("not-an-email") is False
        assert config.is_admin_email("missing-at-symbol.com") is False
        assert config.is_admin_email("@no-local-part.com") is False
        assert config.is_admin_email("") is False

    def test_is_admin_email_empty_configuration(self):
        """Test is_admin_email with no admin configuration."""
        config = AuthenticationConfig(admin_emails=[], admin_email_domains=[])

        assert config.is_admin_email("anyone@anywhere.com") is False
        assert config.is_admin_email("admin@company.com") is False


@pytest.mark.auth
class TestAuthenticationConfigEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def test_config_with_all_features_disabled(self):
        """Test configuration with all optional features disabled."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=False,
            email_whitelist_enabled=False,
            rate_limiting_enabled=False,
            auth_error_detail_enabled=False,
            auth_logging_enabled=False,
            auth_metrics_enabled=False,
        )

        assert config.cloudflare_access_enabled is False
        assert config.email_whitelist_enabled is False
        assert config.rate_limiting_enabled is False
        assert config.auth_error_detail_enabled is False
        assert config.auth_logging_enabled is False
        assert config.auth_metrics_enabled is False

    def test_config_with_extreme_values(self):
        """Test configuration with extreme but valid values."""
        config = AuthenticationConfig(
            jwt_leeway=0,  # Minimum leeway
            jwks_cache_ttl=1,  # Very short cache
            jwks_cache_max_size=1,  # Minimal cache size
            jwks_timeout=1,  # Short timeout
            rate_limit_requests=1,  # Very restrictive
            rate_limit_window=1,  # Very short window
        )

        assert config.jwt_leeway == 0
        assert config.jwks_cache_ttl == 1
        assert config.jwks_cache_max_size == 1
        assert config.jwks_timeout == 1
        assert config.rate_limit_requests == 1
        assert config.rate_limit_window == 1

    def test_config_serialization(self):
        """Test configuration can be serialized and deserialized."""
        original_config = AuthenticationConfig(
            cloudflare_team_domain="test",
            email_whitelist=["user@example.com"],
            admin_emails=["admin@test.com"],
        )

        # Serialize to dict
        config_dict = original_config.model_dump()

        # Deserialize back
        restored_config = AuthenticationConfig(**config_dict)

        assert restored_config.cloudflare_team_domain == original_config.cloudflare_team_domain
        assert restored_config.email_whitelist == original_config.email_whitelist
        assert restored_config.admin_emails == original_config.admin_emails

    def test_config_field_access(self):
        """Test all configuration fields are accessible."""
        config = AuthenticationConfig()

        # Test all fields can be accessed without error
        field_names = [
            "cloudflare_access_enabled",
            "cloudflare_team_domain",
            "cloudflare_jwks_url",
            "cloudflare_audience",
            "cloudflare_issuer",
            "jwt_algorithm",
            "jwt_leeway",
            "jwks_cache_ttl",
            "jwks_cache_max_size",
            "jwks_timeout",
            "email_whitelist",
            "email_whitelist_enabled",
            "admin_emails",
            "admin_email_domains",
            "rate_limiting_enabled",
            "rate_limit_requests",
            "rate_limit_window",
            "rate_limit_key_func",
            "session_cookie_name",
            "session_cookie_secure",
            "session_cookie_httponly",
            "session_cookie_samesite",
            "auth_error_detail_enabled",
            "auth_logging_enabled",
            "auth_metrics_enabled",
        ]

        for field_name in field_names:
            assert hasattr(config, field_name)
            # Should not raise exception
            _ = getattr(config, field_name)

    def test_config_type_validation(self):
        """Test configuration validates field types correctly."""
        # Test invalid types raise ValidationError
        with pytest.raises(ValidationError):
            AuthenticationConfig(cloudflare_access_enabled="not_a_bool")

        with pytest.raises(ValidationError):
            AuthenticationConfig(jwt_leeway="not_an_int")

        with pytest.raises(ValidationError):
            AuthenticationConfig(email_whitelist="not_a_list")

        with pytest.raises(ValidationError):
            AuthenticationConfig(admin_emails=123)


@pytest.mark.auth
class TestAuthenticationConfigIntegration:
    """Integration test cases for complex configuration scenarios."""

    def test_complete_cloudflare_setup(self):
        """Test complete Cloudflare Access configuration setup."""
        config = AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_team_domain="mycompany",
            cloudflare_audience="myapp",
            cloudflare_issuer="https://mycompany.cloudflareaccess.com",
            email_whitelist=["@mycompany.com", "contractor@external.com"],
            admin_emails=["admin@mycompany.com"],
            admin_email_domains=["mycompany.com"],
        )

        # Verify JWKS URL auto-generation
        expected_jwks = "https://mycompany.cloudflareaccess.com/cdn-cgi/access/certs"
        assert config.get_jwks_url() == expected_jwks

        # Verify admin detection
        assert config.is_admin_email("admin@mycompany.com") is True
        assert config.is_admin_email("user@mycompany.com") is True  # Domain rule
        assert config.is_admin_email("contractor@external.com") is False  # Not admin

        # Verify email whitelist validation would work
        assert "@mycompany.com" in config.email_whitelist
        assert "contractor@external.com" in config.email_whitelist

    def test_security_hardened_configuration(self):
        """Test security-hardened configuration setup."""
        config = AuthenticationConfig(
            auth_error_detail_enabled=False,  # Hide error details
            session_cookie_secure=True,  # Require HTTPS
            session_cookie_httponly=True,  # Prevent XSS
            session_cookie_samesite="strict",  # Strict CSRF protection
            rate_limiting_enabled=True,  # Enable rate limiting
            rate_limit_requests=10,  # Very restrictive
            rate_limit_window=300,  # 5 minute window
            jwt_leeway=0,  # No clock skew tolerance
        )

        assert config.auth_error_detail_enabled is False
        assert config.session_cookie_secure is True
        assert config.session_cookie_httponly is True
        assert config.session_cookie_samesite == "strict"
        assert config.rate_limiting_enabled is True
        assert config.rate_limit_requests == 10
        assert config.rate_limit_window == 300
        assert config.jwt_leeway == 0

    def test_development_friendly_configuration(self):
        """Test development-friendly configuration setup."""
        config = AuthenticationConfig(
            auth_error_detail_enabled=True,  # Show detailed errors
            session_cookie_secure=False,  # Allow HTTP for dev
            session_cookie_samesite="lax",  # Relaxed for dev tools
            jwt_leeway=60,  # More tolerant timing
            jwks_cache_ttl=60,  # Short cache for changes
            rate_limiting_enabled=False,  # Disable for testing
        )

        assert config.auth_error_detail_enabled is True
        assert config.session_cookie_secure is False
        assert config.session_cookie_samesite == "lax"
        assert config.jwt_leeway == 60
        assert config.jwks_cache_ttl == 60
        assert config.rate_limiting_enabled is False
