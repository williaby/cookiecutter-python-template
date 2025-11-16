"""JWT validation module for Cloudflare Access authentication.

This module provides secure JWT token validation including:
- Signature verification against JWKS
- Payload extraction and email claim validation
- Expiration and audience validation
- Security-focused error handling
"""

import logging
from typing import Any

import jwt
from jwt.algorithms import RSAAlgorithm

from .config import AuthenticationConfig
from .constants import ADMIN_ROLE_PREFIXES, PERMISSION_NAME_EMAIL_AUTHORIZATION
from .exceptions import AuthExceptionHandler
from .jwks_client import JWKSClient
from .models import AuthenticatedUser, JWTValidationError, UserRole

logger = logging.getLogger(__name__)


class JWTValidator:
    """Validator for Cloudflare Access JWT tokens."""

    def __init__(
        self,
        jwks_client: JWKSClient,
        config: AuthenticationConfig,
        audience: str | None = None,
        issuer: str | None = None,
        algorithm: str = "RS256",
    ) -> None:
        """Initialize JWT validator.

        Args:
            jwks_client: JWKS client for key retrieval
            audience: Expected audience (aud) claim
            issuer: Expected issuer (iss) claim
            algorithm: JWT algorithm (default RS256)
        """
        self.jwks_client = jwks_client
        self.audience = audience
        self.issuer = issuer
        self.algorithm = algorithm

    def validate_token(  # noqa: PLR0912
        self,
        token: str,
        email_whitelist: list[str] | None = None,
    ) -> AuthenticatedUser:
        """Validate JWT token and extract user information.

        Args:
            token: JWT token string
            email_whitelist: Optional list of allowed email addresses/domains

        Returns:
            AuthenticatedUser with validated user information

        Raises:
            JWTValidationError: If token validation fails
        """
        try:
            # Validate token format first
            if not token or not isinstance(token, str):
                raise JWTValidationError("Invalid token format", "invalid_format")

            token_parts = token.split(".")
            if len(token_parts) != 3:
                raise JWTValidationError("Invalid token format", "invalid_format")

            # Decode token header to get key ID
            try:
                unverified_header = jwt.get_unverified_header(token)
            except jwt.DecodeError as e:
                raise JWTValidationError("Invalid token format", "invalid_format") from e
            except Exception as e:
                raise JWTValidationError("Invalid token format", "invalid_format") from e

            kid = unverified_header.get("kid")

            if not kid:
                raise JWTValidationError("JWT token missing 'kid' in header", "missing_kid")

            # Get signing key from JWKS
            key_dict = self.jwks_client.get_key_by_kid(kid)
            if not key_dict:
                raise JWTValidationError(f"Key with kid '{kid}' not found in JWKS", "key_not_found")

            # Convert JWK to RSA public key
            try:
                public_key = RSAAlgorithm.from_jwk(key_dict)
            except Exception as e:
                logger.error(f"Failed to convert JWK to public key: {e}")
                raise JWTValidationError(f"Invalid JWK format: {e}", "invalid_jwk") from e

            # Prepare verification options
            verify_options = {
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "iat", "email"],
            }

            # Add audience/issuer verification if configured
            if self.audience:
                verify_options["verify_aud"] = True
            if self.issuer:
                verify_options["verify_iss"] = True

            # Decode and validate token
            try:
                # Cast to expected type for jwt.decode
                payload = jwt.decode(
                    token,
                    public_key,  # type: ignore[arg-type]
                    algorithms=[self.algorithm],
                    audience=self.audience,
                    issuer=self.issuer,
                    options=verify_options,
                )
            except jwt.ExpiredSignatureError as e:
                logger.warning("JWT token has expired")
                raise JWTValidationError("Token has expired", "expired_token") from e
            except jwt.InvalidSignatureError as e:
                logger.warning("JWT token signature verification failed")
                raise JWTValidationError("Token signature verification failed", "invalid_signature") from e
            except jwt.InvalidAudienceError as e:
                logger.warning("JWT token audience validation failed")
                raise JWTValidationError("Invalid token audience", "invalid_audience") from e
            except jwt.InvalidIssuerError as e:
                logger.warning("JWT token issuer validation failed")
                raise JWTValidationError("Invalid token issuer", "invalid_issuer") from e
            except jwt.InvalidKeyError as e:
                logger.warning("JWT token key validation failed")
                raise JWTValidationError("Unable to verify token signature", "invalid_key") from e
            except jwt.MissingRequiredClaimError as e:
                logger.warning(f"JWT token missing required claim: {e}")
                raise JWTValidationError(f"Token missing required claim: {e}", "missing_claim") from e
            except jwt.InvalidTokenError as e:
                logger.warning(f"JWT token validation failed: {e}")
                raise JWTValidationError(f"Invalid token: {e}", "invalid_token") from e

            # Extract and validate email from payload (CRITICAL: not from headers)
            email = payload.get("email")
            if email is None:
                raise JWTValidationError("JWT payload missing required 'email' claim", "missing_email")

            if email == "":
                raise JWTValidationError("JWT payload missing required 'email' claim", "missing_email")

            if not isinstance(email, str) or "@" not in email:
                raise JWTValidationError("Invalid email format in JWT payload", "invalid_email")

            # Validate email against whitelist if provided
            if email_whitelist and not self.is_email_allowed(email, email_whitelist):
                logger.warning(f"Email '{email}' not in whitelist")
                # Use AuthExceptionHandler for authorization errors
                raise AuthExceptionHandler.handle_permission_error(
                    permission_name=PERMISSION_NAME_EMAIL_AUTHORIZATION,
                    user_identifier=email,
                    detail=f"Email '{email}' not authorized for this application",
                )

            # Determine user role (basic implementation)
            role = self._determine_user_role(email, payload)

            logger.info(f"Successfully validated JWT for user: {email}")

            return AuthenticatedUser(
                email=email,
                role=role,
                jwt_claims=payload,
            )

        except JWTValidationError:
            # Re-raise JWT validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error during JWT validation: {e}")
            # Use AuthExceptionHandler for authentication failures
            raise AuthExceptionHandler.handle_authentication_error(
                detail="Authentication failed",
                log_message=f"JWT validation failed: {e}",
            ) from e

    def _is_email_allowed(self, email: str, email_whitelist: list[str]) -> bool:
        """Check if email is allowed based on whitelist.

        Args:
            email: Email address to check
            email_whitelist: List of allowed emails or domains

        Returns:
            True if email is allowed, False otherwise
        """
        email_lower = email.lower()

        for allowed in email_whitelist:
            allowed_lower = allowed.lower()

            # Exact email match
            if email_lower == allowed_lower:
                return True

            # Domain match (if allowed entry starts with @)
            if allowed_lower.startswith("@") and email_lower.endswith(allowed_lower):
                return True

        return False

    def is_email_allowed(self, email: str, email_whitelist: list[str] | None = None) -> bool:
        """Check if email is allowed based on whitelist.

        Args:
            email: Email address to check
            email_whitelist: List of allowed emails or domains

        Returns:
            True if email is allowed, False otherwise
        """
        if email_whitelist is None:
            return True
        return self._is_email_allowed(email, email_whitelist)

    def determine_admin_role(self, email: str) -> UserRole:
        """Determine if email should have admin role based on exact prefix matching.

        Args:
            email: User email address

        Returns:
            UserRole.ADMIN if email starts with admin keywords, UserRole.USER otherwise
        """
        email_lower = email.lower()
        admin_prefixes = ADMIN_ROLE_PREFIXES

        # Extract username part before @
        username = email_lower.split("@")[0]

        # Check if username starts with any admin prefix
        for prefix in admin_prefixes:
            if username == prefix:  # Exact match for the username part
                return UserRole.ADMIN

        return UserRole.USER

    def _determine_user_role(self, email: str, payload: dict[str, Any]) -> UserRole:
        """Determine user role based on email and JWT claims.

        Args:
            email: User email address
            payload: JWT payload

        Returns:
            UserRole for the user
        """
        # Use the public method for role determination
        role_from_email = self.determine_admin_role(email)
        if role_from_email == UserRole.ADMIN:
            return UserRole.ADMIN

        # Check for admin role in JWT claims (if Cloudflare provides it)
        groups = payload.get("groups", [])
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, str) and "admin" in group.lower():
                    return UserRole.ADMIN

        # Default to user role
        return UserRole.USER

    def validate_token_format(self, token: str) -> bool:
        """Basic token format validation without signature verification.

        Args:
            token: JWT token string

        Returns:
            True if token format is valid, False otherwise
        """
        try:
            # Check basic JWT format (3 parts separated by dots)
            parts = token.split(".")
            if len(parts) != 3:
                return False

            # Try to decode header without verification
            jwt.get_unverified_header(token)
            return True

        except Exception:
            return False

    def validate_token_or_raise(self, token: str, email_whitelist: list[str] | None = None) -> AuthenticatedUser:
        """Validate JWT token and raise HTTPException on failure.

        Wrapper around validate_token that converts JWTValidationError to
        standardized HTTP exceptions using AuthExceptionHandler.

        Args:
            token: JWT token string
            email_whitelist: Optional list of allowed email addresses/domains

        Returns:
            AuthenticatedUser with validated user information

        Raises:
            HTTPException: Standardized authentication/authorization errors
        """
        try:
            return self.validate_token(token, email_whitelist)
        except JWTValidationError as e:
            # Convert specific JWT validation errors to appropriate HTTP exceptions
            if e.error_code in {"expired_token"}:
                raise AuthExceptionHandler.handle_authentication_error(
                    detail="Token has expired",
                    log_message=f"JWT validation failed: {e.message}",
                ) from e
            if e.error_code in {"invalid_signature", "invalid_key"}:
                raise AuthExceptionHandler.handle_authentication_error(
                    detail="Invalid authentication credentials",
                    log_message=f"JWT signature validation failed: {e.message}",
                ) from e
            if e.error_code in {"invalid_format", "missing_kid", "invalid_token"}:
                raise AuthExceptionHandler.handle_validation_error(
                    f"Invalid token format: {e.message}",
                    field_name="authorization_token",
                ) from e
            if e.error_code == "email_not_authorized":
                # This is already handled in validate_token with AuthExceptionHandler
                raise
            # Generic authentication error for other cases
            raise AuthExceptionHandler.handle_authentication_error(
                detail="Authentication failed",
                log_message=f"JWT validation failed: {e.message}",
            ) from e
