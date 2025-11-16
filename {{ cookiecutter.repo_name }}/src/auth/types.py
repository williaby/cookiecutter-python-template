"""Common type aliases for authentication system to reduce verbosity."""

from typing import Union

from src.auth.middleware import ServiceTokenUser
from src.auth.models import AuthenticatedUser

# Type alias for authenticated users (either JWT or service token)
AuthenticatedUserType = Union[AuthenticatedUser, ServiceTokenUser]

__all__ = ["AuthenticatedUserType"]
